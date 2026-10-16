//! Local file logging. Logs never leave the machine unless the user sends the file themselves.
//!
//! The desktop app and its worker processes all write to one session log in the user's
//! Documents folder (`ClipFX Enhance Log.txt`), from the moment the app starts: startup,
//! every click and command in the interface, errors, crashes, job progress and periodic
//! performance samples. One file is easy to find and send when something goes wrong.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use parking_lot::Mutex;

/// File name of the session log in the Documents folder.
pub const SESSION_LOG_NAME: &str = "ClipFX Enhance Log.txt";

/// When the session log grows past this at startup, only the newest `KEEP_ON_TRIM` bytes are kept.
const TRIM_OVER: u64 = 50 << 20;
const KEEP_ON_TRIM: u64 = 10 << 20;
/// Rotation size for the plain per-directory log (`init`).
const MAX_LOG: u64 = 5 << 20;

static SESSION_PATH: OnceLock<PathBuf> = OnceLock::new();

struct FileLogger {
    path: PathBuf,
    file: Mutex<Option<std::fs::File>>,
    /// Rotate to `.log.1` past this size (plain log only).
    rotate_at: Option<u64>,
    /// Explicit level from `CLIPFX_LOG`; otherwise per-target defaults.
    forced: Option<log::LevelFilter>,
    /// Which process wrote the line: "app", "worker", "still", "cli".
    role: &'static str,
    pid: u32,
}

/// Default level per log target: everything from ClipFX itself (and the interface) in detail,
/// other libraries at info, and the chattiest ones only when something is wrong.
fn default_level(target: &str) -> log::LevelFilter {
    const OURS: &[&str] = &["clearframe", "clipfx", "ui"];
    const NOISY: &[&str] = &["ureq", "rustls", "naga", "wgpu_hal", "wgpu_core", "tao", "wry", "hyper", "mio", "tracing", "h2", "tiny_http", "polling", "async_io"];
    if OURS.iter().any(|p| target.starts_with(p)) {
        log::LevelFilter::Debug
    } else if NOISY.iter().any(|p| target.starts_with(p)) {
        log::LevelFilter::Warn
    } else {
        log::LevelFilter::Info
    }
}

impl log::Log for FileLogger {
    fn enabled(&self, m: &log::Metadata) -> bool {
        let max = self.forced.unwrap_or_else(|| default_level(m.target()));
        m.level() <= max
    }
    fn log(&self, r: &log::Record) {
        if !self.enabled(r.metadata()) {
            return;
        }
        let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f");
        let thread = std::thread::current();
        let tname = thread.name().unwrap_or("-");
        let line = format!("{now} [{} {} {tname}] {:<5} {}: {}\n", self.role, self.pid, r.level(), r.target(), r.args());
        let mut g = self.file.lock();
        if let (Some(limit), Some(f)) = (self.rotate_at, g.as_mut()) {
            if f.metadata().map(|m| m.len() > limit).unwrap_or(false) {
                let _ = std::fs::rename(&self.path, self.path.with_extension("log.1"));
                *g = open_append(&self.path);
            }
        }
        if let Some(f) = g.as_mut() {
            let _ = f.write_all(line.as_bytes());
        }
        if cfg!(debug_assertions) {
            eprint!("{line}");
        }
    }
    fn flush(&self) {
        if let Some(f) = self.file.lock().as_mut() {
            let _ = f.flush();
        }
    }
}

fn open_append(path: &Path) -> Option<std::fs::File> {
    std::fs::OpenOptions::new().create(true).append(true).open(path).ok()
}

fn forced_level() -> Option<log::LevelFilter> {
    std::env::var("CLIPFX_LOG").ok().and_then(|l| l.parse().ok())
}

fn install(logger: FileLogger) {
    let max = logger.forced.unwrap_or(log::LevelFilter::Debug);
    if log::set_boxed_logger(Box::new(logger)).is_ok() {
        log::set_max_level(max);
    }
}

/// Install the global logger writing to `<dir>/<name>.log`. Safe to call once per process.
pub fn init(dir: &Path, name: &str) {
    let _ = std::fs::create_dir_all(dir);
    let path = dir.join(format!("{name}.log"));
    let file = open_append(&path);
    install(FileLogger { path, file: Mutex::new(file), rotate_at: Some(MAX_LOG), forced: forced_level(), role: "cli", pid: std::process::id() });
}

/// Where the session log goes: the Documents folder, else the home folder, else the temp folder.
fn session_candidates() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(d) = dirs::document_dir() {
        v.push(d);
    }
    if let Some(h) = dirs::home_dir() {
        // Linux without XDG user folders configured: use ~/Documents when it exists.
        if h.join("Documents").is_dir() {
            v.push(h.join("Documents"));
        }
        v.push(h);
    }
    v.push(std::env::temp_dir());
    v.into_iter().map(|d| d.join(SESSION_LOG_NAME)).collect()
}

/// Keep the log from growing without bound: past `TRIM_OVER`, keep only the newest lines.
fn trim(path: &Path) {
    use std::io::{Read, Seek, SeekFrom};
    let Ok(len) = std::fs::metadata(path).map(|m| m.len()) else { return };
    if len <= TRIM_OVER {
        return;
    }
    let Ok(mut f) = std::fs::File::open(path) else { return };
    let _ = f.seek(SeekFrom::Start(len - KEEP_ON_TRIM));
    let mut tail = Vec::new();
    let _ = f.read_to_end(&mut tail);
    drop(f);
    let start = tail.iter().position(|&b| b == b'\n').map(|i| i + 1).unwrap_or(0);
    let mut out = format!("[older lines removed: the log had grown to {} MB]\n", len >> 20).into_bytes();
    out.extend_from_slice(&tail[start..]);
    let _ = std::fs::write(path, out);
}

/// Install the session logger for this process (`role`: "app", "worker" or "still").
/// The app trims an oversized log first; workers only append. Also records panics.
/// Returns the log file path. Safe to call once per process.
pub fn init_session(role: &'static str) -> Option<PathBuf> {
    let mut chosen = None;
    for path in session_candidates() {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if role == "app" {
            trim(&path);
        }
        if let Some(f) = open_append(&path) {
            chosen = Some((path, f));
            break;
        }
    }
    let (path, file) = chosen?;
    let _ = SESSION_PATH.set(path.clone());
    install(FileLogger { path: path.clone(), file: Mutex::new(Some(file)), rotate_at: None, forced: forced_level(), role, pid: std::process::id() });
    install_panic_hook();
    Some(path)
}

/// Path of this process's session log, once `init_session` has run.
pub fn session_log_path() -> Option<PathBuf> {
    SESSION_PATH.get().cloned()
}

/// The session log path even before (or without) `init_session`: the first usable candidate.
pub fn default_session_log_path() -> PathBuf {
    session_log_path().unwrap_or_else(|| session_candidates().remove(0))
}

fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let thread = std::thread::current();
        let bt = std::backtrace::Backtrace::force_capture();
        log::error!(target: "clipfx::crash", "PANIC in thread '{}': {info}\nbacktrace:\n{bt}", thread.name().unwrap_or("-"));
        log::logger().flush();
        previous(info);
    }));
}

/// Last `max_bytes` of a log file.
pub fn tail(path: &Path, max_bytes: u64) -> String {
    use std::io::{Read, Seek, SeekFrom};
    let Ok(mut f) = std::fs::File::open(path) else { return String::new() };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let _ = f.seek(SeekFrom::Start(len.saturating_sub(max_bytes)));
    let mut b = Vec::new();
    let _ = f.read_to_end(&mut b);
    String::from_utf8_lossy(&b).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels() {
        assert_eq!(default_level("clearframe_pipeline::runner"), log::LevelFilter::Debug);
        assert_eq!(default_level("ui"), log::LevelFilter::Debug);
        assert_eq!(default_level("wgpu_core::device"), log::LevelFilter::Warn);
        assert_eq!(default_level("wgpu"), log::LevelFilter::Info);
    }

    #[test]
    fn trims_oversized_log() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("log.txt");
        let line = "x".repeat(99) + "\n";
        let mut f = std::fs::File::create(&p).unwrap();
        for _ in 0..((TRIM_OVER / 100) + 1000) {
            f.write_all(line.as_bytes()).unwrap();
        }
        drop(f);
        trim(&p);
        let len = std::fs::metadata(&p).unwrap().len();
        assert!(len <= KEEP_ON_TRIM + 200, "{len}");
        let s = std::fs::read_to_string(&p).unwrap();
        assert!(s.starts_with("[older lines removed"));
        assert!(s.lines().skip(1).all(|l| l.len() == 99));
    }
}
