//! Local file logging. Logs never leave the machine unless the user exports diagnostics.

use std::io::Write;
use std::path::{Path, PathBuf};

use parking_lot::Mutex;

struct FileLogger {
    path: PathBuf,
    file: Mutex<Option<std::fs::File>>,
    level: log::LevelFilter,
}

const MAX_LOG: u64 = 5 << 20;

impl log::Log for FileLogger {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.level() <= self.level && !m.target().starts_with("ureq") && !m.target().starts_with("rustls")
    }
    fn log(&self, r: &log::Record) {
        if !self.enabled(r.metadata()) {
            return;
        }
        let line = format!("{} {:<5} {}: {}\n", chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ"), r.level(), r.target(), r.args());
        let mut g = self.file.lock();
        if let Some(f) = g.as_mut() {
            if f.metadata().map(|m| m.len() > MAX_LOG).unwrap_or(false) {
                let _ = std::fs::rename(&self.path, self.path.with_extension("log.1"));
                *g = std::fs::OpenOptions::new().create(true).append(true).open(&self.path).ok();
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

/// Install the global logger writing to `<dir>/<name>.log`. Safe to call once per process.
pub fn init(dir: &Path, name: &str) {
    let _ = std::fs::create_dir_all(dir);
    let path = dir.join(format!("{name}.log"));
    let file = std::fs::OpenOptions::new().create(true).append(true).open(&path).ok();
    let level = std::env::var("CLIPFX_LOG").ok().and_then(|l| l.parse().ok()).unwrap_or(log::LevelFilter::Info);
    if log::set_boxed_logger(Box::new(FileLogger { path, file: Mutex::new(file), level })).is_ok() {
        log::set_max_level(level);
    }
}

/// Last `max_bytes` of a log file.
pub fn tail(path: &Path, max_bytes: u64) -> String {
    use std::io::{Read, Seek, SeekFrom};
    let Ok(mut f) = std::fs::File::open(path) else { return String::new() };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let _ = f.seek(SeekFrom::Start(len.saturating_sub(max_bytes)));
    let mut s = String::new();
    let _ = f.read_to_string(&mut s);
    s
}
