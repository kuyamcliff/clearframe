//! FFmpeg/ffprobe discovery and capability detection.
//!
//! Commands are always built from argument vectors — never through a shell —
//! so file names cannot inject arguments or shell syntax.

use std::collections::BTreeSet;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::MediaError;

/// Where an executable was found.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolSource {
    Configured,
    Environment,
    Bundled,
    SystemPath,
}

/// An available video encoder.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EncoderInfo {
    pub name: String,
    pub description: String,
    pub is_hardware: bool,
}

/// Discovered FFmpeg installation and its capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfmpegTools {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
    pub source: ToolSource,
    pub version: String,
    pub configuration: String,
    pub license_hint: String,
    pub video_encoders: Vec<EncoderInfo>,
    pub audio_encoders: Vec<String>,
    pub filters: BTreeSet<String>,
    pub muxers: BTreeSet<String>,
}

#[cfg(windows)]
const EXE: &str = ".exe";
#[cfg(not(windows))]
const EXE: &str = "";

static DEFAULT_THREADS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Limit threads used by every FFmpeg process this process starts (0 = FFmpeg default).
/// Applied as `-threads` for decoders/encoders and `-filter_threads` for filter graphs.
pub fn set_default_threads(n: usize) {
    DEFAULT_THREADS.store(n, std::sync::atomic::Ordering::Relaxed);
}

pub fn default_threads() -> Option<usize> {
    match DEFAULT_THREADS.load(std::sync::atomic::Ordering::Relaxed) {
        0 => None,
        n => Some(n),
    }
}

/// Environment variable to point at a directory containing ffmpeg and ffprobe.
pub const ENV_FFMPEG_DIR: &str = "CLEARFRAME_FFMPEG_DIR";

fn candidate_dirs(configured: Option<&Path>, extra_bundled: &[PathBuf]) -> Vec<(PathBuf, ToolSource)> {
    let mut v = Vec::new();
    if let Some(c) = configured {
        let dir = if c.is_file() { c.parent().map(Path::to_path_buf).unwrap_or_default() } else { c.to_path_buf() };
        v.push((dir, ToolSource::Configured));
    }
    if let Ok(d) = std::env::var(ENV_FFMPEG_DIR) {
        v.push((PathBuf::from(d), ToolSource::Environment));
    }
    for b in extra_bundled {
        v.push((b.clone(), ToolSource::Bundled));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            v.push((dir.to_path_buf(), ToolSource::Bundled));
            v.push((dir.join("ffmpeg"), ToolSource::Bundled));
            // macOS app bundle: Contents/MacOS/<exe> -> Contents/Resources
            v.push((dir.join("../Resources"), ToolSource::Bundled));
            v.push((dir.join("../Resources/ffmpeg"), ToolSource::Bundled));
            // Linux AppImage / deb: /usr/bin/<exe> -> /usr/lib/<app>
            v.push((dir.join("../lib/clearframe"), ToolSource::Bundled));
            v.push((dir.join("../lib/ClearFrame"), ToolSource::Bundled));
        }
    }
    if let Some(paths) = std::env::var_os("PATH") {
        for p in std::env::split_paths(&paths) {
            v.push((p, ToolSource::SystemPath));
        }
    }
    v
}

/// Tauri sidecar naming: `ffmpeg-<target-triple>[.exe]` next to the executable.
fn sidecar_names(base: &str) -> Vec<String> {
    let mut names = vec![format!("{base}{EXE}")];
    if let Some(triple) = option_env!("CLEARFRAME_TARGET_TRIPLE") {
        names.push(format!("{base}-{triple}{EXE}"));
    }
    names
}

fn find_in(dir: &Path, base: &str) -> Option<PathBuf> {
    for n in sidecar_names(base) {
        let p = dir.join(&n);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

/// Run a command with a timeout, returning (status_ok, stdout, stderr) with bounded output.
pub fn run_capture(cmd: &mut Command, timeout: Duration, max_bytes: usize) -> Result<(bool, String, String), MediaError> {
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    hide_console(cmd);
    let mut child = cmd.spawn().map_err(|e| MediaError::Spawn(format!("{:?}: {e}", cmd.get_program())))?;
    let mut so = child.stdout.take().unwrap();
    let mut se = child.stderr.take().unwrap();
    let t_out = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = (&mut so).take(max_bytes as u64).read_to_end(&mut buf);
        // drain the rest so the child never blocks
        let _ = std::io::copy(&mut so, &mut std::io::sink());
        buf
    });
    let t_err = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = (&mut se).take(max_bytes as u64).read_to_end(&mut buf);
        let _ = std::io::copy(&mut se, &mut std::io::sink());
        buf
    });
    let start = Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait()? {
            break s;
        }
        if start.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(MediaError::Timeout(format!("{:?}", cmd.get_program())));
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let out = t_out.join().unwrap_or_default();
    let err = t_err.join().unwrap_or_default();
    Ok((status.success(), String::from_utf8_lossy(&out).into_owned(), String::from_utf8_lossy(&err).into_owned()))
}

/// Prevent a console window flashing up on Windows.
pub fn hide_console(cmd: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(not(windows))]
    let _ = cmd;
}

impl FfmpegTools {
    /// Discover ffmpeg/ffprobe. `configured` may be a directory or the ffmpeg binary.
    pub fn discover(configured: Option<&Path>, bundled_dirs: &[PathBuf]) -> Result<Self, MediaError> {
        for (dir, source) in candidate_dirs(configured, bundled_dirs) {
            let (Some(ffmpeg), Some(ffprobe)) = (find_in(&dir, "ffmpeg"), find_in(&dir, "ffprobe")) else {
                continue;
            };
            match Self::from_paths(ffmpeg, ffprobe, source) {
                Ok(t) => return Ok(t),
                Err(e) => log::warn!("unusable ffmpeg in {}: {e}", dir.display()),
            }
        }
        Err(MediaError::NotFound)
    }

    /// Validate explicit paths and read capabilities.
    pub fn from_paths(ffmpeg: PathBuf, ffprobe: PathBuf, source: ToolSource) -> Result<Self, MediaError> {
        let t = Duration::from_secs(20);
        let (ok, out, _) = run_capture(Command::new(&ffmpeg).args(["-hide_banner", "-version"]), t, 1 << 20)?;
        if !ok {
            return Err(MediaError::Tool("ffmpeg -version failed".into()));
        }
        let version = out.lines().next().unwrap_or("").replace("ffmpeg version ", "");
        let version = version.split(" Copyright").next().unwrap_or("").trim().to_string();
        let configuration = out
            .lines()
            .find(|l| l.starts_with("configuration:"))
            .map(|l| l.trim_start_matches("configuration:").trim().to_string())
            .unwrap_or_default();
        let license_hint = if configuration.contains("--enable-nonfree") {
            "nonfree (not redistributable)".to_string()
        } else if configuration.contains("--enable-gpl") {
            "GPL".to_string()
        } else {
            "LGPL".to_string()
        };
        let (ok, _, _) = run_capture(Command::new(&ffprobe).args(["-hide_banner", "-version"]), t, 1 << 20)?;
        if !ok {
            return Err(MediaError::Tool("ffprobe -version failed".into()));
        }
        let (_, enc, _) = run_capture(Command::new(&ffmpeg).args(["-hide_banner", "-encoders"]), t, 4 << 20)?;
        let mut video_encoders = Vec::new();
        let mut audio_encoders = Vec::new();
        for line in enc.lines() {
            let l = line.trim_start();
            let mut parts = l.splitn(3, char::is_whitespace);
            let (Some(flags), Some(name)) = (parts.next(), parts.next()) else { continue };
            if flags.len() != 6 || name == "=" {
                continue;
            }
            let desc = parts.next().unwrap_or("").trim().to_string();
            if flags.starts_with('V') {
                let is_hardware = ["nvenc", "qsv", "vaapi", "amf", "videotoolbox", "_mf", "v4l2m2m", "vulkan"]
                    .iter()
                    .any(|h| name.contains(h));
                video_encoders.push(EncoderInfo { name: name.to_string(), description: desc, is_hardware });
            } else if flags.starts_with('A') {
                audio_encoders.push(name.to_string());
            }
        }
        let (_, flt, _) = run_capture(Command::new(&ffmpeg).args(["-hide_banner", "-filters"]), t, 4 << 20)?;
        let filters = flt
            .lines()
            .filter_map(|l| {
                let mut it = l.split_whitespace();
                let flags = it.next()?;
                let name = it.next()?;
                if flags.len() == 3 && it.next().is_some() {
                    Some(name.to_string())
                } else {
                    None
                }
            })
            .collect();
        let (_, mux, _) = run_capture(Command::new(&ffmpeg).args(["-hide_banner", "-muxers"]), t, 4 << 20)?;
        let muxers = mux
            .lines()
            .filter_map(|l| {
                let l = l.trim_start();
                let mut it = l.split_whitespace();
                let flags = it.next()?;
                let names = it.next()?;
                if flags.contains('E') && flags.len() <= 3 {
                    Some(names.split(',').map(String::from).collect::<Vec<_>>())
                } else {
                    None
                }
            })
            .flatten()
            .collect();
        Ok(Self { ffmpeg, ffprobe, source, version, configuration, license_hint, video_encoders, audio_encoders, filters, muxers })
    }

    pub fn has_filter(&self, name: &str) -> bool {
        self.filters.contains(name)
    }

    pub fn has_video_encoder(&self, name: &str) -> bool {
        self.video_encoders.iter().any(|e| e.name == name)
    }

    pub fn has_audio_encoder(&self, name: &str) -> bool {
        self.audio_encoders.iter().any(|e| e == name)
    }

    pub fn ffmpeg_cmd(&self) -> Command {
        let mut c = Command::new(&self.ffmpeg);
        c.args(["-nostdin", "-hide_banner"]);
        if let Some(t) = default_threads() {
            c.args(["-filter_threads", &t.to_string()]);
        }
        hide_console(&mut c);
        c
    }

    pub fn ffprobe_cmd(&self) -> Command {
        let mut c = Command::new(&self.ffprobe);
        c.arg("-hide_banner");
        hide_console(&mut c);
        c
    }

    /// Check a hardware encoder actually works on this machine with a tiny test encode.
    pub fn test_encoder(&self, name: &str) -> bool {
        let mut c = self.ffmpeg_cmd();
        c.args(["-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:r=25:d=0.2", "-frames:v", "3", "-c:v", name, "-f", "null", "-"]);
        matches!(run_capture(&mut c, Duration::from_secs(30), 1 << 16), Ok((true, _, _)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_system_ffmpeg() {
        // CI and dev machines are expected to provide ffmpeg on PATH.
        let t = match FfmpegTools::discover(None, &[]) {
            Ok(t) => t,
            Err(_) => {
                eprintln!("ffmpeg not installed; skipping");
                return;
            }
        };
        assert!(!t.version.is_empty());
        assert!(t.has_filter("scale"));
        assert!(t.video_encoders.iter().any(|e| e.name == "ffv1" || e.name == "libx264"));
        assert!(t.muxers.contains("matroska"));
    }

    #[test]
    fn timeout_kills_process() {
        #[cfg(unix)]
        {
            let r = run_capture(Command::new("sleep").arg("5"), Duration::from_millis(200), 1024);
            assert!(matches!(r, Err(MediaError::Timeout(_))));
        }
    }
}
