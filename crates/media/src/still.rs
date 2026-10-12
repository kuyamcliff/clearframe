//! Single frames, thumbnails and browser-playable proxy clips.

use std::path::Path;
use std::time::Duration;

use crate::tools::{run_capture, FfmpegTools};
use crate::MediaError;

/// Extract one frame at `time` as PNG bytes, scaled to at most `max_width` (keeping aspect).
pub fn frame_png(tools: &FfmpegTools, path: &Path, time: f64, max_width: Option<u32>) -> Result<Vec<u8>, MediaError> {
    let dir = tempfile_dir()?;
    let out = dir.join("frame.png");
    let mut c = tools.ffmpeg_cmd();
    c.args(["-v", "error", "-y", "-ss", &format!("{:.6}", time.max(0.0)), "-i"]).arg(path);
    let mut vf = Vec::new();
    if let Some(w) = max_width {
        vf.push(format!("scale='min({w},iw)':-2:flags=bicubic"));
    }
    if !vf.is_empty() {
        c.args(["-vf", &vf.join(",")]);
    }
    c.args(["-frames:v", "1", "-update", "1"]).arg(&out);
    let (ok, _, err) = run_capture(&mut c, Duration::from_secs(60), 1 << 20)?;
    let res = if ok && out.is_file() { Ok(std::fs::read(&out)?) } else { Err(MediaError::Process(format!("could not extract frame: {}", err.trim()))) };
    let _ = std::fs::remove_dir_all(&dir);
    res
}

/// Evenly spaced thumbnails (time, PNG bytes).
pub fn thumbnails(tools: &FfmpegTools, path: &Path, duration: f64, count: usize, height: u32) -> Result<Vec<(f64, Vec<u8>)>, MediaError> {
    let mut v = Vec::new();
    for i in 0..count {
        let t = duration * (i as f64 + 0.5) / count as f64;
        let dir = tempfile_dir()?;
        let out = dir.join("t.jpg");
        let mut c = tools.ffmpeg_cmd();
        c.args(["-v", "error", "-y", "-ss", &format!("{t:.3}"), "-i"]).arg(path);
        c.args(["-vf", &format!("scale=-2:{height}"), "-frames:v", "1", "-q:v", "5", "-update", "1"]).arg(&out);
        if let Ok((true, _, _)) = run_capture(&mut c, Duration::from_secs(30), 1 << 16) {
            if let Ok(b) = std::fs::read(&out) {
                v.push((t, b));
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    Ok(v)
}

/// Create a browser-playable H.264/AAC MP4 of `[start, start+duration)` at most `max_height` tall.
pub fn proxy_clip(tools: &FfmpegTools, src: &Path, start: f64, duration: f64, max_height: u32, out: &Path) -> Result<(), MediaError> {
    let mut c = tools.ffmpeg_cmd();
    c.args(["-v", "error", "-y", "-ss", &format!("{start:.6}"), "-i"]).arg(src);
    c.args(["-t", &format!("{duration:.6}"), "-map", "0:v:0", "-map", "0:a:0?"]);
    c.args(["-vf", &format!("scale=-2:'min({max_height},ih)':flags=bicubic,format=yuv420p")]);
    c.args(["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart"]);
    c.arg(out);
    let (ok, _, err) = run_capture(&mut c, Duration::from_secs(3600), 1 << 20)?;
    if !ok {
        return Err(MediaError::Process(format!("could not create preview clip: {}", err.trim())));
    }
    Ok(())
}

fn tempfile_dir() -> Result<std::path::PathBuf, MediaError> {
    let d = std::env::temp_dir().join(format!("clearframe-still-{}-{}", std::process::id(), unique()));
    std::fs::create_dir_all(&d)?;
    Ok(d)
}

fn unique() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
    t ^ N.fetch_add(1, Ordering::Relaxed)
}
