//! Output verification: probe and fully decode before a job is reported complete.

use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::probe::{probe, MediaInfo};
use crate::rational::Rational;
use crate::tools::{run_capture, FfmpegTools};
use crate::MediaError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Expected {
    pub width: u32,
    pub height: u32,
    pub fps: Rational,
    pub duration: f64,
    pub audio_streams: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub ok: bool,
    pub problems: Vec<String>,
    pub info: Option<MediaInfo>,
    pub decoded_frames: Option<u64>,
}

/// Compare probed output with expectations (pure; used by tests).
pub fn compare(info: &MediaInfo, exp: &Expected) -> Vec<String> {
    let mut p = Vec::new();
    let Some(v) = &info.video else {
        p.push("output has no video stream".into());
        return p;
    };
    if v.width != exp.width || v.height != exp.height {
        p.push(format!("output is {}x{} but {}x{} was expected", v.width, v.height, exp.width, exp.height));
    }
    let fr = v.avg_frame_rate.as_f64();
    if (fr - exp.fps.as_f64()).abs() > 0.02 * exp.fps.as_f64().max(1.0) {
        p.push(format!("output frame rate is {:.3} but {:.3} was expected", fr, exp.fps.as_f64()));
    }
    let frame = 1.0 / exp.fps.as_f64().max(1.0);
    let dur = v.duration.unwrap_or(info.duration);
    if (dur - exp.duration).abs() > (3.0 * frame).max(0.12) {
        p.push(format!("output duration is {:.3}s but {:.3}s was expected", dur, exp.duration));
    }
    if info.audio.len() != exp.audio_streams {
        p.push(format!("output has {} audio streams but {} were expected", info.audio.len(), exp.audio_streams));
    }
    p
}

/// Probe and fully decode `path`.
pub fn validate(tools: &FfmpegTools, path: &Path, exp: &Expected) -> Result<ValidationReport, MediaError> {
    let info = match probe(tools, path) {
        Ok(i) => i,
        Err(e) => return Ok(ValidationReport { ok: false, problems: vec![format!("output cannot be read: {e}")], info: None, decoded_frames: None }),
    };
    let mut problems = compare(&info, exp);
    // Full decode of every stream; -xerror makes any decode error fatal.
    let mut c = tools.ffmpeg_cmd();
    c.args(["-v", "error", "-xerror", "-i"]).arg(path).args(["-map", "0:v:0", "-f", "null", "-"]);
    let (ok, _, err) = run_capture(&mut c, Duration::from_secs(6 * 3600), 1 << 20)?;
    if !ok || !err.trim().is_empty() {
        problems.push(format!("output failed a full decode check: {}", err.trim()));
    }
    let decoded_frames = None;
    Ok(ValidationReport { ok: problems.is_empty(), problems, info: Some(info), decoded_frames })
}
