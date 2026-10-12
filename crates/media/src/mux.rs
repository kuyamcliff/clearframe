//! Final assembly: concatenate encoded video segments and add audio,
//! subtitles, chapters and metadata from the source.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::probe::MediaInfo;
use crate::tools::{run_capture, FfmpegTools};
use crate::MediaError;

/// Output container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Container {
    Mp4,
    Mov,
    Mkv,
    Webm,
    M4v,
}

impl Container {
    pub fn extension(self) -> &'static str {
        match self {
            Container::Mp4 => "mp4",
            Container::Mov => "mov",
            Container::Mkv => "mkv",
            Container::Webm => "webm",
            Container::M4v => "m4v",
        }
    }
    pub fn muxer(self) -> &'static str {
        match self {
            Container::Mp4 | Container::M4v => "mp4",
            Container::Mov => "mov",
            Container::Mkv => "matroska",
            Container::Webm => "webm",
        }
    }
    pub fn from_extension(ext: &str) -> Option<Self> {
        Some(match ext.to_ascii_lowercase().as_str() {
            "mp4" => Container::Mp4,
            "mov" => Container::Mov,
            "mkv" => Container::Mkv,
            "webm" => Container::Webm,
            "m4v" => Container::M4v,
            _ => return None,
        })
    }
    fn is_isobmff(self) -> bool {
        matches!(self, Container::Mp4 | Container::Mov | Container::M4v)
    }
    /// Can this container hold audio codec `codec` without re-encoding?
    pub fn audio_copy_ok(self, codec: &str) -> bool {
        match self {
            Container::Mkv => true,
            Container::Webm => matches!(codec, "opus" | "vorbis"),
            Container::Mp4 | Container::M4v => matches!(codec, "aac" | "mp3" | "ac3" | "eac3" | "alac" | "flac" | "opus"),
            Container::Mov => matches!(codec, "aac" | "mp3" | "ac3" | "eac3" | "alac" | "pcm_s16le" | "pcm_s24le" | "pcm_s16be" | "pcm_s24be"),
        }
    }
    /// Can this container hold video family `family`?
    pub fn video_ok(self, family: &str) -> bool {
        match self {
            Container::Mkv => true,
            Container::Webm => matches!(family, "vp9" | "av1"),
            Container::Mp4 | Container::M4v => matches!(family, "h264" | "hevc" | "av1" | "vp9"),
            Container::Mov => matches!(family, "h264" | "hevc" | "prores"),
        }
    }
    pub fn default_audio_codec(self) -> (&'static str, u32) {
        match self {
            Container::Webm => ("libopus", 160),
            Container::Mkv => ("libopus", 192),
            _ => ("aac", 192),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum AudioMode {
    /// Copy when compatible, otherwise re-encode with the container default.
    #[default]
    Auto,
    /// Always re-encode.
    Encode { codec: String, bitrate_kbps: u32 },
    /// No audio in the output.
    None,
}

/// What happens to audio when the video is slowed down.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SlowMotionAudio {
    /// Drop audio for slow-motion output.
    #[default]
    Mute,
    /// Stretch audio to the new duration keeping pitch (atempo).
    StretchPreservePitch,
    /// Play audio slower with lowered pitch (like tape).
    StretchLowerPitch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MuxSpec {
    /// Encoded video files to concatenate in order.
    pub video: Vec<PathBuf>,
    /// Exact duration of each video file (frames / fps). Encoders with B-frames
    /// give the first segment a start offset; explicit durations keep joins gap-free.
    #[serde(default)]
    pub durations: Option<Vec<f64>>,
    pub video_family: String,
    pub source: PathBuf,
    /// Source range used for audio/subtitles (start, duration) in seconds.
    pub range: Option<(f64, f64)>,
    pub audio: AudioMode,
    /// Source audio stream indices to keep (None = all).
    pub audio_streams: Option<Vec<usize>>,
    /// Video slowed down by this factor (1.0 = unchanged timing).
    pub slow_motion: f64,
    pub slow_motion_audio: SlowMotionAudio,
    pub subtitles: bool,
    pub metadata: bool,
    pub chapters: bool,
    /// Offset applied to the video relative to the source's audio (seconds).
    pub video_offset: f64,
    /// Expected video duration (seconds) — output is capped to this.
    pub duration: f64,
    pub container: Container,
    pub output: PathBuf,
    pub work_dir: PathBuf,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MuxReport {
    pub warnings: Vec<String>,
    pub audio_copied: bool,
    pub audio_streams: usize,
    pub subtitle_streams: usize,
}

fn atempo_chain(mut f: f64) -> Vec<String> {
    let mut v = Vec::new();
    while f < 0.5 {
        v.push("atempo=0.5".to_string());
        f /= 0.5;
    }
    while f > 2.0 {
        v.push("atempo=2.0".to_string());
        f /= 2.0;
    }
    v.push(format!("atempo={f:.6}"));
    v
}

/// Write an ffconcat list, optionally with per-file durations.
pub fn concat_list(files: &[PathBuf], durations: Option<&[f64]>, path: &Path) -> Result<(), MediaError> {
    let mut s = String::from("ffconcat version 1.0\n");
    for (i, f) in files.iter().enumerate() {
        let abs = std::fs::canonicalize(f)?;
        let p = abs.to_string_lossy().replace('\\', "/").replace('\'', "'\\''");
        s.push_str(&format!("file '{p}'\n"));
        if let Some(d) = durations.and_then(|d| d.get(i)) {
            s.push_str(&format!("duration {d:.6}\n"));
        }
    }
    std::fs::write(path, s)?;
    Ok(())
}

/// Plan the mux and return the ffmpeg arguments and report (exposed for tests).
pub fn build_mux_args(spec: &MuxSpec, info: &MediaInfo, list_file: &Path) -> (Vec<String>, MuxReport) {
    let mut report = MuxReport::default();
    let mut a: Vec<String> = vec!["-v".into(), "error".into(), "-y".into()];
    if spec.video_offset.abs() > 1e-6 {
        a.extend(["-itsoffset".into(), format!("{:.6}", spec.video_offset)]);
    }
    a.extend(["-f".into(), "concat".into(), "-safe".into(), "0".into(), "-i".into(), list_file.to_string_lossy().into_owned()]);
    if let Some((start, dur)) = spec.range {
        a.extend(["-ss".into(), format!("{start:.6}"), "-t".into(), format!("{dur:.6}")]);
    }
    a.extend(["-i".into(), spec.source.to_string_lossy().into_owned()]);
    a.extend(["-map".into(), "0:v:0".into(), "-c:v".into(), "copy".into()]);
    if spec.video_family == "hevc" && spec.container.is_isobmff() {
        a.extend(["-tag:v".into(), "hvc1".into()]);
    }

    // Audio
    let slow = spec.slow_motion > 1.0001;
    let mut audio: Vec<&crate::probe::AudioStream> = info
        .audio
        .iter()
        .filter(|s| spec.audio_streams.as_ref().map_or(true, |keep| keep.contains(&s.index)))
        .collect();
    if matches!(spec.audio, AudioMode::None) || (slow && spec.slow_motion_audio == SlowMotionAudio::Mute) {
        if slow && !audio.is_empty() && spec.audio != AudioMode::None {
            report.warnings.push("Audio was removed because slow-motion output is set to mute audio.".into());
        }
        audio.clear();
    }
    let mut all_copied = true;
    for (oi, s) in audio.iter().enumerate() {
        a.extend(["-map".into(), format!("1:{}", s.index)]);
        let mut filters: Vec<String> = Vec::new();
        if slow {
            match spec.slow_motion_audio {
                SlowMotionAudio::StretchPreservePitch => filters.extend(atempo_chain(1.0 / spec.slow_motion)),
                SlowMotionAudio::StretchLowerPitch => {
                    let sr = s.sample_rate.max(8000);
                    filters.push(format!("asetrate={:.3}", sr as f64 / spec.slow_motion));
                    filters.push(format!("aresample={sr}"));
                }
                SlowMotionAudio::Mute => {}
            }
        }
        let copy = filters.is_empty() && matches!(spec.audio, AudioMode::Auto) && spec.container.audio_copy_ok(&s.codec);
        if copy {
            a.extend([format!("-c:a:{oi}"), "copy".into()]);
        } else {
            all_copied = false;
            let (codec, br) = match &spec.audio {
                AudioMode::Encode { codec, bitrate_kbps } => (codec.clone(), *bitrate_kbps),
                _ => {
                    let (c, b) = spec.container.default_audio_codec();
                    (c.to_string(), if s.channels > 2 { b * 2 } else { b })
                }
            };
            a.extend([format!("-c:a:{oi}"), codec.clone(), format!("-b:a:{oi}"), format!("{br}k")]);
            if !filters.is_empty() {
                a.extend([format!("-filter:a:{oi}"), filters.join(",")]);
            }
            if matches!(spec.audio, AudioMode::Auto) && filters.is_empty() {
                report.warnings.push(format!(
                    "Audio track {} ({}) cannot be stored in {} without conversion; it was re-encoded to {}.",
                    oi + 1,
                    s.codec,
                    spec.container.extension().to_uppercase(),
                    codec
                ));
            }
        }
    }
    report.audio_streams = audio.len();
    report.audio_copied = all_copied && !audio.is_empty();

    // Subtitles
    if spec.subtitles && !info.subtitles.is_empty() {
        if slow {
            report.warnings.push("Subtitles were not included because slow motion changes their timing.".into());
        } else {
            let mut si = 0;
            for s in &info.subtitles {
                let codec = match spec.container {
                    Container::Mkv => Some("copy"),
                    Container::Webm if s.is_text => Some("webvtt"),
                    Container::Mp4 | Container::Mov | Container::M4v if s.is_text => Some("mov_text"),
                    _ => None,
                };
                match codec {
                    Some(c) => {
                        a.extend(["-map".into(), format!("1:{}", s.index), format!("-c:s:{si}"), c.into()]);
                        si += 1;
                    }
                    None => report.warnings.push(format!(
                        "Subtitle track ({}) is image-based and cannot be stored in {}; it was not included. Choose MKV to keep it.",
                        s.codec,
                        spec.container.extension().to_uppercase()
                    )),
                }
            }
            report.subtitle_streams = si;
        }
    }
    if spec.metadata {
        a.extend(["-map_metadata".into(), "1".into()]);
    } else {
        a.extend(["-map_metadata".into(), "-1".into()]);
    }
    if spec.chapters && !slow && spec.range.is_none() {
        a.extend(["-map_chapters".into(), "1".into()]);
    } else {
        if spec.chapters && info.chapters > 0 {
            report.warnings.push("Chapters were not included because the output timing or range differs from the source.".into());
        }
        a.extend(["-map_chapters".into(), "-1".into()]);
    }
    if spec.container.is_isobmff() {
        a.extend(["-movflags".into(), "+faststart".into()]);
    }
    a.extend(["-t".into(), format!("{:.6}", spec.duration + spec.video_offset.max(0.0))]);
    a.extend(["-f".into(), spec.container.muxer().into(), spec.output.to_string_lossy().into_owned()]);
    (a, report)
}

/// Assemble the final file.
pub fn mux(tools: &FfmpegTools, spec: &MuxSpec, info: &MediaInfo) -> Result<MuxReport, MediaError> {
    if spec.video.is_empty() {
        return Err(MediaError::Invalid("no encoded video to assemble".into()));
    }
    if !spec.container.video_ok(&spec.video_family) {
        return Err(MediaError::Invalid(format!(
            "{} video cannot be stored in {}",
            spec.video_family,
            spec.container.extension().to_uppercase()
        )));
    }
    std::fs::create_dir_all(&spec.work_dir)?;
    let list = spec.work_dir.join("concat.ffconcat");
    concat_list(&spec.video, spec.durations.as_deref(), &list)?;
    let (args, report) = build_mux_args(spec, info, &list);
    let mut c = tools.ffmpeg_cmd();
    c.args(&args);
    let (ok, _, err) = run_capture(&mut c, Duration::from_secs(6 * 3600), 1 << 20)?;
    if !ok {
        return Err(MediaError::Process(format!("assembling the output failed: {}", err.trim())));
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::parse_probe_json;

    fn info() -> MediaInfo {
        let j = r#"{"streams":[{"index":0,"codec_type":"video","codec_name":"h264","width":8,"height":8,"pix_fmt":"yuv420p","r_frame_rate":"30/1","avg_frame_rate":"30/1"},
          {"index":1,"codec_type":"audio","codec_name":"pcm_s16le","channels":2,"sample_rate":"48000"},
          {"index":2,"codec_type":"subtitle","codec_name":"hdmv_pgs_subtitle"},
          {"index":3,"codec_type":"subtitle","codec_name":"subrip"}],"format":{"duration":"2"}}"#;
        parse_probe_json("in.mkv", j, None).unwrap()
    }

    fn spec(container: Container) -> MuxSpec {
        MuxSpec {
            video: vec![PathBuf::from("a.mkv")],
            durations: None,
            video_family: "hevc".into(),
            source: PathBuf::from("in.mkv"),
            range: None,
            audio: AudioMode::Auto,
            audio_streams: None,
            slow_motion: 1.0,
            slow_motion_audio: SlowMotionAudio::Mute,
            subtitles: true,
            metadata: true,
            chapters: true,
            video_offset: 0.0,
            duration: 2.0,
            container,
            output: PathBuf::from("out"),
            work_dir: PathBuf::from("w"),
        }
    }

    #[test]
    fn mp4_converts_pcm_and_drops_bitmap_subs() {
        let (a, r) = build_mux_args(&spec(Container::Mp4), &info(), Path::new("l"));
        let s = a.join(" ");
        assert!(s.contains("-c:a:0 aac"));
        assert!(s.contains("-tag:v hvc1"));
        assert!(s.contains("-c:s:0 mov_text"));
        assert!(s.contains("+faststart"));
        assert_eq!(r.subtitle_streams, 1);
        assert!(r.warnings.iter().any(|w| w.contains("image-based")));
        assert!(r.warnings.iter().any(|w| w.contains("re-encoded")));
    }

    #[test]
    fn mkv_copies_everything() {
        let (a, r) = build_mux_args(&spec(Container::Mkv), &info(), Path::new("l"));
        let s = a.join(" ");
        assert!(s.contains("-c:a:0 copy"));
        assert_eq!(r.subtitle_streams, 2);
        assert!(r.audio_copied);
    }

    #[test]
    fn slow_motion_audio_modes() {
        let mut sp = spec(Container::Mkv);
        sp.slow_motion = 4.0;
        let (a, r) = build_mux_args(&sp, &info(), Path::new("l"));
        assert!(!a.iter().any(|x| x.starts_with("-c:a")));
        assert!(r.warnings.iter().any(|w| w.contains("slow-motion")));
        sp.slow_motion_audio = SlowMotionAudio::StretchPreservePitch;
        let (a, _) = build_mux_args(&sp, &info(), Path::new("l"));
        let s = a.join(" ");
        assert!(s.contains("atempo=0.5,atempo=0.500000"), "{s}");
        sp.slow_motion_audio = SlowMotionAudio::StretchLowerPitch;
        let (a, _) = build_mux_args(&sp, &info(), Path::new("l"));
        assert!(a.join(" ").contains("asetrate=12000.000,aresample=48000"));
    }

    #[test]
    fn atempo_bounds() {
        assert_eq!(atempo_chain(0.25), vec!["atempo=0.5", "atempo=0.500000"]);
        assert_eq!(atempo_chain(1.0), vec!["atempo=1.000000"]);
    }
}
