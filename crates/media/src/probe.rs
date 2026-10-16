//! Media probing with ffprobe.

use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::rational::Rational;
use crate::tools::{run_capture, FfmpegTools};
use crate::MediaError;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ColorInfo {
    pub range: Option<String>,
    pub space: Option<String>,
    pub transfer: Option<String>,
    pub primaries: Option<String>,
    pub chroma_location: Option<String>,
    /// e.g. "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)"
    pub mastering_display: Option<String>,
    /// "MaxCLL,MaxFALL"
    pub content_light: Option<String>,
}

impl ColorInfo {
    pub fn is_hdr(&self) -> bool {
        matches!(self.transfer.as_deref(), Some("smpte2084") | Some("arib-std-b67"))
    }
    pub fn hdr_kind(&self) -> Option<&'static str> {
        match self.transfer.as_deref() {
            Some("smpte2084") => Some("HDR10 (PQ)"),
            Some("arib-std-b67") => Some("HLG"),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoStream {
    pub index: usize,
    pub codec: String,
    pub codec_long: String,
    pub profile: Option<String>,
    pub width: u32,
    pub height: u32,
    /// Display dimensions after applying rotation.
    pub display_width: u32,
    pub display_height: u32,
    pub sample_aspect_ratio: Option<String>,
    pub display_aspect_ratio: Option<String>,
    pub pix_fmt: String,
    pub bit_depth: u32,
    pub frame_rate: Rational,
    pub avg_frame_rate: Rational,
    pub variable_frame_rate: bool,
    pub frame_count: Option<u64>,
    pub field_order: Option<String>,
    pub rotation: i32,
    pub start_time: f64,
    pub duration: Option<f64>,
    pub bit_rate: Option<u64>,
    pub color: ColorInfo,
}

impl VideoStream {
    pub fn is_interlaced_flagged(&self) -> bool {
        matches!(self.field_order.as_deref(), Some("tt") | Some("bb") | Some("tb") | Some("bt"))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioStream {
    pub index: usize,
    pub codec: String,
    pub channels: u32,
    pub channel_layout: Option<String>,
    pub sample_rate: u32,
    pub bit_rate: Option<u64>,
    pub language: Option<String>,
    pub title: Option<String>,
    pub start_time: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtitleStream {
    pub index: usize,
    pub codec: String,
    pub language: Option<String>,
    pub title: Option<String>,
    /// Text-based subtitles can be converted between containers; bitmap ones cannot.
    pub is_text: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaInfo {
    pub path: String,
    pub container: String,
    pub container_long: String,
    pub duration: f64,
    pub size_bytes: u64,
    pub bit_rate: Option<u64>,
    pub start_time: f64,
    pub video: Option<VideoStream>,
    pub audio: Vec<AudioStream>,
    pub subtitles: Vec<SubtitleStream>,
    pub chapters: usize,
    pub title: Option<String>,
}

impl MediaInfo {
    pub fn video(&self) -> Result<&VideoStream, MediaError> {
        self.video.as_ref().ok_or_else(|| MediaError::Invalid("the file has no video stream".into()))
    }

    /// Estimated number of frames.
    pub fn estimated_frames(&self) -> u64 {
        match &self.video {
            Some(v) => v.frame_count.unwrap_or_else(|| crate::rational::frames_for(v.duration.unwrap_or(self.duration), v.frame_rate)),
            None => 0,
        }
    }
}

fn s(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(|x| x.as_str()).map(|x| x.to_string()).filter(|x| !x.is_empty() && x != "unknown")
}
fn f(v: &Value, k: &str) -> Option<f64> {
    v.get(k).and_then(|x| x.as_str().and_then(|s| s.parse().ok()).or_else(|| x.as_f64()))
}
fn u(v: &Value, k: &str) -> Option<u64> {
    v.get(k).and_then(|x| x.as_str().and_then(|s| s.parse().ok()).or_else(|| x.as_u64()))
}
/// Stream duration: the `duration` field, or Matroska's `DURATION` tag ("HH:MM:SS.fraction"),
/// which is the only per-stream length MKV/WebM files carry.
fn stream_duration(st: &Value) -> Option<f64> {
    f(st, "duration").filter(|d| *d > 0.0).or_else(|| {
        let t = tag(st, "DURATION")?;
        let mut parts = t.trim().split(':').rev();
        let secs: f64 = parts.next()?.parse().ok()?;
        let mins: f64 = parts.next().unwrap_or("0").parse().ok()?;
        let hours: f64 = parts.next().unwrap_or("0").parse().ok()?;
        Some(hours * 3600.0 + mins * 60.0 + secs).filter(|d| *d > 0.0)
    })
}

fn tag(v: &Value, k: &str) -> Option<String> {
    v.get("tags").and_then(|t| t.as_object().and_then(|o| o.iter().find(|(kk, _)| kk.eq_ignore_ascii_case(k)).and_then(|(_, vv)| vv.as_str().map(String::from))))
}

fn bit_depth_from_pix_fmt(p: &str) -> u32 {
    for (pat, d) in [("p16", 16), ("p14", 14), ("p12", 12), ("p10", 10), ("p9", 9), ("48", 16), ("64", 16), ("f32", 32)] {
        if p.contains(pat) {
            return d;
        }
    }
    8
}

fn text_sub(codec: &str) -> bool {
    matches!(codec, "subrip" | "srt" | "ass" | "ssa" | "mov_text" | "webvtt" | "text" | "microdvd" | "subviewer")
}

/// Parse ffprobe `-show_format -show_streams` JSON.
pub fn parse_probe_json(path: &str, json: &str, first_frame_json: Option<&str>) -> Result<MediaInfo, MediaError> {
    let v: Value = serde_json::from_str(json).map_err(|e| MediaError::Invalid(format!("ffprobe output: {e}")))?;
    let fmt = v.get("format").cloned().unwrap_or(Value::Null);
    let streams = v.get("streams").and_then(|s| s.as_array()).cloned().unwrap_or_default();
    let mut video = None;
    let mut audio = Vec::new();
    let mut subtitles = Vec::new();
    for st in &streams {
        let index = st.get("index").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
        match st.get("codec_type").and_then(|x| x.as_str()) {
            Some("video") if video.is_none() => {
                // Skip cover art / attached pictures.
                if st.pointer("/disposition/attached_pic").and_then(|x| x.as_i64()) == Some(1) {
                    continue;
                }
                let width = u(st, "width").unwrap_or(0) as u32;
                let height = u(st, "height").unwrap_or(0) as u32;
                let mut rotation = 0i32;
                if let Some(sd) = st.get("side_data_list").and_then(|x| x.as_array()) {
                    for d in sd {
                        if let Some(r) = d.get("rotation").and_then(|x| x.as_i64().or_else(|| x.as_f64().map(|f| f as i64))) {
                            rotation = r as i32;
                        }
                    }
                }
                if rotation == 0 {
                    if let Some(r) = tag(st, "rotate").and_then(|r| r.parse::<i32>().ok()) {
                        rotation = -r;
                    }
                }
                let rotation = ((rotation % 360) + 360) % 360;
                let (dw, dh) = if rotation == 90 || rotation == 270 { (height, width) } else { (width, height) };
                let fr = s(st, "r_frame_rate").and_then(|x| Rational::parse(&x)).filter(|r| r.is_valid());
                let afr = s(st, "avg_frame_rate").and_then(|x| Rational::parse(&x)).filter(|r| r.is_valid());
                let frame_rate = fr.or(afr).unwrap_or(Rational::new(25, 1));
                let avg_frame_rate = afr.unwrap_or(frame_rate);
                let vfr = (frame_rate.as_f64() - avg_frame_rate.as_f64()).abs() / frame_rate.as_f64().max(1e-9) > 0.01;
                let pix_fmt = s(st, "pix_fmt").unwrap_or_else(|| "unknown".into());
                let bit_depth = u(st, "bits_per_raw_sample").map(|b| b as u32).filter(|b| *b > 0).unwrap_or_else(|| bit_depth_from_pix_fmt(&pix_fmt));
                let color = ColorInfo {
                    range: s(st, "color_range"),
                    space: s(st, "color_space"),
                    transfer: s(st, "color_transfer"),
                    primaries: s(st, "color_primaries"),
                    chroma_location: s(st, "chroma_location"),
                    mastering_display: None,
                    content_light: None,
                };
                video = Some(VideoStream {
                    index,
                    codec: s(st, "codec_name").unwrap_or_default(),
                    codec_long: s(st, "codec_long_name").unwrap_or_default(),
                    profile: s(st, "profile"),
                    width,
                    height,
                    display_width: dw,
                    display_height: dh,
                    sample_aspect_ratio: s(st, "sample_aspect_ratio").filter(|x| x != "0:1"),
                    display_aspect_ratio: s(st, "display_aspect_ratio").filter(|x| x != "0:1"),
                    pix_fmt,
                    bit_depth,
                    frame_rate,
                    avg_frame_rate,
                    variable_frame_rate: vfr,
                    frame_count: u(st, "nb_frames").filter(|n| *n > 0),
                    field_order: s(st, "field_order"),
                    rotation,
                    start_time: f(st, "start_time").unwrap_or(0.0),
                    duration: stream_duration(st),
                    bit_rate: u(st, "bit_rate"),
                    color,
                });
            }
            Some("audio") => audio.push(AudioStream {
                index,
                codec: s(st, "codec_name").unwrap_or_default(),
                channels: u(st, "channels").unwrap_or(0) as u32,
                channel_layout: s(st, "channel_layout"),
                sample_rate: u(st, "sample_rate").unwrap_or(0) as u32,
                bit_rate: u(st, "bit_rate"),
                language: tag(st, "language").filter(|l| l != "und"),
                title: tag(st, "title"),
                start_time: f(st, "start_time").unwrap_or(0.0),
            }),
            Some("subtitle") => {
                let codec = s(st, "codec_name").unwrap_or_default();
                subtitles.push(SubtitleStream { index, is_text: text_sub(&codec), codec, language: tag(st, "language").filter(|l| l != "und"), title: tag(st, "title") })
            }
            _ => {}
        }
    }
    // HDR side data lives on frames.
    if let (Some(vs), Some(ff)) = (video.as_mut(), first_frame_json) {
        if let Ok(fv) = serde_json::from_str::<Value>(ff) {
            if let Some(frame) = fv.get("frames").and_then(|x| x.as_array()).and_then(|a| a.first()) {
                if let Some(sd) = frame.get("side_data_list").and_then(|x| x.as_array()) {
                    for d in sd {
                        match d.get("side_data_type").and_then(|x| x.as_str()) {
                            Some("Mastering display metadata") => {
                                let g = |k: &str| -> i64 {
                                    d.get(k)
                                        .and_then(|x| x.as_str())
                                        .and_then(|s| {
                                            let (a, b) = s.split_once('/')?;
                                            let (a, b) = (a.parse::<f64>().ok()?, b.parse::<f64>().ok()?);
                                            Some(a / b)
                                        })
                                        .unwrap_or(0.0) as i64
                                };
                                let c = |k: &str, scale: f64| -> i64 {
                                    d.get(k)
                                        .and_then(|x| x.as_str())
                                        .and_then(|s| {
                                            let (a, b) = s.split_once('/')?;
                                            Some(a.parse::<f64>().ok()? / b.parse::<f64>().ok()? * scale)
                                        })
                                        .unwrap_or(0.0)
                                        .round() as i64
                                };
                                let _ = g;
                                vs.color.mastering_display = Some(format!(
                                    "G({},{})B({},{})R({},{})WP({},{})L({},{})",
                                    c("green_x", 50000.0),
                                    c("green_y", 50000.0),
                                    c("blue_x", 50000.0),
                                    c("blue_y", 50000.0),
                                    c("red_x", 50000.0),
                                    c("red_y", 50000.0),
                                    c("white_point_x", 50000.0),
                                    c("white_point_y", 50000.0),
                                    c("max_luminance", 10000.0),
                                    c("min_luminance", 10000.0)
                                ));
                            }
                            Some("Content light level metadata") => {
                                let a = d.get("max_content").and_then(|x| x.as_u64()).unwrap_or(0);
                                let b = d.get("max_average").and_then(|x| x.as_u64()).unwrap_or(0);
                                vs.color.content_light = Some(format!("{a},{b}"));
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }
    let size_bytes = u(&fmt, "size").unwrap_or(0);
    Ok(MediaInfo {
        path: path.to_string(),
        container: s(&fmt, "format_name").unwrap_or_default(),
        container_long: s(&fmt, "format_long_name").unwrap_or_default(),
        duration: f(&fmt, "duration").or_else(|| video.as_ref().and_then(|v| v.duration)).unwrap_or(0.0),
        size_bytes,
        bit_rate: u(&fmt, "bit_rate"),
        start_time: f(&fmt, "start_time").unwrap_or(0.0),
        video,
        audio,
        subtitles,
        chapters: v.get("chapters").and_then(|c| c.as_array()).map(|a| a.len()).unwrap_or(0),
        title: tag(&fmt, "title"),
    })
}

/// Probe a media file.
pub fn probe(tools: &FfmpegTools, path: &Path) -> Result<MediaInfo, MediaError> {
    if !path.is_file() {
        return Err(MediaError::Invalid(format!("file not found: {}", path.display())));
    }
    let mut c = tools.ffprobe_cmd();
    c.args(["-v", "error", "-print_format", "json", "-show_format", "-show_streams", "-show_chapters", "-i"]).arg(path);
    let (ok, out, err) = run_capture(&mut c, Duration::from_secs(60), 32 << 20)?;
    if !ok {
        return Err(MediaError::Invalid(format!("not a readable media file: {}", err.trim())));
    }
    let mut c = tools.ffprobe_cmd();
    c.args(["-v", "error", "-print_format", "json", "-select_streams", "v:0", "-read_intervals", "%+#1", "-show_frames", "-show_entries", "frame=side_data_list", "-i"]).arg(path);
    let ff = run_capture(&mut c, Duration::from_secs(60), 4 << 20).ok().filter(|r| r.0).map(|r| r.1);
    let mut info = parse_probe_json(&path.to_string_lossy(), &out, ff.as_deref())?;
    if let Some(v) = info.video.as_mut() {
        let mut c = tools.ffprobe_cmd();
        c.args(["-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#400", "-show_entries", "packet=pts_time", "-of", "csv=p=0", "-i"]).arg(path);
        if let Ok((true, pk, _)) = run_capture(&mut c, Duration::from_secs(60), 1 << 20) {
            let mut ts: Vec<f64> = pk.lines().filter_map(|l| l.trim().trim_end_matches(',').parse().ok()).collect();
            if let Some((vfr, avg)) = analyse_timestamps(&mut ts) {
                if vfr {
                    v.variable_frame_rate = true;
                    v.avg_frame_rate = avg;
                }
            }
        }
    }
    Ok(info)
}

/// Decide whether presentation timestamps are irregular. Returns (is_vfr, average rate).
pub fn analyse_timestamps(ts: &mut [f64]) -> Option<(bool, Rational)> {
    if ts.len() < 8 {
        return None;
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let deltas: Vec<f64> = ts.windows(2).map(|w| w[1] - w[0]).filter(|d| *d > 1e-6).collect();
    if deltas.len() < 6 {
        return None;
    }
    let mut sorted = deltas.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo = sorted[sorted.len() / 10];
    let hi = sorted[sorted.len() * 9 / 10];
    let span = ts[ts.len() - 1] - ts[0];
    let avg = Rational::from_f64((ts.len() - 1) as f64 / span.max(1e-6));
    Some((hi > lo * 1.3 + 0.0005, avg))
}

/// Result of FFmpeg's `idet` interlace analysis.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct InterlaceAnalysis {
    pub tff: u64,
    pub bff: u64,
    pub progressive: u64,
    pub undetermined: u64,
}

impl InterlaceAnalysis {
    pub fn likely_interlaced(&self) -> bool {
        let inter = self.tff + self.bff;
        inter > 0 && inter as f64 > 0.3 * (inter + self.progressive) as f64
    }
    pub fn field_order(&self) -> &'static str {
        if self.bff > self.tff {
            "bff"
        } else {
            "tff"
        }
    }
}

/// Parse the "Multi frame detection" line printed by `idet`.
pub fn parse_idet(stderr: &str) -> Option<InterlaceAnalysis> {
    let line = stderr.lines().rev().find(|l| l.contains("Multi frame detection:"))?;
    let num = |k: &str| -> u64 { line.split(k).nth(1).and_then(|r| r.split_whitespace().next()).and_then(|v| v.parse().ok()).unwrap_or(0) };
    Some(InterlaceAnalysis { tff: num("TFF:"), bff: num("BFF:"), progressive: num("Progressive:"), undetermined: num("Undetermined:") })
}

/// Analyse up to `frames` frames for interlacing.
pub fn detect_interlace(tools: &FfmpegTools, path: &Path, frames: u32) -> Result<InterlaceAnalysis, MediaError> {
    let mut c = tools.ffmpeg_cmd();
    c.args(["-v", "info", "-i"]).arg(path).args(["-map", "0:v:0", "-vf", "idet", "-frames:v", &frames.to_string(), "-an", "-f", "null", "-"]);
    let (_, _, err) = run_capture(&mut c, Duration::from_secs(300), 8 << 20)?;
    parse_idet(&err).ok_or_else(|| MediaError::Invalid("interlace analysis produced no result".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{"streams":[{"index":0,"codec_name":"h264","codec_type":"video","width":1920,"height":1080,
        "pix_fmt":"yuv420p10le","r_frame_rate":"30000/1001","avg_frame_rate":"30000/1001","field_order":"tt",
        "color_range":"tv","color_space":"bt2020nc","color_transfer":"smpte2084","color_primaries":"bt2020",
        "side_data_list":[{"side_data_type":"Display Matrix","rotation":-90}],"nb_frames":"300","start_time":"0.033"},
        {"index":1,"codec_name":"aac","codec_type":"audio","channels":2,"sample_rate":"48000","tags":{"language":"eng"}},
        {"index":2,"codec_name":"hdmv_pgs_subtitle","codec_type":"subtitle"},
        {"index":3,"codec_name":"subrip","codec_type":"subtitle","tags":{"language":"fra"}}],
        "format":{"format_name":"matroska,webm","duration":"10.01","size":"12345","start_time":"0.000"}}"#;

    #[test]
    fn parses_streams() {
        let m = parse_probe_json("x.mkv", SAMPLE, None).unwrap();
        let v = m.video.as_ref().unwrap();
        assert_eq!(v.bit_depth, 10);
        assert_eq!(v.rotation, 270);
        assert_eq!((v.display_width, v.display_height), (1080, 1920));
        assert!(v.is_interlaced_flagged());
        assert!(v.color.is_hdr());
        assert_eq!(v.frame_rate, Rational::new(30000, 1001));
        assert!(!v.variable_frame_rate);
        assert_eq!(m.audio[0].language.as_deref(), Some("eng"));
        assert!(!m.subtitles[0].is_text);
        assert!(m.subtitles[1].is_text);
        assert_eq!(m.estimated_frames(), 300);
    }

    #[test]
    fn detects_vfr() {
        let j = r#"{"streams":[{"index":0,"codec_type":"video","codec_name":"h264","width":8,"height":8,"pix_fmt":"yuv420p",
            "r_frame_rate":"60/1","avg_frame_rate":"2997/100"}],"format":{"duration":"1"}}"#;
        assert!(parse_probe_json("x", j, None).unwrap().video.unwrap().variable_frame_rate);
    }

    #[test]
    fn matroska_stream_duration_tag_wins_over_container() {
        // A subtitle running past the video makes the container longer than the video.
        let j = r#"{"streams":[{"index":0,"codec_type":"video","codec_name":"h264","width":8,"height":8,"pix_fmt":"yuv420p",
            "r_frame_rate":"24/1","avg_frame_rate":"24/1","tags":{"DURATION":"00:00:00.500000000"}},
            {"index":1,"codec_type":"subtitle","codec_name":"subrip","tags":{"DURATION":"00:00:00.800000000"}}],"format":{"duration":"0.8"}}"#;
        let m = parse_probe_json("x.mkv", j, None).unwrap();
        assert_eq!(m.video.unwrap().duration, Some(0.5));
        let j2 = j.replace("00:00:00.500000000", "01:02:03.250000000");
        assert_eq!(parse_probe_json("x.mkv", &j2, None).unwrap().video.unwrap().duration, Some(3723.25));
    }

    #[test]
    fn timestamp_regularity() {
        let mut cfr: Vec<f64> = (0..50).map(|i| i as f64 / 30.0).collect();
        assert!(!analyse_timestamps(&mut cfr).unwrap().0);
        let mut vfr: Vec<f64> = (0..60).filter(|i| i % 3 != 0).map(|i| i as f64 / 30.0).collect();
        let (is_vfr, avg) = analyse_timestamps(&mut vfr).unwrap();
        assert!(is_vfr);
        assert!((avg.as_f64() - 20.0).abs() < 1.0);
    }

    #[test]
    fn parses_idet() {
        let s = "[Parsed_idet_0 @ 0x1] Single frame detection: TFF: 1 BFF: 0\n[Parsed_idet_0 @ 0x1] Multi frame detection: TFF:    90 BFF:     2 Progressive:     8 Undetermined:     0";
        let a = parse_idet(s).unwrap();
        assert_eq!(a.tff, 90);
        assert!(a.likely_interlaced());
        assert_eq!(a.field_order(), "tff");
    }

    #[test]
    fn parses_hdr_side_data() {
        let ff = r#"{"frames":[{"side_data_list":[{"side_data_type":"Mastering display metadata","red_x":"34000/50000","red_y":"16000/50000",
            "green_x":"13250/50000","green_y":"34500/50000","blue_x":"7500/50000","blue_y":"3000/50000","white_point_x":"15635/50000",
            "white_point_y":"16450/50000","min_luminance":"50/10000","max_luminance":"10000000/10000"},
            {"side_data_type":"Content light level metadata","max_content":1000,"max_average":400}]}]}"#;
        let m = parse_probe_json("x", SAMPLE, Some(ff)).unwrap();
        let c = &m.video.unwrap().color;
        assert_eq!(c.mastering_display.as_deref(), Some("G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)"));
        assert_eq!(c.content_light.as_deref(), Some("1000,400"));
    }
}
