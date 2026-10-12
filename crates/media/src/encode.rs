//! Video codec profiles and encoder arguments.

use serde::{Deserialize, Serialize};

use crate::color::ColorTags;
use crate::tools::FfmpegTools;

/// Rate control.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum RateControl {
    /// Constant quality (CRF / CQ / q:v depending on encoder). Lower is better.
    Quality { value: f32 },
    /// Average bitrate in kbit/s.
    Bitrate { kbps: u32 },
    /// Mathematically lossless where the encoder supports it.
    Lossless,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoEncodeSettings {
    pub encoder: String,
    pub rate: RateControl,
    pub preset: Option<String>,
    pub pix_fmt: String,
    /// Encoder profile (e.g. ProRes 0-5).
    pub profile: Option<String>,
    /// Max keyframe interval in frames (None = encoder default).
    pub gop: Option<u32>,
}

/// Static description of a supported encoder for the UI and validation.
#[derive(Debug, Clone, Serialize)]
pub struct CodecProfile {
    pub encoder: &'static str,
    pub label: &'static str,
    pub family: &'static str,
    pub hardware: bool,
    pub pix_fmts: &'static [&'static str],
    pub presets: &'static [&'static str],
    pub default_preset: Option<&'static str>,
    /// (min, max, default) quality value; lower = better.
    pub quality_range: Option<(f32, f32, f32)>,
    pub supports_bitrate: bool,
    pub supports_lossless: bool,
    pub containers: &'static [&'static str],
    pub description: &'static str,
}

const X26X_PRESETS: &[&str] = &["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"];
const NV_PRESETS: &[&str] = &["p1", "p2", "p3", "p4", "p5", "p6", "p7"];

pub const CODECS: &[CodecProfile] = &[
    CodecProfile {
        encoder: "libx264",
        label: "H.264 (x264)",
        family: "h264",
        hardware: false,
        pix_fmts: &["yuv420p", "yuv422p", "yuv444p", "yuv420p10le", "yuv422p10le", "yuv444p10le"],
        presets: X26X_PRESETS,
        default_preset: Some("medium"),
        quality_range: Some((0.0, 51.0, 18.0)),
        supports_bitrate: true,
        supports_lossless: true,
        containers: &["mp4", "mov", "mkv", "m4v"],
        description: "Most compatible. 8-bit 4:2:0 plays everywhere; 10-bit has limited hardware playback support.",
    },
    CodecProfile {
        encoder: "libx265",
        label: "H.265 / HEVC (x265)",
        family: "hevc",
        hardware: false,
        pix_fmts: &["yuv420p", "yuv420p10le", "yuv422p10le", "yuv444p10le", "yuv420p12le"],
        presets: X26X_PRESETS,
        default_preset: Some("medium"),
        quality_range: Some((0.0, 51.0, 20.0)),
        supports_bitrate: true,
        supports_lossless: true,
        containers: &["mp4", "mov", "mkv", "m4v"],
        description: "About half the size of H.264 at similar quality; supports 10-bit and HDR10.",
    },
    CodecProfile {
        encoder: "libsvtav1",
        label: "AV1 (SVT-AV1)",
        family: "av1",
        hardware: false,
        pix_fmts: &["yuv420p", "yuv420p10le"],
        presets: &["2", "4", "6", "8", "10", "12"],
        default_preset: Some("6"),
        quality_range: Some((1.0, 63.0, 28.0)),
        supports_bitrate: true,
        supports_lossless: false,
        containers: &["mp4", "mkv", "webm"],
        description: "Best compression; slower to encode and needs a recent player.",
    },
    CodecProfile {
        encoder: "libaom-av1",
        label: "AV1 (libaom)",
        family: "av1",
        hardware: false,
        pix_fmts: &["yuv420p", "yuv420p10le", "yuv444p", "yuv444p10le"],
        presets: &["0", "2", "4", "6", "8"],
        default_preset: Some("6"),
        quality_range: Some((0.0, 63.0, 28.0)),
        supports_bitrate: true,
        supports_lossless: true,
        containers: &["mp4", "mkv", "webm"],
        description: "Reference AV1 encoder; very slow.",
    },
    CodecProfile {
        encoder: "libvpx-vp9",
        label: "VP9 (libvpx)",
        family: "vp9",
        hardware: false,
        pix_fmts: &["yuv420p", "yuv420p10le", "yuv444p"],
        presets: &["0", "1", "2", "3", "4", "5"],
        default_preset: Some("2"),
        quality_range: Some((0.0, 63.0, 24.0)),
        supports_bitrate: true,
        supports_lossless: true,
        containers: &["webm", "mkv", "mp4"],
        description: "Royalty-free web codec for WebM.",
    },
    CodecProfile {
        encoder: "prores_ks",
        label: "Apple ProRes (prores_ks)",
        family: "prores",
        hardware: false,
        pix_fmts: &["yuv422p10le", "yuv444p10le"],
        presets: &["0", "1", "2", "3", "4", "5"],
        default_preset: Some("3"),
        quality_range: None,
        supports_bitrate: false,
        supports_lossless: false,
        containers: &["mov", "mkv"],
        description: "Intermediate codec for editing. Presets: 0 Proxy, 1 LT, 2 Standard, 3 HQ, 4 4444, 5 4444 XQ. Very large files.",
    },
    CodecProfile {
        encoder: "ffv1",
        label: "FFV1 (lossless)",
        family: "ffv1",
        hardware: false,
        pix_fmts: &["yuv420p", "yuv422p10le", "yuv444p10le", "yuv444p16le", "gbrp10le", "gbrp16le"],
        presets: &[],
        default_preset: None,
        quality_range: None,
        supports_bitrate: false,
        supports_lossless: true,
        containers: &["mkv"],
        description: "Archival lossless codec. Huge files; limited player support.",
    },
    CodecProfile {
        encoder: "h264_nvenc",
        label: "H.264 (NVIDIA NVENC)",
        family: "h264",
        hardware: true,
        pix_fmts: &["yuv420p", "yuv444p"],
        presets: NV_PRESETS,
        default_preset: Some("p5"),
        quality_range: Some((0.0, 51.0, 20.0)),
        supports_bitrate: true,
        supports_lossless: false,
        containers: &["mp4", "mov", "mkv", "m4v"],
        description: "Hardware encoder on NVIDIA GPUs. Fast, slightly larger files than x264.",
    },
    CodecProfile {
        encoder: "hevc_nvenc",
        label: "HEVC (NVIDIA NVENC)",
        family: "hevc",
        hardware: true,
        pix_fmts: &["yuv420p", "p010le", "yuv444p"],
        presets: NV_PRESETS,
        default_preset: Some("p5"),
        quality_range: Some((0.0, 51.0, 22.0)),
        supports_bitrate: true,
        supports_lossless: false,
        containers: &["mp4", "mov", "mkv", "m4v"],
        description: "Hardware HEVC on NVIDIA GPUs, supports 10-bit (p010).",
    },
    CodecProfile {
        encoder: "av1_nvenc",
        label: "AV1 (NVIDIA NVENC)",
        family: "av1",
        hardware: true,
        pix_fmts: &["yuv420p", "p010le"],
        presets: NV_PRESETS,
        default_preset: Some("p5"),
        quality_range: Some((0.0, 51.0, 28.0)),
        supports_bitrate: true,
        supports_lossless: false,
        containers: &["mp4", "mkv", "webm"],
        description: "Hardware AV1 on RTX 40-series and newer.",
    },
    CodecProfile {
        encoder: "hevc_videotoolbox",
        label: "HEVC (Apple VideoToolbox)",
        family: "hevc",
        hardware: true,
        pix_fmts: &["yuv420p", "p010le"],
        presets: &[],
        default_preset: None,
        quality_range: Some((1.0, 100.0, 65.0)),
        supports_bitrate: true,
        supports_lossless: false,
        containers: &["mp4", "mov", "m4v"],
        description: "Hardware HEVC on Macs. Quality scale is 1-100 where higher is better.",
    },
    CodecProfile {
        encoder: "h264_videotoolbox",
        label: "H.264 (Apple VideoToolbox)",
        family: "h264",
        hardware: true,
        pix_fmts: &["yuv420p"],
        presets: &[],
        default_preset: None,
        quality_range: Some((1.0, 100.0, 65.0)),
        supports_bitrate: true,
        supports_lossless: false,
        containers: &["mp4", "mov", "m4v"],
        description: "Hardware H.264 on Macs. Quality scale is 1-100 where higher is better.",
    },
    CodecProfile {
        encoder: "hevc_qsv",
        label: "HEVC (Intel Quick Sync)",
        family: "hevc",
        hardware: true,
        pix_fmts: &["nv12", "p010le"],
        presets: &["veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
        default_preset: Some("medium"),
        quality_range: Some((1.0, 51.0, 22.0)),
        supports_bitrate: true,
        supports_lossless: false,
        containers: &["mp4", "mov", "mkv", "m4v"],
        description: "Hardware HEVC on Intel GPUs.",
    },
    CodecProfile {
        encoder: "h264_amf",
        label: "H.264 (AMD AMF)",
        family: "h264",
        hardware: true,
        pix_fmts: &["yuv420p", "nv12"],
        presets: &["speed", "balanced", "quality"],
        default_preset: Some("quality"),
        quality_range: Some((0.0, 51.0, 20.0)),
        supports_bitrate: true,
        supports_lossless: false,
        containers: &["mp4", "mov", "mkv", "m4v"],
        description: "Hardware H.264 on AMD GPUs (Windows).",
    },
    CodecProfile {
        encoder: "hevc_amf",
        label: "HEVC (AMD AMF)",
        family: "hevc",
        hardware: true,
        pix_fmts: &["yuv420p", "nv12", "p010le"],
        presets: &["speed", "balanced", "quality"],
        default_preset: Some("quality"),
        quality_range: Some((0.0, 51.0, 22.0)),
        supports_bitrate: true,
        supports_lossless: false,
        containers: &["mp4", "mov", "mkv", "m4v"],
        description: "Hardware HEVC on AMD GPUs (Windows).",
    },
];

pub fn codec_profile(encoder: &str) -> Option<&'static CodecProfile> {
    CODECS.iter().find(|c| c.encoder == encoder)
}

/// Codecs present in this FFmpeg build. Hardware encoders are only listed when a test encode succeeds.
pub fn available_codecs(tools: &FfmpegTools, test_hardware: bool) -> Vec<&'static CodecProfile> {
    CODECS.iter().filter(|c| tools.has_video_encoder(c.encoder)).filter(|c| !c.hardware || (test_hardware && tools.test_encoder(c.encoder))).collect()
}

pub fn pix_fmt_bit_depth(p: &str) -> u32 {
    if p.contains("16") {
        16
    } else if p.contains("12") {
        12
    } else if p.contains("10") || p == "p010le" {
        10
    } else {
        8
    }
}

impl VideoEncodeSettings {
    pub fn x264_default() -> Self {
        Self { encoder: "libx264".into(), rate: RateControl::Quality { value: 18.0 }, preset: Some("medium".into()), pix_fmt: "yuv420p".into(), profile: None, gop: None }
    }

    /// Lossless intermediate used for multi-pass work files.
    pub fn intermediate() -> Self {
        Self { encoder: "ffv1".into(), rate: RateControl::Lossless, preset: None, pix_fmt: "yuv444p16le".into(), profile: None, gop: Some(1) }
    }

    /// Fast, browser-playable preview encode.
    pub fn preview() -> Self {
        Self { encoder: "libx264".into(), rate: RateControl::Quality { value: 14.0 }, preset: Some("veryfast".into()), pix_fmt: "yuv420p".into(), profile: None, gop: None }
    }

    /// Encoder arguments (codec, rate control, preset, HDR parameters).
    pub fn args(&self, color: &ColorTags) -> Vec<String> {
        let mut a: Vec<String> = vec!["-c:v".into(), self.encoder.clone()];
        let push = |a: &mut Vec<String>, k: &str, v: String| {
            a.push(k.to_string());
            a.push(v);
        };
        let e = self.encoder.as_str();
        match (&self.rate, e) {
            (RateControl::Quality { value }, "libx264" | "libx265") => push(&mut a, "-crf", format!("{value}")),
            (RateControl::Quality { value }, "libsvtav1" | "libaom-av1" | "libvpx-vp9") => {
                push(&mut a, "-crf", format!("{}", value.round() as i32));
                if e != "libsvtav1" {
                    push(&mut a, "-b:v", "0".into());
                }
            }
            (RateControl::Quality { value }, e) if e.ends_with("_nvenc") => {
                push(&mut a, "-rc", "vbr".into());
                push(&mut a, "-cq", format!("{}", value.round() as i32));
                push(&mut a, "-b:v", "0".into());
            }
            (RateControl::Quality { value }, e) if e.ends_with("_videotoolbox") => push(&mut a, "-q:v", format!("{}", value.round() as i32)),
            (RateControl::Quality { value }, e) if e.ends_with("_qsv") => push(&mut a, "-global_quality", format!("{}", value.round() as i32)),
            (RateControl::Quality { value }, e) if e.ends_with("_amf") => {
                push(&mut a, "-rc", "cqp".into());
                push(&mut a, "-qp_i", format!("{}", value.round() as i32));
                push(&mut a, "-qp_p", format!("{}", value.round() as i32));
            }
            (RateControl::Quality { .. }, _) => {}
            (RateControl::Bitrate { kbps }, _) => {
                push(&mut a, "-b:v", format!("{kbps}k"));
                push(&mut a, "-maxrate", format!("{}k", (*kbps as f32 * 1.5) as u32));
                push(&mut a, "-bufsize", format!("{}k", kbps * 2));
            }
            (RateControl::Lossless, "libx264") => push(&mut a, "-qp", "0".into()),
            (RateControl::Lossless, "libx265") => {}
            (RateControl::Lossless, "libvpx-vp9") => push(&mut a, "-lossless", "1".into()),
            (RateControl::Lossless, "libaom-av1") => push(&mut a, "-crf", "0".into()),
            (RateControl::Lossless, "ffv1") => {
                push(&mut a, "-level", "3".into());
                push(&mut a, "-slicecrc", "1".into());
            }
            (RateControl::Lossless, _) => {}
        }
        if let Some(p) = &self.preset {
            match e {
                "libx264" | "libx265" => push(&mut a, "-preset", p.clone()),
                "libsvtav1" => push(&mut a, "-preset", p.clone()),
                "libaom-av1" => push(&mut a, "-cpu-used", p.clone()),
                "libvpx-vp9" => {
                    push(&mut a, "-cpu-used", p.clone());
                    push(&mut a, "-row-mt", "1".into());
                }
                "prores_ks" => push(&mut a, "-profile:v", p.clone()),
                e if e.ends_with("_nvenc") || e.ends_with("_qsv") => push(&mut a, "-preset", p.clone()),
                e if e.ends_with("_amf") => push(&mut a, "-quality", p.clone()),
                _ => {}
            }
        }
        if let Some(p) = &self.profile {
            push(&mut a, "-profile:v", p.clone());
        }
        if let Some(g) = self.gop {
            push(&mut a, "-g", g.to_string());
        }
        // x265: lossless flag and HDR10 static metadata.
        if e == "libx265" {
            let mut params: Vec<String> = Vec::new();
            if matches!(self.rate, RateControl::Lossless) {
                params.push("lossless=1".into());
            }
            if color.is_hdr() {
                params.push("hdr10=1".into());
                params.push("repeat-headers=1".into());
                if let Some(p) = &color.primaries {
                    params.push(format!("colorprim={p}"));
                }
                if let Some(t) = &color.transfer {
                    params.push(format!("transfer={t}"));
                }
                params.push(format!("colormatrix={}", color.matrix.colorspace_tag()));
                if let Some(md) = &color.mastering_display {
                    params.push(format!("master-display={md}"));
                }
                if let Some(cll) = &color.content_light {
                    params.push(format!("max-cll={cll}"));
                }
            }
            params.push("log-level=error".into());
            push(&mut a, "-x265-params", params.join(":"));
        }
        if e == "libsvtav1" {
            let mut params = vec!["enable-overlays=1".to_string()];
            if color.is_hdr() {
                if let Some(md) = &color.mastering_display {
                    // SVT-AV1 accepts the same G()B()R()WP()L() syntax.
                    params.push(format!("mastering-display={md}"));
                }
                if let Some(cll) = &color.content_light {
                    params.push(format!("content-light={cll}"));
                }
            }
            push(&mut a, "-svtav1-params", params.join(":"));
        }
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn x264_args() {
        let a = VideoEncodeSettings::x264_default().args(&ColorTags::sdr_bt709());
        assert_eq!(a, vec!["-c:v", "libx264", "-crf", "18", "-preset", "medium"]);
    }

    #[test]
    fn x265_hdr_params() {
        let mut c = ColorTags::sdr_bt709();
        c.transfer = Some("smpte2084".into());
        c.primaries = Some("bt2020".into());
        c.matrix = crate::color::YuvMatrix::Bt2020;
        c.content_light = Some("1000,400".into());
        let s =
            VideoEncodeSettings { encoder: "libx265".into(), rate: RateControl::Quality { value: 20.0 }, preset: None, pix_fmt: "yuv420p10le".into(), profile: None, gop: None };
        let a = s.args(&c);
        let p = a.iter().position(|x| x == "-x265-params").unwrap();
        assert!(a[p + 1].contains("hdr10=1"));
        assert!(a[p + 1].contains("max-cll=1000,400"));
        assert!(a[p + 1].contains("colormatrix=bt2020nc"));
    }

    #[test]
    fn profiles_are_consistent() {
        for c in CODECS {
            assert!(!c.pix_fmts.is_empty(), "{}", c.encoder);
            if let (Some(d), false) = (c.default_preset, c.presets.is_empty()) {
                assert!(c.presets.contains(&d), "{}", c.encoder);
            }
            if let Some((lo, hi, def)) = c.quality_range {
                assert!(lo <= def && def <= hi);
            }
        }
        assert_eq!(pix_fmt_bit_depth("yuv420p10le"), 10);
        assert_eq!(pix_fmt_bit_depth("p010le"), 10);
    }
}
