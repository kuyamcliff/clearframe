//! Presets: built-in and user presets, and the portable `.cfxpreset` format.
//!
//! A `.cfxpreset` file is a small, human-readable JSON manifest. Import parses it
//! strictly (unknown operations are rejected), validates every parameter
//! bound, checks names and sizes, and never executes anything. Presets store
//! settings and model references only — never weights, videos or paths.

use serde::{Deserialize, Serialize};

use clearframe_media::encode::{RateControl, VideoEncodeSettings};
use clearframe_media::mux::{Container, SlowMotionAudio};
use clearframe_media::Rational;

use crate::spec::*;

pub const PRESET_FORMAT: &str = "clipfx-preset";
pub const MAX_PRESET_BYTES: usize = 256 * 1024;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Preset {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub category: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub favorite: bool,
    #[serde(default)]
    pub builtin: bool,
    pub pipeline: Pipeline,
    /// Optional: export settings saved with the preset.
    #[serde(default)]
    pub output: Option<OutputSettings>,
}

/// The on-disk `.cfxpreset` document.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PresetFile {
    pub format: String,
    pub schema_version: u32,
    #[serde(default)]
    pub app_version: String,
    #[serde(default)]
    pub exported_at: String,
    pub preset: Preset,
}

#[derive(Debug, thiserror::Error)]
pub enum PresetError {
    #[error("the file is too large to be a preset")]
    TooLarge,
    #[error("not a ClipFX Enhance preset: {0}")]
    Format(String),
    #[error("this preset was made with a newer version of ClipFX Enhance (schema {0})")]
    TooNew(u32),
    #[error("invalid preset: {0}")]
    Invalid(String),
}

fn check_text(field: &str, s: &str, max: usize) -> Result<(), PresetError> {
    if s.chars().count() > max {
        return Err(PresetError::Invalid(format!("{field} is too long")));
    }
    if s.chars().any(|c| c.is_control() && c != '\n') {
        return Err(PresetError::Invalid(format!("{field} contains control characters")));
    }
    Ok(())
}

impl Preset {
    pub fn validate(&self) -> Result<(), PresetError> {
        if self.name.trim().is_empty() {
            return Err(PresetError::Invalid("the preset needs a name".into()));
        }
        check_text("name", &self.name, 80)?;
        check_text("description", &self.description, 2000)?;
        check_text("category", &self.category, 60)?;
        if self.tags.len() > 20 {
            return Err(PresetError::Invalid("too many tags".into()));
        }
        for t in &self.tags {
            check_text("tag", t, 40)?;
        }
        check_text("id", &self.id, 100)?;
        self.pipeline.validate().map_err(PresetError::Invalid)?;
        if let Some(o) = &self.output {
            let p = clearframe_media::encode::codec_profile(&o.video.encoder).ok_or_else(|| PresetError::Invalid(format!("unknown encoder {}", o.video.encoder)))?;
            if !p.pix_fmts.contains(&o.video.pix_fmt.as_str()) {
                return Err(PresetError::Invalid(format!("{} does not support {}", o.video.encoder, o.video.pix_fmt)));
            }
            if let RateControl::Quality { value } = o.video.rate {
                if !value.is_finite() || !(0.0..=100.0).contains(&value) {
                    return Err(PresetError::Invalid("quality value out of range".into()));
                }
            }
            if let RateControl::Bitrate { kbps } = o.video.rate {
                if !(100..=1_000_000).contains(&kbps) {
                    return Err(PresetError::Invalid("bitrate out of range".into()));
                }
            }
        }
        Ok(())
    }

    /// Serialise as a `.cfxpreset` document.
    pub fn export(&self) -> String {
        let mut p = self.clone();
        p.builtin = false;
        p.favorite = false;
        let f = PresetFile {
            format: PRESET_FORMAT.into(),
            schema_version: SCHEMA_VERSION,
            app_version: env!("CARGO_PKG_VERSION").into(),
            exported_at: chrono::Utc::now().to_rfc3339(),
            preset: p,
        };
        serde_json::to_string_pretty(&f).unwrap_or_default()
    }

    /// Parse and validate a `.cfxpreset` document. The imported preset gets a fresh id and is never built-in.
    pub fn import(data: &[u8]) -> Result<Preset, PresetError> {
        if data.len() > MAX_PRESET_BYTES {
            return Err(PresetError::TooLarge);
        }
        let s = std::str::from_utf8(data).map_err(|_| PresetError::Format("not UTF-8 text".into()))?;
        let v: serde_json::Value = serde_json::from_str(s).map_err(|e| PresetError::Format(e.to_string()))?;
        if v.get("format").and_then(|f| f.as_str()) != Some(PRESET_FORMAT) {
            return Err(PresetError::Format("missing \"format\": \"clipfx-preset\"".into()));
        }
        let ver = v.get("schema_version").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
        if ver > SCHEMA_VERSION {
            return Err(PresetError::TooNew(ver));
        }
        if ver == 0 {
            return Err(PresetError::Format("missing schema_version".into()));
        }
        let f: PresetFile = serde_json::from_value(v).map_err(|e| PresetError::Format(e.to_string()))?;
        let mut p = f.preset;
        p.builtin = false;
        p.favorite = false;
        p.id = new_preset_id();
        // Fresh stage ids avoid clashes with existing pipelines.
        for s in &mut p.pipeline.stages {
            s.id = Stage::new(s.op.clone()).id;
        }
        p.validate()?;
        Ok(p)
    }
}

pub fn new_preset_id() -> String {
    format!("user-{}", uuid_like())
}

fn uuid_like() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(1);
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    format!("{:x}{:04x}", t, N.fetch_add(1, Ordering::Relaxed) & 0xffff)
}

fn stage(id: &str, op: Operation) -> Stage {
    Stage { id: id.into(), enabled: true, op }
}

fn x265_10bit(crf: f32) -> OutputSettings {
    OutputSettings {
        container: Container::Mp4,
        video: VideoEncodeSettings {
            encoder: "libx265".into(),
            rate: RateControl::Quality { value: crf },
            preset: Some("medium".into()),
            pix_fmt: "yuv420p10le".into(),
            profile: None,
            gop: None,
        },
        ..Default::default()
    }
}

/// AnimeVideo v3 enhancement (no denoise control) at `target`.
fn anime(target: SizeTarget) -> EnhanceSettings {
    EnhanceSettings { model_id: "xinntao.realesr-animevideo-v3".into(), target, denoise: None, ..Default::default() }
}

/// Built-in presets (cannot be overwritten or deleted).
pub fn builtin_presets() -> Vec<Preset> {
    let p = |id: &str, name: &str, cat: &str, desc: &str, tags: &[&str], stages: Vec<Stage>, output: Option<OutputSettings>| Preset {
        id: id.into(),
        name: name.into(),
        description: desc.into(),
        category: cat.into(),
        tags: tags.iter().map(|s| s.to_string()).collect(),
        favorite: false,
        builtin: true,
        pipeline: Pipeline { stages },
        output,
    };
    vec![
        p(
            "builtin.anime-4k",
            "Anime upscale to 4K",
            "Enhance",
            "Real-ESRGAN AnimeVideo v3 to 2160p, HEVC 10-bit.",
            &["anime", "4k"],
            vec![stage("enhance", Operation::Enhance(anime(SizeTarget::Height { height: 2160 })))],
            Some(x265_10bit(18.0)),
        ),
        p(
            "builtin.anime-2x",
            "Anime 2x (fast)",
            "Enhance",
            "AnimeVideo v3 at the model's native 4x, resized to 2x the source.",
            &["anime"],
            vec![stage("enhance", Operation::Enhance(anime(SizeTarget::Scale { factor: 2.0 })))],
            None,
        ),
        p(
            "builtin.live-1080",
            "Live action to 1080p",
            "Enhance",
            "Real-ESRGAN General v3, keeping fine texture, to 1080p.",
            &["live action", "1080p"],
            vec![stage(
                "enhance",
                Operation::Enhance(EnhanceSettings {
                    model_id: "xinntao.realesr-general-v3".into(),
                    target: SizeTarget::Height { height: 1080 },
                    ..Default::default()
                }),
            )],
            None,
        ),
        p(
            "builtin.restore-old",
            "Restore old footage",
            "Restore",
            "Deinterlace, denoise and upscale to 1080p with conservative settings.",
            &["archival", "interlaced"],
            vec![
                stage("deinterlace", Operation::Deinterlace(DeinterlaceSettings::default())),
                stage("denoise", Operation::Denoise(DenoiseSettings { strength: 12.0, ..Default::default() })),
                stage(
                    "enhance",
                    Operation::Enhance(EnhanceSettings {
                        model_id: "xinntao.realesr-general-v3".into(),
                        denoise: Some(0.3),
                        target: SizeTarget::Height { height: 1080 },
                        strength: 0.8,
                        ..Default::default()
                    }),
                ),
            ],
            None,
        ),
        p(
            "builtin.smooth-60",
            "Smooth 60 fps",
            "Interpolate",
            "RIFE frame interpolation to 60 fps with scene-cut detection; duration and audio unchanged.",
            &["60fps"],
            vec![stage("interpolate", Operation::Interpolate(InterpolateSettings { target: FpsTarget::Fps { fps: Rational::new(60, 1) }, ..Default::default() }))],
            None,
        ),
        p(
            "builtin.slowmo-4x",
            "Slow motion 4x",
            "Interpolate",
            "RIFE slow motion at 4x with pitch-preserved audio.",
            &["slow motion"],
            vec![stage(
                "interpolate",
                Operation::Interpolate(InterpolateSettings {
                    target: FpsTarget::Multiply { factor: 4 },
                    slow_motion: 4.0,
                    slow_motion_audio: SlowMotionAudio::StretchPreservePitch,
                    ..Default::default()
                }),
            )],
            None,
        ),
        p(
            "builtin.stabilize",
            "Stabilize (auto crop)",
            "Stabilize",
            "Two-pass stabilization that zooms just enough to hide moving borders.",
            &["shaky"],
            vec![stage("stabilize", Operation::Stabilize(StabilizeSettings::default()))],
            None,
        ),
        p(
            "builtin.denoise",
            "Clean noisy footage",
            "Restore",
            "FFDNet AI denoiser at a moderate strength.",
            &["low light", "noise"],
            vec![stage("denoise", Operation::Denoise(DenoiseSettings { strength: 15.0, ..Default::default() }))],
            None,
        ),
        p(
            "builtin.deblur",
            "Motion deblur",
            "Restore",
            "NAFNet motion deblur at full strength.",
            &["blur"],
            vec![stage("deblur", Operation::Deblur(DeblurSettings::default()))],
            None,
        ),
        p(
            "builtin.faces",
            "Recover faces",
            "Restore",
            "GFPGAN face restoration at 70% strength.",
            &["faces"],
            vec![stage("faces", Operation::FaceRestore(FaceSettings::default()))],
            None,
        ),
        p(
            "builtin.hdr-to-sdr",
            "HDR to SDR",
            "Color & HDR",
            "Tone-map HDR10/HLG to SDR BT.709 (Hable).",
            &["hdr"],
            vec![stage("color", Operation::Color(ColorSettings { mode: ColorMode::HdrToSdr, ..Default::default() }))],
            None,
        ),
        p(
            "builtin.film-grain",
            "Film grain",
            "Grain",
            "Deterministic film-like grain.",
            &["grain"],
            vec![stage("grain", Operation::Grain(GrainSettings::preset(GrainPreset::Film)))],
            None,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_are_valid() {
        let b = builtin_presets();
        assert!(b.len() >= 10);
        for p in &b {
            p.validate().unwrap_or_else(|e| panic!("{}: {e}", p.id));
            assert!(p.builtin);
        }
    }

    #[test]
    fn export_import_roundtrip() {
        let p = builtin_presets().remove(0);
        let s = p.export();
        let back = Preset::import(s.as_bytes()).unwrap();
        assert_eq!(back.name, p.name);
        assert!(!back.builtin);
        assert_ne!(back.id, p.id);
        assert_eq!(back.pipeline.stages.len(), p.pipeline.stages.len());
        assert_eq!(back.output, p.output);
    }

    #[test]
    fn rejects_malicious_or_invalid_files() {
        assert!(matches!(Preset::import(&vec![b' '; MAX_PRESET_BYTES + 1]), Err(PresetError::TooLarge)));
        assert!(Preset::import(b"{\"format\":\"other\"}").is_err());
        let p = builtin_presets().remove(0);
        let mut v: serde_json::Value = serde_json::from_str(&p.export()).unwrap();
        v["schema_version"] = 99.into();
        assert!(matches!(Preset::import(v.to_string().as_bytes()), Err(PresetError::TooNew(99))));
        // Unknown operation (e.g. something that tries to run code) is rejected.
        let mut v: serde_json::Value = serde_json::from_str(&p.export()).unwrap();
        v["preset"]["pipeline"]["stages"][0]["type"] = "shell".into();
        assert!(Preset::import(v.to_string().as_bytes()).is_err());
        // Unknown top-level fields (e.g. paths) are rejected.
        let mut v: serde_json::Value = serde_json::from_str(&p.export()).unwrap();
        v["preset"]["source_path"] = "../../etc/passwd".into();
        assert!(Preset::import(v.to_string().as_bytes()).is_err());
        // Out-of-range parameter.
        let mut v: serde_json::Value = serde_json::from_str(&p.export()).unwrap();
        v["preset"]["pipeline"]["stages"][0]["settings"]["strength"] = 50.0.into();
        assert!(matches!(Preset::import(v.to_string().as_bytes()), Err(PresetError::Invalid(_))));
        // Encoder/pixel format mismatch.
        let mut v: serde_json::Value = serde_json::from_str(&p.export()).unwrap();
        v["preset"]["output"]["video"]["pix_fmt"] = "yuv444p16le".into();
        assert!(Preset::import(v.to_string().as_bytes()).is_err());
    }
}
