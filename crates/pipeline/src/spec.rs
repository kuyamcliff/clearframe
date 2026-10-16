//! Processing pipeline and job specifications (serialised into projects,
//! presets and job records).

use std::collections::BTreeMap;
use std::path::PathBuf;

use clearframe_inference::device::BackendKind;
use clearframe_inference::face::FaceParams;
use clearframe_inference::model::Precision;
use clearframe_media::encode::{RateControl, VideoEncodeSettings};
use clearframe_media::mux::{AudioMode, Container, SlowMotionAudio};
use clearframe_media::Rational;
use serde::{Deserialize, Serialize};

/// Current schema version for pipelines/presets/projects.
pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DeinterlaceMethod {
    #[default]
    Bwdif,
    Yadif,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FieldOrder {
    #[default]
    Auto,
    Tff,
    Bff,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DeinterlaceRate {
    /// One output frame per input frame (keeps frame rate).
    #[default]
    Frame,
    /// One output frame per field ("bob"): doubles the frame rate.
    Field,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct DeinterlaceSettings {
    pub method: DeinterlaceMethod,
    pub field_order: FieldOrder,
    pub rate: DeinterlaceRate,
    /// Only deinterlace frames flagged as interlaced.
    pub only_flagged: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DenoiseEngine {
    /// FFDNet (AI, fast, noise-level controlled).
    #[default]
    Ffdnet,
    /// DRUNet (AI, higher quality, slower).
    Drunet,
    /// FFmpeg hqdn3d (classic spatial + temporal).
    Hqdn3d,
    /// FFmpeg nlmeans (classic non-local means).
    Nlmeans,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DenoiseSettings {
    pub engine: DenoiseEngine,
    /// Noise level on a 0-255 scale for AI engines; mapped to filter strength for classic ones.
    pub strength: f32,
    /// Temporal filtering for hqdn3d (can smear motion).
    pub temporal: bool,
    /// 0..1: blend of the denoised result over the original (1 = full effect).
    pub mix: f32,
    pub model_id: Option<String>,
}

impl Default for DenoiseSettings {
    fn default() -> Self {
        Self { engine: DenoiseEngine::Ffdnet, strength: 10.0, temporal: false, mix: 1.0, model_id: None }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DeblurSettings {
    pub model_id: String,
    /// 0..1 blend of the deblurred result over the original.
    pub strength: f32,
}

impl Default for DeblurSettings {
    fn default() -> Self {
        Self { model_id: "megvii.nafnet-gopro-w32".into(), strength: 1.0 }
    }
}

/// Output size for enhancement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SizeTarget {
    /// Native output of the model (input x model scale).
    ModelScale,
    /// Multiply the input size.
    Scale { factor: f32 },
    /// Fit the height (e.g. 720, 1080, 1440, 2160), keeping aspect ratio.
    Height { height: u32 },
    /// Exact size.
    Custom { width: u32, height: u32, fit: FitMode },
    /// Keep the source size (restoration without enlargement).
    Source,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FitMode {
    /// Scale to fit inside; pad with black bars.
    #[default]
    Fit,
    /// Scale to cover; crop overflow.
    Fill,
    /// Scale to the exact size, changing aspect ratio.
    Stretch,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EnhanceSettings {
    pub model_id: String,
    pub target: SizeTarget,
    /// 0..1 blend between a plain (bicubic) resize and the model output.
    pub strength: f32,
    /// Real-ESRGAN General v3 only: 0 = keep noise/texture .. 1 = strong denoise.
    pub denoise: Option<f32>,
    /// Tile size in input pixels (0 = automatic).
    pub tile_size: u32,
    pub tile_pad: u32,
}

impl Default for EnhanceSettings {
    fn default() -> Self {
        Self { model_id: "xinntao.realesr-animevideo-v3".into(), target: SizeTarget::ModelScale, strength: 1.0, denoise: None, tile_size: 0, tile_pad: 0 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FaceSettings {
    pub model_id: String,
    pub params: FaceParams,
}

impl Default for FaceSettings {
    fn default() -> Self {
        Self { model_id: "tencentarc.gfpgan-v1.4".into(), params: FaceParams::default() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum InterpEngine {
    /// RIFE optical-flow network (AI).
    #[default]
    Rife,
    /// FFmpeg motion-compensated interpolation (classic, no model).
    Mci,
    /// Cross-fade between neighbouring frames (fast, soft).
    Blend,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FpsTarget {
    Multiply { factor: u32 },
    Fps { fps: Rational },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct InterpolateSettings {
    pub engine: InterpEngine,
    pub model_id: String,
    pub target: FpsTarget,
    /// >1 slows playback down by this factor (duration grows). 1 = keep duration.
    pub slow_motion: f64,
    pub slow_motion_audio: SlowMotionAudio,
    /// Avoid blending across hard cuts.
    pub scene_detection: bool,
    /// Mean absolute difference (0..1) above which neighbouring frames are treated as a cut.
    pub scene_threshold: f32,
    /// Neighbouring frames closer than this (0..1) are treated as duplicates and repeated instead of interpolated.
    pub duplicate_threshold: f32,
    /// RIFE resolution scale (1.0; use 0.5 for 4K to reduce memory).
    pub rife_scale: f64,
}

impl Default for InterpolateSettings {
    fn default() -> Self {
        Self {
            engine: InterpEngine::Rife,
            model_id: "hzwer.rife-v4.26".into(),
            target: FpsTarget::Multiply { factor: 2 },
            slow_motion: 1.0,
            slow_motion_audio: SlowMotionAudio::Mute,
            scene_detection: true,
            scene_threshold: 0.12,
            duplicate_threshold: 0.0,
            rife_scale: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum StabilizeMode {
    /// Fill uncovered edges from neighbouring frames (keeps full frame; edges may show artifacts).
    FullFrame,
    /// Zoom just enough to hide moving borders (sharp edges, smaller visible area).
    #[default]
    AutoCrop,
    /// Leave uncovered edges black.
    Black,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StabilizeSettings {
    /// Number of frames used to smooth camera motion; 0 disables stabilization.
    pub smoothing: u32,
    /// How shaky the footage is, 1-10.
    pub shakiness: u32,
    pub mode: StabilizeMode,
    /// Lock the camera completely (tripod mode).
    pub tripod: bool,
}

impl Default for StabilizeSettings {
    fn default() -> Self {
        Self { smoothing: 15, shakiness: 5, mode: StabilizeMode::AutoCrop, tripod: false }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ColorMode {
    /// Brightness/contrast/saturation/gamma adjustments.
    #[default]
    Adjust,
    /// Tone-map HDR (PQ/HLG) to SDR BT.709.
    HdrToSdr,
    /// Map SDR into an HDR container (PQ or HLG). Does not recover detail that was never captured.
    SdrToHdr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ToneMapper {
    #[default]
    Hable,
    Mobius,
    Reinhard,
    Clip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum HdrTransfer {
    #[default]
    Pq,
    Hlg,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ColorSettings {
    pub mode: ColorMode,
    /// Exposure in stops (-3..3): multiplies the image.
    pub exposure: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub gamma: f32,
    /// Boosts muted colours more than already saturated ones (-1..1).
    pub vibrance: f32,
    /// Cool (-1) to warm (+1) white balance.
    pub temperature: f32,
    /// Green (-1) to magenta (+1).
    pub tint: f32,
    /// Hue rotation in degrees (-180..180).
    pub hue: f32,
    /// Lift (+) or deepen (-) shadows (-1..1).
    pub shadows: f32,
    /// Raise (+) or recover (-) highlights (-1..1).
    pub highlights: f32,
    pub tone_mapper: ToneMapper,
    /// HDR -> SDR: desaturation strength for highlights (0..1).
    pub desaturate: f32,
    /// SDR -> HDR: output transfer.
    pub hdr_transfer: HdrTransfer,
    /// SDR -> HDR: brightness of SDR reference white in nits (203 is the ITU recommendation).
    pub sdr_white_nits: f32,
    /// SDR -> HDR: mastering peak luminance written as metadata.
    pub peak_nits: f32,
}

impl Default for ColorSettings {
    fn default() -> Self {
        Self {
            mode: ColorMode::Adjust,
            exposure: 0.0,
            brightness: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            gamma: 1.0,
            vibrance: 0.0,
            temperature: 0.0,
            tint: 0.0,
            hue: 0.0,
            shadows: 0.0,
            highlights: 0.0,
            tone_mapper: ToneMapper::Hable,
            desaturate: 0.0,
            hdr_transfer: HdrTransfer::Pq,
            sdr_white_nits: 203.0,
            peak_nits: 1000.0,
        }
    }
}

impl ColorSettings {
    /// Whether any of the colour and light sliders differ from neutral.
    pub fn has_adjustments(&self) -> bool {
        let e = 1e-4;
        self.exposure.abs() > e
            || self.brightness.abs() > e
            || (self.contrast - 1.0).abs() > e
            || (self.saturation - 1.0).abs() > e
            || (self.gamma - 1.0).abs() > e
            || self.vibrance.abs() > e
            || self.temperature.abs() > e
            || self.tint.abs() > e
            || self.hue.abs() > e
            || self.shadows.abs() > e
            || self.highlights.abs() > e
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SharpenMethod {
    /// Contrast-adaptive sharpening: strengthens fine detail without halos on strong edges.
    #[default]
    Adaptive,
    /// Classic unsharp mask with a radius; stronger, can halo when pushed.
    Unsharp,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SharpenSettings {
    pub method: SharpenMethod,
    /// 0..1 (adaptive) or 0..3 (unsharp).
    pub amount: f32,
    /// Unsharp radius in pixels (0.5..5).
    pub radius: f32,
}

impl Default for SharpenSettings {
    fn default() -> Self {
        Self { method: SharpenMethod::Adaptive, amount: 0.4, radius: 1.5 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum GrainPreset {
    #[default]
    Fine,
    Film,
    Coarse,
    Monochrome,
    Color,
    Custom,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GrainSettings {
    pub preset: GrainPreset,
    /// 0..1 amplitude.
    pub strength: f32,
    /// 0.5..4 grain size in pixels.
    pub size: f32,
    /// 0 = monochrome .. 1 = fully independent per-channel noise.
    pub color: f32,
    /// New grain pattern every frame (true) or a static pattern (false).
    pub animated: bool,
    /// Reduce grain in deep shadows and bright highlights (film-like response).
    pub luma_response: bool,
    pub seed: u64,
}

impl Default for GrainSettings {
    fn default() -> Self {
        Self { preset: GrainPreset::Fine, strength: 0.25, size: 1.0, color: 0.15, animated: true, luma_response: true, seed: 1 }
    }
}

impl GrainSettings {
    /// Settings for a named preset (strength preserved).
    pub fn preset(p: GrainPreset) -> Self {
        let base = Self::default();
        match p {
            GrainPreset::Fine => Self { preset: p, size: 1.0, color: 0.15, strength: 0.2, ..base },
            GrainPreset::Film => Self { preset: p, size: 1.6, color: 0.3, strength: 0.3, ..base },
            GrainPreset::Coarse => Self { preset: p, size: 2.6, color: 0.25, strength: 0.4, ..base },
            GrainPreset::Monochrome => Self { preset: p, size: 1.3, color: 0.0, strength: 0.3, ..base },
            GrainPreset::Color => Self { preset: p, size: 1.2, color: 1.0, strength: 0.3, ..base },
            GrainPreset::Custom => Self { preset: p, ..base },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ResizeFilter {
    #[default]
    Lanczos,
    Bicubic,
    Bilinear,
    Spline,
}

impl ResizeFilter {
    pub fn ffmpeg(self) -> &'static str {
        match self {
            ResizeFilter::Lanczos => "lanczos",
            ResizeFilter::Bicubic => "bicubic",
            ResizeFilter::Bilinear => "bilinear",
            ResizeFilter::Spline => "spline",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ResizeSettings {
    pub width: u32,
    pub height: u32,
    pub fit: FitMode,
    pub filter: ResizeFilter,
}

impl Default for ResizeSettings {
    fn default() -> Self {
        Self { width: 1920, height: 1080, fit: FitMode::Fit, filter: ResizeFilter::Lanczos }
    }
}

/// One processing operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "settings", rename_all = "snake_case")]
pub enum Operation {
    Deinterlace(DeinterlaceSettings),
    Denoise(DenoiseSettings),
    Deblur(DeblurSettings),
    Enhance(EnhanceSettings),
    FaceRestore(FaceSettings),
    Interpolate(InterpolateSettings),
    Stabilize(StabilizeSettings),
    Color(ColorSettings),
    Sharpen(SharpenSettings),
    Grain(GrainSettings),
    Resize(ResizeSettings),
}

impl Operation {
    pub fn kind(&self) -> &'static str {
        match self {
            Operation::Deinterlace(_) => "deinterlace",
            Operation::Denoise(_) => "denoise",
            Operation::Deblur(_) => "deblur",
            Operation::Enhance(_) => "enhance",
            Operation::FaceRestore(_) => "face_restore",
            Operation::Interpolate(_) => "interpolate",
            Operation::Stabilize(_) => "stabilize",
            Operation::Color(_) => "color",
            Operation::Sharpen(_) => "sharpen",
            Operation::Grain(_) => "grain",
            Operation::Resize(_) => "resize",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Operation::Deinterlace(_) => "Deinterlace",
            Operation::Denoise(_) => "Denoise",
            Operation::Deblur(_) => "Motion deblur",
            Operation::Enhance(_) => "Enhance",
            Operation::FaceRestore(_) => "Face recovery",
            Operation::Interpolate(_) => "Frame interpolation",
            Operation::Stabilize(_) => "Stabilize",
            Operation::Color(_) => "Color & light",
            Operation::Sharpen(_) => "Sharpen",
            Operation::Grain(_) => "Add grain",
            Operation::Resize(_) => "Resize",
        }
    }

    /// Model IDs this operation needs.
    pub fn required_models(&self) -> Vec<String> {
        match self {
            Operation::Denoise(d) => match d.engine {
                DenoiseEngine::Ffdnet => vec![d.model_id.clone().unwrap_or_else(|| "kair.ffdnet-color".into())],
                DenoiseEngine::Drunet => vec![d.model_id.clone().unwrap_or_else(|| "kair.drunet-color".into())],
                _ => vec![],
            },
            Operation::Deblur(d) => vec![d.model_id.clone()],
            Operation::Enhance(e) => vec![e.model_id.clone()],
            Operation::FaceRestore(f) => vec![f.model_id.clone()],
            Operation::Interpolate(i) if i.engine == InterpEngine::Rife => vec![i.model_id.clone()],
            _ => vec![],
        }
    }

    /// Validate parameter bounds (used for presets/projects coming from disk).
    pub fn validate(&self) -> Result<(), String> {
        fn range(name: &str, v: f64, lo: f64, hi: f64) -> Result<(), String> {
            if v.is_finite() && v >= lo && v <= hi {
                Ok(())
            } else {
                Err(format!("{name} must be between {lo} and {hi} (got {v})"))
            }
        }
        match self {
            Operation::Deinterlace(_) => Ok(()),
            Operation::Denoise(d) => {
                range("denoise strength", d.strength as f64, 0.0, 75.0)?;
                range("denoise mix", d.mix as f64, 0.0, 1.0)
            }
            Operation::Deblur(d) => range("deblur strength", d.strength as f64, 0.0, 1.0),
            Operation::Enhance(e) => {
                range("enhance strength", e.strength as f64, 0.0, 1.0)?;
                if let Some(dn) = e.denoise {
                    range("enhance denoise", dn as f64, 0.0, 1.0)?;
                }
                range("tile size", e.tile_size as f64, 0.0, 4096.0)?;
                range("tile padding", e.tile_pad as f64, 0.0, 256.0)?;
                match &e.target {
                    SizeTarget::Scale { factor } => range("scale factor", *factor as f64, 0.25, 8.0),
                    SizeTarget::Height { height } => range("target height", *height as f64, 16.0, 8640.0),
                    SizeTarget::Custom { width, height, .. } => {
                        range("target width", *width as f64, 16.0, 15360.0)?;
                        range("target height", *height as f64, 16.0, 8640.0)
                    }
                    _ => Ok(()),
                }
            }
            Operation::FaceRestore(f) => {
                range("face strength", f.params.strength as f64, 0.0, 1.0)?;
                range("detection threshold", f.params.detection_threshold as f64, 0.3, 0.999)?;
                range("max faces", f.params.max_faces as f64, 1.0, 64.0)?;
                range("temporal smoothing", f.params.temporal_smoothing as f64, 0.0, 0.95)?;
                range("detection size", f.params.detection_max_side as f64, 320.0, 4096.0)
            }
            Operation::Interpolate(i) => {
                match &i.target {
                    FpsTarget::Multiply { factor } => range("multiplier", *factor as f64, 1.0, 16.0)?,
                    FpsTarget::Fps { fps } => range("target fps", fps.as_f64(), 1.0, 480.0)?,
                }
                range("slow motion", i.slow_motion, 1.0, 16.0)?;
                range("scene threshold", i.scene_threshold as f64, 0.0, 1.0)?;
                range("duplicate threshold", i.duplicate_threshold as f64, 0.0, 0.2)?;
                if !(i.rife_scale == 1.0 || i.rife_scale == 0.5 || i.rife_scale == 0.25 || i.rife_scale == 2.0) {
                    return Err("RIFE scale must be 0.25, 0.5, 1 or 2".into());
                }
                Ok(())
            }
            Operation::Stabilize(s) => {
                range("smoothing", s.smoothing as f64, 0.0, 120.0)?;
                range("shakiness", s.shakiness as f64, 1.0, 10.0)
            }
            Operation::Color(c) => {
                range("exposure", c.exposure as f64, -3.0, 3.0)?;
                range("vibrance", c.vibrance as f64, -1.0, 1.0)?;
                range("temperature", c.temperature as f64, -1.0, 1.0)?;
                range("tint", c.tint as f64, -1.0, 1.0)?;
                range("hue", c.hue as f64, -180.0, 180.0)?;
                range("shadows", c.shadows as f64, -1.0, 1.0)?;
                range("highlights", c.highlights as f64, -1.0, 1.0)?;
                range("brightness", c.brightness as f64, -1.0, 1.0)?;
                range("contrast", c.contrast as f64, 0.0, 3.0)?;
                range("saturation", c.saturation as f64, 0.0, 3.0)?;
                range("gamma", c.gamma as f64, 0.1, 5.0)?;
                range("desaturate", c.desaturate as f64, 0.0, 1.0)?;
                range("SDR white", c.sdr_white_nits as f64, 80.0, 400.0)?;
                range("peak luminance", c.peak_nits as f64, 400.0, 10000.0)
            }
            Operation::Sharpen(sh) => {
                let max = if sh.method == SharpenMethod::Adaptive { 1.0 } else { 3.0 };
                range("sharpen amount", sh.amount as f64, 0.0, max)?;
                range("sharpen radius", sh.radius as f64, 0.5, 5.0)
            }
            Operation::Grain(g) => {
                range("grain strength", g.strength as f64, 0.0, 1.0)?;
                range("grain size", g.size as f64, 0.5, 4.0)?;
                range("grain color", g.color as f64, 0.0, 1.0)
            }
            Operation::Resize(r) => {
                range("width", r.width as f64, 16.0, 15360.0)?;
                range("height", r.height as f64, 16.0, 8640.0)
            }
        }
    }
}

/// Stage id of a Deinterlace step added automatically for an interlaced source.
pub const AUTO_DEINTERLACE_ID: &str = "auto-deinterlace";

/// A pipeline stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stage {
    pub id: String,
    #[serde(default = "yes")]
    pub enabled: bool,
    #[serde(flatten)]
    pub op: Operation,
}

fn yes() -> bool {
    true
}

impl Stage {
    pub fn new(op: Operation) -> Self {
        Self { id: format!("{}-{}", op.kind(), short_id()), enabled: true, op }
    }
}

fn short_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
    format!("{:x}", (t ^ (N.fetch_add(1, Ordering::Relaxed) << 48)) & 0xffff_ffff)
}

/// An ordered list of stages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Pipeline {
    pub stages: Vec<Stage>,
}

impl Pipeline {
    pub fn single(op: Operation) -> Self {
        Self { stages: vec![Stage::new(op)] }
    }
    pub fn enabled(&self) -> impl Iterator<Item = &Stage> {
        self.stages.iter().filter(|s| s.enabled)
    }

    /// For an interlaced source, the same pipeline with a Deinterlace step added first when the
    /// pipeline has none: AI models and most filters expect progressive frames, and comb lines
    /// would be enlarged and baked in. `None` when nothing needs adding.
    pub fn with_auto_deinterlace(&self, source_interlaced: bool) -> Option<Pipeline> {
        let has = self.enabled().any(|s| matches!(s.op, Operation::Deinterlace(_)));
        if !source_interlaced || has || self.enabled().next().is_none() {
            return None;
        }
        let mut p = self.clone();
        p.stages.insert(0, Stage { id: AUTO_DEINTERLACE_ID.into(), enabled: true, op: Operation::Deinterlace(DeinterlaceSettings::default()) });
        Some(p)
    }
    pub fn required_models(&self) -> Vec<String> {
        let mut v: Vec<String> = self.enabled().flat_map(|s| s.op.required_models()).collect();
        v.sort();
        v.dedup();
        v
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.stages.len() > 32 {
            return Err("a pipeline can have at most 32 stages".into());
        }
        let mut ids = std::collections::HashSet::new();
        for s in &self.stages {
            if !ids.insert(&s.id) {
                return Err(format!("duplicate stage id {}", s.id));
            }
            if s.id.len() > 64 {
                return Err("stage id too long".into());
            }
            s.op.validate().map_err(|e| format!("{}: {e}", s.op.label()))?;
        }
        Ok(())
    }
}

/// Encoding and container options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputSettings {
    pub container: Container,
    pub video: VideoEncodeSettings,
    #[serde(default)]
    pub audio: AudioMode,
    #[serde(default = "yes")]
    pub subtitles: bool,
    #[serde(default = "yes")]
    pub metadata: bool,
    #[serde(default = "yes")]
    pub chapters: bool,
}

impl OutputSettings {
    /// For an HDR source whose result stays HDR: the same settings with a 10-bit format (and HEVC
    /// in place of H.264, which players do not accept for HDR), when the chosen format is 8-bit.
    /// `None` when nothing needs changing.
    pub fn for_hdr_result(&self) -> Option<OutputSettings> {
        if clearframe_media::encode::pix_fmt_bit_depth(&self.video.pix_fmt) >= 10 {
            return None;
        }
        let mut o = self.clone();
        let enc = o.video.encoder.as_str();
        if enc == "libx264" || enc.starts_with("h264_") {
            o.video.encoder = "libx265".into();
            o.video.preset = Some("medium".into());
            o.video.profile = None;
            if let RateControl::Quality { value } = o.video.rate {
                // x265 CRF gives similar quality about 2 points higher than x264.
                o.video.rate = RateControl::Quality { value: (value + 2.0).min(51.0) };
            }
        }
        o.video.pix_fmt = match o.video.encoder.as_str() {
            "prores_ks" => "yuv422p10le",
            _ => "yuv420p10le",
        }
        .into();
        if matches!(o.container, Container::Webm) && o.video.encoder == "libx265" {
            o.container = Container::Mkv;
        }
        Some(o)
    }
}

impl Default for OutputSettings {
    fn default() -> Self {
        Self { container: Container::Mp4, video: VideoEncodeSettings::x264_default(), audio: AudioMode::Auto, subtitles: true, metadata: true, chapters: true }
    }
}

/// Inference engine options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct EngineSettings {
    /// None = automatic.
    pub backend: Option<BackendKind>,
    pub precision: Precision,
}

/// A model resolved by the registry for a job, with exact hashes recorded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedModel {
    pub id: String,
    pub version: String,
    /// Named weight files (role -> converted safetensors path).
    pub files: BTreeMap<String, PathBuf>,
    /// role -> sha256 of the converted file.
    pub sha256: BTreeMap<String, String>,
    /// Network architecture id (from the catalog).
    #[serde(default)]
    pub architecture: String,
    /// Output scale factor (super-resolution models).
    #[serde(default = "one")]
    pub scale: u32,
    /// Working memory per input megapixel (catalog estimate), used to size tiles.
    #[serde(default = "default_mb_per_mp")]
    pub memory_per_megapixel_mb: f64,
}

fn default_mb_per_mp() -> f64 {
    1500.0
}

/// Resource limits applied to one job (resolved from the Performance settings).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct JobResources {
    /// Worker threads for inference and FFmpeg (0 = automatic).
    pub threads: u32,
    /// Memory budget for this job in MB (0 = unlimited); used to size model tiles.
    pub memory_budget_mb: u64,
    /// GPU memory budget for model tiles in MB (0 = automatic).
    pub vram_budget_mb: u64,
    /// Retry with smaller tiles when a device runs out of memory.
    pub oom_tile_fallback: bool,
    /// Pause the job when free disk space falls below this (MB).
    pub min_free_disk_mb: u64,
    /// "auto", "cpu", "cuda:N", "metal:N".
    pub device: String,
}

impl Default for JobResources {
    fn default() -> Self {
        Self { threads: 0, memory_budget_mb: 0, vram_budget_mb: 0, oom_tile_fallback: true, min_free_disk_mb: 1024, device: "auto".into() }
    }
}

fn one() -> u32 {
    1
}

/// Seconds in the source timeline.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: f64,
    pub end: f64,
}

impl TimeRange {
    pub fn duration(&self) -> f64 {
        (self.end - self.start).max(0.0)
    }
}

/// Everything needed to run one render.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobSpec {
    pub schema_version: u32,
    pub job_id: String,
    pub source: PathBuf,
    #[serde(default)]
    pub range: Option<TimeRange>,
    pub pipeline: Pipeline,
    pub output: OutputSettings,
    /// Final destination (already collision-resolved by the caller).
    pub destination: PathBuf,
    /// Directory for segments, intermediates and recovery state.
    pub work_dir: PathBuf,
    #[serde(default)]
    pub engine: EngineSettings,
    #[serde(default)]
    pub models: BTreeMap<String, ResolvedModel>,
    /// Segment length in seconds for resumable encoding.
    #[serde(default = "default_segment")]
    pub segment_seconds: f64,
    /// Write a JSON sidecar report next to the output.
    #[serde(default)]
    pub write_sidecar: bool,
    /// Skip mux/validation and produce a quick preview file instead.
    #[serde(default)]
    pub preview: bool,
    #[serde(default)]
    pub resources: JobResources,
    /// Planner's low peak-memory estimate (MB), used by the scheduler.
    #[serde(default)]
    pub estimated_memory_mb: f64,
}

fn default_segment() -> f64 {
    10.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_roundtrip_and_shape() {
        let p = Pipeline {
            stages: vec![
                Stage::new(Operation::Deinterlace(DeinterlaceSettings::default())),
                Stage::new(Operation::Enhance(EnhanceSettings::default())),
                Stage::new(Operation::Interpolate(InterpolateSettings::default())),
                Stage::new(Operation::Grain(GrainSettings::preset(GrainPreset::Film))),
            ],
        };
        let j = serde_json::to_string_pretty(&p).unwrap();
        assert!(j.contains("\"type\": \"enhance\""));
        let back: Pipeline = serde_json::from_str(&j).unwrap();
        assert_eq!(p, back);
        p.validate().unwrap();
        assert_eq!(p.required_models(), vec!["hzwer.rife-v4.26", "xinntao.realesr-animevideo-v3"]);
    }

    #[test]
    fn partial_settings_get_defaults() {
        let j = r#"{"stages":[{"id":"a","type":"grain","settings":{"strength":0.5}}]}"#;
        let p: Pipeline = serde_json::from_str(j).unwrap();
        match &p.stages[0].op {
            Operation::Grain(g) => {
                assert_eq!(g.strength, 0.5);
                assert_eq!(g.size, 1.0);
            }
            _ => panic!(),
        }
        assert!(p.stages[0].enabled);
    }

    #[test]
    fn bounds_rejected() {
        let g = GrainSettings { strength: 5.0, ..Default::default() };
        assert!(Operation::Grain(g).validate().is_err());
        let i = InterpolateSettings { slow_motion: 0.5, ..Default::default() };
        assert!(Operation::Interpolate(i).validate().is_err());
        let e = EnhanceSettings { strength: f32::NAN, ..Default::default() };
        assert!(Operation::Enhance(e).validate().is_err());
        let dup = Pipeline { stages: vec![Stage { id: "x".into(), enabled: true, op: Operation::Grain(GrainSettings::default()) }; 2] };
        assert!(dup.validate().is_err());
    }

    #[test]
    fn unknown_operation_is_rejected() {
        let j = r#"{"stages":[{"id":"a","type":"run_script","settings":{}}]}"#;
        assert!(serde_json::from_str::<Pipeline>(j).is_err());
    }
}
