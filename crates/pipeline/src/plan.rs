//! Turn a pipeline into an execution plan.
//!
//! The planner walks the enabled stages, propagates frame size, frame rate,
//! timing and colour state, decides which stages run as FFmpeg filters and
//! which as native model processors, splits the work into passes when a
//! two-pass filter (stabilisation) follows native stages, and reports
//! warnings, blocking errors and resource estimates. It never adds an
//! operation the user did not choose; technical conversions (pixel format,
//! even dimensions) are listed as notes.

use serde::{Deserialize, Serialize};

use clearframe_media::color::ColorTags;
use clearframe_media::encode::pix_fmt_bit_depth;
use clearframe_media::probe::MediaInfo;
use clearframe_media::Rational;

use crate::spec::*;

/// Frame format flowing between stages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Format {
    pub width: u32,
    pub height: u32,
    pub fps: Rational,
    /// Accumulated slow-motion factor (output time / source time).
    pub time_scale: f64,
    pub transfer: Option<String>,
    pub primaries: Option<String>,
    pub interlaced: bool,
}

impl Format {
    pub fn megapixels(&self) -> f64 {
        self.width as f64 * self.height as f64 / 1e6
    }
    pub fn is_hdr(&self) -> bool {
        matches!(self.transfer.as_deref(), Some("smpte2084") | Some("arib-std-b67"))
    }
}

/// Static model facts the planner needs (provided by the model registry).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFacts {
    pub id: String,
    pub display_name: String,
    pub scale: u32,
    pub installed: bool,
    /// Approximate working memory in MB per input megapixel processed at once.
    pub mb_per_megapixel: f64,
    pub weights_mb: f64,
    pub supports_denoise_strength: bool,
}

pub trait ModelLookup {
    fn facts(&self, id: &str) -> Option<ModelFacts>;
}

/// How a stage executes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StageExec {
    /// FFmpeg filter chain.
    Ffmpeg { filters: Vec<String> },
    /// Native processor (model or Rust implementation), followed by optional FFmpeg filters
    /// (e.g. resize to the requested size after a fixed-scale model).
    Native { post_filters: Vec<String> },
    /// Two-pass stabilisation.
    Stabilize { detect: String, transform_template: String },
    /// Disabled by its own settings (e.g. smoothing 0); the source pixels pass through unchanged.
    Skip { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StagePlan {
    pub stage_id: String,
    pub index: usize,
    pub label: String,
    pub exec: StageExec,
    pub input: Format,
    pub output: Format,
    /// For native stages followed by post filters: the model's own output format.
    pub native_output: Option<Format>,
    /// Model IDs used by this stage.
    pub models: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Issue {
    pub severity: Severity,
    pub stage_id: Option<String>,
    pub message: String,
    /// Machine-readable suggestion for the UI, e.g. "add_deinterlace", "install_model:<id>".
    pub action: Option<String>,
}

/// A contiguous group of work inside a pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Segment {
    /// Native stage, by index into `ExecutionPlan::stages`.
    Native { stage: usize },
    /// FFmpeg filter process between native stages.
    Ffmpeg { filters: Vec<String>, input: Format, output: Format },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassPlan {
    pub index: usize,
    /// Reads the previous pass's intermediate instead of the source.
    pub from_intermediate: bool,
    /// Filters applied by the decoder in the source's native format.
    pub decode_filters: Vec<String>,
    /// Stabilisation analysis to run first (filters before + detect).
    pub stabilize: Option<StabilizePass>,
    pub segments: Vec<Segment>,
    /// Filters applied to RGB frames by the encoder.
    pub encode_filters: Vec<String>,
    /// Frame format entering the encoder (before `encode_filters`).
    pub encode_input: Format,
    pub input: Format,
    pub output: Format,
    pub is_final: bool,
    /// Whether the pass can resume by seeking (no frame-number dependent filters).
    pub seekable: bool,
    /// Stage indices included in this pass.
    pub stages: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilizePass {
    pub stage: usize,
    /// Decoder filters before vidstabdetect.
    pub pre_filters: Vec<String>,
    pub detect_filter: String,
    /// `vidstabtransform` with `{TRF}` placeholder for the transforms path.
    pub transform_template: String,
}

/// Estimates presented as ranges.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Estimates {
    pub peak_memory_mb: (f64, f64),
    pub output_size_mb: (f64, f64),
    pub work_disk_mb: (f64, f64),
    pub output_frames: u64,
    pub input_frames: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub input: Format,
    pub output: Format,
    pub stages: Vec<StagePlan>,
    pub passes: Vec<PassPlan>,
    pub issues: Vec<Issue>,
    pub estimates: Estimates,
    pub output_duration: f64,
    pub input_duration: f64,
    pub output_color: ColorTags,
    /// Source is variable frame rate and will be converted to this constant rate.
    pub cfr_conversion: Option<Rational>,
}

impl ExecutionPlan {
    pub fn errors(&self) -> impl Iterator<Item = &Issue> {
        self.issues.iter().filter(|i| i.severity == Severity::Error)
    }
    pub fn has_errors(&self) -> bool {
        self.errors().next().is_some()
    }
}

/// FFmpeg capabilities relevant to planning.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FilterCaps {
    pub vidstab: bool,
    pub zscale: bool,
    pub tonemap: bool,
    pub bwdif: bool,
    pub nlmeans: bool,
    pub minterpolate: bool,
}

impl FilterCaps {
    pub fn all() -> Self {
        Self { vidstab: true, zscale: true, tonemap: true, bwdif: true, nlmeans: true, minterpolate: true }
    }
    pub fn from_tools(t: &clearframe_media::FfmpegTools) -> Self {
        Self {
            vidstab: t.has_filter("vidstabdetect") && t.has_filter("vidstabtransform"),
            zscale: t.has_filter("zscale"),
            tonemap: t.has_filter("tonemap"),
            bwdif: t.has_filter("bwdif"),
            nlmeans: t.has_filter("nlmeans"),
            minterpolate: t.has_filter("minterpolate"),
        }
    }
}

/// Snap a measured average rate to a common frame rate when close.
pub fn snap_fps(r: Rational) -> Rational {
    let f = r.as_f64();
    let common = [
        Rational::new(24000, 1001),
        Rational::new(24, 1),
        Rational::new(25, 1),
        Rational::new(30000, 1001),
        Rational::new(30, 1),
        Rational::new(48, 1),
        Rational::new(50, 1),
        Rational::new(60000, 1001),
        Rational::new(60, 1),
        Rational::new(90, 1),
        Rational::new(100, 1),
        Rational::new(120, 1),
        Rational::new(240, 1),
    ];
    for c in common {
        if (c.as_f64() - f).abs() / c.as_f64() < 0.03 {
            return c;
        }
    }
    Rational::new(f.round().max(1.0) as i64, 1)
}

fn even(v: f64) -> u32 {
    let r = v.round().max(2.0) as u32;
    r + (r % 2)
}

/// Compute (width, height, filters) for scaling `w x h` into `tw x th` with a fit mode.
pub fn fit_filters(w: u32, h: u32, tw: u32, th: u32, fit: FitMode, flags: &str) -> (u32, u32, Vec<String>) {
    let tw = even(tw as f64);
    let th = even(th as f64);
    if w == tw && h == th {
        return (tw, th, vec![]);
    }
    match fit {
        FitMode::Stretch => (tw, th, vec![format!("scale={tw}:{th}:flags={flags}")]),
        FitMode::Fit => {
            let s = (tw as f64 / w as f64).min(th as f64 / h as f64);
            let sw = even(w as f64 * s).min(tw);
            let sh = even(h as f64 * s).min(th);
            let mut f = vec![format!("scale={sw}:{sh}:flags={flags}")];
            if sw != tw || sh != th {
                f.push(format!("pad={tw}:{th}:{}:{}:color=black", (tw - sw) / 2, (th - sh) / 2));
            }
            (tw, th, f)
        }
        FitMode::Fill => {
            let s = (tw as f64 / w as f64).max(th as f64 / h as f64);
            let sw = even(w as f64 * s).max(tw);
            let sh = even(h as f64 * s).max(th);
            let mut f = vec![format!("scale={sw}:{sh}:flags={flags}")];
            if sw != tw || sh != th {
                f.push(format!("crop={tw}:{th}:{}:{}", (sw - tw) / 2, (sh - th) / 2));
            }
            (tw, th, f)
        }
    }
}

fn transfer_name(t: &Option<String>) -> &str {
    match t.as_deref() {
        Some("smpte2084") => "smpte2084",
        Some("arib-std-b67") => "arib-std-b67",
        Some("linear") => "linear",
        Some("smpte170m") | Some("bt470bg") | Some("bt601") => "601",
        _ => "709",
    }
}

fn primaries_name(p: &Option<String>) -> &str {
    match p.as_deref() {
        Some("bt2020") => "2020",
        Some("smpte170m") => "170m",
        Some("bt470bg") => "bt470bg",
        Some("smpte432") => "smpte432",
        _ => "709",
    }
}

/// Default planner options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanOptions {
    /// Source was analysed with idet and found interlaced.
    pub detected_interlaced: Option<bool>,
    pub caps: FilterCaps,
}

impl Default for PlanOptions {
    fn default() -> Self {
        Self { detected_interlaced: None, caps: FilterCaps::all() }
    }
}

/// Build the execution plan.
pub fn plan(
    info: &MediaInfo,
    job_range: Option<TimeRange>,
    pipeline: &Pipeline,
    output: &OutputSettings,
    models: &dyn ModelLookup,
    opts: &PlanOptions,
) -> Result<ExecutionPlan, String> {
    let v = info.video().map_err(|e| e.to_string())?;
    let mut issues: Vec<Issue> = Vec::new();
    let issue = |issues: &mut Vec<Issue>, sev: Severity, stage: Option<&str>, msg: String, action: Option<&str>| {
        issues.push(Issue { severity: sev, stage_id: stage.map(String::from), message: msg, action: action.map(String::from) });
    };
    pipeline.validate()?;

    let cfr_conversion = if v.variable_frame_rate { Some(snap_fps(v.avg_frame_rate)) } else { None };
    let fps_in = cfr_conversion.unwrap_or(if v.frame_rate.as_f64() > 240.0 { snap_fps(v.avg_frame_rate) } else { v.frame_rate });
    if let Some(r) = cfr_conversion {
        issue(&mut issues, Severity::Info, None, format!("The source has a variable frame rate; it will be converted to a constant {r} fps. Audio stays in sync."), None);
    }
    let flagged = v.is_interlaced_flagged();
    let interlaced = opts.detected_interlaced.unwrap_or(flagged);
    let input_duration = match job_range {
        Some(r) => r.duration().min((info.duration - r.start).max(0.0)),
        None => v.duration.unwrap_or(info.duration),
    };
    if input_duration <= 0.0 {
        return Err("the selected range is empty".into());
    }
    if let Some(r) = job_range {
        if r.start < 0.0 || r.start >= info.duration || r.end <= r.start {
            return Err("the selected range is outside the video".into());
        }
    }
    let input = Format {
        width: v.display_width,
        height: v.display_height,
        fps: fps_in,
        time_scale: 1.0,
        transfer: v.color.transfer.clone(),
        primaries: v.color.primaries.clone(),
        interlaced,
    };
    if v.display_width == 0 || v.display_height == 0 {
        return Err("the video has no valid dimensions".into());
    }
    let mut cur = input.clone();
    let mut stages: Vec<StagePlan> = Vec::new();
    let mut out_color = ColorTags::from_stream(v);
    let mut seen_deinterlace = false;
    let mut seen_native_before_deint = false;
    let enabled: Vec<&Stage> = pipeline.enabled().collect();
    if enabled.is_empty() {
        issue(&mut issues, Severity::Error, None, "Choose at least one operation to apply.".into(), Some("add_stage"));
    }

    for (idx, st) in enabled.iter().enumerate() {
        let sid = st.id.as_str();
        let inp = cur.clone();
        let mut notes = Vec::new();
        let mut is_native = false;
        let mut native_output: Option<Format> = None;
        let exec = match &st.op {
            Operation::Deinterlace(d) => {
                seen_deinterlace = true;
                if !cur.interlaced {
                    issue(&mut issues, Severity::Warning, Some(sid), "The source does not appear to be interlaced; deinterlacing progressive video can soften it.".into(), None);
                }
                let mode = match d.rate {
                    DeinterlaceRate::Frame => "send_frame",
                    DeinterlaceRate::Field => {
                        cur.fps = cur.fps.mul_int(2);
                        notes.push(format!("Outputs one frame per field: {} fps.", cur.fps));
                        "send_field"
                    }
                };
                let parity = match d.field_order {
                    FieldOrder::Auto => "auto",
                    FieldOrder::Tff => "tff",
                    FieldOrder::Bff => "bff",
                };
                let deint = if d.only_flagged { "interlaced" } else { "all" };
                let name = match d.method {
                    DeinterlaceMethod::Bwdif if opts.caps.bwdif => "bwdif",
                    _ => "yadif",
                };
                cur.interlaced = false;
                StageExec::Ffmpeg { filters: vec![format!("{name}=mode={mode}:parity={parity}:deint={deint}")] }
            }
            Operation::Denoise(d) => match d.engine {
                DenoiseEngine::Hqdn3d => {
                    let s = d.strength.max(0.0);
                    let t = if d.temporal { s * 0.6 } else { 0.0 };
                    if d.temporal {
                        notes.push("Temporal filtering can smear fast motion.".into());
                    }
                    let mut f = vec![format!("hqdn3d={:.2}:{:.2}:{:.2}:{:.2}", s * 0.4, s * 0.3, t, t * 0.75)];
                    if d.mix < 0.999 {
                        notes.push("Mix is only available for AI denoisers; classic filters apply at full strength.".into());
                        f.truncate(1);
                    }
                    StageExec::Ffmpeg { filters: f }
                }
                DenoiseEngine::Nlmeans => {
                    if !opts.caps.nlmeans {
                        issue(&mut issues, Severity::Error, Some(sid), "This FFmpeg build has no nlmeans filter.".into(), None);
                    }
                    StageExec::Ffmpeg { filters: vec![format!("nlmeans=s={:.2}:p=7:r=15", (d.strength / 4.0).clamp(1.0, 30.0))] }
                }
                DenoiseEngine::Ffdnet | DenoiseEngine::Drunet => {
                    is_native = true;
                    StageExec::Native { post_filters: vec![] }
                }
            },
            Operation::Deblur(_) | Operation::FaceRestore(_) | Operation::Grain(_) => {
                is_native = true;
                if let Operation::FaceRestore(_) = &st.op {
                    notes.push("Face restoration can change identity details and may invent features. Compare with the original before exporting.".into());
                }
                if let Operation::Deblur(_) = &st.op {
                    notes.push("Deblurring reconstructs detail and can create ringing or invented texture on heavily blurred frames.".into());
                }
                StageExec::Native { post_filters: vec![] }
            }
            Operation::Enhance(e) => {
                is_native = true;
                let facts = models.facts(&e.model_id);
                let scale = facts.as_ref().map(|f| f.scale).unwrap_or(1).max(1);
                let (mw, mh) = (cur.width * scale, cur.height * scale);
                let (tw, th, fit) = match &e.target {
                    SizeTarget::ModelScale => (mw, mh, FitMode::Stretch),
                    SizeTarget::Source => (cur.width, cur.height, FitMode::Stretch),
                    SizeTarget::Scale { factor } => (even(cur.width as f64 * *factor as f64), even(cur.height as f64 * *factor as f64), FitMode::Stretch),
                    SizeTarget::Height { height } => (even(cur.width as f64 * *height as f64 / cur.height as f64), *height, FitMode::Stretch),
                    SizeTarget::Custom { width, height, fit } => (*width, *height, *fit),
                };
                let (ow, oh, post) = fit_filters(mw, mh, tw, th, fit, "lanczos+accurate_rnd");
                native_output = Some(Format { width: mw, height: mh, ..cur.clone() });
                if !post.is_empty() {
                    notes.push(format!("Model output {mw}x{mh} is resized to {ow}x{oh} (Lanczos)."));
                }
                if ow > mw || oh > mh {
                    issue(
                        &mut issues,
                        Severity::Warning,
                        Some(sid),
                        format!("The requested size {ow}x{oh} is larger than the model's {scale}x output ({mw}x{mh}); the remainder is a plain resize without added detail."),
                        None,
                    );
                }
                if e.denoise.is_some() && !facts.as_ref().map(|f| f.supports_denoise_strength).unwrap_or(false) {
                    notes.push("The selected model has no denoise control; the denoise value is ignored.".into());
                }
                cur.width = ow;
                cur.height = oh;
                StageExec::Native { post_filters: post }
            }
            Operation::Interpolate(i) => {
                let target = match &i.target {
                    FpsTarget::Multiply { factor } => cur.fps.mul_int(*factor as i64),
                    FpsTarget::Fps { fps } => *fps,
                };
                if i.slow_motion <= 1.0001 && target.as_f64() < cur.fps.as_f64() - 1e-6 {
                    notes.push(format!("Target {target} fps is lower than the input {} fps: frames will be dropped/re-timed.", cur.fps));
                }
                if i.slow_motion > 1.0001 {
                    notes.push(format!(
                        "Slow motion x{:.2}: the output is {:.2}x longer. Audio: {}.",
                        i.slow_motion,
                        i.slow_motion,
                        match i.slow_motion_audio {
                            clearframe_media::mux::SlowMotionAudio::Mute => "removed",
                            clearframe_media::mux::SlowMotionAudio::StretchPreservePitch => "stretched, pitch preserved",
                            clearframe_media::mux::SlowMotionAudio::StretchLowerPitch => "slowed, pitch lowered",
                        }
                    ));
                } else {
                    notes.push("Duration is preserved; audio timing is unchanged.".into());
                }
                let exec = match i.engine {
                    InterpEngine::Rife | InterpEngine::Blend => {
                        is_native = true;
                        StageExec::Native { post_filters: vec![] }
                    }
                    InterpEngine::Mci => {
                        if !opts.caps.minterpolate {
                            issue(&mut issues, Severity::Error, Some(sid), "This FFmpeg build has no minterpolate filter.".into(), None);
                        }
                        let mut f = Vec::new();
                        if i.slow_motion > 1.0001 {
                            f.push(format!("setpts={:.6}*PTS", i.slow_motion));
                        }
                        f.push(format!(
                            "minterpolate=fps={}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd={}:scd_threshold={:.1}",
                            target.to_ffmpeg(),
                            if i.scene_detection { "fdiff" } else { "none" },
                            (i.scene_threshold * 100.0).clamp(1.0, 50.0)
                        ));
                        StageExec::Ffmpeg { filters: f }
                    }
                };
                cur.fps = target;
                cur.time_scale *= i.slow_motion.max(1.0);
                exec
            }
            Operation::Stabilize(s) => {
                if s.smoothing == 0 {
                    StageExec::Skip { reason: "Smoothing is 0: stabilization is off and frames pass through unchanged.".into() }
                } else {
                    if !opts.caps.vidstab {
                        issue(&mut issues, Severity::Error, Some(sid), "This FFmpeg build does not include vid.stab, which stabilization requires.".into(), Some("ffmpeg_vidstab"));
                    }
                    let (crop, optzoom) = match s.mode {
                        StabilizeMode::FullFrame => ("keep", 0),
                        StabilizeMode::AutoCrop => ("black", 1),
                        StabilizeMode::Black => ("black", 0),
                    };
                    let detect = format!("vidstabdetect=shakiness={}:accuracy=15:result={{TRF}}", s.shakiness.clamp(1, 10));
                    let transform =
                        format!("vidstabtransform=input={{TRF}}:smoothing={}:optzoom={optzoom}:crop={crop}:tripod={}:interpol=bicubic", s.smoothing, if s.tripod { 1 } else { 0 });
                    notes.push("vid.stab processes 8-bit 4:4:4 frames; higher bit depth is reduced to 8 bits for this step.".into());
                    match s.mode {
                        StabilizeMode::FullFrame => notes.push("Full frame: uncovered edges are filled from neighbouring frames and may show artifacts.".into()),
                        StabilizeMode::AutoCrop => notes.push("Auto crop: the image is zoomed to hide moving borders; the zoom is measured during analysis.".into()),
                        StabilizeMode::Black => notes.push("Uncovered edges are left black.".into()),
                    }
                    StageExec::Stabilize { detect, transform_template: transform }
                }
            }
            Operation::Color(c) => {
                let mut f = Vec::new();
                match c.mode {
                    ColorMode::Adjust => {
                        let changed = c.brightness.abs() > 1e-4 || (c.contrast - 1.0).abs() > 1e-4 || (c.saturation - 1.0).abs() > 1e-4 || (c.gamma - 1.0).abs() > 1e-4;
                        if changed {
                            f.push(format!("eq=brightness={:.4}:contrast={:.4}:saturation={:.4}:gamma={:.4}", c.brightness, c.contrast, c.saturation, c.gamma));
                        }
                    }
                    ColorMode::HdrToSdr => {
                        if !cur.is_hdr() {
                            issue(&mut issues, Severity::Error, Some(sid), "HDR to SDR needs an HDR (PQ or HLG) source; this video is SDR.".into(), None);
                        }
                        if !(opts.caps.zscale && opts.caps.tonemap) {
                            issue(&mut issues, Severity::Error, Some(sid), "This FFmpeg build lacks zscale/tonemap required for tone mapping.".into(), None);
                        }
                        let tm = match c.tone_mapper {
                            ToneMapper::Hable => "hable",
                            ToneMapper::Mobius => "mobius",
                            ToneMapper::Reinhard => "reinhard",
                            ToneMapper::Clip => "clip",
                        };
                        f.push(format!("zscale=tin={}:pin={}:t=linear:npl=100", transfer_name(&cur.transfer), primaries_name(&cur.primaries)));
                        f.push("format=gbrpf32le".into());
                        f.push("zscale=p=709".into());
                        f.push(format!("tonemap=tonemap={tm}:desat={:.2}", c.desaturate * 2.0));
                        f.push("zscale=t=709".into());
                        cur.transfer = Some("bt709".into());
                        cur.primaries = Some("bt709".into());
                        out_color.transfer = Some("bt709".into());
                        out_color.primaries = Some("bt709".into());
                        out_color.matrix = clearframe_media::color::YuvMatrix::Bt709;
                        out_color.mastering_display = None;
                        out_color.content_light = None;
                    }
                    ColorMode::SdrToHdr => {
                        if cur.is_hdr() {
                            issue(&mut issues, Severity::Error, Some(sid), "The video is already HDR.".into(), None);
                        }
                        if !opts.caps.zscale {
                            issue(&mut issues, Severity::Error, Some(sid), "This FFmpeg build lacks zscale required for HDR conversion.".into(), None);
                        }
                        notes.push("SDR to HDR maps the existing image into an HDR container; it cannot recover highlight or shadow detail that was never captured.".into());
                        let (t, tname) = match c.hdr_transfer {
                            HdrTransfer::Pq => ("smpte2084", "smpte2084"),
                            HdrTransfer::Hlg => ("arib-std-b67", "arib-std-b67"),
                        };
                        f.push(format!("zscale=tin={}:pin={}:t=linear:npl={:.0}", transfer_name(&cur.transfer), primaries_name(&cur.primaries), c.sdr_white_nits));
                        f.push("format=gbrpf32le".into());
                        f.push(format!("zscale=p=2020:t={t}:npl={:.0}", c.sdr_white_nits));
                        cur.transfer = Some(tname.into());
                        cur.primaries = Some("bt2020".into());
                        out_color.transfer = Some(tname.into());
                        out_color.primaries = Some("bt2020".into());
                        out_color.matrix = clearframe_media::color::YuvMatrix::Bt2020;
                        out_color.mastering_display = Some(format!("G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L({},50)", (c.peak_nits * 10000.0).round() as i64));
                        out_color.content_light = Some(format!("{},{}", c.peak_nits.round() as i64, (c.sdr_white_nits * 1.5).round() as i64));
                    }
                }
                if f.is_empty() {
                    StageExec::Skip { reason: "No colour change is configured.".into() }
                } else {
                    f.push("format=rgb48le".into());
                    StageExec::Ffmpeg { filters: f }
                }
            }
            Operation::Resize(r) => {
                let (ow, oh, f) = fit_filters(cur.width, cur.height, r.width, r.height, r.fit, &format!("{}+accurate_rnd", r.filter.ffmpeg()));
                cur.width = ow;
                cur.height = oh;
                if f.is_empty() {
                    StageExec::Skip { reason: "The video is already this size.".into() }
                } else {
                    StageExec::Ffmpeg { filters: f }
                }
            }
        };
        if is_native {
            if !seen_deinterlace && cur.interlaced {
                seen_native_before_deint = true;
            }
            if inp.is_hdr() {
                issue(
                    &mut issues,
                    Severity::Warning,
                    Some(sid),
                    "AI models are trained on SDR video; processing HDR (PQ/HLG) frames directly can shift tones. Consider tone mapping to SDR first.".into(),
                    None,
                );
            }
            for m in st.op.required_models() {
                match models.facts(&m) {
                    Some(f) if f.installed => {}
                    Some(f) => issue(&mut issues, Severity::Error, Some(sid), format!("{} is not installed.", f.display_name), Some(&format!("install_model:{m}"))),
                    None => issue(&mut issues, Severity::Error, Some(sid), format!("Model {m} is not in the catalog."), Some("choose_model")),
                }
            }
        }
        let models_used = if is_native { st.op.required_models() } else { vec![] };
        stages.push(StagePlan {
            stage_id: st.id.clone(),
            index: idx,
            label: st.op.label().into(),
            exec,
            input: inp,
            output: cur.clone(),
            native_output,
            models: models_used,
            notes,
        });
    }

    if interlaced && !seen_deinterlace {
        issue(
            &mut issues,
            Severity::Warning,
            None,
            "The source appears to be interlaced. Add a Deinterlace step first; enhancement models expect progressive frames.".into(),
            Some("add_deinterlace"),
        );
    } else if seen_native_before_deint {
        issue(&mut issues, Severity::Warning, None, "An AI step runs before deinterlacing; move Deinterlace to the start of the pipeline.".into(), Some("move_deinterlace_first"));
    }
    order_hints(&enabled, &mut issues);

    // Output format checks.
    let depth = pix_fmt_bit_depth(&output.video.pix_fmt);
    if out_color.is_hdr() && depth < 10 {
        issue(
            &mut issues,
            Severity::Error,
            None,
            format!("HDR output needs a 10-bit pixel format; {} is {}-bit. Choose a 10-bit format such as yuv420p10le.", output.video.pix_fmt, depth),
            Some("set_pix_fmt:yuv420p10le"),
        );
    }
    if v.bit_depth > 8 && depth == 8 && !out_color.is_hdr() {
        issue(&mut issues, Severity::Info, None, format!("The source is {}-bit; the output pixel format is 8-bit.", v.bit_depth), None);
    }
    let family = clearframe_media::encode::codec_profile(&output.video.encoder).map(|c| c.family).unwrap_or("h264");
    if !output.container.video_ok(family) {
        issue(
            &mut issues,
            Severity::Error,
            None,
            format!("{} cannot be stored in {}. Choose another container or codec.", family.to_uppercase(), output.container.extension().to_uppercase()),
            None,
        );
    }
    if (output.video.pix_fmt.contains("420") || output.video.pix_fmt.contains("422") || output.video.pix_fmt == "nv12" || output.video.pix_fmt == "p010le")
        && (!cur.width.is_multiple_of(2) || !cur.height.is_multiple_of(2))
    {
        issue(
            &mut issues,
            Severity::Error,
            None,
            format!("{}x{} has odd dimensions, which {} cannot encode. Add a Resize step.", cur.width, cur.height, output.video.pix_fmt),
            None,
        );
    }
    if cur.width > 16384 || cur.height > 16384 {
        issue(&mut issues, Severity::Error, None, format!("Output size {}x{} exceeds supported limits.", cur.width, cur.height), None);
    }

    let passes = build_passes(&stages, &input);
    let output_duration = input_duration * cur.time_scale;
    let input_frames = (input_duration * input.fps.as_f64()).round() as u64;
    let output_frames = (output_duration * cur.fps.as_f64()).round() as u64;
    let estimates = estimate(&stages, &passes, &cur, output, models, input_frames, output_frames, output_duration);
    Ok(ExecutionPlan { input, output: cur, stages, passes, issues, estimates, output_duration, input_duration, output_color: out_color, cfr_conversion })
}

fn order_hints(stages: &[&Stage], issues: &mut Vec<Issue>) {
    let pos = |k: &str| stages.iter().position(|s| s.op.kind() == k);
    let mut hint = |msg: &str, sid: &str| {
        issues.push(Issue { severity: Severity::Info, stage_id: Some(sid.to_string()), message: msg.to_string(), action: None });
    };
    if let (Some(g), Some(e)) = (pos("grain"), pos("enhance")) {
        if g < e {
            hint("Grain added before Enhance will be treated as detail or noise by the model. Grain usually goes last.", &stages[g].id);
        }
    }
    if let (Some(g), Some(d)) = (pos("grain"), pos("denoise")) {
        if g < d {
            hint("Denoise after Add grain will remove the grain you added.", &stages[d].id);
        }
    }
    if let (Some(d), Some(e)) = (pos("denoise"), pos("enhance")) {
        if d > e {
            hint("Denoising after Enhance is slower (more pixels) and the model may have sharpened the noise. Denoise usually goes first.", &stages[d].id);
        }
    }
    if let (Some(i), Some(e)) = (pos("interpolate"), pos("enhance")) {
        if i < e {
            hint("Interpolating before Enhance multiplies the number of frames the model must upscale.", &stages[i].id);
        }
    }
    if let (Some(f), Some(e)) = (pos("face_restore"), pos("enhance")) {
        if f < e {
            hint("Face recovery usually works best after Enhance, at the final resolution.", &stages[f].id);
        }
    }
}

fn build_passes(stages: &[StagePlan], input: &Format) -> Vec<PassPlan> {
    let mut passes: Vec<PassPlan> = Vec::new();
    let new_pass = |index: usize, from_intermediate: bool, fmt: &Format| PassPlan {
        index,
        from_intermediate,
        decode_filters: vec![],
        stabilize: None,
        segments: vec![],
        encode_filters: vec![],
        encode_input: fmt.clone(),
        input: fmt.clone(),
        output: fmt.clone(),
        is_final: false,
        seekable: true,
        stages: vec![],
    };
    let mut p = new_pass(0, false, input);
    // pending ffmpeg filters not yet attached
    let mut pending: Vec<String> = Vec::new();
    let mut pending_in: Option<Format> = None;
    let mut native_seen = false;

    for sp in stages {
        match &sp.exec {
            StageExec::Skip { .. } => {
                p.stages.push(sp.index);
            }
            StageExec::Ffmpeg { filters } => {
                if pending_in.is_none() {
                    pending_in = Some(sp.input.clone());
                }
                pending.extend(filters.iter().cloned());
                p.stages.push(sp.index);
            }
            StageExec::Stabilize { detect, transform_template } => {
                if native_seen {
                    // Close this pass into an intermediate and start a new one.
                    p.encode_input = pending_in.take().unwrap_or_else(|| sp.input.clone());
                    if !pending.is_empty() {
                        p.encode_filters = std::mem::take(&mut pending);
                    }
                    p.output = sp.input.clone();
                    passes.push(p);
                    p = new_pass(passes.len(), true, &sp.input);
                    native_seen = false;
                }
                // Everything before in this pass is FFmpeg-only, held in `pending`.
                let pre = std::mem::take(&mut pending);
                p.stabilize = Some(StabilizePass { stage: sp.index, pre_filters: pre.clone(), detect_filter: detect.clone(), transform_template: transform_template.clone() });
                pending = pre;
                // vid.stab only accepts 8-bit YUV/RGB24; convert explicitly (4:4:4 keeps chroma).
                pending.push("format=yuv444p".into());
                pending.push(transform_template.clone());
                if pending_in.is_none() {
                    pending_in = Some(sp.input.clone());
                }
                p.seekable = false;
                p.stages.push(sp.index);
            }
            StageExec::Native { post_filters } => {
                if !pending.is_empty() {
                    let filters = std::mem::take(&mut pending);
                    let fin = pending_in.take().unwrap_or_else(|| sp.input.clone());
                    if !native_seen {
                        p.decode_filters = filters;
                    } else {
                        p.segments.push(Segment::Ffmpeg { filters, input: fin, output: sp.input.clone() });
                    }
                }
                pending_in = None;
                p.segments.push(Segment::Native { stage: sp.index });
                native_seen = true;
                if !post_filters.is_empty() {
                    pending_in = Some(native_raw_output(sp));
                    pending.extend(post_filters.iter().cloned());
                }
                p.stages.push(sp.index);
            }
        }
    }
    p.output = stages.last().map(|s| s.output.clone()).unwrap_or_else(|| input.clone());
    if !pending.is_empty() {
        if native_seen {
            p.encode_input = pending_in.take().unwrap_or_else(|| p.output.clone());
            p.encode_filters = pending;
        } else {
            p.decode_filters = pending;
            p.encode_input = p.output.clone();
        }
    } else {
        p.encode_input = p.output.clone();
    }
    p.is_final = true;
    passes.push(p);
    passes
}

/// Format produced by a native stage before its post filters.
pub fn native_raw_output(sp: &StagePlan) -> Format {
    sp.native_output.clone().unwrap_or_else(|| sp.output.clone())
}

#[allow(clippy::too_many_arguments)]
fn estimate(
    stages: &[StagePlan],
    passes: &[PassPlan],
    out: &Format,
    output: &OutputSettings,
    models: &dyn ModelLookup,
    input_frames: u64,
    output_frames: u64,
    output_duration: f64,
) -> Estimates {
    let mut peak = 400.0f64; // application + decoder/encoder processes
    for sp in stages {
        let frame_mb = sp.output.megapixels() * 12.0; // f32 RGB
        peak = peak.max(400.0 + frame_mb * 8.0);
        if let StageExec::Native { .. } = sp.exec {
            // Models process tiles of up to ~0.25-1 MP.
            let tile_mp = sp.input.megapixels().min(0.6);
            let model_mb = stages_model_mem(sp, models, tile_mp);
            peak = peak.max(400.0 + frame_mb * 8.0 + model_mb);
        }
    }
    // bits per pixel heuristics by rate control
    let px_per_sec = out.megapixels() * 1e6 * out.fps.as_f64();
    let (lo_bpp, hi_bpp) = match &output.video.rate {
        clearframe_media::encode::RateControl::Bitrate { kbps } => {
            let b = *kbps as f64 * 1000.0 / px_per_sec.max(1.0);
            (b * 0.9, b * 1.1)
        }
        clearframe_media::encode::RateControl::Lossless => (4.0, 12.0),
        clearframe_media::encode::RateControl::Quality { value } => {
            let base = match output.video.encoder.as_str() {
                "prores_ks" => 2.5,
                "ffv1" => 6.0,
                e if e.contains("265") || e.contains("hevc") => 0.06,
                e if e.contains("av1") || e.contains("vp9") => 0.05,
                _ => 0.1,
            };
            let f = 2f64.powf((18.0 - *value as f64) / 6.0);
            (base * f * 0.4, base * f * 2.0)
        }
    };
    let video_mb = |bpp: f64| px_per_sec * bpp * output_duration / 8.0 / 1e6;
    let mut work = 0.0;
    for p in passes.iter().filter(|p| !p.is_final) {
        work += p.output.megapixels() * 1e6 * 6.0 * 0.5 * (output_duration / out.time_scale) * p.output.fps.as_f64() / 1e6;
    }
    let (out_lo, out_hi) = (video_mb(lo_bpp).max(0.1), video_mb(hi_bpp).max(0.2));
    Estimates { peak_memory_mb: (peak * 0.8, peak * 1.5), output_size_mb: (out_lo, out_hi), work_disk_mb: (work * 0.6 + out_lo, work * 1.4 + out_hi), output_frames, input_frames }
}

fn stages_model_mem(sp: &StagePlan, models: &dyn ModelLookup, tile_mp: f64) -> f64 {
    let mut total = 0.0;
    for id in &sp.models {
        match models.facts(id) {
            Some(f) => total += f.weights_mb + f.mb_per_megapixel * tile_mp,
            None => total += 600.0 + 1200.0 * tile_mp,
        }
    }
    if sp.models.is_empty() {
        // Native non-model stages (grain, blend interpolation) need a few frame buffers.
        total += sp.output.megapixels() * 24.0;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use clearframe_media::probe::parse_probe_json;

    struct Models;
    impl ModelLookup for Models {
        fn facts(&self, id: &str) -> Option<ModelFacts> {
            let scale = if id.contains("realesr") || id.contains("esrgan") { 4 } else { 1 };
            Some(ModelFacts {
                id: id.into(),
                display_name: id.into(),
                scale,
                installed: !id.contains("missing"),
                mb_per_megapixel: 1000.0,
                weights_mb: 10.0,
                supports_denoise_strength: id.contains("general"),
            })
        }
    }

    fn info(extra: &str) -> MediaInfo {
        let j = format!(
            r#"{{"streams":[{{"index":0,"codec_type":"video","codec_name":"h264","width":640,"height":360,"pix_fmt":"yuv420p",
            "r_frame_rate":"30/1","avg_frame_rate":"30/1"{extra}}},
            {{"index":1,"codec_type":"audio","codec_name":"aac","channels":2,"sample_rate":"48000"}}],"format":{{"duration":"10"}}}}"#
        );
        parse_probe_json("in.mp4", &j, None).unwrap()
    }

    fn st(op: Operation) -> Stage {
        Stage::new(op)
    }

    #[test]
    fn enhance_then_interpolate() {
        let p = Pipeline {
            stages: vec![
                st(Operation::Enhance(EnhanceSettings { target: SizeTarget::Height { height: 1080 }, ..Default::default() })),
                st(Operation::Interpolate(InterpolateSettings::default())),
            ],
        };
        let plan = plan(&info(""), None, &p, &OutputSettings::default(), &Models, &PlanOptions::default()).unwrap();
        assert!(!plan.has_errors(), "{:?}", plan.issues);
        assert_eq!((plan.output.width, plan.output.height), (1920, 1080));
        assert_eq!(plan.output.fps, Rational::new(60, 1));
        assert_eq!(plan.passes.len(), 1);
        let pass = &plan.passes[0];
        assert!(pass.decode_filters.is_empty());
        // enhance (native) -> ffmpeg resize -> interpolate (native)
        assert_eq!(pass.segments.len(), 3);
        match &pass.segments[1] {
            Segment::Ffmpeg { filters, input, output } => {
                assert!(filters[0].starts_with("scale=1920:1080"));
                assert_eq!((input.width, input.height), (2560, 1440));
                assert_eq!((output.width, output.height), (1920, 1080));
            }
            s => panic!("{s:?}"),
        }
        assert_eq!(plan.estimates.output_frames, 600);
    }

    #[test]
    fn ffmpeg_only_runs_in_decoder() {
        let p = Pipeline {
            stages: vec![
                st(Operation::Deinterlace(DeinterlaceSettings { rate: DeinterlaceRate::Field, ..Default::default() })),
                st(Operation::Denoise(DenoiseSettings { engine: DenoiseEngine::Hqdn3d, ..Default::default() })),
            ],
        };
        let plan = plan(&info(r#","field_order":"tt""#), None, &p, &OutputSettings::default(), &Models, &PlanOptions::default()).unwrap();
        assert_eq!(plan.passes[0].decode_filters.len(), 2);
        assert!(plan.passes[0].segments.is_empty());
        assert_eq!(plan.output.fps, Rational::new(60, 1));
        assert!(!plan.issues.iter().any(|i| i.action.as_deref() == Some("add_deinterlace")));
    }

    #[test]
    fn interlace_warning_and_missing_model_error() {
        let p = Pipeline { stages: vec![st(Operation::Enhance(EnhanceSettings { model_id: "x.realesr-missing".into(), ..Default::default() }))] };
        let plan = plan(&info(r#","field_order":"tt""#), None, &p, &OutputSettings::default(), &Models, &PlanOptions::default()).unwrap();
        assert!(plan.issues.iter().any(|i| i.action.as_deref() == Some("add_deinterlace")));
        assert!(plan.has_errors());
        assert!(plan.errors().any(|i| i.action.as_deref().unwrap_or("").starts_with("install_model:")));
    }

    #[test]
    fn stabilize_after_native_splits_passes() {
        let p = Pipeline {
            stages: vec![
                st(Operation::Denoise(DenoiseSettings::default())),
                st(Operation::Stabilize(StabilizeSettings::default())),
                st(Operation::Grain(GrainSettings::default())),
            ],
        };
        let plan = plan(&info(""), None, &p, &OutputSettings::default(), &Models, &PlanOptions::default()).unwrap();
        assert_eq!(plan.passes.len(), 2);
        assert!(!plan.passes[0].is_final);
        assert!(plan.passes[1].from_intermediate);
        assert!(plan.passes[1].stabilize.is_some());
        assert!(!plan.passes[1].seekable);
        assert_eq!(plan.passes[1].decode_filters[0], "format=yuv444p");
        assert!(plan.passes[1].decode_filters[1].starts_with("vidstabtransform"));
    }

    #[test]
    fn stabilize_off_is_passthrough() {
        let p = Pipeline { stages: vec![st(Operation::Stabilize(StabilizeSettings { smoothing: 0, ..Default::default() }))] };
        let plan = plan(&info(""), None, &p, &OutputSettings::default(), &Models, &PlanOptions::default()).unwrap();
        assert!(matches!(plan.stages[0].exec, StageExec::Skip { .. }));
        assert!(plan.passes[0].decode_filters.is_empty());
    }

    #[test]
    fn hdr_needs_10bit_and_slowmo_duration() {
        let c = ColorSettings { mode: ColorMode::SdrToHdr, ..Default::default() };
        let p = Pipeline {
            stages: vec![st(Operation::Color(c)), st(Operation::Interpolate(InterpolateSettings { slow_motion: 2.0, engine: InterpEngine::Blend, ..Default::default() }))],
        };
        let plan = plan(&info(""), Some(TimeRange { start: 1.0, end: 3.0 }), &p, &OutputSettings::default(), &Models, &PlanOptions::default()).unwrap();
        assert!(plan.errors().any(|e| e.message.contains("10-bit")));
        assert!((plan.output_duration - 4.0).abs() < 1e-9);
        assert_eq!(plan.output_color.transfer.as_deref(), Some("smpte2084"));
    }

    #[test]
    fn fit_modes() {
        let (w, h, f) = fit_filters(1920, 800, 1920, 1080, FitMode::Fit, "lanczos");
        assert_eq!((w, h), (1920, 1080));
        assert!(f[1].starts_with("pad=1920:1080:0:140"));
        let (_, _, f) = fit_filters(1920, 800, 1920, 1080, FitMode::Fill, "lanczos");
        assert!(f[1].starts_with("crop=1920:1080"));
        let (_, _, f) = fit_filters(10, 10, 10, 10, FitMode::Fill, "x");
        assert!(f.is_empty());
    }

    #[test]
    fn vfr_snaps() {
        assert_eq!(snap_fps(Rational::new(2997, 100)), Rational::new(30000, 1001));
        assert_eq!(snap_fps(Rational::new(3333, 100)), Rational::new(33, 1));
    }
}
