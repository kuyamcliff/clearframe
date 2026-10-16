//! Executes a [`JobSpec`]: plans the work, runs each pass through decoder ->
//! processors -> segmented encoder, assembles the output with audio and
//! subtitles, validates it and moves it into place without overwriting.
//!
//! Recovery: every pass encodes ~10 s Matroska segments and records completed
//! segments, so a cancelled, paused or crashed job resumes from the last
//! finished segment. Passes containing frame-number dependent filters
//! (stabilisation) restart from the start of that pass; analysis results are
//! kept.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use crossbeam_channel::bounded;
use serde::{Deserialize, Serialize};

use clearframe_core::fsutil;
use clearframe_inference::device::{self, BackendKind};
use clearframe_media::color::{ColorTags, YuvMatrix};
use clearframe_media::encode::{codec_profile, VideoEncodeSettings};
use clearframe_media::mux::{self, MuxReport, MuxSpec};
use clearframe_media::pipe::{DecodeSpec, EncodeOutput, EncodeSpec, FrameReader, FrameWriter, RawFrame};
use clearframe_media::probe::{probe, MediaInfo};
use clearframe_media::validate::{self, Expected, ValidationReport};
use clearframe_media::FfmpegTools;

use crate::plan::{self, ExecutionPlan, FilterCaps, ModelLookup, PassPlan, PlanOptions, Segment, StageExec};
use crate::processors::{build_native, FfmpegProc, Frame, Timed};
use crate::spec::*;

pub use clearframe_core::jobs::JobState;

/// Time spent per phase.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Timings {
    pub prepare_s: f64,
    pub analyze_s: f64,
    pub decode_wait_s: f64,
    pub stages_s: BTreeMap<String, f64>,
    pub encode_wait_s: f64,
    pub assemble_s: f64,
    pub validate_s: f64,
    pub total_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Progress {
    pub pass: usize,
    pub passes: usize,
    pub phase: String,
    pub frames_done: u64,
    pub frames_total: u64,
    /// Overall fraction 0..1 across passes and phases.
    pub fraction: f64,
    /// Measured output frames per second over the recent window.
    pub fps: f64,
    /// Remaining time range in seconds, once measurable.
    pub eta: Option<(f64, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    State {
        state: JobState,
        detail: String,
    },
    Plan {
        plan: Box<ExecutionPlan>,
    },
    Progress(Progress),
    Log {
        level: String,
        message: String,
    },
    Warning {
        message: String,
    },
    Resumed {
        pass: usize,
        from_frame: u64,
    },
    /// Throttled resource sample of the worker and its FFmpeg children.
    Resources(ResourceSample),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceSample {
    /// Resident memory of the worker plus child processes (MB).
    pub process_memory_mb: u64,
    /// CPU use of the worker plus children (100 = one full core).
    pub process_cpu_percent: f32,
    pub system_memory_used_mb: u64,
    pub system_memory_total_mb: u64,
    pub system_memory_available_mb: u64,
    pub backend: String,
    pub stage: String,
    pub free_disk_mb: Option<u64>,
}

/// Samples resource use of this process tree at most once per interval.
pub struct ResourceMonitor {
    sys: sysinfo::System,
    pid: sysinfo::Pid,
    last: Option<Instant>,
}

impl Default for ResourceMonitor {
    fn default() -> Self {
        Self { sys: sysinfo::System::new(), pid: sysinfo::get_current_pid().unwrap_or(sysinfo::Pid::from(0)), last: None }
    }
}

impl ResourceMonitor {
    pub fn sample(&mut self, backend: &str, stage: &str, disk: Option<&Path>) -> Option<ResourceSample> {
        if self.last.is_some_and(|l| l.elapsed() < std::time::Duration::from_secs(2)) {
            return None;
        }
        self.last = Some(Instant::now());
        self.sys.refresh_memory();
        self.sys.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
        let mut mem = 0u64;
        let mut cpu = 0f32;
        for p in self.sys.processes().values() {
            if p.pid() == self.pid || p.parent() == Some(self.pid) {
                mem += p.memory();
                cpu += p.cpu_usage();
            }
        }
        Some(ResourceSample {
            process_memory_mb: mem / (1 << 20),
            process_cpu_percent: cpu,
            system_memory_used_mb: self.sys.used_memory() / (1 << 20),
            system_memory_total_mb: self.sys.total_memory() / (1 << 20),
            system_memory_available_mb: self.sys.available_memory() / (1 << 20),
            backend: backend.to_string(),
            stage: stage.to_string(),
            free_disk_mb: disk.and_then(fsutil::free_space).map(|b| b / 1_000_000),
        })
    }
}

/// The GPU that image models run on (cross-vendor wgpu backend).
#[cfg(feature = "gpu")]
pub type GpuRef = std::sync::Arc<clearframe_inference::gpu::Gpu>;
#[cfg(not(feature = "gpu"))]
pub type GpuRef = ();

/// Where a job computes: the tensor device, plus the GPU used for image models.
pub struct Compute {
    pub kind: BackendKind,
    pub device: clearframe_inference::Device,
    pub gpu: Option<GpuRef>,
    /// Human-readable name for logs and reports.
    pub label: String,
}

/// Resolve the compute device from job resources ("auto", "cpu", "gpu:N", "cuda:N", "metal:N").
pub fn select_compute(res: &JobResources, engine: &EngineSettings) -> std::result::Result<Compute, String> {
    // CLIPFX_DEVICE overrides the setting (support and automated tests, e.g. "gpu:0" to use a
    // software GPU that is never chosen automatically).
    let d = std::env::var("CLIPFX_DEVICE").unwrap_or_else(|_| res.device.clone()).trim().to_ascii_lowercase();
    let parse = |p: &str| d.strip_prefix(p).and_then(|n| n.parse::<usize>().ok());
    let plain = |kind: BackendKind, device| Compute { kind, device, gpu: None, label: kind.label().to_string() };
    if d == "cpu" {
        return Ok(plain(BackendKind::Cpu, clearframe_inference::Device::Cpu));
    }
    if let Some(i) = parse("cuda:") {
        return device::create(BackendKind::Cuda, i).map(|x| plain(BackendKind::Cuda, x));
    }
    if let Some(i) = parse("metal:") {
        return device::create(BackendKind::Metal, i).map(|x| plain(BackendKind::Metal, x));
    }
    let gpu_index = parse("gpu:");
    let (kind, dev, index) = match gpu_index {
        Some(i) => (BackendKind::Gpu, device::create(BackendKind::Gpu, i)?, i),
        None => {
            let (k, dv) = device::select(engine.backend)?;
            (k, dv, device::best_hardware_gpu().unwrap_or(0))
        }
    };
    if kind != BackendKind::Gpu {
        return Ok(plain(kind, dev));
    }
    gpu_compute(index, dev)
}

#[cfg(feature = "gpu")]
fn gpu_compute(index: usize, device: clearframe_inference::Device) -> std::result::Result<Compute, String> {
    let g = device::gpu_context(index)?;
    let label = format!("GPU · {} ({})", g.info.name, g.info.api);
    Ok(Compute { kind: BackendKind::Gpu, device, gpu: Some(g), label })
}

#[cfg(not(feature = "gpu"))]
fn gpu_compute(_index: usize, _device: clearframe_inference::Device) -> std::result::Result<Compute, String> {
    Err("this build has no GPU backend".into())
}

/// Resolve the compute device from job resources ("auto", "cpu", "gpu:N", "cuda:N", "metal:N").
pub fn select_device(res: &JobResources, engine: &EngineSettings) -> std::result::Result<(BackendKind, clearframe_inference::Device), String> {
    select_compute(res, engine).map(|c| (c.kind, c.device))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobReport {
    pub job_id: String,
    pub app_version: String,
    pub ffmpeg_version: String,
    pub backend: String,
    #[serde(default)]
    pub resources: JobResources,
    pub source: PathBuf,
    pub source_fingerprint: String,
    pub output: PathBuf,
    pub range: Option<TimeRange>,
    pub pipeline: Pipeline,
    pub output_settings: OutputSettings,
    pub models: BTreeMap<String, ResolvedModel>,
    pub output_width: u32,
    pub output_height: u32,
    pub output_fps: String,
    pub output_duration: f64,
    pub frames_written: u64,
    pub started_at: String,
    pub finished_at: String,
    pub timings: Timings,
    pub warnings: Vec<String>,
    pub stage_stats: BTreeMap<String, serde_json::Value>,
    pub stabilize_zoom_percent: Option<f64>,
    pub validation: Option<ValidationReport>,
    pub mux: Option<MuxReport>,
}

/// Why a job stopped.
#[derive(Debug, thiserror::Error)]
pub enum JobError {
    #[error("cancelled")]
    Cancelled,
    #[error("paused")]
    Paused,
    #[error("{0}")]
    Invalid(String),
    /// Stopped at a safe point because a resource ran out (disk space). Resumable.
    #[error("{0}")]
    Resources(String),
    #[error("{0:#}")]
    Failed(anyhow::Error),
}

impl From<anyhow::Error> for JobError {
    fn from(e: anyhow::Error) -> Self {
        JobError::Failed(e)
    }
}

/// Stop signal shared with the controller. `Pause` keeps work for resuming.
#[derive(Clone, Default)]
pub struct StopFlag {
    cancel: Arc<AtomicBool>,
    pause: Arc<AtomicBool>,
}

impl StopFlag {
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }
    pub fn pause(&self) {
        self.pause.store(true, Ordering::SeqCst);
    }
    fn check(&self) -> std::result::Result<(), JobError> {
        if self.cancel.load(Ordering::SeqCst) {
            return Err(JobError::Cancelled);
        }
        if self.pause.load(Ordering::SeqCst) {
            return Err(JobError::Paused);
        }
        Ok(())
    }
}

/// Persisted recovery state in `work_dir/state.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct RunState {
    plan_key: String,
    passes_done: Vec<bool>,
    analysis_done: Vec<bool>,
    zoom: Option<f64>,
    heartbeat: String,
}

impl RunState {
    fn load(dir: &Path) -> Option<Self> {
        serde_json::from_slice(&std::fs::read(dir.join("state.json")).ok()?).ok()
    }
    fn save(&mut self, dir: &Path) -> Result<()> {
        self.heartbeat = chrono::Utc::now().to_rfc3339();
        let tmp = dir.join("state.json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(self)?)?;
        std::fs::rename(&tmp, dir.join("state.json"))?;
        Ok(())
    }
}

/// Information about resumable work for the UI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryInfo {
    pub passes_done: usize,
    pub segments_done: usize,
    pub last_heartbeat: String,
}

pub fn recovery_info(work_dir: &Path) -> Option<RecoveryInfo> {
    let st = RunState::load(work_dir)?;
    let mut segs = 0;
    for i in 0..st.passes_done.len() {
        segs += read_segment_list(&work_dir.join(format!("pass{i}")).join("segments.csv")).len();
    }
    Some(RecoveryInfo { passes_done: st.passes_done.iter().filter(|d| **d).count(), segments_done: segs, last_heartbeat: st.heartbeat })
}

/// (file name, start, end) for each finished segment.
fn read_segment_list(path: &Path) -> Vec<(String, f64, f64)> {
    let Ok(s) = std::fs::read_to_string(path) else { return vec![] };
    s.lines()
        .filter_map(|l| {
            let mut it = l.rsplitn(3, ',');
            let end = it.next()?.trim().parse::<f64>().ok()?;
            let start = it.next()?.trim().parse::<f64>().ok()?;
            let name = it.next()?.trim().trim_matches('"').to_string();
            Some((name, start, end))
        })
        .collect()
}

fn plan_key(spec: &JobSpec, fingerprint: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    serde_json::to_string(&(&spec.pipeline, &spec.output, &spec.range, &spec.engine)).unwrap_or_default().hash(&mut h);
    fingerprint.hash(&mut h);
    for m in spec.models.values() {
        for v in m.sha256.values() {
            v.hash(&mut h);
        }
    }
    format!("{:016x}", h.finish())
}

/// Model facts derived from resolved models (used inside workers).
pub struct ResolvedLookup<'a> {
    pub models: &'a BTreeMap<String, ResolvedModel>,
    /// Optional scale overrides (otherwise `ResolvedModel::scale`).
    pub scales: BTreeMap<String, u32>,
}

impl ModelLookup for ResolvedLookup<'_> {
    fn facts(&self, id: &str) -> Option<plan::ModelFacts> {
        let m = self.models.get(id)?;
        Some(plan::ModelFacts {
            id: id.to_string(),
            display_name: id.to_string(),
            scale: *self.scales.get(id).unwrap_or(&m.scale.max(1)),
            installed: true,
            mb_per_megapixel: 1500.0,
            weights_mb: 100.0,
            supports_denoise_strength: m.files.contains_key("wdn"),
        })
    }
}

/// Read model scale from converted weights without building the network.
pub fn model_scale(m: &ResolvedModel) -> u32 {
    let Some(p) = m.files.get("weights") else { return 1 };
    let Ok(w) = clearframe_inference::Weights::from_safetensors(p, &clearframe_inference::Device::Cpu) else { return 1 };
    match clearframe_inference::Architecture::detect(&w) {
        Some(clearframe_inference::Architecture::SrVggCompact) => {
            let n = w.count_indexed("body.");
            w.get(&format!("body.{}.weight", n - 1)).ok().and_then(|t| t.dim(0).ok()).map(|c| ((c / 3) as f64).sqrt().round() as u32).unwrap_or(1)
        }
        Some(clearframe_inference::Architecture::RrdbNet) => match w.get("conv_first.weight").ok().and_then(|t| t.dim(1).ok()) {
            Some(3) => 4,
            Some(12) => 2,
            _ => 1,
        },
        _ => 1,
    }
}

struct Ctx<'a> {
    tools: &'a FfmpegTools,
    spec: &'a JobSpec,
    info: MediaInfo,
    plan: ExecutionPlan,
    stop: StopFlag,
    events: &'a mut dyn FnMut(Event),
    timings: Timings,
    stage_stats: BTreeMap<String, serde_json::Value>,
    warnings: Vec<String>,
    started: Instant,
    total_units: f64,
    done_units: f64,
    backend_label: String,
    monitor: ResourceMonitor,
}

impl Ctx<'_> {
    fn emit(&mut self, e: Event) {
        (self.events)(e);
    }
    fn state(&mut self, s: JobState, d: &str) {
        self.emit(Event::State { state: s, detail: d.to_string() });
    }
    fn log(&mut self, msg: impl Into<String>) {
        let m = msg.into();
        log::info!("{m}");
        self.emit(Event::Log { level: "info".into(), message: m });
    }
    fn warn(&mut self, msg: impl Into<String>) {
        let m = msg.into();
        self.warnings.push(m.clone());
        self.emit(Event::Warning { message: m });
    }
}

/// Plan a job without running it (used by the UI for validation and estimates).
pub fn plan_job(tools: &FfmpegTools, spec: &JobSpec, lookup: &dyn ModelLookup, detected_interlaced: Option<bool>) -> Result<(MediaInfo, ExecutionPlan)> {
    let info = probe(tools, &spec.source).context("reading the source video")?;
    let p = plan_probed(tools, &info, spec, lookup, detected_interlaced)?;
    Ok((info, p))
}

/// [`plan_job`] for a source that was already probed.
pub fn plan_probed(tools: &FfmpegTools, info: &MediaInfo, spec: &JobSpec, lookup: &dyn ModelLookup, detected_interlaced: Option<bool>) -> Result<ExecutionPlan> {
    let opts = PlanOptions { detected_interlaced, caps: FilterCaps::from_tools(tools) };
    let interlaced = detected_interlaced.unwrap_or_else(|| info.video().map(|v| v.is_interlaced_flagged()).unwrap_or(false));
    let auto = spec.pipeline.with_auto_deinterlace(interlaced);
    plan::plan(info, spec.range, auto.as_ref().unwrap_or(&spec.pipeline), &spec.output, lookup, &opts).map_err(|e| anyhow!(e))
}

/// Run a job to completion.
pub fn run_job(tools: &FfmpegTools, spec: &JobSpec, lookup: &dyn ModelLookup, stop: StopFlag, events: &mut dyn FnMut(Event)) -> std::result::Result<JobReport, JobError> {
    // An interlaced source gets a Deinterlace step first when the pipeline has none (the plan
    // made for the export check does the same, so stage numbers match).
    let auto_spec;
    let spec = match probe(tools, &spec.source).ok().and_then(|i| i.video().ok().map(|v| v.is_interlaced_flagged())).and_then(|il| spec.pipeline.with_auto_deinterlace(il)) {
        Some(p) => {
            auto_spec = JobSpec { pipeline: p, ..spec.clone() };
            &auto_spec
        }
        None => spec,
    };
    let started_at = chrono::Utc::now().to_rfc3339();
    let started = Instant::now();
    events(Event::State { state: JobState::Preparing, detail: "Reading source".into() });
    if spec.schema_version > SCHEMA_VERSION {
        return Err(JobError::Invalid(format!("job was created by a newer ClipFX Enhance (schema {})", spec.schema_version)));
    }
    let (info, plan) = plan_job(tools, spec, lookup, None).map_err(JobError::Failed)?;
    if plan.has_errors() {
        let msgs: Vec<String> = plan.errors().map(|i| i.message.clone()).collect();
        return Err(JobError::Invalid(msgs.join(" ")));
    }
    events(Event::Plan { plan: Box::new(plan.clone()) });
    std::fs::create_dir_all(&spec.work_dir).map_err(|e| JobError::Failed(anyhow!("creating work folder: {e}")))?;
    let fingerprint = fsutil::quick_fingerprint(&spec.source).unwrap_or_default();
    let key = plan_key(spec, &fingerprint);
    let mut state = match RunState::load(&spec.work_dir) {
        Some(s) if s.plan_key == key && s.passes_done.len() == plan.passes.len() => s,
        _ => {
            // Different settings: discard stale work files.
            for i in 0..16 {
                let _ = std::fs::remove_dir_all(spec.work_dir.join(format!("pass{i}")));
            }
            RunState { plan_key: key, passes_done: vec![false; plan.passes.len()], analysis_done: vec![false; plan.passes.len()], ..Default::default() }
        }
    };
    state.save(&spec.work_dir).map_err(JobError::Failed)?;

    let total_units: f64 = plan.passes.iter().map(|p| pass_frames(&plan, p) as f64 * if p.stabilize.is_some() { 1.25 } else { 1.0 }).sum::<f64>() + 1.0;
    let mut ctx = Ctx {
        tools,
        spec,
        info,
        plan,
        stop,
        events,
        timings: Timings::default(),
        stage_stats: BTreeMap::new(),
        warnings: Vec::new(),
        started,
        total_units,
        done_units: 0.0,
        backend_label: String::new(),
        monitor: ResourceMonitor::default(),
    };
    for i in ctx.plan.issues.clone() {
        if i.severity == plan::Severity::Warning {
            ctx.warn(i.message);
        }
    }

    // Device
    let compute = select_compute(&spec.resources, &spec.engine).map_err(|e| JobError::Invalid(format!("the selected compute device is unavailable: {e}")))?;
    ctx.backend_label = compute.label.clone();
    if compute.kind == BackendKind::Cpu && ctx.plan.stages.iter().any(|s| !s.models.is_empty()) {
        ctx.log("Using CPU inference. AI processing on the CPU is slow; no usable GPU was selected or found.");
    } else if compute.gpu.is_some() {
        ctx.log(format!("Using {} for image models.", compute.label));
    }
    ctx.timings.prepare_s = started.elapsed().as_secs_f64();

    let mut frames_written = 0;
    let n_passes = ctx.plan.passes.len();
    for pi in 0..n_passes {
        let pass = ctx.plan.passes[pi].clone();
        let units = pass_frames(&ctx.plan, &pass) as f64 * if pass.stabilize.is_some() { 1.25 } else { 1.0 };
        if state.passes_done[pi] {
            ctx.done_units += units;
            continue;
        }
        let r = run_pass(&mut ctx, &pass, &mut state, &compute);
        match r {
            Ok(n) => {
                frames_written = n;
                state.passes_done[pi] = true;
                state.save(&spec.work_dir).map_err(JobError::Failed)?;
            }
            Err(e) => {
                let _ = state.save(&spec.work_dir);
                return Err(e);
            }
        }
    }

    // Assemble
    ctx.stop.check()?;
    ctx.state(JobState::Encoding, "Assembling video, audio and subtitles");
    let t = Instant::now();
    let final_pass = ctx.plan.passes.len() - 1;
    let seg_dir = spec.work_dir.join(format!("pass{final_pass}"));
    let segments: Vec<PathBuf> = read_segment_list(&seg_dir.join("segments.csv")).into_iter().map(|(n, _, _)| seg_dir.join("segments").join(n)).collect();
    if segments.is_empty() {
        return Err(JobError::Failed(anyhow!("no video was produced")));
    }
    let out_fps = ctx.plan.output.fps.as_f64();
    let durations: Vec<f64> = segments
        .iter()
        .map(|p| count_video_packets(tools, p).map(|n| n as f64 / out_fps).ok_or_else(|| JobError::Failed(anyhow!("could not read {}", p.display()))))
        .collect::<std::result::Result<_, _>>()?;
    let dest_dir = spec.destination.parent().ok_or_else(|| JobError::Invalid("invalid destination".into()))?.to_path_buf();
    fsutil::ensure_writable_dir(&dest_dir, true).map_err(JobError::Invalid)?;
    let tmp = fsutil::temp_output_path(&spec.destination, &spec.job_id);
    let _ = std::fs::remove_file(&tmp);
    let family = codec_profile(&spec.output.video.encoder).map(|c| c.family).unwrap_or("h264");
    let slow = ctx.plan.output.time_scale;
    let slow_audio = spec
        .pipeline
        .enabled()
        .find_map(|s| match &s.op {
            Operation::Interpolate(i) if i.slow_motion > 1.0001 => Some(i.slow_motion_audio),
            _ => None,
        })
        .unwrap_or_default();
    let video_offset = if spec.range.is_none() {
        let v = ctx.info.video.as_ref().map(|v| v.start_time).unwrap_or(0.0);
        let a = ctx.info.audio.iter().map(|a| a.start_time).fold(v, f64::min);
        let off = v - a;
        if off.abs() > 0.0005 && off < 10.0 {
            off
        } else {
            0.0
        }
    } else {
        0.0
    };
    let mux_spec = MuxSpec {
        video: segments,
        durations: Some(durations),
        video_family: family.to_string(),
        source: spec.source.clone(),
        range: spec.range.map(|r| (r.start, r.duration())),
        audio: spec.output.audio.clone(),
        audio_streams: None,
        slow_motion: slow,
        slow_motion_audio: slow_audio,
        subtitles: spec.output.subtitles,
        metadata: spec.output.metadata,
        chapters: spec.output.chapters,
        video_offset,
        duration: ctx.plan.output_duration,
        container: spec.output.container,
        output: tmp.clone(),
        work_dir: spec.work_dir.join("mux"),
    };
    let mux_report = mux::mux(tools, &mux_spec, &ctx.info).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        JobError::Failed(anyhow!("{e}"))
    })?;
    for w in &mux_report.warnings {
        ctx.warn(w.clone());
    }
    ctx.timings.assemble_s = t.elapsed().as_secs_f64();

    // Validate
    ctx.state(JobState::Validating, "Checking the output can be read and decoded");
    let t = Instant::now();
    let expected = Expected {
        width: ctx.plan.output.width,
        height: ctx.plan.output.height,
        fps: ctx.plan.output.fps,
        duration: ctx.plan.output_duration,
        audio_streams: mux_report.audio_streams,
    };
    let report = validate::validate(tools, &tmp, &expected).map_err(|e| JobError::Failed(anyhow!("{e}")))?;
    ctx.timings.validate_s = t.elapsed().as_secs_f64();
    if !report.ok {
        let _ = std::fs::remove_file(&tmp);
        return Err(JobError::Failed(anyhow!("Export validation failed; your source file was not changed. {}", report.problems.join("; "))));
    }

    // Finalise without overwriting.
    let mut dest = spec.destination.clone();
    if dest.exists() {
        let stem = dest.file_stem().and_then(|s| s.to_str()).unwrap_or("output").to_string();
        let ext = dest.extension().and_then(|s| s.to_str()).unwrap_or("mp4").to_string();
        dest = fsutil::unique_path(&dest_dir, &stem, &ext, &[&spec.source]);
        ctx.warn(format!("A file appeared at the destination while rendering; saved as {} instead.", dest.display()));
    }
    fsutil::finalize_no_overwrite(&tmp, &dest, Some(&spec.source)).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        JobError::Failed(anyhow!("{e}"))
    })?;

    ctx.timings.total_s = ctx.started.elapsed().as_secs_f64();
    let report = JobReport {
        job_id: spec.job_id.clone(),
        app_version: env!("CARGO_PKG_VERSION").to_string(),
        ffmpeg_version: tools.version.clone(),
        backend: compute.label.clone(),
        resources: spec.resources.clone(),
        source: spec.source.clone(),
        source_fingerprint: fingerprint,
        output: dest.clone(),
        range: spec.range,
        pipeline: spec.pipeline.clone(),
        output_settings: spec.output.clone(),
        models: spec.models.clone(),
        output_width: ctx.plan.output.width,
        output_height: ctx.plan.output.height,
        output_fps: ctx.plan.output.fps.to_ffmpeg(),
        output_duration: ctx.plan.output_duration,
        frames_written,
        started_at,
        finished_at: chrono::Utc::now().to_rfc3339(),
        timings: ctx.timings.clone(),
        warnings: ctx.warnings.clone(),
        stage_stats: ctx.stage_stats.clone(),
        stabilize_zoom_percent: state.zoom,
        validation: Some(report),
        mux: Some(mux_report),
    };
    if spec.write_sidecar {
        let side = PathBuf::from(format!("{}.clipfx.json", dest.display()));
        if !side.exists() {
            let _ = std::fs::write(&side, serde_json::to_vec_pretty(&report).unwrap_or_default());
        }
    }
    let _ = std::fs::remove_dir_all(&spec.work_dir);
    (ctx.events)(Event::Progress(Progress {
        pass: n_passes,
        passes: n_passes,
        phase: "done".into(),
        frames_done: frames_written,
        frames_total: frames_written,
        fraction: 1.0,
        fps: 0.0,
        eta: Some((0.0, 0.0)),
    }));
    (ctx.events)(Event::State { state: JobState::Completed, detail: dest.display().to_string() });
    Ok(report)
}

fn pass_frames(plan: &ExecutionPlan, p: &PassPlan) -> u64 {
    let dur = plan.input_duration * p.output.time_scale;
    (dur * p.output.fps.as_f64()).round().max(1.0) as u64
}

fn decode_spec_for(ctx: &Ctx, pass: &PassPlan, pass_dir: &Path, filters: Vec<String>, seek_frames: u64, in_fmt: &plan::Format) -> Result<DecodeSpec> {
    let v = ctx.info.video()?;
    let spec = ctx.spec;
    let seek_s = seek_frames as f64 / in_fmt.fps.as_f64() / 1.0;
    if pass.from_intermediate {
        let prev = pass.index - 1;
        let prev_dir = spec.work_dir.join(format!("pass{prev}"));
        let segs: Vec<PathBuf> = read_segment_list(&prev_dir.join("segments.csv")).into_iter().map(|(n, _, _)| prev_dir.join("segments").join(n)).collect();
        if segs.is_empty() {
            bail!("the intermediate from pass {prev} is missing");
        }
        let list = pass_dir.join("input.ffconcat");
        let prev_fps = ctx.plan.passes[prev].output.fps.as_f64();
        let durs: Vec<f64> =
            segs.iter().map(|p| count_video_packets(ctx.tools, p).map(|n| n as f64 / prev_fps).ok_or_else(|| anyhow!("could not read {}", p.display()))).collect::<Result<_>>()?;
        clearframe_media::mux::concat_list(&segs, Some(&durs), &list).map_err(|e| anyhow!("{e}"))?;
        let mut d = DecodeSpec::new(list, pass.input.width as usize, pass.input.height as usize);
        d.input_args = vec!["-f".into(), "concat".into(), "-safe".into(), "0".into()];
        d.filters = filters;
        d.matrix = YuvMatrix::Bt709;
        d.full_range = true;
        if seek_frames > 0 {
            d.start = Some((seek_frames as f64 - 0.5) / in_fmt.fps.as_f64());
        }
        d.width = in_fmt.width as usize;
        d.height = in_fmt.height as usize;
        d.cwd = Some(pass_dir.to_path_buf());
        return Ok(d);
    }
    let tags = ColorTags::from_stream(v);
    let mut d = DecodeSpec::new(spec.source.clone(), in_fmt.width as usize, in_fmt.height as usize);
    d.stream_index = Some(v.index);
    d.filters = filters;
    d.cfr = ctx.plan.cfr_conversion;
    d.matrix = tags.matrix;
    d.full_range = tags.full_range;
    d.cwd = Some(pass_dir.to_path_buf());
    let range_start = spec.range.map(|r| r.start).unwrap_or(0.0);
    let range_end = spec.range.map(|r| r.end);
    let start = if seek_frames > 0 { range_start + (seek_frames as f64 - 0.5) / ctx.plan.input.fps.as_f64() } else { range_start };
    let _ = seek_s;
    if start > 0.0 {
        d.start = Some(start);
    }
    if let Some(end) = range_end {
        d.duration = Some((end - start).max(0.0));
    }
    Ok(d)
}

fn run_pass(ctx: &mut Ctx, pass: &PassPlan, state: &mut RunState, compute: &Compute) -> std::result::Result<u64, JobError> {
    let spec = ctx.spec;
    let tools = ctx.tools;
    let pass_dir = spec.work_dir.join(format!("pass{}", pass.index));
    let seg_dir = pass_dir.join("segments");
    std::fs::create_dir_all(&seg_dir).map_err(|e| JobError::Failed(e.into()))?;
    let n_passes = ctx.plan.passes.len();

    // 1. Stabilisation analysis.
    let mut decode_filters = pass.decode_filters.clone();
    if let Some(stab) = &pass.stabilize {
        let trf = pass_dir.join("transforms.trf");
        if !(state.analysis_done[pass.index] && trf.is_file()) {
            ctx.state(JobState::Processing, "Analysing camera motion");
            let t = Instant::now();
            run_stabilize_detect(ctx, pass, &pass_dir, stab)?;
            ctx.timings.analyze_s += t.elapsed().as_secs_f64();
            state.analysis_done[pass.index] = true;
            // Measure auto-crop zoom.
            if stab.transform_template.contains("optzoom=1") {
                state.zoom = measure_zoom(ctx, pass, &pass_dir, stab).ok();
                if let Some(z) = state.zoom {
                    let s = 1.0 / (1.0 + z / 100.0);
                    let msg = format!(
                        "Stabilization auto crop zooms in {:.1}%: the visible area corresponds to about {}x{} of the original frame.",
                        z,
                        (pass.input.width as f64 * s).round(),
                        (pass.input.height as f64 * s).round()
                    );
                    ctx.log(msg);
                }
            }
            state.save(&spec.work_dir).map_err(JobError::Failed)?;
        }
        ctx.done_units += pass_frames(&ctx.plan, pass) as f64 * 0.25;
        for f in decode_filters.iter_mut() {
            *f = f.replace("{TRF}", "transforms.trf");
        }
    }

    // 2. Resume point.
    let list_file = pass_dir.join("segments.csv");
    // Merge lists left behind by a run that was killed before it could merge them.
    let mut parts: Vec<PathBuf> = std::fs::read_dir(&pass_dir)
        .map(|rd| {
            rd.flatten().map(|e| e.path()).filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("segments.part") && n.ends_with(".csv"))).collect()
        })
        .unwrap_or_default();
    parts.sort_by_key(|p| p.file_name().and_then(|n| n.to_str()).and_then(|n| n.trim_start_matches("segments.part").trim_end_matches(".csv").parse::<u64>().ok()).unwrap_or(0));
    for part in parts {
        if let Ok(extra) = std::fs::read_to_string(&part) {
            let mut cur = std::fs::read_to_string(&list_file).unwrap_or_default();
            cur.push_str(&extra);
            let _ = std::fs::write(&list_file, cur);
        }
        let _ = std::fs::remove_file(&part);
    }
    let mut done_segments = read_segment_list(&list_file);
    // Drop list entries whose files are missing or empty.
    done_segments.retain(|(n, _, _)| std::fs::metadata(seg_dir.join(n)).map(|m| m.len() > 0).unwrap_or(false));
    if !pass.seekable && !done_segments.is_empty() {
        ctx.log("This pass uses stabilization, which cannot resume mid-way; it restarts from the beginning of the pass (motion analysis is kept).");
        done_segments.clear();
    }
    // Rewrite the list with only verified segments and remove stray files.
    let keep: std::collections::HashSet<String> = done_segments.iter().map(|s| s.0.clone()).collect();
    if let Ok(rd) = std::fs::read_dir(&seg_dir) {
        for e in rd.flatten() {
            let n = e.file_name().to_string_lossy().to_string();
            if !keep.contains(&n) {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }
    let list_text: String = done_segments.iter().map(|(n, s, e)| format!("{n},{s:.6},{e:.6}\n")).collect();
    std::fs::write(&list_file, list_text).map_err(|e| JobError::Failed(e.into()))?;
    let out_fps = pass.output.fps;
    // Resume after the frames actually stored in completed segments (segment
    // timestamps include encoder delay, so frames are counted instead).
    let mut resume_out = 0u64;
    for (n, _, _) in &done_segments {
        match count_video_packets(tools, &seg_dir.join(n)) {
            Some(c) => resume_out += c,
            None => {
                ctx.log("A finished segment could not be read; restarting this pass.");
                done_segments.clear();
                resume_out = 0;
                break;
            }
        }
    }
    if done_segments.is_empty() {
        let _ = std::fs::write(&list_file, "");
        if let Ok(rd) = std::fs::read_dir(&seg_dir) {
            for e in rd.flatten() {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }
    // Corresponding input frame (with ~1 s of pre-roll for temporal filters).
    let in_fps = pass.input.fps.as_f64();
    let ratio_out_per_in = (out_fps.as_f64() * pass.output.time_scale) / (in_fps * pass.input.time_scale);
    let seek_in = if resume_out > 0 {
        let exact = (resume_out as f64 / ratio_out_per_in).floor() as i64;
        (exact - (in_fps.ceil() as i64).max(2)).max(0) as u64
    } else {
        0
    };
    if resume_out > 0 {
        ctx.emit(Event::Resumed { pass: pass.index, from_frame: resume_out });
        ctx.log(format!("Resuming pass {} at output frame {resume_out}.", pass.index + 1));
    }

    // 3. Processors.
    ctx.state(JobState::Preparing, "Loading models");
    let enabled: Vec<&Stage> = spec.pipeline.enabled().collect();
    let mut procs: Vec<Timed> = Vec::new();
    for seg in &pass.segments {
        match seg {
            Segment::Native { stage } => {
                let sp = &ctx.plan.stages[*stage];
                let op = &enabled[sp.index].op;
                let p = build_native(op, sp, &spec.models, compute, spec.engine.precision, &spec.resources).map_err(JobError::Failed)?;
                procs.push(Timed { inner: p, seconds: 0.0, stage_id: sp.stage_id.clone() });
            }
            Segment::Ffmpeg { filters, input, output } => {
                let p = FfmpegProc::new(tools, filters, input, output).map_err(JobError::Failed)?;
                procs.push(Timed { inner: Box::new(p), seconds: 0.0, stage_id: format!("ffmpeg-{}", procs.len()) });
            }
        }
    }

    // 4. Decoder (frames are in the pass input timeline, decoded after `decode_filters`).
    let dec_fmt = match pass.segments.first() {
        Some(Segment::Native { stage }) => ctx.plan.stages[*stage].input.clone(),
        Some(Segment::Ffmpeg { input, .. }) => input.clone(),
        None => pass.encode_input.clone(),
    };
    // Frame indices after decode filters are in the decode-output timeline.
    let dec_ratio = (dec_fmt.fps.as_f64() * dec_fmt.time_scale) / (in_fps * pass.input.time_scale);
    let dspec = decode_spec_for(ctx, pass, &pass_dir, decode_filters, seek_in, &dec_fmt).map_err(JobError::Failed)?;
    let mut reader = FrameReader::spawn(tools, &dspec).map_err(|e| JobError::Failed(anyhow!("{e}")))?;
    let first_index = (seek_in as f64 * dec_ratio).round() as u64;

    // 5. Encoder.
    let (video, color, sar) = if pass.is_final {
        let mut color = ctx.plan.output_color.clone();
        if spec.output.video.encoder == "ffv1" {
            color.full_range = color.full_range || spec.output.video.pix_fmt.starts_with("gbr");
        }
        let sar = ctx.info.video.as_ref().and_then(|v| v.sample_aspect_ratio.clone()).filter(|s| s != "1:1");
        (spec.output.video.clone(), color, sar)
    } else {
        let mut v = VideoEncodeSettings::intermediate();
        v.pix_fmt = "gbrp16le".into();
        (v, ColorTags { matrix: YuvMatrix::Bt709, full_range: true, primaries: None, transfer: None, mastering_display: None, content_light: None }, None)
    };
    // Resume bookkeeping in the encoder-input timeline (trailing filters may change the rate).
    let enc_ratio = (out_fps.as_f64() * pass.output.time_scale) / (pass.encode_input.fps.as_f64() * pass.encode_input.time_scale);
    let resume_enc = (resume_out as f64 / enc_ratio).round() as u64;
    let enc_filters = pass.encode_filters.clone();
    // Each run writes its own list; it is merged into the main list afterwards.
    let new_list = pass_dir.join(format!("segments.part{}.csv", done_segments.len()));
    let _ = std::fs::remove_file(&new_list);
    let enc = EncodeSpec {
        width: pass.encode_input.width as usize,
        height: pass.encode_input.height as usize,
        fps: pass.encode_input.fps,
        filters: enc_filters,
        video,
        color,
        sar,
        output: EncodeOutput::Segments {
            dir: seg_dir.clone(),
            seconds: spec.segment_seconds.max(1.0),
            start_number: done_segments.len() as u32,
            list_file: new_list.clone(),
            time_offset: 0.0,
        },
    };
    let writer = FrameWriter::spawn(tools, &enc).map_err(|e| JobError::Failed(anyhow!("{e}")))?;

    ctx.state(JobState::Processing, &format!("Processing (pass {} of {n_passes})", pass.index + 1));
    let total = pass_frames(&ctx.plan, pass);
    let units_before = ctx.done_units;

    // Reader thread.
    let (rtx, rrx) = bounded::<std::result::Result<RawFrame, String>>(3);
    let reader_handle = std::thread::spawn(move || loop {
        match reader.read() {
            Ok(Some(f)) => {
                if rtx.send(Ok(f)).is_err() {
                    reader.kill();
                    return;
                }
            }
            Ok(None) => return,
            Err(e) => {
                let _ = rtx.send(Err(e.to_string()));
                return;
            }
        }
    });
    // Writer thread.
    let (wtx, wrx) = bounded::<RawFrame>(3);
    let writer_handle = std::thread::spawn(move || -> std::result::Result<u64, String> {
        let mut w = writer;
        for f in wrx {
            if let Err(e) = w.write(&f) {
                w.kill();
                return Err(e.to_string());
            }
        }
        w.finish().map_err(|e| e.to_string())
    });

    let mut written: u64 = 0;
    let mut decoded: u64 = 0;
    let mut window: std::collections::VecDeque<(Instant, u64)> = std::collections::VecDeque::new();
    let mut last_emit = Instant::now() - std::time::Duration::from_secs(1);
    let expected_out = total.saturating_sub(resume_out);
    // With several stages each runs on its own thread, so a GPU stage keeps working while CPU stages
    // and FFmpeg filters handle the neighbouring frames. The last stage hands frames to the encoder.
    let shared_written = Arc::new(AtomicU64::new(0));
    let shared_encode_us = Arc::new(AtomicU64::new(0));
    let mut stages = if procs.len() > 1 {
        let (wtx, written, encode_us) = (wtx.clone(), shared_written.clone(), shared_encode_us.clone());
        let sink = move |f: Frame| -> std::result::Result<(), String> {
            if f.index >= resume_enc {
                let t = Instant::now();
                wtx.send(f.into_raw()).map_err(|_| "the encoder stopped unexpectedly".to_string())?;
                encode_us.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
                written.fetch_add(1, Ordering::Relaxed);
            }
            Ok(())
        };
        Some(StageThreads::spawn(std::mem::take(&mut procs), sink))
    } else {
        None
    };

    let result: std::result::Result<(), JobError> = (|| {
        loop {
            ctx.stop.check()?;
            let t = Instant::now();
            let item = rrx.recv();
            ctx.timings.decode_wait_s += t.elapsed().as_secs_f64();
            let frames: Vec<Frame> = match item {
                Ok(Ok(raw)) => {
                    let f = Frame::raw(first_index + decoded, raw);
                    decoded += 1;
                    vec![f]
                }
                Ok(Err(e)) => return Err(JobError::Failed(anyhow!("decoding failed: {e}"))),
                Err(_) => break,
            };
            let outs = match &stages {
                Some(st) => {
                    if !st.send(frames) {
                        // A stage stopped; its error is collected when the threads are joined.
                        break;
                    }
                    written = shared_written.load(Ordering::Relaxed);
                    Vec::new()
                }
                None => run_chain(&mut procs, 0, frames, false)?,
            };
            for f in outs {
                if f.index < resume_enc {
                    continue;
                }
                let t = Instant::now();
                wtx.send(f.into_raw()).map_err(|_| JobError::Failed(anyhow!("the encoder stopped unexpectedly")))?;
                ctx.timings.encode_wait_s += t.elapsed().as_secs_f64();
                written += 1;
            }
            // Resource sample and disk-space guard (throttled to every 2 s).
            let stage_name = match &stages {
                Some(st) => st.busiest(),
                None => procs.iter().max_by(|a, b| a.seconds.partial_cmp(&b.seconds).unwrap_or(std::cmp::Ordering::Equal)).map(|p| p.inner.name()).unwrap_or_default(),
            };
            let backend_label = ctx.backend_label.clone();
            if let Some(sample) = ctx.monitor.sample(&backend_label, &stage_name, Some(&spec.work_dir)) {
                let min = spec.resources.min_free_disk_mb;
                let dest_free = spec.destination.parent().and_then(fsutil::free_space).map(|b| b / 1_000_000);
                let low = [sample.free_disk_mb, dest_free].into_iter().flatten().min();
                ctx.emit(Event::Resources(sample));
                if let Some(free) = low {
                    if min > 0 && free < min {
                        return Err(JobError::Resources(format!(
                            "Paused: only {free} MB of disk space is left (minimum {min} MB). Free some space and resume; finished segments are kept."
                        )));
                    }
                }
            }
            // Progress at most ~4 times a second.
            if last_emit.elapsed().as_millis() >= 250 {
                last_emit = Instant::now();
                window.push_back((Instant::now(), written));
                while window.len() > 40 {
                    window.pop_front();
                }
                let fps = match (window.front(), window.back()) {
                    (Some(a), Some(b)) if b.0 > a.0 => (b.1 - a.1) as f64 / (b.0 - a.0).as_secs_f64(),
                    _ => 0.0,
                };
                let remaining = expected_out.saturating_sub((written as f64 * enc_ratio) as u64) as f64 / enc_ratio;
                let eta = if fps > 0.0 && written > 2 { Some((remaining / fps * 0.85, remaining / fps * 1.25 + 2.0)) } else { None };
                ctx.done_units = units_before + resume_out as f64 + written as f64 * enc_ratio;
                let fraction = (ctx.done_units / ctx.total_units).min(0.99);
                ctx.emit(Event::Progress(Progress {
                    pass: pass.index,
                    passes: n_passes,
                    phase: "processing".into(),
                    frames_done: resume_out + (written as f64 * enc_ratio) as u64,
                    frames_total: total,
                    fraction,
                    fps,
                    eta,
                }));
            }
        }
        if let Some(st) = stages.take() {
            // Let every stage finish in order, then collect the processors (for statistics) and the first error.
            let (back, err) = st.join(false);
            procs = back;
            ctx.timings.encode_wait_s += shared_encode_us.load(Ordering::Relaxed) as f64 / 1e6;
            written = shared_written.load(Ordering::Relaxed);
            return match err {
                Some(e) => Err(JobError::Failed(anyhow!("{e}"))),
                None => Ok(()),
            };
        }
        // Flush processors in order.
        let outs = run_chain(&mut procs, 0, Vec::new(), true)?;
        for f in outs {
            if f.index < resume_enc {
                continue;
            }
            wtx.send(f.into_raw()).map_err(|_| JobError::Failed(anyhow!("the encoder stopped unexpectedly")))?;
            written += 1;
        }
        Ok(())
    })();
    if let Some(st) = stages.take() {
        // Stopped or failed early: end the stage threads without flushing them.
        let (back, _) = st.join(true);
        procs = back;
    }
    drop(wtx);
    drop(rrx);
    let _ = reader_handle.join();
    let enc_result = writer_handle.join().unwrap_or_else(|_| Err("encoder thread panicked".into()));
    for p in &procs {
        *ctx.timings.stages_s.entry(p.stage_id.clone()).or_insert(0.0) += p.seconds;
        if let Some(s) = p.inner.stats() {
            if let Some(w) = s.get("warning").and_then(|w| w.as_str()) {
                if !ctx.warnings.iter().any(|x| x == w) {
                    ctx.warn(w.to_string());
                }
            }
            ctx.stage_stats.insert(p.stage_id.clone(), s);
        }
    }
    // Merge the new segment list into the main list (only fully written segments are listed).
    let merge = |list_file: &Path, new_list: &Path| {
        if let Ok(s) = std::fs::read_to_string(new_list) {
            let mut cur = std::fs::read_to_string(list_file).unwrap_or_default();
            cur.push_str(&s);
            let _ = std::fs::write(list_file, cur);
            let _ = std::fs::remove_file(new_list);
        }
    };
    if let Err(e) = result {
        merge(&list_file, &new_list);
        return Err(e);
    }
    if let Err(e) = enc_result {
        merge(&list_file, &new_list);
        return Err(JobError::Failed(anyhow!("encoding failed: {e}")));
    }
    merge(&list_file, &new_list);
    if written == 0 && resume_out == 0 {
        return Err(JobError::Failed(anyhow!("no frames were produced; check the selected range")));
    }
    ctx.done_units = units_before + total as f64;
    Ok(resume_out + (written as f64 * enc_ratio).round() as u64)
}

/// Number of video packets (frames) in a finished segment.
fn count_video_packets(tools: &FfmpegTools, p: &Path) -> Option<u64> {
    let mut c = tools.ffprobe_cmd();
    c.args(["-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0"]).arg(p);
    let (ok, out, _) = clearframe_media::tools::run_capture(&mut c, std::time::Duration::from_secs(120), 4096).ok()?;
    if !ok {
        return None;
    }
    out.trim().trim_end_matches(',').parse().ok()
}

/// The processors of a pass, each on its own thread, connected by short queues.
struct StageThreads {
    input: Option<crossbeam_channel::Sender<Frame>>,
    handles: Vec<std::thread::JoinHandle<(Timed, std::result::Result<(), String>)>>,
    names: Vec<String>,
    busy_us: Arc<Vec<AtomicU64>>,
    abort: Arc<AtomicBool>,
}

impl StageThreads {
    /// Start one thread per processor; the last one passes its frames to `sink`.
    fn spawn<S>(procs: Vec<Timed>, sink: S) -> Self
    where
        S: FnMut(Frame) -> std::result::Result<(), String> + Send + 'static,
    {
        let n = procs.len();
        let names = procs.iter().map(|p| p.inner.name()).collect();
        let busy_us: Arc<Vec<AtomicU64>> = Arc::new((0..n).map(|_| AtomicU64::new(0)).collect());
        let abort = Arc::new(AtomicBool::new(false));
        let (input, mut rx) = bounded::<Frame>(2);
        let mut sink = Some(sink);
        let mut handles = Vec::new();
        for (i, mut p) in procs.into_iter().enumerate() {
            let (next, next_rx) = if i + 1 < n {
                let (t, r) = bounded::<Frame>(2);
                (Some(t), Some(r))
            } else {
                (None, None)
            };
            let my_rx = match next_rx {
                Some(r) => std::mem::replace(&mut rx, r),
                None => rx.clone(),
            };
            let mut sink = if i + 1 == n { sink.take() } else { None };
            let (busy, abort) = (busy_us.clone(), abort.clone());
            handles.push(std::thread::spawn(move || {
                let mut emit = |outs: Vec<Frame>| -> std::result::Result<(), String> {
                    for f in outs {
                        match (&next, &mut sink) {
                            (Some(tx), _) => tx.send(f).map_err(|_| DOWNSTREAM_STOPPED.to_string())?,
                            (None, Some(s)) => s(f)?,
                            (None, None) => {}
                        }
                    }
                    Ok(())
                };
                let r = (|| {
                    for f in my_rx.iter() {
                        if abort.load(Ordering::Relaxed) {
                            return Ok(());
                        }
                        let mut out = Vec::new();
                        p.push(f, &mut out).map_err(|e| format!("{e:#}"))?;
                        busy[i].store((p.seconds * 1e6) as u64, Ordering::Relaxed);
                        emit(out)?;
                    }
                    if abort.load(Ordering::Relaxed) {
                        return Ok(());
                    }
                    let mut out = Vec::new();
                    p.finish(&mut out).map_err(|e| format!("{e:#}"))?;
                    emit(out)
                })();
                if r.is_err() {
                    abort.store(true, Ordering::Relaxed);
                }
                (p, r)
            }));
        }
        Self { input: Some(input), handles, names, busy_us, abort }
    }

    /// Queue frames for the first stage; false when the stages have stopped.
    fn send(&self, frames: Vec<Frame>) -> bool {
        let Some(tx) = &self.input else { return false };
        frames.into_iter().all(|f| !self.abort.load(Ordering::Relaxed) && tx.send(f).is_ok())
    }

    /// Name of the stage that has spent the most time so far.
    fn busiest(&self) -> String {
        let i = (0..self.names.len()).max_by_key(|&i| self.busy_us[i].load(Ordering::Relaxed)).unwrap_or(0);
        self.names.get(i).cloned().unwrap_or_default()
    }

    /// Close the input and wait for every stage. With `abort` the stages stop without flushing.
    /// Returns the processors in order and the first real error.
    fn join(mut self, abort: bool) -> (Vec<Timed>, Option<String>) {
        if abort {
            self.abort.store(true, Ordering::Relaxed);
        }
        drop(self.input.take());
        let mut procs = Vec::new();
        let mut first: Option<String> = None;
        let mut downstream = false;
        for h in self.handles {
            match h.join() {
                Ok((p, r)) => {
                    procs.push(p);
                    match r {
                        Err(e) if e == DOWNSTREAM_STOPPED => downstream = true,
                        Err(e) if first.is_none() => first = Some(e),
                        _ => {}
                    }
                }
                Err(_) => {
                    if first.is_none() {
                        first = Some("a processing thread panicked".into());
                    }
                }
            }
        }
        if first.is_none() && downstream && !abort {
            first = Some("processing stopped unexpectedly".into());
        }
        (procs, first)
    }
}

const DOWNSTREAM_STOPPED: &str = "a later stage stopped";

/// Push frames through processors starting at `from`; when `flush`, finish each
/// processor in order and push its remaining output downstream.
fn run_chain(procs: &mut [Timed], from: usize, frames: Vec<Frame>, flush: bool) -> std::result::Result<Vec<Frame>, JobError> {
    let mut cur = frames;
    for p in procs.iter_mut().skip(from) {
        let mut next = Vec::new();
        for f in cur {
            p.push(f, &mut next).map_err(JobError::Failed)?;
        }
        if flush {
            p.finish(&mut next).map_err(JobError::Failed)?;
        }
        cur = next;
    }
    Ok(cur)
}

fn run_stabilize_detect(ctx: &mut Ctx, pass: &PassPlan, pass_dir: &Path, stab: &plan::StabilizePass) -> std::result::Result<(), JobError> {
    let mut filters = stab.pre_filters.clone();
    filters.push("format=yuv444p".into());
    filters.push(stab.detect_filter.replace("{TRF}", "transforms.trf.partial"));
    let in_fmt = &pass.input;
    let d = decode_spec_for(ctx, pass, pass_dir, filters, 0, in_fmt).map_err(JobError::Failed)?;
    // Build an analysis command from the decode spec, writing to null.
    let mut c = d.command(ctx.tools);
    // Replace the rawvideo output: ffmpeg accepts a second output; simplest is a dedicated command.
    let args: Vec<String> = c.get_args().map(|a| a.to_string_lossy().to_string()).collect();
    let mut a2: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "-vf" {
            // keep filters but drop rgb conversion
            let vf = args[i + 1].clone();
            let kept: Vec<&str> = vf.split(',').filter(|f| !f.starts_with("scale=in_color_matrix") && *f != "format=rgb48le").collect();
            a2.push("-vf".into());
            a2.push(kept.join(","));
            i += 2;
            continue;
        }
        if args[i] == "-fps_mode" {
            break;
        }
        a2.push(args[i].clone());
        i += 1;
    }
    a2.extend(["-progress".into(), "pipe:1".into(), "-nostats".into(), "-f".into(), "null".into(), "-".into()]);
    c = std::process::Command::new(&ctx.tools.ffmpeg);
    clearframe_media::tools::hide_console(&mut c);
    c.current_dir(pass_dir).args(&a2).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = c.spawn().map_err(|e| JobError::Failed(anyhow!("starting motion analysis: {e}")))?;
    let stderr = child.stderr.take().unwrap();
    let err_h = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = std::io::Read::read_to_string(&mut BufReader::new(stderr), &mut s);
        s
    });
    let total = (ctx.plan.input_duration * pass.input.fps.as_f64()).max(1.0);
    let units_before = ctx.done_units;
    let pass_units = pass_frames(&ctx.plan, pass) as f64 * 0.25;
    let n_passes = ctx.plan.passes.len();
    for line in BufReader::new(child.stdout.take().unwrap()).lines() {
        if ctx.stop.check().is_err() {
            let _ = child.kill();
            let _ = child.wait();
            return Err(ctx.stop.check().unwrap_err());
        }
        let Ok(line) = line else { break };
        if let Some(v) = line.strip_prefix("frame=") {
            if let Ok(n) = v.trim().parse::<u64>() {
                ctx.done_units = units_before + pass_units * (n as f64 / total).min(1.0);
                let fraction = ctx.done_units / ctx.total_units;
                ctx.emit(Event::Progress(Progress {
                    pass: pass.index,
                    passes: n_passes,
                    phase: "analyzing".into(),
                    frames_done: n,
                    frames_total: total as u64,
                    fraction,
                    fps: 0.0,
                    eta: None,
                }));
            }
        }
    }
    let status = child.wait().map_err(|e| JobError::Failed(e.into()))?;
    let err = err_h.join().unwrap_or_default();
    if !status.success() {
        return Err(JobError::Failed(anyhow!("motion analysis failed: {}", err.trim())));
    }
    std::fs::rename(pass_dir.join("transforms.trf.partial"), pass_dir.join("transforms.trf")).map_err(|e| JobError::Failed(anyhow!("motion analysis produced no result: {e}")))?;
    ctx.done_units = units_before;
    Ok(())
}

fn measure_zoom(ctx: &Ctx, pass: &PassPlan, pass_dir: &Path, stab: &plan::StabilizePass) -> Result<f64> {
    let mut c = ctx.tools.ffmpeg_cmd();
    c.current_dir(pass_dir);
    c.args(["-v", "info"]);
    if pass.from_intermediate {
        c.args(["-f", "concat", "-safe", "0", "-i", "input.ffconcat"]);
    } else {
        if let Some(r) = ctx.spec.range {
            c.args(["-ss", &format!("{:.6}", r.start)]);
        }
        c.arg("-i").arg(&ctx.spec.source);
    }
    let mut f = stab.pre_filters.clone();
    f.push("format=yuv444p".into());
    f.push(stab.transform_template.replace("{TRF}", "transforms.trf"));
    c.args(["-map", "0:v:0", "-vf", &f.join(","), "-frames:v", "1", "-f", "null", "-"]);
    let (_, _, err) = clearframe_media::tools::run_capture(&mut c, std::time::Duration::from_secs(120), 1 << 20).map_err(|e| anyhow!("{e}"))?;
    for l in err.lines() {
        if let Some(rest) = l.split("Final zoom:").nth(1) {
            if let Ok(z) = rest.trim().parse::<f64>() {
                return Ok(z);
            }
        }
    }
    bail!("zoom not reported")
}

/// Mark the plan stage exec helper used by the UI.
pub fn stage_is_native(e: &StageExec) -> bool {
    matches!(e, StageExec::Native { .. })
}

#[cfg(test)]
mod stage_thread_tests {
    use super::*;
    use crate::processors::Processor;
    use clearframe_inference::RgbImage;

    /// Adds `add` to the first pixel; with `hold`, keeps one frame back until the next (like a temporal stage).
    struct Step {
        add: f32,
        hold: bool,
        held: Option<Frame>,
        fail_at: Option<u64>,
    }

    impl Processor for Step {
        fn name(&self) -> String {
            format!("step {}", self.add)
        }
        fn push(&mut self, f: Frame, out: &mut Vec<Frame>) -> Result<()> {
            if Some(f.index) == self.fail_at {
                bail!("failed on frame {}", f.index);
            }
            let (idx, mut img) = f.into_img();
            img.data[0] += self.add;
            let f = Frame::img(idx, img);
            if self.hold {
                out.extend(self.held.replace(f));
            } else {
                out.push(f);
            }
            Ok(())
        }
        fn finish(&mut self, out: &mut Vec<Frame>) -> Result<()> {
            out.extend(self.held.take());
            Ok(())
        }
    }

    fn chain(fail_at: Option<u64>) -> Vec<Timed> {
        [(1.0, false, None), (10.0, true, fail_at), (100.0, false, None)]
            .into_iter()
            .map(|(add, hold, fail_at)| Timed { inner: Box::new(Step { add, hold, held: None, fail_at }), seconds: 0.0, stage_id: format!("s{add}") })
            .collect()
    }

    fn frames(n: u64) -> Vec<Frame> {
        (0..n).map(|i| Frame::img(i, RgbImage::new(2, 2))).collect()
    }

    #[test]
    fn stages_on_threads_keep_order_and_flush() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let st = StageThreads::spawn(chain(None), move |f: Frame| tx.send(f).map_err(|e| e.to_string()));
        assert!(st.send(frames(20)));
        let (procs, err) = st.join(false);
        assert!(err.is_none());
        assert_eq!(procs.iter().map(|p| p.stage_id.as_str()).collect::<Vec<_>>(), ["s1", "s10", "s100"]);
        let out: Vec<Frame> = rx.try_iter().collect();
        // Every frame went through every stage, in order, including the one the middle stage held back.
        assert_eq!(out.iter().map(|f| f.index).collect::<Vec<_>>(), (0..20).collect::<Vec<_>>());
        assert!(out.into_iter().all(|f| f.into_img().1.data[0] == 111.0));
    }

    #[test]
    fn a_failing_stage_stops_the_pipeline_with_its_error() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let st = StageThreads::spawn(chain(Some(5)), move |f: Frame| tx.send(f).map_err(|e| e.to_string()));
        let mut sent = 0;
        for f in frames(1000) {
            if !st.send(vec![f]) {
                break;
            }
            sent += 1;
        }
        assert!(sent < 1000, "the input should stop accepting frames");
        let (procs, err) = st.join(false);
        assert_eq!(procs.len(), 3);
        assert_eq!(err.as_deref(), Some("failed on frame 5"));
        // Nothing past the failure was flushed.
        assert!(rx.try_iter().count() <= 5);
    }

    #[test]
    fn abort_skips_flushing() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let st = StageThreads::spawn(chain(None), move |f: Frame| tx.send(f).map_err(|e| e.to_string()));
        assert!(st.send(frames(3)));
        let (_, err) = st.join(true);
        assert!(err.is_none());
        // The held-back last frame is not flushed when stopping.
        assert!(rx.try_iter().count() < 3);
    }
}
