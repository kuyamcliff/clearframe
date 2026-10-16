//! Frame processors that make up a pass.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clearframe_inference::face::{FaceParams, FaceRestorer};
use clearframe_inference::model::{FaceModels, ImageModel, ImageParams, InterpModel, Precision};
use clearframe_inference::{Device, RgbImage, Weights};
use clearframe_media::pipe::{FilterProcess, RawFrame};
use clearframe_media::FfmpegTools;

use crate::grain::apply_grain;
use crate::interp::{BlendSynth, InterpStats, Retimer, Synthesizer};
use crate::plan::{Format, StagePlan};
use crate::spec::*;

/// Pixel data in whichever representation the last stage produced.
pub enum FrameData {
    Raw(RawFrame),
    Img(RgbImage),
}

/// A frame with its absolute index in the current timeline.
pub struct Frame {
    pub index: u64,
    pub data: FrameData,
}

impl Frame {
    pub fn raw(index: u64, raw: RawFrame) -> Self {
        Self { index, data: FrameData::Raw(raw) }
    }
    pub fn img(index: u64, img: RgbImage) -> Self {
        Self { index, data: FrameData::Img(img) }
    }
    pub fn into_img(self) -> (u64, RgbImage) {
        match self.data {
            FrameData::Img(i) => (self.index, i),
            FrameData::Raw(r) => (self.index, RgbImage::from_rgb48le(r.width, r.height, &r.data)),
        }
    }
    pub fn into_raw(self) -> RawFrame {
        match self.data {
            FrameData::Raw(r) => r,
            FrameData::Img(i) => {
                let mut data = Vec::new();
                i.write_rgb48le(&mut data);
                RawFrame { width: i.width, height: i.height, data }
            }
        }
    }
    pub fn dims(&self) -> (usize, usize) {
        match &self.data {
            FrameData::Raw(r) => (r.width, r.height),
            FrameData::Img(i) => (i.width, i.height),
        }
    }
}

pub trait Processor: Send {
    fn name(&self) -> String;
    fn push(&mut self, f: Frame, out: &mut Vec<Frame>) -> Result<()>;
    fn finish(&mut self, out: &mut Vec<Frame>) -> Result<()>;
    /// Optional statistics for the job report.
    fn stats(&self) -> Option<serde_json::Value> {
        None
    }
}

const MIN_TILE: usize = 64;

/// Whether an inference error message indicates memory exhaustion.
pub fn is_oom(msg: &str) -> bool {
    let m = msg.to_ascii_lowercase();
    m.contains("out of memory")
        || m.contains("cuda_error_out_of_memory")
        || m.contains("failed to allocate")
        || m.contains("allocation failed")
        || m.contains("insufficient memory")
        || m.contains("use a smaller tile")
}

/// Whether an error came from the GPU engine (driver reset, hang, validation), so the work can
/// be retried on a fresh GPU context or on the CPU.
pub fn is_gpu_failure(msg: &str) -> bool {
    msg.contains("GPU")
}

/// Image model stage (enhance / denoise / deblur) with optional strength blend.
pub struct ImageProc {
    label: String,
    model: ImageModel,
    params: ImageParams,
    /// 0..1: amount of model output blended over the (resized) input.
    mix: f32,
    oom_fallback: bool,
    oom_retries: u32,
    /// Video-memory budget, for reattaching to a reopened GPU.
    vram_budget_mb: u64,
    /// The GPU was reopened once after a failure; a second failure moves the stage to the CPU.
    gpu_reopened: bool,
    /// Frames worked on at once. On a GPU two keep it busy: one frame's forward pass runs while the
    /// other's tiles are prepared, read back and blended on the CPU.
    in_flight: usize,
    pending: Vec<Frame>,
}

impl ImageProc {
    fn new(label: String, model: ImageModel, params: ImageParams, mix: f32, res: &JobResources, sp: &StagePlan) -> Self {
        let in_flight = frames_in_flight(model.gpu_name().is_some(), sp, res);
        Self {
            label,
            model,
            params,
            mix,
            oom_fallback: res.oom_tile_fallback,
            oom_retries: 0,
            vram_budget_mb: res.vram_budget_mb,
            gpu_reopened: false,
            in_flight,
            pending: Vec::new(),
        }
    }

    /// Run the model on one frame and blend by `mix`.
    fn render(&self, img: &RgbImage) -> Result<RgbImage> {
        let res = self.model.process(img, &self.params)?;
        Ok(if self.mix < 0.999 {
            let base = if (res.width, res.height) != (img.width, img.height) { img.resize_bilinear(res.width, res.height) } else { img.clone() };
            base.blend(&res, self.mix.clamp(0.0, 1.0))
        } else {
            res
        })
    }

    /// One frame at a time, retrying with smaller tiles when the device runs out of memory.
    fn render_retrying(&mut self, img: &RgbImage) -> Result<RgbImage> {
        loop {
            match self.render(img) {
                Ok(r) => return Ok(r),
                Err(e) if self.oom_fallback && is_oom(&e.to_string()) && self.model.tile > MIN_TILE => {
                    // Device ran out of memory: retry this frame with smaller tiles.
                    let old = if self.model.tile == 0 { img.width.max(img.height) } else { self.model.tile };
                    self.model.tile = (old / 2).max(MIN_TILE);
                    self.model.tile_pad = self.model.tile_pad.min(self.model.tile / 4).max(8);
                    log::warn!("{}: out of memory, retrying with {}px tiles", self.label, self.model.tile);
                    self.oom_retries += 1;
                }
                Err(e) if self.model.gpu_name().is_some() && is_gpu_failure(&e.to_string()) => self.recover_gpu(&e.to_string()),
                Err(e) => return Err(anyhow!("{}: {e}", self.label)),
            }
        }
    }

    /// After a GPU failure: reopen the GPU once (a driver reset leaves the old context unusable)
    /// and retry; if it fails again, finish the job on the CPU rather than failing it.
    fn recover_gpu(&mut self, err: &str) {
        self.in_flight = 1;
        #[cfg(feature = "gpu")]
        if !self.gpu_reopened {
            self.gpu_reopened = true;
            if let Some(index) = self.model.gpu_index() {
                log::warn!("{}: {err}; reopening the GPU and retrying", self.label);
                let tile = self.model.tile;
                match clearframe_inference::gpu::context(index).and_then(|g| self.model.use_gpu_with_budget(g, self.vram_budget_mb).map_err(|e| e.to_string())) {
                    Ok(()) => {
                        // Smaller tiles than before: less work per piece, less memory.
                        self.model.tile = self.model.tile.min((tile / 2).max(256));
                        return;
                    }
                    Err(e) => log::warn!("{}: could not reopen the GPU ({e})", self.label),
                }
            }
        }
        log::warn!("{}: {err}; continuing on the CPU", self.label);
        self.model.use_cpu();
    }

    fn run_pending(&mut self, out: &mut Vec<Frame>) -> Result<()> {
        let frames: Vec<(u64, RgbImage)> = std::mem::take(&mut self.pending).into_iter().map(Frame::into_img).collect();
        if frames.len() > 1 {
            let this = &*self;
            let results: Vec<Result<RgbImage>> = std::thread::scope(|s| {
                let hs: Vec<_> = frames.iter().map(|(_, img)| s.spawn(move || this.render(img))).collect();
                hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(anyhow!("a processing thread panicked")))).collect()
            });
            if results.iter().all(|r| r.is_ok()) {
                for ((idx, _), r) in frames.into_iter().zip(results) {
                    out.push(Frame::img(idx, r?));
                }
                return Ok(());
            }
            match results.into_iter().find_map(|r| r.err()) {
                // Two frames' working memory did not fit: go on one at a time (the frames are redone in order).
                Some(e) if self.oom_fallback && is_oom(&e.to_string()) => {
                    log::warn!("{}: out of memory with {} frames in flight; continuing one at a time", self.label, self.in_flight);
                    self.in_flight = 1;
                    self.oom_retries += 1;
                }
                // A GPU failure: redo the frames one at a time, which recovers or moves to the CPU.
                Some(e) if is_gpu_failure(&e.to_string()) => {
                    log::warn!("{}: {e}; continuing one frame at a time", self.label);
                    self.in_flight = 1;
                }
                Some(e) => return Err(anyhow!("{}: {e}", self.label)),
                None => unreachable!(),
            }
        }
        for (idx, img) in frames {
            let r = self.render_retrying(&img)?;
            out.push(Frame::img(idx, r));
        }
        Ok(())
    }
}

/// How many frames an image stage works on at once: two on a GPU when both frames' buffers fit the
/// memory budget comfortably, otherwise one.
fn frames_in_flight(on_gpu: bool, sp: &StagePlan, res: &JobResources) -> usize {
    if !on_gpu {
        // On the CPU one frame already uses every core.
        return 1;
    }
    let out = sp.native_output.as_ref().unwrap_or(&sp.output);
    // Input, output, blend base and tile copies, as f32 RGB.
    let per_frame_mb = ((sp.input.width as u64 * sp.input.height as u64 + 3 * out.width as u64 * out.height as u64) * 12) >> 20;
    if res.memory_budget_mb == 0 || 2 * per_frame_mb <= res.memory_budget_mb / 4 {
        2
    } else {
        1
    }
}

impl Processor for ImageProc {
    fn name(&self) -> String {
        self.label.clone()
    }
    fn push(&mut self, f: Frame, out: &mut Vec<Frame>) -> Result<()> {
        self.pending.push(f);
        if self.pending.len() >= self.in_flight {
            self.run_pending(out)?;
        }
        Ok(())
    }
    fn finish(&mut self, out: &mut Vec<Frame>) -> Result<()> {
        self.run_pending(out)
    }
    fn stats(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({ "tile": self.model.tile, "oom_retries": self.oom_retries, "frames_in_flight": self.in_flight }))
    }
}

pub struct FaceProc {
    restorer: FaceRestorer,
    params: FaceParams,
    frames: u64,
    faces: u64,
}

impl Processor for FaceProc {
    fn name(&self) -> String {
        "Face recovery".into()
    }
    fn push(&mut self, f: Frame, out: &mut Vec<Frame>) -> Result<()> {
        let (idx, mut img) = f.into_img();
        let s = self.restorer.process(&mut img, &self.params).map_err(|e| anyhow!("face recovery: {e}"))?;
        self.frames += 1;
        self.faces += s.faces_restored as u64;
        out.push(Frame::img(idx, img));
        Ok(())
    }
    fn finish(&mut self, _out: &mut Vec<Frame>) -> Result<()> {
        Ok(())
    }
    fn stats(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({ "frames": self.frames, "faces_restored": self.faces }))
    }
}

pub struct GrainProc {
    settings: GrainSettings,
}

impl Processor for GrainProc {
    fn name(&self) -> String {
        "Add grain".into()
    }
    fn push(&mut self, f: Frame, out: &mut Vec<Frame>) -> Result<()> {
        let (idx, mut img) = f.into_img();
        apply_grain(&mut img, &self.settings, idx);
        out.push(Frame::img(idx, img));
        Ok(())
    }
    fn finish(&mut self, _out: &mut Vec<Frame>) -> Result<()> {
        Ok(())
    }
}

struct RifeSynth {
    model: InterpModel,
    scale: f64,
}

impl Synthesizer for RifeSynth {
    fn synthesize(&mut self, a: &RgbImage, b: &RgbImage, ts: &[f32]) -> Result<Vec<RgbImage>> {
        self.model.interpolate_many(a, b, ts, self.scale).map_err(|e| anyhow!("RIFE: {e}"))
    }
}

pub struct InterpProc {
    retimer: Retimer,
    synth: Box<dyn Synthesizer>,
}

impl Processor for InterpProc {
    fn name(&self) -> String {
        "Frame interpolation".into()
    }
    fn push(&mut self, f: Frame, out: &mut Vec<Frame>) -> Result<()> {
        let (idx, img) = f.into_img();
        for (j, im) in self.retimer.push(idx, img, self.synth.as_mut())? {
            out.push(Frame::img(j, im));
        }
        Ok(())
    }
    fn finish(&mut self, out: &mut Vec<Frame>) -> Result<()> {
        for (j, im) in self.retimer.finish() {
            out.push(Frame::img(j, im));
        }
        Ok(())
    }
    fn stats(&self) -> Option<serde_json::Value> {
        let s: &InterpStats = &self.retimer.stats;
        serde_json::to_value(s).ok()
    }
}

/// FFmpeg filter process between native stages.
pub struct FfmpegProc {
    label: String,
    proc_: FilterProcess,
    /// Output frames per input frame.
    ratio: f64,
    base: Option<u64>,
    count: u64,
    raw_out: Vec<RawFrame>,
}

impl FfmpegProc {
    pub fn new(tools: &FfmpegTools, filters: &[String], input: &Format, output: &Format) -> Result<Self> {
        let p = FilterProcess::spawn(tools, input.width as usize, input.height as usize, input.fps, filters, output.width as usize, output.height as usize)
            .context("starting FFmpeg filter process")?;
        let ratio = (output.fps.as_f64() * output.time_scale) / (input.fps.as_f64() * input.time_scale);
        Ok(Self { label: format!("FFmpeg: {}", filters.join(",")), proc_: p, ratio, base: None, count: 0, raw_out: Vec::new() })
    }

    fn emit(&mut self, out: &mut Vec<Frame>) {
        let base = self.base.unwrap_or(0);
        for r in self.raw_out.drain(..) {
            out.push(Frame::raw(base + self.count, r));
            self.count += 1;
        }
    }
}

impl Processor for FfmpegProc {
    fn name(&self) -> String {
        self.label.clone()
    }
    fn push(&mut self, f: Frame, out: &mut Vec<Frame>) -> Result<()> {
        if self.base.is_none() {
            self.base = Some((f.index as f64 * self.ratio).round() as u64);
        }
        let raw = f.into_raw();
        let mut tmp = Vec::new();
        self.proc_.push(raw, &mut tmp)?;
        self.raw_out.extend(tmp);
        self.emit(out);
        Ok(())
    }
    fn finish(&mut self, out: &mut Vec<Frame>) -> Result<()> {
        let mut tmp = Vec::new();
        self.proc_.finish(&mut tmp)?;
        self.raw_out.extend(tmp);
        self.emit(out);
        Ok(())
    }
}

/// Wraps a processor and accumulates the time spent in it.
pub struct Timed {
    pub inner: Box<dyn Processor>,
    pub seconds: f64,
    pub stage_id: String,
}

impl Timed {
    pub fn push(&mut self, f: Frame, out: &mut Vec<Frame>) -> Result<()> {
        let t = Instant::now();
        let r = self.inner.push(f, out);
        self.seconds += t.elapsed().as_secs_f64();
        r
    }
    pub fn finish(&mut self, out: &mut Vec<Frame>) -> Result<()> {
        let t = Instant::now();
        let r = self.inner.finish(out);
        self.seconds += t.elapsed().as_secs_f64();
        r
    }
}

fn load_weights(m: &ResolvedModel, role: &str, device: &Device) -> Result<Weights> {
    let p = m.files.get(role).ok_or_else(|| anyhow!("model {} has no `{role}` weights", m.id))?;
    load_weights_path(p, device).with_context(|| format!("loading {} ({role})", m.id))
}

fn load_weights_path(p: &Path, device: &Device) -> Result<Weights> {
    Weights::from_safetensors(p, device).map_err(|e| anyhow!("{}: {e}", p.display()))
}

fn model<'a>(models: &'a BTreeMap<String, ResolvedModel>, id: &str) -> Result<&'a ResolvedModel> {
    models.get(id).ok_or_else(|| anyhow!("model {id} is not installed"))
}

/// Default tile when the user chose automatic, scaled to the device and
/// limited so one tile's working memory fits the job's memory budget.
fn auto_tile(arch_default: usize, device: &Device, res: &JobResources, mb_per_mp: f64) -> usize {
    let mut t = match device {
        Device::Cpu => arch_default,
        _ => arch_default.max(512),
    };
    let budget = match device {
        Device::Cpu => res.memory_budget_mb,
        _ if res.vram_budget_mb > 0 => res.vram_budget_mb,
        _ => res.memory_budget_mb,
    };
    if budget > 0 && mb_per_mp > 0.0 {
        // Keep half the budget for frame buffers, decoder and encoder.
        let mp = (budget as f64 * 0.5 / mb_per_mp).max(0.004);
        let side = (mp * 1e6).sqrt() as usize;
        if t == 0 || side < t {
            t = side.max(MIN_TILE);
        }
    }
    t
}

/// Tile size limited by the memory budget (for models whose default is whole-frame).
pub fn budget_tile(res: &JobResources, mb_per_mp: f64) -> usize {
    auto_tile(0, &Device::Cpu, res, mb_per_mp)
}

/// Move an image model onto the job's GPU when one was selected. Returns whether it runs there;
/// on failure the model stays on the CPU and the job continues.
fn attach_gpu(im: &mut ImageModel, compute: &crate::runner::Compute, id: &str, res: &JobResources) -> bool {
    #[cfg(feature = "gpu")]
    if let Some(g) = &compute.gpu {
        match im.use_gpu_with_budget(g.clone(), res.vram_budget_mb) {
            Ok(()) => {
                // Each tile's input and output (and the read-back copy) still pass through
                // system memory, so half the RAM budget bounds the tile as well.
                if res.memory_budget_mb > 0 {
                    let s = im.scale().max(1) as f64;
                    let bytes_per_px = 2.0 * 12.0 * (1.0 + s * s);
                    let side = ((res.memory_budget_mb as f64 * 0.5 * 1048576.0 / bytes_per_px).sqrt() as usize).max(MIN_TILE);
                    im.tile = im.tile.min(side);
                }
                log::info!("{id}: running on {} with {}px tiles", g.info.name, im.tile);
                return true;
            }
            Err(e) => log::warn!("{id}: GPU unavailable ({e}); using the CPU"),
        }
    }
    let _ = (im, compute, id, res);
    false
}

/// Build the native processor for a stage.
pub fn build_native(
    op: &Operation,
    sp: &StagePlan,
    models: &BTreeMap<String, ResolvedModel>,
    compute: &crate::runner::Compute,
    precision: Precision,
    res: &JobResources,
) -> Result<Box<dyn Processor>> {
    let device = &compute.device;
    Ok(match op {
        Operation::Enhance(e) => {
            let m = model(models, &e.model_id)?;
            let mut w = load_weights(m, "weights", device)?;
            if let (Some(dn), Some(_)) = (e.denoise, m.files.get("wdn")) {
                // Real-ESRGAN DNI: x4v3 * dn + wdn * (1 - dn)
                let wdn = load_weights(m, "wdn", device)?;
                w = w.interpolate(&wdn, dn as f64).map_err(|e| anyhow!("{e}"))?;
            }
            let mut im = ImageModel::load(&w, precision).map_err(|e| anyhow!("{}: {e}", m.id))?;
            let on_gpu = attach_gpu(&mut im, compute, &m.id, res);
            if e.tile_size > 0 {
                im.tile = e.tile_size as usize;
            } else if !on_gpu {
                im.tile = auto_tile(im.tile, device, res, m.memory_per_megapixel_mb);
            }
            if e.tile_pad > 0 {
                im.tile_pad = e.tile_pad as usize;
            }
            let scale = im.scale() as u32;
            let expect = sp.native_output.as_ref().map(|f| f.width).unwrap_or(sp.output.width);
            if sp.input.width * scale != expect {
                anyhow::bail!("model {} scale x{scale} does not match the plan", m.id);
            }
            Box::new(ImageProc::new(format!("Enhance ({})", m.id), im, ImageParams::default(), e.strength, res, sp))
        }
        Operation::Denoise(d) => {
            let id = op.required_models().into_iter().next().ok_or_else(|| anyhow!("denoise engine has no model"))?;
            let m = model(models, &id)?;
            let w = load_weights(m, "weights", device)?;
            let _ = d.engine;
            let mut im = ImageModel::load(&w, precision).map_err(|e| anyhow!("{}: {e}", m.id))?;
            if !attach_gpu(&mut im, compute, &m.id, res) {
                im.tile = auto_tile(im.tile, device, res, m.memory_per_megapixel_mb);
            }
            Box::new(ImageProc::new(format!("Denoise ({})", m.id), im, ImageParams { noise_sigma: d.strength / 255.0 }, d.mix, res, sp))
        }
        Operation::Deblur(d) => {
            let m = model(models, &d.model_id)?;
            let w = load_weights(m, "weights", device)?;
            let mut im = ImageModel::load(&w, precision).map_err(|e| anyhow!("{}: {e}", m.id))?;
            if !attach_gpu(&mut im, compute, &m.id, res) {
                im.tile = auto_tile(im.tile, device, res, m.memory_per_megapixel_mb).min(512);
            }
            Box::new(ImageProc::new(format!("Deblur ({})", m.id), im, ImageParams::default(), d.strength, res, sp))
        }
        Operation::FaceRestore(f) => {
            let m = model(models, &f.model_id)?;
            let det = load_weights(m, "detector", device)?;
            let res = load_weights(m, "restorer", device)?;
            #[allow(unused_mut)]
            let mut fm = FaceModels::load(&det, &res).map_err(|e| anyhow!("{}: {e}", m.id))?;
            #[cfg(feature = "gpu")]
            if let Some(g) = &compute.gpu {
                match fm.use_gpu(g.clone()) {
                    Ok(()) => log::info!("{}: face restoration on {}", m.id, g.info.name),
                    Err(e) => log::warn!("{}: GPU unavailable ({e}); using the CPU", m.id),
                }
            }
            Box::new(FaceProc { restorer: FaceRestorer::new(fm), params: f.params.clone(), frames: 0, faces: 0 })
        }
        Operation::Interpolate(i) => {
            let ratio = (sp.output.fps.as_f64() * sp.output.time_scale) / (sp.input.fps.as_f64() * sp.input.time_scale);
            let synth: Box<dyn Synthesizer> = match i.engine {
                InterpEngine::Rife => {
                    let m = model(models, &i.model_id)?;
                    let w = load_weights(m, "weights", device)?;
                    #[allow(unused_mut)]
                    let mut im = InterpModel::load(&w, precision).map_err(|e| anyhow!("{}: {e}", m.id))?;
                    #[cfg(feature = "gpu")]
                    if let Some(g) = &compute.gpu {
                        match im.use_gpu(g.clone()) {
                            Ok(()) => log::info!("{}: frame interpolation on {}", m.id, g.info.name),
                            Err(e) => log::warn!("{}: GPU unavailable ({e}); using the CPU", m.id),
                        }
                    }
                    Box::new(RifeSynth { model: im, scale: i.rife_scale })
                }
                _ => Box::new(BlendSynth),
            };
            Box::new(InterpProc { retimer: Retimer::new(ratio, if i.scene_detection { Some(i.scene_threshold) } else { None }, i.duplicate_threshold), synth })
        }
        Operation::Grain(g) => Box::new(GrainProc { settings: g.clone() }),
        other => anyhow::bail!("{} is not a native stage", other.label()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_conversions_roundtrip() {
        let img = RgbImage::from_data(2, 1, vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.1]);
        let raw = Frame::img(3, img.clone()).into_raw();
        assert_eq!(raw.data.len(), 12);
        let (i, back) = Frame::raw(3, raw).into_img();
        assert_eq!(i, 3);
        for (a, b) in img.data.iter().zip(&back.data) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn tiles_shrink_with_memory_budget() {
        let mut r = JobResources::default();
        assert_eq!(auto_tile(512, &Device::Cpu, &r, 1000.0), 512);
        r.memory_budget_mb = 200; // 0.1 MP per tile at 1000 MB/MP
        let t = auto_tile(512, &Device::Cpu, &r, 1000.0);
        assert!((300..512).contains(&t), "{t}");
        r.memory_budget_mb = 1;
        assert_eq!(auto_tile(512, &Device::Cpu, &r, 1000.0), MIN_TILE);
        assert!(is_oom("CUDA_ERROR_OUT_OF_MEMORY"));
        assert!(!is_oom("shape mismatch"));
    }

    #[test]
    fn grain_processor_keeps_index() {
        let mut p = GrainProc { settings: GrainSettings::default() };
        let mut out = Vec::new();
        p.push(Frame::img(9, RgbImage::from_data(4, 4, vec![0.5; 48])), &mut out).unwrap();
        assert_eq!(out[0].index, 9);
    }
}
