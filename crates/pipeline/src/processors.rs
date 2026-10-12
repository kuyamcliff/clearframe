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

/// Image model stage (enhance / denoise / deblur) with optional strength blend.
pub struct ImageProc {
    label: String,
    model: ImageModel,
    params: ImageParams,
    /// 0..1: amount of model output blended over the (resized) input.
    mix: f32,
}

impl Processor for ImageProc {
    fn name(&self) -> String {
        self.label.clone()
    }
    fn push(&mut self, f: Frame, out: &mut Vec<Frame>) -> Result<()> {
        let (idx, img) = f.into_img();
        let res = self.model.process(&img, &self.params).map_err(|e| anyhow!("{}: {e}", self.label))?;
        let res = if self.mix < 0.999 {
            let base = if (res.width, res.height) != (img.width, img.height) { img.resize_bilinear(res.width, res.height) } else { img };
            base.blend(&res, self.mix.clamp(0.0, 1.0))
        } else {
            res
        };
        out.push(Frame::img(idx, res));
        Ok(())
    }
    fn finish(&mut self, _out: &mut Vec<Frame>) -> Result<()> {
        Ok(())
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
        let p = FilterProcess::spawn(
            tools,
            input.width as usize,
            input.height as usize,
            input.fps,
            filters,
            output.width as usize,
            output.height as usize,
        )
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

/// Default tile when the user chose automatic, scaled to the device.
fn auto_tile(arch_default: usize, device: &Device) -> usize {
    match device {
        Device::Cpu => arch_default,
        _ => arch_default.max(512),
    }
}

/// Build the native processor for a stage.
pub fn build_native(op: &Operation, sp: &StagePlan, models: &BTreeMap<String, ResolvedModel>, device: &Device, precision: Precision) -> Result<Box<dyn Processor>> {
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
            if e.tile_size > 0 {
                im.tile = e.tile_size as usize;
            } else {
                im.tile = auto_tile(im.tile, device);
            }
            if e.tile_pad > 0 {
                im.tile_pad = e.tile_pad as usize;
            }
            let scale = im.scale() as u32;
            let expect = sp.native_output.as_ref().map(|f| f.width).unwrap_or(sp.output.width);
            if sp.input.width * scale != expect {
                anyhow::bail!("model {} scale x{scale} does not match the plan", m.id);
            }
            Box::new(ImageProc { label: format!("Enhance ({})", m.id), model: im, params: ImageParams::default(), mix: e.strength })
        }
        Operation::Denoise(d) => {
            let id = op.required_models().into_iter().next().ok_or_else(|| anyhow!("denoise engine has no model"))?;
            let m = model(models, &id)?;
            let w = load_weights(m, "weights", device)?;
            let _ = d.engine;
            let mut im = ImageModel::load(&w, precision).map_err(|e| anyhow!("{}: {e}", m.id))?;
            im.tile = auto_tile(im.tile, device);
            Box::new(ImageProc { label: format!("Denoise ({})", m.id), model: im, params: ImageParams { noise_sigma: d.strength / 255.0 }, mix: d.mix })
        }
        Operation::Deblur(d) => {
            let m = model(models, &d.model_id)?;
            let w = load_weights(m, "weights", device)?;
            let mut im = ImageModel::load(&w, precision).map_err(|e| anyhow!("{}: {e}", m.id))?;
            im.tile = auto_tile(im.tile, device).min(512);
            Box::new(ImageProc { label: format!("Deblur ({})", m.id), model: im, params: ImageParams::default(), mix: d.strength })
        }
        Operation::FaceRestore(f) => {
            let m = model(models, &f.model_id)?;
            let det = load_weights(m, "detector", device)?;
            let res = load_weights(m, "restorer", device)?;
            let fm = FaceModels::load(&det, &res).map_err(|e| anyhow!("{}: {e}", m.id))?;
            Box::new(FaceProc { restorer: FaceRestorer::new(fm), params: f.params.clone(), frames: 0, faces: 0 })
        }
        Operation::Interpolate(i) => {
            let ratio = (sp.output.fps.as_f64() * sp.output.time_scale) / (sp.input.fps.as_f64() * sp.input.time_scale);
            let synth: Box<dyn Synthesizer> = match i.engine {
                InterpEngine::Rife => {
                    let m = model(models, &i.model_id)?;
                    let w = load_weights(m, "weights", device)?;
                    Box::new(RifeSynth { model: InterpModel::load(&w, precision).map_err(|e| anyhow!("{}: {e}", m.id))?, scale: i.rife_scale })
                }
                _ => Box::new(BlendSynth),
            };
            Box::new(InterpProc {
                retimer: Retimer::new(ratio, if i.scene_detection { Some(i.scene_threshold) } else { None }, i.duplicate_threshold),
                synth,
            })
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
    fn grain_processor_keeps_index() {
        let mut p = GrainProc { settings: GrainSettings::default() };
        let mut out = Vec::new();
        p.push(Frame::img(9, RgbImage::from_data(4, 4, vec![0.5; 48])), &mut out).unwrap();
        assert_eq!(out[0].index, 9);
    }
}
