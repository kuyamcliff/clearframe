//! Still-frame previews rendered with the real pipeline.
//!
//! A still preview decodes a few frames at the requested time, runs them
//! through exactly the same processors as a full render and returns the
//! original and processed frames as PNG. Clip previews use the normal job
//! runner with a short range.

use std::collections::BTreeMap;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};

use clearframe_media::color::ColorTags;
use clearframe_media::pipe::{DecodeSpec, FrameReader, RawFrame};
use clearframe_media::probe::MediaInfo;
use clearframe_media::FfmpegTools;

use crate::plan::{self, ExecutionPlan, FilterCaps, ModelLookup, PlanOptions, Segment};
use crate::processors::{build_native, FfmpegProc, Frame, Timed};
use crate::spec::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StillRequest {
    pub source: PathBuf,
    pub time: f64,
    pub pipeline: Pipeline,
    pub output: OutputSettings,
    #[serde(default)]
    pub engine: EngineSettings,
    #[serde(default)]
    pub models: BTreeMap<String, ResolvedModel>,
    #[serde(default)]
    pub resources: JobResources,
    /// Optional crop of the processed frame (x, y, w, h) in output pixels to speed up comparisons.
    #[serde(default)]
    pub crop: Option<(u32, u32, u32, u32)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StillResult {
    /// PNG of the source frame (after rotation, before processing).
    pub original_png: Vec<u8>,
    /// PNG of the processed frame.
    pub processed_png: Vec<u8>,
    pub original_size: (u32, u32),
    pub processed_size: (u32, u32),
    pub seconds: f64,
    pub notes: Vec<String>,
}

/// Convert packed RGB48 to an 8-bit PNG.
pub fn raw_to_png(f: &RawFrame) -> Result<Vec<u8>> {
    let mut rgb8 = Vec::with_capacity(f.width * f.height * 3);
    for c in f.data.as_chunks::<2>().0 {
        let v = u16::from_le_bytes([c[0], c[1]]);
        rgb8.push(((v as u32 + 128) / 257).min(255) as u8);
    }
    let img = image::RgbImage::from_raw(f.width as u32, f.height as u32, rgb8).ok_or_else(|| anyhow!("bad frame buffer"))?;
    let mut out = Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png)?;
    Ok(out.into_inner())
}

fn crop_raw(f: &RawFrame, x: u32, y: u32, w: u32, h: u32) -> RawFrame {
    let (x, y) = (x.min(f.width as u32 - 1) as usize, y.min(f.height as u32 - 1) as usize);
    let w = (w as usize).min(f.width - x).max(1);
    let h = (h as usize).min(f.height - y).max(1);
    let mut data = Vec::with_capacity(w * h * 6);
    for row in y..y + h {
        let s = (row * f.width + x) * 6;
        data.extend_from_slice(&f.data[s..s + w * 6]);
    }
    RawFrame { width: w, height: h, data }
}

#[allow(clippy::too_many_arguments)]
fn decode_at(
    tools: &FfmpegTools,
    info: &MediaInfo,
    plan: &ExecutionPlan,
    source: &Path,
    time: f64,
    filters: Vec<String>,
    fmt: &plan::Format,
    frames: usize,
) -> Result<Vec<RawFrame>> {
    let v = info.video()?;
    let tags = ColorTags::from_stream(v);
    let mut d = DecodeSpec::new(source.to_path_buf(), fmt.width as usize, fmt.height as usize);
    d.stream_index = Some(v.index);
    d.filters = filters;
    d.cfr = plan.cfr_conversion;
    d.matrix = tags.matrix;
    d.full_range = tags.full_range;
    d.start = Some(time.max(0.0));
    let mut r = FrameReader::spawn(tools, &d).map_err(|e| anyhow!("{e}"))?;
    let mut out = Vec::new();
    while out.len() < frames {
        match r.read().map_err(|e| anyhow!("{e}"))? {
            Some(f) => out.push(f),
            None => break,
        }
    }
    r.kill();
    if out.is_empty() {
        bail!("no frame could be decoded at {time:.3}s");
    }
    Ok(out)
}

/// Render a still preview.
pub fn render_still(tools: &FfmpegTools, info: &MediaInfo, req: &StillRequest, lookup: &dyn ModelLookup) -> Result<StillResult> {
    let started = std::time::Instant::now();
    let opts = PlanOptions { detected_interlaced: None, caps: FilterCaps::from_tools(tools) };
    let range = TimeRange { start: req.time.max(0.0), end: (req.time + 1.0).min(info.duration.max(req.time + 0.04)) };
    let auto_req;
    let req = match req.pipeline.with_auto_deinterlace(info.video().map(|v| v.is_interlaced_flagged()).unwrap_or(false)) {
        Some(p) => {
            auto_req = StillRequest { pipeline: p, ..req.clone() };
            &auto_req
        }
        None => req,
    };
    let plan = plan::plan(info, Some(range), &req.pipeline, &req.output, lookup, &opts).map_err(|e| anyhow!(e))?;
    if let Some(e) = plan.errors().next() {
        bail!("{}", e.message);
    }
    if plan.passes.len() > 1 || plan.passes.iter().any(|p| p.stabilize.is_some()) {
        bail!("Stabilization needs motion analysis of the whole clip; use a clip preview instead of a still.");
    }
    let pass = &plan.passes[0];
    let needs_pair = req.pipeline.enabled().any(|s| matches!(s.op, Operation::Interpolate(_)));
    let n_frames = if needs_pair { 2 } else { 1 };
    let compute = crate::runner::select_compute(&req.resources, &req.engine).map_err(|e| anyhow!(e))?;
    let enabled: Vec<&Stage> = req.pipeline.enabled().collect();
    let mut procs: Vec<Timed> = Vec::new();
    for seg in &pass.segments {
        match seg {
            Segment::Native { stage } => {
                let sp = &plan.stages[*stage];
                procs.push(Timed {
                    inner: build_native(&enabled[sp.index].op, sp, &req.models, &compute, req.engine.precision, &req.resources)?,
                    seconds: 0.0,
                    stage_id: sp.stage_id.clone(),
                });
            }
            Segment::Ffmpeg { filters, input, output } => {
                procs.push(Timed { inner: Box::new(FfmpegProc::new(tools, filters, input, output)?), seconds: 0.0, stage_id: "ffmpeg".into() });
            }
        }
    }
    if !pass.encode_filters.is_empty() {
        procs.push(Timed { inner: Box::new(FfmpegProc::new(tools, &pass.encode_filters, &pass.encode_input, &pass.output)?), seconds: 0.0, stage_id: "ffmpeg-out".into() });
    }
    let dec_fmt = match pass.segments.first() {
        Some(Segment::Native { stage }) => plan.stages[*stage].input.clone(),
        Some(Segment::Ffmpeg { input, .. }) => input.clone(),
        None => pass.encode_input.clone(),
    };
    let frames = decode_at(tools, info, &plan, &req.source, req.time, pass.decode_filters.clone(), &dec_fmt, n_frames)?;
    let original = decode_at(tools, info, &plan, &req.source, req.time, vec![], &plan.input, 1)?.remove(0);
    let mut outs: Vec<Frame> = Vec::new();
    let mut cur: Vec<Frame> = frames.into_iter().enumerate().map(|(i, f)| Frame::raw(i as u64, f)).collect();
    for p in procs.iter_mut() {
        let mut next = Vec::new();
        for f in cur {
            p.push(f, &mut next)?;
        }
        p.finish(&mut next)?;
        cur = next;
    }
    outs.extend(cur);
    if outs.is_empty() {
        bail!("the pipeline produced no frame for this position");
    }
    // For interpolation show a synthesised in-between frame (the middle output).
    let pick = if needs_pair { outs.len() / 2 } else { 0 };
    let processed = outs.swap_remove(pick).into_raw();
    let (processed, original) = match req.crop {
        Some((x, y, w, h)) => {
            let sx = original.width as f64 / processed.width as f64;
            let sy = original.height as f64 / processed.height as f64;
            (
                crop_raw(&processed, x, y, w, h),
                crop_raw(&original, (x as f64 * sx) as u32, (y as f64 * sy) as u32, ((w as f64 * sx).ceil() as u32).max(1), ((h as f64 * sy).ceil() as u32).max(1)),
            )
        }
        None => (processed, original),
    };
    let mut notes: Vec<String> = plan.stages.iter().flat_map(|s| s.notes.clone()).collect();
    if needs_pair {
        notes.push("Showing a synthesised frame between the selected frame and the next one.".into());
    }
    Ok(StillResult {
        original_size: (original.width as u32, original.height as u32),
        processed_size: (processed.width as u32, processed.height as u32),
        original_png: raw_to_png(&original)?,
        processed_png: raw_to_png(&processed)?,
        seconds: started.elapsed().as_secs_f64(),
        notes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_and_crop() {
        let f = RawFrame { width: 4, height: 2, data: vec![0xff; 4 * 2 * 6] };
        let png = raw_to_png(&f).unwrap();
        assert_eq!(&png[1..4], b"PNG");
        let c = crop_raw(&f, 1, 0, 2, 2);
        assert_eq!((c.width, c.height, c.data.len()), (2, 2, 24));
        let c = crop_raw(&f, 3, 1, 10, 10);
        assert_eq!((c.width, c.height), (1, 1));
    }
}
