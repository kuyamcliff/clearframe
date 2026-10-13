//! Source inspection, thumbnails, proxies and real-pipeline previews.
//! Everything is written under the managed cache and keyed so that a change
//! of source, settings, models or app version produces a new preview.

use std::collections::BTreeSet;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use clearframe_core::fsutil;
use clearframe_core::jobs::{JobKind, JobState};
use clearframe_core::paths::CacheArea;
use clearframe_media::encode::VideoEncodeSettings;
use clearframe_media::mux::{AudioMode, Container};
use clearframe_media::probe::MediaInfo;
use clearframe_pipeline::preview::StillRequest;
use clearframe_pipeline::spec::*;

use crate::export::NewJob;
use crate::App;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceInfo {
    pub info: MediaInfo,
    pub fingerprint: String,
    pub thumbnails: Vec<(f64, String)>,
    /// The webview can likely play the file directly (otherwise use proxies).
    pub browser_playable: bool,
    pub interlace: Option<clearframe_media::probe::InterlaceAnalysis>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StillPreview {
    pub key: String,
    pub original: String,
    pub processed: String,
    pub original_size: (u32, u32),
    pub processed_size: (u32, u32),
    pub seconds: f64,
    pub notes: Vec<String>,
    pub cached: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipPreview {
    pub key: String,
    /// Ready file when cached.
    pub path: Option<String>,
    /// Preview job id when rendering.
    pub job_id: Option<String>,
    pub original_proxy: String,
}

fn key_of(parts: &[&str]) -> String {
    let mut h = Sha256::new();
    h.update(env!("CARGO_PKG_VERSION"));
    for p in parts {
        h.update([0]);
        h.update(p.as_bytes());
    }
    hex::encode(&h.finalize()[..12])
}

impl App {
    /// Probe a source and prepare thumbnails (cached by fingerprint).
    pub fn source_info(&self, path: &Path, detect_interlace: bool) -> Result<SourceInfo> {
        let tools = self.tools()?;
        let info = self.probe_cached(&tools, path)?;
        let fp = fsutil::quick_fingerprint(path)?;
        let paths = self.paths();
        let tdir = paths.cache(CacheArea::Thumbnails).join(&fp);
        let mut thumbs: Vec<(f64, String)> = Vec::new();
        if tdir.is_dir() {
            if let Ok(rd) = std::fs::read_dir(&tdir) {
                for e in rd.flatten() {
                    let n = e.file_name().to_string_lossy().to_string();
                    if let Some(t) = n.strip_prefix("t_").and_then(|x| x.strip_suffix(".jpg")).and_then(|x| x.parse::<f64>().ok()) {
                        thumbs.push((t / 1000.0, e.path().to_string_lossy().into()));
                    }
                }
            }
        }
        if thumbs.is_empty() && info.duration > 0.0 {
            std::fs::create_dir_all(&tdir)?;
            let temp = paths.cache(CacheArea::Temp);
            for (t, bytes) in clearframe_media::still::thumbnails(&tools, path, info.duration, 16, 90, &temp)? {
                let p = tdir.join(format!("t_{}.jpg", (t * 1000.0).round() as u64));
                std::fs::write(&p, bytes)?;
                thumbs.push((t, p.to_string_lossy().into()));
            }
        }
        thumbs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let v = info.video()?;
        let browser_playable = matches!(v.codec.as_str(), "h264" | "vp8" | "vp9" | "av1")
            && v.bit_depth <= 8
            && (info.container.contains("mp4") || info.container.contains("webm") || info.container.contains("mov"))
            && !v.color.is_hdr();
        let interlace = if detect_interlace { clearframe_media::probe::detect_interlace(&tools, path, 300).ok() } else { None };
        let _ = self.db.add_recent(&path.to_string_lossy(), "video", &path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(), None);
        Ok(SourceInfo { info, fingerprint: fp, thumbnails: thumbs, browser_playable, interlace })
    }

    /// A single original frame as PNG (scrubbing). Cached.
    pub fn source_frame(&self, path: &Path, time: f64, max_width: u32) -> Result<String> {
        let tools = self.tools()?;
        let fp = fsutil::quick_fingerprint(path)?;
        let paths = self.paths();
        let out = paths.cache(CacheArea::Thumbnails).join(&fp).join(format!("f_{}_{max_width}.png", (time * 1000.0).round() as u64));
        if !out.is_file() {
            std::fs::create_dir_all(out.parent().unwrap())?;
            let png = clearframe_media::still::frame_png(&tools, path, time, Some(max_width), &paths.cache(CacheArea::Temp))?;
            std::fs::write(&out, png)?;
        }
        Ok(out.to_string_lossy().into())
    }

    /// Browser-playable proxy of a source range (H.264/AAC). Cached.
    pub fn source_proxy(&self, path: &Path, start: f64, duration: f64, max_height: u32) -> Result<String> {
        let tools = self.tools()?;
        let fp = fsutil::quick_fingerprint(path)?;
        let key = key_of(&[&fp, &format!("{start:.3}"), &format!("{duration:.3}"), &max_height.to_string()]);
        let out = self.paths().cache(CacheArea::Proxies).join(format!("{key}.mp4"));
        if !out.is_file() {
            let tmp = out.with_extension("partial.mp4");
            clearframe_media::still::proxy_clip(&tools, path, start, duration, max_height, &tmp)?;
            std::fs::rename(&tmp, &out)?;
        }
        Ok(out.to_string_lossy().into())
    }

    fn still_key(&self, req: &StillRequest, fp: &str) -> String {
        let shas: Vec<String> = req.models.values().flat_map(|m| m.sha256.values().cloned()).collect();
        key_of(&[
            fp,
            &serde_json::to_string(&req.pipeline).unwrap_or_default(),
            &format!("{:.4}", req.time),
            &format!("{:?}", req.crop),
            &shas.join(","),
            &serde_json::to_string(&req.engine).unwrap_or_default(),
        ])
    }

    /// Render a still preview with the real models in a worker process (cached).
    pub fn preview_still(
        &self,
        source: &Path,
        time: f64,
        pipeline: Pipeline,
        output: OutputSettings,
        engine: EngineSettings,
        crop: Option<(u32, u32, u32, u32)>,
    ) -> Result<StillPreview> {
        if self.storage_busy.load(std::sync::atomic::Ordering::SeqCst) {
            bail!("Previews are unavailable while storage is being moved.");
        }
        let tools = self.tools()?;
        let models = self.resolve_models(&pipeline)?;
        let fp = fsutil::quick_fingerprint(source)?;
        let req = StillRequest { source: source.to_path_buf(), time, pipeline, output, engine, models, resources: self.job_resources(), crop };
        let key = self.still_key(&req, &fp);
        let dir = self.paths().cache(CacheArea::Previews).join(format!("still-{key}"));
        let meta = dir.join("result.json");
        if let Some(mut v) = std::fs::read(&meta).ok().and_then(|b| serde_json::from_slice::<StillPreview>(&b).ok()) {
            if Path::new(&v.processed).is_file() {
                v.cached = true;
                return Ok(v);
            }
        }
        std::fs::create_dir_all(&dir)?;
        let req_path = dir.join("request.json");
        let plan = self.resource_plan();
        let body = serde_json::json!({
            "env": { "ffmpeg": tools.ffmpeg, "ffprobe": tools.ffprobe, "priority": plan.priority, "threads": plan.threads },
            "req": req,
        });
        std::fs::write(&req_path, serde_json::to_vec(&body)?)?;
        let mut cmd = Command::new(&self.queue_worker().program);
        cmd.arg("--still").arg(&req_path).arg(&dir);
        for (k, v) in self.paths().child_env() {
            cmd.env(k, v);
        }
        cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null());
        clearframe_media::tools::hide_console(&mut cmd);
        let mut child = cmd.spawn().map_err(|e| anyhow!("could not start the preview worker: {e}"))?;
        let mut result: Option<serde_json::Value> = None;
        for line in BufReader::new(child.stdout.take().unwrap()).lines().map_while(Result::ok) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                if v["event"] == "result" {
                    result = Some(v);
                }
            }
        }
        let status = child.wait()?;
        let v = result.ok_or_else(|| anyhow!("the preview worker stopped unexpectedly ({status})"))?;
        if v["ok"] != true {
            let _ = std::fs::remove_dir_all(&dir);
            bail!("{}", v["message"].as_str().unwrap_or("preview failed"));
        }
        let size = |k: &str| -> (u32, u32) { (v[k][0].as_u64().unwrap_or(0) as u32, v[k][1].as_u64().unwrap_or(0) as u32) };
        let sp = StillPreview {
            key,
            original: v["original"].as_str().unwrap_or("").into(),
            processed: v["processed"].as_str().unwrap_or("").into(),
            original_size: size("original_size"),
            processed_size: size("processed_size"),
            seconds: v["seconds"].as_f64().unwrap_or(0.0),
            notes: v["notes"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default(),
            cached: false,
        };
        std::fs::write(&meta, serde_json::to_vec(&sp)?)?;
        Ok(sp)
    }

    fn queue_worker(&self) -> crate::WorkerCommand {
        self.queue_worker_cmd()
    }

    /// Render a short clip preview through the real pipeline (preview lane of the queue).
    pub fn preview_clip(self: &Arc<Self>, job: NewJob, start: f64, duration: f64) -> Result<ClipPreview> {
        if self.storage_busy.load(std::sync::atomic::Ordering::SeqCst) {
            bail!("Previews are unavailable while storage is being moved.");
        }
        let fp = fsutil::quick_fingerprint(&job.source)?;
        let s = self.settings();
        let models = self.resolve_models(&job.pipeline)?;
        let shas: Vec<String> = models.values().flat_map(|m| m.sha256.values().cloned()).collect();
        let key = key_of(&[
            &fp,
            &serde_json::to_string(&job.pipeline).unwrap_or_default(),
            &format!("{start:.3}"),
            &format!("{duration:.3}"),
            &shas.join(","),
            &serde_json::to_string(&job.engine).unwrap_or_default(),
        ]);
        let out = self.paths().cache(CacheArea::Previews).join(format!("clip-{key}.mp4"));
        let original_proxy = self.source_proxy(&job.source, start, duration, s.preview_max_height)?;
        if out.is_file() {
            return Ok(ClipPreview { key, path: Some(out.to_string_lossy().into()), job_id: None, original_proxy });
        }
        // Only the newest preview matters: cancel older pending previews.
        for j in self.db.jobs(true).unwrap_or_default() {
            if j.kind == JobKind::Preview && !j.state.is_terminal() && j.name != key {
                let _ = self.cancel_job(&j.id);
            }
            if j.kind == JobKind::Preview
                && j.name == key
                && matches!(j.state, JobState::Queued | JobState::Preparing | JobState::Processing | JobState::Encoding | JobState::Validating)
            {
                return Ok(ClipPreview { key, path: None, job_id: Some(j.id), original_proxy });
            }
        }
        let mut p = job;
        p.range = Some(TimeRange { start, end: start + duration });
        p.output = OutputSettings {
            container: Container::Mp4,
            video: VideoEncodeSettings::preview(),
            audio: AudioMode::Encode { codec: "aac".into(), bitrate_kbps: 160 },
            subtitles: false,
            metadata: false,
            chapters: false,
        };
        p.preset_name = Some(key.clone());
        let rec = self.enqueue_with(p, &BTreeSet::new(), JobKind::Preview, Some(out))?;
        Ok(ClipPreview { key, path: None, job_id: Some(rec.id), original_proxy })
    }
}
