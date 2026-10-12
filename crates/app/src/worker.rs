//! Worker process: runs one job (or one still preview) isolated from the UI
//! process, so a driver crash or memory exhaustion cannot take the app down.
//!
//! Protocol: JSON lines on stdout (`runner::Event` values, then one `result`
//! line). Control commands on stdin: `pause`, `cancel`.

use std::io::{BufRead, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use clearframe_core::settings::ProcessPriority;
use clearframe_media::FfmpegTools;
use clearframe_pipeline::preview::{render_still, StillRequest};
use clearframe_pipeline::runner::{run_job, JobError, JobReport, ResolvedLookup, StopFlag};
use clearframe_pipeline::spec::JobSpec;

/// Everything a worker needs besides the job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerEnvelope {
    pub spec: JobSpec,
    pub ffmpeg: std::path::PathBuf,
    pub ffprobe: std::path::PathBuf,
    pub priority: ProcessPriority,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResultKind {
    Completed,
    Paused,
    Cancelled,
    Invalid,
    Failed,
    Resources,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerResult {
    pub event: String,
    pub kind: ResultKind,
    pub message: String,
    pub report: Option<JobReport>,
}

fn emit<T: Serialize>(v: &T) {
    let mut out = std::io::stdout().lock();
    if let Ok(s) = serde_json::to_string(v) {
        let _ = writeln!(out, "{s}");
        let _ = out.flush();
    }
}

fn apply_threads(threads: u32) {
    if threads > 0 {
        // candle's CPU kernels use rayon's global pool: size it before any work starts.
        let _ = rayon::ThreadPoolBuilder::new().num_threads(threads as usize).build_global();
        clearframe_media::tools::set_default_threads(threads as usize);
    }
}

fn tools_from(env: &WorkerEnvelope) -> Result<FfmpegTools, String> {
    FfmpegTools::from_paths(env.ffmpeg.clone(), env.ffprobe.clone(), clearframe_media::tools::ToolSource::Configured).map_err(|e| e.to_string())
}

/// Entry point for `--worker <envelope.json>`. Returns the process exit code.
pub fn run_worker(envelope_path: &Path) -> i32 {
    let env: WorkerEnvelope = match std::fs::read(envelope_path).ok().and_then(|b| serde_json::from_slice(&b).ok()) {
        Some(e) => e,
        None => {
            emit(&WorkerResult { event: "result".into(), kind: ResultKind::Invalid, message: "unreadable job file".into(), report: None });
            return 2;
        }
    };
    crate::resources::apply_process_priority(env.priority);
    apply_threads(env.spec.resources.threads);
    let tools = match tools_from(&env) {
        Ok(t) => t,
        Err(e) => {
            emit(&WorkerResult { event: "result".into(), kind: ResultKind::Failed, message: e, report: None });
            return 1;
        }
    };
    // Integrity: hash every model file before use (detects corruption since install).
    for m in env.spec.models.values() {
        for (role, p) in &m.files {
            let ok = clearframe_core::fsutil::sha256_file(p).ok().as_deref() == m.sha256.get(role).map(|s| s.as_str());
            if !ok {
                emit(&WorkerResult {
                    event: "result".into(),
                    kind: ResultKind::Invalid,
                    message: format!("Model {} is missing or damaged ({}). Repair it in the Model Library.", m.id, p.display()),
                    report: None,
                });
                return 1;
            }
        }
    }
    let stop = StopFlag::default();
    let s2 = stop.clone();
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines() {
            match line.as_deref().map(str::trim) {
                Ok("pause") => s2.pause(),
                Ok("cancel") => s2.cancel(),
                Ok(_) => {}
                Err(_) => break,
            }
        }
        // stdin closed: the app went away; stop at a safe point so the job can resume.
        s2.pause();
    });
    let lookup = ResolvedLookup { models: &env.spec.models, scales: Default::default() };
    let r = run_job(&tools, &env.spec, &lookup, stop, &mut |e| emit(&e));
    let (kind, message, report, code) = match r {
        Ok(rep) => (ResultKind::Completed, rep.output.display().to_string(), Some(rep), 0),
        Err(JobError::Paused) => (ResultKind::Paused, "Paused; finished segments are kept.".into(), None, 0),
        Err(JobError::Cancelled) => (ResultKind::Cancelled, "Cancelled".into(), None, 0),
        Err(JobError::Invalid(m)) => (ResultKind::Invalid, m, None, 1),
        Err(JobError::Resources(m)) => (ResultKind::Resources, m, None, 0),
        Err(JobError::Failed(e)) => {
            let m = format!("{e:#}");
            let hint = if clearframe_pipeline::processors::is_oom(&m) {
                " The device ran out of memory: lower the memory budget or tile size in Settings > Performance, or use the CPU."
            } else {
                ""
            };
            (ResultKind::Failed, format!("{m}{hint}"), None, 1)
        }
    };
    emit(&WorkerResult { event: "result".into(), kind, message, report });
    code
}

/// Entry point for `--still <request.json> <out-dir>`: renders a still preview and writes PNGs.
pub fn run_still(request_path: &Path, out_dir: &Path) -> i32 {
    #[derive(Deserialize)]
    struct Req {
        env: WorkerEnvelopeStill,
        req: StillRequest,
    }
    #[derive(Deserialize)]
    struct WorkerEnvelopeStill {
        ffmpeg: std::path::PathBuf,
        ffprobe: std::path::PathBuf,
        priority: ProcessPriority,
        threads: u32,
    }
    let r: Req = match std::fs::read(request_path).ok().and_then(|b| serde_json::from_slice(&b).ok()) {
        Some(r) => r,
        None => {
            emit(&serde_json::json!({"event": "result", "ok": false, "message": "unreadable request"}));
            return 2;
        }
    };
    crate::resources::apply_process_priority(r.env.priority);
    apply_threads(r.env.threads);
    let result = (|| -> Result<serde_json::Value, String> {
        let tools = FfmpegTools::from_paths(r.env.ffmpeg, r.env.ffprobe, clearframe_media::tools::ToolSource::Configured).map_err(|e| e.to_string())?;
        let info = clearframe_media::probe::probe(&tools, &r.req.source).map_err(|e| e.to_string())?;
        let lookup = ResolvedLookup { models: &r.req.models, scales: Default::default() };
        let s = render_still(&tools, &info, &r.req, &lookup).map_err(|e| format!("{e:#}"))?;
        std::fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
        let o = out_dir.join("original.png");
        let p = out_dir.join("processed.png");
        std::fs::write(&o, &s.original_png).map_err(|e| e.to_string())?;
        std::fs::write(&p, &s.processed_png).map_err(|e| e.to_string())?;
        Ok(serde_json::json!({
            "event": "result", "ok": true,
            "original": o, "processed": p,
            "original_size": s.original_size, "processed_size": s.processed_size,
            "seconds": s.seconds, "notes": s.notes
        }))
    })();
    match result {
        Ok(v) => {
            emit(&v);
            0
        }
        Err(m) => {
            emit(&serde_json::json!({"event": "result", "ok": false, "message": m}));
            1
        }
    }
}
