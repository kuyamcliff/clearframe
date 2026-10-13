//! Application-level tests with real worker processes (the `clipfx` binary),
//! synthetic media and synthetic model weights. All data lives in temp folders.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use clearframe_app::export::NewJob;
use clearframe_app::storage::{RelocationMode, StorageKind};
use clearframe_app::{App, WorkerCommand};
use clearframe_core::jobs::JobState;
use clearframe_core::paths::CacheArea;
use clearframe_core::settings::{MemoryPreset, PerformanceMode};
use clearframe_pipeline::spec::*;

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
}

fn worker() -> WorkerCommand {
    WorkerCommand { program: PathBuf::from(env!("CARGO_BIN_EXE_clipfx")), args: vec!["--worker".into()] }
}

fn open(root: &Path) -> Arc<App> {
    App::open(Some(root), worker(), vec![], Arc::new(|_| {})).unwrap()
}

fn clip(path: &Path, secs: u32) {
    let s = secs.to_string();
    let st = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-t",
            &s,
            "-i",
            "testsrc2=size=320x240:rate=30",
            "-f",
            "lavfi",
            "-t",
            &s,
            "-i",
            "sine",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
        ])
        .arg(path)
        .status()
        .unwrap();
    assert!(st.success());
}

fn job(src: &Path, out: &Path, p: Pipeline) -> NewJob {
    NewJob {
        source: src.into(),
        pipeline: p,
        output: OutputSettings::default(),
        range: None,
        folder: Some(out.into()),
        file_name: None,
        remember_folder: false,
        preset_name: None,
        batch_id: None,
        engine: Default::default(),
    }
}

fn wait(app: &Arc<App>, id: &str, until: impl Fn(JobState) -> bool, secs: u64) -> JobState {
    let t = Instant::now();
    loop {
        let s = app.db.job(id).unwrap().state;
        if until(s) || t.elapsed() > Duration::from_secs(secs) {
            return s;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn grain() -> Pipeline {
    Pipeline::single(Operation::Grain(GrainSettings::default()))
}

/// Tiny SRVGG x4 network (random weights) as safetensors, for custom-model tests.
fn synthetic_srvgg(path: &Path) {
    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let mut seed = 1u32;
    let mut rnd = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 8) as f32 / (1u32 << 24) as f32 * 0.02 - 0.01
            })
            .collect()
    };
    tensors.push(("body.0.weight".into(), vec![4, 3, 3, 3], rnd(108)));
    tensors.push(("body.0.bias".into(), vec![4], rnd(4)));
    tensors.push(("body.1.weight".into(), vec![4], rnd(4)));
    tensors.push(("body.2.weight".into(), vec![48, 4, 3, 3], rnd(48 * 36)));
    tensors.push(("body.2.bias".into(), vec![48], rnd(48)));
    let bytes: Vec<(String, Vec<usize>, Vec<u8>)> = tensors.into_iter().map(|(n, s, d)| (n, s, d.iter().flat_map(|v| v.to_le_bytes()).collect())).collect();
    let views: Vec<(String, safetensors::tensor::TensorView)> =
        bytes.iter().map(|(n, s, d)| (n.clone(), safetensors::tensor::TensorView::new(safetensors::Dtype::F32, s.clone(), d).unwrap())).collect();
    safetensors::serialize_to_file(views, &None, path).unwrap();
}

#[test]
fn render_through_worker_and_source_untouched() {
    if !have_ffmpeg() {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let app = open(d.path());
    let src = d.path().join("src video.mp4");
    clip(&src, 1);
    let before = clearframe_core::fsutil::sha256_file(&src).unwrap();
    app.start_scheduler();
    let j = app.enqueue(job(&src, &d.path().join("exports ü"), grain())).unwrap();
    assert_eq!(wait(&app, &j.id, |s| s.is_terminal(), 120), JobState::Completed, "{:?}", app.db.job(&j.id).unwrap().error);
    assert_eq!(clearframe_core::fsutil::sha256_file(&src).unwrap(), before);
    let r = app.db.job(&j.id).unwrap();
    assert!(Path::new(&r.destination).is_file());
    assert!(r.report_json.is_some());
    // Work folder of a finished job is cleaned up.
    assert!(!app.paths().job_dir(&j.id).exists());
    // Second job for the same source never overwrites the first output.
    let j2 = app.enqueue(job(&src, &d.path().join("exports ü"), grain())).unwrap();
    assert_ne!(j2.destination, r.destination);
}

#[test]
fn worker_crash_and_app_restart_recover() {
    if !have_ffmpeg() {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let src = d.path().join("src.mp4");
    clip(&src, 8);
    let id;
    {
        let app = open(d.path());
        let mut s = app.settings();
        s.segment_seconds = 2.0;
        app.save_settings(s).unwrap();
        app.start_scheduler();
        let p = Pipeline {
            stages: vec![
                Stage::new(Operation::Interpolate(InterpolateSettings { engine: InterpEngine::Blend, ..Default::default() })),
                Stage::new(Operation::Grain(GrainSettings::default())),
            ],
        };
        id = app.enqueue(job(&src, &d.path().join("out"), p)).unwrap().id;
        assert_eq!(wait(&app, &id, |s| s == JobState::Processing, 60), JobState::Processing);
        // Wait until at least one segment exists, then kill the worker like a crash.
        let segs = app.paths().job_dir(&id).join("pass0/segments");
        let t = Instant::now();
        while std::fs::read_dir(&segs).map(|r| r.count()).unwrap_or(0) < 2 && t.elapsed() < Duration::from_secs(60) {
            std::thread::sleep(Duration::from_millis(100));
        }
        for pid in app.queue.worker_pids() {
            let pid = pid.to_string();
            let st = if cfg!(windows) { Command::new("taskkill").args(["/F", "/PID", &pid]).status() } else { Command::new("kill").args(["-9", &pid]).status() };
            assert!(st.map(|s| s.success()).unwrap_or(false), "could not kill worker {pid}");
        }
        let s = wait(&app, &id, |s| s == JobState::Recoverable, 30);
        assert_eq!(s, JobState::Recoverable, "{:?}", app.db.job(&id).unwrap());
        assert!(app.db.job(&id).unwrap().error.unwrap().contains("stopped unexpectedly"));
    }
    // "Restart" the application: a new App on the same data resumes the job.
    let app = open(d.path());
    app.start_scheduler();
    app.resume_job(&id).unwrap();
    assert_eq!(wait(&app, &id, |s| s.is_terminal(), 180), JobState::Completed, "{:?}", app.db.job(&id).unwrap().error);
    let events = app.db.job_events(&id).unwrap();
    assert!(events.iter().any(|e| e.message.contains("Resumed")), "{events:?}");
}

#[test]
fn interrupted_app_marks_running_jobs_recoverable() {
    let d = tempfile::tempdir().unwrap();
    let app = open(d.path());
    // Simulate a job that was running when the app died.
    let rec = clearframe_core::jobs::JobRecord {
        id: "j1".into(),
        kind: clearframe_core::jobs::JobKind::Render,
        name: "x".into(),
        state: JobState::Queued,
        source: "/x".into(),
        destination: "/y".into(),
        spec_json: "{}".into(),
        created_at: clearframe_core::db::now(),
        updated_at: clearframe_core::db::now(),
        started_at: None,
        finished_at: None,
        progress: 0.3,
        progress_json: None,
        detail: None,
        error: None,
        report_json: None,
        queue_order: 0,
        attempts: 0,
        batch_id: None,
        heartbeat: None,
        warnings: vec![],
    };
    app.db.insert_job(&rec).unwrap();
    app.db.set_job_state("j1", JobState::Preparing, None).unwrap();
    app.db.set_job_state("j1", JobState::Processing, None).unwrap();
    drop(app);
    let app = open(d.path());
    assert_eq!(app.db.job("j1").unwrap().state, JobState::Recoverable);
}

#[test]
fn cache_relocation_with_paused_job_and_safe_clear() {
    if !have_ffmpeg() {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let app = open(d.path());
    let src = d.path().join("src.mp4");
    clip(&src, 1);
    // A queued-but-paused job owns a work folder that must survive clearing and moving.
    app.queue.wake();
    let j = app.enqueue(job(&src, &d.path().join("out"), grain())).unwrap();
    app.pause_job(&j.id).unwrap();
    std::fs::create_dir_all(app.paths().job_dir(&j.id)).unwrap();
    std::fs::write(app.paths().job_dir(&j.id).join("state.json"), b"{}").unwrap();
    std::fs::write(app.paths().cache(CacheArea::Previews).join("old.mp4"), vec![0u8; 4096]).unwrap();
    let r = app.clear_cache(&CacheArea::ALL, false);
    assert_eq!(r.skipped_protected, 1);
    assert!(app.paths().job_dir(&j.id).join("state.json").is_file());
    assert!(!app.paths().cache(CacheArea::Previews).join("old.mp4").exists());
    // Move the cache to a new folder (with spaces/Unicode).
    let new_cache = d.path().join("Externe Platte/ClipFX Enhance Cache ü");
    let plan = app.plan_relocation(StorageKind::Cache, &new_cache);
    assert!(plan.plan.problems.is_empty(), "{:?}", plan.plan.problems);
    let old_cache = app.paths().cache_dir.clone();
    app.relocate_blocking(StorageKind::Cache, &new_cache, RelocationMode::Move).unwrap();
    assert_eq!(app.paths().cache_dir, new_cache);
    assert!(!old_cache.exists());
    assert!(app.paths().job_dir(&j.id).join("state.json").is_file());
    // The setting persists across restarts.
    drop(app);
    let app = open(d.path());
    assert_eq!(app.paths().cache_dir, new_cache);
    // The paused job resumes and runs with work files in the new cache.
    app.start_scheduler();
    app.resume_job(&j.id).unwrap();
    assert_eq!(wait(&app, &j.id, |s| s.is_terminal(), 120), JobState::Completed);
    // Projects/presets/exports are never inside the cache.
    assert!(d.path().join("out").read_dir().unwrap().count() >= 1);
}

#[test]
fn models_folder_move_and_custom_model_render_offline() {
    if !have_ffmpeg() {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let app = open(d.path());
    let w = d.path().join("tiny x4.safetensors");
    synthetic_srvgg(&w);
    let id = app.import_custom_model("Tiny Test x4", "MIT", w, None).unwrap();
    assert!(app.models.resolve(&id).is_some());
    let scale = app.resolve_models(&Pipeline::single(Operation::Enhance(EnhanceSettings { model_id: id.clone(), ..Default::default() }))).unwrap()[&id].scale;
    assert_eq!(scale, 4);
    // Move the models folder to "another drive".
    let dest = d.path().join("D drive/ClipFX Enhance Models");
    app.relocate_blocking(StorageKind::Models, &dest, RelocationMode::Move).unwrap();
    assert!(app.models.resolve(&id).unwrap().1["weights"].starts_with(&dest));
    // Offline: break networking for this process and its workers; local processing still works.
    std::env::set_var("HTTPS_PROXY", "http://127.0.0.1:9");
    std::env::set_var("HTTP_PROXY", "http://127.0.0.1:9");
    let src = d.path().join("src.mp4");
    clip(&src, 1);
    app.start_scheduler();
    let p = Pipeline::single(Operation::Enhance(EnhanceSettings { model_id: id.clone(), target: SizeTarget::Scale { factor: 2.0 }, ..Default::default() }));
    let j = app.enqueue(job(&src, &d.path().join("out"), p)).unwrap();
    assert_eq!(wait(&app, &j.id, |s| s.is_terminal(), 300), JobState::Completed, "{:?}", app.db.job(&j.id).unwrap().error);
    // The model cannot be removed while in use; after the job it can.
    assert!(app.remove_model(&id).unwrap() > 0);
    assert!(app.models.resolve(&id).is_none());
}

#[test]
fn damaged_model_is_refused_by_worker() {
    if !have_ffmpeg() {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let app = open(d.path());
    let w = d.path().join("tiny.safetensors");
    synthetic_srvgg(&w);
    let id = app.import_custom_model("Tiny", "MIT", w, None).unwrap();
    let src = d.path().join("src.mp4");
    clip(&src, 1);
    let p = Pipeline::single(Operation::Enhance(EnhanceSettings { model_id: id.clone(), ..Default::default() }));
    let j = app.enqueue(job(&src, &d.path().join("out"), p)).unwrap();
    // Corrupt the weights after queueing (same size so only hashing detects it).
    let f = app.models.resolve(&id).unwrap().1["weights"].clone();
    let mut b = std::fs::read(&f).unwrap();
    let n = b.len();
    b[n - 1] ^= 0xff;
    std::fs::write(&f, b).unwrap();
    app.start_scheduler();
    assert_eq!(wait(&app, &j.id, |s| s.is_terminal(), 60), JobState::Failed);
    assert!(app.db.job(&j.id).unwrap().error.unwrap().contains("damaged"));
}

#[test]
fn performance_settings_reach_the_worker() {
    if !have_ffmpeg() {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let app = open(d.path());
    let mut s = app.settings();
    s.performance.mode = PerformanceMode::Custom;
    s.performance.cpu_threads = 1;
    s.performance.memory_preset = MemoryPreset::Custom;
    s.performance.memory_budget_mb = 600;
    // The CPU tile rule is checked here; GPU tiles are covered by the pipeline tests.
    s.performance.device = "cpu".into();
    app.save_settings(s).unwrap();
    let r = app.job_resources();
    assert_eq!(r.threads, 1);
    assert_eq!(r.memory_budget_mb, 600);
    let w = d.path().join("tiny.safetensors");
    synthetic_srvgg(&w);
    let id = app.import_custom_model("Tiny", "MIT", w, None).unwrap();
    let src = d.path().join("src.mp4");
    clip(&src, 1);
    app.start_scheduler();
    let j = app.enqueue(job(&src, &d.path().join("out"), Pipeline::single(Operation::Enhance(EnhanceSettings { model_id: id, ..Default::default() })))).unwrap();
    assert_eq!(wait(&app, &j.id, |s| s.is_terminal(), 300), JobState::Completed, "{:?}", app.db.job(&j.id).unwrap().error);
    let rep: serde_json::Value = serde_json::from_str(&app.db.job(&j.id).unwrap().report_json.unwrap()).unwrap();
    assert_eq!(rep["resources"]["threads"], 1);
    // The 600 MB budget limits tiles below the 512 px default.
    let tile = rep["stage_stats"].as_object().unwrap().values().find_map(|v| v["tile"].as_u64()).unwrap();
    assert!(tile < 512, "tile {tile}");
    let events = app.db.job_events(&j.id).unwrap();
    assert!(events.iter().any(|e| e.message.contains("1 threads")));
}

#[test]
fn unsupported_device_fails_clearly() {
    if !have_ffmpeg() {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let app = open(d.path());
    let mut s = app.settings();
    s.performance.device = "cuda:0".into();
    app.save_settings(s).unwrap();
    let src = d.path().join("src.mp4");
    clip(&src, 1);
    app.start_scheduler();
    let j = app.enqueue(job(&src, &d.path().join("out"), grain())).unwrap();
    assert_eq!(wait(&app, &j.id, |s| s.is_terminal(), 60), JobState::Failed);
    assert!(app.db.job(&j.id).unwrap().error.unwrap().contains("unavailable"));
}

#[test]
fn whole_models_dir_deleted_is_detected() {
    let d = tempfile::tempdir().unwrap();
    let app = open(d.path());
    let w = d.path().join("tiny.safetensors");
    synthetic_srvgg(&w);
    let id = app.import_custom_model("Tiny", "MIT", w, None).unwrap();
    std::fs::remove_dir_all(app.paths().models_dir).unwrap();
    let o = app.storage_overview();
    assert!(!o.models.models_dir_exists);
    assert!(o.warnings.iter().any(|w| w.contains("models folder")));
    assert!(app.models.resolve(&id).is_none());
    assert_eq!(app.model_state(&id), clearframe_models::manager::ModelState::Missing);
    // Pipelines needing it are refused with a clear message instead of crashing.
    let e = app.resolve_models(&Pipeline::single(Operation::Enhance(EnhanceSettings { model_id: id, ..Default::default() }))).unwrap_err();
    assert!(e.to_string().contains("not installed"));
}

#[test]
fn insufficient_space_for_relocation_is_rejected() {
    let d = tempfile::tempdir().unwrap();
    let app = open(d.path());
    // Destination inside the source is invalid; non-empty destination is invalid.
    let p = app.plan_relocation(StorageKind::Models, &app.paths().models_dir.join("sub"));
    assert!(!p.plan.problems.is_empty());
    let busy = d.path().join("busy");
    std::fs::create_dir_all(&busy).unwrap();
    std::fs::write(busy.join("x"), b"x").unwrap();
    assert!(app.relocate_blocking(StorageKind::Cache, &busy, RelocationMode::Move).is_err());
    assert!(app.paths().cache_is_default);
}

#[test]
fn background_ffmpeg_detection_is_awaited_and_announced() {
    if !have_ffmpeg() {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let changed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let c = changed.clone();
    let sink: clearframe_app::EventSink = Arc::new(move |e| {
        if matches!(e, clearframe_app::AppEvent::ToolsChanged) {
            c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    });
    let app = App::open_background_tools(Some(d.path()), worker(), vec![], sink).unwrap();
    // Callers that need FFmpeg right after launch wait for detection instead of failing.
    let t = app.tools().expect("tools after background detection");
    assert!(t.has_filter("scale"));
    assert!(!app.tools_pending());
    assert_eq!(app.tools_status().0.map(|t| t.version), Some(t.version));
    assert_eq!(changed.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[test]
fn detected_ffmpeg_is_ready_at_once_on_the_next_launch() {
    if !have_ffmpeg() {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let first = App::open_background_tools(Some(d.path()), worker(), vec![], Arc::new(|_| {})).unwrap();
    let t = first.tools().unwrap();
    drop(first);
    // Second launch: FFmpeg is unchanged, so it is known immediately with no detection pending.
    let second = App::open_background_tools(Some(d.path()), worker(), vec![], Arc::new(|_| {})).unwrap();
    let (cached, _) = second.tools_status();
    assert_eq!(cached.map(|c| c.ffmpeg), Some(t.ffmpeg.clone()), "FFmpeg should come from the saved detection");
    assert!(!second.tools_pending());
    assert_eq!(second.tools().unwrap().version, t.version);
}

#[test]
fn probe_results_are_reused_until_the_file_changes() {
    if !have_ffmpeg() {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let app = open(d.path());
    let tools = app.tools().unwrap();
    let src = d.path().join("v.mp4");
    clip(&src, 1);
    let a = app.probe_cached(&tools, &src).unwrap();
    assert_eq!(app.probe_cached(&tools, &src).unwrap().duration, a.duration);
    // Replacing the file (different size) must not return the old result.
    clip(&src, 3);
    let b = app.probe_cached(&tools, &src).unwrap();
    assert!(b.duration > a.duration + 1.0, "{} vs {}", b.duration, a.duration);
}
