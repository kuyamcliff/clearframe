//! What the desktop app writes to the session log (see `clearframe_app::logging`): a startup
//! header describing the machine, every backend event, periodic performance samples and
//! interface responsiveness. Interface clicks and errors arrive through the `log_ui` command.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use tauri::AppHandle;

use clearframe_app::{App, AppEvent};

/// How often the performance sampler runs.
const PERF_EVERY: Duration = Duration::from_secs(15);

fn trunc(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}… [{} more bytes]", &s[..end], s.len() - end)
}

/// First lines of a session: who, what and where, so a log explains itself.
pub fn header(args: &[String]) {
    log::info!("==================================================================");
    log::info!("ClipFX Enhance {} session start", env!("CARGO_PKG_VERSION"));
    log::info!("os: {} {} ({})", std::env::consts::OS, std::env::consts::ARCH, clearframe_models::manager::platform_id());
    log::info!("exe: {:?}", std::env::current_exe().ok());
    log::info!("args: {:?}", args);
    log::info!("cwd: {:?}", std::env::current_dir().ok());
    log::info!("log file: {:?}", clearframe_app::logging::session_log_path());
    match tauri::webview_version() {
        Ok(v) => log::info!("webview: {v}"),
        Err(e) => log::warn!("webview version unknown: {e}"),
    }
}

/// Details that need the opened app state.
pub fn state_details(app: &App) {
    log::info!("paths: {}", serde_json::to_string(&app.paths()).unwrap_or_default());
    log::info!("settings: {}", serde_json::to_string(&app.settings()).unwrap_or_default());
    log::info!("resource plan: {}", serde_json::to_string(&app.resource_plan()).unwrap_or_default());
}

/// Slower checks (hardware probe, GPU adapters) on a background thread so startup is not delayed.
pub fn background_details(app: Arc<App>) {
    let _ = std::thread::Builder::new().name("log-details".into()).spawn(move || {
        let t = Instant::now();
        log::info!("hardware: {}", serde_json::to_string(&app.hardware()).unwrap_or_default());
        log::info!("compute backends: {}", serde_json::to_string(&clearframe_inference::device::probe()).unwrap_or_default());
        let adapters = clearframe_inference::device::gpu_adapters();
        if adapters.is_empty() {
            log::warn!("no GPU adapter found; processing will use the CPU");
        }
        log::info!("gpu adapters: {}", serde_json::to_string(&adapters).unwrap_or_default());
        log::info!("hardware details took {} ms", t.elapsed().as_millis());
    });
}

/// Logs backend events, throttling the frequent progress ones.
#[derive(Default)]
pub struct EventLogger {
    last: Mutex<HashMap<String, Instant>>,
    seen_jobs: Mutex<std::collections::HashSet<String>>,
}

impl EventLogger {
    /// True when `key` has not been logged within `every`.
    fn due(&self, key: String, every: Duration) -> bool {
        let mut m = self.last.lock();
        let now = Instant::now();
        match m.get(&key) {
            Some(t) if now.duration_since(*t) < every => false,
            _ => {
                m.insert(key, now);
                true
            }
        }
    }

    pub fn log(&self, e: &AppEvent) {
        match e {
            AppEvent::JobUpdated { job } => {
                if self.seen_jobs.lock().insert(job.id.clone()) {
                    log::info!("job {} '{}' ({:?}): {} -> {}", job.id, job.name, job.kind, job.source, job.destination);
                    log::debug!("job {} spec: {}", job.id, trunc(&job.spec_json, 8000));
                }
                let level = if job.error.is_some() { log::Level::Error } else { log::Level::Info };
                log::log!(
                    level,
                    "job {} state {:?} {:.1}% attempts {}{}{}{}",
                    job.id,
                    job.state,
                    job.progress * 100.0,
                    job.attempts,
                    job.detail.as_deref().map(|d| format!(" detail: {d}")).unwrap_or_default(),
                    job.error.as_deref().map(|d| format!(" ERROR: {d}")).unwrap_or_default(),
                    if job.warnings.is_empty() { String::new() } else { format!(" warnings: {:?}", job.warnings) },
                );
                if let Some(r) = &job.report_json {
                    log::info!("job {} report: {}", job.id, trunc(r, 8000));
                }
            }
            AppEvent::JobProgress { id, progress: p } => {
                if self.due(format!("progress:{id}"), Duration::from_secs(5)) {
                    log::info!(
                        "job {id} progress: pass {}/{} {} frames {}/{} {:.1}% {:.2} fps eta {}",
                        p.pass,
                        p.passes,
                        p.phase,
                        p.frames_done,
                        p.frames_total,
                        p.fraction * 100.0,
                        p.fps,
                        p.eta.map(|(a, b)| format!("{a:.0}-{b:.0}s")).unwrap_or_else(|| "-".into())
                    );
                }
            }
            AppEvent::JobLog { id, level, message } => {
                let l = match level.as_str() {
                    "error" => log::Level::Error,
                    "warn" | "warning" => log::Level::Warn,
                    "debug" => log::Level::Debug,
                    _ => log::Level::Info,
                };
                log::log!(l, "job {id}: {message}");
            }
            AppEvent::JobResources { id, sample } => {
                if self.due(format!("res:{id}"), Duration::from_secs(30)) {
                    log::info!("job {id} resources: {}", serde_json::to_string(sample).unwrap_or_default());
                }
            }
            AppEvent::ModelProgress { progress } => {
                let key = format!("model:{}:{:?}", progress.id, progress.state);
                if self.due(key, Duration::from_secs(3)) {
                    log::info!("model {} {:?} {}/{} bytes {}", progress.id, progress.state, progress.bytes_done, progress.bytes_total, progress.message);
                }
            }
            AppEvent::ModelError { id, message } => log::error!("model {id} error: {message}"),
            AppEvent::PreviewFailed { key, message } => log::error!("preview {key} failed: {message}"),
            AppEvent::StorageProgress { .. } | AppEvent::UpdateProgress { .. } => {
                let kind = if matches!(e, AppEvent::UpdateProgress { .. }) { "update" } else { "storage" };
                if self.due(kind.into(), Duration::from_secs(3)) {
                    log::info!("event: {}", serde_json::to_string(e).unwrap_or_default());
                }
            }
            AppEvent::Notice { level, message } => {
                let l = if level == "error" {
                    log::Level::Error
                } else if level.starts_with("warn") {
                    log::Level::Warn
                } else {
                    log::Level::Info
                };
                log::log!(l, "notice: {message}");
            }
            _ => log::info!("event: {}", serde_json::to_string(e).unwrap_or_default()),
        }
    }
}

/// Every `PERF_EVERY`: system and app CPU, memory, GPU, temperature, and how long the
/// interface thread takes to respond (long delays mean the window froze).
pub fn start_perf_sampler(handle: AppHandle, app: Arc<App>) {
    let _ = std::thread::Builder::new().name("log-perf".into()).spawn(move || loop {
        std::thread::sleep(PERF_EVERY);
        let t = Instant::now();
        let s = app.system_sample();
        let gpus: Vec<String> = s
            .gpus
            .iter()
            .map(|g| {
                format!(
                    "{} util {} mem {}/{} MB temp {}",
                    g.name,
                    g.utilization_percent.map(|v| format!("{v:.0}%")).unwrap_or_else(|| "?".into()),
                    g.memory_used_mb.map(|v| v.to_string()).unwrap_or_else(|| "?".into()),
                    g.memory_total_mb.map(|v| v.to_string()).unwrap_or_else(|| "?".into()),
                    g.temperature_c.map(|v| format!("{v:.0}C")).unwrap_or_else(|| "?".into()),
                )
            })
            .collect();
        log::info!(
            "perf: cpu {:.0}% (clipfx {:.0}%) of {} cores | ram used {}/{} MB, available {} MB, clipfx {} MB{} | gpu [{}] | temp {} | sample {} ms{}",
            s.cpu_percent,
            s.clearframe_cpu_percent,
            s.logical_cores,
            s.memory_used_mb,
            s.memory_total_mb,
            s.memory_available_mb,
            s.clearframe_memory_mb,
            if s.memory_pressure { " MEMORY PRESSURE" } else { "" },
            gpus.join("; "),
            s.max_temperature_c.map(|v| format!("{v:.0}C")).unwrap_or_else(|| "?".into()),
            t.elapsed().as_millis(),
            s.thermal_warning.as_deref().map(|w| format!(" | thermal: {w}")).unwrap_or_default(),
        );
        // Interface responsiveness: time for the main thread to pick up a no-op.
        let (tx, rx) = std::sync::mpsc::channel();
        let sent = Instant::now();
        if handle
            .run_on_main_thread(move || {
                let _ = tx.send(());
            })
            .is_ok()
        {
            match rx.recv_timeout(Duration::from_secs(10)) {
                Ok(()) => {
                    let ms = sent.elapsed().as_millis();
                    if ms > 250 {
                        log::warn!("interface thread slow to respond: {ms} ms");
                    }
                }
                Err(_) => log::error!("interface thread did not respond within 10 s (window frozen?)"),
            }
        }
    });
}
