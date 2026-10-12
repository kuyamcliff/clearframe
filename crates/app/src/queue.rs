//! Job queue and scheduler.
//!
//! Each job runs in its own worker process. The scheduler starts queued jobs
//! in order while the concurrency limit, memory budget and storage state
//! allow, pauses running jobs under memory pressure (when configured), and
//! records every state transition in the database.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};
use crossbeam_channel::{unbounded, Receiver, Sender};
use parking_lot::Mutex;

use clearframe_core::jobs::{JobKind, JobRecord, JobState};
use clearframe_core::settings::MemoryPressureAction;
use clearframe_pipeline::runner::Event;
use clearframe_pipeline::spec::JobSpec;

use crate::worker::{ResultKind, WorkerEnvelope, WorkerResult};
use crate::{App, AppEvent, WorkerCommand};

struct Running {
    stdin: Arc<Mutex<Option<ChildStdin>>>,
    pid: u32,
    kind: JobKind,
    models: Vec<String>,
    /// Paused automatically because of memory pressure.
    pressure_paused: bool,
}

pub struct Queue {
    worker: WorkerCommand,
    running: Mutex<HashMap<String, Running>>,
    paused: AtomicBool,
    wake_tx: Sender<()>,
    wake_rx: Receiver<()>,
    started: AtomicBool,
    /// Jobs paused by the memory-pressure guard (resumed automatically).
    pressure_held: Mutex<Vec<String>>,
}

impl Queue {
    pub fn new(worker: WorkerCommand) -> Self {
        let (wake_tx, wake_rx) = unbounded();
        Self { worker, running: Mutex::new(HashMap::new()), paused: AtomicBool::new(false), wake_tx, wake_rx, started: AtomicBool::new(false), pressure_held: Mutex::new(vec![]) }
    }

    pub fn wake(&self) {
        let _ = self.wake_tx.send(());
    }

    pub fn worker_pids(&self) -> Vec<u32> {
        self.running.lock().values().map(|r| r.pid).collect()
    }

    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::SeqCst)
    }

    fn send(&self, id: &str, cmd: &str) -> bool {
        let g = self.running.lock();
        let Some(r) = g.get(id) else { return false };
        let mut s = r.stdin.lock();
        match s.as_mut() {
            Some(w) => writeln!(w, "{cmd}").and_then(|_| w.flush()).is_ok(),
            None => false,
        }
    }

    /// Model ids used by running work (never removed while in use).
    pub fn models_in_use(&self) -> Vec<String> {
        self.running.lock().values().flat_map(|r| r.models.clone()).collect()
    }
}

impl App {
    pub fn queue_worker_cmd(&self) -> WorkerCommand {
        self.queue.worker.clone()
    }

    pub fn running_jobs(&self) -> usize {
        self.queue.running.lock().len()
    }

    fn update_job_event(&self, id: &str) {
        if let Ok(j) = self.db.job(id) {
            self.emit(AppEvent::JobUpdated { job: Box::new(j) });
        }
    }

    fn set_state(&self, id: &str, s: JobState, detail: Option<&str>) {
        match self.db.set_job_state(id, s, detail) {
            Ok(_) => self.update_job_event(id),
            Err(e) => log::debug!("job {id}: {e}"),
        }
    }

    pub fn queue_state_event(&self) {
        let jobs = self.db.jobs(false).unwrap_or_default();
        let queued = jobs.iter().filter(|j| j.state == JobState::Queued || j.state == JobState::WaitingForResources).count();
        self.emit(AppEvent::QueueState { paused: self.queue.is_paused(), running: self.running_jobs(), queued });
    }

    /// Start the scheduler thread (idempotent).
    pub fn start_scheduler(self: &Arc<Self>) {
        if self.queue.started.swap(true, Ordering::SeqCst) {
            return;
        }
        let app = self.clone();
        std::thread::Builder::new()
            .name("scheduler".into())
            .spawn(move || {
                let mut last_policy = Instant::now() - Duration::from_secs(3600);
                loop {
                    let _ = app.queue.wake_rx.recv_timeout(Duration::from_millis(1000));
                    while app.queue.wake_rx.try_recv().is_ok() {}
                    app.check_memory_pressure();
                    app.schedule();
                    if last_policy.elapsed() > Duration::from_secs(600) && app.running_jobs() == 0 {
                        let r = app.enforce_cache_policy();
                        if r.removed_entries > 0 {
                            log::info!("cache policy removed {} entries ({} MB)", r.removed_entries, r.freed_bytes / 1_000_000);
                        }
                        last_policy = Instant::now();
                    }
                }
            })
            .expect("spawn scheduler");
    }

    fn check_memory_pressure(&self) {
        let s = self.settings();
        if s.performance.memory_pressure != MemoryPressureAction::Pause {
            return;
        }
        let plan = self.resource_plan();
        let sample = self.system_sample();
        let critical = sample.memory_available_mb < plan.reserve_memory_mb / 2;
        let ok_again = sample.memory_available_mb > plan.reserve_memory_mb;
        if critical {
            let ids: Vec<String> = self.queue.running.lock().iter().filter(|(_, r)| r.kind == JobKind::Render && !r.pressure_paused).map(|(k, _)| k.clone()).collect();
            if let Some(id) = ids.last() {
                log::warn!("memory pressure ({} MB available): pausing {id}", sample.memory_available_mb);
                if let Some(r) = self.queue.running.lock().get_mut(id) {
                    r.pressure_paused = true;
                }
                self.queue.send(id, "pause");
                let _ = self.db.log_job_event(id, "warning", &format!("Paused because only {} MB of memory was available; it continues automatically when memory is freed.", sample.memory_available_mb));
            }
        } else if ok_again {
            let held: Vec<String> = std::mem::take(&mut *self.queue.pressure_held.lock());
            for id in held {
                if self.db.job(&id).map(|j| j.state == JobState::WaitingForResources).unwrap_or(false) {
                    self.set_state(&id, JobState::Queued, Some("Memory is available again"));
                }
            }
        }
    }

    fn schedule(self: &Arc<Self>) {
        if self.queue.is_paused() || self.storage_busy.load(Ordering::SeqCst) {
            return;
        }
        let Ok(jobs) = self.db.jobs(true) else { return };
        let plan = self.resource_plan();
        // Previews have their own lane (one at a time) so they never wait behind long renders.
        let running = self.queue.running.lock().iter().map(|(_, r)| r.kind).collect::<Vec<_>>();
        let renders = running.iter().filter(|k| **k == JobKind::Render).count() as u32;
        let previews = running.iter().filter(|k| **k == JobKind::Preview).count();
        if previews == 0 {
            if let Some(j) = jobs.iter().find(|j| j.kind == JobKind::Preview && j.state == JobState::Queued) {
                self.start_job(j.clone());
            }
        }
        if renders >= plan.max_concurrent_jobs {
            return;
        }
        let Some(j) = jobs.iter().find(|j| j.kind == JobKind::Render && matches!(j.state, JobState::Queued | JobState::WaitingForResources)) else { return };
        // Memory gate: start only when the estimate fits what is available (always allow when idle).
        let est = serde_json::from_str::<JobSpec>(&j.spec_json).ok().map(|s| s.estimated_memory_mb).unwrap_or(0.0) as u64;
        let sample = self.system_sample();
        let need = est.min(plan.memory_budget_mb / plan.max_concurrent_jobs.max(1) as u64);
        if renders > 0 && sample.memory_available_mb < need + plan.reserve_memory_mb / 2 {
            if j.state != JobState::WaitingForResources {
                self.set_state(&j.id, JobState::WaitingForResources, Some(&format!("Waiting for memory: needs about {need} MB, {} MB available", sample.memory_available_mb)));
            }
            return;
        }
        if renders == 0 && need > 0 && sample.memory_available_mb < need {
            let _ = self.db.log_job_event(&j.id, "warning", &format!("Starting with about {} MB available although about {need} MB is estimated; tiles are reduced to fit the memory budget.", sample.memory_available_mb));
        }
        self.start_job(j.clone());
    }

    fn start_job(self: &Arc<Self>, j: JobRecord) {
        if let Err(e) = self.try_start_job(&j) {
            log::warn!("job {} could not start: {e}", j.id);
            let _ = self.db.set_job_state(&j.id, JobState::Preparing, None);
            let _ = self.db.set_job_error(&j.id, &e.to_string());
            self.set_state(&j.id, JobState::Failed, Some(&e.to_string()));
        }
    }

    fn try_start_job(self: &Arc<Self>, j: &JobRecord) -> Result<()> {
        let tools = self.tools()?;
        let mut spec: JobSpec = serde_json::from_str(&j.spec_json)?;
        let paths = self.paths();
        // Work folders always live in the current cache location (it may have moved).
        spec.work_dir = paths.job_dir(&spec.job_id);
        // Apply the current performance settings.
        spec.resources = self.job_resources();
        std::fs::create_dir_all(&spec.work_dir)?;
        let envelope = WorkerEnvelope { ffmpeg: tools.ffmpeg.clone(), ffprobe: tools.ffprobe.clone(), priority: self.resource_plan().priority, spec: spec.clone() };
        let env_path = spec.work_dir.join("job.json");
        std::fs::write(&env_path, serde_json::to_vec_pretty(&envelope)?)?;
        self.db.set_job_spec(&j.id, &serde_json::to_string(&spec)?, &spec.destination.to_string_lossy()).map_err(|e| anyhow!("{e}"))?;
        let mut cmd = Command::new(&self.queue.worker.program);
        cmd.args(&self.queue.worker.args).arg(&env_path);
        for (k, v) in paths.child_env() {
            cmd.env(k, v);
        }
        if spec.resources.threads > 0 {
            cmd.env("RAYON_NUM_THREADS", spec.resources.threads.to_string());
        }
        let log_file = std::fs::File::create(spec.work_dir.join("worker.log"))?;
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::from(log_file));
        clearframe_media::tools::hide_console(&mut cmd);
        let mut child = cmd.spawn().map_err(|e| anyhow!("could not start the processing worker: {e}"))?;
        let stdin = Arc::new(Mutex::new(child.stdin.take()));
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("no worker output"))?;
        self.queue.running.lock().insert(j.id.clone(), Running { stdin, pid: child.id(), kind: j.kind, models: spec.models.keys().cloned().collect(), pressure_paused: false });
        let _ = self.db.log_job_event(&j.id, "info", &format!("Started (worker pid {}, {} threads, budget {} MB, device {})", child.id(), spec.resources.threads, spec.resources.memory_budget_mb, spec.resources.device));
        if j.state == JobState::WaitingForResources {
            self.set_state(&j.id, JobState::Queued, None);
        }
        self.set_state(&j.id, JobState::Preparing, Some("Starting"));
        self.queue_state_event();
        let app = self.clone();
        let id = j.id.clone();
        let kind = j.kind;
        std::thread::Builder::new().name(format!("job-{id}")).spawn(move || {
            let mut result: Option<WorkerResult> = None;
            let mut last_db = Instant::now() - Duration::from_secs(10);
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if line.contains("\"event\":\"result\"") {
                    result = serde_json::from_str(&line).ok();
                    continue;
                }
                let Ok(ev) = serde_json::from_str::<Event>(&line) else { continue };
                match ev {
                    Event::State { state, detail } => app.set_state(&id, state, Some(&detail)),
                    Event::Progress(p) => {
                        if last_db.elapsed() > Duration::from_millis(900) || p.fraction >= 1.0 {
                            let _ = app.db.set_job_progress(&id, p.fraction, &serde_json::to_string(&p).unwrap_or_default());
                            last_db = Instant::now();
                        }
                        app.emit(AppEvent::JobProgress { id: id.clone(), progress: p });
                    }
                    Event::Log { level, message } => {
                        let _ = app.db.log_job_event(&id, &level, &message);
                        app.emit(AppEvent::JobLog { id: id.clone(), level, message });
                    }
                    Event::Warning { message } => {
                        let _ = app.db.add_job_warning(&id, &message);
                        let _ = app.db.log_job_event(&id, "warning", &message);
                        app.emit(AppEvent::JobLog { id: id.clone(), level: "warning".into(), message });
                    }
                    Event::Resumed { from_frame, .. } => {
                        let _ = app.db.log_job_event(&id, "info", &format!("Resumed from frame {from_frame}"));
                    }
                    Event::Resources(sample) => app.emit(AppEvent::JobResources { id: id.clone(), sample }),
                    Event::Plan { .. } => {}
                }
            }
            let status = child.wait();
            let pressure = app.queue.running.lock().remove(&id).map(|r| r.pressure_paused).unwrap_or(false);
            app.finish_job(&id, kind, result, status.ok().and_then(|s| s.code()), pressure);
            app.queue_state_event();
            app.queue.wake();
        })?;
        Ok(())
    }

    fn finish_job(&self, id: &str, kind: JobKind, result: Option<WorkerResult>, code: Option<i32>, pressure: bool) {
        let s = self.settings();
        match result {
            Some(r) => match r.kind {
                ResultKind::Completed => {
                    if let Some(rep) = &r.report {
                        let _ = self.db.set_job_report(id, &serde_json::to_string(rep).unwrap_or_default(), &rep.output.to_string_lossy());
                    }
                    self.set_state(id, JobState::Completed, Some(&r.message));
                    if kind == JobKind::Preview {
                        if let Ok(j) = self.db.job(id) {
                            self.emit(AppEvent::PreviewReady { key: j.name.clone(), path: j.destination.clone() });
                        }
                    }
                }
                ResultKind::Paused if pressure => {
                    self.set_state(id, JobState::Paused, Some("Paused for memory"));
                    self.set_state(id, JobState::Queued, None);
                    self.set_state(id, JobState::WaitingForResources, Some("Waiting for memory to be freed"));
                    self.queue.pressure_held.lock().push(id.to_string());
                }
                ResultKind::Paused => self.set_state(id, JobState::Paused, Some(&r.message)),
                ResultKind::Resources => {
                    let _ = self.db.set_job_error(id, &r.message);
                    self.set_state(id, JobState::Paused, Some(&r.message));
                    self.emit(AppEvent::Notice { level: "warning".into(), message: r.message.clone() });
                }
                ResultKind::Cancelled => {
                    self.set_state(id, JobState::Cancelled, Some("Cancelled"));
                    let _ = std::fs::remove_dir_all(self.paths().job_dir(id));
                }
                ResultKind::Invalid | ResultKind::Failed => {
                    let _ = self.db.set_job_error(id, &r.message);
                    self.set_state(id, JobState::Failed, Some(&r.message));
                    if kind == JobKind::Preview {
                        self.emit(AppEvent::PreviewFailed { key: self.db.job(id).map(|j| j.name).unwrap_or_default(), message: r.message.clone() });
                    }
                    if s.stop_queue_on_error && kind == JobKind::Render {
                        self.pause_queue();
                    }
                }
            },
            None => {
                let msg = format!(
                    "The processing worker stopped unexpectedly (exit code {}). This usually means the system ran out of memory or a GPU driver failed. Finished segments were kept; resume to continue.",
                    code.map(|c| c.to_string()).unwrap_or_else(|| "none".into())
                );
                let _ = self.db.set_job_error(id, &msg);
                self.set_state(id, JobState::Recoverable, Some(&msg));
                if kind == JobKind::Preview {
                    self.emit(AppEvent::PreviewFailed { key: self.db.job(id).map(|j| j.name).unwrap_or_default(), message: msg });
                }
            }
        }
        if s.notifications && kind == JobKind::Render {
            if let Ok(j) = self.db.job(id) {
                if j.state.is_terminal() {
                    self.emit(AppEvent::Notice { level: if j.state == JobState::Completed { "success".into() } else { "info".into() }, message: format!("{}: {}", j.name, j.state.as_str()) });
                }
            }
        }
    }

    // ---- controls ----

    pub fn pause_queue(&self) {
        self.queue.paused.store(true, Ordering::SeqCst);
        self.queue_state_event();
    }

    pub fn resume_queue(&self) {
        self.queue.paused.store(false, Ordering::SeqCst);
        self.queue.wake();
        self.queue_state_event();
    }

    /// Pause: running jobs stop at a safe point (finished segments kept); queued jobs are held.
    pub fn pause_job(&self, id: &str) -> Result<()> {
        if self.queue.send(id, "pause") {
            return Ok(());
        }
        let j = self.db.job(id).map_err(|e| anyhow!("{e}"))?;
        if matches!(j.state, JobState::Queued | JobState::WaitingForResources) {
            self.set_state(id, JobState::Paused, Some("Paused"));
            return Ok(());
        }
        bail!("the job cannot be paused in its current state ({})", j.state.as_str())
    }

    /// Resume / retry: back to the queue; resumable work is reused.
    pub fn resume_job(&self, id: &str) -> Result<()> {
        let j = self.db.job(id).map_err(|e| anyhow!("{e}"))?;
        if !matches!(j.state, JobState::Paused | JobState::Recoverable | JobState::Failed | JobState::Cancelled) {
            bail!("the job is {}", j.state.as_str());
        }
        self.db.set_job_state(id, JobState::Queued, Some("Queued")).map_err(|e| anyhow!("{e}"))?;
        self.update_job_event(id);
        self.queue.wake();
        Ok(())
    }

    pub fn cancel_job(&self, id: &str) -> Result<()> {
        if self.queue.send(id, "cancel") {
            return Ok(());
        }
        let j = self.db.job(id).map_err(|e| anyhow!("{e}"))?;
        if j.state.is_terminal() {
            return Ok(());
        }
        self.set_state(id, JobState::Cancelled, Some("Cancelled"));
        let _ = std::fs::remove_dir_all(self.paths().job_dir(id));
        Ok(())
    }

    /// Remove a job from the list (not while running). Its work files are deleted; outputs are kept.
    pub fn remove_job(&self, id: &str) -> Result<()> {
        if self.queue.running.lock().contains_key(id) {
            bail!("cancel or pause the job before removing it");
        }
        self.db.delete_job(id).map_err(|e| anyhow!("{e}"))?;
        let _ = std::fs::remove_dir_all(self.paths().job_dir(id));
        self.emit(AppEvent::JobRemoved { id: id.to_string() });
        self.queue_state_event();
        Ok(())
    }

    pub fn reorder_job(&self, id: &str, index: usize) -> Result<()> {
        self.db.reorder_job(id, index).map_err(|e| anyhow!("{e}"))?;
        self.queue_state_event();
        Ok(())
    }

    pub fn clear_finished_jobs(&self) -> Result<usize> {
        let n = self.db.clear_finished_jobs().map_err(|e| anyhow!("{e}"))?;
        self.enforce_cache_policy();
        Ok(n)
    }

    /// Pause all running jobs and wait (up to `timeout`) for them to stop — used before quitting.
    pub fn pause_all_and_wait(&self, timeout: Duration) {
        let ids: Vec<String> = self.queue.running.lock().keys().cloned().collect();
        for id in &ids {
            self.queue.send(id, "pause");
        }
        let t = Instant::now();
        while self.running_jobs() > 0 && t.elapsed() < timeout {
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}
