//! ClipFX Enhance application services.
//!
//! [`App`] owns the database, settings, FFmpeg tools, model manager and job
//! queue. The desktop shell and the CLI are thin layers over it.

pub mod diagnostics;
pub mod events;
pub mod export;
pub mod ipc;
pub mod logging;
pub mod models;
pub mod presets;
pub mod preview;
pub mod queue;
pub mod resources;
pub mod storage;
pub mod updates;
pub mod worker;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use parking_lot::RwLock;

use clearframe_core::db::Db;
use clearframe_core::paths::AppPaths;
use clearframe_core::settings::Settings;
use clearframe_hardware::HardwareInfo;
use clearframe_media::FfmpegTools;
use clearframe_models::catalog::Catalog;
use clearframe_models::manager::ModelManager;

pub use events::{AppEvent, EventSink};

/// How to start a worker process for jobs.
#[derive(Debug, Clone)]
pub struct WorkerCommand {
    pub program: PathBuf,
    /// Arguments placed before the job file path (e.g. `["--worker"]`).
    pub args: Vec<String>,
}

impl WorkerCommand {
    /// The current executable with `--worker`.
    pub fn current_exe() -> Self {
        Self { program: std::env::current_exe().unwrap_or_else(|_| PathBuf::from("clipfx")), args: vec!["--worker".into()] }
    }
}

pub struct App {
    paths: RwLock<AppPaths>,
    pub db: Arc<Db>,
    settings: RwLock<Settings>,
    tools: RwLock<Option<FfmpegTools>>,
    tools_error: RwLock<Option<String>>,
    /// FFmpeg is still being detected in the background (see [`App::open_background_tools`]).
    tools_pending: parking_lot::Mutex<bool>,
    tools_ready: parking_lot::Condvar,
    pub models: Arc<ModelManager>,
    hardware: RwLock<Option<HardwareInfo>>,
    /// Held while hardware is detected, so concurrent callers wait for one detection.
    hardware_detect: parking_lot::Mutex<()>,
    /// Recent probe results by file, reused while the file is unchanged (size, modified time).
    probes: parking_lot::Mutex<std::collections::HashMap<PathBuf, ProbeEntry>>,
    pub events: EventSink,
    pub queue: queue::Queue,
    pub installs: models::Installs,
    /// Extra directories searched for bundled ffmpeg/ffprobe.
    pub bundled_tool_dirs: Vec<PathBuf>,
    /// A storage relocation is running: no new jobs or cache writes start.
    pub(crate) storage_busy: std::sync::atomic::AtomicBool,
    pub(crate) storage_cancel: std::sync::atomic::AtomicBool,
    pub(crate) monitor: resources::SystemMonitor,
}

impl App {
    /// Open (or create) the application state.
    pub fn open(root: Option<&Path>, worker: WorkerCommand, bundled_tool_dirs: Vec<PathBuf>, events: EventSink) -> Result<Arc<Self>> {
        let app = Self::open_inner(root, worker, bundled_tool_dirs, events)?;
        app.refresh_tools();
        Ok(app)
    }

    /// Like [`App::open`], but detects FFmpeg on a background thread so a window can show at once
    /// (detection runs several FFmpeg commands, which can take seconds on a slow or busy machine).
    /// Until it finishes, [`App::tools`] waits for it and [`App::tools_pending`] is true;
    /// [`AppEvent::ToolsChanged`] is sent when it is done.
    pub fn open_background_tools(root: Option<&Path>, worker: WorkerCommand, bundled_tool_dirs: Vec<PathBuf>, events: EventSink) -> Result<Arc<Self>> {
        let app = Self::open_inner(root, worker, bundled_tool_dirs, events)?;
        // Use what was detected last time when FFmpeg has not changed since, so it is ready at
        // once; the background check below still runs and corrects anything that changed.
        match app.cached_tools() {
            Some(t) => *app.tools.write() = Some(t),
            None => *app.tools_pending.lock() = true,
        }
        let a = app.clone();
        std::thread::Builder::new().name("ffmpeg-detect".into()).spawn(move || a.refresh_tools())?;
        // Hardware too (cores, memory, GPUs), so the first export check doesn't wait for it.
        let a = app.clone();
        std::thread::Builder::new().name("hardware-detect".into()).spawn(move || {
            a.hardware();
        })?;
        Ok(app)
    }

    fn open_inner(root: Option<&Path>, worker: WorkerCommand, bundled_tool_dirs: Vec<PathBuf>, events: EventSink) -> Result<Arc<Self>> {
        let paths = AppPaths::resolve(root);
        std::fs::create_dir_all(&paths.data_dir)?;
        // Finish or roll back a storage move interrupted by a crash.
        storage::recover_moves(&paths);
        paths.ensure()?;
        let db = Arc::new(Db::open(&paths.database()).map_err(|e| anyhow!("{e}"))?);
        let settings = db.load_settings().map_err(|e| anyhow!("{e}"))?;
        let catalog = models::effective_catalog(&db);
        let models = Arc::new(ModelManager::new(db.clone(), paths.models_dir.clone(), paths.cache(clearframe_core::paths::CacheArea::Downloads), catalog));
        let app = Arc::new(Self {
            paths: RwLock::new(paths),
            db: db.clone(),
            settings: RwLock::new(settings),
            tools: RwLock::new(None),
            tools_error: RwLock::new(None),
            tools_pending: parking_lot::Mutex::new(false),
            tools_ready: parking_lot::Condvar::new(),
            models,
            hardware: RwLock::new(None),
            hardware_detect: parking_lot::Mutex::new(()),
            probes: Default::default(),
            events: events.clone(),
            queue: queue::Queue::new(worker),
            installs: models::Installs::default(),
            bundled_tool_dirs,
            storage_busy: Default::default(),
            storage_cancel: Default::default(),
            monitor: resources::SystemMonitor::default(),
        });
        presets::seed_builtins(&app.db)?;
        let recovered = db.recover_interrupted().map_err(|e| anyhow!("{e}"))?;
        if !recovered.is_empty() {
            log::warn!("{} job(s) were interrupted and can be resumed", recovered.len());
        }
        Ok(app)
    }

    /// Current storage locations.
    pub fn paths(&self) -> AppPaths {
        self.paths.read().clone()
    }

    pub(crate) fn set_paths(&self, p: AppPaths) {
        self.models.set_locations(p.models_dir.clone(), p.cache(clearframe_core::paths::CacheArea::Downloads));
        *self.paths.write() = p;
    }

    pub fn settings(&self) -> Settings {
        self.settings.read().clone()
    }

    pub fn save_settings(&self, mut s: Settings) -> Result<Settings> {
        s.sanitize();
        let ffmpeg_changed = s.ffmpeg_dir != self.settings.read().ffmpeg_dir;
        self.db.save_settings(&s).map_err(|e| anyhow!("{e}"))?;
        *self.settings.write() = s.clone();
        if ffmpeg_changed {
            self.refresh_tools();
        }
        Ok(s)
    }

    /// Re-discover FFmpeg.
    pub fn refresh_tools(&self) {
        let s = self.settings();
        let before = self.tools.read().as_ref().map(|t| (t.ffmpeg.clone(), t.version.clone()));
        match FfmpegTools::discover(s.ffmpeg_dir.as_deref(), &self.bundled_tool_dirs) {
            Ok(t) => {
                log::info!("using ffmpeg {} from {}", t.version, t.ffmpeg.display());
                self.save_tools_cache(&t);
                *self.tools.write() = Some(t);
                *self.tools_error.write() = None;
            }
            Err(e) => {
                log::warn!("ffmpeg not found: {e}");
                let _ = std::fs::remove_file(self.tools_cache_path());
                *self.tools.write() = None;
                *self.tools_error.write() = Some(e.to_string());
            }
        }
        let changed = before != self.tools.read().as_ref().map(|t| (t.ffmpeg.clone(), t.version.clone()));
        let mut pending = self.tools_pending.lock();
        let was_pending = *pending;
        *pending = false;
        self.tools_ready.notify_all();
        drop(pending);
        if was_pending || changed {
            (self.events)(AppEvent::ToolsChanged);
        }
    }

    fn tools_cache_path(&self) -> PathBuf {
        self.paths().data_dir.join("ffmpeg-detected.json")
    }

    /// The FFmpeg found last time, if this app version detected it and the configured folder and
    /// both executables (size and modification time) are unchanged.
    fn cached_tools(&self) -> Option<FfmpegTools> {
        let c: ToolsCache = serde_json::from_slice(&std::fs::read(self.tools_cache_path()).ok()?).ok()?;
        let fresh = c.app_version == env!("CARGO_PKG_VERSION")
            && c.ffmpeg_dir == self.settings().ffmpeg_dir
            && c.ffmpeg_stamp == file_stamp(&c.tools.ffmpeg)?
            && c.ffprobe_stamp == file_stamp(&c.tools.ffprobe)?;
        fresh.then_some(c.tools)
    }

    fn save_tools_cache(&self, t: &FfmpegTools) {
        let (Some(ffmpeg_stamp), Some(ffprobe_stamp)) = (file_stamp(&t.ffmpeg), file_stamp(&t.ffprobe)) else { return };
        let c = ToolsCache { app_version: env!("CARGO_PKG_VERSION").into(), ffmpeg_dir: self.settings().ffmpeg_dir, ffmpeg_stamp, ffprobe_stamp, tools: t.clone() };
        // Write then rename, so a crash never leaves a half-written file behind.
        let path = self.tools_cache_path();
        let tmp = path.with_extension("json.part");
        if serde_json::to_vec(&c).ok().is_some_and(|b| std::fs::write(&tmp, b).is_ok()) {
            let _ = std::fs::rename(&tmp, &path);
        }
    }

    /// FFmpeg detection started by [`App::open_background_tools`] has not finished yet.
    pub fn tools_pending(&self) -> bool {
        *self.tools_pending.lock()
    }

    pub fn tools(&self) -> Result<FfmpegTools> {
        let mut pending = self.tools_pending.lock();
        while *pending {
            if self.tools_ready.wait_for(&mut pending, std::time::Duration::from_secs(120)).timed_out() {
                return Err(anyhow!("FFmpeg detection is taking too long"));
            }
        }
        drop(pending);
        self.tools.read().clone().ok_or_else(|| anyhow!(self.tools_error.read().clone().unwrap_or_else(|| "FFmpeg was not found".into())))
    }

    pub fn tools_status(&self) -> (Option<FfmpegTools>, Option<String>) {
        (self.tools.read().clone(), self.tools_error.read().clone())
    }

    /// Hardware info (cached after the first call).
    pub fn hardware(&self) -> HardwareInfo {
        if let Some(h) = self.hardware.read().clone() {
            return h;
        }
        let _detecting = self.hardware_detect.lock();
        if let Some(h) = self.hardware.read().clone() {
            return h; // another caller finished detecting while we waited
        }
        let h = clearframe_hardware::detect();
        *self.hardware.write() = Some(h.clone());
        h
    }

    /// Probe a video, reusing the last result while the file is unchanged. Opening a video, every
    /// export check and every enqueue need it, and each probe runs several FFprobe commands.
    pub fn probe_cached(&self, tools: &FfmpegTools, path: &Path) -> Result<clearframe_media::probe::MediaInfo> {
        let stamp = file_stamp(path);
        if let Some(st) = stamp {
            if let Some((s, info)) = self.probes.lock().get(path) {
                if *s == st {
                    return Ok(info.clone());
                }
            }
        }
        let info = clearframe_media::probe::probe(tools, path)?;
        if let Some(st) = stamp {
            let mut m = self.probes.lock();
            if m.len() >= 64 {
                m.clear();
            }
            m.insert(path.to_path_buf(), (st, info.clone()));
        }
        Ok(info)
    }

    pub fn refresh_hardware(&self) -> HardwareInfo {
        *self.hardware.write() = None;
        self.hardware()
    }

    pub fn catalog(&self) -> Catalog {
        self.models.catalog()
    }

    pub fn emit(&self, e: AppEvent) {
        (self.events)(e);
    }
}

/// A cached probe: the file's (size, modified time) and what FFprobe reported.
type ProbeEntry = ((u64, i64), clearframe_media::probe::MediaInfo);

/// What [`App::cached_tools`] keeps between launches.
#[derive(serde::Serialize, serde::Deserialize)]
struct ToolsCache {
    app_version: String,
    ffmpeg_dir: Option<PathBuf>,
    ffmpeg_stamp: (u64, i64),
    ffprobe_stamp: (u64, i64),
    tools: FfmpegTools,
}

/// Size and modification time (seconds) of a file, to notice when it was replaced.
fn file_stamp(p: &Path) -> Option<(u64, i64)> {
    let m = std::fs::metadata(p).ok()?;
    let t = m.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_secs() as i64;
    Some((m.len(), t))
}
