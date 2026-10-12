//! ClearFrame application services.
//!
//! [`App`] owns the database, settings, FFmpeg tools, model manager and job
//! queue. The desktop shell and the CLI are thin layers over it.

pub mod diagnostics;
pub mod events;
pub mod export;
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
        Self { program: std::env::current_exe().unwrap_or_else(|_| PathBuf::from("clearframe")), args: vec!["--worker".into()] }
    }
}

pub struct App {
    paths: RwLock<AppPaths>,
    pub db: Arc<Db>,
    settings: RwLock<Settings>,
    tools: RwLock<Option<FfmpegTools>>,
    tools_error: RwLock<Option<String>>,
    pub models: Arc<ModelManager>,
    hardware: RwLock<Option<HardwareInfo>>,
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
            models,
            hardware: RwLock::new(None),
            events: events.clone(),
            queue: queue::Queue::new(worker),
            installs: models::Installs::default(),
            bundled_tool_dirs,
            storage_busy: Default::default(),
            storage_cancel: Default::default(),
            monitor: resources::SystemMonitor::default(),
        });
        app.refresh_tools();
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
        match FfmpegTools::discover(s.ffmpeg_dir.as_deref(), &self.bundled_tool_dirs) {
            Ok(t) => {
                log::info!("using ffmpeg {} from {}", t.version, t.ffmpeg.display());
                *self.tools.write() = Some(t);
                *self.tools_error.write() = None;
            }
            Err(e) => {
                log::warn!("ffmpeg not found: {e}");
                *self.tools.write() = None;
                *self.tools_error.write() = Some(e.to_string());
            }
        }
    }

    pub fn tools(&self) -> Result<FfmpegTools> {
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
        let h = clearframe_hardware::detect();
        *self.hardware.write() = Some(h.clone());
        h
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
