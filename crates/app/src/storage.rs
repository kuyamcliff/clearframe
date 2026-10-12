//! Storage & cache services: overview, relocation of the models and cache
//! folders (journaled, cancellable, crash-recoverable), cache clearing and
//! cleanup policies, and opening folders in the OS file manager.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};

use clearframe_core::cache::{self, CacheReport, ClearResult};
use clearframe_core::fsutil;
use clearframe_core::paths::{AppPaths, CacheArea, StorageConfig};
use clearframe_core::relocate::{self, MovePlan};
use clearframe_models::manager::StorageReport;

use crate::{App, AppEvent};

const JOURNAL: &str = "storage-move.journal.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageKind {
    Models,
    Cache,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelocationMode {
    /// Copy everything, verify, switch, then delete the old copy.
    Move,
    /// Cache only: delete disposable cache first, then move what remains (resumable job data).
    ClearDisposableAndMove,
    /// Point at the new location without moving (the old folder is left untouched).
    SwitchOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageOverview {
    pub paths: AppPaths,
    pub models: StorageReport,
    pub cache: CacheReport,
    pub data_bytes: u64,
    pub data_free: Option<u64>,
    pub warnings: Vec<String>,
    pub relocation_in_progress: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelocationPlan {
    pub kind: StorageKind,
    pub plan: MovePlan,
    /// For the cache: bytes that can simply be deleted instead of moved.
    pub disposable_bytes: u64,
    pub notes: Vec<String>,
}

/// Finish or roll back an interrupted relocation (called before anything else on startup).
pub fn recover_moves(paths: &AppPaths) {
    let j = paths.config_dir.join(JOURNAL);
    let Some(journal) = relocate::read_journal(&j) else { return };
    let active = if journal.destination == paths.models_dir || journal.destination == paths.cache_dir { journal.destination.clone() } else { journal.source.clone() };
    let r = relocate::recover(&j, &active);
    log::warn!("recovered an interrupted storage move ({r:?}): {} -> {}", journal.source.display(), journal.destination.display());
}

/// Open a folder in Explorer / Finder / the desktop file manager.
pub fn open_in_file_manager(path: &Path) -> Result<()> {
    if !path.exists() {
        bail!("{} does not exist", path.display());
    }
    let (cmd, args): (&str, Vec<&std::ffi::OsStr>) = if cfg!(windows) {
        ("explorer", vec![path.as_os_str()])
    } else if cfg!(target_os = "macos") {
        ("open", vec![path.as_os_str()])
    } else {
        ("xdg-open", vec![path.as_os_str()])
    };
    std::process::Command::new(cmd).args(args).spawn().map_err(|e| anyhow!("could not open the file manager: {e}"))?;
    Ok(())
}

impl App {
    /// Job ids whose work folders must be kept (unfinished jobs), plus active downloads.
    pub fn protected_cache_entries(&self, include_resumable: bool) -> HashSet<String> {
        let mut p = HashSet::new();
        for j in self.db.jobs(true).unwrap_or_default() {
            let keep = if include_resumable { j.state.is_active() } else { !j.state.is_terminal() };
            if keep {
                p.insert(fsutil::sanitize_component(&j.id));
            }
        }
        for id in self.installs_running() {
            p.insert(fsutil::sanitize_component(&id));
        }
        p
    }

    pub fn storage_overview(&self) -> StorageOverview {
        let paths = self.paths();
        let s = self.settings();
        let cache = cache::report(&paths, &self.protected_cache_entries(false), s.low_disk_warning_mb);
        let models = self.models.storage();
        let mut warnings = Vec::new();
        // A default folder that was never used is created on first install; only warn when
        // a chosen folder disappeared or installed models went missing with it.
        let lost = !paths.models_is_default || self.models.status().iter().any(|m| m.state == clearframe_models::manager::ModelState::Missing);
        if !models.models_dir_exists && lost {
            warnings.push(format!("The models folder {} is missing. Choose its new location or reinstall models.", paths.models_dir.display()));
        }
        for (label, free) in [("cache", cache.free_bytes), ("models", models.free_bytes)] {
            if let Some(f) = free {
                if f < s.low_disk_warning_mb * 1_000_000 {
                    warnings.push(format!("The {label} drive is nearly full ({} MB free).", f / 1_000_000));
                }
            }
        }
        StorageOverview {
            data_bytes: fsutil::dir_size(&paths.data_dir)
                .saturating_sub(if paths.models_dir.starts_with(&paths.data_dir) { models.total_bytes } else { 0 })
                .saturating_sub(if paths.cache_dir.starts_with(&paths.data_dir) { cache.total_bytes } else { 0 }),
            data_free: fsutil::free_space(&paths.data_dir),
            relocation_in_progress: self.storage_busy.load(Ordering::SeqCst),
            paths,
            models,
            cache,
            warnings,
        }
    }

    pub fn plan_relocation(&self, kind: StorageKind, dest: &Path) -> RelocationPlan {
        let paths = self.paths();
        let src = match kind {
            StorageKind::Models => paths.models_dir.clone(),
            StorageKind::Cache => paths.cache_dir.clone(),
        };
        let plan = relocate::plan(&src, dest);
        let mut notes = Vec::new();
        let mut disposable = 0;
        match kind {
            StorageKind::Models => {
                notes.push("Installed models are copied and verified, then the old folder is deleted. Nothing is downloaded again.".into());
            }
            StorageKind::Cache => {
                for a in [CacheArea::Previews, CacheArea::Thumbnails, CacheArea::Proxies, CacheArea::Temp, CacheArea::Downloads] {
                    disposable += fsutil::dir_size(&paths.cache(a));
                }
                notes.push("Cache holds only regenerable data: previews, thumbnails, proxies, partial downloads and job work files. Your videos, exports, projects, presets and models are never stored there.".into());
                notes.push("Choose \"Clear and move\" to delete regenerable cache instead of copying it; resumable job data is still moved.".into());
            }
        }
        if self.running_jobs() > 0 {
            notes.push("Running jobs must be paused first; the move waits for no active work.".into());
        }
        RelocationPlan { kind, plan, disposable_bytes: disposable, notes }
    }

    /// Move (or switch) the models or cache folder in the background, reporting progress events.
    pub fn relocate(self: &Arc<Self>, kind: StorageKind, dest: PathBuf, mode: RelocationMode) -> Result<()> {
        if self.running_jobs() > 0 {
            bail!("Pause or wait for running jobs before moving storage.");
        }
        if kind == StorageKind::Models && !self.installs_running().is_empty() {
            bail!("Wait for model installs to finish before moving the models folder.");
        }
        if mode == RelocationMode::ClearDisposableAndMove && kind != StorageKind::Cache {
            bail!("Only the cache can be cleared during a move.");
        }
        if self.storage_busy.swap(true, Ordering::SeqCst) {
            bail!("Another storage move is already running.");
        }
        self.storage_cancel.store(false, Ordering::SeqCst);
        let app = self.clone();
        std::thread::Builder::new().name("storage-move".into()).spawn(move || {
            let r = app.relocate_blocking(kind, &dest, mode);
            app.storage_busy.store(false, Ordering::SeqCst);
            let (ok, message) = match r {
                Ok(m) => (true, m),
                Err(e) => (false, e.to_string()),
            };
            app.emit(AppEvent::StorageDone { kind: format!("{kind:?}").to_lowercase(), ok, message });
        })?;
        Ok(())
    }

    pub fn cancel_relocation(&self) {
        self.storage_cancel.store(true, Ordering::SeqCst);
    }

    /// Synchronous relocation (used by `relocate` and tests).
    pub fn relocate_blocking(&self, kind: StorageKind, dest: &Path, mode: RelocationMode) -> Result<String> {
        let paths = self.paths();
        let src = match kind {
            StorageKind::Models => paths.models_dir.clone(),
            StorageKind::Cache => paths.cache_dir.clone(),
        };
        let mut cfg: StorageConfig = paths.storage_config();
        let set = |cfg: &mut StorageConfig, p: PathBuf| match kind {
            StorageKind::Models => cfg.models_dir = Some(p),
            StorageKind::Cache => cfg.cache_dir = Some(p),
        };
        if mode == RelocationMode::SwitchOnly {
            std::fs::create_dir_all(dest)?;
            fsutil::ensure_writable_dir(dest, false).map_err(|e| anyhow!(e))?;
            set(&mut cfg, dest.to_path_buf());
            paths.save_storage_config(&cfg)?;
            let np = paths.with_config(&cfg);
            np.ensure()?;
            self.set_paths(np);
            return Ok(format!("Now using {}. The previous folder was left unchanged.", dest.display()));
        }
        if mode == RelocationMode::ClearDisposableAndMove {
            let r = cache::clear(
                &paths,
                &[CacheArea::Previews, CacheArea::Thumbnails, CacheArea::Proxies, CacheArea::Temp, CacheArea::Downloads],
                &self.protected_cache_entries(false),
            );
            log::info!("cleared {} MB of cache before moving", r.freed_bytes / 1_000_000);
        }
        let journal = paths.config_dir.join(JOURNAL);
        let events = self.events.clone();
        let k = format!("{kind:?}").to_lowercase();
        relocate::copy(&src, dest, &journal, &self.storage_cancel, &mut |done, total| {
            events(AppEvent::StorageProgress { kind: k.clone(), bytes_done: done, bytes_total: total });
        })
        .map_err(|e| anyhow!("{e}"))?;
        // Switch configuration only after a verified copy exists.
        set(&mut cfg, dest.to_path_buf());
        paths.save_storage_config(&cfg)?;
        let np = paths.with_config(&cfg);
        self.set_paths(np.clone());
        relocate::finish(&journal)?;
        np.ensure()?;
        Ok(format!("Moved to {}.", dest.display()))
    }

    /// Restore the default location setting for models or cache (does not move data).
    pub fn reset_storage_location(&self, kind: StorageKind) -> Result<()> {
        let paths = self.paths();
        let mut cfg = paths.storage_config();
        match kind {
            StorageKind::Models => cfg.models_dir = None,
            StorageKind::Cache => cfg.cache_dir = None,
        }
        paths.save_storage_config(&cfg)?;
        let np = AppPaths::resolve(if paths.data_dir.ends_with("data") && paths.config_dir.ends_with("config") { paths.data_dir.parent() } else { None });
        np.ensure()?;
        self.set_paths(np);
        Ok(())
    }

    /// Clear cache areas. Data of unfinished jobs is kept unless `include_resumable`.
    pub fn clear_cache(&self, areas: &[CacheArea], include_resumable: bool) -> ClearResult {
        let r = cache::clear(&self.paths(), areas, &self.protected_cache_entries(include_resumable));
        if include_resumable {
            // Jobs whose recovery data was removed restart from the beginning when resumed.
            for j in self.db.jobs(true).unwrap_or_default() {
                if !j.state.is_terminal() && !j.state.is_active() && !self.paths().job_dir(&j.id).exists() {
                    let _ = self.db.log_job_event(&j.id, "info", "Resumable work files were cleared; the job will restart from the beginning.");
                }
            }
        }
        r
    }

    /// Apply size/age policies and remove work folders of finished or deleted jobs.
    pub fn enforce_cache_policy(&self) -> ClearResult {
        let s = self.settings();
        let paths = self.paths();
        let mut r = cache::enforce_policy(&paths, s.cache_limit_mb, s.cache_max_age_days, &self.protected_cache_entries(false));
        let keep: HashSet<String> = self
            .db
            .jobs(true)
            .unwrap_or_default()
            .into_iter()
            .filter(|j| !j.state.is_terminal() || (s.keep_failed_work && j.state == clearframe_core::jobs::JobState::Failed))
            .map(|j| fsutil::sanitize_component(&j.id))
            .collect();
        let o = cache::remove_orphan_job_dirs(&paths, &keep);
        r.freed_bytes += o.freed_bytes;
        r.removed_entries += o.removed_entries;
        r
    }
}
