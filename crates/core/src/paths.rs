//! Centralised, platform-aware storage locations.
//!
//! Defaults (resolved with the OS known-folder APIs via the `dirs` crate —
//! `SHGetKnownFolderPath` on Windows, never hard-coded):
//!
//! | | Windows | macOS | Linux |
//! |---|---|---|---|
//! | data (DB, logs, models) | `%LOCALAPPDATA%\ClearFrame` | `~/Library/Application Support/ClearFrame` | `$XDG_DATA_HOME/ClearFrame` (`~/.local/share/ClearFrame`) |
//! | cache | `%LOCALAPPDATA%\ClearFrame\cache` | `~/Library/Caches/ClearFrame` | `$XDG_CACHE_HOME/ClearFrame` (`~/.cache/ClearFrame`) |
//! | storage config | `%LOCALAPPDATA%\ClearFrame\storage.json` | `~/Library/Application Support/ClearFrame/storage.json` | `$XDG_CONFIG_HOME/ClearFrame/storage.json` |
//!
//! The model and cache locations are user-configurable and recorded in
//! `storage.json`, which lives outside both so it survives relocations.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub const APP_DIR: &str = "ClearFrame";
pub const EXPORTS_DIR: &str = "ClearFrame Exports";
pub const STORAGE_FILE: &str = "storage.json";

/// User-chosen locations (None = platform default).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct StorageConfig {
    pub models_dir: Option<PathBuf>,
    pub cache_dir: Option<PathBuf>,
}

/// Cache sub-areas. Everything here is disposable except `jobs` entries that
/// belong to unfinished (queued/paused/recoverable) jobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "snake_case")]
pub enum CacheArea {
    /// Rendered preview clips and stills.
    Previews,
    /// Timeline thumbnails.
    Thumbnails,
    /// Browser-playable proxies of sources.
    Proxies,
    /// Model download staging (partial downloads resume from here).
    Downloads,
    /// Short-lived temporary files (frame extraction, analysis).
    Temp,
    /// Per-job work folders: segments, intermediates, recovery state.
    Jobs,
}

impl CacheArea {
    pub const ALL: [CacheArea; 6] = [CacheArea::Previews, CacheArea::Thumbnails, CacheArea::Proxies, CacheArea::Downloads, CacheArea::Temp, CacheArea::Jobs];
    pub fn dir_name(self) -> &'static str {
        match self {
            CacheArea::Previews => "previews",
            CacheArea::Thumbnails => "thumbnails",
            CacheArea::Proxies => "proxies",
            CacheArea::Downloads => "downloads",
            CacheArea::Temp => "temp",
            CacheArea::Jobs => "jobs",
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            CacheArea::Previews => "Preview renders",
            CacheArea::Thumbnails => "Thumbnails",
            CacheArea::Proxies => "Playback proxies",
            CacheArea::Downloads => "Partial model downloads",
            CacheArea::Temp => "Temporary files",
            CacheArea::Jobs => "Job work files (segments, intermediates)",
        }
    }
}

/// Resolved locations used by the application.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AppPaths {
    /// Database, logs.
    pub data_dir: PathBuf,
    /// Installed models.
    pub models_dir: PathBuf,
    /// All managed cache and temporary processing data.
    pub cache_dir: PathBuf,
    pub logs_dir: PathBuf,
    /// Directory holding `storage.json`.
    pub config_dir: PathBuf,
    pub models_is_default: bool,
    pub cache_is_default: bool,
}

/// Platform default directories `(data, cache, config)`.
pub fn platform_defaults() -> (PathBuf, PathBuf, PathBuf) {
    let fallback = || std::env::temp_dir().join(APP_DIR);
    let data = dirs::data_local_dir().map(|d| d.join(APP_DIR)).unwrap_or_else(fallback);
    let cache = if cfg!(windows) {
        // %LOCALAPPDATA%\ClearFrame\cache
        data.join("cache")
    } else {
        dirs::cache_dir().map(|d| d.join(APP_DIR)).unwrap_or_else(|| data.join("cache"))
    };
    let config = if cfg!(target_os = "linux") { dirs::config_dir().map(|d| d.join(APP_DIR)).unwrap_or_else(|| data.clone()) } else { data.clone() };
    (data, cache, config)
}

impl AppPaths {
    /// Resolve locations. With `root`, everything lives under it (portable mode and tests).
    pub fn resolve(root: Option<&Path>) -> Self {
        let (data, cache, config) = match root {
            Some(r) => (r.join("data"), r.join("cache"), r.join("config")),
            None => platform_defaults(),
        };
        let cfg = load_storage_config(&config);
        Self::from_parts(data, cache, config, &cfg)
    }

    fn from_parts(data: PathBuf, default_cache: PathBuf, config: PathBuf, cfg: &StorageConfig) -> Self {
        Self {
            models_dir: cfg.models_dir.clone().unwrap_or_else(|| data.join("models")),
            cache_dir: cfg.cache_dir.clone().unwrap_or(default_cache),
            logs_dir: data.join("logs"),
            models_is_default: cfg.models_dir.is_none(),
            cache_is_default: cfg.cache_dir.is_none(),
            config_dir: config,
            data_dir: data,
        }
    }

    pub fn default_models_dir(&self) -> PathBuf {
        self.data_dir.join("models")
    }

    /// Create directories that must exist. A missing *custom* models or cache
    /// directory is not silently recreated elsewhere; it is created in place
    /// (the user can pick another location in Settings).
    pub fn ensure(&self) -> std::io::Result<()> {
        for d in [&self.data_dir, &self.logs_dir, &self.config_dir] {
            std::fs::create_dir_all(d)?;
        }
        std::fs::create_dir_all(&self.cache_dir)?;
        for a in CacheArea::ALL {
            std::fs::create_dir_all(self.cache(a))?;
        }
        Ok(())
    }

    pub fn database(&self) -> PathBuf {
        self.data_dir.join("clearframe.sqlite3")
    }

    pub fn storage_file(&self) -> PathBuf {
        self.config_dir.join(STORAGE_FILE)
    }

    pub fn cache(&self, area: CacheArea) -> PathBuf {
        self.cache_dir.join(area.dir_name())
    }

    /// Work folder for a job.
    pub fn job_dir(&self, job_id: &str) -> PathBuf {
        self.cache(CacheArea::Jobs).join(crate::fsutil::sanitize_component(job_id))
    }

    /// A new unique temporary directory under the managed temp area.
    pub fn temp_subdir(&self, prefix: &str) -> std::io::Result<PathBuf> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let d = self.cache(CacheArea::Temp).join(format!("{prefix}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir_all(&d)?;
        Ok(d)
    }

    pub fn storage_config(&self) -> StorageConfig {
        StorageConfig { models_dir: (!self.models_is_default).then(|| self.models_dir.clone()), cache_dir: (!self.cache_is_default).then(|| self.cache_dir.clone()) }
    }

    /// Persist new locations atomically.
    pub fn save_storage_config(&self, cfg: &StorageConfig) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.config_dir)?;
        let tmp = self.storage_file().with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(cfg).unwrap_or_default())?;
        std::fs::rename(&tmp, self.storage_file())
    }

    /// Re-resolve after a configuration change.
    pub fn with_config(&self, cfg: &StorageConfig) -> Self {
        let default_cache = if self.cache_is_default { self.cache_dir.clone() } else { Self::default_cache_for(&self.data_dir) };
        Self::from_parts(self.data_dir.clone(), default_cache, self.config_dir.clone(), cfg)
    }

    fn default_cache_for(data: &Path) -> PathBuf {
        let (d, c, _) = platform_defaults();
        if data == d {
            c
        } else {
            data.parent().map(|p| p.join("cache")).unwrap_or_else(|| data.join("cache"))
        }
    }

    /// Environment for child processes so libraries that honour TMPDIR/TEMP
    /// write into the managed temp area instead of the system drive.
    pub fn child_env(&self) -> Vec<(String, PathBuf)> {
        let t = self.cache(CacheArea::Temp);
        vec![("TMPDIR".into(), t.clone()), ("TEMP".into(), t.clone()), ("TMP".into(), t)]
    }
}

pub fn load_storage_config(config_dir: &Path) -> StorageConfig {
    std::fs::read(config_dir.join(STORAGE_FILE)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}

/// The documented default export folder: `<Videos>/ClearFrame Exports`, falling
/// back to Documents, then the home directory.
pub fn default_export_dir() -> PathBuf {
    dirs::video_dir().or_else(dirs::document_dir).or_else(dirs::home_dir).unwrap_or_else(std::env::temp_dir).join(EXPORTS_DIR)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portable_root_and_config_roundtrip() {
        let d = tempfile::tempdir().unwrap();
        let p = AppPaths::resolve(Some(d.path()));
        p.ensure().unwrap();
        assert!(p.models_dir.starts_with(d.path()));
        assert!(p.cache(CacheArea::Jobs).is_dir());
        assert!(p.job_dir("a/b").starts_with(p.cache(CacheArea::Jobs)));
        let custom = d.path().join("Mödels with spaces");
        let cfg = StorageConfig { models_dir: Some(custom.clone()), cache_dir: None };
        p.save_storage_config(&cfg).unwrap();
        let p2 = AppPaths::resolve(Some(d.path()));
        assert_eq!(p2.models_dir, custom);
        assert!(!p2.models_is_default);
        assert!(p2.cache_is_default);
        assert_eq!(p2.storage_config(), cfg);
        assert!(default_export_dir().ends_with(EXPORTS_DIR));
    }

    #[test]
    fn platform_defaults_are_per_user() {
        let (data, cache, config) = platform_defaults();
        assert!(data.ends_with(APP_DIR));
        assert!(cache.to_string_lossy().contains(APP_DIR));
        assert!(config.to_string_lossy().contains(APP_DIR));
        #[cfg(target_os = "linux")]
        {
            assert!(cache.to_string_lossy().contains(".cache") || std::env::var("XDG_CACHE_HOME").is_ok());
        }
    }
}
