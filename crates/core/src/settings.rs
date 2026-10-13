//! User preferences.
//!
//! Stored as a versioned JSON document in SQLite. Every field has a default so
//! settings written by older versions load cleanly; unknown fields from newer
//! versions are preserved in `extra` and written back unchanged.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

pub const SETTINGS_VERSION: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Theme {
    #[default]
    Dark,
    Light,
    System,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MotionPreference {
    /// Follow the operating system setting.
    #[default]
    System,
    Reduced,
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum QualityDefault {
    Fast,
    #[default]
    Balanced,
    Quality,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BackendPreference {
    #[default]
    Auto,
    Cpu,
    Cuda,
    Metal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FallbackPolicy {
    /// Ask before falling back to a slower backend.
    #[default]
    Ask,
    /// Fall back to CPU automatically.
    Allow,
    /// Never fall back; fail instead.
    Never,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub version: u32,
    // Appearance & accessibility
    pub theme: Theme,
    pub motion: MotionPreference,
    pub language: String,
    pub ui_scale: f32,
    // Export
    /// Default export folder. None = documented default (`Videos/ClipFX Enhance Exports`).
    pub output_dir: Option<PathBuf>,
    pub naming_template: String,
    pub write_sidecar_report: bool,
    pub remember_last_export_folder: bool,
    pub last_export_folder: Option<PathBuf>,
    // Preview
    pub preview_seconds: f32,
    pub preview_max_height: u32,
    pub quality_default: QualityDefault,
    // Processing
    pub backend: BackendPreference,
    pub fallback: FallbackPolicy,
    pub half_precision: bool,
    /// 0 = automatic.
    pub memory_limit_mb: u32,
    pub max_concurrent_jobs: u32,
    pub stop_queue_on_error: bool,
    pub segment_seconds: f32,
    pub keep_failed_work: bool,
    // Locations: model and cache folders live in storage.json (see paths.rs).
    pub ffmpeg_dir: Option<PathBuf>,
    // Cache policy
    /// Soft limit for disposable cache (previews, thumbnails, proxies, temp); 0 = no limit.
    pub cache_limit_mb: u64,
    /// Delete disposable cache entries not used for this many days; 0 = never.
    pub cache_max_age_days: u32,
    /// Warn when a drive used by ClipFX Enhance has less free space than this.
    pub low_disk_warning_mb: u64,
    // Performance
    pub performance: Performance,
    // Behaviour
    pub notifications: bool,
    pub inspector_mode_advanced: bool,
    pub confirm_before_quit_with_jobs: bool,
    /// Optional network features are off until the user turns them on.
    pub check_updates_on_launch: bool,
    pub allow_catalog_updates: bool,
    pub onboarding_complete: bool,
    pub shortcuts: BTreeMap<String, String>,
    /// Unknown keys from newer versions, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            version: SETTINGS_VERSION,
            theme: Theme::Dark,
            motion: MotionPreference::System,
            language: "en".into(),
            ui_scale: 1.0,
            output_dir: None,
            naming_template: "{name}_{task}".into(),
            write_sidecar_report: false,
            remember_last_export_folder: false,
            last_export_folder: None,
            preview_seconds: 3.0,
            preview_max_height: 1080,
            quality_default: QualityDefault::Balanced,
            backend: BackendPreference::Auto,
            fallback: FallbackPolicy::Ask,
            half_precision: false,
            memory_limit_mb: 0,
            max_concurrent_jobs: 1,
            stop_queue_on_error: false,
            segment_seconds: 10.0,
            keep_failed_work: true,
            ffmpeg_dir: None,
            cache_limit_mb: 10_240,
            cache_max_age_days: 30,
            low_disk_warning_mb: 5_000,
            performance: Performance::default(),
            notifications: true,
            inspector_mode_advanced: false,
            confirm_before_quit_with_jobs: true,
            check_updates_on_launch: true,
            allow_catalog_updates: false,
            onboarding_complete: false,
            shortcuts: default_shortcuts(),
            extra: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PerformanceMode {
    /// Keep the computer responsive: leaves CPU cores and memory for other apps, lower priority.
    #[default]
    Balanced,
    /// Favour ClipFX Enhance while keeping basic responsiveness.
    HighPerformance,
    /// Use everything ClipFX Enhance can; other applications may become slow.
    Maximum,
    /// Use the explicit values below.
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ProcessPriority {
    Low,
    #[default]
    BelowNormal,
    Normal,
    AboveNormal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemoryPreset {
    /// Leave about half of the memory for other applications.
    MoreForOthers,
    /// Leave about a quarter (at least 4 GB) for other applications.
    #[default]
    Balanced,
    /// Leave about 2 GB for the system.
    PrioritizeApp,
    /// Use `memory_budget_mb`.
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemoryPressureAction {
    /// Pause running jobs at a safe point until memory frees up, then continue.
    #[default]
    Pause,
    /// Keep going (the OS may swap and slow down).
    Continue,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Performance {
    pub mode: PerformanceMode,
    /// Custom: worker threads for inference/decoding (0 = automatic).
    pub cpu_threads: u32,
    /// Custom: CPU threads kept free for the rest of the system.
    pub reserve_cpu_threads: u32,
    pub priority: ProcessPriority,
    pub memory_preset: MemoryPreset,
    /// Custom memory budget for processing (MB).
    pub memory_budget_mb: u64,
    pub memory_pressure: MemoryPressureAction,
    /// Run more than one job at a time when the budget allows.
    pub max_concurrent_jobs: u32,
    /// Compute device: "auto", "cpu", "gpu:0" (any GPU via Vulkan/DX12/Metal), "cuda:0", "metal:0".
    pub device: String,
    /// Limit for GPU memory used by models (MB, 0 = automatic).
    pub vram_budget_mb: u64,
    /// Retry with smaller tiles when a device runs out of memory.
    pub oom_tile_fallback: bool,
}

impl Default for Performance {
    fn default() -> Self {
        Self {
            // Use the machine fully by default: every core at normal priority, the GPU and all
            // but about 2 GB of the memory.
            mode: PerformanceMode::Maximum,
            cpu_threads: 0,
            reserve_cpu_threads: 2,
            priority: ProcessPriority::Normal,
            memory_preset: MemoryPreset::PrioritizeApp,
            memory_budget_mb: 8192,
            memory_pressure: MemoryPressureAction::Pause,
            max_concurrent_jobs: 1,
            device: "auto".into(),
            vram_budget_mb: 0,
            oom_tile_fallback: true,
        }
    }
}

/// Concrete limits derived from `Performance` and the machine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourcePlan {
    pub threads: u32,
    pub priority: ProcessPriority,
    pub memory_budget_mb: u64,
    pub reserve_memory_mb: u64,
    pub max_concurrent_jobs: u32,
    pub explanation: Vec<String>,
}

impl Performance {
    /// Resolve the effective limits for a machine with `cores` logical CPUs and `ram_mb` memory.
    pub fn resolve(&self, cores: u32, ram_mb: u64) -> ResourcePlan {
        let cores = cores.max(1);
        let mut why = Vec::new();
        let (threads, priority) = match self.mode {
            PerformanceMode::Balanced => {
                let t = if cores <= 2 { 1 } else { (cores * 3 / 4).min(cores - 1).max(1) };
                why.push(format!("Balanced: {t} of {cores} CPU threads, below-normal priority so the desktop stays responsive."));
                (t, ProcessPriority::BelowNormal)
            }
            PerformanceMode::HighPerformance => {
                let t = (cores - 1).max(1);
                why.push(format!("High performance: {t} of {cores} CPU threads (one kept free), normal priority."));
                (t, ProcessPriority::Normal)
            }
            PerformanceMode::Maximum => {
                why.push(format!("Maximum performance: all {cores} CPU threads, normal priority. Other applications may become slower."));
                (cores, ProcessPriority::Normal)
            }
            PerformanceMode::Custom => {
                let avail = cores.saturating_sub(self.reserve_cpu_threads).max(1);
                let t = if self.cpu_threads == 0 { avail } else { self.cpu_threads.min(cores).max(1) };
                why.push(format!("Custom: {t} CPU threads, {} kept free.", cores - t.min(cores)));
                (t, self.priority)
            }
        };
        let balanced_reserve = (ram_mb / 4).max(4096.min(ram_mb / 2));
        let reserve = match self.memory_preset {
            MemoryPreset::MoreForOthers => ram_mb / 2,
            MemoryPreset::Balanced => balanced_reserve,
            MemoryPreset::PrioritizeApp => 2048.min(ram_mb / 4),
            // A small budget means "use less", not "keep everything else free": without the
            // cap, 600 MB on an 8 GB machine would pause jobs whenever less than ~3.7 GB
            // was free and never resume them.
            MemoryPreset::Custom => ram_mb.saturating_sub(self.memory_budget_mb).min(balanced_reserve),
        };
        let budget = match self.memory_preset {
            MemoryPreset::Custom => self.memory_budget_mb.min(ram_mb).max(512),
            _ => ram_mb.saturating_sub(reserve).max(512),
        };
        why.push(format!("Processing memory budget about {} MB; about {} MB left for other applications. The budget is a target: the OS can still page memory.", budget, reserve));
        ResourcePlan { threads, priority, memory_budget_mb: budget, reserve_memory_mb: reserve, max_concurrent_jobs: self.max_concurrent_jobs.clamp(1, 8), explanation: why }
    }
}

pub fn default_shortcuts() -> BTreeMap<String, String> {
    [
        ("open", "CmdOrCtrl+O"),
        ("export", "CmdOrCtrl+E"),
        ("quick_export", "CmdOrCtrl+Shift+E"),
        ("preview", "CmdOrCtrl+P"),
        ("play_pause", "Space"),
        ("frame_back", "ArrowLeft"),
        ("frame_forward", "ArrowRight"),
        ("set_in", "I"),
        ("set_out", "O"),
        ("compare_toggle", "C"),
        ("zoom_fit", "CmdOrCtrl+0"),
        ("zoom_100", "CmdOrCtrl+1"),
        ("toggle_queue", "CmdOrCtrl+J"),
        ("settings", "CmdOrCtrl+Comma"),
        ("undo", "CmdOrCtrl+Z"),
        ("redo", "CmdOrCtrl+Shift+Z"),
    ]
    .into_iter()
    .map(|(a, b)| (a.to_string(), b.to_string()))
    .collect()
}

impl Settings {
    /// Clamp values into supported ranges (applied on load and save).
    pub fn sanitize(&mut self) {
        // Version 2 made Maximum the default: settings still on the old default move to it once.
        if self.version < 2 && self.performance.mode == PerformanceMode::HighPerformance {
            self.performance.mode = PerformanceMode::Maximum;
        }
        self.ui_scale = self.ui_scale.clamp(0.75, 2.0);
        self.preview_seconds = self.preview_seconds.clamp(0.5, 30.0);
        self.preview_max_height = self.preview_max_height.clamp(240, 4320);
        self.max_concurrent_jobs = self.max_concurrent_jobs.clamp(1, 8);
        self.performance.max_concurrent_jobs = self.performance.max_concurrent_jobs.clamp(1, 8);
        self.performance.memory_budget_mb = self.performance.memory_budget_mb.clamp(512, 4 << 20);
        let d = self.performance.device.trim().to_ascii_lowercase();
        let valid = d == "auto" || d == "cpu" || ["gpu:", "cuda:", "metal:"].iter().any(|p| d.strip_prefix(p).is_some_and(|n| n.parse::<u32>().is_ok()));
        self.performance.device = if valid { d } else { "auto".into() };
        self.segment_seconds = self.segment_seconds.clamp(2.0, 120.0);
        if self.naming_template.trim().is_empty() || self.naming_template.len() > 200 {
            self.naming_template = "{name}_{task}".into();
        }
        if self.language.is_empty() {
            self.language = "en".into();
        }
        for (k, v) in default_shortcuts() {
            self.shortcuts.entry(k).or_insert(v);
        }
        self.version = SETTINGS_VERSION;
    }

    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        let mut v: Settings = serde_json::from_str(s)?;
        v.sanitize();
        Ok(v)
    }
}

/// What a settings reset should remove. Models and videos are never touched
/// unless explicitly requested.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResetScope {
    pub preferences: bool,
    pub recent_files: bool,
    pub job_history: bool,
    pub cache: bool,
    pub user_presets: bool,
    pub models: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_partial_load() {
        let s = Settings::from_json(r#"{"theme":"light","preview_seconds":999}"#).unwrap();
        assert_eq!(s.theme, Theme::Light);
        assert_eq!(s.preview_seconds, 30.0);
        assert_eq!(s.naming_template, "{name}_{task}");
        assert!(s.shortcuts.contains_key("play_pause"));
    }

    #[test]
    fn performance_modes() {
        // Out of the box the app uses every thread and nearly all memory.
        let r = Performance::default().resolve(16, 32768);
        assert_eq!((r.threads, r.priority, r.memory_budget_mb), (16, ProcessPriority::Normal, 32768 - 2048));
        // Settings saved on the old default (High performance) move to Maximum once; later choices stay.
        let s = Settings::from_json(r#"{"version":1,"performance":{"mode":"high_performance"}}"#).unwrap();
        assert_eq!(s.performance.mode, PerformanceMode::Maximum);
        let s = Settings::from_json(r#"{"version":2,"performance":{"mode":"high_performance"}}"#).unwrap();
        assert_eq!(s.performance.mode, PerformanceMode::HighPerformance);
        let mut p = Performance { mode: PerformanceMode::Balanced, memory_preset: MemoryPreset::Balanced, ..Performance::default() };
        let r = p.resolve(16, 32768);
        assert_eq!(r.threads, 12);
        assert_eq!(r.priority, ProcessPriority::BelowNormal);
        assert_eq!(r.memory_budget_mb, 32768 - 8192);
        p.mode = PerformanceMode::Maximum;
        p.memory_preset = MemoryPreset::PrioritizeApp;
        let r = p.resolve(16, 32768);
        assert_eq!(r.threads, 16);
        assert_eq!(r.memory_budget_mb, 32768 - 2048);
        p.mode = PerformanceMode::Custom;
        p.cpu_threads = 0;
        p.reserve_cpu_threads = 4;
        p.memory_preset = MemoryPreset::Custom;
        p.memory_budget_mb = 6000;
        let r = p.resolve(8, 16000);
        assert_eq!((r.threads, r.memory_budget_mb, r.reserve_memory_mb), (4, 6000, 4096));
        // A small custom budget must not demand that nearly all memory stays free.
        p.memory_budget_mb = 600;
        let r = p.resolve(8, 7168);
        assert_eq!((r.memory_budget_mb, r.reserve_memory_mb), (600, 3584));
        // A budget close to all RAM still leaves what's outside it.
        p.memory_budget_mb = 15000;
        assert_eq!(p.resolve(8, 16000).reserve_memory_mb, 1000);
        assert_eq!(Performance::default().resolve(1, 2048).threads, 1);
        let s = Settings::from_json(r#"{"performance":{"device":"rm -rf"}}"#).unwrap();
        assert_eq!(s.performance.device, "auto");
    }

    #[test]
    fn unknown_fields_preserved() {
        let s = Settings::from_json(r#"{"future_option":{"a":1}}"#).unwrap();
        let out = serde_json::to_string(&s).unwrap();
        assert!(out.contains("future_option"));
    }
}
