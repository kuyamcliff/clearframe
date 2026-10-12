// TypeScript mirrors of the Rust types exchanged over IPC (serde shapes).

export type Rational = { num: number; den: number };

export type OperationType =
  | "deinterlace" | "denoise" | "deblur" | "enhance" | "face_restore"
  | "interpolate" | "stabilize" | "color" | "grain" | "resize";

export interface Stage {
  id: string;
  enabled: boolean;
  type: OperationType;
  settings: any;
}
export interface Pipeline { stages: Stage[] }

export type RateControl =
  | { mode: "quality"; value: number }
  | { mode: "bitrate"; kbps: number }
  | { mode: "lossless" };

export interface VideoEncodeSettings {
  encoder: string;
  rate: RateControl;
  preset: string | null;
  pix_fmt: string;
  profile: string | null;
  gop: number | null;
}

export type AudioMode = { mode: "auto" } | { mode: "encode"; codec: string; bitrate_kbps: number } | { mode: "none" };
export type Container = "mp4" | "mov" | "mkv" | "webm" | "m4v";

export interface OutputSettings {
  container: Container;
  video: VideoEncodeSettings;
  audio: AudioMode;
  subtitles: boolean;
  metadata: boolean;
  chapters: boolean;
}

export interface EngineSettings { backend: "cpu" | "gpu" | "cuda" | "metal" | null; precision: "fp32" | "fp16" }
export interface TimeRange { start: number; end: number }

export interface NewJob {
  source: string;
  pipeline: Pipeline;
  output: OutputSettings;
  range: TimeRange | null;
  folder: string | null;
  file_name: string | null;
  remember_folder: boolean;
  preset_name: string | null;
  batch_id: string | null;
  engine: EngineSettings;
}

export interface ColorInfo {
  range?: string; space?: string; transfer?: string; primaries?: string;
  chroma_location?: string; mastering_display?: string; content_light?: string;
}
export interface VideoStream {
  index: number; codec: string; codec_long: string; profile?: string;
  width: number; height: number; display_width: number; display_height: number;
  sample_aspect_ratio?: string; display_aspect_ratio?: string; pix_fmt: string; bit_depth: number;
  frame_rate: Rational; avg_frame_rate: Rational; variable_frame_rate: boolean; frame_count?: number;
  field_order?: string; rotation: number; start_time: number; duration?: number; bit_rate?: number; color: ColorInfo;
}
export interface AudioStream { index: number; codec: string; channels: number; channel_layout?: string; sample_rate: number; language?: string; title?: string }
export interface SubtitleStream { index: number; codec: string; language?: string; title?: string; is_text: boolean }
export interface MediaInfo {
  path: string; container: string; container_long: string; duration: number; size_bytes: number;
  bit_rate?: number; video?: VideoStream; audio: AudioStream[]; subtitles: SubtitleStream[]; chapters: number; title?: string;
}
export interface InterlaceAnalysis { tff: number; bff: number; progressive: number; undetermined: number }
export interface SourceInfo {
  info: MediaInfo; fingerprint: string; thumbnails: [number, string][]; browser_playable: boolean; interlace: InterlaceAnalysis | null;
}

export type Severity = "info" | "warning" | "error";
export interface Issue { severity: Severity; stage_id: string | null; message: string; action: string | null }
export interface Format { width: number; height: number; fps: Rational; time_scale: number; transfer?: string; primaries?: string; interlaced: boolean }
export interface StagePlan { stage_id: string; index: number; label: string; exec: { kind: string }; input: Format; output: Format; notes: string[]; models: string[] }
export interface Estimates { peak_memory_mb: [number, number]; output_size_mb: [number, number]; work_disk_mb: [number, number]; output_frames: number; input_frames: number }
export interface ExecutionPlan {
  input: Format; output: Format; stages: StagePlan[]; passes: unknown[]; issues: Issue[];
  estimates: Estimates; output_duration: number; input_duration: number; cfr_conversion: Rational | null;
}
export interface ExportCheck { plan: ExecutionPlan; destination: string; destination_folder_ok: { Ok?: null; Err?: string }; free_bytes: number | null }

export type JobState =
  | "queued" | "waiting_for_resources" | "preparing" | "previewing" | "processing" | "encoding"
  | "validating" | "completed" | "paused" | "cancelled" | "failed" | "recoverable";

export interface Progress {
  pass: number; passes: number; phase: string; frames_done: number; frames_total: number;
  fraction: number; fps: number; eta: [number, number] | null;
}
export interface JobRecord {
  id: string; kind: "render" | "preview"; name: string; state: JobState; source: string; destination: string;
  spec_json: string; created_at: string; updated_at: string; started_at: string | null; finished_at: string | null;
  progress: number; progress_json: string | null; detail: string | null; error: string | null; report_json: string | null;
  queue_order: number; attempts: number; batch_id: string | null; heartbeat: string | null; warnings: string[];
}
export interface JobEvent { at: string; level: string; message: string }

export interface ResourceSample {
  process_memory_mb: number; process_cpu_percent: number; system_memory_used_mb: number; system_memory_total_mb: number;
  system_memory_available_mb: number; backend: string; stage: string; free_disk_mb: number | null;
}

export type ModelState =
  | "not_installed" | "downloading" | "paused" | "verifying" | "preparing" | "testing" | "installed"
  | "update_available" | "missing" | "broken" | "failed";

export interface Artifact { role: string; filename: string; source_url: string; size_bytes: number; sha256: string; format: string }
export interface ModelManifest {
  id: string; version: string; display_name: string; publisher: string; summary: string; best_for: string[]; weaknesses: string[];
  categories: string[]; tasks: string[]; architecture: string; scale: number; supports_denoise_strength: boolean;
  input_types: string[]; artifacts: Artifact[]; download_size_bytes: number; installed_size_bytes: number;
  supported_platforms: string[]; backends: string[]; precisions: string[]; minimum_ram_bytes: number;
  memory_per_megapixel_mb: number; performance_tier: string; weights_license: string; code_license: string;
  commercial_use_allowed: boolean; redistribution_allowed: boolean; license_notes: string; attribution: string[];
  source_repository: string; training_data: string; provenance_reviewed_at: string; quality_status: string; recommended: boolean;
}
export interface StoredFile { path: string; sha256: string; size: number }
export interface InstalledModel {
  id: string; version: string; state: ModelState; architecture: string; dir: string; files: Record<string, StoredFile>;
  installed_size: number; installed_at: string; verified_at: string | null; smoke_test: string | null; error: string | null;
  custom: boolean; custom_info: { display_name: string; license: string; source: string; scale: number } | null;
}
export type CapabilityLevel = "supported" | "compatible_not_certified" | "cpu_fallback_only" | "unavailable";
export interface ModelCard {
  manifest: ModelManifest | null; installed: InstalledModel | null; state: ModelState; partial_bytes: number;
  path: string | null; disk_bytes: number; capability: { level: CapabilityLevel; backend: string | null; reason: string }; in_progress: boolean;
}
export interface InstallProgress { id: string; state: ModelState; role: string | null; bytes_done: number; bytes_total: number; message: string }

export interface Preset {
  id: string; name: string; description: string; category: string; tags: string[]; favorite: boolean; builtin: boolean;
  pipeline: Pipeline; output: OutputSettings | null;
}

export type CacheArea = "previews" | "thumbnails" | "proxies" | "downloads" | "temp" | "jobs";
export interface AppPaths { data_dir: string; models_dir: string; cache_dir: string; logs_dir: string; config_dir: string; models_is_default: boolean; cache_is_default: boolean }
export interface AreaUsage { area: CacheArea; label: string; bytes: number; entries: number; protected_bytes: number }
export interface StorageOverview {
  paths: AppPaths;
  models: { models_dir: string; models_dir_exists: boolean; total_bytes: number; per_model: Record<string, number>; staging_bytes: number; free_bytes: number | null };
  cache: { cache_dir: string; total_bytes: number; free_bytes: number | null; areas: AreaUsage[]; low_space: boolean };
  data_bytes: number; data_free: number | null; warnings: string[]; relocation_in_progress: boolean;
}
export interface MovePlan { source: string; destination: string; total_bytes: number; file_count: number; destination_free: number | null; same_filesystem: boolean; problems: string[] }
export interface RelocationPlan { kind: "models" | "cache"; plan: MovePlan; disposable_bytes: number; notes: string[] }
export interface ClearResult { freed_bytes: number; removed_entries: number; skipped_protected: number; errors: string[] }

export type PerformanceMode = "balanced" | "high_performance" | "maximum" | "custom";
export type ProcessPriority = "low" | "below_normal" | "normal" | "above_normal";
export type MemoryPreset = "more_for_others" | "balanced" | "prioritize_app" | "custom";
export interface Performance {
  mode: PerformanceMode; cpu_threads: number; reserve_cpu_threads: number; priority: ProcessPriority;
  memory_preset: MemoryPreset; memory_budget_mb: number; memory_pressure: "pause" | "continue";
  max_concurrent_jobs: number; device: string; vram_budget_mb: number; oom_tile_fallback: boolean;
}
export interface Settings {
  version: number; theme: "dark" | "light" | "system"; motion: "system" | "reduced" | "full"; language: string; ui_scale: number;
  output_dir: string | null; naming_template: string; write_sidecar_report: boolean; remember_last_export_folder: boolean;
  last_export_folder: string | null; preview_seconds: number; preview_max_height: number;
  quality_default: "fast" | "balanced" | "quality"; backend: string; fallback: string; half_precision: boolean;
  memory_limit_mb: number; max_concurrent_jobs: number; stop_queue_on_error: boolean; segment_seconds: number;
  keep_failed_work: boolean; ffmpeg_dir: string | null; cache_limit_mb: number; cache_max_age_days: number;
  low_disk_warning_mb: number; performance: Performance; notifications: boolean; inspector_mode_advanced: boolean;
  confirm_before_quit_with_jobs: boolean; check_updates_on_launch: boolean; allow_catalog_updates: boolean;
  onboarding_complete: boolean; shortcuts: Record<string, string>; [k: string]: unknown;
}
export interface ResourcePlan { threads: number; priority: ProcessPriority; memory_budget_mb: number; reserve_memory_mb: number; max_concurrent_jobs: number; explanation: string[] }
export interface GpuInfo { vendor: string; name: string; memory_total_mb: number | null; memory_free_mb: number | null; driver: string | null; source: string }
export interface BackendStatus { kind: "cpu" | "gpu" | "cuda" | "metal"; compiled: boolean; available: boolean; detail: string }
/** A GPU usable by the cross-vendor backend (Vulkan / DirectX 12 / Metal). */
export interface GpuAdapter { index: number; name: string; vendor: string; kind: string; api: string; max_buffer_mb: number }

export interface HardwareInfo {
  os: string; os_version: string; arch: string; cpu_brand: string; logical_cores: number; physical_cores: number | null;
  ram_total_mb: number; ram_available_mb: number; gpus: GpuInfo[]; backends: BackendStatus[]; gpu_adapters?: GpuAdapter[]; precisions: string[]; cpu_features: string[];
}
export interface GpuSample { name: string; utilization_percent: number | null; memory_used_mb: number | null; memory_total_mb: number | null; temperature_c: number | null; measured: boolean }
export interface SystemSample {
  cpu_percent: number; logical_cores: number; memory_total_mb: number; memory_used_mb: number; memory_available_mb: number;
  clearframe_memory_mb: number; clearframe_cpu_percent: number; gpus: GpuSample[]; max_temperature_c: number | null;
  thermal_warning: string | null; memory_pressure: boolean;
}
export interface CodecProfile {
  encoder: string; label: string; family: string; hardware: boolean; pix_fmts: string[]; presets: string[];
  default_preset: string | null; quality_range: [number, number, number] | null; supports_bitrate: boolean;
  supports_lossless: boolean; containers: string[]; description: string;
}
export interface AppInfo {
  version: string; os: string; arch: string; platform: string;
  ffmpeg: { version: string; path: string; source: string; license: string; vidstab: boolean; zscale: boolean } | null;
  ffmpeg_error: string | null; queue_paused: boolean; compiled_backends: string[];
  /** Present only in the browser demo backend. */
  demo?: boolean;
}
export interface StillPreview { key: string; original: string; processed: string; original_size: [number, number]; processed_size: [number, number]; seconds: number; notes: string[]; cached: boolean }
export interface ClipPreview { key: string; path: string | null; job_id: string | null; original_proxy: string }
export interface RecentFile { path: string; kind: string; title: string; opened_at: string }
export interface Project {
  format: string; schema_version: number; name: string;
  source: { path: string; relative_path?: string | null; fingerprint: string; size_bytes: number; summary?: unknown };
  pipeline: Pipeline; output: OutputSettings; range: TimeRange | null; preview_range: TimeRange | null;
  export: { folder: string | null; file_name: string | null }; engine: EngineSettings; models: Record<string, string>;
  job_ids: string[]; preset_id: string | null; notes: string; created_at: string; updated_at: string;
}
export interface OpenReport { project: Project; source_found: boolean; source_changed: boolean; suggested_source: string | null; missing_models: string[]; migrated_from: number | null; backup_path: string | null }
export interface UpdateInfo { current: string; latest: string; newer: boolean; url: string; notes: string }

export type AppEvent =
  | { type: "job_updated"; job: JobRecord }
  | { type: "job_progress"; id: string; progress: Progress }
  | { type: "job_log"; id: string; level: string; message: string }
  | { type: "job_removed"; id: string }
  | { type: "queue_state"; paused: boolean; running: number; queued: number }
  | { type: "model_progress"; progress: InstallProgress }
  | { type: "model_changed"; id: string }
  | { type: "model_error"; id: string; message: string }
  | { type: "preview_ready"; key: string; path: string }
  | { type: "preview_failed"; key: string; message: string }
  | { type: "job_resources"; id: string; sample: ResourceSample }
  | { type: "storage_progress"; kind: string; bytes_done: number; bytes_total: number }
  | { type: "storage_done"; kind: string; ok: boolean; message: string }
  | { type: "notice"; level: string; message: string };
