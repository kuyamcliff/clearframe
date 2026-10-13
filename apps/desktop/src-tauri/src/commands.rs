//! Tauri commands: thin wrappers over `clearframe_app::App`. Long operations
//! run on blocking threads so the UI never waits on processing.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager, State};

use clearframe_app::export::NewJob;
use clearframe_app::storage::{RelocationMode, StorageKind};
use clearframe_app::App;
use clearframe_core::paths::CacheArea;
use clearframe_core::settings::Settings;
use clearframe_pipeline::presets::Preset;
use clearframe_pipeline::project::Project;
use clearframe_pipeline::spec::*;

type R<T> = Result<T, String>;
type S<'a> = State<'a, Arc<App>>;

fn e<E: std::fmt::Display>(x: E) -> String {
    format!("{x:#}")
}

/// Run blocking work off the async runtime.
async fn blocking<T: Send + 'static>(f: impl FnOnce() -> anyhow::Result<T> + Send + 'static) -> R<T> {
    tauri::async_runtime::spawn_blocking(f).await.map_err(e)?.map_err(e)
}

/// Allow the webview to load a file through the asset protocol.
fn allow_file(h: &AppHandle, p: &std::path::Path) {
    let _ = h.asset_protocol_scope().allow_file(p);
    crate::media_server::allow_file(p);
}

/// The cache (previews, thumbnails, proxies) is always readable by the webview.
pub fn allow_cache_assets(h: &AppHandle, app: &App) {
    let dir = app.paths().cache_dir;
    let _ = h.asset_protocol_scope().allow_directory(&dir, true);
    crate::media_server::allow_dir(&dir);
}

#[derive(Serialize)]
struct AppInfo {
    version: String,
    os: String,
    arch: String,
    platform: String,
    ffmpeg: Option<serde_json::Value>,
    ffmpeg_error: Option<String>,
    /// FFmpeg is still being detected; a `tools_changed` event follows.
    ffmpeg_checking: bool,
    queue_paused: bool,
    compiled_backends: Vec<String>,
}

#[tauri::command]
fn app_info(app: S<'_>) -> AppInfo {
    let (t, err) = app.tools_status();
    AppInfo {
        version: env!("CARGO_PKG_VERSION").into(),
        os: std::env::consts::OS.into(),
        arch: std::env::consts::ARCH.into(),
        platform: clearframe_models::manager::platform_id().into(),
        ffmpeg: t.map(|t| serde_json::json!({ "version": t.version, "path": t.ffmpeg, "source": t.source, "license": t.license_hint, "vidstab": t.has_filter("vidstabdetect"), "zscale": t.has_filter("zscale") })),
        ffmpeg_error: err,
        ffmpeg_checking: app.tools_pending(),
        queue_paused: app.queue.is_paused(),
        compiled_backends: {
            use clearframe_inference::device::{compiled, BackendKind};
            [BackendKind::Cpu, BackendKind::Gpu, BackendKind::Cuda, BackendKind::Metal].into_iter().filter(|k| compiled(*k)).map(|k| k.id().to_string()).collect()
        },
    }
}

#[tauri::command]
fn settings_get(app: S<'_>) -> Settings {
    app.settings()
}

#[tauri::command]
fn settings_set(app: S<'_>, settings: Settings) -> R<Settings> {
    app.save_settings(settings).map_err(e)
}

#[tauri::command]
fn settings_reset(app: S<'_>) -> R<Settings> {
    app.save_settings(Settings { onboarding_complete: true, ..Default::default() }).map_err(e)
}

#[tauri::command]
async fn hardware(app: S<'_>, refresh: bool) -> R<serde_json::Value> {
    let a = app.inner().clone();
    blocking(move || {
        let hw = if refresh { a.refresh_hardware() } else { a.hardware() };
        Ok(serde_json::json!({ "hardware": hw, "plan": a.resource_plan(), "probe": clearframe_inference::device::probe() }))
    })
    .await
}

#[tauri::command]
async fn system_sample(app: S<'_>) -> R<clearframe_app::resources::SystemSample> {
    let a = app.inner().clone();
    blocking(move || Ok(a.system_sample())).await
}

#[tauri::command]
async fn codecs(app: S<'_>) -> R<serde_json::Value> {
    // Off the main thread: it waits for FFmpeg detection right after launch.
    let a = app.inner().clone();
    blocking(move || {
        let t = a.tools()?;
        let list: Vec<_> = clearframe_media::encode::CODECS.iter().map(|c| serde_json::json!({ "profile": c, "available": t.has_video_encoder(c.encoder) })).collect();
        Ok(serde_json::json!(list))
    })
    .await
}

#[tauri::command]
fn operation_defaults() -> serde_json::Value {
    clearframe_app::ipc::operation_defaults()
}

// ---- sources & previews ----

#[tauri::command]
async fn source_info(h: AppHandle, app: S<'_>, path: PathBuf, detect_interlace: bool) -> R<clearframe_app::preview::SourceInfo> {
    allow_file(&h, &path);
    let a = app.inner().clone();
    blocking(move || a.source_info(&path, detect_interlace)).await
}

#[tauri::command]
async fn source_frame(app: S<'_>, path: PathBuf, time: f64, max_width: u32) -> R<String> {
    let a = app.inner().clone();
    blocking(move || a.source_frame(&path, time, max_width)).await
}

#[tauri::command]
async fn source_proxy(app: S<'_>, path: PathBuf, start: f64, duration: f64, max_height: u32) -> R<String> {
    let a = app.inner().clone();
    blocking(move || a.source_proxy(&path, start, duration, max_height)).await
}

#[tauri::command]
async fn preview_still(
    app: S<'_>,
    source: PathBuf,
    time: f64,
    pipeline: Pipeline,
    output: OutputSettings,
    engine: EngineSettings,
    crop: Option<(u32, u32, u32, u32)>,
) -> R<clearframe_app::preview::StillPreview> {
    let a = app.inner().clone();
    blocking(move || a.preview_still(&source, time, pipeline, output, engine, crop)).await
}

#[tauri::command]
async fn preview_clip(app: S<'_>, job: NewJob, start: f64, duration: f64) -> R<clearframe_app::preview::ClipPreview> {
    let a = app.inner().clone();
    blocking(move || a.preview_clip(job, start, duration)).await
}

// ---- export & jobs ----

#[tauri::command]
async fn check_export(app: S<'_>, job: NewJob) -> R<clearframe_app::export::ExportCheck> {
    let a = app.inner().clone();
    blocking(move || a.check_export(&job)).await
}

#[tauri::command]
async fn enqueue(h: AppHandle, app: S<'_>, job: NewJob) -> R<clearframe_core::jobs::JobRecord> {
    let a = app.inner().clone();
    let r = blocking(move || a.enqueue(job)).await?;
    allow_file(&h, std::path::Path::new(&r.destination));
    Ok(r)
}

#[tauri::command]
async fn enqueue_batch(app: S<'_>, sources: Vec<PathBuf>, template: NewJob, recursive: bool) -> R<serde_json::Value> {
    let a = app.inner().clone();
    blocking(move || {
        let files = App::collect_videos(&sources, recursive);
        let r = a.enqueue_batch(files, template);
        Ok(serde_json::json!({ "created": r.created, "failed": r.failed.iter().map(|(p, m)| serde_json::json!({"path": p, "error": m})).collect::<Vec<_>>() }))
    })
    .await
}

#[tauri::command]
fn collect_videos(paths: Vec<PathBuf>, recursive: bool) -> Vec<PathBuf> {
    App::collect_videos(&paths, recursive)
}

#[tauri::command]
fn default_export_folder(app: S<'_>) -> PathBuf {
    app.default_export_folder()
}

#[tauri::command]
fn jobs_list(h: AppHandle, app: S<'_>, include_previews: bool) -> R<Vec<clearframe_core::jobs::JobRecord>> {
    let jobs = app.db.jobs(include_previews).map_err(e)?;
    for j in &jobs {
        if j.state == clearframe_core::jobs::JobState::Completed {
            allow_file(&h, std::path::Path::new(&j.destination));
        }
    }
    Ok(jobs)
}

#[tauri::command]
fn job_events(app: S<'_>, id: String) -> R<Vec<clearframe_core::db::JobEvent>> {
    app.db.job_events(&id).map_err(e)
}

#[tauri::command]
fn job_action(app: S<'_>, id: String, action: String, index: Option<usize>) -> R<()> {
    match action.as_str() {
        "pause" => app.pause_job(&id),
        "resume" | "retry" => app.resume_job(&id),
        "cancel" => app.cancel_job(&id),
        "remove" => app.remove_job(&id),
        "reorder" => app.reorder_job(&id, index.unwrap_or(0)),
        _ => Err(anyhow::anyhow!("unknown action {action}")),
    }
    .map_err(e)
}

#[tauri::command]
fn queue_action(app: S<'_>, action: String) -> R<usize> {
    match action.as_str() {
        "pause" => {
            app.pause_queue();
            Ok(0)
        }
        "resume" => {
            app.resume_queue();
            Ok(0)
        }
        "clear_finished" => app.clear_finished_jobs().map_err(e),
        _ => Err(format!("unknown action {action}")),
    }
}

// ---- models ----

#[tauri::command]
async fn models_list(app: S<'_>) -> R<Vec<clearframe_app::models::ModelCard>> {
    let a = app.inner().clone();
    blocking(move || Ok(a.model_cards())).await
}

#[tauri::command]
fn model_install(app: S<'_>, id: String, local_files: Option<BTreeMap<String, PathBuf>>) -> R<()> {
    app.install_model(&id, local_files.unwrap_or_default()).map_err(e)
}

#[tauri::command]
async fn model_action(app: S<'_>, id: String, action: String) -> R<serde_json::Value> {
    let a = app.inner().clone();
    blocking(move || {
        Ok(match action.as_str() {
            "pause" => serde_json::json!(a.pause_install(&id)),
            "cancel" => serde_json::json!(a.cancel_install(&id)),
            "remove" => serde_json::json!({ "freed": a.remove_model(&id)? }),
            "verify" => serde_json::to_value(a.verify_model(&id)?)?,
            "repair" => serde_json::json!({ "ok_without_reinstall": a.repair_model(&id)? }),
            "usage" => serde_json::json!(a.model_usage(&id)),
            "in_use" => serde_json::json!(a.queue.models_in_use().contains(&id)),
            "open" => {
                let dir = a.models.model_dir(&id);
                let target = if dir.exists() { dir } else { a.models.models_dir() };
                clearframe_app::storage::open_in_file_manager(&target)?;
                serde_json::json!(true)
            }
            _ => anyhow::bail!("unknown action {action}"),
        })
    })
    .await
}

#[tauri::command]
async fn model_import_custom(app: S<'_>, name: String, license: String, file: PathBuf) -> R<String> {
    let a = app.inner().clone();
    blocking(move || a.import_custom_model(&name, &license, file, None)).await
}

// ---- storage ----

#[tauri::command]
async fn storage_overview(app: S<'_>) -> R<clearframe_app::storage::StorageOverview> {
    let a = app.inner().clone();
    blocking(move || Ok(a.storage_overview())).await
}

#[tauri::command]
async fn storage_plan(app: S<'_>, kind: StorageKind, dest: PathBuf) -> R<clearframe_app::storage::RelocationPlan> {
    let a = app.inner().clone();
    blocking(move || Ok(a.plan_relocation(kind, &dest))).await
}

#[tauri::command]
fn storage_relocate(h: AppHandle, app: S<'_>, kind: StorageKind, dest: PathBuf, mode: RelocationMode) -> R<()> {
    if kind == StorageKind::Cache {
        let _ = h.asset_protocol_scope().allow_directory(&dest, true);
    }
    app.relocate(kind, dest.clone(), mode).map_err(e)?;
    if kind == StorageKind::Cache {
        crate::media_server::allow_dir(&dest);
    }
    Ok(())
}

#[tauri::command]
fn storage_cancel(app: S<'_>) {
    app.cancel_relocation();
}

/// URL prefix for `<video>` sources; the path is appended percent-encoded.
#[tauri::command]
fn media_base() -> R<String> {
    crate::media_server::start().map_err(|x| x.to_string())
}

#[tauri::command]
fn storage_reset(h: AppHandle, app: S<'_>, kind: StorageKind) -> R<()> {
    app.reset_storage_location(kind).map_err(e)?;
    allow_cache_assets(&h, &app);
    Ok(())
}

#[tauri::command]
async fn cache_clear(app: S<'_>, areas: Vec<CacheArea>, include_resumable: bool) -> R<clearframe_core::cache::ClearResult> {
    let a = app.inner().clone();
    blocking(move || Ok(a.clear_cache(&areas, include_resumable))).await
}

#[tauri::command]
async fn cache_enforce(app: S<'_>) -> R<clearframe_core::cache::ClearResult> {
    let a = app.inner().clone();
    blocking(move || Ok(a.enforce_cache_policy())).await
}

#[tauri::command]
fn open_path(path: PathBuf) -> R<()> {
    let target = if path.is_file() { path.parent().map(|p| p.to_path_buf()).unwrap_or(path) } else { path };
    clearframe_app::storage::open_in_file_manager(&target).map_err(e)
}

// ---- presets & projects ----

#[tauri::command]
fn presets_list(app: S<'_>) -> Vec<Preset> {
    app.presets()
}

#[tauri::command]
fn preset_action(
    app: S<'_>,
    action: String,
    id: Option<String>,
    preset: Option<Preset>,
    name: Option<String>,
    favorite: Option<bool>,
    path: Option<PathBuf>,
) -> R<serde_json::Value> {
    let id = id.unwrap_or_default();
    Ok(match action.as_str() {
        "save" => serde_json::to_value(app.save_preset(preset.ok_or("missing preset")?).map_err(e)?).map_err(e)?,
        "duplicate" => serde_json::to_value(app.duplicate_preset(&id).map_err(e)?).map_err(e)?,
        "rename" => serde_json::to_value(app.rename_preset(&id, &name.unwrap_or_default()).map_err(e)?).map_err(e)?,
        "delete" => {
            app.delete_preset(&id).map_err(e)?;
            serde_json::json!(true)
        }
        "favorite" => {
            app.set_preset_favorite(&id, favorite.unwrap_or(true)).map_err(e)?;
            serde_json::json!(true)
        }
        "export" => {
            app.export_preset(&id, &path.ok_or("missing path")?).map_err(e)?;
            serde_json::json!(true)
        }
        "import" => {
            let (p, missing) = app.import_preset(&path.ok_or("missing path")?).map_err(e)?;
            serde_json::json!({ "preset": p, "missing_models": missing })
        }
        _ => return Err(format!("unknown action {action}")),
    })
}

#[derive(Deserialize)]
struct SaveProject {
    path: PathBuf,
    project: Project,
}

#[tauri::command]
fn project_save(app: S<'_>, req: SaveProject) -> R<Project> {
    let mut p = req.project;
    if p.source.fingerprint.is_empty() {
        p.source.fingerprint = clearframe_core::fsutil::quick_fingerprint(std::path::Path::new(&p.source.path)).unwrap_or_default();
    }
    p.save(&req.path).map_err(e)?;
    let _ = app.db.touch_project(&req.path.to_string_lossy(), &p.name, Some(&p.source.path));
    let _ = app.db.add_recent(&req.path.to_string_lossy(), "project", &p.name, None);
    Ok(p)
}

#[tauri::command]
fn project_open(h: AppHandle, app: S<'_>, path: PathBuf) -> R<clearframe_pipeline::project::OpenReport> {
    let a = app.inner().clone();
    let r = Project::open(&path, &move |id| a.models.resolve(id).is_some()).map_err(e)?;
    allow_file(&h, std::path::Path::new(&r.project.source.path));
    if let Some(s) = &r.suggested_source {
        allow_file(&h, std::path::Path::new(s));
    }
    let _ = app.db.touch_project(&path.to_string_lossy(), &r.project.name, Some(&r.project.source.path));
    let _ = app.db.add_recent(&path.to_string_lossy(), "project", &r.project.name, None);
    Ok(r)
}

#[tauri::command]
fn new_project(source: PathBuf, name: String) -> Project {
    Project::new(&name, &source)
}

#[tauri::command]
fn recent_list(app: S<'_>) -> R<Vec<clearframe_core::db::RecentFile>> {
    let all = app.db.recent(20).map_err(e)?;
    Ok(all.into_iter().filter(|r| std::path::Path::new(&r.path).exists() || r.kind == "project").collect())
}

#[tauri::command]
fn recent_action(app: S<'_>, action: String, path: Option<String>) -> R<()> {
    match action.as_str() {
        "remove" => app.db.remove_recent(&path.unwrap_or_default()).map_err(e),
        "clear" => app.db.clear_recent().map_err(e),
        _ => Err(format!("unknown action {action}")),
    }
}

// ---- diagnostics & updates ----

#[tauri::command]
async fn diagnostics(app: S<'_>, redact: bool) -> R<String> {
    let a = app.inner().clone();
    blocking(move || Ok(a.diagnostics_text(redact))).await
}

#[tauri::command]
fn write_text_file(path: PathBuf, contents: String) -> R<()> {
    std::fs::write(path, contents).map_err(e)
}

#[tauri::command]
async fn check_update(app: S<'_>) -> R<clearframe_app::updates::UpdateInfo> {
    let a = app.inner().clone();
    blocking(move || a.check_for_update()).await
}

/// Download the app update for this computer (verified), reporting progress as events.
#[tauri::command]
async fn update_download(app: S<'_>) -> R<()> {
    let a = app.inner().clone();
    blocking(move || {
        let mut last = 0u64;
        a.download_update(|done, total| {
            // About 100 progress events per download.
            if done == total || done >= last + (total / 100).max(1 << 16) {
                last = done;
                a.emit(clearframe_app::AppEvent::UpdateProgress { done, total });
            }
        })
        .map(|_| ())
    })
    .await
}

/// Start the downloaded update and quit so it can replace the app.
#[tauri::command]
async fn update_install(h: AppHandle, app: S<'_>) -> R<()> {
    let a = app.inner().clone();
    blocking(move || a.install_update()).await?;
    h.exit(0);
    Ok(())
}

#[tauri::command]
async fn update_catalog(app: S<'_>) -> R<String> {
    let a = app.inner().clone();
    blocking(move || a.update_catalog()).await
}

#[tauri::command]
async fn refresh_tools(app: S<'_>) -> R<()> {
    let a = app.inner().clone();
    blocking(move || {
        a.refresh_tools();
        a.tools().map(|_| ())
    })
    .await
}

#[tauri::command]
fn allow_asset(h: AppHandle, path: PathBuf) {
    allow_file(&h, &path);
}

/// Files passed on the command line (file associations, "Open with"). Returned once.
pub struct StartupFiles(pub std::sync::Mutex<Vec<PathBuf>>);

#[tauri::command]
fn startup_files(h: AppHandle, files: State<'_, StartupFiles>) -> Vec<PathBuf> {
    let v = std::mem::take(&mut *files.0.lock().unwrap());
    for p in &v {
        allow_file(&h, p);
    }
    v
}

pub fn handler() -> impl Fn(tauri::ipc::Invoke) -> bool + Send + Sync + 'static {
    tauri::generate_handler![
        media_base,
        app_info,
        settings_get,
        settings_set,
        settings_reset,
        hardware,
        system_sample,
        codecs,
        operation_defaults,
        source_info,
        source_frame,
        source_proxy,
        preview_still,
        preview_clip,
        check_export,
        enqueue,
        enqueue_batch,
        collect_videos,
        default_export_folder,
        jobs_list,
        job_events,
        job_action,
        queue_action,
        models_list,
        model_install,
        model_action,
        model_import_custom,
        storage_overview,
        storage_plan,
        storage_relocate,
        storage_cancel,
        storage_reset,
        cache_clear,
        cache_enforce,
        open_path,
        presets_list,
        preset_action,
        project_save,
        project_open,
        new_project,
        recent_list,
        recent_action,
        diagnostics,
        write_text_file,
        check_update,
        update_download,
        update_install,
        update_catalog,
        refresh_tools,
        allow_asset,
        startup_files,
    ]
}
