// Typed access to the Rust backend. Inside the desktop app every call goes
// through Tauri IPC; in a plain browser (UI development and automated UI tests)
// an in-memory demo backend answers instead and the UI says so.

import type {
  AppEvent, AppInfo, ClearResult, ClipPreview, CodecProfile, EngineSettings, ExportCheck, HardwareInfo, JobEvent,
  JobRecord, ModelCard, NewJob, OpenReport, OutputSettings, Pipeline, Preset, Project, RecentFile, RelocationPlan,
  ResourcePlan, Settings, SourceInfo, StillPreview, StorageOverview, SystemSample, UpdateInfo, CacheArea,
} from "./types";

type Invoke = <T>(cmd: string, args?: Record<string, unknown>) => Promise<T>;

export const inTauri = (): boolean => typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

let invokeImpl: Invoke | null = null;
let mockModule: typeof import("./mock") | null = null;

async function backend(): Promise<Invoke> {
  if (invokeImpl) return invokeImpl;
  if (inTauri()) {
    const core = await import("@tauri-apps/api/core");
    invokeImpl = core.invoke as Invoke;
  } else {
    mockModule = await import("./mock");
    invokeImpl = mockModule.mockInvoke as Invoke;
  }
  return invokeImpl;
}

export async function call<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  const inv = await backend();
  try {
    return await inv<T>(cmd, args);
  } catch (e) {
    throw new Error(typeof e === "string" ? e : e instanceof Error ? e.message : JSON.stringify(e));
  }
}

/** Subscribe to backend events. Returns an unsubscribe function. */
export async function onEvent(handler: (e: AppEvent) => void): Promise<() => void> {
  if (inTauri()) {
    const ev = await import("@tauri-apps/api/event");
    return ev.listen<AppEvent>("cf://event", (m) => handler(m.payload));
  }
  await backend();
  return mockModule!.mockListen(handler);
}

/** URL the webview can load for a local file (asset protocol). */
export function fileUrl(path: string | null | undefined, bust?: string | number): string {
  if (!path) return "";
  if (inTauri()) {
    // convertFileSrc is synchronous; resolved lazily to keep this module browser-safe.
    const w = window as unknown as { __TAURI_INTERNALS__: { convertFileSrc: (p: string, proto?: string) => string } };
    const u = w.__TAURI_INTERNALS__.convertFileSrc(path, "asset");
    return bust !== undefined ? `${u}?v=${bust}` : u;
  }
  return mockModule ? mockModule.mockFileUrl(path) : "";
}

let mediaBase: string | null = null;

/** Fetch the local media server prefix once at boot (Tauri only). */
export async function initMedia(): Promise<void> {
  if (!inTauri()) return;
  try { mediaBase = await call<string>("media_base"); } catch { mediaBase = null; }
}

/**
 * URL for a `<video>` source. WebKitGTK cannot play from the asset protocol,
 * so videos go through the app's local HTTP server when it is running.
 */
export function mediaUrl(path: string | null | undefined, bust?: string | number): string {
  if (!path) return "";
  if (inTauri() && mediaBase) {
    const enc = path.replace(/\\/g, "/").split("/").map(encodeURIComponent).join("/").replace(/^\//, "");
    return bust !== undefined ? `${mediaBase}${enc}?v=${bust}` : `${mediaBase}${enc}`;
  }
  return fileUrl(path, bust);
}

// ---------- native dialogs ----------

export interface FileFilter { name: string; extensions: string[] }
export const VIDEO_EXTENSIONS = ["mp4", "mkv", "mov", "avi", "webm", "m4v", "mts", "m2ts", "ts", "mpg", "mpeg", "wmv", "flv", "3gp", "vob", "ogv", "mxf", "dv"];

export async function pickFile(title: string, filters: FileFilter[], multiple = false): Promise<string[]> {
  if (inTauri()) {
    const d = await import("@tauri-apps/plugin-dialog");
    const r = await d.open({ title, filters, multiple, directory: false });
    if (!r) return [];
    return Array.isArray(r) ? r : [r];
  }
  await backend();
  return mockModule!.mockPick(title, false, multiple);
}

export async function pickFolder(title: string, defaultPath?: string): Promise<string | null> {
  if (inTauri()) {
    const d = await import("@tauri-apps/plugin-dialog");
    const r = await d.open({ title, directory: true, multiple: false, defaultPath });
    return typeof r === "string" ? r : null;
  }
  await backend();
  return (await mockModule!.mockPick(title, true, false))[0] ?? null;
}

export async function pickSave(title: string, defaultPath: string, filters: FileFilter[]): Promise<string | null> {
  if (inTauri()) {
    const d = await import("@tauri-apps/plugin-dialog");
    return (await d.save({ title, defaultPath, filters })) ?? null;
  }
  await backend();
  return mockModule!.mockSave(defaultPath);
}

export async function notify(title: string, body: string): Promise<void> {
  if (!inTauri()) return;
  try {
    const n = await import("@tauri-apps/plugin-notification");
    let ok = await n.isPermissionGranted();
    if (!ok) ok = (await n.requestPermission()) === "granted";
    if (ok) n.sendNotification({ title, body });
  } catch {
    // Notifications are optional.
  }
}

/** Files dropped onto the window (desktop only). */
export async function onFileDrop(handler: (paths: string[]) => void, hover?: (over: boolean) => void): Promise<() => void> {
  if (!inTauri()) return () => {};
  const wv = await import("@tauri-apps/api/webview");
  return wv.getCurrentWebview().onDragDropEvent((e) => {
    const p = e.payload;
    if (p.type === "drop") {
      hover?.(false);
      handler(p.paths);
    } else if (p.type === "enter" || p.type === "over") hover?.(true);
    else hover?.(false);
  });
}

// ---------- typed commands ----------

export interface HardwareReport { hardware: HardwareInfo; plan: ResourcePlan; probe: unknown }
export interface CodecEntry { profile: CodecProfile; available: boolean }
export interface OperationDefaults { [k: string]: any }

export const api = {
  appInfo: () => call<AppInfo>("app_info"),
  settingsGet: () => call<Settings>("settings_get"),
  settingsSet: (settings: Settings) => call<Settings>("settings_set", { settings }),
  settingsReset: () => call<Settings>("settings_reset"),
  hardware: (refresh = false) => call<HardwareReport>("hardware", { refresh }),
  systemSample: () => call<SystemSample>("system_sample"),
  codecs: () => call<CodecEntry[]>("codecs"),
  operationDefaults: () => call<OperationDefaults>("operation_defaults"),

  sourceInfo: (path: string, detectInterlace: boolean) => call<SourceInfo>("source_info", { path, detectInterlace }),
  sourceFrame: (path: string, time: number, maxWidth: number) => call<string>("source_frame", { path, time, maxWidth }),
  sourceProxy: (path: string, start: number, duration: number, maxHeight: number) => call<string>("source_proxy", { path, start, duration, maxHeight }),
  previewStill: (source: string, time: number, pipeline: Pipeline, output: OutputSettings, engine: EngineSettings, crop: [number, number, number, number] | null = null) =>
    call<StillPreview>("preview_still", { source, time, pipeline, output, engine, crop }),
  previewClip: (job: NewJob, start: number, duration: number) => call<ClipPreview>("preview_clip", { job, start, duration }),

  checkExport: (job: NewJob) => call<ExportCheck>("check_export", { job }),
  enqueue: (job: NewJob) => call<JobRecord>("enqueue", { job }),
  enqueueBatch: (sources: string[], template: NewJob, recursive: boolean) =>
    call<{ created: JobRecord[]; failed: { path: string; error: string }[] }>("enqueue_batch", { sources, template, recursive }),
  collectVideos: (paths: string[], recursive: boolean) => call<string[]>("collect_videos", { paths, recursive }),
  defaultExportFolder: () => call<string>("default_export_folder"),
  jobsList: (includePreviews = false) => call<JobRecord[]>("jobs_list", { includePreviews }),
  jobEvents: (id: string) => call<JobEvent[]>("job_events", { id }),
  jobAction: (id: string, action: "pause" | "resume" | "retry" | "cancel" | "remove" | "reorder", index?: number) =>
    call<void>("job_action", { id, action, index: index ?? null }),
  queueAction: (action: "pause" | "resume" | "clear_finished") => call<number>("queue_action", { action }),

  modelsList: () => call<ModelCard[]>("models_list"),
  modelInstall: (id: string, localFiles?: Record<string, string>) => call<void>("model_install", { id, localFiles: localFiles ?? null }),
  modelAction: <T = unknown>(id: string, action: "pause" | "cancel" | "remove" | "verify" | "repair" | "usage" | "in_use" | "open") =>
    call<T>("model_action", { id, action }),
  modelImportCustom: (name: string, license: string, file: string) => call<string>("model_import_custom", { name, license, file }),

  storageOverview: () => call<StorageOverview>("storage_overview"),
  storagePlan: (kind: "models" | "cache", dest: string) => call<RelocationPlan>("storage_plan", { kind, dest }),
  storageRelocate: (kind: "models" | "cache", dest: string, mode: "move" | "clear_disposable_and_move" | "switch_only") =>
    call<void>("storage_relocate", { kind, dest, mode }),
  storageCancel: () => call<void>("storage_cancel"),
  storageReset: (kind: "models" | "cache") => call<void>("storage_reset", { kind }),
  cacheClear: (areas: CacheArea[], includeResumable: boolean) => call<ClearResult>("cache_clear", { areas, includeResumable }),
  cacheEnforce: () => call<ClearResult>("cache_enforce"),
  openPath: (path: string) => call<void>("open_path", { path }),

  presetsList: () => call<Preset[]>("presets_list"),
  presetAction: <T = unknown>(action: "save" | "duplicate" | "rename" | "delete" | "favorite" | "export" | "import", a: { id?: string; preset?: Preset; name?: string; favorite?: boolean; path?: string }) =>
    call<T>("preset_action", { action, id: a.id ?? null, preset: a.preset ?? null, name: a.name ?? null, favorite: a.favorite ?? null, path: a.path ?? null }),

  projectSave: (path: string, project: Project) => call<Project>("project_save", { req: { path, project } }),
  projectOpen: (path: string) => call<OpenReport>("project_open", { path }),
  newProject: (source: string, name: string) => call<Project>("new_project", { source, name }),
  recentList: () => call<RecentFile[]>("recent_list"),
  recentAction: (action: "remove" | "clear", path?: string) => call<void>("recent_action", { action, path: path ?? null }),

  diagnostics: (redact: boolean) => call<string>("diagnostics", { redact }),
  writeTextFile: (path: string, contents: string) => call<void>("write_text_file", { path, contents }),
  checkUpdate: () => call<UpdateInfo>("check_update"),
  updateCatalog: () => call<string>("update_catalog"),
  refreshTools: () => call<void>("refresh_tools"),
  allowAsset: (path: string) => call<void>("allow_asset", { path }),
  startupFiles: () => call<string[]>("startup_files"),
};
