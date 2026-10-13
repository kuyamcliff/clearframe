import { create } from "zustand";

import { api, notify, onEvent, type CodecEntry, type HardwareReport, type OperationDefaults } from "../lib/ipc";
import { basename } from "../lib/format";
import { addStageCanonical } from "../lib/ops";
import type {
  AppEvent, AppInfo, ClipPreview, OperationType, EngineSettings, InstallProgress, JobRecord, ModelCard, NewJob, OutputSettings, Pipeline,
  Preset, Progress, ResourceSample, Settings, SourceInfo, StillPreview, TimeRange,
  UpdateInfo,
} from "../lib/types";

export type Panel = "sources" | "queue" | null;
export type DialogKind = "models" | "prefs" | "start" | null;
export type SettingsSection = "general" | "export" | "storage" | "performance" | "advanced" | "diagnostics" | "about";
export type CompareMode = "split" | "side" | "original" | "processed";
export type Zoom = "fit" | "1" | "2";

export interface Toast { id: number; level: "info" | "success" | "warning" | "error"; message: string; action?: { label: string; run: () => void } }

interface History { past: Pipeline[]; future: Pipeline[] }

export interface StorageProgress { kind: string; done: number; total: number; finished?: { ok: boolean; message: string } }

export interface State {
  ready: boolean;
  bootError: string | null;
  app: AppInfo | null;
  settings: Settings | null;
  defaults: OperationDefaults | null;
  codecs: CodecEntry[];
  hardware: HardwareReport | null;
  /** A newer version found by an update check (shown by the update banner). */
  update: UpdateInfo | null;

  panel: Panel;
  dialog: DialogKind;
  settingsSection: SettingsSection;
  sideTab: "adjust" | "codec";
  sideOpen: boolean;
  zoom: Zoom;
  muted: boolean;
  /** Source playback through a proxy copy, started at `start` seconds. */
  sourcePlay: { src: string; start: number } | null;
  sourcePlayLoading: boolean;
  sources: string[];
  modelFocus: string | null;

  // workspace
  sourcePath: string | null;
  source: SourceInfo | null;
  sourceLoading: boolean;
  sourceError: string | null;
  playhead: number;
  range: TimeRange | null;
  compare: CompareMode;
  still: StillPreview | null;
  stillAt: number;
  stillBusy: boolean;
  stillError: string | null;
  clip: ClipPreview | null;
  clipRange: TimeRange | null;
  clipError: string | null;

  // edit state
  pipeline: Pipeline;
  history: History;
  output: OutputSettings | null;
  engine: EngineSettings;
  exportFolder: string | null;
  exportFileName: string | null;
  presetName: string | null;
  projectPath: string | null;
  projectName: string | null;
  dirty: boolean;

  // live data
  jobs: JobRecord[];
  progress: Record<string, Progress>;
  resources: Record<string, ResourceSample>;
  queuePaused: boolean;
  models: ModelCard[];
  installs: Record<string, InstallProgress>;
  presets: Preset[];
  storageProgress: StorageProgress | null;
  toasts: Toast[];
}

const emptyPipeline: Pipeline = { stages: [] };

export const useStore = create<State>(() => ({
  ready: false,
  bootError: null,
  app: null,
  settings: null,
  defaults: null,
  codecs: [],
  update: null,
  hardware: null,
  panel: null,
  dialog: null,
  settingsSection: "general",
  sideTab: "adjust",
  sideOpen: true,
  zoom: "fit",
  muted: true,
  sourcePlay: null,
  sourcePlayLoading: false,
  sources: [],
  modelFocus: null,
  sourcePath: null,
  source: null,
  sourceLoading: false,
  sourceError: null,
  playhead: 0,
  range: null,
  compare: "split",
  still: null,
  stillAt: 0,
  stillBusy: false,
  stillError: null,
  clip: null,
  clipRange: null,
  clipError: null,
  pipeline: emptyPipeline,
  history: { past: [], future: [] },
  output: null,
  engine: { backend: null, precision: "fp32" },
  exportFolder: null,
  exportFileName: null,
  presetName: null,
  projectPath: null,
  projectName: null,
  dirty: false,
  jobs: [],
  progress: {},
  resources: {},
  queuePaused: false,
  models: [],
  installs: {},
  presets: [],
  storageProgress: null,
  toasts: [],
}));

const set = useStore.setState;
const get = useStore.getState;

// ---------- toasts ----------

let toastSeq = 1;
export function toast(level: Toast["level"], message: string, action?: Toast["action"]) {
  const id = toastSeq++;
  set((s) => ({ toasts: [...s.toasts.slice(-4), { id, level, message, action }] }));
  window.setTimeout(() => dismissToast(id), level === "error" ? 12000 : 6000);
}
export function dismissToast(id: number) {
  set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }));
}
export function errorToast(e: unknown) {
  toast("error", e instanceof Error ? e.message : String(e));
}

// ---------- boot & events ----------

let unlisten: (() => void) | null = null;

export async function boot() {
  try {
    const [app, settings, defaults] = await Promise.all([api.appInfo(), api.settingsGet(), api.operationDefaults()]);
    set({ app, settings, defaults, output: structuredClone(defaults.output), queuePaused: app.queue_paused, engine: { backend: null, precision: settings.half_precision ? "fp16" : "fp32" } });
    applyAppearance(settings);
    if (!unlisten) unlisten = await onEvent(handleEvent);
    // Codecs need FFmpeg, which may still be being detected right after launch; don't hold the
    // interface back for them (they load again on "tools_changed").
    api.codecs().then((codecs) => set({ codecs })).catch(() => {});
    await Promise.all([refreshModels(), refreshJobs(), refreshPresets()]);
    api.hardware(false).then((hardware) => set({ hardware })).catch(() => {});
    set({ ready: true });
  } catch (e) {
    set({ bootError: e instanceof Error ? e.message : String(e), ready: true });
  }
}

export function applyAppearance(s: Settings) {
  const root = document.documentElement;
  const dark = s.theme === "dark" || (s.theme === "system" && window.matchMedia?.("(prefers-color-scheme: dark)").matches);
  root.dataset.theme = dark ? "dark" : "light";
  const reduced = s.motion === "reduced" || (s.motion === "system" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches);
  root.dataset.motion = reduced ? "reduced" : "full";
  root.style.fontSize = `${Math.round(14 * (s.ui_scale || 1))}px`;
}

let modelRefresh: number | null = null;
function refreshModelsSoon() {
  if (modelRefresh) window.clearTimeout(modelRefresh);
  modelRefresh = window.setTimeout(() => {
    modelRefresh = null;
    refreshModels();
  }, 150);
}

export function handleEvent(e: AppEvent) {
  switch (e.type) {
    case "job_updated": {
      const prev = get().jobs.find((j) => j.id === e.job.id);
      set((s) => {
        const jobs = s.jobs.some((j) => j.id === e.job.id) ? s.jobs.map((j) => (j.id === e.job.id ? e.job : j)) : [...s.jobs, e.job];
        return { jobs };
      });
      onJobTransition(prev, e.job);
      break;
    }
    case "job_progress":
      set((s) => ({ progress: { ...s.progress, [e.id]: e.progress } }));
      break;
    case "job_resources":
      set((s) => ({ resources: { ...s.resources, [e.id]: e.sample } }));
      break;
    case "job_removed":
      set((s) => ({ jobs: s.jobs.filter((j) => j.id !== e.id) }));
      break;
    case "job_log":
      if (e.level === "error") toast("error", e.message);
      break;
    case "queue_state":
      set({ queuePaused: e.paused });
      break;
    case "model_progress":
      set((s) => ({ installs: { ...s.installs, [e.progress.id]: e.progress } }));
      break;
    case "model_changed":
      set((s) => {
        const installs = { ...s.installs };
        delete installs[e.id];
        return { installs };
      });
      refreshModelsSoon();
      break;
    case "model_error":
      set((s) => {
        const installs = { ...s.installs };
        delete installs[e.id];
        return { installs };
      });
      toast("error", `${e.id}: ${e.message}`);
      refreshModelsSoon();
      break;
    case "preview_ready": {
      const clip = get().clip;
      if (clip && clip.key === e.key) set({ clip: { ...clip, path: e.path, job_id: null }, clipError: null });
      break;
    }
    case "preview_failed": {
      const clip = get().clip;
      if (clip && clip.key === e.key) set({ clipError: e.message, clip: { ...clip, job_id: null } });
      break;
    }
    case "storage_progress":
      set({ storageProgress: { kind: e.kind, done: e.bytes_done, total: e.bytes_total } });
      break;
    case "storage_done":
      set({ storageProgress: { kind: e.kind, done: 1, total: 1, finished: { ok: e.ok, message: e.message } } });
      toast(e.ok ? "success" : "error", e.message);
      refreshModelsSoon();
      break;
    case "notice":
      toast(e.level === "error" ? "error" : e.level === "warning" ? "warning" : "info", e.message);
      break;
    case "tools_changed":
      // FFmpeg detection finished after launch: pick up its version and the available codecs.
      api.appInfo().then((app) => set({ app })).catch(() => {});
      api.codecs().then((codecs) => set({ codecs })).catch(() => {});
      break;
  }
}

function onJobTransition(prev: JobRecord | undefined, j: JobRecord) {
  if (prev?.state === j.state) return;
  const clip = get().clip;
  if (j.kind === "preview") {
    if (clip && clip.job_id === j.id) {
      if (j.state === "completed") set({ clip: { ...clip, path: j.destination, job_id: null }, clipError: null });
      if (j.state === "failed") set({ clipError: j.error ?? "Preview failed", clip: { ...clip, job_id: null } });
      if (j.state === "cancelled") set({ clip: { ...clip, job_id: null } });
    }
    return;
  }
  const s = get().settings;
  if (j.state === "completed") {
    toast("success", `Finished: ${basename(j.destination)}`, { label: "Show", run: () => api.openPath(j.destination).catch(errorToast) });
    if (s?.notifications && document.hidden) notify("ClipFX Enhance: export finished", basename(j.destination));
  } else if (j.state === "failed") {
    toast("error", `Failed: ${j.name}${j.error ? ` — ${j.error}` : ""}`);
    if (s?.notifications && document.hidden) notify("ClipFX Enhance: export failed", j.name);
  }
}

export async function refreshModels() {
  try {
    set({ models: await api.modelsList() });
  } catch (e) {
    errorToast(e);
  }
}

export async function refreshJobs() {
  try {
    const jobs = await api.jobsList(true);
    const progress: Record<string, Progress> = { ...get().progress };
    for (const j of jobs) {
      if (j.progress_json && !progress[j.id]) {
        try {
          progress[j.id] = JSON.parse(j.progress_json);
        } catch {
          // ignore malformed progress snapshots
        }
      }
    }
    set({ jobs, progress });
  } catch (e) {
    errorToast(e);
  }
}

export async function refreshPresets() {
  try {
    set({ presets: await api.presetsList() });
  } catch (e) {
    errorToast(e);
  }
}

export async function saveSettings(patch: (s: Settings) => Settings) {
  const cur = get().settings;
  if (!cur) return;
  const next = patch(structuredClone(cur));
  set({ settings: next });
  try {
    const saved = await api.settingsSet(next);
    set({ settings: saved });
    applyAppearance(saved);
    return saved;
  } catch (e) {
    set({ settings: cur });
    errorToast(e);
  }
}

// ---------- editing ----------

export function setPipeline(p: Pipeline, record = true) {
  set((s) => ({
    pipeline: p,
    history: record ? { past: [...s.history.past.slice(-99), s.pipeline], future: [] } : s.history,
    still: null,
    clip: null,
    stillError: null,
    clipError: null,
    dirty: true,
  }));
}

export function undo() {
  const { history, pipeline } = get();
  const prev = history.past[history.past.length - 1];
  if (!prev) return;
  set({ pipeline: prev, history: { past: history.past.slice(0, -1), future: [pipeline, ...history.future] }, still: null, clip: null, dirty: true });
}

export function redo() {
  const { history, pipeline } = get();
  const next = history.future[0];
  if (!next) return;
  set({ pipeline: next, history: { past: [...history.past, pipeline], future: history.future.slice(1) }, still: null, clip: null, dirty: true });
}

export function applyPreset(p: Preset) {
  setPipeline(structuredClone(p.pipeline));
  set((s) => ({ output: p.output ? structuredClone(p.output) : s.output, presetName: p.name }));
}

export function clearPipeline() {
  setPipeline({ stages: [] });
  set({ presetName: null });
}

/** Turn an operation on, adding it with defaults at its canonical position if needed. */
export function enableOp(type: OperationType, on = true) {
  const { pipeline, defaults } = get();
  const stage = pipeline.stages.find((s) => s.type === type);
  if (stage) setPipeline({ stages: pipeline.stages.map((s) => (s.type === type ? { ...s, enabled: on } : s)) });
  else if (on && defaults) setPipeline(addStageCanonical(pipeline, type, defaults[type].settings));
}

export function currentJob(overrides: Partial<NewJob> = {}): NewJob | null {
  const s = get();
  if (!s.sourcePath || !s.output) return null;
  return {
    source: s.sourcePath,
    pipeline: s.pipeline,
    output: s.output,
    range: s.range,
    folder: s.exportFolder,
    file_name: s.exportFileName,
    remember_folder: !!s.settings?.remember_last_export_folder,
    preset_name: s.presetName,
    batch_id: null,
    engine: s.engine,
    ...overrides,
  };
}

// ---------- sources ----------

export async function openSource(path: string) {
  set((s) => ({ sourcePlay: null, sourcePath: path, source: null, sourceLoading: true, sourceError: null, still: null, clip: null, stillError: null, clipError: null, playhead: 0, range: null, sources: s.sources.includes(path) ? s.sources : [...s.sources, path] }));
  try {
    const info = await api.sourceInfo(path, true);
    if (get().sourcePath !== path) return;
    set({ source: info, sourceLoading: false, playhead: Math.min(info.info.duration / 3, info.info.duration) });
    const il = info.interlace;
    const hasDeint = get().pipeline.stages.some((s) => s.type === "deinterlace");
    if (il && il.tff + il.bff > il.progressive && !hasDeint) toast("warning", "Interlaced video detected.", { label: "Deinterlace", run: () => enableOp("deinterlace") });
  } catch (e) {
    if (get().sourcePath !== path) return;
    set({ sourceLoading: false, sourceError: e instanceof Error ? e.message : String(e) });
  }
}

const PLAY_CHUNK = 10;

/** Play the source from the playhead using browser-playable proxy copies, 10 s at a time. */
export async function playSource() {
  const s = get();
  if (!s.sourcePath || !s.source || s.sourcePlayLoading || s.app?.demo) return;
  const dur = s.source.info.duration;
  const start = s.playhead >= dur - 0.05 ? 0 : s.playhead;
  set({ sourcePlayLoading: true });
  try {
    const src = await api.sourceProxy(s.sourcePath, start, Math.min(PLAY_CHUNK, dur - start), Math.min(720, s.settings?.preview_max_height ?? 720));
    if (get().sourcePath === s.sourcePath) set({ sourcePlay: { src, start }, playhead: start });
  } catch (e) {
    errorToast(e);
  } finally {
    set({ sourcePlayLoading: false });
  }
}

export function stopSource() {
  set({ sourcePlay: null });
}

export function closeSource() {
  set((s) => ({ sources: s.sources.filter((x) => x !== s.sourcePath), sourcePath: null, source: null, sourceError: null, still: null, clip: null, range: null, projectPath: null, projectName: null, dirty: false }));
}

let stillSeq = 0;
export async function renderStill() {
  const s = get();
  if (!s.sourcePath || !s.output) return;
  const my = ++stillSeq;
  const at = s.playhead;
  set({ stillBusy: true, stillError: null });
  try {
    const r = await api.previewStill(s.sourcePath, at, s.pipeline, s.output, s.engine);
    if (my !== stillSeq) return;
    set({ still: r, stillAt: at, stillBusy: false, compare: s.compare === "original" ? "split" : s.compare });
  } catch (e) {
    if (my !== stillSeq) return;
    set({ stillBusy: false, stillError: e instanceof Error ? e.message : String(e) });
  }
}

export async function renderClip(seconds?: number) {
  const s = get();
  const job = currentJob();
  if (!job || !s.settings) return;
  const dur = Math.min(seconds ?? s.settings.preview_seconds, Math.max(0.1, (s.source?.info.duration ?? 0) - s.playhead));
  set({ clipError: null, clipRange: { start: s.playhead, end: s.playhead + dur } });
  try {
    const r = await api.previewClip(job, s.playhead, dur);
    set({ clip: r });
  } catch (e) {
    set({ clipError: e instanceof Error ? e.message : String(e) });
  }
}
