// In-memory demo backend used only when the UI runs in a plain browser
// (UI development and automated UI tests). It never processes video: the UI
// shows a "demo mode" banner whenever this backend is active.

import type {
  AppEvent, CacheArea, JobRecord, JobState, ModelCard, NewJob, Preset, Settings, StorageOverview,
} from "./types";
import modelsFixture from "./fixtures/models.json";
import presetsFixture from "./fixtures/presets.json";
import settingsFixture from "./fixtures/settings.json";
import opDefaultsFixture from "./fixtures/operation_defaults.json";

type Handler = (e: AppEvent) => void;
const listeners = new Set<Handler>();
const emit = (e: AppEvent) => listeners.forEach((h) => h(e));

export function mockListen(h: Handler): () => void {
  listeners.add(h);
  return () => listeners.delete(h);
}

declare global {
  interface Window {
    __cfMockPick?: string[][];
    __cfMockSave?: (string | null)[];
    __cfMock?: typeof state;
  }
}

export async function mockPick(_title: string, folder: boolean, _multiple: boolean): Promise<string[]> {
  const q = window.__cfMockPick;
  if (q && q.length) return q.shift()!;
  return [folder ? "/home/demo/ClearFrame Exports" : "/home/demo/Videos/holiday clip.mp4"];
}
export async function mockSave(defaultPath: string): Promise<string | null> {
  const q = window.__cfMockSave;
  if (q && q.length) return q.shift()!;
  return defaultPath;
}

function svg(label: string, hue: number, w = 640, h = 360): string {
  const s = `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="hsl(${hue},55%,35%)"/><stop offset="1" stop-color="hsl(${(hue + 60) % 360},60%,18%)"/></linearGradient></defs>
<rect width="100%" height="100%" fill="url(#g)"/><circle cx="${w * 0.68}" cy="${h * 0.42}" r="${h * 0.18}" fill="hsla(${(hue + 30) % 360},70%,70%,.5)"/>
<text x="24" y="${h - 24}" font-family="sans-serif" font-size="22" fill="#fff" opacity=".85">${label}</text></svg>`;
  return "data:image/svg+xml;charset=utf-8," + encodeURIComponent(s);
}

export function mockFileUrl(path: string): string {
  if (path.startsWith("mock://")) {
    const [kind, arg] = path.slice(7).split("/");
    const t = parseFloat(arg || "0");
    if (kind === "processed") return svg(`Processed (demo) ${t.toFixed(2)}s`, 200 + t * 7);
    if (kind === "thumb") return svg("", 190 + t * 9, 160, 90);
    return svg(`Original (demo) ${t.toFixed(2)}s`, 200 + t * 7);
  }
  return "";
}

const now = () => new Date().toISOString();
let seq = 1;

const defaultSettings = (): Settings => structuredClone(settingsFixture) as unknown as Settings;

const state = {
  settings: defaultSettings(),
  models: structuredClone(modelsFixture) as unknown as ModelCard[],
  presets: structuredClone(presetsFixture) as unknown as Preset[],
  jobs: [] as JobRecord[],
  queuePaused: false,
  modelsDir: "/home/demo/.local/share/ClearFrame/models",
  cacheDir: "/home/demo/.cache/ClearFrame",
  cacheBytes: { previews: 120e6, thumbnails: 8e6, proxies: 300e6, downloads: 0, temp: 2e6, jobs: 50e6 } as Record<CacheArea, number>,
  recent: [] as { path: string; kind: string; title: string; opened_at: string }[],
  timers: new Map<string, number>(),
};
if (typeof window !== "undefined") window.__cfMock = state;

function resolvePlan(p: Settings["performance"]) {
  const cores = 8, ram = 16384;
  const threads = p.mode === "balanced" ? 6 : p.mode === "high_performance" ? 7 : p.mode === "maximum" ? 8 : Math.max(1, p.cpu_threads || cores - p.reserve_cpu_threads);
  const reserve = p.memory_preset === "more_for_others" ? ram / 2 : p.memory_preset === "balanced" ? 4096 : p.memory_preset === "prioritize_clear_frame" ? 2048 : ram - p.memory_budget_mb;
  const budget = p.memory_preset === "custom" ? p.memory_budget_mb : ram - reserve;
  return {
    threads, priority: p.mode === "balanced" ? "below_normal" : p.mode === "custom" ? p.priority : "normal", memory_budget_mb: budget,
    reserve_memory_mb: reserve, max_concurrent_jobs: p.max_concurrent_jobs,
    explanation: [`Demo machine: ${threads} of ${cores} CPU threads.`, `Processing memory budget about ${budget} MB; about ${reserve} MB left for other applications.`],
  };
}

function modelId(c: ModelCard) {
  return c.manifest?.id ?? c.installed?.id ?? "";
}

function setModel(id: string, f: (c: ModelCard) => void) {
  const c = state.models.find((m) => modelId(m) === id);
  if (c) f(c);
  emit({ type: "model_changed", id });
}

function startInstall(id: string) {
  const c = state.models.find((m) => modelId(m) === id);
  if (!c || !c.manifest) throw "unknown model";
  if (c.in_progress) throw "already installing";
  const total = c.manifest.download_size_bytes;
  let done = c.partial_bytes || 0;
  c.in_progress = true;
  c.state = "downloading";
  const t = window.setInterval(() => {
    done = Math.min(total, done + Math.max(total / 12, 1));
    c.partial_bytes = done;
    emit({ type: "model_progress", progress: { id, state: "downloading", role: "weights", bytes_done: done, bytes_total: total, message: "Downloading" } });
    if (done >= total) {
      window.clearInterval(t);
      state.timers.delete(id);
      emit({ type: "model_progress", progress: { id, state: "testing", role: null, bytes_done: total, bytes_total: total, message: "Running a smoke test" } });
      window.setTimeout(() => {
        setModel(id, (m) => {
          m.in_progress = false;
          m.state = "installed";
          m.partial_bytes = 0;
          m.disk_bytes = m.manifest!.installed_size_bytes;
          m.path = `${state.modelsDir}/other/${id}`;
          m.installed = {
            id, version: m.manifest!.version, state: "installed", architecture: m.manifest!.architecture, dir: m.path, files: {},
            installed_size: m.disk_bytes, installed_at: now(), verified_at: now(), smoke_test: "demo", error: null, custom: false, custom_info: null,
          };
        });
      }, 300);
    }
  }, 150);
  state.timers.set(id, t);
}

function job(id: string) {
  const j = state.jobs.find((x) => x.id === id);
  if (!j) throw "job not found";
  return j;
}

function setJob(j: JobRecord, s: JobState, extra: Partial<JobRecord> = {}) {
  Object.assign(j, { state: s, updated_at: now() }, extra);
  emit({ type: "job_updated", job: { ...j } });
}

function runJob(j: JobRecord) {
  setJob(j, "processing", { started_at: j.started_at ?? now() });
  const t = window.setInterval(() => {
    if (j.state !== "processing") {
      window.clearInterval(t);
      return;
    }
    j.progress = Math.min(1, j.progress + 0.05);
    const total = 300;
    emit({ type: "job_progress", id: j.id, progress: { pass: 1, passes: 1, phase: "processing", frames_done: Math.round(j.progress * total), frames_total: total, fraction: j.progress, fps: 12.5, eta: [(1 - j.progress) * 20, (1 - j.progress) * 26] } });
    emit({ type: "job_resources", id: j.id, sample: { process_memory_mb: 1800, process_cpu_percent: 340, system_memory_used_mb: 9000, system_memory_total_mb: 16384, system_memory_available_mb: 7384, backend: "cpu", stage: "enhance", free_disk_mb: 120000 } });
    if (j.progress >= 1) {
      window.clearInterval(t);
      setJob(j, "completed", { finished_at: now(), progress: 1 });
      schedule();
    }
  }, 200);
}

function schedule() {
  if (state.queuePaused) return;
  if (state.jobs.some((j) => ["processing", "preparing", "encoding", "validating"].includes(j.state))) return;
  const next = state.jobs.filter((j) => j.state === "queued").sort((a, b) => a.queue_order - b.queue_order)[0];
  if (next) runJob(next);
  emit({ type: "queue_state", paused: state.queuePaused, running: state.jobs.filter((j) => j.state === "processing").length, queued: state.jobs.filter((j) => j.state === "queued").length });
}

function baseName(p: string) {
  return p.split(/[\\/]/).pop() ?? p;
}

function makeJob(n: NewJob, kind: "render" | "preview" = "render"): JobRecord {
  const id = `demo-${seq++}`;
  const folder = n.folder ?? "/home/demo/Videos/ClearFrame";
  const name = n.file_name ?? baseName(n.source).replace(/\.[^.]+$/, "") + "_enhanced." + n.output.container;
  const j: JobRecord = {
    id, kind, name, state: "queued", source: n.source, destination: `${folder}/${name}`, spec_json: JSON.stringify({ pipeline: n.pipeline }),
    created_at: now(), updated_at: now(), started_at: null, finished_at: null, progress: 0, progress_json: null, detail: null, error: null,
    report_json: null, queue_order: state.jobs.length, attempts: 0, batch_id: n.batch_id, heartbeat: null, warnings: [],
  };
  state.jobs.push(j);
  emit({ type: "job_updated", job: { ...j } });
  window.setTimeout(schedule, 50);
  return j;
}

function storage(): StorageOverview {
  const areas: { area: CacheArea; label: string }[] = [
    { area: "previews", label: "Preview renders" }, { area: "thumbnails", label: "Thumbnails" }, { area: "proxies", label: "Playback proxies" },
    { area: "downloads", label: "Partial model downloads" }, { area: "temp", label: "Temporary files" }, { area: "jobs", label: "Job work files (segments, intermediates)" },
  ];
  const per: Record<string, number> = {};
  for (const m of state.models) if (m.state === "installed") per[modelId(m)] = m.disk_bytes;
  const modelsTotal = Object.values(per).reduce((a, b) => a + b, 0);
  const cacheTotal = Object.values(state.cacheBytes).reduce((a, b) => a + b, 0);
  return {
    paths: {
      data_dir: "/home/demo/.local/share/ClearFrame", models_dir: state.modelsDir, cache_dir: state.cacheDir, logs_dir: "/home/demo/.local/share/ClearFrame/logs",
      config_dir: "/home/demo/.config/ClearFrame", models_is_default: state.modelsDir.endsWith("ClearFrame/models"), cache_is_default: state.cacheDir.endsWith(".cache/ClearFrame"),
    },
    models: { models_dir: state.modelsDir, models_dir_exists: true, total_bytes: modelsTotal, per_model: per, staging_bytes: 0, free_bytes: 120e9 },
    cache: {
      cache_dir: state.cacheDir, total_bytes: cacheTotal, free_bytes: 120e9, low_space: false,
      areas: areas.map((a) => ({ ...a, bytes: state.cacheBytes[a.area], entries: state.cacheBytes[a.area] > 0 ? 3 : 0, protected_bytes: a.area === "jobs" ? 50e6 : 0 })),
    },
    data_bytes: 2e6, data_free: 120e9, warnings: [], relocation_in_progress: false,
  };
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export async function mockInvoke(cmd: string, a: Record<string, any> = {}): Promise<any> {
  await sleep(10);
  switch (cmd) {
    case "app_info":
      return {
        version: "1.0.0", os: "browser", arch: "demo", platform: "demo", ffmpeg: { version: "demo", path: "(none)", source: "demo", license: "n/a", vidstab: true, zscale: true },
        ffmpeg_error: null, queue_paused: state.queuePaused, compiled_backends: ["cpu"], demo: true,
      };
    case "settings_get":
      return structuredClone(state.settings);
    case "settings_set":
      state.settings = structuredClone(a.settings);
      return structuredClone(state.settings);
    case "settings_reset":
      state.settings = { ...defaultSettings(), onboarding_complete: true };
      return structuredClone(state.settings);
    case "hardware":
      return {
        hardware: {
          os: "demo", os_version: "Browser demo", arch: "demo", cpu_brand: "Demo CPU (8 threads)", logical_cores: 8, physical_cores: 4, ram_total_mb: 16384, ram_available_mb: 9000,
          gpus: [{ vendor: "Demo", name: "Demo GPU", memory_total_mb: 8192, memory_free_mb: 7000, driver: null, source: "demo" }],
          backends: [{ kind: "cpu", compiled: true, available: true, detail: "device created" }, { kind: "cuda", compiled: false, available: false, detail: "not included in this build" }, { kind: "metal", compiled: false, available: false, detail: "not included in this build" }],
          precisions: ["fp32"], cpu_features: ["avx2"],
        },
        plan: resolvePlan(state.settings.performance), probe: null,
      };
    case "system_sample": {
      const cpu = 20 + Math.random() * 30;
      return {
        cpu_percent: cpu, logical_cores: 8, memory_total_mb: 16384, memory_used_mb: 7000 + Math.random() * 600, memory_available_mb: 9000, clearframe_memory_mb: 300 + Math.random() * 50,
        clearframe_cpu_percent: cpu / 3, gpus: [{ name: "Demo GPU", utilization_percent: null, memory_used_mb: null, memory_total_mb: 8192, temperature_c: null, measured: false }],
        max_temperature_c: null, thermal_warning: null, memory_pressure: false,
      };
    }
    case "codecs":
      return [
        ["libx264", "H.264 (x264)", "h264", false, ["yuv420p", "yuv422p", "yuv444p", "yuv420p10le"], ["ultrafast", "veryfast", "fast", "medium", "slow", "slower", "veryslow"], "medium", [0, 18, 51], ["mp4", "mov", "mkv", "m4v"], true],
        ["libx265", "H.265 / HEVC (x265)", "hevc", false, ["yuv420p", "yuv420p10le", "yuv444p10le"], ["fast", "medium", "slow"], "medium", [0, 20, 51], ["mp4", "mov", "mkv"], true],
        ["libsvtav1", "AV1 (SVT-AV1)", "av1", false, ["yuv420p", "yuv420p10le"], ["4", "6", "8", "10"], "8", [1, 30, 63], ["mp4", "mkv", "webm"], true],
        ["prores_ks", "Apple ProRes", "prores", false, ["yuv422p10le", "yuv444p10le"], [], null, null, ["mov", "mkv"], true],
        ["ffv1", "FFV1 (lossless archive)", "ffv1", false, ["yuv444p16le", "yuv420p"], [], null, null, ["mkv"], true],
        ["h264_nvenc", "H.264 (NVIDIA NVENC)", "h264", true, ["yuv420p"], ["p1", "p4", "p7"], "p4", [0, 21, 51], ["mp4", "mov", "mkv"], false],
      ].map(([encoder, label, family, hardware, pix_fmts, presets, default_preset, quality_range, containers, available]) => ({
        profile: { encoder, label, family, hardware, pix_fmts, presets, default_preset, quality_range, supports_bitrate: encoder !== "ffv1", supports_lossless: encoder === "ffv1" || encoder === "libx264", containers, description: `${label} encoder` },
        available,
      }));
    case "operation_defaults":
      return structuredClone(opDefaultsFixture);
    case "source_info": {
      const path: string = a.path;
      if (/missing/i.test(path)) throw `could not read ${path}: No such file`;
      state.recent = [{ path, kind: "video", title: baseName(path), opened_at: now() }, ...state.recent.filter((r) => r.path !== path)].slice(0, 10);
      const interlaced = /interlaced|dv/i.test(path);
      return {
        info: {
          path, container: "mov,mp4,m4a,3gp,3g2,mj2", container_long: "QuickTime / MOV", duration: 12.5, size_bytes: 24e6, bit_rate: 15e6,
          video: {
            index: 0, codec: "h264", codec_long: "H.264", width: 1280, height: 720, display_width: 1280, display_height: 720, pix_fmt: "yuv420p", bit_depth: 8,
            frame_rate: { num: 30000, den: 1001 }, avg_frame_rate: { num: 30000, den: 1001 }, variable_frame_rate: false, frame_count: 375, field_order: interlaced ? "tt" : "progressive",
            rotation: 0, start_time: 0, color: { range: "tv", space: "bt709", transfer: "bt709", primaries: "bt709" },
          },
          audio: [{ index: 1, codec: "aac", channels: 2, channel_layout: "stereo", sample_rate: 48000 }], subtitles: [], chapters: 0,
        },
        fingerprint: "demo", thumbnails: Array.from({ length: 16 }, (_, i) => [i * 0.78, `mock://thumb/${i}`]), browser_playable: false,
        interlace: a.detectInterlace ? { tff: interlaced ? 300 : 0, bff: 0, progressive: interlaced ? 0 : 300, undetermined: 0 } : null,
      };
    }
    case "source_frame":
      return `mock://original/${a.time}`;
    case "source_proxy":
      return `mock://proxy/${a.start}`;
    case "preview_still": {
      await sleep(400);
      const missing = (a.pipeline.stages as any[]).filter((s) => s.enabled && s.settings?.model_id).map((s) => s.settings.model_id)
        .filter((id: string) => state.models.find((m) => modelId(m) === id)?.state !== "installed");
      if (missing.length) throw `Install ${missing.join(", ")} in the Model Library first.`;
      return { key: `k${a.time}`, original: `mock://original/${a.time}`, processed: `mock://processed/${a.time}`, original_size: [1280, 720], processed_size: [2560, 1440], seconds: 0.4, notes: ["Demo mode: images are placeholders."], cached: false };
    }
    case "preview_clip": {
      const j = makeJob(a.job, "preview");
      return { key: j.id, path: null, job_id: j.id, original_proxy: `mock://proxy/${a.start}` };
    }
    case "check_export": {
      const n: NewJob = a.job;
      const issues: any[] = [];
      for (const s of n.pipeline.stages) {
        const id = s.enabled ? s.settings?.model_id : null;
        if (id && state.models.find((m) => modelId(m) === id)?.state !== "installed") issues.push({ severity: "error", stage_id: s.id, message: `The model ${id} is not installed.`, action: "Install it in the Model Library." });
      }
      if (!n.pipeline.stages.some((s) => s.enabled)) issues.push({ severity: "warning", stage_id: null, message: "No processing stage is enabled: the video will only be re-encoded.", action: null });
      const fmt = { width: 1280, height: 720, fps: { num: 30000, den: 1001 }, time_scale: 1, interlaced: false };
      const ups = n.pipeline.stages.some((s) => s.enabled && s.type === "enhance");
      const out = ups ? { ...fmt, width: 2560, height: 1440 } : fmt;
      return {
        plan: {
          input: fmt, output: out, issues, passes: [], cfr_conversion: null, input_duration: 12.5, output_duration: 12.5,
          stages: n.pipeline.stages.filter((s) => s.enabled).map((s, i) => ({ stage_id: s.id, index: i, label: s.type, exec: { kind: "native" }, input: fmt, output: s.type === "enhance" ? out : fmt, notes: [], models: s.settings?.model_id ? [s.settings.model_id] : [] })),
          estimates: { peak_memory_mb: [900, 1600], output_size_mb: [40, 90], work_disk_mb: [200, 400], output_frames: 375, input_frames: 375 },
        },
        destination: `${n.folder ?? "/home/demo/Videos/ClearFrame"}/${n.file_name ?? baseName(n.source).replace(/\.[^.]+$/, "") + "_enhanced." + n.output.container}`,
        destination_folder_ok: { Ok: null }, free_bytes: 120e9,
      };
    }
    case "enqueue":
      return makeJob(a.job);
    case "enqueue_batch": {
      const batch = `batch-${seq++}`;
      return { created: (a.sources as string[]).map((s) => makeJob({ ...a.template, source: s, batch_id: batch, file_name: null })), failed: [] };
    }
    case "collect_videos":
      return a.paths;
    case "default_export_folder":
      return "/home/demo/Videos/ClearFrame";
    case "jobs_list":
      return state.jobs.filter((j) => a.includePreviews || j.kind === "render").map((j) => ({ ...j }));
    case "job_events":
      return [{ at: now(), level: "info", message: "Demo job (no real processing)." }];
    case "job_action": {
      const j = job(a.id);
      if (a.action === "pause") setJob(j, "paused");
      else if (a.action === "resume" || a.action === "retry") {
        setJob(j, "queued", { error: null });
        schedule();
      } else if (a.action === "cancel") setJob(j, "cancelled");
      else if (a.action === "remove") {
        state.jobs = state.jobs.filter((x) => x.id !== j.id);
        emit({ type: "job_removed", id: j.id });
      } else if (a.action === "reorder") {
        const q = state.jobs.filter((x) => x.state === "queued" && x.id !== j.id).sort((x, y) => x.queue_order - y.queue_order);
        q.splice(a.index ?? 0, 0, j);
        q.forEach((x, i) => (x.queue_order = i));
        emit({ type: "job_updated", job: { ...j } });
      }
      return null;
    }
    case "queue_action":
      if (a.action === "pause") state.queuePaused = true;
      if (a.action === "resume") {
        state.queuePaused = false;
        schedule();
      }
      if (a.action === "clear_finished") {
        const n = state.jobs.length;
        state.jobs = state.jobs.filter((j) => !["completed", "cancelled", "failed"].includes(j.state));
        return n - state.jobs.length;
      }
      emit({ type: "queue_state", paused: state.queuePaused, running: 0, queued: state.jobs.filter((j) => j.state === "queued").length });
      return 0;
    case "models_list":
      return structuredClone(state.models);
    case "model_install":
      startInstall(a.id);
      return null;
    case "model_action": {
      const id: string = a.id;
      switch (a.action) {
        case "pause":
        case "cancel": {
          const t = state.timers.get(id);
          if (t) window.clearInterval(t);
          state.timers.delete(id);
          setModel(id, (m) => {
            m.in_progress = false;
            m.state = a.action === "pause" ? "paused" : "not_installed";
            if (a.action === "cancel") m.partial_bytes = 0;
          });
          return true;
        }
        case "remove": {
          if (state.jobs.some((j) => !["completed", "cancelled", "failed"].includes(j.state) && j.spec_json.includes(id))) throw "This model is used by a queued or running job.";
          let freed = 0;
          setModel(id, (m) => {
            freed = m.disk_bytes;
            m.state = "not_installed";
            m.installed = null;
            m.disk_bytes = 0;
            m.path = null;
          });
          return { freed };
        }
        case "verify":
          return { id, ok: true, problems: [] };
        case "repair":
          return { ok_without_reinstall: true };
        case "usage":
          return state.presets.filter((p) => JSON.stringify(p.pipeline).includes(id)).map((p) => `Preset: ${p.name}`);
        case "in_use":
          return false;
        case "open":
          return true;
      }
      throw `unknown action ${a.action}`;
    }
    case "model_import_custom":
      return "custom.demo";
    case "storage_overview":
      return storage();
    case "storage_plan": {
      const s = storage();
      const src = a.kind === "models" ? s.paths.models_dir : s.paths.cache_dir;
      const total = a.kind === "models" ? s.models.total_bytes : s.cache.total_bytes;
      const problems = a.dest.startsWith(src) ? ["The destination is inside the current folder."] : [];
      return { kind: a.kind, plan: { source: src, destination: a.dest, total_bytes: total, file_count: 12, destination_free: 200e9, same_filesystem: false, problems }, disposable_bytes: a.kind === "cache" ? total - state.cacheBytes.jobs : 0, notes: ["Demo mode: nothing is moved."] };
    }
    case "storage_relocate": {
      const s = storage();
      const total = a.kind === "models" ? s.models.total_bytes || 1e8 : s.cache.total_bytes;
      let done = 0;
      const t = window.setInterval(() => {
        done = Math.min(total, done + total / 8);
        emit({ type: "storage_progress", kind: a.kind, bytes_done: done, bytes_total: total });
        if (done >= total) {
          window.clearInterval(t);
          if (a.kind === "models") state.modelsDir = a.dest;
          else state.cacheDir = a.dest;
          emit({ type: "storage_done", kind: a.kind, ok: true, message: `Moved to ${a.dest}.` });
        }
      }, 120);
      return null;
    }
    case "storage_cancel":
      return null;
    case "storage_reset":
      if (a.kind === "models") state.modelsDir = "/home/demo/.local/share/ClearFrame/models";
      else state.cacheDir = "/home/demo/.cache/ClearFrame";
      return null;
    case "cache_clear": {
      let freed = 0;
      for (const ar of a.areas as CacheArea[]) {
        const keep = ar === "jobs" && !a.includeResumable ? 50e6 : 0;
        freed += state.cacheBytes[ar] - keep;
        state.cacheBytes[ar] = keep;
      }
      return { freed_bytes: freed, removed_entries: 4, skipped_protected: a.includeResumable ? 0 : 1, errors: [] };
    }
    case "cache_enforce":
      return { freed_bytes: 0, removed_entries: 0, skipped_protected: 0, errors: [] };
    case "open_path":
      return null;
    case "presets_list":
      return structuredClone(state.presets);
    case "preset_action": {
      const p = state.presets.find((x) => x.id === a.id);
      switch (a.action) {
        case "save": {
          const np = { ...a.preset, id: a.preset.id || `user.${seq++}`, builtin: false };
          state.presets = [...state.presets.filter((x) => x.id !== np.id), np];
          return np;
        }
        case "duplicate": {
          const np = { ...structuredClone(p!), id: `user.${seq++}`, name: `${p!.name} (copy)`, builtin: false };
          state.presets.push(np);
          return np;
        }
        case "rename":
          if (p!.builtin) throw "Built-in presets cannot be renamed. Duplicate it first.";
          p!.name = a.name;
          return p;
        case "delete":
          if (p!.builtin) throw "Built-in presets cannot be deleted.";
          state.presets = state.presets.filter((x) => x.id !== a.id);
          return true;
        case "favorite":
          p!.favorite = a.favorite;
          return true;
        case "export":
          return true;
        case "import":
          return { preset: { ...state.presets[0], id: `user.${seq++}`, name: "Imported preset", builtin: false }, missing_models: [] };
      }
      throw `unknown action ${a.action}`;
    }
    case "new_project":
      return {
        format: "clearframe-project", schema_version: 1, name: a.name, source: { path: a.source, fingerprint: "", size_bytes: 0 }, pipeline: { stages: [] },
        output: (await mockInvoke("operation_defaults")).output, range: null, preview_range: null, export: { folder: null, file_name: null },
        engine: { backend: null, precision: "fp32" }, models: {}, job_ids: [], preset_id: null, notes: "", created_at: now(), updated_at: now(),
      };
    case "project_save":
      state.recent.unshift({ path: a.req.path, kind: "project", title: a.req.project.name, opened_at: now() });
      (state as any).projects = { ...((state as any).projects ?? {}), [a.req.path]: a.req.project };
      return a.req.project;
    case "project_open": {
      const p = (state as any).projects?.[a.path];
      if (!p) throw "Not a ClearFrame project (demo mode can only reopen projects saved in this session).";
      return { project: p, source_found: true, source_changed: false, suggested_source: null, missing_models: [], migrated_from: null, backup_path: null };
    }
    case "recent_list":
      return state.recent;
    case "recent_action":
      if (a.action === "clear") state.recent = [];
      else state.recent = state.recent.filter((r) => r.path !== a.path);
      return null;
    case "diagnostics":
      return `ClearFrame diagnostics (demo mode)\nredacted: ${a.redact}\n`;
    case "write_text_file":
      return null;
    case "check_update":
      return { current: "1.0.0", latest: "1.0.0", newer: false, url: "", notes: "" };
    case "update_catalog":
      return "Demo mode: the catalog was not changed.";
    case "refresh_tools":
    case "allow_asset":
      return null;
  }
  throw `demo backend: unknown command ${cmd}`;
}
