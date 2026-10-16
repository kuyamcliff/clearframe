import { useEffect, useRef, useState } from "react";
import { Activity, Cpu, Database, FileText, FolderOpen, Info, Settings2, Upload, Wrench } from "lucide-react";

import { Check, Confirm, Dialog, F, NumberInput, Progress, Range, Select, Seg, useInterval } from "../components/ui";
import { api, pickFolder, pickSave } from "../lib/ipc";
import { bytes, mb } from "../lib/format";
import type { CacheArea, ClearResult, MemoryPreset, PerformanceMode, RelocationPlan, Settings, StorageOverview, SystemSample } from "../lib/types";
import { errorToast, saveSettings, toast, useStore, type SettingsSection } from "../state/store";

type Patch = (f: (s: Settings) => Settings) => void;

const SECTIONS: { key: SettingsSection; label: string; icon: typeof Info }[] = [
  { key: "general", label: "General", icon: Settings2 },
  { key: "export", label: "Output", icon: Upload },
  { key: "storage", label: "Storage", icon: Database },
  { key: "performance", label: "Performance", icon: Cpu },
  { key: "advanced", label: "Advanced", icon: Wrench },
  { key: "diagnostics", label: "Diagnostics", icon: FileText },
  { key: "about", label: "About", icon: Info },
];

function General({ s, patch }: { s: Settings; patch: Patch }) {
  return (
    <div className="prefs">
      <div className="pref-h">Appearance</div>
      <F label="Theme"><Seg label="Theme" value={s.theme} onChange={(v) => patch((x) => ({ ...x, theme: v }))} options={[{ value: "dark", label: "Dark" }, { value: "light", label: "Light" }, { value: "system", label: "System" }]} /></F>
      <F label="Animations"><Seg label="Animations" value={s.motion} onChange={(v) => patch((x) => ({ ...x, motion: v }))} options={[{ value: "system", label: "System" }, { value: "reduced", label: "Reduced" }, { value: "full", label: "Full" }]} /></F>
      <F label="Interface size"><Range label="Interface size" value={s.ui_scale} min={0.75} max={2} step={0.05} scale={100} onChange={(v) => patch((x) => ({ ...x, ui_scale: v }))} /></F>
      <F label="Language"><Select label="Language" value={s.language} onChange={(v) => patch((x) => ({ ...x, language: v }))} options={[{ value: "en", label: "English" }]} /></F>
      <div className="pref-h">Behaviour</div>
      <F label="Notifications"><Check checked={s.notifications} onChange={(v) => patch((x) => ({ ...x, notifications: v }))} label="When exports finish in the background" /></F>
      <F label="Closing"><Check checked={s.confirm_before_quit_with_jobs} onChange={(v) => patch((x) => ({ ...x, confirm_before_quit_with_jobs: v }))} label="Ask while exports are running" /></F>
      <F label="Start dialog"><Check checked={!s.hide_start_dialog} onChange={(v) => patch((x) => ({ ...x, hide_start_dialog: !v }))} label="Show after opening a video" /></F>
      <div className="pref-h">Network</div>
      <F label="Updates"><Check checked={s.check_updates_on_launch} onChange={(v) => patch((x) => ({ ...x, check_updates_on_launch: v }))} label="Check for updates automatically" /></F>
      <F label="Model catalog"><Check checked={s.allow_catalog_updates} onChange={(v) => patch((x) => ({ ...x, allow_catalog_updates: v }))} label="Allow signed updates" /></F>
    </div>
  );
}

function Output({ s, patch }: { s: Settings; patch: Patch }) {
  const [def, setDef] = useState("");
  useEffect(() => { api.defaultExportFolder().then(setDef).catch(() => {}); }, []);
  return (
    <div className="prefs">
      <div className="pref-h">Exports</div>
      <F label="Default folder">
        <div className="row">
          <span className="path grow ellipsis" title={s.output_dir ?? def}>{s.output_dir ?? def}</span>
          <button className="btn sm" onClick={async () => { const f = await pickFolder("Default export folder", s.output_dir ?? def); if (f) patch((x) => ({ ...x, output_dir: f })); }}>Browse</button>
          {s.output_dir && <button className="btn sm ghost" onClick={() => patch((x) => ({ ...x, output_dir: null }))}>Reset</button>}
        </div>
      </F>
      <F label="Remember last folder"><Check checked={s.remember_last_export_folder} onChange={(v) => patch((x) => ({ ...x, remember_last_export_folder: v }))} label="" /></F>
      <F label="File name" htmlFor="nt" title="{name} {task} {model} {preset} {width} {height} {fps} {date}">
        <input id="nt" className="input" value={s.naming_template} onChange={(e) => patch((x) => ({ ...x, naming_template: e.target.value }))} />
      </F>
      <F label="Sidecar report"><Check checked={s.write_sidecar_report} onChange={(v) => patch((x) => ({ ...x, write_sidecar_report: v }))} label=".clipfx.json next to each export" /></F>
      <div className="pref-h">Previews</div>
      <F label="Default clip length"><Range label="Preview seconds" value={s.preview_seconds} min={0.5} max={30} step={0.5} onChange={(v) => patch((x) => ({ ...x, preview_seconds: v }))} /></F>
      <F label="Playback copies">
        <Select label="Proxy height" value={String(s.preview_max_height)} onChange={(v) => patch((x) => ({ ...x, preview_max_height: Number(v) }))} options={["540", "720", "1080", "1440", "2160"].map((h) => ({ value: h, label: `${h}p` }))} />
      </F>
    </div>
  );
}

function Relocate({ kind, overview, onDone }: { kind: "models" | "cache"; overview: StorageOverview; onDone: () => void }) {
  const [dest, setDest] = useState<string | null>(null);
  const [plan, setPlan] = useState<RelocationPlan | null>(null);
  const [mode, setMode] = useState<"move" | "clear_disposable_and_move" | "switch_only">(kind === "cache" ? "clear_disposable_and_move" : "move");
  const [started, setStarted] = useState(false);
  const progress = useStore((s) => s.storageProgress);
  const running = useStore((s) => s.jobs.some((j) => ["preparing", "processing", "encoding", "validating", "previewing"].includes(j.state)));
  const src = kind === "models" ? overview.paths.models_dir : overview.paths.cache_dir;
  const choose = async () => {
    const f = await pickFolder(kind === "models" ? "New models folder" : "New cache folder");
    if (!f) return;
    setDest(f);
    setPlan(null);
    api.storagePlan(kind, f).then(setPlan).catch(errorToast);
  };
  const start = async () => {
    if (!dest) return;
    useStore.setState({ storageProgress: { kind, done: 0, total: plan?.plan.total_bytes ?? 0 } });
    try { await api.storageRelocate(kind, dest, mode); setStarted(true); } catch (e) { errorToast(e); useStore.setState({ storageProgress: null }); }
  };
  const finished = started && progress?.finished;
  const close = () => { if (started && !finished) return; useStore.setState({ storageProgress: null }); onDone(); };
  const p = plan?.plan;
  const need = mode === "clear_disposable_and_move" ? (p?.total_bytes ?? 0) - (plan?.disposable_bytes ?? 0) : mode === "switch_only" ? 0 : p?.total_bytes ?? 0;
  const tooSmall = !!p && p.destination_free !== null && p.destination_free < need;
  return (
    <Dialog title={`Move ${kind === "models" ? "models" : "cache"}`} onClose={close} closeDisabled={started && !finished} footer={
      started ? (finished ? <button className="btn primary" onClick={close}>Done</button> : <button className="btn danger" onClick={() => api.storageCancel().catch(errorToast)}>Cancel move</button>)
        : <>
          <button className="btn ghost" onClick={close}>Cancel</button>
          <button className="btn primary" disabled={!p || p.problems.length > 0 || tooSmall || running} onClick={start} data-testid="start-move">{mode === "switch_only" ? "Switch" : "Move"}</button>
        </>
    }>
      {!started ? (
        <div className="prefs" style={{ gap: 10 }}>
          <F label="From"><span className="path">{src}</span></F>
          <F label="To">
            <div className="row"><span className="path grow">{dest ?? "—"}</span><button className="btn sm" onClick={choose}>Browse</button></div>
          </F>
          {p && <F label="Size">{bytes(p.total_bytes)} · {p.file_count} files · {bytes(p.destination_free)} free there</F>}
          {p && (
            <F label="Method">
              <Select label="Method" value={mode} onChange={setMode} options={[
                ...(kind === "cache" ? [{ value: "clear_disposable_and_move" as const, label: `Clear regenerable files, move the rest (${bytes(need)})` }] : []),
                { value: "move" as const, label: `Copy, verify, switch, delete old (${bytes(p.total_bytes)})` },
                { value: "switch_only" as const, label: "Use the new folder without copying" },
              ]} />
            </F>
          )}
          {running && <div className="issue warn">Pause running exports first.</div>}
          {p?.problems.map((x) => <div key={x} className="issue err">{x}</div>)}
          {tooSmall && <div className="issue err">Not enough free space at the destination.</div>}
          {mode === "switch_only" && kind === "models" && p && <div className="issue warn">Installed models will show as missing until moved back or reinstalled.</div>}
        </div>
      ) : (
        <div className="prefs" style={{ gap: 8 }}>
          <Progress value={progress && progress.total ? progress.done / progress.total : 0} indeterminate={!progress?.total && !finished} tone={finished ? (finished.ok ? "ok" : "err") : undefined} label="Move progress" />
          <span className="muted">{finished ? finished.message : `${bytes(progress?.done ?? 0)} of ${bytes(progress?.total ?? 0)}`}</span>
        </div>
      )}
    </Dialog>
  );
}

function Storage({ s, patch }: { s: Settings; patch: Patch }) {
  const [ov, setOv] = useState<StorageOverview | null>(null);
  const [areas, setAreas] = useState<CacheArea[]>(["previews", "thumbnails", "proxies", "temp"]);
  const [resumable, setResumable] = useState(false);
  const [moving, setMoving] = useState<"models" | "cache" | null>(null);
  const [resetting, setResetting] = useState<"models" | "cache" | null>(null);
  const [last, setLast] = useState<ClearResult | null>(null);
  const load = () => api.storageOverview().then(setOv).catch(errorToast);
  useEffect(() => { load(); }, []);
  if (!ov) return <span className="faint">Measuring…</span>;
  const clear = async () => {
    try {
      const r = await api.cacheClear(areas, resumable);
      setLast(r);
      toast(r.errors.length ? "warning" : "success", `Freed ${bytes(r.freed_bytes)}${r.skipped_protected ? ` · kept ${r.skipped_protected} in use` : ""}`);
      load();
    } catch (e) { errorToast(e); }
  };
  const loc = (kind: "models" | "cache", label: string, path: string, isDefault: boolean, used: number, free: number | null) => (
    <div className="loc" data-testid={`loc-${kind}`}>
      <div className="name">{label} <span className={`tag${isDefault ? "" : " accent"}`}>{isDefault ? "default" : "custom"}</span></div>
      <div className="path">{path}</div>
      <div className="meta">{bytes(used)} used · {bytes(free)} free</div>
      <div className="acts">
        <button className="btn sm ghost icon" aria-label={`Open ${label} folder`} title="Open" onClick={() => api.openPath(path).catch(errorToast)}><FolderOpen size={13} /></button>
        <button className="btn sm" disabled={ov.relocation_in_progress} onClick={() => setMoving(kind)}>Change…</button>
        {!isDefault && <button className="btn sm ghost" onClick={() => setResetting(kind)}>Default</button>}
      </div>
    </div>
  );
  return (
    <div className="prefs">
      <div className="pref-h">Locations</div>
      {ov.warnings.map((w) => <div key={w} className="issue warn">{w}</div>)}
      {loc("models", "Models", ov.paths.models_dir, ov.paths.models_is_default, ov.models.total_bytes, ov.models.free_bytes)}
      {loc("cache", "Cache", ov.paths.cache_dir, ov.paths.cache_is_default, ov.cache.total_bytes, ov.cache.free_bytes)}
      <div className="loc">
        <div className="name">Settings & logs</div>
        <div className="path">{ov.paths.data_dir}</div>
        <div className="meta">{bytes(ov.data_bytes)}</div>
        <div className="acts"><button className="btn sm ghost icon" aria-label="Open data folder" onClick={() => api.openPath(ov.paths.data_dir).catch(errorToast)}><FolderOpen size={13} /></button></div>
      </div>
      <div className="pref-h">Cache</div>
      <table className="mtable" data-testid="cache-table">
        <tbody>
          {ov.cache.areas.map((a) => (
            <tr key={a.area} onClick={() => setAreas(areas.includes(a.area) ? areas.filter((x) => x !== a.area) : [...areas, a.area])}>
              <td style={{ width: 28 }}><span className="chk"><input type="checkbox" aria-label={`Select ${a.label}`} checked={areas.includes(a.area)} readOnly /></span></td>
              <td>{a.label}</td>
              <td className="r muted">{bytes(a.bytes)}{a.protected_bytes > 0 && <span className="faint"> · {bytes(a.protected_bytes)} in use</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <F label="Unfinished exports"><Check checked={resumable} onChange={setResumable} label="Also clear their work files (they restart)" /></F>
      <div className="row">
        <button className="btn" disabled={!areas.length} onClick={clear} data-testid="clear-cache">Clear selected</button>
        <button className="btn ghost" onClick={() => api.cacheEnforce().then((r) => { toast("info", `Freed ${bytes(r.freed_bytes)}`); load(); }).catch(errorToast)}>Run cleanup</button>
        <span className="grow" />
        <span className="muted">Total {bytes(ov.cache.total_bytes)}</span>
      </div>
      {last && last.errors.length > 0 && <div className="issue warn">{last.errors.slice(0, 2).join("; ")}</div>}
      <div className="pref-h">Automatic cleanup</div>
      <F label="Size limit"><NumberInput label="Cache size limit" value={Math.round(s.cache_limit_mb / 1024)} min={1} max={10000} suffix="GB" onChange={(v) => patch((x) => ({ ...x, cache_limit_mb: Math.round(v * 1024) }))} /></F>
      <F label="Remove after"><NumberInput label="Maximum cache age" value={s.cache_max_age_days} min={0} max={3650} suffix="days" onChange={(v) => patch((x) => ({ ...x, cache_max_age_days: Math.round(v) }))} /></F>
      <F label="Low disk warning"><NumberInput label="Low disk warning" value={Math.round(s.low_disk_warning_mb / 1000)} min={1} max={1000} suffix="GB" onChange={(v) => patch((x) => ({ ...x, low_disk_warning_mb: Math.round(v * 1000) }))} /></F>
      {moving && <Relocate kind={moving} overview={ov} onDone={() => { setMoving(null); load(); }} />}
      {resetting && <Confirm title={`Use the default ${resetting} folder?`} confirmLabel="Use default" onCancel={() => setResetting(null)}
        body="Nothing is moved or deleted." onConfirm={async () => { const k = resetting; setResetting(null); try { await api.storageReset(k); load(); } catch (e) { errorToast(e); } }} />}
    </div>
  );
}

const MODES: { value: PerformanceMode; label: string }[] = [
  { value: "balanced", label: "Balanced" }, { value: "high_performance", label: "High" }, { value: "maximum", label: "Maximum" }, { value: "custom", label: "Custom" },
];
const MEMORY: { value: MemoryPreset; label: string }[] = [
  { value: "more_for_others", label: "Other apps" }, { value: "balanced", label: "Balanced" }, { value: "prioritize_app", label: "This app" }, { value: "custom", label: "Custom" },
];

function Chart({ series, max, color }: { series: number[]; max: number; color: string }) {
  const w = 300, h = 64;
  const pts = series.map((v, i) => `${(i / 59) * w},${h - (Math.min(v, max) / max) * (h - 4) - 2}`).join(" ");
  return <svg className="chart" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden><polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" vectorEffect="non-scaling-stroke" /></svg>;
}

function Monitor() {
  const [hist, setHist] = useState<SystemSample[]>([]);
  const visible = useRef(true);
  useEffect(() => {
    const f = () => (visible.current = !document.hidden);
    document.addEventListener("visibilitychange", f);
    return () => document.removeEventListener("visibilitychange", f);
  }, []);
  useInterval(() => { if (visible.current) api.systemSample().then((x) => setHist((h) => [...h.slice(-59), x])).catch(() => {}); }, 1000);
  const c = hist[hist.length - 1];
  if (!c) return <span className="faint">…</span>;
  return (
    <div data-testid="monitor" style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div className="stats">
        <div className="stat"><div className="v">{c.cpu_percent.toFixed(0)}%</div><div className="k">CPU</div></div>
        <div className="stat"><div className="v">{(c.clearframe_cpu_percent / Math.max(1, c.logical_cores)).toFixed(0)}%</div><div className="k">CPU · this app</div></div>
        <div className="stat"><div className="v">{mb(c.memory_used_mb)}</div><div className="k">RAM of {mb(c.memory_total_mb)}</div></div>
        <div className="stat"><div className="v">{mb(c.clearframe_memory_mb)}</div><div className="k">RAM · this app</div></div>
        {c.gpus.map((g) => (
          <div className="stat" key={g.name}><div className="v">{g.measured && g.utilization_percent !== null ? `${g.utilization_percent.toFixed(0)}%` : "—"}</div><div className="k" title={g.name}>{g.name}</div></div>
        ))}
        {c.max_temperature_c !== null && <div className="stat"><div className="v">{c.max_temperature_c.toFixed(0)}°C</div><div className="k">Temperature</div></div>}
      </div>
      <div className="stats" style={{ gridTemplateColumns: "1fr 1fr" }}>
        <Chart series={hist.map((x) => x.cpu_percent)} max={100} color="var(--accent)" />
        <Chart series={hist.map((x) => x.memory_used_mb)} max={c.memory_total_mb} color="var(--ok)" />
      </div>
      {c.thermal_warning && <div className="issue warn">{c.thermal_warning}</div>}
      {c.memory_pressure && <div className="issue warn">System memory is low.</div>}
    </div>
  );
}

function Performance({ s, patch }: { s: Settings; patch: Patch }) {
  const hw = useStore((x) => x.hardware);
  const [plan, setPlan] = useState(hw?.plan ?? null);
  const p = s.performance;
  const setP = (f: (q: Settings["performance"]) => Settings["performance"]) => patch((x) => ({ ...x, performance: f({ ...x.performance }) }));
  useEffect(() => { api.hardware(false).then((h) => { setPlan(h.plan); useStore.setState({ hardware: h }); }).catch(() => {}); }, [s.performance]);
  const h = hw?.hardware;
  const cores = h?.logical_cores ?? 8;
  const ram = h?.ram_total_mb ?? 16384;
  const devices: { value: string; label: string; disabled?: boolean }[] = [{ value: "auto", label: "Automatic" }, { value: "cpu", label: `CPU · ${h?.cpu_brand ?? ""}` }];
  const API: Record<string, string> = { vulkan: "Vulkan", dx12: "DirectX 12", metal: "Metal" };
  for (const a of h?.gpu_adapters ?? []) {
    const soft = a.kind === "cpu";
    devices.push({ value: `gpu:${a.index}`, label: `${a.name} · ${API[a.api] ?? a.api}${soft ? " (software, slow)" : ""}` });
  }
  for (const b of h?.backends ?? []) {
    if (b.kind === "cpu" || b.kind === "gpu") continue;
    const gpus = (h?.gpus ?? []).filter((g) => (b.kind === "cuda" ? /nvidia/i.test(g.vendor) : /apple|amd|intel/i.test(g.vendor) || h?.os === "macos"));
    (gpus.length ? gpus : [{ name: b.kind.toUpperCase() }]).forEach((g, i) => devices.push({ value: `${b.kind}:${i}`, label: `${g.name} · ${b.kind.toUpperCase()}${b.available ? "" : " (unavailable)"}`, disabled: !b.available }));
  }
  if (!devices.some((d) => d.value === p.device)) devices.push({ value: p.device, label: `${p.device} (not present)`, disabled: true });
  const anyGpu = (h?.backends ?? []).some((b) => b.kind !== "cpu" && b.available);
  return (
    <div className="prefs">
      <div className="pref-h">Processing</div>
      <F label="Mode"><Seg label="Performance mode" value={p.mode} onChange={(v) => setP((q) => ({ ...q, mode: v }))} options={MODES} /></F>
      {plan && <F label=""><span className="faint">{plan.threads} of {cores} threads · {plan.priority.replace("_", " ")} priority</span></F>}
      {p.mode === "custom" && (
        <>
          <F label="CPU threads"><NumberInput label="CPU threads" value={p.cpu_threads} min={0} max={cores} onChange={(v) => setP((q) => ({ ...q, cpu_threads: Math.round(v) }))} suffix={p.cpu_threads === 0 ? "auto" : `of ${cores}`} /></F>
          <F label="Keep free"><NumberInput label="Reserved threads" value={p.reserve_cpu_threads} min={0} max={Math.max(0, cores - 1)} onChange={(v) => setP((q) => ({ ...q, reserve_cpu_threads: Math.round(v) }))} suffix="threads" /></F>
          <F label="Priority"><Select label="Process priority" value={p.priority} onChange={(v) => setP((q) => ({ ...q, priority: v }))} options={[{ value: "low", label: "Low" }, { value: "below_normal", label: "Below normal" }, { value: "normal", label: "Normal" }, { value: "above_normal", label: "Above normal" }]} /></F>
        </>
      )}
      <F label="Parallel exports"><NumberInput label="Concurrent exports" value={p.max_concurrent_jobs} min={1} max={8} onChange={(v) => setP((q) => ({ ...q, max_concurrent_jobs: Math.round(v) }))} /></F>
      <div className="pref-h">Memory</div>
      <F label="Prioritize"><Seg label="Memory preset" value={p.memory_preset} onChange={(v) => setP((q) => ({ ...q, memory_preset: v }))} options={MEMORY} /></F>
      {p.memory_preset === "custom" && <F label="Budget (MB)"><Range label="Memory budget" value={p.memory_budget_mb} min={512} max={ram} step={256} onChange={(v) => setP((q) => ({ ...q, memory_budget_mb: Math.round(v) }))} /></F>}
      {plan && <F label=""><span className="faint">{mb(plan.memory_budget_mb)} for processing · {mb(plan.reserve_memory_mb)} kept free</span></F>}
      <F label="When memory is low"><Select label="Memory pressure action" value={p.memory_pressure} onChange={(v) => setP((q) => ({ ...q, memory_pressure: v }))} options={[{ value: "pause", label: "Pause exports" }, { value: "continue", label: "Keep going" }]} /></F>
      <div className="pref-h">Graphics</div>
      <F label="Device"><Select label="Compute device" value={p.device} onChange={(v) => setP((q) => ({ ...q, device: v }))} options={devices} /></F>
      {!anyGpu && <F label=""><span className="faint">No GPU found. Processing runs on the CPU.</span></F>}
      {anyGpu && <F label="GPU memory (MB)"><NumberInput label="GPU memory limit" value={p.vram_budget_mb} min={0} max={262144} step={256} suffix={p.vram_budget_mb === 0 ? "auto" : "MB"} onChange={(v) => setP((q) => ({ ...q, vram_budget_mb: Math.round(v) }))} /></F>}
      <F label="Out of memory"><Check checked={p.oom_tile_fallback} onChange={(v) => setP((q) => ({ ...q, oom_tile_fallback: v }))} label="Retry with smaller tiles" /></F>
      <div className="pref-h"><Activity size={11} style={{ verticalAlign: -1 }} /> Live</div>
      <Monitor />
    </div>
  );
}

function Advanced({ s, patch }: { s: Settings; patch: Patch }) {
  const app = useStore((x) => x.app);
  const [ask, setAsk] = useState(false);
  return (
    <div className="prefs">
      <div className="pref-h">FFmpeg</div>
      <F label="Location">
        <div className="row">
          <span className="path grow ellipsis">{s.ffmpeg_dir ?? (app?.ffmpeg ? app.ffmpeg.path : "Not found")}</span>
          <button className="btn sm" onClick={async () => { const f = await pickFolder("Folder with ffmpeg and ffprobe"); if (f) patch((x) => ({ ...x, ffmpeg_dir: f })); }}>Browse</button>
          {s.ffmpeg_dir && <button className="btn sm ghost" onClick={() => patch((x) => ({ ...x, ffmpeg_dir: null }))}>Auto</button>}
          <button className="btn sm ghost" onClick={async () => { try { await api.refreshTools(); useStore.setState({ app: await api.appInfo() }); toast("success", "FFmpeg found"); } catch (e) { errorToast(e); } }}>Detect</button>
        </div>
      </F>
      <F label="Version">{app?.ffmpeg ? `${app.ffmpeg.version} · ${app.ffmpeg.license}` : app?.ffmpeg_checking ? null : <span className="err">{app?.ffmpeg_error}</span>}</F>
      {app?.ffmpeg && <F label="Filters"><span className="row"><span className={`tag ${app.ffmpeg.vidstab ? "ok" : "err"}`}>vid.stab</span><span className={`tag ${app.ffmpeg.zscale ? "ok" : "err"}`}>zscale</span></span></F>}
      <div className="pref-h">Processing</div>
      <F label="Segment length"><NumberInput label="Segment seconds" value={s.segment_seconds} min={2} max={120} suffix="s" onChange={(v) => patch((x) => ({ ...x, segment_seconds: v }))} /></F>
      <F label="GPU fallback"><Select label="Fallback" value={s.fallback} onChange={(v) => patch((x) => ({ ...x, fallback: v }))} options={[{ value: "ask", label: "Ask" }, { value: "allow", label: "Use CPU" }, { value: "never", label: "Fail" }]} /></F>
      <F label="Half precision"><Check checked={s.half_precision} onChange={(v) => patch((x) => ({ ...x, half_precision: v }))} label="FP16 on supported GPUs" /></F>
      <F label="On failure"><Check checked={s.stop_queue_on_error} onChange={(v) => patch((x) => ({ ...x, stop_queue_on_error: v }))} label="Pause the queue" /></F>
      <F label="Failed work files"><Check checked={s.keep_failed_work} onChange={(v) => patch((x) => ({ ...x, keep_failed_work: v }))} label="Keep for retry" /></F>
      <div className="pref-h">Reset</div>
      <F label=""><button className="btn danger" style={{ justifySelf: "start" }} onClick={() => setAsk(true)}>Reset preferences</button></F>
      {ask && <Confirm title="Reset preferences?" danger confirmLabel="Reset" body="Storage locations, models, presets and exports are kept." onCancel={() => setAsk(false)}
        onConfirm={async () => { setAsk(false); try { useStore.setState({ settings: await api.settingsReset() }); } catch (e) { errorToast(e); } }} />}
    </div>
  );
}

function Diagnostics() {
  const [redact, setRedact] = useState(true);
  const [text, setText] = useState<string | null>(null);
  const [logPath, setLogPath] = useState<string | null>(null);
  useEffect(() => { api.logPath().then(setLogPath).catch(() => setLogPath(null)); }, []);
  return (
    <div className="prefs">
      <div className="pref-h">Activity log</div>
      <F label="Log file">
        <div className="row">
          <span className="mono" style={{ wordBreak: "break-all" }}>{logPath ?? "…"}</span>
          {logPath && <button className="btn ghost" onClick={() => api.openPath(logPath).catch(errorToast)}>Show</button>}
        </div>
      </F>
      <p className="muted">Everything the app does is recorded here from the moment it opens: clicks, errors, crashes and performance. Send this file when reporting a problem. It never leaves your computer on its own.</p>
      <div className="pref-h">Report</div>
      <F label="Privacy"><Check checked={redact} onChange={setRedact} label="Hide paths and file names" /></F>
      <div className="row">
        <button className="btn" onClick={() => api.diagnostics(redact).then(setText).catch(errorToast)}>Create</button>
        {text && <button className="btn ghost" onClick={() => navigator.clipboard?.writeText(text).then(() => toast("success", "Copied"), errorToast)}>Copy</button>}
        {text && <button className="btn ghost" onClick={async () => { const p = await pickSave("Save report", "clipfx-diagnostics.txt", [{ name: "Text", extensions: ["txt"] }]); if (p) api.writeTextFile(p, text).then(() => toast("success", "Saved"), errorToast); }}>Save…</button>}
      </div>
      {text && <textarea className="input" rows={20} readOnly value={text} aria-label="Diagnostics report" />}
    </div>
  );
}

function About() {
  const app = useStore((x) => x.app);
  const [upd, setUpd] = useState<string | null>(null);
  return (
    <div className="prefs">
      <div className="row" style={{ gap: 12, marginBottom: 8 }}>
        <img src="/logo.svg" alt="" width={40} height={40} />
        <div><div style={{ fontSize: 15 }}>ClipFX <b style={{ color: "var(--accent-2)", fontWeight: 500 }}>Enhance</b></div><div className="faint">Version {app?.version} · MIT License</div></div>
      </div>
      <F label="Platform">{app?.os} {app?.arch}</F>
      <F label="Compute">{app?.compiled_backends.join(", ").toUpperCase()}</F>
      <F label="FFmpeg">{app?.ffmpeg ? app.ffmpeg.version : app?.ffmpeg_checking ? null : app?.ffmpeg_error}</F>
      <F label="Updates">
        <div className="row">
          <button className="btn sm" onClick={() => api.checkUpdate().then((u) => { useStore.setState({ update: u.newer ? u : null }); setUpd(u.newer ? `${u.latest} is available` : "Up to date"); }).catch((e) => setUpd(e instanceof Error ? e.message : String(e)))}>Check now</button>
          {upd && <span className="muted selectable">{upd}</span>}
        </div>
      </F>
    </div>
  );
}

export function PreferencesDialog() {
  const section = useStore((x) => x.settingsSection);
  const s = useStore((x) => x.settings);
  if (!s) return null;
  const patch: Patch = (f) => { saveSettings(f); };
  return (
    <Dialog title="Preferences" onClose={() => useStore.setState({ dialog: null })} size="wide" flush>
      <div className="split-dialog">
        <nav aria-label="Preferences sections">
          {SECTIONS.map(({ key, label, icon: I }) => (
            <button key={key} aria-current={section === key ? "page" : undefined} onClick={() => useStore.setState({ settingsSection: key })}><I size={14} /> {label}</button>
          ))}
        </nav>
        <div className="content">
          {section === "general" && <General s={s} patch={patch} />}
          {section === "export" && <Output s={s} patch={patch} />}
          {section === "storage" && <Storage s={s} patch={patch} />}
          {section === "performance" && <Performance s={s} patch={patch} />}
          {section === "advanced" && <Advanced s={s} patch={patch} />}
          {section === "diagnostics" && <Diagnostics />}
          {section === "about" && <About />}
        </div>
      </div>
    </Dialog>
  );
}
