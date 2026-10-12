import { useEffect, useRef, useState } from "react";
import { Activity, Cpu, Database, FileText, FolderOpen, Info, Settings2, Upload, Wrench } from "lucide-react";

import { Check, Confirm, Field, Modal, Notice, NumberInput, Progress, Select, Seg, Slider, useInterval } from "../components/ui";
import { api, pickFolder, pickSave } from "../lib/ipc";
import { bytes, mb } from "../lib/format";
import type { CacheArea, ClearResult, MemoryPreset, PerformanceMode, RelocationPlan, Settings, StorageOverview, SystemSample } from "../lib/types";
import { errorToast, saveSettings, toast, useStore, type SettingsSection } from "../state/store";

const SECTIONS: { key: SettingsSection; label: string; icon: typeof Info }[] = [
  { key: "general", label: "General", icon: Settings2 },
  { key: "export", label: "Export & preview", icon: Upload },
  { key: "storage", label: "Storage & cache", icon: Database },
  { key: "performance", label: "Performance", icon: Cpu },
  { key: "advanced", label: "Advanced", icon: Wrench },
  { key: "diagnostics", label: "Diagnostics", icon: FileText },
  { key: "about", label: "About", icon: Info },
];

type Patch = (f: (s: Settings) => Settings) => void;

function General({ s, patch }: { s: Settings; patch: Patch }) {
  return (
    <section className="section">
      <h2>General</h2>
      <p className="lead">Appearance and behaviour.</p>
      <div className="grid2">
        <Field label="Theme"><Seg label="Theme" value={s.theme} onChange={(v) => patch((x) => ({ ...x, theme: v }))} options={[{ value: "dark", label: "Dark" }, { value: "light", label: "Light" }, { value: "system", label: "System" }]} /></Field>
        <Field label="Motion"><Seg label="Motion" value={s.motion} onChange={(v) => patch((x) => ({ ...x, motion: v }))} options={[{ value: "system", label: "System" }, { value: "reduced", label: "Reduced" }, { value: "full", label: "Full" }]} /></Field>
        <Field label="Interface size"><Slider label="Interface size" value={s.ui_scale} min={0.75} max={2} step={0.05} onChange={(v) => patch((x) => ({ ...x, ui_scale: v }))} /></Field>
        <Field label="Language" hint="More languages can be contributed; the interface is currently English only.">
          <Select label="Language" value={s.language} onChange={(v) => patch((x) => ({ ...x, language: v }))} options={[{ value: "en", label: "English" }]} />
        </Field>
      </div>
      <Check checked={s.notifications} onChange={(v) => patch((x) => ({ ...x, notifications: v }))} label="Notify me when exports finish or fail" desc="Shown only while ClearFrame is in the background." />
      <Check checked={s.confirm_before_quit_with_jobs} onChange={(v) => patch((x) => ({ ...x, confirm_before_quit_with_jobs: v }))} label="Ask before closing while exports are running" desc="Running exports are paused safely and resume next time either way." />
      <Check checked={s.check_updates_on_launch} onChange={(v) => patch((x) => ({ ...x, check_updates_on_launch: v }))} label="Check for new ClearFrame versions on launch" desc="Contacts GitHub only. Off by default; ClearFrame works fully offline." />
      <Check checked={s.allow_catalog_updates} onChange={(v) => patch((x) => ({ ...x, allow_catalog_updates: v }))} label="Allow signed model catalog updates" desc="Only lists newer models; nothing is downloaded until you press Install." />
    </section>
  );
}

function ExportSettings({ s, patch }: { s: Settings; patch: Patch }) {
  const [def, setDef] = useState("");
  useEffect(() => { api.defaultExportFolder().then(setDef).catch(() => {}); }, []);
  return (
    <section className="section">
      <h2>Export & preview</h2>
      <p className="lead">Where results go and how they are named. Originals are never overwritten.</p>
      <Field label="Default export folder">
        <div className="row">
          <span className="path grow">{s.output_dir ?? `${def} (default)`}</span>
          <button className="btn small" onClick={async () => { const f = await pickFolder("Default export folder", s.output_dir ?? def); if (f) patch((x) => ({ ...x, output_dir: f })); }}>Choose…</button>
          {s.output_dir && <button className="btn ghost small" onClick={() => patch((x) => ({ ...x, output_dir: null }))}>Reset</button>}
        </div>
      </Field>
      <Check checked={s.remember_last_export_folder} onChange={(v) => patch((x) => ({ ...x, remember_last_export_folder: v }))} label="Remember the last folder I exported to" />
      <Field label="File naming" htmlFor="nt" hint={<>Placeholders: <code>{"{name}"}</code> source name, <code>{"{task}"}</code> main task, <code>{"{model}"}</code>, <code>{"{preset}"}</code>, <code>{"{width}"}</code>, <code>{"{height}"}</code>, <code>{"{fps}"}</code>, <code>{"{date}"}</code>.</>}>
        <input id="nt" className="input" value={s.naming_template} onChange={(e) => patch((x) => ({ ...x, naming_template: e.target.value }))} />
      </Field>
      <Check checked={s.write_sidecar_report} onChange={(v) => patch((x) => ({ ...x, write_sidecar_report: v }))} label="Save a processing report next to each export (.clearframe.json)" />
      <div className="grid2 mt">
        <Field label="Preview clip length"><Slider label="Preview seconds" value={s.preview_seconds} min={0.5} max={30} step={0.5} onChange={(v) => patch((x) => ({ ...x, preview_seconds: v }))} /></Field>
        <Field label="Preview proxy height" help="Maximum height of the playback copies used for comparison.">
          <Select label="Preview proxy height" value={String(s.preview_max_height)} onChange={(v) => patch((x) => ({ ...x, preview_max_height: Number(v) }))} options={["540", "720", "1080", "1440", "2160"].map((h) => ({ value: h, label: `${h}p` }))} />
        </Field>
      </div>
    </section>
  );
}

const AREA_SAFE: Record<CacheArea, string> = {
  previews: "Regenerated when you preview again.", thumbnails: "Regenerated when you open a video.", proxies: "Regenerated for playback.",
  downloads: "Partial model downloads. Clearing means a paused install starts over.", temp: "Leftovers from interrupted operations.",
  jobs: "Work files of exports. Data of unfinished exports is kept unless you choose otherwise.",
};

function Relocate({ kind, overview, onDone }: { kind: "models" | "cache"; overview: StorageOverview; onDone: () => void }) {
  const [dest, setDest] = useState<string | null>(null);
  const [plan, setPlan] = useState<RelocationPlan | null>(null);
  const [mode, setMode] = useState<"move" | "clear_disposable_and_move" | "switch_only">(kind === "cache" ? "clear_disposable_and_move" : "move");
  const [started, setStarted] = useState(false);
  const progress = useStore((s) => s.storageProgress);
  const running = useStore((s) => s.jobs.some((j) => ["preparing", "processing", "encoding", "validating", "previewing"].includes(j.state)));
  const choose = async () => {
    const f = await pickFolder(kind === "models" ? "Choose the new models folder" : "Choose the new cache folder");
    if (!f) return;
    setDest(f);
    setPlan(null);
    try {
      setPlan(await api.storagePlan(kind, f));
    } catch (e) {
      errorToast(e);
    }
  };
  const start = async () => {
    if (!dest) return;
    useStore.setState({ storageProgress: { kind, done: 0, total: plan?.plan.total_bytes ?? 0 } });
    try {
      await api.storageRelocate(kind, dest, mode);
      setStarted(true);
    } catch (e) {
      errorToast(e);
      useStore.setState({ storageProgress: null });
    }
  };
  const finished = started && progress?.finished;
  const close = () => {
    if (started && !finished) return;
    useStore.setState({ storageProgress: null });
    onDone();
  };
  const p = plan?.plan;
  const src = kind === "models" ? overview.paths.models_dir : overview.paths.cache_dir;
  const need = mode === "clear_disposable_and_move" ? (p?.total_bytes ?? 0) - (plan?.disposable_bytes ?? 0) : mode === "switch_only" ? 0 : p?.total_bytes ?? 0;
  const tooSmall = p && p.destination_free !== null && p.destination_free < need;
  return (
    <Modal title={kind === "models" ? "Change models location" : "Change cache location"} onClose={close} closeDisabled={started && !finished} footer={
      started ? (
        finished ? <button className="btn primary" onClick={close}>Close</button> : <button className="btn danger" onClick={() => api.storageCancel().catch(errorToast)}>Cancel move</button>
      ) : (
        <>
          <button className="btn" onClick={close}>Cancel</button>
          <button className="btn primary" disabled={!p || p.problems.length > 0 || !!tooSmall || running} onClick={start} data-testid="start-move">
            {mode === "switch_only" ? "Use this folder" : "Move"}
          </button>
        </>
      )
    }>
      {!started && (
        <>
          <Field label="Current location"><div className="path">{src}</div></Field>
          <Field label="New location">
            <div className="row"><span className="path grow">{dest ?? "Not chosen"}</span><button className="btn small" onClick={choose}><FolderOpen size={14} /> Choose…</button></div>
          </Field>
          {running && <Notice tone="warn">Pause running exports and previews first. A move never runs while files are being written.</Notice>}
          {plan && (
            <>
              <div className="stats mt">
                <div className="stat"><span className="v">{bytes(p!.total_bytes)}</span><span className="l">to move ({p!.file_count} files)</span></div>
                <div className="stat"><span className="v">{bytes(p!.destination_free)}</span><span className="l">free at destination</span></div>
                {kind === "cache" && <div className="stat"><span className="v">{bytes(plan.disposable_bytes)}</span><span className="l">regenerable (can be cleared)</span></div>}
              </div>
              {p!.problems.map((x) => <div key={x} className="mt"><Notice tone="err">{x}</Notice></div>)}
              {tooSmall && <div className="mt"><Notice tone="err">Not enough free space at the destination.</Notice></div>}
              <Field label="How">
                <div className="col" style={{ gap: 2 }}>
                  {kind === "cache" && <Check checked={mode === "clear_disposable_and_move"} onChange={() => setMode("clear_disposable_and_move")} label="Clear regenerable cache, then move the rest (recommended)" desc="Previews, thumbnails, proxies and temp files are deleted instead of copied. Unfinished exports keep their work files." />}
                  <Check checked={mode === "move"} onChange={() => setMode("move")} label="Move everything" desc="Copies and verifies every file, switches to the new folder, then deletes the old copy. Interrupted moves are rolled back or finished on next start." />
                  <Check checked={mode === "switch_only"} onChange={() => setMode("switch_only")} label="Just use the new folder" desc={kind === "models" ? "Nothing is copied. Installed models in the old folder will show as missing until you move them back or reinstall." : "Nothing is copied; the old cache is left as it is."} />
                </div>
              </Field>
              {plan.notes.map((n) => <p key={n} className="small muted">{n}</p>)}
            </>
          )}
        </>
      )}
      {started && (
        <div className="col">
          <Progress value={progress && progress.total ? progress.done / progress.total : 0} indeterminate={!progress?.total && !finished} tone={finished ? (finished.ok ? "ok" : "err") : undefined} label="Move progress" />
          <div className="small muted">{finished ? finished.message : `${bytes(progress?.done ?? 0)} of ${bytes(progress?.total ?? 0)} copied…`}</div>
          {!finished && <p className="small faint">The old folder stays in use until every file has been copied and verified. You can cancel safely.</p>}
          {finished && !finished.ok && <Notice tone="warn">Nothing was switched. Your data is still in {src}.</Notice>}
        </div>
      )}
    </Modal>
  );
}

function Storage({ s, patch }: { s: Settings; patch: Patch }) {
  const [ov, setOv] = useState<StorageOverview | null>(null);
  const [areas, setAreas] = useState<CacheArea[]>(["previews", "thumbnails", "proxies", "temp"]);
  const [includeResumable, setIncludeResumable] = useState(false);
  const [moving, setMoving] = useState<"models" | "cache" | null>(null);
  const [resetting, setResetting] = useState<"models" | "cache" | null>(null);
  const [last, setLast] = useState<ClearResult | null>(null);
  const load = () => api.storageOverview().then(setOv).catch(errorToast);
  useEffect(() => { load(); }, []);
  if (!ov) return <p className="muted">Measuring storage…</p>;
  const clear = async () => {
    try {
      const r = await api.cacheClear(areas, includeResumable);
      setLast(r);
      toast(r.errors.length ? "warning" : "success", `Freed ${bytes(r.freed_bytes)}.${r.skipped_protected ? ` Kept ${r.skipped_protected} item(s) that unfinished exports need.` : ""}`);
      load();
    } catch (e) {
      errorToast(e);
    }
  };
  const locRow = (kind: "models" | "cache", path: string, isDefault: boolean, used: number, free: number | null) => (
    <div className="card" data-testid={`loc-${kind}`}>
      <div className="row wrap">
        <div className="grow" style={{ minWidth: 260 }}>
          <h3>{kind === "models" ? "Models" : "Cache"} {isDefault ? <span className="badge">default</span> : <span className="badge accent">custom</span>}</h3>
          <div className="path">{path}</div>
          <div className="small faint">{bytes(used)} used · {bytes(free)} free on this drive</div>
        </div>
        <button className="btn small" onClick={() => api.openPath(path).catch(errorToast)}><FolderOpen size={14} /> Open</button>
        <button className="btn small" disabled={ov.relocation_in_progress} onClick={() => setMoving(kind)}>Change…</button>
        {!isDefault && <button className="btn ghost small" onClick={() => setResetting(kind)}>Use default</button>}
      </div>
    </div>
  );
  const total = ov.cache.total_bytes;
  return (
    <section className="section">
      <h2>Storage & cache</h2>
      <p className="lead">ClearFrame keeps everything it writes in these folders. Clearing the cache never deletes your videos, exports, projects, presets or models.</p>
      {ov.warnings.map((w) => <div key={w} style={{ marginBottom: 8 }}><Notice tone="warn">{w}</Notice></div>)}
      {locRow("models", ov.paths.models_dir, ov.paths.models_is_default, ov.models.total_bytes, ov.models.free_bytes)}
      {locRow("cache", ov.paths.cache_dir, ov.paths.cache_is_default, total, ov.cache.free_bytes)}
      <div className="card">
        <h3>Settings, database and logs</h3>
        <div className="path">{ov.paths.data_dir}</div>
        <div className="small faint">{bytes(ov.data_bytes)} · logs in {ov.paths.logs_dir}</div>
        <div className="row mt"><button className="btn small" onClick={() => api.openPath(ov.paths.data_dir).catch(errorToast)}><FolderOpen size={14} /> Open</button>
          <button className="btn small" onClick={() => api.openPath(ov.paths.logs_dir).catch(errorToast)}><FolderOpen size={14} /> Open logs</button></div>
      </div>

      <h3 className="mt-l">Cache contents</h3>
      <table className="table mt" data-testid="cache-table">
        <thead><tr><th style={{ width: 30 }}></th><th>Area</th><th className="num">Size</th><th>Notes</th></tr></thead>
        <tbody>
          {ov.cache.areas.map((a) => (
            <tr key={a.area}>
              <td><input type="checkbox" aria-label={`Select ${a.label}`} checked={areas.includes(a.area)} onChange={(e) => setAreas(e.target.checked ? [...areas, a.area] : areas.filter((x) => x !== a.area))} /></td>
              <td>{a.label}</td>
              <td className="num">{bytes(a.bytes)}{a.protected_bytes > 0 && <div className="tiny faint">{bytes(a.protected_bytes)} in use</div>}</td>
              <td className="small faint">{AREA_SAFE[a.area]}</td>
            </tr>
          ))}
          <tr><td></td><td><strong>Total</strong></td><td className="num"><strong>{bytes(total)}</strong></td><td></td></tr>
        </tbody>
      </table>
      <Check checked={includeResumable} onChange={setIncludeResumable} label="Also delete work files of paused or interrupted exports" desc="Those exports will start again from the beginning." />
      <div className="row">
        <button className="btn" disabled={!areas.length} onClick={clear} data-testid="clear-cache">Clear selected</button>
        <button className="btn ghost" onClick={() => api.cacheEnforce().then((r) => { toast("info", `Cleanup freed ${bytes(r.freed_bytes)}.`); load(); }).catch(errorToast)}>Run automatic cleanup now</button>
      </div>
      {last && last.errors.length > 0 && <div className="mt"><Notice tone="warn">Some files could not be removed (they may be open in another program): {last.errors.slice(0, 3).join("; ")}</Notice></div>}

      <h3 className="mt-l">Automatic cleanup</h3>
      <div className="grid2 mt">
        <Field label="Cache size limit" help="When the cache grows beyond this, the oldest regenerable files are removed first. Work files of unfinished exports are never removed automatically.">
          <NumberInput label="Cache size limit" value={Math.round(s.cache_limit_mb / 1024)} min={1} max={10000} suffix="GB" onChange={(v) => patch((x) => ({ ...x, cache_limit_mb: Math.round(v * 1024) }))} />
        </Field>
        <Field label="Remove cache files older than">
          <NumberInput label="Maximum cache age" value={s.cache_max_age_days} min={0} max={3650} suffix="days (0 = never)" onChange={(v) => patch((x) => ({ ...x, cache_max_age_days: Math.round(v) }))} />
        </Field>
        <Field label="Warn when free space is below" help="Exports pause safely before the disk is full, and resume when you free space.">
          <NumberInput label="Low disk warning" value={Math.round(s.low_disk_warning_mb / 1000)} min={1} max={1000} suffix="GB" onChange={(v) => patch((x) => ({ ...x, low_disk_warning_mb: Math.round(v * 1000) }))} />
        </Field>
      </div>
      <p className="small faint">GPU drivers keep their own shader caches outside ClearFrame's control (for example NVIDIA's ComputeCache or the macOS Metal cache). ClearFrame does not move or clear them.</p>

      {moving && <Relocate kind={moving} overview={ov} onDone={() => { setMoving(null); load(); }} />}
      {resetting && (
        <Confirm title={`Use the default ${resetting} folder?`} confirmLabel="Use default" onCancel={() => setResetting(null)}
          body={resetting === "models" ? "ClearFrame will look for models in the default folder. Models in the current custom folder are not moved or deleted; they will show as missing until you move them." : "ClearFrame will use the default cache folder. The current cache folder is left as it is."}
          onConfirm={async () => { const k = resetting; setResetting(null); try { await api.storageReset(k); load(); } catch (e) { errorToast(e); } }} />
      )}
    </section>
  );
}

const MODES: { value: PerformanceMode; title: string; desc: string }[] = [
  { value: "balanced", title: "Balanced", desc: "Leaves CPU and memory for other apps; ClearFrame runs at lower priority. Best while you keep working." },
  { value: "high_performance", title: "High performance", desc: "Uses all but one CPU thread at normal priority. Other apps may feel slower." },
  { value: "maximum", title: "Maximum", desc: "Uses every CPU thread. Fastest, but your computer will be busy; laptops may get hot and loud." },
  { value: "custom", title: "Custom", desc: "Choose threads, priority and memory yourself." },
];
const MEMORY: { value: MemoryPreset; label: string }[] = [
  { value: "more_for_others", label: "More for other apps" }, { value: "balanced", label: "Balanced" },
  { value: "prioritize_clear_frame", label: "Prioritize ClearFrame" }, { value: "custom", label: "Custom" },
];

function Chart({ series, max, color }: { series: number[]; max: number; color: string }) {
  const w = 300, h = 80;
  const pts = series.map((v, i) => `${(i / Math.max(1, 59)) * w},${h - (Math.min(v, max) / max) * (h - 4) - 2}`).join(" ");
  return (
    <svg className="chart" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="2" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function Monitor() {
  const [hist, setHist] = useState<SystemSample[]>([]);
  const visible = useRef(true);
  useEffect(() => {
    const f = () => (visible.current = !document.hidden);
    document.addEventListener("visibilitychange", f);
    return () => document.removeEventListener("visibilitychange", f);
  }, []);
  useInterval(() => {
    if (!visible.current) return;
    api.systemSample().then((x) => setHist((h) => [...h.slice(-59), x])).catch(() => {});
  }, 1000);
  const cur = hist[hist.length - 1];
  if (!cur) return <p className="muted small">Measuring…</p>;
  return (
    <div data-testid="monitor">
      <div className="stats">
        <div className="stat"><span className="v">{cur.cpu_percent.toFixed(0)}%</span><span className="l">CPU (all apps)</span></div>
        <div className="stat"><span className="v">{(cur.clearframe_cpu_percent / Math.max(1, cur.logical_cores)).toFixed(0)}%</span><span className="l">CPU (ClearFrame)</span></div>
        <div className="stat"><span className="v">{mb(cur.memory_used_mb)}</span><span className="l">RAM used of {mb(cur.memory_total_mb)}</span></div>
        <div className="stat"><span className="v">{mb(cur.clearframe_memory_mb)}</span><span className="l">RAM (ClearFrame)</span></div>
        {cur.gpus.map((g) => (
          <div className="stat" key={g.name}>
            <span className="v">{g.measured && g.utilization_percent !== null ? `${g.utilization_percent.toFixed(0)}%` : "—"}</span>
            <span className="l">{g.name}{g.measured ? (g.memory_used_mb !== null ? ` · ${mb(g.memory_used_mb)} / ${mb(g.memory_total_mb)}` : "") : " · usage not measurable"}</span>
          </div>
        ))}
        {cur.max_temperature_c !== null && <div className="stat"><span className="v">{cur.max_temperature_c.toFixed(0)} °C</span><span className="l">Hottest sensor</span></div>}
      </div>
      <div className="grid2 mt">
        <div><div className="tiny faint">CPU, last minute</div><Chart series={hist.map((x) => x.cpu_percent)} max={100} color="var(--accent)" /></div>
        <div><div className="tiny faint">Memory, last minute</div><Chart series={hist.map((x) => x.memory_used_mb)} max={cur.memory_total_mb} color="var(--ok)" /></div>
      </div>
      {cur.thermal_warning && <div className="mt"><Notice tone="warn">{cur.thermal_warning}</Notice></div>}
      {cur.memory_pressure && <div className="mt"><Notice tone="warn">The system is low on memory. Exports pause at a safe point when this happens (if enabled below).</Notice></div>}
    </div>
  );
}

function Performance({ s, patch }: { s: Settings; patch: Patch }) {
  const hw = useStore((x) => x.hardware);
  const [plan, setPlan] = useState(hw?.plan ?? null);
  const p = s.performance;
  const setP = (f: (q: Settings["performance"]) => Settings["performance"]) => patch((x) => ({ ...x, performance: f({ ...x.performance }) }));
  useEffect(() => {
    api.hardware(false).then((h) => { setPlan(h.plan); useStore.setState({ hardware: h }); }).catch(() => {});
  }, [s.performance]);
  const h = hw?.hardware;
  const cores = h?.logical_cores ?? 8;
  const ram = h?.ram_total_mb ?? 16384;
  const devices: { value: string; label: string; disabled?: boolean }[] = [{ value: "auto", label: "Automatic (best available)" }, { value: "cpu", label: "CPU" }];
  for (const b of h?.backends ?? []) {
    if (b.kind === "cpu") continue;
    const gpus = (h?.gpus ?? []).filter((g) => (b.kind === "cuda" ? /nvidia/i.test(g.vendor) : b.kind === "metal" ? /apple|amd|intel/i.test(g.vendor) || h?.os === "macos" : false));
    const list = gpus.length ? gpus : [{ name: b.kind.toUpperCase() }];
    list.forEach((g, i) => devices.push({ value: `${b.kind}:${i}`, label: `${g.name} (${b.kind.toUpperCase()})${b.available ? "" : ` — ${b.detail}`}`, disabled: !b.available }));
  }
  if (!devices.some((d) => d.value === p.device)) devices.push({ value: p.device, label: `${p.device} — not available on this computer`, disabled: true });
  const anyGpu = (h?.backends ?? []).some((b) => b.kind !== "cpu" && b.available);
  return (
    <section className="section">
      <h2>Performance</h2>
      <p className="lead">How much of your computer ClearFrame may use. Changes apply to exports that start after you change them.</p>
      <div className="mode-cards" role="radiogroup" aria-label="Performance mode">
        {MODES.map((m) => (
          <button key={m.value} className="mode-card" role="radio" aria-checked={p.mode === m.value} aria-pressed={p.mode === m.value} onClick={() => setP((q) => ({ ...q, mode: m.value }))}>
            <span className="t">{m.title}</span><span className="d">{m.desc}</span>
          </button>
        ))}
      </div>
      {plan && <div className="mt"><Notice tone="info">{plan.explanation.map((l) => <div key={l}>{l}</div>)}</Notice></div>}

      {p.mode === "custom" && (
        <div className="grid2 mt">
          <Field label="CPU threads" help="0 = all threads except the reserved ones.">
            <NumberInput label="CPU threads" value={p.cpu_threads} min={0} max={cores} onChange={(v) => setP((q) => ({ ...q, cpu_threads: Math.round(v) }))} suffix={`of ${cores}`} />
          </Field>
          <Field label="Keep threads free for other apps">
            <NumberInput label="Reserved threads" value={p.reserve_cpu_threads} min={0} max={Math.max(0, cores - 1)} onChange={(v) => setP((q) => ({ ...q, reserve_cpu_threads: Math.round(v) }))} />
          </Field>
          <Field label="Process priority" help="Lower priority keeps the desktop responsive; processing gets whatever is left.">
            <Select label="Process priority" value={p.priority} onChange={(v) => setP((q) => ({ ...q, priority: v }))} options={[{ value: "low", label: "Low" }, { value: "below_normal", label: "Below normal" }, { value: "normal", label: "Normal" }, { value: "above_normal", label: "Above normal" }]} />
          </Field>
        </div>
      )}

      <h3 className="mt-l">Memory</h3>
      <Field label="Memory budget for processing" help="ClearFrame sizes tiles and decides how many exports run at once from this budget. It is a target: the operating system can still swap.">
        <Seg label="Memory preset" value={p.memory_preset} onChange={(v) => setP((q) => ({ ...q, memory_preset: v }))} options={MEMORY} />
      </Field>
      {p.memory_preset === "custom" && (
        <Field label="Budget"><Slider label="Memory budget (MB)" value={p.memory_budget_mb} min={512} max={ram} step={256} onChange={(v) => setP((q) => ({ ...q, memory_budget_mb: Math.round(v) }))} /></Field>
      )}
      {plan && <p className="small muted">About {mb(plan.memory_budget_mb)} for processing, {mb(plan.reserve_memory_mb)} left for other apps (of {mb(ram)}).</p>}
      <Field label="When the system runs low on memory">
        <Select label="Memory pressure action" value={p.memory_pressure} onChange={(v) => setP((q) => ({ ...q, memory_pressure: v }))} options={[{ value: "pause", label: "Pause exports safely until memory frees up" }, { value: "continue", label: "Keep going (may slow everything down)" }]} />
      </Field>
      <Field label="Exports at the same time" help="More than one runs only when the memory budget fits all of them.">
        <NumberInput label="Concurrent exports" value={p.max_concurrent_jobs} min={1} max={8} onChange={(v) => setP((q) => ({ ...q, max_concurrent_jobs: Math.round(v) }))} />
      </Field>

      <h3 className="mt-l">Graphics card</h3>
      {h && (
        <div className="small muted" style={{ marginBottom: 10 }}>
          {h.gpus.length ? h.gpus.map((g) => <div key={g.name}>{g.vendor} {g.name}{g.memory_total_mb ? ` · ${mb(g.memory_total_mb)} VRAM` : ""}{g.driver ? ` · driver ${g.driver}` : ""}</div>) : <div>No graphics card detected.</div>}
          {h.backends.filter((b) => b.kind !== "cpu").map((b) => <div key={b.kind} className="faint">{b.kind.toUpperCase()}: {b.available ? "available" : b.detail}</div>)}
        </div>
      )}
      {!anyGpu && <Notice tone="info">GPU acceleration is not available in this build or on this computer. Everything runs on the CPU, which is slower but produces the same results.</Notice>}
      <Field label="Compute device">
        <Select label="Compute device" value={p.device} onChange={(v) => setP((q) => ({ ...q, device: v }))} options={devices} />
      </Field>
      {anyGpu && (
        <Field label="GPU memory limit" help="0 = automatic. Lower values use smaller tiles.">
          <NumberInput label="GPU memory limit" value={p.vram_budget_mb} min={0} max={262144} step={256} suffix="MB" onChange={(v) => setP((q) => ({ ...q, vram_budget_mb: Math.round(v) }))} />
        </Field>
      )}
      <Check checked={p.oom_tile_fallback} onChange={(v) => setP((q) => ({ ...q, oom_tile_fallback: v }))} label="If the device runs out of memory, retry with smaller tiles" desc="Slower, but avoids failed exports on large videos." />

      <h3 className="mt-l"><Activity size={16} style={{ verticalAlign: -2 }} /> Live usage</h3>
      <Monitor />
    </section>
  );
}

function Advanced({ s, patch }: { s: Settings; patch: Patch }) {
  const app = useStore((x) => x.app);
  return (
    <section className="section">
      <h2>Advanced</h2>
      <p className="lead">Processing internals. Defaults suit most people.</p>
      <Field label="FFmpeg folder" hint={app?.ffmpeg ? `Using ${app.ffmpeg.path} (${app.ffmpeg.version}, ${app.ffmpeg.source})` : app?.ffmpeg_error ?? ""}>
        <div className="row">
          <span className="path grow">{s.ffmpeg_dir ?? "Automatic (bundled, then system)"}</span>
          <button className="btn small" onClick={async () => { const f = await pickFolder("Folder containing ffmpeg and ffprobe"); if (f) patch((x) => ({ ...x, ffmpeg_dir: f })); }}>Choose…</button>
          {s.ffmpeg_dir && <button className="btn ghost small" onClick={() => patch((x) => ({ ...x, ffmpeg_dir: null }))}>Automatic</button>}
          <button className="btn small" onClick={async () => {
            try { await api.refreshTools(); const a = await api.appInfo(); useStore.setState({ app: a }); toast("success", `FFmpeg ${a.ffmpeg?.version ?? ""} found.`); } catch (e) { errorToast(e); }
          }}>Re-detect</button>
        </div>
      </Field>
      {app?.ffmpeg && (!app.ffmpeg.vidstab || !app.ffmpeg.zscale) && (
        <Notice tone="warn">This FFmpeg lacks {[!app.ffmpeg.vidstab && "vid.stab (stabilization)", !app.ffmpeg.zscale && "zscale (HDR conversion)"].filter(Boolean).join(" and ")}. Those features are disabled until you use an FFmpeg build that includes them.</Notice>
      )}
      <div className="grid2 mt">
        <Field label="Segment length" help="Exports are written in segments; after a crash only the current segment is redone.">
          <NumberInput label="Segment seconds" value={s.segment_seconds} min={2} max={120} suffix="s" onChange={(v) => patch((x) => ({ ...x, segment_seconds: v }))} />
        </Field>
        <Field label="If a GPU is not usable">
          <Select label="Fallback" value={s.fallback} onChange={(v) => patch((x) => ({ ...x, fallback: v }))} options={[{ value: "ask", label: "Ask me" }, { value: "allow", label: "Use the CPU automatically" }, { value: "never", label: "Fail the export" }]} />
        </Field>
      </div>
      <Check checked={s.half_precision} onChange={(v) => patch((x) => ({ ...x, half_precision: v }))} label="Use half precision (FP16) on GPUs that support it" desc="Faster and uses less memory; tiny numerical differences. Ignored on the CPU." />
      <Check checked={s.stop_queue_on_error} onChange={(v) => patch((x) => ({ ...x, stop_queue_on_error: v }))} label="Pause the queue when an export fails" />
      <Check checked={s.keep_failed_work} onChange={(v) => patch((x) => ({ ...x, keep_failed_work: v }))} label="Keep work files of failed exports" desc="Lets Retry continue where it failed. Cleared by cache cleanup." />
      <div className="row mt-l">
        <ResetSettings />
      </div>
    </section>
  );
}

function ResetSettings() {
  const [ask, setAsk] = useState(false);
  return (
    <>
      <button className="btn danger" onClick={() => setAsk(true)}>Reset all settings</button>
      {ask && (
        <Confirm title="Reset settings?" danger confirmLabel="Reset" onCancel={() => setAsk(false)}
          body="Preferences return to their defaults. Storage locations, models, presets, projects, exports and the queue are not changed."
          onConfirm={async () => { setAsk(false); try { const s = await api.settingsReset(); useStore.setState({ settings: s }); toast("success", "Settings reset."); } catch (e) { errorToast(e); } }} />
      )}
    </>
  );
}

function Diagnostics() {
  const [redact, setRedact] = useState(true);
  const [text, setText] = useState<string | null>(null);
  const gen = () => api.diagnostics(redact).then(setText).catch(errorToast);
  return (
    <section className="section">
      <h2>Diagnostics</h2>
      <p className="lead">A plain-text report about your system, FFmpeg, models and recent errors, for bug reports. It is never uploaded; you decide whether to share it.</p>
      <Check checked={redact} onChange={setRedact} label="Hide personal paths and file names" />
      <div className="row">
        <button className="btn" onClick={gen}>Create report</button>
        {text && <button className="btn" onClick={() => navigator.clipboard?.writeText(text).then(() => toast("success", "Copied."), errorToast)}>Copy</button>}
        {text && <button className="btn" onClick={async () => { const p = await pickSave("Save diagnostics", "clearframe-diagnostics.txt", [{ name: "Text", extensions: ["txt"] }]); if (p) api.writeTextFile(p, text).then(() => toast("success", "Saved."), errorToast); }}>Save…</button>}
      </div>
      {text && <textarea className="input mt" rows={18} readOnly value={text} aria-label="Diagnostics report" style={{ width: "100%" }} />}
    </section>
  );
}

function About() {
  const app = useStore((x) => x.app);
  const [upd, setUpd] = useState<string | null>(null);
  return (
    <section className="section">
      <h2>About ClearFrame</h2>
      <p className="lead">Free, open-source, offline video enhancement. No accounts, no telemetry, no cloud processing.</p>
      <dl className="kv">
        <dt>Version</dt><dd>{app?.version}</dd>
        <dt>Platform</dt><dd>{app?.os} {app?.arch}</dd>
        <dt>Compute backends</dt><dd>{app?.compiled_backends.join(", ")}</dd>
        <dt>FFmpeg</dt><dd>{app?.ffmpeg ? `${app.ffmpeg.version} (${app.ffmpeg.license})` : app?.ffmpeg_error}</dd>
      </dl>
      <p className="small muted mt">ClearFrame is MIT licensed. FFmpeg and each AI model have their own licenses; model licenses are shown in the Model Library and in MODEL_LICENSES.md. Some model licenses do not allow commercial use of results — check before using output commercially.</p>
      <div className="row mt">
        <button className="btn" onClick={() => api.checkUpdate().then((u) => setUpd(u.newer ? `Version ${u.latest} is available: ${u.url}` : `You have the latest version (${u.current}).`)).catch((e) => setUpd(`Could not check: ${e instanceof Error ? e.message : e}`))}>Check for updates</button>
      </div>
      {upd && <p className="small mt selectable">{upd}</p>}
    </section>
  );
}

export function SettingsView() {
  const section = useStore((s) => s.settingsSection);
  const s = useStore((x) => x.settings);
  if (!s) return null;
  const patch: Patch = (f) => { saveSettings(f); };
  return (
    <div className="settings">
      <nav aria-label="Settings sections">
        {SECTIONS.map(({ key, label, icon: I }) => (
          <button key={key} aria-current={section === key ? "page" : undefined} onClick={() => useStore.setState({ settingsSection: key })}><I size={16} /> {label}</button>
        ))}
      </nav>
      <div className="content">
        {section === "general" && <General s={s} patch={patch} />}
        {section === "export" && <ExportSettings s={s} patch={patch} />}
        {section === "storage" && <Storage s={s} patch={patch} />}
        {section === "performance" && <Performance s={s} patch={patch} />}
        {section === "advanced" && <Advanced s={s} patch={patch} />}
        {section === "diagnostics" && <Diagnostics />}
        {section === "about" && <About />}
      </div>
    </div>
  );
}

