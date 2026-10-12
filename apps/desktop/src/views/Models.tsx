import { useEffect, useState } from "react";
import { CheckCircle2, Download, FolderOpen, HardDrive, Pause, Play, RefreshCw, ShieldCheck, Trash2, Upload, Wrench, X } from "lucide-react";

import { Field, Modal, Notice, Progress } from "../components/ui";
import { api, pickFile } from "../lib/ipc";
import { bytes } from "../lib/format";
import { cardId, cardName } from "../lib/ops";
import type { ModelCard, ModelState, StorageOverview } from "../lib/types";
import { errorToast, refreshModels, toast, useStore } from "../state/store";

const STATE_LABEL: Record<ModelState, string> = {
  not_installed: "Not installed", downloading: "Downloading", paused: "Paused", verifying: "Verifying", preparing: "Preparing", testing: "Testing",
  installed: "Installed", update_available: "Update available", missing: "Files missing", broken: "Damaged", failed: "Install failed",
};
const STATE_TONE: Partial<Record<ModelState, string>> = { installed: "ok", update_available: "accent", missing: "err", broken: "err", failed: "err", paused: "warn", downloading: "accent", verifying: "accent", preparing: "accent", testing: "accent" };

const CATEGORIES: { key: string; label: string; match: (c: ModelCard) => boolean }[] = [
  { key: "all", label: "All", match: () => true },
  { key: "installed", label: "Installed", match: (c) => c.state === "installed" || c.state === "update_available" },
  { key: "problems", label: "Needs attention", match: (c) => ["missing", "broken", "failed", "paused"].includes(c.state) },
  { key: "enhancement", label: "Upscale", match: (c) => !!c.manifest?.categories.includes("enhancement") || !!c.installed?.custom },
  { key: "anime", label: "Anime", match: (c) => !!c.manifest?.categories.includes("anime") },
  { key: "denoise", label: "Denoise", match: (c) => !!c.manifest?.categories.includes("denoise") },
  { key: "deblur", label: "Deblur", match: (c) => !!c.manifest?.categories.includes("deblur") },
  { key: "frame_interpolation", label: "Interpolation", match: (c) => !!c.manifest?.categories.includes("frame_interpolation") },
  { key: "face_recovery", label: "Faces", match: (c) => !!c.manifest?.categories.includes("face_recovery") },
];

function UninstallDialog({ card, onClose }: { card: ModelCard; onClose: () => void }) {
  const id = cardId(card);
  const [usage, setUsage] = useState<string[] | null>(null);
  const [inUse, setInUse] = useState(false);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    Promise.all([api.modelAction<string[]>(id, "usage"), api.modelAction<boolean>(id, "in_use")])
      .then(([u, b]) => { setUsage(u); setInUse(b); })
      .catch(errorToast);
  }, [id]);
  const remove = async () => {
    setBusy(true);
    try {
      const r = await api.modelAction<{ freed: number }>(id, "remove");
      toast("success", `Removed ${cardName(card)} (${bytes(r.freed)} freed).`);
      await refreshModels();
      onClose();
    } catch (e) {
      errorToast(e);
      setBusy(false);
    }
  };
  return (
    <Modal title={`Uninstall ${cardName(card)}?`} onClose={onClose} footer={
      <>
        <button className="btn" onClick={onClose}>Keep</button>
        <button className="btn danger" disabled={busy || inUse || usage === null} onClick={remove}>Uninstall</button>
      </>
    }>
      {usage === null ? (
        <p className="muted">Checking where this model is used…</p>
      ) : inUse ? (
        <Notice tone="err">A running export or preview is using this model right now. Wait for it to finish or pause it first.</Notice>
      ) : usage.length === 0 ? (
        <Notice tone="ok">Safe to remove: no preset or queued export uses this model.</Notice>
      ) : (
        <Notice tone="warn">
          Used by:
          <ul style={{ margin: "4px 0 0", paddingLeft: 18 }}>{usage.map((u) => <li key={u}>{u}</li>)}</ul>
          They will ask you to reinstall the model before running.
        </Notice>
      )}
      <p className="small muted mt">This frees {bytes(card.disk_bytes)}. You can reinstall it later; videos, presets, projects and exports are not touched.</p>
    </Modal>
  );
}

function ImportDialog({ onClose }: { onClose: () => void }) {
  const [file, setFile] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [license, setLicense] = useState("");
  const [busy, setBusy] = useState(false);
  const go = async () => {
    if (!file) return;
    setBusy(true);
    try {
      const id = await api.modelImportCustom(name.trim(), license.trim(), file);
      toast("success", `Imported ${name} as ${id}.`);
      await refreshModels();
      onClose();
    } catch (e) {
      errorToast(e);
      setBusy(false);
    }
  };
  return (
    <Modal title="Import a custom upscaling model" onClose={onClose} footer={
      <>
        <button className="btn" onClick={onClose}>Cancel</button>
        <button className="btn primary" disabled={!file || !name.trim() || !license.trim() || busy} onClick={go}>{busy ? "Importing…" : "Import"}</button>
      </>
    }>
      <Notice tone="info">
        Supported: Real-ESRGAN-compatible weights (SRVGGNetCompact or RRDBNet) as <code>.pth</code> or <code>.safetensors</code>. Files are read with a restricted loader that never runs code from the file, converted, and smoke-tested before use.
      </Notice>
      <Field label="Weights file">
        <div className="row">
          <span className="path grow">{file ?? "No file chosen"}</span>
          <button className="btn small" onClick={async () => {
            const [f] = await pickFile("Choose model weights", [{ name: "Model weights", extensions: ["pth", "pt", "safetensors", "ckpt"] }]);
            if (f) {
              setFile(f);
              if (!name) setName(f.split(/[\\/]/).pop()!.replace(/\.[^.]+$/, ""));
            }
          }}>Choose…</button>
        </div>
      </Field>
      <Field label="Display name" htmlFor="imp-name"><input id="imp-name" className="input" value={name} maxLength={80} onChange={(e) => setName(e.target.value)} /></Field>
      <Field label="License" htmlFor="imp-lic" hint="Record the license the weights were published under, so you know how you may use the results.">
        <input id="imp-lic" className="input" value={license} maxLength={200} placeholder="e.g. BSD-3-Clause, CC BY-NC 4.0, unknown" onChange={(e) => setLicense(e.target.value)} />
      </Field>
    </Modal>
  );
}

function ModelCardView({ c }: { c: ModelCard }) {
  const id = cardId(c);
  const prog = useStore((s) => s.installs[id]);
  const [removing, setRemoving] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const m = c.manifest;
  const state: ModelState = prog?.state ?? c.state;
  const working = c.in_progress || !!prog;
  const run = async (label: string, fn: () => Promise<unknown>, ok?: string) => {
    setBusy(label);
    try {
      await fn();
      if (ok) toast("success", ok);
    } catch (e) {
      errorToast(e);
    } finally {
      setBusy(null);
      refreshModels();
    }
  };
  const install = () => run("install", () => api.modelInstall(id));
  const total = prog?.bytes_total || m?.download_size_bytes || 0;
  const done = prog?.bytes_done ?? c.partial_bytes;
  const capTone = c.capability.level === "supported" ? "ok" : c.capability.level === "unavailable" ? "err" : c.capability.level === "cpu_fallback_only" ? "warn" : "";
  const capLabel = { supported: "Tested on this system", compatible_not_certified: "Compatible (not certified)", cpu_fallback_only: "CPU only on this system", unavailable: "Not supported here" }[c.capability.level];

  return (
    <article className="model" data-testid={`model-${id}`} aria-label={cardName(c)}>
      <div className="head">
        <div className="grow">
          <h3>{cardName(c)}</h3>
          <div className="tiny faint">{m ? `${m.publisher} · v${m.version}` : `Custom import · ${c.installed?.custom_info?.license ?? ""}`}</div>
        </div>
        <span className={`badge ${STATE_TONE[state] ?? ""}`} data-testid="model-state">{STATE_LABEL[state]}</span>
      </div>
      {m && <p className="small muted">{m.summary}</p>}
      <div className="row wrap" style={{ gap: 4 }}>
        {m?.recommended && <span className="badge accent">Recommended</span>}
        {m && <span className="badge">{m.scale > 1 ? `${m.scale}× ` : ""}{m.architecture}</span>}
        {m && <span className="badge">{m.performance_tier}</span>}
        <span className={`badge ${capTone}`} title={c.capability.reason}>{capLabel}</span>
      </div>
      <div className="small faint">
        {m && <>Download {bytes(m.download_size_bytes)} · on disk {bytes(c.disk_bytes || m.installed_size_bytes)}</>}
        {!m && <>On disk {bytes(c.disk_bytes)}</>}
        {m && <> · needs {Math.round(m.minimum_ram_bytes / 2 ** 30)} GB RAM</>}
      </div>

      {working && (
        <div className="col" style={{ gap: 4 }}>
          <Progress value={total ? done / total : 0} indeterminate={!total || state !== "downloading"} label={`Installing ${cardName(c)}`} />
          <div className="tiny faint">{prog?.message ?? STATE_LABEL[state]}{total ? ` · ${bytes(done)} of ${bytes(total)}` : ""}</div>
        </div>
      )}
      {state === "paused" && !working && <div className="tiny faint">Paused at {bytes(c.partial_bytes)} of {bytes(total)}. The download resumes where it stopped.</div>}
      {state === "missing" && (
        <Notice tone="err">The model folder was deleted or moved outside ClearFrame. Nothing is downloaded automatically: reinstall it, or remove it from the library.{c.path && <div className="path">{c.path}</div>}</Notice>
      )}
      {state === "broken" && <Notice tone="err">Some files failed verification{c.installed?.error ? `: ${c.installed.error}` : "."} Repair re-checks every file; if the weights themselves are damaged you are asked before anything is downloaded again.</Notice>}
      {state === "failed" && c.installed?.error && <Notice tone="err">{c.installed.error}</Notice>}
      {state === "update_available" && <Notice tone="info">A newer version is in the catalog. The installed version keeps working until you update.</Notice>}

      {m && (
        <details>
          <summary>Best for, limitations, license and source</summary>
          <div className="col mt" style={{ gap: 8 }}>
            {m.best_for.length > 0 && <div><div className="tiny faint">Best for</div><ul>{m.best_for.map((x) => <li key={x}>{x}</li>)}</ul></div>}
            {m.weaknesses.length > 0 && <div><div className="tiny faint">Limitations</div><ul>{m.weaknesses.map((x) => <li key={x}>{x}</li>)}</ul></div>}
            <dl className="kv">
              <dt>Weights license</dt><dd>{m.weights_license}</dd>
              <dt>Code license</dt><dd>{m.code_license}</dd>
              <dt>Commercial use</dt><dd className={m.commercial_use_allowed ? "ok" : "warn"}>{m.commercial_use_allowed ? "Allowed by the license" : "Not allowed / unclear"}</dd>
              <dt>Redistribution</dt><dd>{m.redistribution_allowed ? "Allowed" : "Not allowed"}</dd>
              <dt>Source</dt><dd><span className="path">{m.source_repository}</span></dd>
              <dt>Training data</dt><dd>{m.training_data}</dd>
              <dt>Reviewed</dt><dd>{m.provenance_reviewed_at} · {m.quality_status}</dd>
              {m.artifacts.map((a) => (
                <span key={a.role} style={{ display: "contents" }}>
                  <dt>{a.role}</dt><dd><span className="path">{a.filename} · sha256 {a.sha256.slice(0, 12)}…</span></dd>
                </span>
              ))}
            </dl>
            {m.license_notes && <p className="small muted">{m.license_notes}</p>}
            {m.attribution.length > 0 && <p className="tiny faint">{m.attribution.join(" ")}</p>}
          </div>
        </details>
      )}
      {c.path && (state === "installed" || state === "update_available" || state === "broken") && <div className="path" title={c.path}>{c.path}</div>}

      <div className="actions">
        {working ? (
          <>
            {state === "downloading" && <button className="btn small" onClick={() => run("pause", () => api.modelAction(id, "pause"))}><Pause size={14} /> Pause</button>}
            <button className="btn small danger" onClick={() => run("cancel", () => api.modelAction(id, "cancel"), "Install cancelled. Nothing was left installed.")}><X size={14} /> Cancel</button>
          </>
        ) : (
          <>
            {(state === "not_installed" || state === "failed") && m && (
              <button className="btn small primary" disabled={!!busy || c.capability.level === "unavailable"} onClick={install}>
                <Download size={14} /> {state === "failed" ? "Retry install" : c.partial_bytes > 0 ? "Resume install" : "Install"}
              </button>
            )}
            {state === "paused" && (
              <>
                <button className="btn small primary" onClick={install}><Play size={14} /> Resume</button>
                <button className="btn small danger" onClick={() => run("cancel", () => api.modelAction(id, "cancel"), "Partial download discarded.")}><X size={14} /> Discard</button>
              </>
            )}
            {state === "update_available" && m && <button className="btn small primary" onClick={install}><RefreshCw size={14} /> Update</button>}
            {state === "missing" && m && <button className="btn small primary" onClick={install}><Download size={14} /> Reinstall</button>}
            {state === "broken" && (
              <button className="btn small primary" disabled={!!busy} onClick={() => run("repair", async () => {
                const r = await api.modelAction<{ ok_without_reinstall: boolean }>(id, "repair");
                if (r.ok_without_reinstall) toast("success", `${cardName(c)} repaired without downloading.`);
                else toast("warning", `${cardName(c)} has damaged weights that must be downloaded again (${bytes(m?.download_size_bytes ?? 0)}).`, m ? { label: "Reinstall", run: install } : undefined);
              })}><Wrench size={14} /> {busy === "repair" ? "Repairing…" : "Repair"}</button>
            )}
            {(state === "installed" || state === "update_available") && (
              <button className="btn small" disabled={!!busy} onClick={() => run("verify", async () => {
                const r = await api.modelAction<{ ok: boolean; problems: string[] }>(id, "verify");
                if (r.ok) toast("success", `${cardName(c)}: all files verified.`);
                else toast("error", `${cardName(c)}: ${r.problems.join("; ")}`);
              })}><ShieldCheck size={14} /> {busy === "verify" ? "Verifying…" : "Verify"}</button>
            )}
            {(state === "installed" || state === "update_available" || state === "broken") && (
              <button className="btn small" onClick={() => run("open", () => api.modelAction(id, "open"))}><FolderOpen size={14} /> Open folder</button>
            )}
            {(state === "installed" || state === "update_available" || state === "broken" || state === "missing" || (state === "failed" && !m)) && (
              <button className="btn small danger" onClick={() => setRemoving(true)}><Trash2 size={14} /> {state === "missing" ? "Remove from library" : "Uninstall"}</button>
            )}
          </>
        )}
      </div>
      {removing && <UninstallDialog card={c} onClose={() => { setRemoving(false); refreshModels(); }} />}
    </article>
  );
}

export function ModelsView() {
  const models = useStore((s) => s.models);
  const settings = useStore((s) => s.settings);
  const [cat, setCat] = useState("all");
  const [storage, setStorage] = useState<StorageOverview | null>(null);
  const [importing, setImporting] = useState(false);
  const loadStorage = () => api.storageOverview().then(setStorage).catch(() => {});
  useEffect(() => {
    refreshModels();
    loadStorage();
  }, [models.length]);
  const list = models.filter(CATEGORIES.find((c) => c.key === cat)!.match);
  const installed = models.filter((c) => c.state === "installed" || c.state === "update_available");
  const problems = models.filter((c) => c.state === "missing" || c.state === "broken");
  const s = storage?.models;
  return (
    <div className="page">
      <div className="page-head">
        <div className="grow">
          <h1>Model Library</h1>
          <p className="muted small">Models are downloaded only when you press Install, verified by SHA-256, converted to a safe format and tested before use. After that everything works offline.</p>
        </div>
        <button className="btn" onClick={() => { refreshModels(); loadStorage(); }}><RefreshCw size={15} /> Rescan</button>
        <button className="btn" onClick={() => setImporting(true)}><Upload size={15} /> Import custom…</button>
        {settings?.allow_catalog_updates && (
          <button className="btn" onClick={() => api.updateCatalog().then((m) => { toast("info", m); refreshModels(); }).catch(errorToast)}>Update catalog</button>
        )}
      </div>

      <div className="card" style={{ marginBottom: 14 }}>
        <div className="row wrap gap-l">
          <HardDrive size={20} className="faint" />
          <div className="grow" style={{ minWidth: 240 }}>
            <div className="small faint">Models folder{storage && !storage.paths.models_is_default ? " (custom location)" : ""}</div>
            <div className="path" data-testid="models-dir">{s?.models_dir ?? "…"}</div>
          </div>
          <div className="small"><strong>{installed.length}</strong> installed · {bytes(s?.total_bytes ?? 0)} used · {bytes(s?.free_bytes)} free</div>
          <button className="btn small" disabled={!s?.models_dir_exists} onClick={() => s && api.openPath(s.models_dir).catch(errorToast)}><FolderOpen size={14} /> Open</button>
          <button className="btn small" onClick={() => useStore.setState({ view: "settings", settingsSection: "storage" })}>Change location…</button>
        </div>
        {storage?.warnings.map((w) => <div key={w} className="mt"><Notice tone="warn">{w}</Notice></div>)}
        {problems.length > 0 && (
          <div className="mt"><Notice tone="err">{problems.length} model(s) have missing or damaged files. They are not used until you repair or reinstall them; nothing is downloaded automatically.</Notice></div>
        )}
        {(s?.staging_bytes ?? 0) > 0 && <div className="tiny faint mt">{bytes(s!.staging_bytes)} of partial downloads can be resumed or cleared in Settings → Storage.</div>}
      </div>

      <div className="filters" role="group" aria-label="Category">
        {CATEGORIES.map((c) => {
          const n = models.filter(c.match).length;
          return n || c.key === "all" ? <button key={c.key} className="chip" aria-pressed={cat === c.key} onClick={() => setCat(c.key)}>{c.label} <span className="faint">{n}</span></button> : null;
        })}
      </div>
      {list.length === 0 ? (
        <div className="empty"><CheckCircle2 /> Nothing here.</div>
      ) : (
        <div className="model-grid">{list.map((c) => <ModelCardView key={cardId(c)} c={c} />)}</div>
      )}
      {importing && <ImportDialog onClose={() => setImporting(false)} />}
    </div>
  );
}
