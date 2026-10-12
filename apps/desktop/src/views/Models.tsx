import { useEffect, useState } from "react";
import { Download, FolderOpen, Pause, Play, RefreshCw, ShieldCheck, Trash2, Upload, Wrench, X } from "lucide-react";

import { Dialog, F, Progress } from "../components/ui";
import { api, pickFile } from "../lib/ipc";
import { bytes } from "../lib/format";
import { cardId, cardName, isInstalled, shortName, tagline } from "../lib/ops";
import type { ModelCard, ModelState, StorageOverview } from "../lib/types";
import { errorToast, refreshModels, toast, useStore } from "../state/store";

const LABEL: Record<ModelState, string> = {
  not_installed: "Not installed", downloading: "Downloading", paused: "Paused", verifying: "Verifying", preparing: "Preparing", testing: "Testing",
  installed: "Installed", update_available: "Update", missing: "Missing", broken: "Damaged", failed: "Failed",
};
const TONE: Partial<Record<ModelState, string>> = { installed: "ok", update_available: "accent", missing: "err", broken: "err", failed: "err", paused: "warn" };

const CATS: { key: string; label: string; match: (c: ModelCard) => boolean }[] = [
  { key: "all", label: "All models", match: () => true },
  { key: "installed", label: "Installed", match: (c) => isInstalled(c) },
  { key: "enhancement", label: "Upscaling", match: (c) => !!c.manifest?.categories.includes("enhancement") || !!c.installed?.custom },
  { key: "denoise", label: "Denoise", match: (c) => !!c.manifest?.categories.includes("denoise") },
  { key: "deblur", label: "Deblur", match: (c) => !!c.manifest?.categories.includes("deblur") },
  { key: "frame_interpolation", label: "Interpolation", match: (c) => !!c.manifest?.categories.includes("frame_interpolation") },
  { key: "face_recovery", label: "Faces", match: (c) => !!c.manifest?.categories.includes("face_recovery") },
  { key: "problems", label: "Needs attention", match: (c) => ["missing", "broken", "failed"].includes(c.state) },
];

function Status({ c }: { c: ModelCard }) {
  const p = useStore((s) => s.installs[cardId(c)]);
  if (p || c.in_progress) {
    const frac = p && p.bytes_total ? p.bytes_done / p.bytes_total : 0;
    return (
      <div style={{ width: 110 }}>
        <Progress value={frac} indeterminate={!p || p.state !== "downloading"} label={`Installing ${shortName(c)}`} />
        <div className="faint" style={{ fontSize: 10.5, marginTop: 3 }}>{p?.state === "downloading" ? `${Math.round(frac * 100)}%` : LABEL[p?.state ?? "downloading"]}</div>
      </div>
    );
  }
  return <span className={`tag ${TONE[c.state] ?? ""}`} data-testid="model-state">{LABEL[c.state]}</span>;
}

function Actions({ c, onUninstall }: { c: ModelCard; onUninstall: () => void }) {
  const id = cardId(c);
  const p = useStore((s) => s.installs[id]);
  const m = c.manifest;
  const run = (fn: () => Promise<unknown>) => fn().catch(errorToast).finally(refreshModels);
  const working = !!p || c.in_progress;
  if (working) {
    return (
      <div className="row">
        {(p?.state ?? "downloading") === "downloading" && <button className="btn sm" onClick={() => run(() => api.modelAction(id, "pause"))}><Pause size={13} /> Pause</button>}
        <button className="btn sm ghost" onClick={() => run(() => api.modelAction(id, "cancel"))}><X size={13} /> Cancel</button>
      </div>
    );
  }
  const install = () => run(() => api.modelInstall(id));
  return (
    <div className="row" style={{ flexWrap: "wrap" }}>
      {m && (c.state === "not_installed" || c.state === "failed") && <button className="btn sm primary" disabled={c.capability.level === "unavailable"} onClick={install}><Download size={13} /> {c.state === "failed" ? "Retry" : c.partial_bytes > 0 ? "Resume" : "Install"}</button>}
      {m && c.state === "paused" && <><button className="btn sm primary" onClick={install}><Play size={13} /> Resume</button><button className="btn sm ghost" onClick={() => run(() => api.modelAction(id, "cancel"))}>Discard</button></>}
      {m && c.state === "update_available" && <button className="btn sm primary" onClick={install}><RefreshCw size={13} /> Update</button>}
      {m && c.state === "missing" && <button className="btn sm primary" onClick={install}><Download size={13} /> Reinstall</button>}
      {c.state === "broken" && (
        <button className="btn sm primary" onClick={() => run(async () => {
          const r = await api.modelAction<{ ok_without_reinstall: boolean }>(id, "repair");
          if (r.ok_without_reinstall) toast("success", `${shortName(c)} repaired`);
          else toast("warning", `${shortName(c)} needs a fresh download (${bytes(m?.download_size_bytes ?? 0)})`, m ? { label: "Reinstall", run: install } : undefined);
        })}><Wrench size={13} /> Repair</button>
      )}
      {isInstalled(c) && <button className="btn sm" onClick={() => run(async () => {
        const r = await api.modelAction<{ ok: boolean; problems: string[] }>(id, "verify");
        r.ok ? toast("success", `${shortName(c)} verified`) : toast("error", `${shortName(c)}: ${r.problems.join("; ")}`);
      })}><ShieldCheck size={13} /> Verify</button>}
      {(isInstalled(c) || c.state === "broken") && <button className="btn sm" onClick={() => run(() => api.modelAction(id, "open"))}><FolderOpen size={13} /> Open</button>}
      {(isInstalled(c) || c.state === "broken" || c.state === "missing") && <button className="btn sm danger" onClick={onUninstall}><Trash2 size={13} /> {c.state === "missing" ? "Remove" : "Uninstall"}</button>}
    </div>
  );
}

function Uninstall({ c, onClose }: { c: ModelCard; onClose: () => void }) {
  const id = cardId(c);
  const [usage, setUsage] = useState<string[] | null>(null);
  const [inUse, setInUse] = useState(false);
  useEffect(() => {
    Promise.all([api.modelAction<string[]>(id, "usage"), api.modelAction<boolean>(id, "in_use")]).then(([u, b]) => { setUsage(u); setInUse(b); }).catch(errorToast);
  }, [id]);
  const go = async () => {
    try {
      const r = await api.modelAction<{ freed: number }>(id, "remove");
      toast("success", `Removed ${shortName(c)} · ${bytes(r.freed)} freed`);
      onClose();
    } catch (e) { errorToast(e); }
    refreshModels();
  };
  return (
    <Dialog title={`Uninstall ${shortName(c)}?`} onClose={onClose} footer={<>
      <button className="btn ghost" onClick={onClose}>Keep</button>
      <button className="btn danger" disabled={usage === null || inUse} onClick={go}>Uninstall</button>
    </>}>
      <div className="prefs" style={{ gap: 8 }}>
        <F label="Frees">{bytes(c.disk_bytes)}</F>
        <F label="In use">{usage === null ? "…" : inUse ? <span className="err">By a running export</span> : usage.length === 0 ? <span className="ok">Not used — safe to remove</span> : <span className="warn">{usage.join(", ")}</span>}</F>
      </div>
    </Dialog>
  );
}

function Detail({ c }: { c: ModelCard }) {
  const [uninstall, setUninstall] = useState(false);
  const m = c.manifest;
  return (
    <div className="detail" data-testid="model-detail">
      <div>
        <div style={{ fontSize: 14 }}>{cardName(c)}</div>
        <div className="faint">{m ? `${m.publisher} · v${m.version}` : "Custom import"}</div>
      </div>
      <Actions c={c} onUninstall={() => setUninstall(true)} />
      {c.state === "missing" && <div className="issue err">Folder deleted or moved outside the app. Nothing is downloaded until you choose Reinstall.</div>}
      {c.state === "broken" && <div className="issue err">{c.installed?.error ?? "Files failed verification."}</div>}
      {c.state === "failed" && c.installed?.error && <div className="issue err">{c.installed.error}</div>}
      {m && <p className="muted" style={{ fontSize: 11.5, fontWeight: 400 }}>{m.summary}</p>}
      {m && (
        <dl className="kv">
          <dt>Scale</dt><dd>{m.scale}× · {m.architecture}</dd>
          <dt>Download</dt><dd>{bytes(m.download_size_bytes)}</dd>
          <dt>On disk</dt><dd>{bytes(c.disk_bytes || m.installed_size_bytes)}</dd>
          <dt>RAM</dt><dd>{Math.round(m.minimum_ram_bytes / 2 ** 30)} GB minimum</dd>
          <dt>This PC</dt><dd>{c.capability.reason}</dd>
        </dl>
      )}
      {m && m.best_for.length > 0 && <div><div className="faint" style={{ fontSize: 11, marginBottom: 4 }}>Good for</div><ul className="bullets">{m.best_for.map((x) => <li key={x}>{x}</li>)}</ul></div>}
      {m && m.weaknesses.length > 0 && <div><div className="faint" style={{ fontSize: 11, marginBottom: 4 }}>Limitations</div><ul className="bullets">{m.weaknesses.map((x) => <li key={x}>{x}</li>)}</ul></div>}
      {m && (
        <dl className="kv">
          <dt>Weights</dt><dd>{m.weights_license}</dd>
          <dt>Code</dt><dd>{m.code_license}</dd>
          <dt>Commercial</dt><dd className={m.commercial_use_allowed ? "ok" : "warn"}>{m.commercial_use_allowed ? "Allowed" : "Not allowed / unclear"}</dd>
          <dt>Source</dt><dd className="path">{m.source_repository}</dd>
          <dt>Training</dt><dd>{m.training_data}</dd>
          <dt>Reviewed</dt><dd>{m.provenance_reviewed_at}</dd>
          {m.license_notes && <><dt>Notes</dt><dd className="muted">{m.license_notes}</dd></>}
        </dl>
      )}
      {c.installed?.custom_info && <dl className="kv"><dt>License</dt><dd>{c.installed.custom_info.license}</dd><dt>From</dt><dd className="path">{c.installed.custom_info.source}</dd></dl>}
      {c.path && <dl className="kv"><dt>Folder</dt><dd className="path">{c.path}</dd></dl>}
      {uninstall && <Uninstall c={c} onClose={() => setUninstall(false)} />}
    </div>
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
      await api.modelImportCustom(name.trim(), license.trim(), file);
      toast("success", `Imported ${name}`);
      await refreshModels();
      onClose();
    } catch (e) { errorToast(e); setBusy(false); }
  };
  return (
    <Dialog title="Import upscaling model" onClose={onClose} footer={<>
      <button className="btn ghost" onClick={onClose}>Cancel</button>
      <button className="btn primary" disabled={!file || !name.trim() || !license.trim() || busy} onClick={go}>{busy ? "Testing…" : "Import"}</button>
    </>}>
      <div className="prefs" style={{ gap: 10 }}>
        <F label="Weights (.pth, .safetensors)">
          <div className="row">
            <span className="path grow ellipsis">{file ?? "—"}</span>
            <button className="btn sm" onClick={async () => {
              const [f] = await pickFile("Model weights", [{ name: "Weights", extensions: ["pth", "pt", "safetensors", "ckpt"] }]);
              if (f) { setFile(f); if (!name) setName(f.split(/[\\/]/).pop()!.replace(/\.[^.]+$/, "")); }
            }}>Browse</button>
          </div>
        </F>
        <F label="Name" htmlFor="im-n"><input id="im-n" className="input" value={name} maxLength={80} onChange={(e) => setName(e.target.value)} /></F>
        <F label="License" htmlFor="im-l"><input id="im-l" className="input" value={license} maxLength={200} placeholder="e.g. BSD-3-Clause" onChange={(e) => setLicense(e.target.value)} /></F>
      </div>
    </Dialog>
  );
}

export function ModelsDialog() {
  const models = useStore((s) => s.models);
  const focus = useStore((s) => s.modelFocus);
  const settings = useStore((s) => s.settings);
  const [cat, setCat] = useState("all");
  const [sel, setSel] = useState<string | null>(focus);
  const [storage, setStorage] = useState<StorageOverview | null>(null);
  const [importing, setImporting] = useState(false);
  useEffect(() => { refreshModels(); api.storageOverview().then(setStorage).catch(() => {}); }, []);
  const list = models.filter(CATS.find((c) => c.key === cat)!.match);
  const selected = models.find((c) => cardId(c) === sel) ?? list[0];
  const close = () => useStore.setState({ dialog: null, modelFocus: null });
  const s = storage?.models;
  return (
    <Dialog title="Models" onClose={close} size="wide" flush headerExtra={<>
      {settings?.allow_catalog_updates && <button className="btn ghost sm" onClick={() => api.updateCatalog().then((m) => { toast("info", m); refreshModels(); }).catch(errorToast)}>Update catalog</button>}
      <button className="btn ghost sm" onClick={() => setImporting(true)}><Upload size={13} /> Import</button>
      <button className="btn ghost sm icon" aria-label="Rescan" title="Rescan" onClick={() => refreshModels()}><RefreshCw size={13} /></button>
    </>}>
      <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
        <div className="models-dlg split-dialog" style={{ flex: 1, minHeight: 0 }}>
          <nav aria-label="Categories">
            {CATS.map((c) => {
              const n = models.filter(c.match).length;
              return (n > 0 || c.key === "all") && (
                <button key={c.key} aria-current={cat === c.key ? "page" : undefined} onClick={() => setCat(c.key)}>{c.label}<span className="n">{n}</span></button>
              );
            })}
          </nav>
          <div className="list">
            <table className="mtable">
              <thead><tr><th>Model</th><th>Use</th><th className="r">Size</th><th>Status</th></tr></thead>
              <tbody>
                {list.map((c) => (
                  <tr key={cardId(c)} className={selected && cardId(selected) === cardId(c) ? "sel" : ""} onClick={() => setSel(cardId(c))} data-testid={`model-${cardId(c)}`}>
                    <td style={{ whiteSpace: "nowrap" }}>{shortName(c)}{c.manifest?.recommended && <span className="star on" title="Recommended"> ★</span>}</td>
                    <td className="faint" style={{ fontStyle: "italic", maxWidth: 220 }}><div className="ellipsis">{tagline(c)}</div></td>
                    <td className="r muted">{bytes(c.manifest?.download_size_bytes ?? c.disk_bytes)}</td>
                    <td><Status c={c} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {selected ? <Detail key={cardId(selected)} c={selected} /> : <div className="detail" />}
        </div>
        <div className="dlg-foot">
          <span className="path ellipsis grow" data-testid="models-dir" title={s?.models_dir}>{s?.models_dir ?? "…"}</span>
          <span>{bytes(s?.total_bytes ?? 0)} used · {bytes(s?.free_bytes)} free</span>
          <button className="btn ghost sm" disabled={!s?.models_dir_exists} onClick={() => s && api.openPath(s.models_dir).catch(errorToast)}><FolderOpen size={13} /> Open</button>
          <button className="btn ghost sm" onClick={() => useStore.setState({ dialog: "prefs", settingsSection: "storage" })}>Change location</button>
        </div>
        {storage?.warnings.map((w) => <div key={w} className="issue warn" style={{ padding: "0 12px 8px" }}>{w}</div>)}
      </div>
      {importing && <ImportDialog onClose={() => setImporting(false)} />}
    </Dialog>
  );
}
