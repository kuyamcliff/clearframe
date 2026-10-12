import { useState } from "react";
import { Copy, Download, Pencil, Star, Trash2, Upload } from "lucide-react";

import { Check, Confirm, Field, Modal } from "../components/ui";
import { api, pickFile, pickSave } from "../lib/ipc";
import { cardId, cardName, opInfo, requiredModels } from "../lib/ops";
import type { Preset } from "../lib/types";
import { applyPreset, errorToast, refreshPresets, toast, useStore } from "../state/store";

const PRESET_FILTER = [{ name: "ClearFrame preset", extensions: ["cframe"] }];

export function SavePresetDialog({ onClose }: { onClose: () => void }) {
  const pipeline = useStore((s) => s.pipeline);
  const output = useStore((s) => s.output);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [category, setCategory] = useState("My presets");
  const [withOutput, setWithOutput] = useState(false);
  const save = async () => {
    try {
      const p: Preset = { id: "", name: name.trim(), description, category, tags: [], favorite: false, builtin: false, pipeline, output: withOutput ? output : null };
      await api.presetAction("save", { preset: p });
      await refreshPresets();
      toast("success", `Saved preset “${p.name}”.`);
      onClose();
    } catch (e) {
      errorToast(e);
    }
  };
  return (
    <Modal title="Save as preset" onClose={onClose} footer={
      <>
        <button className="btn" onClick={onClose}>Cancel</button>
        <button className="btn primary" disabled={!name.trim() || pipeline.stages.length === 0} onClick={save}>Save preset</button>
      </>
    }>
      <Field label="Name" htmlFor="pr-name"><input id="pr-name" className="input" value={name} maxLength={80} onChange={(e) => setName(e.target.value)} /></Field>
      <Field label="Description" htmlFor="pr-desc"><input id="pr-desc" className="input" value={description} maxLength={400} onChange={(e) => setDescription(e.target.value)} /></Field>
      <Field label="Category" htmlFor="pr-cat"><input id="pr-cat" className="input" value={category} maxLength={40} onChange={(e) => setCategory(e.target.value)} /></Field>
      <Check checked={withOutput} onChange={setWithOutput} label="Include export format (codec, quality, audio)" />
      {pipeline.stages.length === 0 && <p className="small warn">The current pipeline has no stages.</p>}
    </Modal>
  );
}

function PresetCard({ p }: { p: Preset }) {
  const models = useStore((s) => s.models);
  const [renaming, setRenaming] = useState(false);
  const [name, setName] = useState(p.name);
  const [deleting, setDeleting] = useState(false);
  const missing = requiredModels(p.pipeline).filter((id) => {
    const c = models.find((m) => cardId(m) === id);
    return !c || !(c.state === "installed" || c.state === "update_available");
  });
  const act = async (fn: () => Promise<unknown>, ok?: string) => {
    try {
      await fn();
      await refreshPresets();
      if (ok) toast("success", ok);
    } catch (e) {
      errorToast(e);
    }
  };
  return (
    <div className="card" data-testid={`preset-${p.id}`}>
      <div className="row" style={{ alignItems: "flex-start" }}>
        <div className="grow">
          <div className="row">
            <h3>{p.name}</h3>
            {p.builtin && <span className="badge">built-in</span>}
          </div>
          <div className="tiny faint">{p.category}</div>
        </div>
        <button className="btn ghost icon small" aria-label={p.favorite ? "Remove from favorites" : "Add to favorites"} aria-pressed={p.favorite}
          onClick={() => act(() => api.presetAction("favorite", { id: p.id, favorite: !p.favorite }))}>
          <Star size={15} fill={p.favorite ? "currentColor" : "none"} className={p.favorite ? "warn" : ""} />
        </button>
      </div>
      {p.description && <p className="small muted mt" style={{ marginTop: 6 }}>{p.description}</p>}
      <div className="row wrap mt" style={{ marginTop: 8, gap: 4 }}>
        {p.pipeline.stages.filter((s) => s.enabled).map((s) => <span key={s.id} className="badge">{opInfo(s.type).label}</span>)}
        {p.output && <span className="badge accent">export format</span>}
      </div>
      {missing.length > 0 && (
        <p className="tiny warn mt" style={{ marginTop: 6 }}>
          Needs: {missing.map((id) => { const c = models.find((m) => cardId(m) === id); return c ? cardName(c) : id; }).join(", ")}
        </p>
      )}
      <div className="row wrap mt" style={{ marginTop: 10 }}>
        <button className="btn small primary" onClick={() => { applyPreset(p); useStore.setState({ view: "workspace" }); }}>Use</button>
        <button className="btn ghost small" onClick={() => act(() => api.presetAction("duplicate", { id: p.id }), "Duplicated.")}><Copy size={13} /> Duplicate</button>
        {!p.builtin && <button className="btn ghost small" onClick={() => setRenaming(true)}><Pencil size={13} /> Rename</button>}
        <button className="btn ghost small" onClick={async () => {
          const path = await pickSave("Export preset", `${p.name.replace(/[^\w\- ]+/g, "_")}.cframe`, PRESET_FILTER);
          if (path) act(() => api.presetAction("export", { id: p.id, path: path.endsWith(".cframe") ? path : `${path}.cframe` }), "Preset exported.");
        }}><Download size={13} /> Export</button>
        {!p.builtin && <button className="btn ghost small danger" onClick={() => setDeleting(true)}><Trash2 size={13} /> Delete</button>}
      </div>
      {renaming && (
        <Modal title="Rename preset" onClose={() => setRenaming(false)} footer={
          <>
            <button className="btn" onClick={() => setRenaming(false)}>Cancel</button>
            <button className="btn primary" disabled={!name.trim()} onClick={() => { setRenaming(false); act(() => api.presetAction("rename", { id: p.id, name: name.trim() })); }}>Rename</button>
          </>
        }>
          <Field label="Name" htmlFor="rn"><input id="rn" className="input" value={name} onChange={(e) => setName(e.target.value)} /></Field>
        </Modal>
      )}
      {deleting && (
        <Confirm title={`Delete “${p.name}”?`} danger confirmLabel="Delete preset" onCancel={() => setDeleting(false)}
          body="Only the preset is deleted. Videos, projects and exports that used it are not affected."
          onConfirm={() => { setDeleting(false); act(() => api.presetAction("delete", { id: p.id }), "Preset deleted."); }} />
      )}
    </div>
  );
}

export function PresetsView() {
  const presets = useStore((s) => s.presets);
  const [q, setQ] = useState("");
  const [cat, setCat] = useState<string>("All");
  const [saving, setSaving] = useState(false);
  const cats = ["All", "Favorites", ...Array.from(new Set(presets.map((p) => p.category))).sort()];
  const list = presets
    .filter((p) => cat === "All" || (cat === "Favorites" ? p.favorite : p.category === cat))
    .filter((p) => !q || `${p.name} ${p.description} ${p.tags.join(" ")}`.toLowerCase().includes(q.toLowerCase()))
    .sort((a, b) => Number(b.favorite) - Number(a.favorite) || Number(a.builtin) - Number(b.builtin) || a.name.localeCompare(b.name));
  const importPreset = async () => {
    const [path] = await pickFile("Import preset", PRESET_FILTER);
    if (!path) return;
    try {
      const r = await api.presetAction<{ preset: Preset; missing_models: string[] }>("import", { path });
      await refreshPresets();
      toast(r.missing_models.length ? "warning" : "success", `Imported “${r.preset.name}”.${r.missing_models.length ? ` It needs models you have not installed: ${r.missing_models.join(", ")}. Nothing was downloaded.` : ""}`);
    } catch (e) {
      errorToast(e);
    }
  };
  return (
    <div className="page">
      <div className="page-head">
        <div className="grow">
          <h1>Presets</h1>
          <p className="muted small">Reusable pipelines. Presets never download models by themselves.</p>
        </div>
        <input className="input" placeholder="Search presets" aria-label="Search presets" value={q} onChange={(e) => setQ(e.target.value)} />
        <button className="btn" onClick={importPreset}><Upload size={15} /> Import…</button>
        <button className="btn primary" onClick={() => setSaving(true)}>Save current as preset</button>
      </div>
      <div className="filters" role="group" aria-label="Category">
        {cats.map((c) => <button key={c} className="chip" aria-pressed={cat === c} onClick={() => setCat(c)}>{c}</button>)}
      </div>
      {list.length === 0 ? <div className="empty">No presets match.</div> : <div className="preset-grid">{list.map((p) => <PresetCard key={p.id} p={p} />)}</div>}
      {saving && <SavePresetDialog onClose={() => setSaving(false)} />}
    </div>
  );
}
