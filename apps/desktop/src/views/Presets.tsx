import { useRef, useState } from "react";
import { Box, MoreHorizontal, Plus, Star, Upload, X } from "lucide-react";

import { Check, Confirm, Dialog, F, useDismiss, useMenu } from "../components/ui";
import { api, pickFile, pickSave } from "../lib/ipc";
import type { Preset } from "../lib/types";
import { applyPreset, clearPipeline, errorToast, refreshPresets, saveSettings, toast, useStore } from "../state/store";

const PRESET_FILTER = [{ name: "ClipFX Enhance preset", extensions: ["cfxpreset"] }];

function sortPresets(list: Preset[]) {
  return [...list].sort((a, b) => Number(b.favorite) - Number(a.favorite) || Number(a.builtin) - Number(b.builtin) || a.name.localeCompare(b.name));
}

async function act(fn: () => Promise<unknown>, ok?: string) {
  try {
    await fn();
    await refreshPresets();
    if (ok) toast("success", ok);
  } catch (e) {
    errorToast(e);
  }
}

function PresetItem({ p, onPick, manage }: { p: Preset; onPick: (p: Preset) => void; manage?: boolean }) {
  const { open, node } = useMenu();
  const [renaming, setRenaming] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [name, setName] = useState(p.name);
  return (
    <div className="item" role="button" tabIndex={0} onClick={() => onPick(p)} onKeyDown={(e) => e.key === "Enter" && onPick(p)} data-testid={`preset-${p.id}`}>
      <div style={{ minWidth: 0 }}>
        <div className="t ellipsis">{p.name}</div>
        <div className="d ellipsis">{p.description || p.category}</div>
      </div>
      <button className="btn ghost sm icon" aria-label={p.favorite ? "Unfavorite" : "Favorite"} aria-pressed={p.favorite}
        onClick={(e) => { e.stopPropagation(); act(() => api.presetAction("favorite", { id: p.id, favorite: !p.favorite })); }}>
        <Star size={14} className={`star${p.favorite ? " on" : ""}`} fill={p.favorite ? "currentColor" : "none"} />
      </button>
      {manage ? (
        <button className="btn ghost sm icon" aria-label={`More for ${p.name}`} onClick={(e) => open(e, [
          { label: "Duplicate", onClick: () => act(() => api.presetAction("duplicate", { id: p.id })) },
          { label: "Rename", onClick: () => setRenaming(true), disabled: p.builtin },
          { label: "Export…", onClick: async () => {
            const path = await pickSave("Export preset", `${p.name.replace(/[^\w\- ]+/g, "_")}.cfxpreset`, PRESET_FILTER);
            if (path) act(() => api.presetAction("export", { id: p.id, path: path.endsWith(".cfxpreset") ? path : `${path}.cfxpreset` }), "Preset exported");
          } },
          { label: "Delete", onClick: () => setDeleting(true), danger: true, disabled: p.builtin, sep: true },
        ])}><MoreHorizontal size={14} /></button>
      ) : <span />}
      {node}
      {renaming && (
        <Dialog title="Rename preset" onClose={() => setRenaming(false)} footer={<>
          <button className="btn ghost" onClick={() => setRenaming(false)}>Cancel</button>
          <button className="btn primary" disabled={!name.trim()} onClick={() => { setRenaming(false); act(() => api.presetAction("rename", { id: p.id, name: name.trim() })); }}>Rename</button>
        </>}>
          <F label="Name" htmlFor="rn"><input id="rn" className="input" value={name} onChange={(e) => setName(e.target.value)} /></F>
        </Dialog>
      )}
      {deleting && <Confirm title={`Delete “${p.name}”?`} danger confirmLabel="Delete" onCancel={() => setDeleting(false)} onConfirm={() => { setDeleting(false); act(() => api.presetAction("delete", { id: p.id }), "Preset deleted"); }} />}
    </div>
  );
}

export function SavePresetDialog({ onClose }: { onClose: () => void }) {
  const pipeline = useStore((s) => s.pipeline);
  const output = useStore((s) => s.output);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [withOutput, setWithOutput] = useState(false);
  const save = async () => {
    const p: Preset = { id: "", name: name.trim(), description: description.trim(), category: "My presets", tags: [], favorite: false, builtin: false, pipeline, output: withOutput ? output : null };
    try {
      await api.presetAction("save", { preset: p });
      await refreshPresets();
      useStore.setState({ presetName: p.name });
      toast("success", `Saved “${p.name}”`);
      onClose();
    } catch (e) {
      errorToast(e);
    }
  };
  return (
    <Dialog title="Create preset" onClose={onClose} footer={<>
      <button className="btn ghost" onClick={onClose}>Cancel</button>
      <button className="btn primary" disabled={!name.trim() || !pipeline.stages.some((s) => s.enabled)} onClick={save}>Save</button>
    </>}>
      <div className="prefs" style={{ gap: 10 }}>
        <F label="Name" htmlFor="pr-name"><input id="pr-name" className="input" value={name} maxLength={80} onChange={(e) => setName(e.target.value)} /></F>
        <F label="Description" htmlFor="pr-desc"><input id="pr-desc" className="input" value={description} maxLength={200} onChange={(e) => setDescription(e.target.value)} /></F>
        <F label="Codec settings"><Check checked={withOutput} onChange={setWithOutput} label="Include" /></F>
      </div>
    </Dialog>
  );
}

export function PresetsPopover({ onClose }: { onClose: () => void }) {
  const presets = useStore((s) => s.presets);
  const ref = useRef<HTMLDivElement>(null);
  const [creating, setCreating] = useState(false);
  useDismiss(ref, onClose, !creating && !document.querySelector(".backdrop"));
  const importPreset = async () => {
    const [path] = await pickFile("Import preset", PRESET_FILTER);
    if (!path) return;
    try {
      const r = await api.presetAction<{ preset: Preset; missing_models: string[] }>("import", { path });
      await refreshPresets();
      toast(r.missing_models.length ? "warning" : "success", `Imported “${r.preset.name}”${r.missing_models.length ? ` · needs ${r.missing_models.length} model(s)` : ""}`);
    } catch (e) {
      errorToast(e);
    }
  };
  const list = sortPresets(presets);
  const mine = list.filter((p) => !p.builtin);
  const builtin = list.filter((p) => p.builtin);
  const pick = (p: Preset) => { applyPreset(p); onClose(); };
  return (
    <div className="pop" ref={ref} style={{ top: 44, right: 120 }} role="dialog" aria-label="Presets">
      <div className="pop-h">
        <span className="grow">Presets</span>
        <button className="btn ghost sm icon" aria-label="Import preset" title="Import" onClick={importPreset}><Upload size={14} /></button>
        <button className="btn ghost sm icon" aria-label="Close presets" onClick={onClose}><X size={14} /></button>
      </div>
      <div className="pop-b plist">
        {mine.length > 0 && <div className="grp">Yours</div>}
        {mine.map((p) => <PresetItem key={p.id} p={p} onPick={pick} manage />)}
        {mine.length > 0 && <div className="grp">Built-in</div>}
        {builtin.map((p) => <PresetItem key={p.id} p={p} onPick={pick} manage />)}
      </div>
      <div className="pop-f">
        <button className="btn ghost" onClick={() => setCreating(true)}><Plus size={14} /> Create new</button>
      </div>
      {creating && <SavePresetDialog onClose={() => setCreating(false)} />}
    </div>
  );
}

/** “How do you want to edit?” shown after opening a video. */
export function StartDialog() {
  const presets = useStore((s) => s.presets);
  const settings = useStore((s) => s.settings);
  const [hide, setHide] = useState(!!settings?.hide_start_dialog);
  const close = () => {
    useStore.setState({ dialog: null });
    if (hide !== !!settings?.hide_start_dialog) saveSettings((s) => ({ ...s, hide_start_dialog: hide }));
  };
  return (
    <Dialog title="How do you want to edit?" onClose={close} size="mid" flush>
      <div className="start">
        <div>
          <Box size={30} strokeWidth={1.3} className="ico" />
          <span className="t">Start from scratch</span>
          <span className="d">No adjustments applied</span>
          <button className="btn primary" style={{ alignSelf: "flex-start" }} onClick={() => { clearPipeline(); close(); }}>Start editing</button>
          <span className="grow" />
          <Check checked={hide} onChange={setHide} label="Don't show again" />
        </div>
        <div>
          <Star size={30} strokeWidth={1.3} className="ico" />
          <span className="t">Start from preset</span>
          <span className="d">Choose a preset below to get started</span>
          <div className="plist">
            {sortPresets(presets).map((p) => <PresetItem key={p.id} p={p} onPick={(x) => { applyPreset(x); close(); }} />)}
          </div>
        </div>
      </div>
    </Dialog>
  );
}
