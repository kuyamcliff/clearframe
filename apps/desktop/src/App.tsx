import { useEffect, useState } from "react";
import { Blend, Boxes, ChevronRight, Columns2, FolderInput, Layers, PanelRightClose, PanelRightOpen, RectangleHorizontal, Settings as Gear, Sparkles, SquareSplitHorizontal } from "lucide-react";

import { Confirm, Toasts, useMenu, type MenuItem } from "./components/ui";
import { UpdateBanner } from "./components/UpdateBanner";
import { api, inTauri, onFileDrop, VIDEO_EXTENSIONS } from "./lib/ipc";
import { basename, timecode } from "./lib/format";
import { boot, clearPipeline, playSource, refreshPresets, stopSource, closeSource, errorToast, openSource, redo, renderClip, renderStill, toast, undo, useStore, type CompareMode } from "./state/store";
import { ModelsDialog } from "./views/Models";
import { chooseVideos, Player } from "./views/Player";
import { PreferencesDialog } from "./views/Preferences";
import { PresetsPopover, StartDialog } from "./views/Presets";
import { openProjectFile, saveProject } from "./views/Projects";
import { exportNow, Sidebar } from "./views/Sidebar";

const RUNNING = ["preparing", "previewing", "processing", "encoding", "validating"];

function menus(): Record<string, MenuItem[]> {
  const s = useStore.getState();
  const has = !!s.source;
  const mode = (c: CompareMode) => () => useStore.setState({ compare: c });
  return {
    File: [
      { label: "Open video…", onClick: () => chooseVideos().catch(errorToast) },
      { label: "Open project…", onClick: () => openProjectFile().catch(errorToast) },
      { label: "Save project", onClick: () => saveProject(false).catch(errorToast), disabled: !has, sep: true },
      { label: "Save project as…", onClick: () => saveProject(true).catch(errorToast), disabled: !has },
      { label: "Close video", onClick: closeSource, disabled: !s.sourcePath, sep: true },
    ],
    Edit: [
      { label: "Undo", onClick: undo, disabled: !s.history.past.length },
      { label: "Redo", onClick: redo, disabled: !s.history.future.length },
      { label: "Clear adjustments", onClick: clearPipeline, disabled: !s.pipeline.stages.length, sep: true },
      { label: "Clear in/out", onClick: () => useStore.setState({ range: null }), disabled: !s.range },
    ],
    View: [
      { label: "Original", onClick: mode("original") },
      { label: "Preview", onClick: mode("processed") },
      { label: "Split", onClick: mode("split") },
      { label: "Side by side", onClick: mode("side") },
      { label: s.sideOpen ? "Hide controls" : "Show controls", onClick: () => useStore.setState({ sideOpen: !s.sideOpen }), sep: true },
      { label: "Sources", onClick: () => useStore.setState({ panel: s.panel === "sources" ? null : "sources" }) },
      { label: "Export queue", onClick: () => useStore.setState({ panel: s.panel === "queue" ? null : "queue" }) },
    ],
    Process: [
      { label: "Render frame", onClick: () => renderStill(), disabled: !has },
      { label: `Render ${s.settings?.preview_seconds ?? 3}s`, onClick: () => renderClip(), disabled: !has },
      { label: "Export as…", onClick: () => exportNow(true), disabled: !has, sep: true },
      { label: "Quick export", onClick: () => exportNow(false), disabled: !has },
      { label: s.queuePaused ? "Resume queue" : "Pause queue", onClick: () => api.queueAction(s.queuePaused ? "resume" : "pause").then(() => useStore.setState({ queuePaused: !s.queuePaused })).catch(errorToast), sep: true },
    ],
    Help: [
      { label: "Models", onClick: () => useStore.setState({ dialog: "models" }) },
      { label: "Diagnostics", onClick: () => useStore.setState({ dialog: "prefs", settingsSection: "diagnostics" }) },
      { label: "About", onClick: () => useStore.setState({ dialog: "prefs", settingsSection: "about" }) },
    ],
  };
}

function TopBar({ presets, setPresets }: { presets: boolean; setPresets: (v: boolean) => void }) {
  const sourcePath = useStore((s) => s.sourcePath);
  const projectName = useStore((s) => s.projectName);
  const dirty = useStore((s) => s.dirty);
  const compare = useStore((s) => s.compare);
  const sideOpen = useStore((s) => s.sideOpen);
  const app = useStore((s) => s.app);
  const problems = useStore((s) => s.models.some((m) => m.state === "missing" || m.state === "broken"));
  const { open, node } = useMenu();
  const modes: { value: CompareMode; label: string; icon: typeof Sparkles }[] = [
    { value: "original", label: "Original", icon: RectangleHorizontal },
    { value: "processed", label: "Preview", icon: Sparkles },
    { value: "split", label: "Split view", icon: SquareSplitHorizontal },
    { value: "side", label: "Side by side", icon: Columns2 },
  ];
  return (
    <header className="topbar">
      <div className="brand"><img src="/logo.svg" alt="" />ClipFX <b>Enhance</b></div>
      <nav className="row" style={{ gap: 0, padding: "0 6px" }} aria-label="Menu">
        {Object.keys(menus()).map((k) => <button key={k} className="tb" onClick={(e) => open(e, menus()[k])}>{k}</button>)}
      </nav>
      <div className="crumbs">
        <span>Projects</span><ChevronRight size={12} />
        <span className="cur ellipsis" style={{ maxWidth: 260 }}>{projectName ?? (sourcePath ? basename(sourcePath) : "Default")}{dirty && projectName ? " •" : ""}</span>
        {app?.demo && <span className="tag warn" title="Running in a browser without the processing engine">Demo</span>}
        {app && !app.ffmpeg && !app.ffmpeg_checking && <span className="tag err" title={app.ffmpeg_error ?? ""}>FFmpeg missing</span>}
      </div>
      <div className="top-center" role="group" aria-label="Player mode">
        {modes.map(({ value, label, icon: I }) => (
          <button key={value} className="tb icon" aria-label={label} title={label} aria-pressed={compare === value} onClick={() => useStore.setState({ compare: value })}><I size={15} /></button>
        ))}
      </div>
      <div className="top-right">
        <button className="tb" aria-pressed={presets} onClick={() => setPresets(!presets)}><Blend size={15} /> Presets</button>
        <span className="sep" />
        <button className="tb icon" aria-label="Models" title="Models" onClick={() => useStore.setState({ dialog: "models" })}><Boxes size={16} />{problems && <span className="dot-badge" />}</button>
        <button className="tb icon" aria-label="Preferences" title="Preferences (Ctrl+,)" onClick={() => useStore.setState({ dialog: "prefs" })}><Gear size={16} /></button>
        <span className="sep" />
        <button className="tb icon" aria-label={sideOpen ? "Hide controls" : "Show controls"} title={sideOpen ? "Hide controls" : "Show controls"} onClick={() => useStore.setState({ sideOpen: !sideOpen })}>
          {sideOpen ? <PanelRightClose size={16} /> : <PanelRightOpen size={16} />}
        </button>
      </div>
      {node}
    </header>
  );
}

function BottomBar() {
  const panel = useStore((s) => s.panel);
  const range = useStore((s) => s.range);
  const playhead = useStore((s) => s.playhead);
  const source = useStore((s) => s.source);
  const sources = useStore((s) => s.sources.length);
  const active = useStore((s) => s.jobs.filter((j) => j.kind === "render" && (RUNNING.includes(j.state) || j.state === "queued" || j.state === "waiting_for_resources")).length);
  const dur = source?.info.duration ?? 0;
  const v = source?.info.video;
  const step = v ? v.frame_rate.den / v.frame_rate.num : 1 / 30;
  const toggle = (p: "sources" | "queue") => useStore.setState({ panel: panel === p ? null : p });
  return (
    <footer className="bottombar">
      <button className="tb icon mono" aria-label="Set in point" title="Set in (I)" disabled={!source} onClick={() => useStore.setState({ range: { start: playhead, end: Math.max(playhead + step, range?.end ?? dur) } })}>[</button>
      <button className="tb icon mono" aria-label="Set out point" title="Set out (O)" disabled={!source} onClick={() => useStore.setState({ range: { start: Math.min(range?.start ?? 0, playhead - step), end: playhead } })}>]</button>
      <button className="tb icon mono" aria-label="Clear in and out" title="Clear in/out" disabled={!range} onClick={() => useStore.setState({ range: null })}>[×]</button>
      {range && <span className="faint mono" style={{ marginLeft: 6 }}>{timecode(range.start)} – {timecode(range.end)}</span>}
      <span className="spacer" />
      <button className="tb" aria-pressed={panel === "sources"} onClick={() => toggle("sources")}><FolderInput size={14} /> Sources{sources > 1 && <span className="count">{sources}</span>}</button>
      <button className="tb" aria-pressed={panel === "queue"} onClick={() => toggle("queue")}><Layers size={14} /> Export queue{active > 0 && <span className="count">{active}</span>}</button>
    </footer>
  );
}

function useShortcuts() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable);
      const mod = e.ctrlKey || e.metaKey;
      const s = useStore.getState();
      if (document.querySelector(".backdrop")) return;
      const k = e.key.toLowerCase();
      if (mod && k === "o") { e.preventDefault(); chooseVideos().catch(errorToast); return; }
      if (mod && k === "s") { e.preventDefault(); if (s.sourcePath) saveProject(e.shiftKey).catch(errorToast); return; }
      if (mod && k === "e") { e.preventDefault(); if (s.source) exportNow(!e.shiftKey); return; }
      if (mod && k === "r") { e.preventDefault(); if (s.source) (e.shiftKey ? renderClip() : renderStill()); return; }
      if (mod && k === "j") { e.preventDefault(); useStore.setState({ panel: s.panel === "queue" ? null : "queue" }); return; }
      if (mod && e.key === ",") { e.preventDefault(); useStore.setState({ dialog: "prefs" }); return; }
      if (typing) return;
      if (mod && k === "z") { e.preventDefault(); e.shiftKey ? redo() : undo(); return; }
      if (mod && k === "y") { e.preventDefault(); redo(); return; }
      if (!s.source) return;
      const v = s.source.info.video;
      const step = v ? v.frame_rate.den / v.frame_rate.num : 1 / 30;
      const dur = s.source.info.duration;
      if (e.key === " ") { e.preventDefault(); if (s.sourcePlay) stopSource(); else playSource(); return; }
      if (e.key === "ArrowLeft") { e.preventDefault(); useStore.setState({ playhead: Math.max(0, s.playhead - (e.shiftKey ? 1 : step)) }); }
      else if (e.key === "ArrowRight") { e.preventDefault(); useStore.setState({ playhead: Math.min(dur, s.playhead + (e.shiftKey ? 1 : step)) }); }
      else if (k === "i") useStore.setState({ range: { start: s.playhead, end: Math.max(s.playhead + step, s.range?.end ?? dur) } });
      else if (k === "o") useStore.setState({ range: { start: Math.min(s.range?.start ?? 0, s.playhead - step), end: s.playhead } });
      else if (k === "1") useStore.setState({ compare: "original" });
      else if (k === "2") useStore.setState({ compare: "processed" });
      else if (k === "3") useStore.setState({ compare: "split" });
      else if (k === "4") useStore.setState({ compare: "side" });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}

function useCloseGuard(setAsk: (v: boolean) => void) {
  useEffect(() => {
    if (!inTauri()) return;
    let un: (() => void) | undefined;
    import("@tauri-apps/api/window").then(({ getCurrentWindow }) =>
      getCurrentWindow().onCloseRequested((e) => {
        const s = useStore.getState();
        if (s.settings?.confirm_before_quit_with_jobs && s.jobs.some((j) => j.kind === "render" && RUNNING.includes(j.state))) {
          e.preventDefault();
          setAsk(true);
        }
      }).then((f) => (un = f)));
    return () => un?.();
  }, [setAsk]);
}

/** Open files passed by the OS (file associations, "Open with"). */
async function openFiles(paths: string[]) {
  const proj = paths.find((p) => p.toLowerCase().endsWith(".cfxproj"));
  if (proj) return openProjectFile(proj).catch(errorToast);
  for (const p of paths.filter((x) => x.toLowerCase().endsWith(".cfxpreset"))) {
    await api.presetAction("import", { path: p }).then(() => { refreshPresets(); toast("success", `Imported ${basename(p)}`); }).catch(errorToast);
  }
  const vids = paths.filter((p) => VIDEO_EXTENSIONS.includes(p.split(".").pop()?.toLowerCase() ?? ""));
  if (!vids.length) return;
  const s = useStore.getState();
  useStore.setState({ sources: [...s.sources, ...vids.filter((v) => !s.sources.includes(v))] });
  await openSource(vids[0]);
  if (vids.length > 1) useStore.setState({ panel: "sources" });
}

/** Offer the start dialog when a video is opened without adjustments. */
function useStartDialog() {
  const sourcePath = useStore((s) => s.sourcePath);
  useEffect(() => {
    const s = useStore.getState();
    if (sourcePath && !s.settings?.hide_start_dialog && !s.projectPath && !s.pipeline.stages.some((x) => x.enabled) && !s.dialog) useStore.setState({ dialog: "start" });
  }, [sourcePath]);
}

export function App() {
  const ready = useStore((s) => s.ready);
  const bootError = useStore((s) => s.bootError);
  const sideOpen = useStore((s) => s.sideOpen);
  const dialog = useStore((s) => s.dialog);
  const [presets, setPresets] = useState(false);
  const [askClose, setAskClose] = useState(false);

  useEffect(() => { boot(); }, []);
  useShortcuts();
  useCloseGuard(setAskClose);
  useStartDialog();
  useEffect(() => {
    let un: (() => void) | undefined;
    onFileDrop((paths) => {
      const known = paths.some((p) => /\.(cfxproj|cfxpreset)$/i.test(p) || VIDEO_EXTENSIONS.includes(p.split(".").pop()?.toLowerCase() ?? ""));
      if (!known) return toast("warning", "Unsupported file type");
      openFiles(paths);
    }).then((f) => (un = f));
    return () => un?.();
  }, []);
  useEffect(() => {
    if (ready && !bootError) api.startupFiles().then((f) => { if (f.length) openFiles(f); }).catch(() => {});
  }, [ready, bootError]);

  if (!ready) return <div className="empty-player" style={{ height: "100%" }}><img src="/logo.svg" alt="" width={36} /></div>;
  if (bootError) {
    return (
      <div className="empty-player" style={{ height: "100%" }}>
        <span className="err">ClipFX Enhance could not start</span>
        <span className="faint selectable">{bootError}</span>
        <button className="btn" onClick={() => location.reload()}>Retry</button>
      </div>
    );
  }
  return (
    <div className={`app${sideOpen ? "" : " no-side"}`}>
      <TopBar presets={presets} setPresets={setPresets} />
      <UpdateBanner />
      <Player />
      <Sidebar />
      <BottomBar />
      {presets && <PresetsPopover onClose={() => setPresets(false)} />}
      {dialog === "start" && <StartDialog />}
      {dialog === "models" && <ModelsDialog />}
      {dialog === "prefs" && <PreferencesDialog />}
      {askClose && (
        <Confirm title="Exports are running" confirmLabel="Pause and quit" body="They resume next time you open the app." onCancel={() => setAskClose(false)}
          onConfirm={() => import("@tauri-apps/api/window").then(({ getCurrentWindow }) => getCurrentWindow().destroy())} />
      )}
      <Toasts />
    </div>
  );
}
