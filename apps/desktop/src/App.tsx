import { useEffect, useState } from "react";
import { Boxes, Clapperboard, FolderOpen, ListVideo, Save, Settings as SettingsIcon, Sparkles } from "lucide-react";

import { Confirm, Modal, Notice, Toasts } from "./components/ui";
import { api, inTauri, onFileDrop, VIDEO_EXTENSIONS } from "./lib/ipc";
import { bytes } from "./lib/format";
import { cardId, cardName } from "./lib/ops";
import { boot, errorToast, openSource, redo, renderClip, renderStill, saveSettings, toast, undo, useStore, type View } from "./state/store";
import { ExportDialog } from "./views/ExportDialog";
import { ModelsView } from "./views/Models";
import { PresetsView } from "./views/Presets";
import { openProjectFile, saveProject } from "./views/Projects";
import { QueueView } from "./views/Queue";
import { SettingsView } from "./views/Settings";
import { chooseVideo, Workspace } from "./views/Workspace";

const NAV: { view: View; label: string; icon: typeof Clapperboard }[] = [
  { view: "workspace", label: "Workspace", icon: Clapperboard },
  { view: "queue", label: "Queue", icon: ListVideo },
  { view: "presets", label: "Presets", icon: Sparkles },
  { view: "models", label: "Model Library", icon: Boxes },
  { view: "settings", label: "Settings", icon: SettingsIcon },
];

const ACTIVE = ["preparing", "previewing", "processing", "encoding", "validating", "queued", "waiting_for_resources"];

function Nav() {
  const view = useStore((s) => s.view);
  const active = useStore((s) => s.jobs.filter((j) => j.kind === "render" && ACTIVE.includes(j.state)).length);
  const problems = useStore((s) => s.models.filter((m) => m.state === "missing" || m.state === "broken").length);
  return (
    <nav className="nav" aria-label="Main">
      {NAV.map(({ view: v, label, icon: I }) => (
        <button key={v} aria-label={label} title={label} aria-current={view === v ? "page" : undefined} onClick={() => useStore.setState({ view: v })}>
          <I size={22} />
          {v === "queue" && active > 0 && <span className="badge-dot">{active}</span>}
          {v === "models" && problems > 0 && <span className="badge-dot" style={{ background: "var(--err)" }}>!</span>}
        </button>
      ))}
    </nav>
  );
}

function TopBar() {
  const source = useStore((s) => s.sourcePath);
  const projectName = useStore((s) => s.projectName);
  const dirty = useStore((s) => s.dirty);
  const app = useStore((s) => s.app);
  return (
    <header className="topbar">
      <div className="brand"><img src="/logo.svg" alt="" /> ClearFrame</div>
      <span className="title ellipsis">{projectName ? `${projectName}${dirty ? " •" : ""}` : ""}</span>
      <span className="spacer" />
      {app && !app.ffmpeg && <span className="badge err" title={app.ffmpeg_error ?? ""}>FFmpeg missing</span>}
      <button className="btn ghost small" onClick={() => chooseVideo().catch(errorToast)}><FolderOpen size={15} /> Open video</button>
      <button className="btn ghost small" onClick={() => openProjectFile().catch(errorToast)}>Open project</button>
      <button className="btn ghost small" disabled={!source} onClick={() => saveProject(false).catch(errorToast)} title="Save project (Ctrl+S)"><Save size={15} /> Save</button>
    </header>
  );
}

function Onboarding() {
  const app = useStore((s) => s.app);
  const hw = useStore((s) => s.hardware);
  const models = useStore((s) => s.models);
  const recommended = models.filter((m) => m.manifest?.recommended && m.state !== "installed");
  const [picked, setPicked] = useState<string[]>([]);
  const finish = async () => {
    for (const id of picked) {
      try {
        await api.modelInstall(id);
      } catch (e) {
        errorToast(e);
      }
    }
    if (picked.length) toast("info", `Installing ${picked.length} model(s). Progress is shown in the Model Library.`);
    await saveSettings((s) => ({ ...s, onboarding_complete: true }));
  };
  const size = picked.reduce((a, id) => a + (models.find((m) => cardId(m) === id)?.manifest?.download_size_bytes ?? 0), 0);
  return (
    <Modal title="Welcome to ClearFrame" onClose={finish} wide footer={
      <>
        <span className="small faint grow">{picked.length ? `${picked.length} model(s), ${bytes(size)} download` : "You can install models any time from the Model Library."}</span>
        <button className="btn primary" onClick={finish}>{picked.length ? "Install and start" : "Start"}</button>
      </>
    }>
      <p className="muted">ClearFrame enhances videos on your own computer. Nothing is uploaded, there is no account, and after models are installed it works offline.</p>
      <div className="col mt">
        {app?.ffmpeg ? (
          <Notice tone="ok">FFmpeg {app.ffmpeg.version} found{app.ffmpeg.vidstab ? "" : " (without vid.stab: stabilization unavailable)"}.</Notice>
        ) : (
          <Notice tone="err">FFmpeg was not found: {app?.ffmpeg_error}. Install FFmpeg or choose its folder in Settings → Advanced. Video processing needs it.</Notice>
        )}
        {hw && (
          <Notice tone="info">
            {hw.hardware.cpu_brand} · {hw.hardware.logical_cores} threads · {Math.round(hw.hardware.ram_total_mb / 1024)} GB RAM
            {hw.hardware.backends.some((b) => b.kind !== "cpu" && b.available) ? " · GPU acceleration available" : " · processing runs on the CPU (slower, same quality)"}
          </Notice>
        )}
      </div>
      {recommended.length > 0 && (
        <>
          <h3 className="mt-l">Choose models to install now (optional)</h3>
          <p className="small faint">Nothing is downloaded unless you tick it. Each model shows its license; some do not allow commercial use.</p>
          <div className="col mt">
            {recommended.map((m) => (
              <label key={cardId(m)} className="check">
                <input type="checkbox" checked={picked.includes(cardId(m))} onChange={(e) => setPicked(e.target.checked ? [...picked, cardId(m)] : picked.filter((x) => x !== cardId(m)))} />
                <span>
                  {cardName(m)} <span className="faint small">· {bytes(m.manifest!.download_size_bytes)} · {m.manifest!.weights_license}</span>
                  <div className="desc">{m.manifest!.summary}</div>
                </span>
              </label>
            ))}
          </div>
        </>
      )}
    </Modal>
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
      if (mod && k === "o") { e.preventDefault(); chooseVideo().catch(errorToast); return; }
      if (mod && k === "s") { e.preventDefault(); if (s.sourcePath) saveProject(e.shiftKey).catch(errorToast); return; }
      if (mod && k === "e") { e.preventDefault(); if (s.source) useStore.setState({ exportOpen: true }); return; }
      if (mod && k === "p") { e.preventDefault(); if (s.source) (e.shiftKey ? renderClip() : renderStill()); return; }
      if (mod && k === "j") { e.preventDefault(); useStore.setState({ view: s.view === "queue" ? "workspace" : "queue" }); return; }
      if (mod && e.key === ",") { e.preventDefault(); useStore.setState({ view: "settings" }); return; }
      if (typing) return;
      if (mod && k === "z") { e.preventDefault(); e.shiftKey ? redo() : undo(); return; }
      if (mod && k === "y") { e.preventDefault(); redo(); return; }
      if (s.view !== "workspace" || !s.source) return;
      const v = s.source.info.video;
      const step = v ? v.frame_rate.den / v.frame_rate.num : 1 / 30;
      const dur = s.source.info.duration;
      if (e.key === "ArrowLeft") { e.preventDefault(); useStore.setState({ playhead: Math.max(0, s.playhead - (e.shiftKey ? 1 : step)) }); }
      else if (e.key === "ArrowRight") { e.preventDefault(); useStore.setState({ playhead: Math.min(dur, s.playhead + (e.shiftKey ? 1 : step)) }); }
      else if (k === "i") useStore.setState({ range: { start: s.playhead, end: Math.max(s.playhead + step, s.range?.end ?? dur) } });
      else if (k === "o") useStore.setState({ range: { start: Math.min(s.range?.start ?? 0, s.playhead - step), end: s.playhead } });
      else if (k === "c") useStore.setState({ compare: s.compare === "split" ? "side" : s.compare === "side" ? "original" : s.compare === "original" ? "processed" : "split" });
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
        const running = s.jobs.some((j) => j.kind === "render" && ["preparing", "processing", "encoding", "validating"].includes(j.state));
        if (running && s.settings?.confirm_before_quit_with_jobs) {
          e.preventDefault();
          setAsk(true);
        }
      }).then((f) => (un = f)),
    );
    return () => un?.();
  }, [setAsk]);
}

export function App() {
  const ready = useStore((s) => s.ready);
  const bootError = useStore((s) => s.bootError);
  const view = useStore((s) => s.view);
  const app = useStore((s) => s.app);
  const settings = useStore((s) => s.settings);
  const exportOpen = useStore((s) => s.exportOpen);
  const [askClose, setAskClose] = useState(false);

  useEffect(() => { boot(); }, []);
  useShortcuts();
  useCloseGuard(setAskClose);
  useEffect(() => {
    let un: (() => void) | undefined;
    onFileDrop(async (paths) => {
      const proj = paths.find((p) => p.toLowerCase().endsWith(".cfproj"));
      if (proj) return openProjectFile(proj).catch(errorToast);
      const vids = paths.filter((p) => VIDEO_EXTENSIONS.includes(p.split(".").pop()?.toLowerCase() ?? ""));
      if (vids.length === 1) openSource(vids[0]);
      else if (vids.length > 1) {
        await openSource(vids[0]);
        toast("info", `${vids.length} videos dropped. The first is open; use Export → “Add more files” to process the rest with the same settings.`);
      } else toast("warning", "That file type is not supported.");
    }).then((f) => (un = f));
    return () => un?.();
  }, []);

  if (!ready) return <div className="empty" style={{ height: "100%" }}>Starting ClearFrame…</div>;
  if (bootError) {
    return (
      <div className="empty" style={{ height: "100%" }}>
        <h2>ClearFrame could not start</h2>
        <p className="selectable">{bootError}</p>
        <button className="btn" onClick={() => location.reload()}>Try again</button>
      </div>
    );
  }
  return (
    <div className={`app${app?.demo ? " with-banner" : ""}`}>
      <TopBar />
      {app?.demo && <div className="demo-banner" role="status">Demo mode: running in a browser without the ClearFrame engine. Nothing is processed or downloaded.</div>}
      <Nav />
      <main className="main">
        {view === "workspace" && <Workspace />}
        {view === "queue" && <QueueView />}
        {view === "presets" && <PresetsView />}
        {view === "models" && <ModelsView />}
        {view === "settings" && <SettingsView />}
      </main>
      {exportOpen && <ExportDialog />}
      {settings && !settings.onboarding_complete && <Onboarding />}
      {askClose && (
        <Confirm title="Exports are running" confirmLabel="Pause and quit" onCancel={() => setAskClose(false)}
          body="Running exports will be paused safely and resume from where they stopped next time you open ClearFrame."
          onConfirm={() => import("@tauri-apps/api/window").then(({ getCurrentWindow }) => getCurrentWindow().destroy())} />
      )}
      <Toasts />
    </div>
  );
}
