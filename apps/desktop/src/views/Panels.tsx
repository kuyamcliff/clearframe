import { useEffect, useMemo, useState, type PointerEvent as RPointerEvent, type ReactNode } from "react";
import { FileVideo, FolderOpen, Layers, MoreHorizontal, Pause, Play, Plus, RotateCcw, ScrollText, Square, Trash2, X } from "lucide-react";

import { Confirm, Progress, useMenu } from "../components/ui";
import { api, fileUrl, pickFile, VIDEO_EXTENSIONS } from "../lib/ipc";
import { basename, duration, eta, mb } from "../lib/format";
import type { JobEvent, JobRecord, JobState } from "../lib/types";
import { currentJob, errorToast, openSource, refreshJobs, toast, useStore } from "../state/store";
import { chooseVideos } from "./Player";

const ACTIVE: JobState[] = ["preparing", "previewing", "processing", "encoding", "validating"];
const LABEL: Record<JobState, string> = {
  queued: "Queued", waiting_for_resources: "Waiting for memory", preparing: "Preparing", previewing: "Previewing", processing: "Processing",
  encoding: "Encoding", validating: "Verifying", completed: "Done", paused: "Paused", cancelled: "Cancelled", failed: "Failed", recoverable: "Interrupted",
};
const TONE: Partial<Record<JobState, string>> = { completed: "ok", failed: "err", paused: "warn", recoverable: "warn", waiting_for_resources: "warn", processing: "accent", encoding: "accent", preparing: "accent", validating: "accent" };

function Sheet({ title, icon, actions, children, testId }: { title: string; icon: ReactNode; actions?: ReactNode; children: ReactNode; testId: string }) {
  const [h, setH] = useState(46);
  const grip = (e: RPointerEvent) => {
    const parent = ((e.currentTarget as HTMLElement).closest(".player") as HTMLElement).getBoundingClientRect();
    setH(Math.max(22, Math.min(92, ((parent.bottom - e.clientY) / parent.height) * 100)));
  };
  return (
    <div className="sheet" style={{ ["--h" as string]: `${h}%` }} data-testid={testId} role="region" aria-label={title}>
      <div className="sheet-grip" onPointerDown={(e) => (e.target as Element).setPointerCapture?.(e.pointerId)} onPointerMove={(e) => e.buttons && grip(e)} />
      <div className="sheet-h">
        {icon}<span className="grow">{title}</span>{actions}
        <button className="btn ghost sm icon" aria-label="Close" onClick={() => useStore.setState({ panel: null })}><X size={15} /></button>
      </div>
      <div className="sheet-b">{children}</div>
    </div>
  );
}

function SourceThumb({ path }: { path: string }) {
  const current = useStore((s) => s.sourcePath);
  const [img, setImg] = useState<string | null>(null);
  const { open, node } = useMenu();
  useEffect(() => { api.sourceFrame(path, 1, 320).then(setImg).catch(() => setImg("")); }, [path]);
  const remove = () => {
    const s = useStore.getState();
    useStore.setState({ sources: s.sources.filter((x) => x !== path) });
  };
  return (
    <div className={`thumb${current === path ? " sel" : ""}`} onClick={() => current !== path && openSource(path)} role="button" tabIndex={0} title={path}
      onKeyDown={(e) => e.key === "Enter" && openSource(path)}>
      <div className="img">{img ? <img src={fileUrl(img)} alt="" /> : <FileVideo size={20} />}</div>
      <div className="cap">
        <span className="grow ellipsis">{basename(path)}</span>
        <button className="btn ghost sm icon" aria-label={`More for ${basename(path)}`} onClick={(e) => open(e, [
          { label: "Open", onClick: () => openSource(path) },
          { label: "Show in folder", onClick: () => api.openPath(path).catch(errorToast) },
          { label: "Remove from list", onClick: remove, sep: true },
        ])}><MoreHorizontal size={13} /></button>
      </div>
      {node}
    </div>
  );
}

function SourcesPanel() {
  const sources = useStore((s) => s.sources);
  const exportAll = async () => {
    const t = currentJob();
    if (!t) return;
    try {
      const r = await api.enqueueBatch(sources, { ...t, file_name: null, range: null }, false);
      toast(r.failed.length ? "warning" : "success", `Queued ${r.created.length} of ${sources.length}${r.failed.length ? ` — ${r.failed.map((f) => `${basename(f.path)}: ${f.error}`).join("; ")}` : ""}`);
      refreshJobs();
    } catch (e) { errorToast(e); }
  };
  return (
    <Sheet title="Sources" icon={<Layers size={14} className="faint" />} testId="sources" actions={
      sources.length > 1 && <button className="btn sm" onClick={exportAll}>Export all with current settings</button>
    }>
      <div className="thumbs">
        <button className="thumb add" aria-label="Add videos" onClick={() => chooseVideos().catch(errorToast)}><Plus size={22} /></button>
        {sources.map((p) => <SourceThumb key={p} path={p} />)}
      </div>
    </Sheet>
  );
}

function JobRow({ j }: { j: JobRecord }) {
  const progress = useStore((s) => s.progress[j.id]);
  const res = useStore((s) => s.resources[j.id]);
  const [log, setLog] = useState<JobEvent[] | null>(null);
  const [ask, setAsk] = useState(false);
  const act = (a: Parameters<typeof api.jobAction>[1], idx?: number) => api.jobAction(j.id, a, idx).then(refreshJobs).catch(errorToast);
  const active = ACTIVE.includes(j.state);
  const frac = j.state === "completed" ? 1 : progress?.fraction ?? j.progress;
  const report = useMemo(() => { try { return j.report_json ? JSON.parse(j.report_json) : null; } catch { return null; } }, [j.report_json]);
  const toggleLog = async () => { if (log) return setLog(null); try { setLog(await api.jobEvents(j.id)); } catch (e) { errorToast(e); } };
  const finished = ["completed", "failed", "cancelled"].includes(j.state);
  return (
    <>
      <tr data-testid={`job-${j.id}`}>
        <td style={{ maxWidth: 320 }}>
          <div className="ellipsis" title={j.destination}>{basename(j.destination) || j.name}</div>
          <div className="faint ellipsis" style={{ fontSize: 11 }} title={j.source}>{basename(j.source)}</div>
        </td>
        <td><span className={`tag ${TONE[j.state] ?? ""}`}>{LABEL[j.state]}</span></td>
        <td className="pc">
          <Progress value={frac} tone={j.state === "completed" ? "ok" : j.state === "failed" ? "err" : undefined} indeterminate={active && !progress} label={`Progress of ${j.name}`} />
          <div className="faint" style={{ fontSize: 11, marginTop: 3 }}>
            {Math.round(frac * 100)}%{progress && active ? ` · pass ${progress.pass}/${progress.passes}` : ""}
          </div>
        </td>
        <td className="muted">{active && progress && progress.fps > 0 ? `${progress.fps.toFixed(1)} fps` : report?.timings?.total_s ? duration(report.timings.total_s) : ""}{active && res ? ` · ${mb(res.process_memory_mb)} RAM` : ""}</td>
        <td className="muted">{active ? eta(progress?.eta) : j.state === "completed" && report?.validation?.ok ? "Verified" : ""}</td>
        <td className="act">
          {(active || j.state === "queued" || j.state === "waiting_for_resources") && <button className="btn ghost sm icon" aria-label="Pause" title="Pause" onClick={() => act("pause")}><Pause size={13} /></button>}
          {(j.state === "paused" || j.state === "recoverable") && <button className="btn ghost sm icon" aria-label="Resume" title="Resume" onClick={() => act("resume")}><Play size={13} /></button>}
          {(j.state === "failed" || j.state === "cancelled") && <button className="btn ghost sm icon" aria-label="Retry" title="Retry" onClick={() => act("retry")}><RotateCcw size={13} /></button>}
          {!finished && <button className="btn ghost sm icon" aria-label="Cancel" title="Cancel" onClick={() => setAsk(true)}><Square size={12} /></button>}
          {j.state === "completed" && <button className="btn ghost sm icon" aria-label="Show file" title="Show in folder" onClick={() => api.openPath(j.destination).catch(errorToast)}><FolderOpen size={13} /></button>}
          <button className="btn ghost sm icon" aria-label="Log" title="Log" onClick={toggleLog}><ScrollText size={13} /></button>
          {finished && <button className="btn ghost sm icon" aria-label="Remove" title="Remove from list" onClick={() => act("remove")}><Trash2 size={13} /></button>}
          {ask && <Confirm title="Cancel this export?" danger confirmLabel="Cancel export" onCancel={() => setAsk(false)} onConfirm={() => { setAsk(false); act("cancel"); }} />}
        </td>
      </tr>
      {(j.error || j.warnings.length > 0 || log) && (
        <tr>
          <td colSpan={6} style={{ paddingTop: 0 }}>
            {j.error && <div className="err selectable" style={{ fontSize: 11.5 }}>{j.error}</div>}
            {j.warnings.map((w, i) => <div key={i} className="warn" style={{ fontSize: 11.5 }}>{w}</div>)}
            {log && <div className="log">{log.length ? log.map((e, i) => <div key={i}>{e.at.slice(11, 19)}  {e.message}</div>) : "No entries."}</div>}
          </td>
        </tr>
      )}
    </>
  );
}

function QueuePanel() {
  const jobs = useStore((s) => s.jobs);
  const paused = useStore((s) => s.queuePaused);
  const list = jobs.filter((j) => j.kind === "render").sort((a, b) => {
    const rank = (j: JobRecord) => (ACTIVE.includes(j.state) ? 0 : j.state === "queued" || j.state === "waiting_for_resources" ? 1 : j.state === "paused" || j.state === "recoverable" ? 2 : 3);
    return rank(a) - rank(b) || (rank(a) === 1 ? a.queue_order - b.queue_order : b.created_at.localeCompare(a.created_at));
  });
  const addVideos = async () => {
    const t = currentJob();
    if (!t) return toast("info", "Open a video and choose settings first.");
    const files = await pickFile("Add videos with the current settings", [{ name: "Video", extensions: VIDEO_EXTENSIONS }], true);
    if (!files.length) return;
    try {
      const r = await api.enqueueBatch(files, { ...t, file_name: null, range: null }, false);
      toast(r.failed.length ? "warning" : "success", `Queued ${r.created.length} of ${files.length}`);
      refreshJobs();
    } catch (e) { errorToast(e); }
  };
  return (
    <Sheet title="Export queue" icon={<Layers size={14} className="faint" />} testId="queue" actions={<>
      <button className="btn ghost sm" onClick={addVideos}><Plus size={13} /> Add videos</button>
      <button className="btn ghost sm" onClick={() => api.queueAction("clear_finished").then(refreshJobs).catch(errorToast)}>Clear finished</button>
      <button className="btn sm" onClick={() => api.queueAction(paused ? "resume" : "pause").then(() => useStore.setState({ queuePaused: !paused })).catch(errorToast)}>
        {paused ? <><Play size={13} /> Resume queue</> : <><Pause size={13} /> Pause queue</>}
      </button>
    </>}>
      {list.length === 0 ? <div className="faint" style={{ padding: "24px 0", textAlign: "center" }}>No exports</div> : (
        <table className="q">
          <thead><tr><th>Output</th><th>Status</th><th>Progress</th><th>Speed</th><th>Remaining</th><th /></tr></thead>
          <tbody>{list.map((j) => <JobRow key={j.id} j={j} />)}</tbody>
        </table>
      )}
    </Sheet>
  );
}

export function Panels() {
  const panel = useStore((s) => s.panel);
  return panel === "sources" ? <SourcesPanel /> : panel === "queue" ? <QueuePanel /> : null;
}
