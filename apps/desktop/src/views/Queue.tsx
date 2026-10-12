import { useMemo, useState } from "react";
import { ArrowDown, ArrowUp, FolderOpen, Pause, Play, RotateCcw, ScrollText, Square, Trash2 } from "lucide-react";

import { Confirm, Notice, Progress } from "../components/ui";
import { api } from "../lib/ipc";
import { basename, duration, eta, mb, relTime } from "../lib/format";
import type { JobEvent, JobRecord, JobState } from "../lib/types";
import { errorToast, refreshJobs, toast, useStore } from "../state/store";

const ACTIVE: JobState[] = ["preparing", "previewing", "processing", "encoding", "validating"];
const LABEL: Record<JobState, string> = {
  queued: "Queued", waiting_for_resources: "Waiting for memory", preparing: "Preparing", previewing: "Previewing", processing: "Processing",
  encoding: "Encoding", validating: "Checking output", completed: "Completed", paused: "Paused", cancelled: "Cancelled", failed: "Failed",
  recoverable: "Interrupted — can resume",
};

function tone(s: JobState) {
  return s === "completed" ? "ok" : s === "failed" ? "err" : s === "paused" || s === "recoverable" || s === "waiting_for_resources" ? "warn" : ACTIVE.includes(s) ? "accent" : "";
}

function JobItem({ j, index, queuedCount }: { j: JobRecord; index: number; queuedCount: number }) {
  const progress = useStore((s) => s.progress[j.id]);
  const res = useStore((s) => s.resources[j.id]);
  const [log, setLog] = useState<JobEvent[] | null>(null);
  const [confirm, setConfirm] = useState<"cancel" | null>(null);
  const act = (a: Parameters<typeof api.jobAction>[1], idx?: number) => api.jobAction(j.id, a, idx).then(refreshJobs).catch(errorToast);
  const active = ACTIVE.includes(j.state);
  const frac = progress?.fraction ?? j.progress;
  const report = useMemo(() => {
    try {
      return j.report_json ? JSON.parse(j.report_json) : null;
    } catch {
      return null;
    }
  }, [j.report_json]);
  const showLog = async () => {
    if (log) return setLog(null);
    try {
      setLog(await api.jobEvents(j.id));
    } catch (e) {
      errorToast(e);
    }
  };
  return (
    <div className="job" data-testid={`job-${j.id}`}>
      <div className="col" style={{ gap: 6, minWidth: 0 }}>
        <div className="row" style={{ minWidth: 0 }}>
          <strong className="ellipsis" title={j.destination}>{basename(j.destination) || j.name}</strong>
          <span className={`badge ${tone(j.state)}`}>{LABEL[j.state]}</span>
          {j.kind === "preview" && <span className="badge">preview</span>}
        </div>
        <div className="meta">
          <span title={j.source}>from {basename(j.source)}</span>
          <span>{relTime(j.created_at)}</span>
          {j.attempts > 1 && <span>attempt {j.attempts}</span>}
        </div>
        {(active || j.state === "paused" || j.state === "recoverable" || j.state === "queued") && (
          <>
            <Progress value={frac} indeterminate={active && !progress} label={`Progress of ${j.name}`} />
            <div className="meta">
              <span>{Math.round(frac * 100)}%</span>
              {progress && <span>pass {progress.pass}/{progress.passes} · {progress.phase}</span>}
              {progress && progress.frames_total > 0 && <span>{progress.frames_done}/{progress.frames_total} frames</span>}
              {active && progress && progress.fps > 0 && <span>{progress.fps.toFixed(1)} fps</span>}
              {active && <span>{eta(progress?.eta)}</span>}
              {active && res && <span title="Memory used by this job's worker process">{mb(res.process_memory_mb)} RAM · {res.backend.toUpperCase()}</span>}
            </div>
          </>
        )}
        {j.detail && !active && j.state !== "completed" && <div className="small muted">{j.detail}</div>}
        {j.error && <Notice tone="err"><span className="selectable">{j.error}</span></Notice>}
        {j.warnings.length > 0 && j.warnings.map((w, i) => <Notice key={i} tone="warn">{w}</Notice>)}
        {report && j.state === "completed" && (
          <div className="meta">
            {report.timings?.total_s > 0 && <span>took {duration(report.timings.total_s)}</span>}
            {report.output_width > 0 && <span>{report.output_width}×{report.output_height} · {report.output_fps} fps</span>}
            {report.frames_written > 0 && <span>{report.frames_written} frames</span>}
            {report.backend && <span>{String(report.backend).toUpperCase()}</span>}
            {report.validation && <span className={report.validation.ok ? "ok" : "err"}>{report.validation.ok ? `output verified${report.validation.decoded_frames ? ` (${report.validation.decoded_frames} frames decoded)` : ""}` : `verification problems: ${report.validation.problems.join("; ")}`}</span>}
          </div>
        )}
      </div>
      <div className="actions">
        {j.state === "queued" && queuedCount > 1 && (
          <>
            <button className="btn ghost icon small" aria-label="Move up" disabled={index === 0} onClick={() => act("reorder", index - 1)}><ArrowUp size={14} /></button>
            <button className="btn ghost icon small" aria-label="Move down" disabled={index === queuedCount - 1} onClick={() => act("reorder", index + 1)}><ArrowDown size={14} /></button>
          </>
        )}
        {(active || j.state === "queued" || j.state === "waiting_for_resources") && (
          <button className="btn small" onClick={() => act("pause")}><Pause size={14} /> Pause</button>
        )}
        {(j.state === "paused" || j.state === "recoverable") && <button className="btn small" onClick={() => act("resume")}><Play size={14} /> Resume</button>}
        {(j.state === "failed" || j.state === "cancelled") && <button className="btn small" onClick={() => act("retry")}><RotateCcw size={14} /> Retry</button>}
        {!["completed", "failed", "cancelled"].includes(j.state) && (
          <button className="btn small danger" onClick={() => setConfirm("cancel")}><Square size={13} /> Cancel</button>
        )}
        {j.state === "completed" && <button className="btn small" onClick={() => api.openPath(j.destination).catch(errorToast)}><FolderOpen size={14} /> Show file</button>}
        <button className="btn ghost icon small" aria-label="Show log" title="Log" onClick={showLog}><ScrollText size={14} /></button>
        {["completed", "failed", "cancelled"].includes(j.state) && (
          <button className="btn ghost icon small" aria-label="Remove from list" title="Remove from list (the exported file is kept)" onClick={() => act("remove")}><Trash2 size={14} /></button>
        )}
      </div>
      {log && (
        <div className="log">
          {log.length === 0 ? <div>No log entries.</div> : log.map((e, i) => <div key={i} className={e.level === "error" ? "err" : e.level === "warning" ? "warn" : ""}>{e.at.slice(11, 19)} {e.message}</div>)}
        </div>
      )}
      {confirm && (
        <Confirm title="Cancel this export?" danger confirmLabel="Cancel export" onCancel={() => setConfirm(null)}
          body="Work done so far is discarded. Your source video is not affected. Use Pause instead if you want to continue later."
          onConfirm={() => { setConfirm(null); act("cancel"); }} />
      )}
    </div>
  );
}

export function QueueView() {
  const jobs = useStore((s) => s.jobs);
  const paused = useStore((s) => s.queuePaused);
  const [filter, setFilter] = useState<"all" | "active" | "done" | "problems">("all");
  const [showPreviews, setShowPreviews] = useState(false);
  const list = jobs
    .filter((j) => showPreviews || j.kind === "render")
    .filter((j) => filter === "all" || (filter === "active" ? !["completed", "cancelled", "failed"].includes(j.state) : filter === "done" ? j.state === "completed" : ["failed", "recoverable", "cancelled"].includes(j.state)))
    .sort((a, b) => {
      const rank = (j: JobRecord) => (ACTIVE.includes(j.state) ? 0 : j.state === "queued" || j.state === "waiting_for_resources" ? 1 : j.state === "paused" || j.state === "recoverable" ? 2 : 3);
      return rank(a) - rank(b) || (rank(a) === 1 ? a.queue_order - b.queue_order : b.created_at.localeCompare(a.created_at));
    });
  const queued = list.filter((j) => j.state === "queued");
  const recoverable = jobs.filter((j) => j.state === "recoverable");
  return (
    <div className="page">
      <div className="page-head">
        <div className="grow">
          <h1>Queue</h1>
          <p className="muted small">Exports run in a separate process. If ClearFrame closes or crashes, unfinished jobs resume from the last completed segment.</p>
        </div>
        {paused ? (
          <button className="btn primary" onClick={() => api.queueAction("resume").then(() => useStore.setState({ queuePaused: false })).catch(errorToast)}><Play size={15} /> Resume queue</button>
        ) : (
          <button className="btn" onClick={() => api.queueAction("pause").then(() => useStore.setState({ queuePaused: true })).catch(errorToast)}><Pause size={15} /> Pause queue</button>
        )}
        <button className="btn" onClick={() => api.queueAction("clear_finished").then((n) => { toast("info", `Removed ${n} finished item(s) from the list. Exported files are kept.`); refreshJobs(); }).catch(errorToast)}>Clear finished</button>
      </div>
      {paused && <Notice tone="warn">The queue is paused. Running exports finish their current segment and stop; nothing new starts.</Notice>}
      {recoverable.length > 0 && <Notice tone="info">{recoverable.length} export(s) were interrupted and can be resumed.</Notice>}
      <div className="filters mt" role="group" aria-label="Filter">
        {(["all", "active", "done", "problems"] as const).map((f) => (
          <button key={f} className="chip" aria-pressed={filter === f} onClick={() => setFilter(f)}>{f === "all" ? "All" : f === "active" ? "In progress" : f === "done" ? "Completed" : "Needs attention"}</button>
        ))}
        <span className="spacer" />
        <label className="row small muted"><input type="checkbox" checked={showPreviews} onChange={(e) => setShowPreviews(e.target.checked)} /> Show preview renders</label>
      </div>
      {list.length === 0 ? (
        <div className="empty">
          <p>No exports {filter === "all" ? "yet" : "here"}.</p>
          {filter === "all" && <button className="btn" onClick={() => useStore.setState({ view: "workspace" })}>Open a video</button>}
        </div>
      ) : (
        <div className="list">{list.map((j) => <JobItem key={j.id} j={j} index={queued.indexOf(j)} queuedCount={queued.length} />)}</div>
      )}
    </div>
  );
}
