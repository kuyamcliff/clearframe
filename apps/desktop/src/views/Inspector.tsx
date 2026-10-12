import { useEffect, useState } from "react";
import { ArrowDown, ArrowUp, ChevronDown, ChevronRight, Film, Image as ImageIcon, Plus, Redo2, Save, Trash2, Undo2, Upload } from "lucide-react";

import { StageForm } from "../components/StageForm";
import { Notice, Seg, Switch } from "../components/ui";
import { api } from "../lib/ipc";
import { fps, mb } from "../lib/format";
import { addStageCanonical, moveStage, OPS, opInfo, removeStage, stageSummary, updateStage, newStageId } from "../lib/ops";
import type { ExportCheck, OperationType } from "../lib/types";
import { currentJob, redo, renderClip, renderStill, saveSettings, setPipeline, undo, useStore } from "../state/store";
import { SavePresetDialog } from "./Presets";

export function usePlanCheck(): { check: ExportCheck | null; error: string | null; busy: boolean } {
  const pipeline = useStore((s) => s.pipeline);
  const output = useStore((s) => s.output);
  const sourcePath = useStore((s) => s.sourcePath);
  const source = useStore((s) => s.source);
  const range = useStore((s) => s.range);
  const models = useStore((s) => s.models);
  const folder = useStore((s) => s.exportFolder);
  const name = useStore((s) => s.exportFileName);
  const [state, setState] = useState<{ check: ExportCheck | null; error: string | null; busy: boolean }>({ check: null, error: null, busy: false });
  useEffect(() => {
    if (!sourcePath || !source || !output) {
      setState({ check: null, error: null, busy: false });
      return;
    }
    let live = true;
    setState((s) => ({ ...s, busy: true }));
    const t = window.setTimeout(async () => {
      const job = currentJob();
      if (!job) return;
      try {
        const check = await api.checkExport(job);
        if (live) setState({ check, error: null, busy: false });
      } catch (e) {
        if (live) setState({ check: null, error: e instanceof Error ? e.message : String(e), busy: false });
      }
    }, 300);
    return () => {
      live = false;
      window.clearTimeout(t);
    };
  }, [pipeline, output, sourcePath, source, range, models, folder, name]);
  return state;
}

function AddStageMenu({ onAdd }: { onAdd: (t: OperationType) => void }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ position: "relative" }}>
      <button className="btn small" onClick={() => setOpen(!open)} aria-expanded={open}>
        <Plus size={14} /> Add stage
      </button>
      {open && (
        <div className="card" style={{ position: "absolute", zIndex: 20, right: 0, top: 30, width: 240, padding: 6, boxShadow: "var(--shadow)" }} role="menu">
          {OPS.map((o) => (
            <button key={o.type} role="menuitem" className="btn ghost" style={{ width: "100%", justifyContent: "flex-start" }} onClick={() => { onAdd(o.type); setOpen(false); }}>
              {o.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function SimpleTasks() {
  const pipeline = useStore((s) => s.pipeline);
  const defaults = useStore((s) => s.defaults);
  const models = useStore((s) => s.models);
  const [open, setOpen] = useState<string | null>(null);
  return (
    <div>
      {OPS.map((o) => {
        const stages = pipeline.stages.filter((s) => s.type === o.type);
        const stage = stages[0];
        const on = !!stage && stage.enabled;
        const expanded = open === o.type;
        return (
          <div key={o.type} className={`task${on ? " on" : ""}`} data-testid={`task-${o.type}`}>
            <div className="task-head" onClick={() => setOpen(expanded ? null : o.type)}>
              {expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
              <div className="grow">
                <div className="name">{o.task}</div>
                <div className="sum ellipsis">{stage ? (stage.enabled ? stageSummary(stage, models) : "off") : o.builtin ? "built-in" : "AI model"}</div>
              </div>
              <span onClick={(e) => e.stopPropagation()}>
                <Switch label={o.task} checked={on} onChange={(v) => {
                  if (stage) setPipeline(updateStage(pipeline, stage.id, (s) => ({ ...s, enabled: v })));
                  else if (v && defaults) {
                    setPipeline(addStageCanonical(pipeline, o.type, defaults[o.type].settings));
                    setOpen(o.type);
                  }
                }} />
              </span>
            </div>
            {expanded && (
              <div className="task-body">
                <p className="desc">{o.description}</p>
                {stage ? (
                  <>
                    <StageForm stage={stage} advanced={false} onChange={(settings) => setPipeline(updateStage(pipeline, stage.id, (s) => ({ ...s, settings })))} />
                    {stages.length > 1 && <div className="hint small faint">This pipeline has {stages.length} {o.label} stages; edit the others in Advanced.</div>}
                  </>
                ) : (
                  <button className="btn small" onClick={() => defaults && setPipeline(addStageCanonical(pipeline, o.type, defaults[o.type].settings))}>
                    <Plus size={14} /> Turn on
                  </button>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function PipelineEditor() {
  const pipeline = useStore((s) => s.pipeline);
  const defaults = useStore((s) => s.defaults);
  const models = useStore((s) => s.models);
  const [sel, setSel] = useState<string | null>(pipeline.stages[0]?.id ?? null);
  const selected = pipeline.stages.find((s) => s.id === sel) ?? null;
  return (
    <div>
      <div className="row" style={{ marginBottom: 8 }}>
        <span className="small muted grow">Stages run top to bottom.</span>
        <AddStageMenu onAdd={(t) => {
          if (!defaults) return;
          const id = newStageId(t, pipeline.stages);
          setPipeline({ stages: [...pipeline.stages, { id, enabled: true, type: t, settings: structuredClone(defaults[t].settings) }] });
          setSel(id);
        }} />
      </div>
      {pipeline.stages.length === 0 && <div className="empty small">No stages yet. Add one, or apply a preset.</div>}
      {pipeline.stages.map((s, i) => (
        <div key={s.id} className={`stage-row${s.enabled ? "" : " off"}${sel === s.id ? " sel" : ""}`} data-testid={`stage-${s.id}`}>
          <span className="idx">{i + 1}</span>
          <button className="btn ghost grow" style={{ justifyContent: "flex-start", height: "auto", padding: "2px 4px", flexDirection: "column", alignItems: "flex-start" }} onClick={() => setSel(s.id)}>
            <span style={{ fontWeight: 600 }}>{opInfo(s.type).label}</span>
            <span className="tiny faint ellipsis" style={{ maxWidth: 170 }}>{stageSummary(s, models)}</span>
          </button>
          <Switch label={`Enable ${opInfo(s.type).label}`} checked={s.enabled} onChange={(v) => setPipeline(updateStage(pipeline, s.id, (x) => ({ ...x, enabled: v })))} />
          <button className="btn ghost icon small" aria-label="Move up" disabled={i === 0} onClick={() => setPipeline(moveStage(pipeline, s.id, -1))}><ArrowUp size={14} /></button>
          <button className="btn ghost icon small" aria-label="Move down" disabled={i === pipeline.stages.length - 1} onClick={() => setPipeline(moveStage(pipeline, s.id, 1))}><ArrowDown size={14} /></button>
          <button className="btn ghost icon small" aria-label="Remove stage" onClick={() => setPipeline(removeStage(pipeline, s.id))}><Trash2 size={14} /></button>
        </div>
      ))}
      {selected && (
        <div className="card mt" style={{ padding: 12 }}>
          <h3 style={{ marginBottom: 4 }}>{opInfo(selected.type).label}</h3>
          <p className="small faint" style={{ marginBottom: 10 }}>{opInfo(selected.type).description}</p>
          <StageForm stage={selected} advanced onChange={(settings) => setPipeline(updateStage(pipeline, selected.id, (x) => ({ ...x, settings })))} />
        </div>
      )}
    </div>
  );
}

export function PlanSummary({ check, error, busy }: { check: ExportCheck | null; error: string | null; busy: boolean }) {
  if (error) return <Notice tone="err">{error}</Notice>;
  if (!check) return busy ? <div className="small faint">Checking…</div> : null;
  const p = check.plan;
  const errors = p.issues.filter((i) => i.severity === "error");
  const go = (stageModel: boolean) => stageModel && useStore.setState({ view: "models" });
  return (
    <div className="col" style={{ gap: 6 }} data-testid="plan-summary">
      <div className="plan-line"><span>Output</span><span>{p.output.width}×{p.output.height} · {fps(p.output.fps)}</span></div>
      <div className="plan-line"><span>Peak memory</span><span>{mb(p.estimates.peak_memory_mb[0])} – {mb(p.estimates.peak_memory_mb[1])}</span></div>
      <div className="plan-line"><span>Output size</span><span>{mb(p.estimates.output_size_mb[0])} – {mb(p.estimates.output_size_mb[1])}</span></div>
      {p.cfr_conversion && <div className="plan-line"><span>Frame rate</span><span>VFR → {fps(p.cfr_conversion)}</span></div>}
      {p.issues.length > 0 && (
        <div className="issues">
          {p.issues.map((i, k) => (
            <Notice key={k} tone={i.severity === "error" ? "err" : i.severity === "warning" ? "warn" : "info"}>
              {i.message}
              {i.action && <div className="small faint">{i.action}</div>}
              {/model/i.test(i.message) && i.severity === "error" && (
                <button className="btn small mt" style={{ marginTop: 6 }} onClick={() => go(true)}>Open Model Library</button>
              )}
            </Notice>
          ))}
        </div>
      )}
      {errors.length === 0 && busy && <div className="tiny faint">Updating…</div>}
    </div>
  );
}

export function Inspector() {
  const settings = useStore((s) => s.settings);
  const source = useStore((s) => s.source);
  const history = useStore((s) => s.history);
  const stillBusy = useStore((s) => s.stillBusy);
  const clip = useStore((s) => s.clip);
  const presetName = useStore((s) => s.presetName);
  const advanced = !!settings?.inspector_mode_advanced;
  const plan = usePlanCheck();
  const [saving, setSaving] = useState(false);
  const blocked = !!plan.check?.plan.issues.some((i) => i.severity === "error");
  const anyStage = useStore((s) => s.pipeline.stages.some((x) => x.enabled));

  return (
    <aside className="inspector" aria-label="Inspector">
      <div className="inspector-head">
        <Seg label="Inspector mode" value={advanced ? "advanced" : "simple"} onChange={(v) => saveSettings((s) => ({ ...s, inspector_mode_advanced: v === "advanced" }))}
          options={[{ value: "simple", label: "Tasks" }, { value: "advanced", label: "Pipeline" }]} />
        <span className="spacer" />
        <button className="btn ghost icon small" aria-label="Undo" title="Undo" disabled={!history.past.length} onClick={undo}><Undo2 size={15} /></button>
        <button className="btn ghost icon small" aria-label="Redo" title="Redo" disabled={!history.future.length} onClick={redo}><Redo2 size={15} /></button>
        <button className="btn ghost icon small" aria-label="Save as preset" title="Save as preset" onClick={() => setSaving(true)}><Save size={15} /></button>
      </div>
      {presetName && <div className="tiny faint" style={{ padding: "6px 12px 0" }}>Based on preset: {presetName}</div>}
      <div className="inspector-body">{advanced ? <PipelineEditor /> : <SimpleTasks />}</div>
      <div className="inspector-foot">
        {source && <PlanSummary {...plan} />}
        <div className="row">
          <button className="btn grow" disabled={!source || stillBusy || !anyStage || blocked} onClick={renderStill} title="Render the current frame with the real models">
            <ImageIcon size={15} /> {stillBusy ? "Rendering…" : "Preview frame"}
          </button>
          <button className="btn grow" disabled={!source || !anyStage || !!clip?.job_id || blocked} onClick={renderClip} title={`Render ${settings?.preview_seconds ?? 3} s from the playhead`}>
            <Film size={15} /> {clip?.job_id ? "Rendering clip…" : "Preview clip"}
          </button>
        </div>
        <button className="btn primary large" disabled={!source || blocked} onClick={() => useStore.setState({ exportOpen: true })} data-testid="open-export">
          <Upload size={16} /> Export…
        </button>
      </div>
      {saving && <SavePresetDialog onClose={() => setSaving(false)} />}
    </aside>
  );
}
