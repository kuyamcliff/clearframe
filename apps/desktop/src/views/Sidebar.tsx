import { useEffect, useMemo, useState } from "react";
import { ChevronRight, FolderOpen, Plus, Share, Trash2 } from "lucide-react";

import { StageForm } from "../components/StageForm";
import { Check, F, Issue, NumberInput, Range, Select, Seg, Switch } from "../components/ui";
import { api, pickFolder, pickSave } from "../lib/ipc";
import { basename, dirname, fps, mb, timecode } from "../lib/format";
import { addStageCanonical, newStageId, SECTION_ORDER, SECTION_TITLE, shortName, stageSummary, updateStage } from "../lib/ops";
import type { AudioMode, Container, ExportCheck, OperationType, OutputSettings, RateControl } from "../lib/types";
import { currentJob, enableOp, errorToast, setPipeline, toast, useStore } from "../state/store";

export function usePlanCheck(): { check: ExportCheck | null; error: string | null; busy: boolean } {
  const pipeline = useStore((s) => s.pipeline);
  const output = useStore((s) => s.output);
  const source = useStore((s) => s.source);
  const range = useStore((s) => s.range);
  const models = useStore((s) => s.models);
  const folder = useStore((s) => s.exportFolder);
  const name = useStore((s) => s.exportFileName);
  const [st, setSt] = useState<{ check: ExportCheck | null; error: string | null; busy: boolean }>({ check: null, error: null, busy: false });
  useEffect(() => {
    if (!source || !output) return setSt({ check: null, error: null, busy: false });
    let live = true;
    setSt((x) => ({ ...x, busy: true }));
    const t = window.setTimeout(async () => {
      const job = currentJob();
      if (!job) return;
      try {
        const check = await api.checkExport(job);
        if (live) setSt({ check, error: null, busy: false });
      } catch (e) {
        if (live) setSt({ check: null, error: e instanceof Error ? e.message : String(e), busy: false });
      }
    }, 250);
    return () => { live = false; window.clearTimeout(t); };
  }, [pipeline, output, source, range, models, folder, name]);
  return st;
}

function Section({ type }: { type: OperationType }) {
  const pipeline = useStore((s) => s.pipeline);
  const models = useStore((s) => s.models);
  const defaults = useStore((s) => s.defaults);
  const stages = pipeline.stages.filter((s) => s.type === type);
  const on = stages.some((s) => s.enabled);
  const [open, setOpen] = useState(on);
  useEffect(() => { if (on) setOpen(true); }, [on]);
  const first = stages[0];
  const edit = (id: string) => (settings: Record<string, unknown>) => setPipeline(updateStage(useStore.getState().pipeline, id, (x) => ({ ...x, settings })));
  const addSecond = () => {
    if (!defaults) return;
    const p = useStore.getState().pipeline;
    const last = p.stages.map((s) => s.type).lastIndexOf(type);
    const stage = { id: newStageId(type, p.stages), enabled: true, type, settings: structuredClone(defaults[type].settings) };
    const next = [...p.stages];
    next.splice(last + 1, 0, stage);
    setPipeline({ stages: next });
  };
  return (
    <div className="sec" data-testid={`sec-${type}`}>
      <div className="sec-h" onClick={() => setOpen(!open)} role="button" aria-expanded={open}>
        <ChevronRight size={14} className={`chev${open ? " open" : ""}`} />
        <span className="t">{SECTION_TITLE[type]}</span>
        {!open && on && first && <span className="sum">{stageSummary(first, models)}</span>}
        <Switch label={SECTION_TITLE[type]} checked={on} onChange={(v) => { enableOp(type, v); if (v) setOpen(true); }} />
      </div>
      {open && (
        <div className="sec-b">
          {first ? (
            <StageForm stage={first} onChange={edit(first.id)} />
          ) : (
            defaults && <StageForm stage={{ id: `${type}-new`, enabled: false, type, settings: defaults[type].settings }}
              onChange={(settings) => setPipeline(addStageCanonical(useStore.getState().pipeline, type, settings))} />
          )}
          {stages.slice(1).map((st, i) => (
            <div className="sub" key={st.id}>
              <div className="sub-h">
                <span className="grow">{i === 0 ? "Second" : `Pass ${i + 2}`} {SECTION_TITLE[type].toLowerCase()}</span>
                <Switch label={`${SECTION_TITLE[type]} ${i + 2}`} checked={st.enabled} onChange={(v) => setPipeline(updateStage(useStore.getState().pipeline, st.id, (x) => ({ ...x, enabled: v })))} />
                <button className="btn ghost sm icon" aria-label="Remove pass" onClick={() => setPipeline({ stages: useStore.getState().pipeline.stages.filter((x) => x.id !== st.id) })}><Trash2 size={13} /></button>
              </div>
              <StageForm stage={st} onChange={edit(st.id)} />
            </div>
          ))}
          {(type === "enhance" || type === "denoise") && first && stages.length < 3 && (
            <button className="btn sm ghost addbtn" onClick={addSecond}><Plus size={13} /> Add {stages.length === 1 ? "second" : "another"} {SECTION_TITLE[type].toLowerCase()}</button>
          )}
        </div>
      )}
    </div>
  );
}

const CONTAINERS: { value: Container; label: string }[] = [{ value: "mp4", label: "mp4" }, { value: "mkv", label: "mkv" }, { value: "mov", label: "mov" }, { value: "webm", label: "webm" }];

function Codec() {
  const output = useStore((s) => s.output)!;
  const codecs = useStore((s) => s.codecs);
  const folder = useStore((s) => s.exportFolder);
  const fileName = useStore((s) => s.exportFileName);
  const settings = useStore((s) => s.settings);
  const range = useStore((s) => s.range);
  const [def, setDef] = useState("");
  useEffect(() => { api.defaultExportFolder().then(setDef).catch(() => {}); }, []);
  const setOut = (patch: Partial<OutputSettings>) => useStore.setState({ output: { ...output, ...patch } });
  const setVideo = (patch: Partial<OutputSettings["video"]>) => setOut({ video: { ...output.video, ...patch } });
  const compatible = useMemo(() => codecs.filter((c) => c.profile.containers.includes(output.container)), [codecs, output.container]);
  const profile = codecs.find((c) => c.profile.encoder === output.video.encoder)?.profile;
  const rate = output.video.rate;
  const audio = output.audio;

  const pickEncoder = (enc: string, base = output) => {
    const p = codecs.find((c) => c.profile.encoder === enc)?.profile;
    if (!p) return;
    const r: RateControl = p.quality_range ? { mode: "quality", value: p.quality_range[1] } : p.supports_lossless ? { mode: "lossless" } : { mode: "bitrate", kbps: 20000 };
    useStore.setState({ output: { ...base, video: { encoder: enc, preset: p.default_preset, pix_fmt: p.pix_fmts.includes(base.video.pix_fmt) ? base.video.pix_fmt : p.pix_fmts[0], rate: r, profile: null, gop: null } } });
  };
  const pickContainer = (c: Container) => {
    const next = { ...output, container: c };
    if (codecs.some((x) => x.profile.encoder === output.video.encoder && x.profile.containers.includes(c))) useStore.setState({ output: next });
    else {
      const first = codecs.find((x) => x.available && x.profile.containers.includes(c));
      if (first) pickEncoder(first.profile.encoder, next);
      else useStore.setState({ output: next });
    }
  };

  return (
    <div className="sec-b" style={{ paddingTop: 14 }}>
      <F label="Container" htmlFor="c-cont"><Select id="c-cont" value={output.container} onChange={pickContainer} options={CONTAINERS} /></F>
      <F label="Codec" htmlFor="c-enc">
        <Select id="c-enc" value={output.video.encoder} onChange={(e) => pickEncoder(e)} options={[
          ...compatible.map((c) => ({ value: c.profile.encoder, label: c.available ? c.profile.label : `${c.profile.label} (unavailable)`, disabled: !c.available })),
          ...(compatible.some((c) => c.profile.encoder === output.video.encoder) ? [] : [{ value: output.video.encoder, label: output.video.encoder, disabled: true }]),
        ]} />
      </F>
      {profile && profile.presets.length > 0 && (
        <F label="Speed" htmlFor="c-preset">
          <Select id="c-preset" value={output.video.preset ?? profile.default_preset ?? profile.presets[0]} onChange={(x) => setVideo({ preset: x })} options={profile.presets.map((p) => ({ value: p, label: p }))} />
        </F>
      )}
      {profile && (
        <F label="Bitrate">
          <Seg label="Rate control" value={rate.mode} onChange={(m) => setVideo({ rate: m === "quality" ? { mode: "quality", value: profile.quality_range?.[1] ?? 20 } : m === "bitrate" ? { mode: "bitrate", kbps: 20000 } : { mode: "lossless" } })}
            options={[
              ...(profile.quality_range ? [{ value: "quality" as const, label: "Dynamic" }] : []),
              ...(profile.supports_bitrate ? [{ value: "bitrate" as const, label: "Constant" }] : []),
              ...(profile.supports_lossless ? [{ value: "lossless" as const, label: "Lossless" }] : []),
            ]} />
        </F>
      )}
      {rate.mode === "quality" && profile?.quality_range && (
        <F label="Quality" title="Lower = higher quality, larger file"><Range label="Quality" value={rate.value} min={profile.quality_range[0]} max={profile.quality_range[2]} onChange={(x) => setVideo({ rate: { mode: "quality", value: x } })} /></F>
      )}
      {rate.mode === "bitrate" && (
        <F label="Mbps"><NumberInput label="Bitrate (Mbps)" value={rate.kbps / 1000} min={0.1} max={1000} step={1} onChange={(x) => setVideo({ rate: { mode: "bitrate", kbps: Math.round(x * 1000) } })} /></F>
      )}
      {profile && (
        <F label="Pixel format" htmlFor="c-pix"><Select id="c-pix" value={output.video.pix_fmt} onChange={(x) => setVideo({ pix_fmt: x })} options={profile.pix_fmts.map((p) => ({ value: p, label: p }))} /></F>
      )}
      <F label="Audio" htmlFor="c-audio">
        <Select id="c-audio" value={audio.mode} onChange={(m) => {
          const a: AudioMode = m === "encode" ? { mode: "encode", codec: output.container === "webm" ? "libopus" : "aac", bitrate_kbps: 192 } : m === "none" ? { mode: "none" } : { mode: "auto" };
          setOut({ audio: a });
        }} options={[{ value: "auto", label: "Copy" }, { value: "encode", label: "Convert" }, { value: "none", label: "None" }]} />
      </F>
      {audio.mode === "encode" && (
        <>
          <F label="Audio codec" htmlFor="c-ac">
            <Select id="c-ac" value={audio.codec} onChange={(x) => setOut({ audio: { ...audio, codec: x } })} options={[{ value: "aac", label: "AAC" }, { value: "libopus", label: "Opus" }, { value: "flac", label: "FLAC" }, { value: "pcm_s16le", label: "PCM 16-bit" }, { value: "ac3", label: "AC-3" }]} />
          </F>
          <F label="Audio kbps"><NumberInput label="Audio bitrate" value={audio.bitrate_kbps} min={32} max={1536} step={32} onChange={(x) => setOut({ audio: { ...audio, bitrate_kbps: Math.round(x) } })} /></F>
        </>
      )}
      <F label="Keep">
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <Check checked={output.subtitles} onChange={(x) => setOut({ subtitles: x })} label="Subtitles" />
          <Check checked={output.chapters} onChange={(x) => setOut({ chapters: x })} label="Chapters" />
          <Check checked={output.metadata} onChange={(x) => setOut({ metadata: x })} label="Metadata" />
        </div>
      </F>
      <div className="sub">
        <F label="Folder">
          <button className="btn sm ghost" style={{ maxWidth: "100%", justifyContent: "flex-start" }} title={folder ?? settings?.output_dir ?? def}
            onClick={async () => { const f = await pickFolder("Output folder", folder ?? settings?.output_dir ?? def); if (f) useStore.setState({ exportFolder: f }); }}>
            <FolderOpen size={13} /><span className="ellipsis">{basename(folder ?? settings?.output_dir ?? def) || "Choose…"}</span>
          </button>
        </F>
        <F label="File name" htmlFor="c-name">
          <input id="c-name" className="input" placeholder={settings?.naming_template ?? "{name}_{task}"} value={fileName ?? ""} onChange={(e) => useStore.setState({ exportFileName: e.target.value.trim() ? e.target.value : null })} />
        </F>
        {range && <F label="Range"><span className="muted">{timecode(range.start)} – {timecode(range.end)}</span></F>}
      </div>
    </div>
  );
}

export async function exportNow(saveAs: boolean, destination?: string) {
  const s = useStore.getState();
  let job = currentJob();
  if (!job) return;
  if (saveAs) {
    const ext = s.output?.container ?? "mp4";
    const path = await pickSave("Export as", destination ?? `${s.exportFolder ?? ""}`, [{ name: ext.toUpperCase(), extensions: [ext] }]);
    if (!path) return;
    const name = basename(path).replace(/\.[^.]+$/, "");
    useStore.setState({ exportFolder: dirname(path), exportFileName: name });
    job = { ...job, folder: dirname(path), file_name: name };
  }
  try {
    const rec = await api.enqueue(job);
    toast("success", `Queued ${basename(rec.destination)}`, { label: "Show queue", run: () => useStore.setState({ panel: "queue" }) });
  } catch (e) {
    errorToast(e);
  }
}

/** Backend messages name models by id; show their display names instead. */
function useNiceText() {
  const models = useStore((s) => s.models);
  return (msg: string) => models.reduce((m, c) => (c.manifest ? m.split(c.manifest.id).join(shortName(c)) : m), msg);
}

export function Sidebar() {
  const nice = useNiceText();
  const tab = useStore((s) => s.sideTab);
  const source = useStore((s) => s.source);
  const output = useStore((s) => s.output);
  const plan = usePlanCheck();
  const errors = plan.check?.plan.issues.filter((i) => i.severity === "error") ?? [];
  const other = plan.check?.plan.issues.filter((i) => i.severity !== "error") ?? [];
  const blocked = !source || !plan.check || errors.length > 0 || !!plan.check.destination_folder_ok?.Err;
  const p = plan.check?.plan;
  return (
    <aside className="side" aria-label="Controls">
      <div className="side-tabs" role="tablist">
        <button role="tab" aria-selected={tab === "adjust"} onClick={() => useStore.setState({ sideTab: "adjust" })}>Adjustments</button>
        <button role="tab" aria-selected={tab === "codec"} onClick={() => useStore.setState({ sideTab: "codec" })}>Codec settings</button>
      </div>
      <div className="side-body">
        {tab === "adjust" ? SECTION_ORDER.map((t) => <Section key={t} type={t} />) : output && <Codec />}
      </div>
      <div className="side-foot">
        {source && (plan.error || errors.length > 0 || other.length > 0 || plan.check?.destination_folder_ok?.Err) && (
          <div className="issues" data-testid="issues">
            {plan.error && <Issue tone="err">{plan.error}</Issue>}
            {plan.check?.destination_folder_ok?.Err && <Issue tone="err">{plan.check.destination_folder_ok.Err}</Issue>}
            {errors.map((i, k) => <Issue key={k} tone="err">{nice(i.message)}</Issue>)}
            {other.map((i, k) => <Issue key={`o${k}`} tone={i.severity === "warning" ? "warn" : "info"}>{nice(i.message)}</Issue>)}
          </div>
        )}
        {p && (
          <div className="est" data-testid="estimate">
            <span>{p.output.width}×{p.output.height} · {fps(p.output.fps)}</span>
            <span title="Estimated output size">{mb(p.estimates.output_size_mb[0])}–{mb(p.estimates.output_size_mb[1])}</span>
          </div>
        )}
        <div className="actions">
          <button className="main" disabled={blocked} onClick={() => exportNow(true, plan.check?.destination)} data-testid="export-as"><FolderOpen size={14} /> Export as…</button>
          <button className="q" disabled={blocked} onClick={() => exportNow(false)} title={plan.check ? `Export to ${plan.check.destination}` : "Quick export"} aria-label="Quick export" data-testid="quick-export"><Share size={15} /></button>
        </div>
      </div>
    </aside>
  );
}
