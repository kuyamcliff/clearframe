import { useEffect, useMemo, useRef, useState, type PointerEvent as RPointerEvent, type RefObject } from "react";
import { ChevronDown, ChevronsLeftRight, FileVideo, Loader2, MoreHorizontal, Pause, Play, SkipBack, SkipForward, StepBack, StepForward, Upload, Volume2, VolumeX } from "lucide-react";

import { useMenu } from "../components/ui";
import { api, fileUrl, pickFile, VIDEO_EXTENSIONS } from "../lib/ipc";
import { basename, relTime, timecode } from "../lib/format";
import { moveStage, SECTION_TITLE, stageSummary } from "../lib/ops";
import type { RecentFile } from "../lib/types";
import { errorToast, openSource, renderClip, renderStill, setPipeline, useStore } from "../state/store";
import { Panels } from "./Panels";
import { openProjectFile } from "./Projects";

export async function chooseVideos() {
  const files = await pickFile("Open video", [{ name: "Video", extensions: VIDEO_EXTENSIONS }, { name: "All files", extensions: ["*"] }], true);
  if (!files.length) return;
  const s = useStore.getState();
  useStore.setState({ sources: [...s.sources, ...files.filter((f) => !s.sources.includes(f))] });
  await openSource(files[0]);
}

/** Frame step and duration of the open source. */
function useClock() {
  const source = useStore((s) => s.source);
  const v = source?.info.video;
  return { step: v ? v.frame_rate.den / v.frame_rate.num : 1 / 30, dur: source?.info.duration ?? 0 };
}

function useOriginalFrame(): string | null {
  const path = useStore((s) => s.sourcePath);
  const t = useStore((s) => s.playhead);
  const ready = useStore((s) => !!s.source);
  const [img, setImg] = useState<string | null>(null);
  useEffect(() => {
    if (!path || !ready) return;
    let live = true;
    const h = window.setTimeout(() => api.sourceFrame(path, t, 1920).then((p) => live && setImg(p)).catch(() => {}), 90);
    return () => { live = false; window.clearTimeout(h); };
  }, [path, t, ready]);
  return img;
}

function useSync(a: RefObject<HTMLVideoElement>, b: RefObject<HTMLVideoElement>, on: boolean) {
  useEffect(() => {
    const x = a.current, y = b.current;
    if (!on || !x || !y) return;
    const sync = () => { if (Math.abs(x.currentTime - y.currentTime) > 0.05) x.currentTime = y.currentTime; };
    const play = () => { x.play().catch(() => {}); };
    const pause = () => x.pause();
    y.addEventListener("timeupdate", sync);
    y.addEventListener("play", play);
    y.addEventListener("pause", pause);
    return () => { y.removeEventListener("timeupdate", sync); y.removeEventListener("play", play); y.removeEventListener("pause", pause); };
  }, [a, b, on]);
}

function Media({ src, kind, alt, vref, zoomW }: { src: string; kind: "img" | "video"; alt: string; vref?: RefObject<HTMLVideoElement>; zoomW?: number }) {
  const muted = useStore((s) => s.muted);
  const style = zoomW ? { width: zoomW, maxWidth: "none", maxHeight: "none" } : undefined;
  return kind === "img" ? <img src={src} alt={alt} draggable={false} style={style} /> : <video ref={vref} src={src} muted={muted} loop playsInline style={style} />;
}

function Wipe({ left, right, kind, l, r }: { left: string; right: string; kind: "img" | "video"; l: RefObject<HTMLVideoElement>; r: RefObject<HTMLVideoElement> }) {
  const [pos, setPos] = useState(0.5);
  const box = useRef<HTMLDivElement>(null);
  const drag = (e: RPointerEvent) => {
    const b = box.current!.getBoundingClientRect();
    setPos(Math.max(0, Math.min(1, (e.clientX - b.left) / b.width)));
  };
  return (
    <div className="wipe" ref={box} onPointerDown={(e) => { (e.target as Element).setPointerCapture?.(e.pointerId); drag(e); }} onPointerMove={(e) => e.buttons && drag(e)}
      role="slider" aria-label="Compare position" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(pos * 100)} tabIndex={0}
      onKeyDown={(e) => { if (e.key === "ArrowLeft") setPos((p) => Math.max(0, p - 0.05)); if (e.key === "ArrowRight") setPos((p) => Math.min(1, p + 0.05)); }}>
      <div className="layer"><Media src={right} kind={kind} alt="Preview" vref={r} /></div>
      <div className="layer" style={{ clipPath: `inset(0 ${(1 - pos) * 100}% 0 0)` }}><Media src={left} kind={kind} alt="Original" vref={l} /></div>
      <div className="bar" style={{ left: `${pos * 100}%` }} />
      <div className="knob" style={{ left: `${pos * 100}%` }}><ChevronsLeftRight size={14} /></div>
      <span className="label l">Original</span>
      <span className="label r">Preview</span>
    </div>
  );
}

function FxStrip() {
  const pipeline = useStore((s) => s.pipeline);
  const models = useStore((s) => s.models);
  const { open, node } = useMenu();
  const on = pipeline.stages.filter((s) => s.enabled);
  const orderMenu = (e: React.MouseEvent) => {
    const items = pipeline.stages.flatMap((s, i) => [
      { label: `${i + 1}. ${SECTION_TITLE[s.type]}${s.enabled ? "" : " (off)"} — move up`, onClick: () => setPipeline(moveStage(useStore.getState().pipeline, s.id, -1)), disabled: i === 0, sep: i > 0 },
      { label: `${i + 1}. ${SECTION_TITLE[s.type]} — move down`, onClick: () => setPipeline(moveStage(useStore.getState().pipeline, s.id, 1)), disabled: i === pipeline.stages.length - 1 },
    ]);
    open(e, items.length ? items : [{ label: "No adjustments", onClick: () => {}, disabled: true }]);
  };
  return (
    <div className="fx" data-testid="fx">
      <span className="fn">fx</span>
      <div className="chips">
        {on.length === 0 ? <span className="faint">No adjustments</span> : on.map((s, i) => (
          <span key={s.id} className="chip">{i > 0 && <span className="dot" style={{ marginRight: 10 }}>·</span>}<b>{SECTION_TITLE[s.type]}</b><span>{stageSummary(s, models)}</span></span>
        ))}
      </div>
      <button className="btn ghost sm icon" aria-label="Processing order" title="Processing order" onClick={orderMenu}><MoreHorizontal size={14} /></button>
      {node}
    </div>
  );
}

function Stage({ l, r }: { l: RefObject<HTMLVideoElement>; r: RefObject<HTMLVideoElement> }) {
  const source = useStore((s) => s.source);
  const still = useStore((s) => s.still);
  const stillAt = useStore((s) => s.stillAt);
  const stillBusy = useStore((s) => s.stillBusy);
  const stillError = useStore((s) => s.stillError);
  const clip = useStore((s) => s.clip);
  const clipError = useStore((s) => s.clipError);
  const mode = useStore((s) => s.compare);
  const zoom = useStore((s) => s.zoom);
  const playhead = useStore((s) => s.playhead);
  const progress = useStore((s) => (clip?.job_id ? s.progress[clip.job_id] : undefined));
  const original = useOriginalFrame();
  const stillFresh = !!still && Math.abs(stillAt - playhead) < 0.02;
  const clipReady = !!clip?.path;
  let left = fileUrl(original), right = "", kind: "img" | "video" = "img";
  let lw = source?.info.video?.width ?? 0, rw = lw;
  if (clipReady) { left = fileUrl(clip!.original_proxy); right = fileUrl(clip!.path); kind = "video"; }
  else if (stillFresh) { left = fileUrl(still!.original); right = fileUrl(still!.processed); lw = still!.original_size[0]; rw = still!.processed_size[0]; }
  useSync(l, r, kind === "video");
  const z = zoom === "fit" ? 0 : Number(zoom);
  const single = (src: string, alt: string, w: number, ref?: RefObject<HTMLVideoElement>) => (
    <div className={`view${z ? " zoomed" : ""}`}>{src ? <Media src={src} kind={kind} alt={alt} vref={ref} zoomW={z ? w * z : undefined} /> : <Loader2 className="spin faint" />}</div>
  );
  const effective = right ? mode : "original";

  let status: React.ReactNode = null;
  if (stillBusy) status = <div className="status"><Loader2 size={13} className="spin" /> Rendering frame…</div>;
  else if (clip?.job_id) status = <div className="status"><Loader2 size={13} className="spin" /> Rendering clip {progress ? `${Math.round(progress.fraction * 100)}%` : ""}</div>;
  else if (stillError) status = <div className="status err" role="alert">{stillError}</div>;
  else if (clipError) status = <div className="status err" role="alert">{clipError}</div>;

  return (
    <div className="stage" data-testid="viewer">
      {effective === "split" ? <Wipe left={left} right={right} kind={kind} l={l} r={r} />
        : effective === "side" ? (
          <div className="side-by-side">
            {single(left, "Original", lw, l)}
            {single(right, "Preview", rw, r)}
          </div>
        ) : effective === "processed" ? single(right, "Preview", rw, r) : single(left, "Original", lw, l)}
      {status}
    </div>
  );
}

function Ruler({ dur }: { dur: number }) {
  const ticks = useMemo(() => {
    const steps = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600];
    const major = steps.find((s) => dur / s <= 10) ?? 3600;
    const minor = major / 5;
    const out: { t: number; major: boolean }[] = [];
    for (let i = 0; i * minor <= dur + 1e-6; i++) out.push({ t: i * minor, major: i % 5 === 0 });
    return out;
  }, [dur]);
  const fmt = (t: number) => {
    const r = Math.round(t * 1000) / 1000;
    const h = Math.floor(r / 3600), m = Math.floor((r % 3600) / 60), s = Math.floor(r % 60);
    return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  };
  return (
    <>
      {ticks.map((k) => (
        <span key={k.t}>
          <span className={`tick${k.major ? " major" : ""}`} style={{ left: `${(k.t / dur) * 100}%` }} />
          {k.major && k.t / dur < 0.94 && <span className="t" style={{ left: `${(k.t / dur) * 100}%` }}>{fmt(k.t)}</span>}
        </span>
      ))}
    </>
  );
}

function Timeline() {
  const source = useStore((s) => s.source);
  const playhead = useStore((s) => s.playhead);
  const range = useStore((s) => s.range);
  const clip = useStore((s) => s.clip);
  const clipRange = useStore((s) => s.clipRange);
  const still = useStore((s) => s.still);
  const stillAt = useStore((s) => s.stillAt);
  const ref = useRef<HTMLDivElement>(null);
  if (!source) return null;
  const dur = source.info.duration || 1;
  const pct = (t: number) => `${(Math.max(0, Math.min(dur, t)) / dur) * 100}%`;
  const seek = (e: RPointerEvent) => {
    const b = ref.current!.getBoundingClientRect();
    useStore.setState({ playhead: Math.max(0, Math.min(dur, ((e.clientX - b.left) / b.width) * dur)) });
  };
  return (
    <div className="timeline">
      <div ref={ref} style={{ position: "relative" }} onPointerDown={(e) => { (e.target as Element).setPointerCapture?.(e.pointerId); seek(e); }} onPointerMove={(e) => e.buttons && seek(e)}
        role="slider" aria-label="Timeline" aria-valuemin={0} aria-valuemax={dur} aria-valuenow={playhead} aria-valuetext={timecode(playhead)} tabIndex={0}>
        <div className="ruler"><Ruler dur={dur} /></div>
        <div className="track">
          {still && <div className="rendered" style={{ left: pct(stillAt), width: 3 }} title="Rendered frame" />}
          {clipRange && <div className={clip?.job_id ? "rendering" : clip?.path ? "rendered" : "rendering"} style={{ left: pct(clipRange.start), width: `calc(${pct(clipRange.end)} - ${pct(clipRange.start)})` }} />}
        </div>
        <div className="film">
          {source.thumbnails.map(([t, p]) => <img key={t} src={fileUrl(p)} alt="" draggable={false} />)}
          {range && <div className="out" style={{ left: 0, width: pct(range.start) }} />}
          {range && <div className="out" style={{ left: pct(range.end), right: 0 }} />}
        </div>
        {range && <div className="inout" style={{ left: pct(range.start), width: `calc(${pct(range.end)} - ${pct(range.start)})` }} />}
        <div className="playhead" style={{ left: pct(playhead) }} />
      </div>
    </div>
  );
}

function RenderButton() {
  const stillBusy = useStore((s) => s.stillBusy);
  const clip = useStore((s) => s.clip);
  const anyStage = useStore((s) => s.pipeline.stages.some((x) => x.enabled));
  const [last, setLast] = useState<number>(0);
  const { open, node } = useMenu();
  const run = (sec: number) => { setLast(sec); sec === 0 ? renderStill() : renderClip(sec); };
  const busy = stillBusy || !!clip?.job_id;
  const label = last === 0 ? "Render frame" : `Render ${last}s`;
  return (
    <div className="split-btn">
      <button className="btn primary sm" disabled={!anyStage || busy} onClick={() => run(last)} data-testid="render">{label}</button>
      <button className="btn primary sm" disabled={!anyStage || busy} aria-label="Render options" onClick={(e) => open(e, [
        { label: "Frame", onClick: () => run(0) },
        { label: "1 second", onClick: () => run(1) },
        { label: "3 seconds", onClick: () => run(3) },
        { label: "5 seconds", onClick: () => run(5) },
        { label: "10 seconds", onClick: () => run(10) },
      ])}><ChevronDown size={13} /></button>
      {node}
    </div>
  );
}

function Playback({ r }: { r: RefObject<HTMLVideoElement> }) {
  const playhead = useStore((s) => s.playhead);
  const range = useStore((s) => s.range);
  const zoom = useStore((s) => s.zoom);
  const clip = useStore((s) => s.clip);
  const muted = useStore((s) => s.muted);
  const { step, dur } = useClock();
  const [playing, setPlaying] = useState(false);
  const set = (t: number) => useStore.setState({ playhead: Math.max(0, Math.min(dur, t)) });
  const canPlay = !!clip?.path;
  useEffect(() => {
    const v = r.current;
    if (!v) return setPlaying(false);
    const on = () => setPlaying(true), off = () => setPlaying(false);
    v.addEventListener("play", on); v.addEventListener("pause", off);
    return () => { v.removeEventListener("play", on); v.removeEventListener("pause", off); };
  }, [r, clip?.path]);
  const toggle = () => { const v = r.current; if (!v) return; v.paused ? v.play().catch(() => {}) : v.pause(); };
  return (
    <div className="playback">
      <span className="tc">{timecode(playhead)}</span>
      <div className="center">
        <button className="tb icon" aria-label="Go to start" title="Start" onClick={() => set(range?.start ?? 0)}><SkipBack size={15} /></button>
        <button className="tb icon" aria-label="Previous frame" title="Previous frame (←)" onClick={() => set(playhead - step)}><StepBack size={15} /></button>
        <button className="tb icon" aria-label={playing ? "Pause" : "Play"} disabled={!canPlay} title={canPlay ? "Play preview (Space)" : "Render a clip to play it"} onClick={toggle}>{playing ? <Pause size={15} /> : <Play size={15} />}</button>
        <button className="tb icon" aria-label="Next frame" title="Next frame (→)" onClick={() => set(playhead + step)}><StepForward size={15} /></button>
        <button className="tb icon" aria-label="Go to end" title="End" onClick={() => set(range?.end ?? dur)}><SkipForward size={15} /></button>
      </div>
      <div className="right">
        <button className="tb icon" aria-label={muted ? "Unmute" : "Mute"} onClick={() => useStore.setState({ muted: !muted })}>{muted ? <VolumeX size={15} /> : <Volume2 size={15} />}</button>
        <select className="input fit" aria-label="Zoom" value={zoom} onChange={(e) => useStore.setState({ zoom: e.target.value as "fit" | "1" | "2" })}>
          <option value="fit">Fit</option>
          <option value="1">100%</option>
          <option value="2">200%</option>
        </select>
        <RenderButton />
      </div>
    </div>
  );
}

function Empty() {
  const [over, setOver] = useState(false);
  const [recent, setRecent] = useState<RecentFile[]>([]);
  useEffect(() => { api.recentList().then(setRecent).catch(() => {}); }, []);
  return (
    <div className="empty-player" onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)} onDrop={(e) => { e.preventDefault(); setOver(false); }}>
      <div className={`drop${over ? " over" : ""}`}>
        <Upload size={26} />
        <span className="t">Drop videos here</span>
        <button className="btn primary" onClick={() => chooseVideos().catch(errorToast)}>Browse</button>
      </div>
      {recent.length > 0 && (
        <div className="recent">
          <div className="h">Recent</div>
          {recent.slice(0, 6).map((f) => (
            <button key={f.path} onClick={() => (f.kind === "project" ? openProjectFile(f.path) : openSource(f.path)).catch(errorToast)} title={f.path}>
              <FileVideo size={14} className="faint" />
              <span className="grow ellipsis">{f.title || basename(f.path)}</span>
              <span className="faint">{relTime(f.opened_at)}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function Player() {
  const sourcePath = useStore((s) => s.sourcePath);
  const source = useStore((s) => s.source);
  const loading = useStore((s) => s.sourceLoading);
  const error = useStore((s) => s.sourceError);
  const mode = useStore((s) => s.compare);
  const presetName = useStore((s) => s.presetName);
  const projectName = useStore((s) => s.projectName);
  const l = useRef<HTMLVideoElement>(null);
  const r = useRef<HTMLVideoElement>(null);
  const previewName = projectName ?? presetName ?? "Untitled";
  return (
    <section className="player" aria-label="Player">
      <div className="tabs" role="tablist">
        <button className="tab" role="tab" aria-selected={mode === "original" || mode === "split" || mode === "side"} onClick={() => useStore.setState({ compare: "original" })}>Original</button>
        <button className="tab" role="tab" aria-selected={mode !== "original"} onClick={() => useStore.setState({ compare: mode === "original" ? "processed" : mode })} disabled={!source}>
          <FileVideo size={13} /> {previewName}
        </button>
      </div>
      <FxStrip />
      {!sourcePath ? <Empty />
        : loading ? <div className="stage"><div className="view"><Loader2 className="spin faint" /></div></div>
        : error ? (
          <div className="empty-player">
            <span className="err" role="alert">Could not open {basename(sourcePath)}</span>
            <span className="faint selectable" style={{ maxWidth: 520, textAlign: "center" }}>{error}</span>
            <button className="btn" onClick={() => chooseVideos().catch(errorToast)}>Choose another file</button>
          </div>
        ) : (
          <>
            <Stage l={l} r={r} />
            <Playback r={r} />
            <Timeline />
          </>
        )}
      <Panels />
    </section>
  );
}
