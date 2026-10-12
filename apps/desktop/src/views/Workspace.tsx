import { useEffect, useRef, useState, type PointerEvent as RPointerEvent } from "react";
import { ChevronLeft, ChevronRight, FileVideo, FolderOpen, Loader2, X } from "lucide-react";

import { Notice, Seg } from "../components/ui";
import { api, fileUrl, pickFile, VIDEO_EXTENSIONS } from "../lib/ipc";
import { basename, bytes, fps, relTime, timecode } from "../lib/format";
import { closeSource, errorToast, openSource, useStore, type CompareMode } from "../state/store";
import { Inspector } from "./Inspector";
import { openProjectFile } from "./Projects";

export async function chooseVideo() {
  const [p] = await pickFile("Open video", [{ name: "Video", extensions: VIDEO_EXTENSIONS }, { name: "All files", extensions: ["*"] }]);
  if (p) await openSource(p);
}

function Facts() {
  const source = useStore((s) => s.source);
  if (!source) return null;
  const v = source.info.video;
  const il = source.interlace;
  const interlaced = il ? il.tff + il.bff > il.progressive : false;
  const hdr = v && ["smpte2084", "arib-std-b67"].includes(v.color.transfer ?? "");
  return (
    <div className="facts">
      {v && <span className="badge">{v.width}×{v.height}</span>}
      {v && <span className="badge">{fps(v.frame_rate)}{v.variable_frame_rate ? " VFR" : ""}</span>}
      {v && <span className="badge">{v.codec.toUpperCase()} {v.bit_depth}-bit</span>}
      <span className="badge">{timecode(source.info.duration, false)}</span>
      <span className="badge">{bytes(source.info.size_bytes)}</span>
      {source.info.audio.length > 0 && <span className="badge">{source.info.audio.length} audio</span>}
      {source.info.subtitles.length > 0 && <span className="badge">{source.info.subtitles.length} subtitles</span>}
      {hdr && <span className="badge accent">HDR {v!.color.transfer === "smpte2084" ? "PQ" : "HLG"}</span>}
      {interlaced && <span className="badge warn">Interlaced</span>}
      {v?.variable_frame_rate && <span className="badge warn" title="Converted to constant frame rate on export">VFR</span>}
    </div>
  );
}

function useOriginalFrame(): string | null {
  const path = useStore((s) => s.sourcePath);
  const t = useStore((s) => s.playhead);
  const ready = useStore((s) => !!s.source);
  const [img, setImg] = useState<string | null>(null);
  useEffect(() => {
    if (!path || !ready) return;
    let live = true;
    const h = window.setTimeout(async () => {
      try {
        const p = await api.sourceFrame(path, t, 1920);
        if (live) setImg(p);
      } catch {
        // keep the previous frame; errors surface on preview
      }
    }, 120);
    return () => {
      live = false;
      window.clearTimeout(h);
    };
  }, [path, t, ready]);
  return img;
}

function Wipe({ left, right, kind }: { left: string; right: string; kind: "img" | "video" }) {
  const [pos, setPos] = useState(0.5);
  const box = useRef<HTMLDivElement>(null);
  const drag = (e: RPointerEvent) => {
    const r = box.current!.getBoundingClientRect();
    setPos(Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)));
  };
  const leftRef = useRef<HTMLVideoElement>(null);
  const rightRef = useRef<HTMLVideoElement>(null);
  useSyncedVideos(leftRef, rightRef, kind === "video");
  return (
    <div className="compare" ref={box} onPointerDown={(e) => { (e.target as Element).setPointerCapture?.(e.pointerId); drag(e); }} onPointerMove={(e) => e.buttons && drag(e)}
      role="slider" aria-label="Compare position" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(pos * 100)} tabIndex={0}
      onKeyDown={(e) => { if (e.key === "ArrowLeft") setPos((p) => Math.max(0, p - 0.05)); if (e.key === "ArrowRight") setPos((p) => Math.min(1, p + 0.05)); }}>
      <div className="layer">{kind === "img" ? <img src={right} alt="Processed" draggable={false} /> : <video ref={rightRef} src={right} muted loop playsInline />}</div>
      <div className="layer" style={{ clipPath: `inset(0 ${(1 - pos) * 100}% 0 0)` }}>
        {kind === "img" ? <img src={left} alt="Original" draggable={false} /> : <video ref={leftRef} src={left} muted loop playsInline />}
      </div>
      <div className="divider" style={{ left: `calc(${pos * 100}% - 1px)` }} />
      <span className="tag left">Original</span>
      <span className="tag right">Processed</span>
    </div>
  );
}

function useSyncedVideos(a: React.RefObject<HTMLVideoElement>, b: React.RefObject<HTMLVideoElement>, on: boolean) {
  useEffect(() => {
    if (!on) return;
    const x = a.current, y = b.current;
    if (!x || !y) return;
    const sync = () => {
      if (Math.abs(x.currentTime - y.currentTime) > 0.05) x.currentTime = y.currentTime;
    };
    const play = () => { x.play().catch(() => {}); };
    const pause = () => x.pause();
    y.addEventListener("timeupdate", sync);
    y.addEventListener("play", play);
    y.addEventListener("pause", pause);
    y.play().catch(() => {});
    return () => {
      y.removeEventListener("timeupdate", sync);
      y.removeEventListener("play", play);
      y.removeEventListener("pause", pause);
    };
  }, [a, b, on]);
}

function SideBySide({ left, right, kind }: { left: string; right: string; kind: "img" | "video" }) {
  const l = useRef<HTMLVideoElement>(null);
  const r = useRef<HTMLVideoElement>(null);
  useSyncedVideos(l, r, kind === "video");
  return (
    <div className="side">
      <div>{kind === "img" ? <img src={left} alt="Original" /> : <video ref={l} src={left} muted loop playsInline />}<span className="tag left">Original</span></div>
      <div>{kind === "img" ? <img src={right} alt="Processed" /> : <video ref={r} src={right} muted loop playsInline controls />}<span className="tag left">Processed</span></div>
    </div>
  );
}

function Viewer() {
  const source = useStore((s) => s.source);
  const still = useStore((s) => s.still);
  const stillAt = useStore((s) => s.stillAt);
  const stillBusy = useStore((s) => s.stillBusy);
  const stillError = useStore((s) => s.stillError);
  const clip = useStore((s) => s.clip);
  const clipError = useStore((s) => s.clipError);
  const compare = useStore((s) => s.compare);
  const playhead = useStore((s) => s.playhead);
  const progress = useStore((s) => (clip?.job_id ? s.progress[clip.job_id] : undefined));
  const original = useOriginalFrame();
  const [showClip, setShowClip] = useState(true);
  useEffect(() => setShowClip(true), [clip?.path]);

  const stillFresh = still && Math.abs(stillAt - playhead) < 0.02;
  const clipReady = clip?.path && showClip;
  let left = "", right = "", kind: "img" | "video" = "img";
  if (clipReady) {
    left = fileUrl(clip!.original_proxy);
    right = fileUrl(clip!.path);
    kind = "video";
  } else if (stillFresh) {
    left = fileUrl(still!.original);
    right = fileUrl(still!.processed);
  } else {
    left = fileUrl(original);
  }
  const hasProcessed = !!right;
  const mode: CompareMode = hasProcessed ? compare : "original";

  return (
    <div className="viewer" data-testid="viewer">
      {!source ? null : mode === "split" ? (
        <Wipe left={left} right={right} kind={kind} />
      ) : mode === "side" ? (
        <SideBySide left={left} right={right} kind={kind} />
      ) : mode === "processed" ? (
        kind === "img" ? <img src={right} alt="Processed preview" /> : <video src={right} controls loop autoPlay muted playsInline />
      ) : left ? (
        kind === "img" ? <img src={left} alt="Original frame" /> : <video src={left} controls loop autoPlay muted playsInline />
      ) : (
        <Loader2 className="spin" />
      )}
      <div className="overlay">
        {stillBusy ? (
          <div className="status"><Loader2 size={14} /> Rendering preview frame with the selected models…</div>
        ) : clip?.job_id ? (
          <div className="status"><Loader2 size={14} /> Rendering preview clip… {progress ? `${Math.round(progress.fraction * 100)}%` : ""}</div>
        ) : stillError ? (
          <div className="status err" role="alert">Preview failed: {stillError}</div>
        ) : clipError ? (
          <div className="status err" role="alert">Clip preview failed: {clipError}</div>
        ) : still && !stillFresh && !clipReady ? (
          <div className="status">The preview frame is from {timecode(stillAt)}. Press “Preview frame” to render this position.</div>
        ) : !hasProcessed ? (
          <div className="status">Original. Press “Preview frame” to see the result with your settings.</div>
        ) : stillFresh && still!.notes.length ? (
          <div className="status">{still!.notes.join(" ")}</div>
        ) : null}
      </div>
      {clip?.path && (
        <div style={{ position: "absolute", top: 10, left: "50%", transform: "translateX(-50%)" }}>
          <Seg label="Preview source" value={showClip ? "clip" : "frame"} onChange={(v) => setShowClip(v === "clip")} options={[{ value: "clip", label: "Clip" }, { value: "frame", label: "Frame" }]} />
        </div>
      )}
    </div>
  );
}

function Timeline() {
  const source = useStore((s) => s.source);
  const playhead = useStore((s) => s.playhead);
  const range = useStore((s) => s.range);
  const clip = useStore((s) => s.clip);
  const previewSeconds = useStore((s) => s.settings?.preview_seconds ?? 3);
  const ref = useRef<HTMLDivElement>(null);
  if (!source) return null;
  const dur = source.info.duration || 1;
  const seek = (e: RPointerEvent) => {
    const r = ref.current!.getBoundingClientRect();
    useStore.setState({ playhead: Math.max(0, Math.min(dur, ((e.clientX - r.left) / r.width) * dur)) });
  };
  const thumbs = source.thumbnails;
  return (
    <div className="timeline" ref={ref} onPointerDown={(e) => { (e.target as Element).setPointerCapture?.(e.pointerId); seek(e); }} onPointerMove={(e) => e.buttons && seek(e)}
      role="slider" aria-label="Timeline" aria-valuemin={0} aria-valuemax={dur} aria-valuenow={playhead} aria-valuetext={timecode(playhead)} tabIndex={0}>
      <div className="thumbs">{thumbs.map(([t, p]) => <img key={t} src={fileUrl(p)} alt="" draggable={false} />)}</div>
      {range && <div className="range" style={{ left: `${(range.start / dur) * 100}%`, width: `${((range.end - range.start) / dur) * 100}%` }} />}
      {clip && <div className="clipmark" style={{ left: `${(playhead / dur) * 100}%`, width: `${(previewSeconds / dur) * 100}%` }} />}
      <div className="head" style={{ left: `${(playhead / dur) * 100}%` }} />
    </div>
  );
}

function Transport() {
  const source = useStore((s) => s.source);
  const playhead = useStore((s) => s.playhead);
  const range = useStore((s) => s.range);
  const compare = useStore((s) => s.compare);
  if (!source) return null;
  const v = source.info.video;
  const step = v ? v.frame_rate.den / v.frame_rate.num : 1 / 30;
  const dur = source.info.duration;
  const set = (t: number) => useStore.setState({ playhead: Math.max(0, Math.min(dur, t)) });
  return (
    <div className="transport">
      <button className="btn ghost icon small" aria-label="Previous frame" onClick={() => set(playhead - step)}><ChevronLeft size={16} /></button>
      <button className="btn ghost icon small" aria-label="Next frame" onClick={() => set(playhead + step)}><ChevronRight size={16} /></button>
      <span className="timecode">{timecode(playhead)} / {timecode(dur)}</span>
      <button className="btn small" onClick={() => useStore.setState({ range: { start: playhead, end: Math.max(playhead + step, range?.end ?? dur) } })} title="Set export start (I)">In</button>
      <button className="btn small" onClick={() => useStore.setState({ range: { start: Math.min(range?.start ?? 0, playhead - step), end: playhead } })} title="Set export end (O)">Out</button>
      {range ? (
        <span className="small muted row">
          Range {timecode(range.start)} – {timecode(range.end)}
          <button className="btn ghost icon small" aria-label="Clear range" onClick={() => useStore.setState({ range: null })}><X size={14} /></button>
        </span>
      ) : (
        <span className="small faint">Whole video</span>
      )}
      <span className="spacer" />
      <Seg label="Compare mode" value={compare} onChange={(c) => useStore.setState({ compare: c })} options={[
        { value: "split", label: "Split", title: "Drag the divider (C)" }, { value: "side", label: "Side by side" },
        { value: "original", label: "Original" }, { value: "processed", label: "Processed" },
      ]} />
    </div>
  );
}

function Welcome() {
  const [over, setOver] = useState(false);
  const [recent, setRecent] = useState<{ path: string; kind: string; title: string; opened_at: string }[]>([]);
  useEffect(() => {
    api.recentList().then(setRecent).catch(() => {});
  }, []);
  return (
    <div className="viewer" style={{ flexDirection: "column", background: "var(--bg)" }} onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}>
      <div className={`dropzone${over ? " over" : ""}`}>
        <FileVideo size={40} />
        <h2 style={{ color: "var(--text)" }}>Open a video to enhance</h2>
        <p className="small">Drop a file here, or choose one. Your original is never modified: results are always saved as new files.</p>
        <div className="row">
          <button className="btn primary large" onClick={() => chooseVideo().catch(errorToast)}><FolderOpen size={16} /> Open video…</button>
          <button className="btn large" onClick={() => openProjectFile().catch(errorToast)}>Open project…</button>
        </div>
      </div>
      {recent.length > 0 && (
        <div className="recent">
          <div className="row"><h3 className="grow">Recent</h3><button className="btn ghost small" onClick={() => api.recentAction("clear").then(() => setRecent([])).catch(errorToast)}>Clear</button></div>
          <ul>
            {recent.map((r) => (
              <li key={r.path}>
                <button className="item" onClick={() => (r.kind === "project" ? openProjectFile(r.path) : openSource(r.path)).catch(errorToast)}>
                  <FileVideo size={16} className="faint" />
                  <span className="grow ellipsis">{r.title || basename(r.path)}<span className="tiny faint" style={{ display: "block" }}>{r.path}</span></span>
                  <span className="tiny faint">{r.kind === "project" ? "project · " : ""}{relTime(r.opened_at)}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export function Workspace() {
  const sourcePath = useStore((s) => s.sourcePath);
  const source = useStore((s) => s.source);
  const loading = useStore((s) => s.sourceLoading);
  const error = useStore((s) => s.sourceError);
  return (
    <div className="workspace">
      <section className="stage-area" aria-label="Preview">
        {sourcePath && (
          <div className="source-bar">
            <FileVideo size={16} className="faint" />
            <strong className="ellipsis" style={{ maxWidth: 320 }} title={sourcePath}>{basename(sourcePath)}</strong>
            <Facts />
            <span className="spacer" />
            <button className="btn ghost small" onClick={() => chooseVideo().catch(errorToast)}>Replace…</button>
            <button className="btn ghost icon small" aria-label="Close video" onClick={closeSource}><X size={15} /></button>
          </div>
        )}
        {!sourcePath ? (
          <Welcome />
        ) : loading ? (
          <div className="viewer"><div className="status"><Loader2 size={14} /> Reading video…</div></div>
        ) : error ? (
          <div className="viewer" style={{ background: "var(--bg)" }}>
            <div style={{ maxWidth: 520 }}>
              <Notice tone="err">
                <strong>This video could not be opened.</strong>
                <div className="small selectable">{error}</div>
              </Notice>
              <div className="row mt"><button className="btn" onClick={() => chooseVideo().catch(errorToast)}>Choose another file…</button></div>
            </div>
          </div>
        ) : (
          <>
            <Viewer />
            {source && <Transport />}
            <Timeline />
          </>
        )}
      </section>
      <Inspector />
    </div>
  );
}
