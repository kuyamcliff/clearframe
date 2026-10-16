import { AlertTriangle, Check as CheckIcon, Download, Loader2 } from "lucide-react";

import { api } from "../lib/ipc";
import { bytes } from "../lib/format";
import { cardId, isInstalled, modelsFor, shortName, tagline } from "../lib/ops";
import type { ModelCard, OperationType, Stage } from "../lib/types";
import { errorToast, useStore } from "../state/store";
import { Check, F, NumberInput, Range, Select, Seg } from "./ui";

type S = Record<string, any>;

const NEUTRAL_COLOR = { exposure: 0, brightness: 0, contrast: 1, saturation: 1, gamma: 1, vibrance: 0, temperature: 0, tint: 0, hue: 0, shadows: 0, highlights: 0 };
const isNeutral = (s: S) => Object.entries(NEUTRAL_COLOR).every(([k, v]) => Math.abs((s[k] ?? v) - v) < 1e-4);

/** A built-in (non-model) choice shown in the same list as models. */
interface Builtin { value: string; name: string; tag: string }

export function installModel(id: string) {
  api.modelInstall(id).catch(errorToast);
}

/** Model list: name, italic use case, install state. Selecting a missing model never downloads it; the download icon does. */
export function ModelList({ type, value, onChange, builtins, filter, label = "AI model" }: {
  type: OperationType; value: string; onChange: (id: string) => void; builtins?: Builtin[]; filter?: (c: ModelCard) => boolean; label?: string;
}) {
  const models = useStore((s) => s.models);
  const installs = useStore((s) => s.installs);
  const list = modelsFor(type, models).filter((c) => !filter || filter(c));
  const status = (c: ModelCard) => {
    const id = cardId(c);
    const p = installs[id];
    if (p || c.in_progress) {
      const pct = p && p.bytes_total ? Math.round((p.bytes_done / p.bytes_total) * 100) : null;
      return <span className="s"><Loader2 size={11} className="spin" />{pct !== null ? `${pct}%` : ""}</span>;
    }
    if (isInstalled(c)) return id === value ? <span className="s"><CheckIcon size={12} /></span> : null;
    if (c.state === "missing" || c.state === "broken") return <span className="s bad" title={c.state === "missing" ? "Files missing" : "Damaged"}><AlertTriangle size={11} /></span>;
    return (
      <span className="s dl" role="button" tabIndex={0} title={`Install (${bytes(c.manifest?.download_size_bytes ?? 0)})`} aria-label={`Install ${shortName(c)}`}
        onClick={(e) => { e.stopPropagation(); installModel(id); }} onKeyDown={(e) => e.key === "Enter" && installModel(id)}>
        <Download size={11} />{bytes(c.manifest?.download_size_bytes ?? 0, 0)}
      </span>
    );
  };
  const sel = models.find((c) => cardId(c) === value);
  return (
    <F label={label} stack>
      <div className="models" role="radiogroup" aria-label={label}>
        {builtins?.map((b) => (
          <button key={b.value} type="button" className="mrow" role="radio" aria-checked={value === b.value} onClick={() => onChange(b.value)}>
            <span className="n">{b.name}</span><span className="d">{b.tag}</span>{value === b.value ? <span className="s"><CheckIcon size={12} /></span> : <span />}
          </button>
        ))}
        {list.map((c) => (
          <button key={cardId(c)} type="button" className="mrow" role="radio" aria-checked={cardId(c) === value} onClick={() => onChange(cardId(c))} title={c.manifest?.summary}>
            <span className="n">{shortName(c)}</span><span className="d">{tagline(c)}</span>{status(c) ?? <span />}
          </button>
        ))}
      </div>
      {sel && !isInstalled(sel) && !sel.in_progress && !installs[value] && sel.manifest && (
        <button className="btn sm primary" style={{ alignSelf: "flex-start" }} onClick={() => installModel(value)}>
          <Download size={12} /> {sel.state === "missing" || sel.state === "broken" ? "Reinstall" : "Install"} {shortName(sel)} · {bytes(sel.manifest.download_size_bytes)}
        </button>
      )}
    </F>
  );
}

const SIZE_PRESETS: { value: string; label: string }[] = [
  { value: "model_scale", label: "Model scale" },
  { value: "scale:2", label: "2× source" },
  { value: "height:1080", label: "1920×1080 (FHD)" },
  { value: "height:1440", label: "2560×1440 (QHD)" },
  { value: "height:2160", label: "3840×2160 (4K)" },
  { value: "height:4320", label: "7680×4320 (8K)" },
  { value: "source", label: "Same as source" },
  { value: "custom", label: "Custom…" },
];

function sizeKey(t: S): string {
  if (t.kind === "scale") return t.factor === 2 ? "scale:2" : "custom-scale";
  if (t.kind === "height") return [1080, 1440, 2160, 4320].includes(t.height) ? `height:${t.height}` : "custom";
  return t.kind;
}

export function StageForm({ stage, onChange }: { stage: Stage; onChange: (settings: S) => void }) {
  const s: S = stage.settings ?? {};
  const up = (patch: S) => onChange({ ...s, ...patch });
  const id = (k: string) => `${stage.id}-${k}`;
  const source = useStore((st) => st.source);
  const models = useStore((st) => st.models);
  const v = source?.info.video;
  const srcFps = v ? v.frame_rate.num / v.frame_rate.den : 30;

  switch (stage.type) {
    case "enhance": {
      const t = s.target ?? { kind: "model_scale" };
      const card = models.find((c) => cardId(c) === s.model_id);
      const scale = card?.manifest?.scale ?? 4;
      const key = sizeKey(t);
      return (
        <>
          <F label="Output size" htmlFor={id("size")}>
            <Select id={id("size")} value={key === "custom-scale" ? "custom" : key} onChange={(k) => {
              if (k === "model_scale" || k === "source") up({ target: { kind: k } });
              else if (k.startsWith("scale:")) up({ target: { kind: "scale", factor: Number(k.slice(6)) } });
              else if (k.startsWith("height:")) up({ target: { kind: "height", height: Number(k.slice(7)) } });
              else up({ target: { kind: "custom", width: v ? v.width * 2 : 3840, height: v ? v.height * 2 : 2160, fit: "fit" } });
            }} options={SIZE_PRESETS.map((o) => o.value === "model_scale" ? { ...o, label: `${scale}× (model)${v ? ` · ${v.width * scale}×${v.height * scale}` : ""}` } : o)} />
          </F>
          {t.kind === "custom" && (
            <F label="Size">
              <span className="row">
                <NumberInput label="Width" value={t.width} min={16} max={15360} onChange={(w) => up({ target: { ...t, width: Math.round(w) } })} />
                <span className="faint">×</span>
                <NumberInput label="Height" value={t.height} min={16} max={8640} onChange={(h) => up({ target: { ...t, height: Math.round(h) } })} />
              </span>
            </F>
          )}
          {t.kind === "custom" && (
            <F label="Fit"><Seg label="Fit" value={t.fit} onChange={(f) => up({ target: { ...t, fit: f } })} options={[{ value: "fit", label: "Fit" }, { value: "fill", label: "Fill" }, { value: "stretch", label: "Stretch" }]} /></F>
          )}
          {t.kind === "scale" && t.factor !== 2 && (
            <F label="Scale"><Range label="Scale" value={t.factor} min={0.25} max={8} step={0.25} onChange={(f) => up({ target: { ...t, factor: f } })} /></F>
          )}
          <ModelList type="enhance" value={s.model_id} onChange={(m) => up({ model_id: m, denoise: null })} />
          <F label="Strength"><Range label="Strength" value={s.strength} min={0} max={1} step={0.01} scale={100} onChange={(x) => up({ strength: x })} /></F>
          {card?.manifest?.supports_denoise_strength && (
            <F label="Reduce noise" title="Auto measures the video's noise: none for clean video (keeps texture), more for grainy video">
              <Check checked={s.denoise == null} label="Auto" onChange={(auto) => up({ denoise: auto ? null : useStore.getState().defaults?.general_denoise_default ?? 0.1 })} />
              {s.denoise != null && <Range label="Reduce noise" value={s.denoise} min={0} max={1} step={0.01} scale={100} onChange={(x) => up({ denoise: x })} />}
            </F>
          )}
          <F label="Tile size" title="0 = automatic">
            <NumberInput label="Tile size" value={s.tile_size} min={0} max={4096} step={32} onChange={(x) => up({ tile_size: Math.round(x) })} suffix={s.tile_size === 0 ? "auto" : "px"} />
          </F>
        </>
      );
    }
    case "denoise": {
      const ai = s.engine === "ffdnet" || s.engine === "drunet";
      const value = ai ? s.model_id ?? (s.engine === "ffdnet" ? "kair.ffdnet-color" : "kair.drunet-color") : `builtin:${s.engine}`;
      return (
        <>
          <ModelList type="denoise" label="Method" value={value}
            builtins={[{ value: "builtin:hqdn3d", name: "hqdn3d", tag: "Fast filter" }, { value: "builtin:nlmeans", name: "NL-means", tag: "Detail-preserving filter" }]}
            onChange={(m) => {
              if (m.startsWith("builtin:")) up({ engine: m.slice(8), model_id: null, strength: m === "builtin:hqdn3d" ? 4 : 6 });
              else {
                const c = models.find((x) => cardId(x) === m);
                up({ engine: c?.manifest?.architecture === "drunet" ? "drunet" : "ffdnet", model_id: m });
              }
            }} />
          <F label={ai ? "Noise level" : "Strength"}><Range label="Denoise strength" value={s.strength} min={0} max={ai ? 75 : 30} step={0.5} onChange={(x) => up({ strength: x })} /></F>
          <F label="Mix"><Range label="Mix" value={s.mix} min={0} max={1} step={0.01} scale={100} onChange={(x) => up({ mix: x })} /></F>
          {!ai && <F label="Temporal"><Check checked={!!s.temporal} onChange={(x) => up({ temporal: x })} label="Use neighbouring frames" /></F>}
        </>
      );
    }
    case "deblur":
      return (
        <>
          <ModelList type="deblur" value={s.model_id} onChange={(m) => up({ model_id: m })} />
          <F label="Strength"><Range label="Deblur strength" value={s.strength} min={0} max={1} step={0.01} scale={100} onChange={(x) => up({ strength: x })} /></F>
        </>
      );
    case "face_restore": {
      const p = s.params ?? {};
      const upP = (patch: S) => up({ params: { ...p, ...patch } });
      return (
        <>
          <ModelList type="face_restore" value={s.model_id} onChange={(m) => up({ model_id: m })} />
          <F label="Strength"><Range label="Face strength" value={p.strength} min={0} max={1} step={0.01} scale={100} onChange={(x) => upP({ strength: x })} /></F>
          <F label="Detection"><Range label="Detection confidence" value={p.detection_threshold} min={0.3} max={0.999} step={0.001} scale={100} onChange={(x) => upP({ detection_threshold: x })} /></F>
          <F label="Smoothing"><Range label="Temporal smoothing" value={p.temporal_smoothing} min={0} max={0.95} step={0.01} scale={100} onChange={(x) => upP({ temporal_smoothing: x })} /></F>
          <F label="Max faces"><NumberInput label="Maximum faces" value={p.max_faces} min={1} max={64} onChange={(x) => upP({ max_faces: Math.round(x) })} /></F>
          <F label="Detect at"><NumberInput label="Detection size" value={p.detection_max_side} min={320} max={4096} step={32} onChange={(x) => upP({ detection_max_side: Math.round(x) })} suffix="px" /></F>
        </>
      );
    }
    case "interpolate": {
      const t = s.target ?? { kind: "multiply", factor: 2 };
      const key = t.kind === "multiply" ? `x${t.factor}` : `${t.fps.num}/${t.fps.den}`;
      const rates = ["24/1", "25/1", "30000/1001", "30/1", "50/1", "60000/1001", "60/1", "120/1", "144/1", "240/1"];
      return (
        <>
          <F label="Frame rate" htmlFor={id("fps")}>
            <Select id={id("fps")} value={key} onChange={(k) => {
              if (k.startsWith("x")) up({ target: { kind: "multiply", factor: Number(k.slice(1)) } });
              else { const [n, d] = k.split("/").map(Number); up({ target: { kind: "fps", fps: { num: n, den: d } } }); }
            }} options={[
              ...[2, 3, 4, 8].map((f) => ({ value: `x${f}`, label: `${f}× · ${+(srcFps * f).toFixed(3)} fps` })),
              ...rates.map((r) => { const [n, d] = r.split("/").map(Number); return { value: r, label: `${+(n / d).toFixed(3)} fps` }; }),
            ]} />
          </F>
          <F label="Slow motion" htmlFor={id("sm")}>
            <Select id={id("sm")} value={String(s.slow_motion)} onChange={(x) => up({ slow_motion: Number(x) })} options={[1, 2, 4, 8, 16].map((x) => ({ value: String(x), label: x === 1 ? "None" : `${x}×` }))} />
          </F>
          {s.slow_motion > 1 && (
            <F label="Audio" htmlFor={id("sma")}>
              <Select id={id("sma")} value={s.slow_motion_audio} onChange={(x) => up({ slow_motion_audio: x })} options={[
                { value: "mute", label: "Remove" }, { value: "stretch_preserve_pitch", label: "Stretch, keep pitch" }, { value: "stretch_lower_pitch", label: "Stretch, lower pitch" },
              ]} />
            </F>
          )}
          <ModelList type="interpolate" value={s.engine === "rife" ? s.model_id : `builtin:${s.engine}`}
            builtins={[{ value: "builtin:mci", name: "Motion", tag: "Built-in motion interpolation" }, { value: "builtin:blend", name: "Blend", tag: "Frame blending" }]}
            onChange={(m) => (m.startsWith("builtin:") ? up({ engine: m.slice(8) }) : up({ engine: "rife", model_id: m }))} />
          <F label="Scene cuts"><Check checked={!!s.scene_detection} onChange={(x) => up({ scene_detection: x })} label="Don't blend across cuts" /></F>
          {s.scene_detection && <F label="Cut threshold"><Range label="Scene threshold" value={s.scene_threshold} min={0} max={1} step={0.01} scale={100} onChange={(x) => up({ scene_threshold: x })} /></F>}
          <F label="Duplicates"><Range label="Duplicate threshold" value={s.duplicate_threshold} min={0} max={0.2} step={0.005} scale={100} onChange={(x) => up({ duplicate_threshold: x })} /></F>
          {s.engine === "rife" && (
            <F label="Flow scale"><Seg label="RIFE flow scale" value={String(s.rife_scale)} onChange={(x) => up({ rife_scale: Number(x) })} options={["0.25", "0.5", "1", "2"].map((x) => ({ value: x, label: x }))} /></F>
          )}
        </>
      );
    }
    case "stabilize":
      return (
        <>
          <F label="Smoothness"><Range label="Smoothness" value={s.smoothing} min={0} max={120} onChange={(x) => up({ smoothing: Math.round(x) })} /></F>
          <F label="Shakiness"><Range label="Shakiness" value={s.shakiness} min={1} max={10} onChange={(x) => up({ shakiness: Math.round(x) })} /></F>
          <F label="Edges"><Seg label="Edges" value={s.mode} onChange={(x) => up({ mode: x })} options={[{ value: "auto_crop", label: "Auto crop" }, { value: "full_frame", label: "Full frame" }, { value: "black", label: "Black" }]} /></F>
          <F label="Tripod"><Check checked={!!s.tripod} onChange={(x) => up({ tripod: x })} label="Lock to first frame" /></F>
        </>
      );
    case "deinterlace":
      return (
        <>
          <F label="Method"><Seg label="Method" value={s.method} onChange={(x) => up({ method: x })} options={[{ value: "bwdif", label: "bwdif" }, { value: "yadif", label: "yadif" }]} /></F>
          <F label="Output"><Seg label="Output rate" value={s.rate} onChange={(x) => up({ rate: x })} options={[{ value: "frame", label: "Same rate" }, { value: "field", label: "Double rate" }]} /></F>
          <F label="Field order"><Seg label="Field order" value={s.field_order} onChange={(x) => up({ field_order: x })} options={[{ value: "auto", label: "Auto" }, { value: "tff", label: "Top" }, { value: "bff", label: "Bottom" }]} /></F>
          <F label="Flagged only"><Check checked={!!s.only_flagged} onChange={(x) => up({ only_flagged: x })} label="Only flagged frames" /></F>
        </>
      );
    case "color":
      return (
        <>
          <F label="Mode"><Seg label="Color mode" value={s.mode} onChange={(x) => up({ mode: x })} options={[{ value: "adjust", label: "Grade" }, { value: "hdr_to_sdr", label: "HDR→SDR" }, { value: "sdr_to_hdr", label: "SDR→HDR" }]} /></F>
          <div className="subhead"><span>Light</span>
            <button type="button" className="link" onClick={() => up(NEUTRAL_COLOR)} disabled={isNeutral(s)}>Reset</button>
          </div>
          <F label="Exposure"><Range label="Exposure" value={s.exposure ?? 0} min={-3} max={3} step={0.05} reset={0} onChange={(x) => up({ exposure: x })} /></F>
          <F label="Brightness"><Range label="Brightness" value={s.brightness} min={-1} max={1} step={0.01} scale={100} reset={0} onChange={(x) => up({ brightness: x })} /></F>
          <F label="Contrast"><Range label="Contrast" value={s.contrast} min={0} max={3} step={0.01} reset={1} onChange={(x) => up({ contrast: x })} /></F>
          <F label="Shadows"><Range label="Shadows" value={s.shadows ?? 0} min={-1} max={1} step={0.01} scale={100} reset={0} onChange={(x) => up({ shadows: x })} /></F>
          <F label="Highlights"><Range label="Highlights" value={s.highlights ?? 0} min={-1} max={1} step={0.01} scale={100} reset={0} onChange={(x) => up({ highlights: x })} /></F>
          <F label="Gamma"><Range label="Gamma" value={s.gamma} min={0.1} max={5} step={0.01} reset={1} onChange={(x) => up({ gamma: x })} /></F>
          <div className="subhead"><span>Color</span></div>
          <F label="Temperature"><Range label="Temperature" value={s.temperature ?? 0} min={-1} max={1} step={0.01} scale={100} reset={0} onChange={(x) => up({ temperature: x })} /></F>
          <F label="Tint"><Range label="Tint" value={s.tint ?? 0} min={-1} max={1} step={0.01} scale={100} reset={0} onChange={(x) => up({ tint: x })} /></F>
          <F label="Saturation"><Range label="Saturation" value={s.saturation} min={0} max={3} step={0.01} reset={1} onChange={(x) => up({ saturation: x })} /></F>
          <F label="Vibrance"><Range label="Vibrance" value={s.vibrance ?? 0} min={-1} max={1} step={0.01} scale={100} reset={0} onChange={(x) => up({ vibrance: x })} /></F>
          <F label="Hue"><Range label="Hue" value={s.hue ?? 0} min={-180} max={180} step={1} reset={0} onChange={(x) => up({ hue: x })} /></F>
          {s.mode === "hdr_to_sdr" && (
            <>
              <F label="Tone map" htmlFor={id("tm")}>
                <Select id={id("tm")} value={s.tone_mapper} onChange={(x) => up({ tone_mapper: x })} options={["hable", "mobius", "reinhard", "clip"].map((x) => ({ value: x, label: x[0].toUpperCase() + x.slice(1) }))} />
              </F>
              <F label="Desaturate"><Range label="Desaturate" value={s.desaturate} min={0} max={1} step={0.01} scale={100} onChange={(x) => up({ desaturate: x })} /></F>
            </>
          )}
          {s.mode === "sdr_to_hdr" && (
            <>
              <F label="Format"><Seg label="HDR format" value={s.hdr_transfer} onChange={(x) => up({ hdr_transfer: x })} options={[{ value: "pq", label: "HDR10 (PQ)" }, { value: "hlg", label: "HLG" }]} /></F>
              <F label="SDR white"><NumberInput label="SDR white" value={s.sdr_white_nits} min={80} max={400} onChange={(x) => up({ sdr_white_nits: x })} suffix="nits" /></F>
              <F label="Peak"><NumberInput label="Peak" value={s.peak_nits} min={400} max={10000} step={100} onChange={(x) => up({ peak_nits: x })} suffix="nits" /></F>
            </>
          )}
        </>
      );
    case "sharpen":
      return (
        <>
          <F label="Method"><Seg label="Sharpen method" value={s.method} onChange={(x) => up({ method: x, amount: x === "adaptive" ? Math.min(s.amount, 1) : s.amount })}
            options={[{ value: "adaptive", label: "Adaptive", title: "Brings out fine detail without halos" }, { value: "unsharp", label: "Unsharp mask", title: "Stronger, with a radius" }]} /></F>
          <F label="Amount"><Range label="Sharpen amount" value={s.amount} min={0} max={s.method === "adaptive" ? 1 : 3} step={0.01} scale={100} reset={0.4} onChange={(x) => up({ amount: x })} /></F>
          {s.method === "unsharp" && <F label="Radius"><Range label="Sharpen radius" value={s.radius} min={0.5} max={5} step={0.1} reset={1.5} onChange={(x) => up({ radius: x })} /></F>}
        </>
      );
    case "resize":
      return (
        <>
          <F label="Size">
            <span className="row">
              <NumberInput label="Width" value={s.width} min={16} max={15360} onChange={(x) => up({ width: Math.round(x) })} />
              <span className="faint">×</span>
              <NumberInput label="Height" value={s.height} min={16} max={8640} onChange={(x) => up({ height: Math.round(x) })} />
            </span>
          </F>
          <F label="Fit"><Seg label="Fit" value={s.fit} onChange={(x) => up({ fit: x })} options={[{ value: "fit", label: "Fit" }, { value: "fill", label: "Fill" }, { value: "stretch", label: "Stretch" }]} /></F>
          <F label="Filter" htmlFor={id("flt")}>
            <Select id={id("flt")} value={s.filter} onChange={(x) => up({ filter: x })} options={["lanczos", "bicubic", "bilinear", "spline"].map((x) => ({ value: x, label: x[0].toUpperCase() + x.slice(1) }))} />
          </F>
        </>
      );
    case "grain": {
      const presets = useStore.getState().defaults?.grain_presets as S[] | undefined;
      return (
        <>
          <F label="Look" htmlFor={id("look")}>
            <Select id={id("look")} value={s.preset} onChange={(x) => {
              const p = presets?.find((g) => g.preset === x);
              up(p ? { ...p, seed: s.seed } : { preset: x });
            }} options={["fine", "film", "coarse", "monochrome", "color", "custom"].map((x) => ({ value: x, label: x[0].toUpperCase() + x.slice(1) }))} />
          </F>
          <F label="Amount"><Range label="Grain amount" value={s.strength} min={0} max={1} step={0.01} scale={100} onChange={(x) => up({ strength: x, preset: "custom" })} /></F>
          <F label="Size"><Range label="Grain size" value={s.size} min={0.5} max={4} step={0.1} onChange={(x) => up({ size: x, preset: "custom" })} /></F>
          <F label="Color"><Range label="Grain color" value={s.color} min={0} max={1} step={0.01} scale={100} onChange={(x) => up({ color: x, preset: "custom" })} /></F>
          <F label="Animated"><Check checked={!!s.animated} onChange={(x) => up({ animated: x })} label="Changes every frame" /></F>
          <F label="Shadows"><Check checked={!!s.luma_response} onChange={(x) => up({ luma_response: x })} label="Luminance response" /></F>
          <F label="Seed"><NumberInput label="Seed" value={s.seed} min={0} max={4294967295} onChange={(x) => up({ seed: Math.round(x) })} /></F>
        </>
      );
    }
  }
  return null;
}
