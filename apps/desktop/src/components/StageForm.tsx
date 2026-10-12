import { Download } from "lucide-react";

import { api } from "../lib/ipc";
import { bytes } from "../lib/format";
import { cardId, cardName, isInstalled, modelsFor } from "../lib/ops";
import type { ModelCard, OperationType, Stage } from "../lib/types";
import { errorToast, toast, useStore } from "../state/store";
import { Check, Field, Notice, NumberInput, Select, Slider } from "./ui";

type Settings = Record<string, any>;

interface Props {
  stage: Stage;
  onChange: (settings: Settings) => void;
  advanced: boolean;
}

export function ModelPicker({ type, value, onChange, filter }: { type: OperationType; value: string; onChange: (id: string) => void; filter?: (c: ModelCard) => boolean }) {
  const models = useStore((s) => s.models);
  const installs = useStore((s) => s.installs);
  const options = modelsFor(type, models).filter((c) => !filter || filter(c));
  const current = models.find((c) => cardId(c) === value);
  const prog = installs[value];
  const install = async () => {
    try {
      await api.modelInstall(value);
      toast("info", `Installing ${current ? cardName(current) : value}…`);
    } catch (e) {
      errorToast(e);
    }
  };
  return (
    <Field label="Model" htmlFor={`model-${type}`}>
      <Select id={`model-${type}`} label="Model" value={value} onChange={onChange}
        options={[
          ...options.map((c) => ({ value: cardId(c), label: `${cardName(c)}${isInstalled(c) ? "" : " — not installed"}` })),
          ...(current || !value ? [] : [{ value, label: `${value} — unknown model` }]),
        ]} />
      {current?.manifest && <div className="hint">{current.manifest.summary}</div>}
      {current && !isInstalled(current) && (
        <div className="row mt" style={{ marginTop: 6 }}>
          {current.in_progress || prog ? (
            <span className="small muted">
              Installing… {prog && prog.bytes_total ? `${bytes(prog.bytes_done)} of ${bytes(prog.bytes_total)}` : prog?.message}
            </span>
          ) : current.state === "missing" || current.state === "broken" ? (
            <Notice tone="warn">This model's files are {current.state === "missing" ? "missing" : "damaged"}. Repair it in the Model Library.</Notice>
          ) : current.manifest ? (
            <button className="btn small" onClick={install}>
              <Download size={14} /> Install ({bytes(current.manifest.download_size_bytes)})
            </button>
          ) : null}
          <button className="btn ghost small" onClick={() => useStore.setState({ view: "models" })}>Model Library</button>
        </div>
      )}
    </Field>
  );
}

const pctFmt = (v: number) => `${Math.round(v * 100)}%`;

export function StageForm({ stage, onChange, advanced }: Props) {
  const s: Settings = stage.settings ?? {};
  const up = (patch: Settings) => onChange({ ...s, ...patch });
  const id = (k: string) => `${stage.id}-${k}`;
  const source = useStore((st) => st.source);
  const srcFps = source?.info.video ? source.info.video.frame_rate.num / source.info.video.frame_rate.den : 30;

  switch (stage.type) {
    case "deinterlace":
      return (
        <>
          <Field label="Method" htmlFor={id("m")} help="bwdif gives cleaner motion; yadif is faster on old machines.">
            <Select id={id("m")} value={s.method} onChange={(v) => up({ method: v })} options={[{ value: "bwdif", label: "bwdif (recommended)" }, { value: "yadif", label: "yadif" }]} />
          </Field>
          <Field label="Output rate" htmlFor={id("r")} help="Double rate keeps every field as a frame (e.g. 25i → 50p) for the smoothest motion.">
            <Select id={id("r")} value={s.rate} onChange={(v) => up({ rate: v })} options={[{ value: "frame", label: "Same frame rate" }, { value: "field", label: "Double rate (one frame per field)" }]} />
          </Field>
          {advanced && (
            <>
              <Field label="Field order" htmlFor={id("fo")}>
                <Select id={id("fo")} value={s.field_order} onChange={(v) => up({ field_order: v })} options={[{ value: "auto", label: "Automatic" }, { value: "tff", label: "Top field first" }, { value: "bff", label: "Bottom field first" }]} />
              </Field>
              <Check checked={!!s.only_flagged} onChange={(v) => up({ only_flagged: v })} label="Only frames flagged as interlaced" />
            </>
          )}
        </>
      );
    case "denoise": {
      const ai = s.engine === "ffdnet" || s.engine === "drunet";
      return (
        <>
          <Field label="Engine" htmlFor={id("e")}>
            <Select id={id("e")} value={s.engine} onChange={(v) => up({ engine: v, model_id: null })} options={[
              { value: "ffdnet", label: "FFDNet (AI, fast)" }, { value: "drunet", label: "DRUNet (AI, best quality, slow)" },
              { value: "hqdn3d", label: "hqdn3d (built-in, very fast)" }, { value: "nlmeans", label: "NL-means (built-in)" },
            ]} />
          </Field>
          {ai && <ModelPicker type="denoise" value={s.model_id ?? (s.engine === "ffdnet" ? "kair.ffdnet-color" : "kair.drunet-color")} onChange={(v) => up({ model_id: v })}
            filter={(c) => c.manifest?.architecture === s.engine} />}
          <Field label="Strength" help={ai ? "Noise level σ the model removes (0–75). 5–15 suits typical camera noise; higher values smooth detail." : "Filter strength. Higher removes more noise and more detail."}>
            <Slider label="Strength" value={s.strength} min={0} max={ai ? 75 : 30} step={0.5} onChange={(v) => up({ strength: v })} />
          </Field>
          <Field label="Mix with original" help="Blend some of the original back in to keep texture.">
            <Slider label="Mix" value={s.mix} min={0} max={1} step={0.05} format={pctFmt} onChange={(v) => up({ mix: v })} />
          </Field>
          {s.engine !== "drunet" && s.engine !== "ffdnet" && advanced && (
            <Check checked={!!s.temporal} onChange={(v) => up({ temporal: v })} label="Temporal (uses neighbouring frames)" />
          )}
        </>
      );
    }
    case "deblur":
      return (
        <>
          <ModelPicker type="deblur" value={s.model_id} onChange={(v) => up({ model_id: v })} />
          <Field label="Strength">
            <Slider label="Deblur strength" value={s.strength} min={0} max={1} step={0.05} format={pctFmt} onChange={(v) => up({ strength: v })} />
          </Field>
        </>
      );
    case "enhance": {
      const t = s.target ?? { kind: "model_scale" };
      const models = useStore.getState().models;
      const card = models.find((c) => cardId(c) === s.model_id);
      return (
        <>
          <ModelPicker type="enhance" value={s.model_id} onChange={(v) => up({ model_id: v, denoise: null })} />
          <Field label="Output size" htmlFor={id("t")} help="Models upscale by their native factor; other sizes are reached by resizing the model output with Lanczos.">
            <Select id={id("t")} value={t.kind} onChange={(k) => {
              const nt = k === "scale" ? { kind: "scale", factor: 2 } : k === "height" ? { kind: "height", height: 2160 } : k === "custom" ? { kind: "custom", width: 3840, height: 2160, fit: "fit" } : { kind: k };
              up({ target: nt });
            }} options={[
              { value: "model_scale", label: `Model's native scale${card?.manifest ? ` (${card.manifest.scale}×)` : ""}` },
              { value: "scale", label: "Scale factor" }, { value: "height", label: "Target height (e.g. 1080p, 4K)" },
              { value: "custom", label: "Exact size" }, { value: "source", label: "Keep source size (enhance only)" },
            ]} />
          </Field>
          {t.kind === "scale" && (
            <Field label="Scale factor">
              <Slider label="Scale factor" value={t.factor} min={0.25} max={8} step={0.25} onChange={(v) => up({ target: { ...t, factor: v } })} />
            </Field>
          )}
          {t.kind === "height" && (
            <Field label="Height" htmlFor={id("h")}>
              <Select id={id("h")} value={String(t.height)} onChange={(v) => up({ target: { ...t, height: Number(v) } })}
                options={[480, 720, 1080, 1440, 2160, 4320].map((h) => ({ value: String(h), label: h === 2160 ? "2160p (4K UHD)" : h === 4320 ? "4320p (8K)" : `${h}p` }))} />
            </Field>
          )}
          {t.kind === "custom" && (
            <div className="row wrap">
              <Field label="Width"><NumberInput label="Width" value={t.width} min={16} max={15360} onChange={(v) => up({ target: { ...t, width: Math.round(v) } })} /></Field>
              <Field label="Height"><NumberInput label="Height" value={t.height} min={16} max={8640} onChange={(v) => up({ target: { ...t, height: Math.round(v) } })} /></Field>
              <Field label="Fit" htmlFor={id("fit")}>
                <Select id={id("fit")} value={t.fit} onChange={(v) => up({ target: { ...t, fit: v } })} options={[{ value: "fit", label: "Fit (pad)" }, { value: "fill", label: "Fill (crop)" }, { value: "stretch", label: "Stretch" }]} />
              </Field>
            </div>
          )}
          <Field label="Strength" help="Blend between a plain Lanczos upscale (0%) and the full model result (100%).">
            <Slider label="Enhance strength" value={s.strength} min={0} max={1} step={0.05} format={pctFmt} onChange={(v) => up({ strength: v })} />
          </Field>
          {card?.manifest?.supports_denoise_strength && (
            <Field label="Denoise amount" help="This model can mix its denoising variant: 0 keeps more grain and detail, 1 removes more noise.">
              <Slider label="Denoise amount" value={s.denoise ?? 0.5} min={0} max={1} step={0.05} format={pctFmt} onChange={(v) => up({ denoise: v })} />
            </Field>
          )}
          {advanced && (
            <div className="row wrap">
              <Field label="Tile size" help="0 = automatic from the memory budget. Smaller tiles use less memory but are slower.">
                <NumberInput label="Tile size" value={s.tile_size} min={0} max={4096} step={32} onChange={(v) => up({ tile_size: Math.round(v) })} suffix="px" />
              </Field>
              <Field label="Tile padding" help="0 = automatic. Overlap between tiles to hide seams.">
                <NumberInput label="Tile padding" value={s.tile_pad} min={0} max={256} step={2} onChange={(v) => up({ tile_pad: Math.round(v) })} suffix="px" />
              </Field>
            </div>
          )}
        </>
      );
    }
    case "face_restore": {
      const p = s.params ?? {};
      const upP = (patch: Settings) => up({ params: { ...p, ...patch } });
      return (
        <>
          <ModelPicker type="face_restore" value={s.model_id} onChange={(v) => up({ model_id: v })} />
          <Notice tone="warn">Face recovery can change how people look at high strength. Compare carefully before exporting.</Notice>
          <Field label="Strength" help="How much of the restored face replaces the original.">
            <Slider label="Face strength" value={p.strength} min={0} max={1} step={0.05} format={pctFmt} onChange={(v) => upP({ strength: v })} />
          </Field>
          {advanced && (
            <>
              <Field label="Detection confidence" help="Higher values ignore uncertain detections (fewer false faces).">
                <Slider label="Detection confidence" value={p.detection_threshold} min={0.3} max={0.999} step={0.01} onChange={(v) => upP({ detection_threshold: v })} />
              </Field>
              <Field label="Maximum faces per frame">
                <NumberInput label="Maximum faces" value={p.max_faces} min={1} max={64} onChange={(v) => upP({ max_faces: Math.round(v) })} />
              </Field>
              <Field label="Temporal smoothing" help="Smooths face positions between frames to reduce jitter.">
                <Slider label="Temporal smoothing" value={p.temporal_smoothing} min={0} max={0.95} step={0.05} onChange={(v) => upP({ temporal_smoothing: v })} />
              </Field>
              <Field label="Detection size" help="Frames are downscaled to this longest side for detection. Larger finds small faces but is slower.">
                <NumberInput label="Detection size" value={p.detection_max_side} min={320} max={4096} step={32} onChange={(v) => upP({ detection_max_side: Math.round(v) })} suffix="px" />
              </Field>
            </>
          )}
        </>
      );
    }
    case "interpolate": {
      const t = s.target ?? { kind: "multiply", factor: 2 };
      const outFps = t.kind === "multiply" ? srcFps * t.factor : t.fps.num / t.fps.den;
      return (
        <>
          <Field label="Engine" htmlFor={id("e")} help="RIFE is an AI model with the best quality. MCI is FFmpeg motion-compensated interpolation; Blend cross-fades frames.">
            <Select id={id("e")} value={s.engine} onChange={(v) => up({ engine: v })} options={[{ value: "rife", label: "RIFE (AI)" }, { value: "mci", label: "Motion interpolation (built-in)" }, { value: "blend", label: "Frame blending (built-in)" }]} />
          </Field>
          {s.engine === "rife" && <ModelPicker type="interpolate" value={s.model_id} onChange={(v) => up({ model_id: v })} />}
          <Field label="Target" htmlFor={id("t")}>
            <Select id={id("t")} value={t.kind === "multiply" ? `x${t.factor}` : "fps"} onChange={(v) => {
              if (v === "fps") up({ target: { kind: "fps", fps: { num: 60, den: 1 } } });
              else up({ target: { kind: "multiply", factor: Number(v.slice(1)) } });
            }} options={[
              { value: "x2", label: "2× frames" }, { value: "x3", label: "3× frames" }, { value: "x4", label: "4× frames" }, { value: "x8", label: "8× frames" },
              { value: "fps", label: "Specific frame rate" },
            ]} />
          </Field>
          {t.kind === "fps" && (
            <Field label="Frame rate" htmlFor={id("fps")}>
              <Select id={id("fps")} value={`${t.fps.num}/${t.fps.den}`} onChange={(v) => {
                const [n, d] = v.split("/").map(Number);
                up({ target: { kind: "fps", fps: { num: n, den: d } } });
              }} options={["24/1", "25/1", "30000/1001", "30/1", "48/1", "50/1", "60000/1001", "60/1", "100/1", "120/1", "144/1", "240/1"].map((r) => {
                const [n, d] = r.split("/").map(Number);
                return { value: r, label: `${+(n / d).toFixed(3)} fps` };
              })} />
            </Field>
          )}
          <Field label="Slow motion" help="Plays the result slower: 2× slow motion of 30 fps footage at 2× frames keeps 30 fps playback but lasts twice as long.">
            <Select id={id("sm")} label="Slow motion" value={String(s.slow_motion)} onChange={(v) => up({ slow_motion: Number(v) })}
              options={[1, 2, 4, 8, 16].map((x) => ({ value: String(x), label: x === 1 ? "Off (smoother motion)" : `${x}× slower` }))} />
          </Field>
          {s.slow_motion > 1 && (
            <Field label="Audio in slow motion" htmlFor={id("sma")}>
              <Select id={id("sma")} value={s.slow_motion_audio} onChange={(v) => up({ slow_motion_audio: v })} options={[
                { value: "mute", label: "Remove audio" }, { value: "stretch_preserve_pitch", label: "Stretch, keep pitch" }, { value: "stretch_lower_pitch", label: "Stretch, lower pitch" },
              ]} />
            </Field>
          )}
          <div className="hint small faint">Output: {(outFps / Math.max(1, s.slow_motion)).toFixed(3).replace(/\.?0+$/, "")} fps playback{s.slow_motion > 1 ? `, ${s.slow_motion}× longer` : ""}.</div>
          <Check checked={!!s.scene_detection} onChange={(v) => up({ scene_detection: v })} label="Don't interpolate across scene cuts" desc="Duplicates frames at cuts instead of morphing between shots." />
          {advanced && (
            <>
              {s.scene_detection && (
                <Field label="Scene cut sensitivity" help="Lower values detect more cuts.">
                  <Slider label="Scene threshold" value={s.scene_threshold} min={0} max={1} step={0.01} onChange={(v) => up({ scene_threshold: v })} />
                </Field>
              )}
              <Field label="Duplicate frame threshold" help="Treat nearly identical frames as duplicates (0 = off). Helps with animation on twos.">
                <Slider label="Duplicate threshold" value={s.duplicate_threshold} min={0} max={0.2} step={0.005} onChange={(v) => up({ duplicate_threshold: v })} />
              </Field>
              {s.engine === "rife" && (
                <Field label="RIFE flow scale" help="0.5 suits 4K input (faster, larger motion); 2 can help small, detailed motion at low resolution.">
                  <Select id={id("rs")} label="RIFE flow scale" value={String(s.rife_scale)} onChange={(v) => up({ rife_scale: Number(v) })} options={["0.25", "0.5", "1", "2"].map((x) => ({ value: x, label: x }))} />
                </Field>
              )}
            </>
          )}
        </>
      );
    }
    case "stabilize":
      return (
        <>
          <Field label="Smoothing" help="Frames used to smooth the camera path. Higher is steadier but crops more.">
            <Slider label="Smoothing" value={s.smoothing} min={0} max={120} onChange={(v) => up({ smoothing: Math.round(v) })} />
          </Field>
          <Field label="Borders" htmlFor={id("b")}>
            <Select id={id("b")} value={s.mode} onChange={(v) => up({ mode: v })} options={[{ value: "auto_crop", label: "Auto crop (zoom to hide borders)" }, { value: "black", label: "Black borders" }, { value: "full_frame", label: "Keep full frame (fill edges)" }]} />
          </Field>
          {advanced && (
            <>
              <Field label="Shakiness" help="How shaky the source is (1–10). Higher analyses larger motion.">
                <Slider label="Shakiness" value={s.shakiness} min={1} max={10} onChange={(v) => up({ shakiness: Math.round(v) })} />
              </Field>
              <Check checked={!!s.tripod} onChange={(v) => up({ tripod: v })} label="Tripod mode (lock to the first frame)" />
            </>
          )}
        </>
      );
    case "color":
      return (
        <>
          <Field label="Mode" htmlFor={id("m")}>
            <Select id={id("m")} value={s.mode} onChange={(v) => up({ mode: v })} options={[{ value: "adjust", label: "Adjust colors" }, { value: "hdr_to_sdr", label: "HDR → SDR (tone map)" }, { value: "sdr_to_hdr", label: "SDR → HDR" }]} />
          </Field>
          {s.mode === "sdr_to_hdr" && <Notice tone="info">SDR → HDR re-encodes the same picture into an HDR container. It does not invent highlight detail.</Notice>}
          <Field label="Brightness"><Slider label="Brightness" value={s.brightness} min={-1} max={1} step={0.01} onChange={(v) => up({ brightness: v })} /></Field>
          <Field label="Contrast"><Slider label="Contrast" value={s.contrast} min={0} max={3} step={0.01} onChange={(v) => up({ contrast: v })} /></Field>
          <Field label="Saturation"><Slider label="Saturation" value={s.saturation} min={0} max={3} step={0.01} onChange={(v) => up({ saturation: v })} /></Field>
          {advanced && <Field label="Gamma"><Slider label="Gamma" value={s.gamma} min={0.1} max={5} step={0.01} onChange={(v) => up({ gamma: v })} /></Field>}
          {s.mode === "hdr_to_sdr" && (
            <>
              <Field label="Tone mapper" htmlFor={id("tm")} help="Hable keeps highlight detail; Mobius keeps colors closer; Reinhard is softer; Clip simply cuts highlights.">
                <Select id={id("tm")} value={s.tone_mapper} onChange={(v) => up({ tone_mapper: v })} options={["hable", "mobius", "reinhard", "clip"].map((x) => ({ value: x, label: x[0].toUpperCase() + x.slice(1) }))} />
              </Field>
              {advanced && <Field label="Highlight desaturation"><Slider label="Desaturate" value={s.desaturate} min={0} max={1} step={0.05} onChange={(v) => up({ desaturate: v })} /></Field>}
            </>
          )}
          {s.mode === "sdr_to_hdr" && (
            <>
              <Field label="HDR format" htmlFor={id("ht")}>
                <Select id={id("ht")} value={s.hdr_transfer} onChange={(v) => up({ hdr_transfer: v })} options={[{ value: "pq", label: "HDR10 (PQ)" }, { value: "hlg", label: "HLG" }]} />
              </Field>
              {advanced && (
                <>
                  <Field label="SDR white level" help="Brightness that SDR white maps to (203 nits is the ITU recommendation)."><NumberInput label="SDR white" value={s.sdr_white_nits} min={80} max={400} onChange={(v) => up({ sdr_white_nits: v })} suffix="nits" /></Field>
                  <Field label="Mastering peak"><NumberInput label="Peak" value={s.peak_nits} min={400} max={10000} step={100} onChange={(v) => up({ peak_nits: v })} suffix="nits" /></Field>
                </>
              )}
            </>
          )}
        </>
      );
    case "resize":
      return (
        <>
          <div className="row wrap">
            <Field label="Width"><NumberInput label="Width" value={s.width} min={16} max={15360} onChange={(v) => up({ width: Math.round(v) })} /></Field>
            <Field label="Height"><NumberInput label="Height" value={s.height} min={16} max={8640} onChange={(v) => up({ height: Math.round(v) })} /></Field>
          </div>
          <Field label="Fit" htmlFor={id("fit")}>
            <Select id={id("fit")} value={s.fit} onChange={(v) => up({ fit: v })} options={[{ value: "fit", label: "Fit inside (pad)" }, { value: "fill", label: "Fill (crop)" }, { value: "stretch", label: "Stretch" }]} />
          </Field>
          <Field label="Filter" htmlFor={id("f")}>
            <Select id={id("f")} value={s.filter} onChange={(v) => up({ filter: v })} options={["lanczos", "bicubic", "bilinear", "spline"].map((x) => ({ value: x, label: x[0].toUpperCase() + x.slice(1) }))} />
          </Field>
        </>
      );
    case "grain": {
      const presets = useStore.getState().defaults?.grain_presets as Settings[] | undefined;
      return (
        <>
          <Field label="Look" htmlFor={id("p")}>
            <Select id={id("p")} value={s.preset} onChange={(v) => {
              const p = presets?.find((x) => x.preset === v);
              up(p ? { ...p, seed: s.seed } : { preset: v });
            }} options={["fine", "film", "coarse", "monochrome", "color", "custom"].map((x) => ({ value: x, label: x[0].toUpperCase() + x.slice(1) }))} />
          </Field>
          <Field label="Amount"><Slider label="Grain amount" value={s.strength} min={0} max={1} step={0.01} format={pctFmt} onChange={(v) => up({ strength: v, preset: "custom" })} /></Field>
          <Field label="Grain size"><Slider label="Grain size" value={s.size} min={0.5} max={4} step={0.1} onChange={(v) => up({ size: v, preset: "custom" })} /></Field>
          {advanced && (
            <>
              <Field label="Color amount"><Slider label="Grain color" value={s.color} min={0} max={1} step={0.05} onChange={(v) => up({ color: v, preset: "custom" })} /></Field>
              <Check checked={!!s.animated} onChange={(v) => up({ animated: v })} label="Animated (changes every frame)" />
              <Check checked={!!s.luma_response} onChange={(v) => up({ luma_response: v })} label="Less grain in highlights and shadows" />
              <Field label="Seed" help="Same seed gives the same grain pattern every export."><NumberInput label="Seed" value={s.seed} min={0} max={4294967295} onChange={(v) => up({ seed: Math.round(v) })} /></Field>
            </>
          )}
        </>
      );
    }
  }
  return null;
}
