// Operation metadata shared by the simple task inspector and the pipeline editor.

import type { ModelCard, OperationType, Pipeline, Stage } from "./types";

export interface OpInfo {
  type: OperationType;
  label: string;
  task: string;
  description: string;
  /** Model catalog task(s) that can run this operation. */
  modelTasks: string[];
  /** Runs entirely in FFmpeg (no model download needed). */
  builtin?: boolean;
}

export const OPS: OpInfo[] = [
  { type: "deinterlace", label: "Deinterlace", task: "Fix interlacing", description: "Removes comb lines from interlaced footage (DV, broadcast, old camcorders) using FFmpeg's bwdif or yadif.", modelTasks: [], builtin: true },
  { type: "denoise", label: "Denoise", task: "Remove noise", description: "Reduces sensor noise and grain. FFDNet/DRUNet are AI models; hqdn3d and NL-means are fast built-in filters.", modelTasks: ["denoise"] },
  { type: "deblur", label: "Deblur", task: "Reduce motion blur", description: "NAFNet trained on GoPro motion blur. Works best on mild camera-shake blur; cannot recover severe or out-of-focus blur.", modelTasks: ["deblur"] },
  { type: "stabilize", label: "Stabilize", task: "Stabilize shaky video", description: "Two-pass vid.stab analysis and smoothing. Auto crop hides moving borders by zooming slightly.", modelTasks: [], builtin: true },
  { type: "enhance", label: "Enhance / Upscale", task: "Upscale & enhance", description: "AI super-resolution with Real-ESRGAN models. Choose an anime model for animation and a general model for live action.", modelTasks: ["super_resolution"] },
  { type: "face_restore", label: "Face recovery", task: "Recover faces", description: "Detects faces with RetinaFace and restores them with GFPGAN, blending back with the strength you choose. Can change identity at high strength.", modelTasks: ["face_restoration"] },
  { type: "interpolate", label: "Frame interpolation", task: "Smooth motion / slow motion", description: "Creates in-between frames with RIFE (AI) or FFmpeg motion interpolation for higher frame rates or slow motion.", modelTasks: ["frame_interpolation"] },
  { type: "color", label: "Color & light", task: "Color & light", description: "Exposure, white balance, shadows, highlights, contrast, saturation, vibrance and hue; also HDR to SDR tone mapping and SDR to HDR (PQ/HLG) conversion.", modelTasks: [], builtin: true },
  { type: "resize", label: "Resize", task: "Resize", description: "Classic scaling (Lanczos, bicubic…) to an exact size with fit, fill or stretch.", modelTasks: [], builtin: true },
  { type: "sharpen", label: "Sharpen", task: "Sharpen", description: "Adaptive sharpening brings out fine detail without halos; unsharp mask is stronger and has a radius. Runs at the final size.", modelTasks: [], builtin: true },
  { type: "grain", label: "Film grain", task: "Add film grain", description: "Adds natural-looking, luminance-aware grain. Useful after denoising or upscaling to avoid a plastic look.", modelTasks: [], builtin: true },
];

export const OP_ORDER: OperationType[] = OPS.map((o) => o.type);

export function opInfo(t: OperationType): OpInfo {
  return OPS.find((o) => o.type === t)!;
}

let counter = 0;
export function newStageId(type: OperationType, existing: Stage[]): string {
  let id = type as string;
  while (existing.some((s) => s.id === id)) id = `${type}-${++counter}`;
  return id;
}

/** Insert a stage at its canonical position (used by the simple task view). */
export function addStageCanonical(p: Pipeline, type: OperationType, settings: unknown): Pipeline {
  const stage: Stage = { id: newStageId(type, p.stages), enabled: true, type, settings: structuredClone(settings) };
  const rank = OP_ORDER.indexOf(type);
  const stages = [...p.stages];
  let at = stages.findIndex((s) => OP_ORDER.indexOf(s.type) > rank);
  if (at < 0) at = stages.length;
  stages.splice(at, 0, stage);
  return { stages };
}

export function moveStage(p: Pipeline, id: string, delta: number): Pipeline {
  const stages = [...p.stages];
  const i = stages.findIndex((s) => s.id === id);
  const j = i + delta;
  if (i < 0 || j < 0 || j >= stages.length) return p;
  [stages[i], stages[j]] = [stages[j], stages[i]];
  return { stages };
}

export function updateStage(p: Pipeline, id: string, f: (s: Stage) => Stage): Pipeline {
  return { stages: p.stages.map((s) => (s.id === id ? f(s) : s)) };
}

export function removeStage(p: Pipeline, id: string): Pipeline {
  return { stages: p.stages.filter((s) => s.id !== id) };
}

/** Models this pipeline needs (mirrors `Operation::required_models` in Rust). */
export function requiredModels(p: Pipeline): string[] {
  const out = new Set<string>();
  for (const s of p.stages) {
    if (!s.enabled) continue;
    const st = s.settings ?? {};
    switch (s.type) {
      case "denoise":
        if (st.engine === "ffdnet") out.add(st.model_id ?? "kair.ffdnet-color");
        if (st.engine === "drunet") out.add(st.model_id ?? "kair.drunet-color");
        break;
      case "deblur":
      case "enhance":
      case "face_restore":
        if (st.model_id) out.add(st.model_id);
        break;
      case "interpolate":
        if (st.engine === "rife" && st.model_id) out.add(st.model_id);
        break;
    }
  }
  return [...out].sort();
}

export function cardId(c: ModelCard): string {
  return c.manifest?.id ?? c.installed?.id ?? "";
}

export function cardName(c: ModelCard): string {
  return c.manifest?.display_name ?? c.installed?.custom_info?.display_name ?? cardId(c);
}

/** Models that can serve an operation (catalog + custom imports). */
export function modelsFor(type: OperationType, cards: ModelCard[]): ModelCard[] {
  const info = opInfo(type);
  return cards.filter((c) => {
    if (c.manifest) return c.manifest.tasks.some((t) => info.modelTasks.includes(t));
    // Custom imports are image-to-image upscalers.
    return type === "enhance" && !!c.installed?.custom;
  });
}

export function isInstalled(c: ModelCard | undefined): boolean {
  return !!c && (c.state === "installed" || c.state === "update_available");
}

export function stageSummary(s: Stage, cards: ModelCard[]): string {
  const st = s.settings ?? {};
  const model = (id?: string | null) => {
    const c = cards.find((x) => cardId(x) === id);
    return c ? shortName(c) : id ?? "";
  };
  switch (s.type) {
    case "deinterlace":
      return `${st.method} · ${st.rate === "field" ? "double rate" : "same rate"}`;
    case "denoise":
      return ["ffdnet", "drunet"].includes(st.engine) ? `${model(st.model_id ?? (st.engine === "ffdnet" ? "kair.ffdnet-color" : "kair.drunet-color"))} · σ ${st.strength}` : `${st.engine} · ${st.strength}`;
    case "deblur":
      return `${model(st.model_id)} · ${Math.round(st.strength * 100)}%`;
    case "enhance": {
      const t = st.target ?? {};
      const size = t.kind === "scale" ? `${t.factor}×` : t.kind === "height" ? `${t.height}p` : t.kind === "custom" ? `${t.width}×${t.height}` : t.kind === "source" ? "source size" : "model scale";
      return `${model(st.model_id)} · ${size}`;
    }
    case "face_restore":
      return `${model(st.model_id)} · ${Math.round((st.params?.strength ?? 0) * 100)}%`;
    case "interpolate": {
      const t = st.target ?? {};
      const target = t.kind === "multiply" ? `${t.factor}× frames` : `${(t.fps.num / t.fps.den).toFixed(2)} fps`;
      return `${st.engine.toUpperCase()} · ${target}${st.slow_motion > 1 ? ` · ${st.slow_motion}× slow motion` : ""}`;
    }
    case "stabilize":
      return `smoothing ${st.smoothing} · ${String(st.mode).replace("_", " ")}`;
    case "sharpen":
      return `${st.method === "adaptive" ? "adaptive" : "unsharp"} · ${Math.round(st.amount * 100)}%`;
    case "color":
      return st.mode === "adjust" ? "adjustments" : st.mode === "hdr_to_sdr" ? `HDR → SDR (${st.tone_mapper})` : `SDR → HDR (${String(st.hdr_transfer).toUpperCase()})`;
    case "resize":
      return `${st.width}×${st.height} · ${st.fit}`;
    case "grain":
      return `${st.preset} · ${Math.round(st.strength * 100)}%`;
  }
  return "";
}

/** Compact model name for dense lists ("Real-ESRGAN AnimeVideo v3" -> "AnimeVideo v3"). */
export function shortName(c: ModelCard): string {
  return cardName(c).replace(/^Real-ESRGAN\s+/, "").replace(/\s*\(.*\)$/, "").replace(/\s*\+ RetinaFace$/, "");
}

/** One-line use case shown next to a model. */
export function tagline(c: ModelCard): string {
  return c.manifest?.best_for[0] ?? (c.installed?.custom ? "Custom import" : "");
}

/** Section titles in the Adjustments panel. */
export const SECTION_TITLE: Record<OperationType, string> = {
  enhance: "Enhancement",
  denoise: "Denoise",
  deblur: "Motion deblur",
  face_restore: "Face recovery",
  interpolate: "Frame interpolation",
  stabilize: "Stabilization",
  deinterlace: "Deinterlace",
  color: "Color & light",
  sharpen: "Sharpen",
  resize: "Resize",
  grain: "Film grain",
};

export const SECTION_ORDER: OperationType[] = ["enhance", "denoise", "deblur", "face_restore", "interpolate", "stabilize", "deinterlace", "color", "resize", "sharpen", "grain"];
