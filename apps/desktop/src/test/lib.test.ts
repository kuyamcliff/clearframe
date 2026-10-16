import { describe, expect, it } from "vitest";

import { basename, bytes, dirname, eta, fps, mb, stem, timecode } from "../lib/format";
import { addStageCanonical, moveStage, opInfo, OPS, removeStage, requiredModels, stageSummary } from "../lib/ops";
import type { ModelCard, Pipeline, Settings } from "../lib/types";
import defaults from "../lib/fixtures/operation_defaults.json";
import settings from "../lib/fixtures/settings.json";
import models from "../lib/fixtures/models.json";
import presets from "../lib/fixtures/presets.json";

const D = defaults as any;

describe("format", () => {
  it("formats sizes and times", () => {
    expect(bytes(0)).toBe("0 B");
    expect(bytes(2_504_012)).toBe("2.5 MB");
    expect(bytes(null)).toBe("—");
    expect(mb(512)).toBe("512 MB");
    expect(mb(2048)).toBe("2.0 GB");
    expect(timecode(83.4)).toBe("1:23.40");
    expect(timecode(3725, false)).toBe("1:02:05");
    expect(fps({ num: 30000, den: 1001 })).toBe("29.97 fps");
    expect(fps({ num: 25, den: 1 })).toBe("25 fps");
    expect(eta(null)).toBe("estimating…");
    expect(eta([100, 104])).toMatch(/^about /);
  });
  it("splits Windows and POSIX paths", () => {
    expect(basename("C:\\Videos\\Ünïcode clip.mp4")).toBe("Ünïcode clip.mp4");
    expect(dirname("/home/a b/c.mkv")).toBe("/home/a b");
    expect(stem("/x/y.tar.mp4")).toBe("y.tar");
  });
});

describe("operations", () => {
  it("has metadata and defaults for every operation the backend knows", () => {
    for (const o of OPS) {
      expect(D[o.type], o.type).toBeDefined();
      expect(D[o.type].type).toBe(o.type);
    }
    expect(Object.keys(D).filter((k) => !["output", "grain_presets", "general_denoise_default"].includes(k)).sort()).toEqual(OPS.map((o) => o.type).sort());
  });

  it("inserts tasks at their canonical position", () => {
    let p: Pipeline = { stages: [] };
    p = addStageCanonical(p, "grain", D.grain.settings);
    p = addStageCanonical(p, "enhance", D.enhance.settings);
    p = addStageCanonical(p, "deinterlace", D.deinterlace.settings);
    expect(p.stages.map((s) => s.type)).toEqual(["deinterlace", "enhance", "grain"]);
    p = addStageCanonical(p, "enhance", D.enhance.settings);
    expect(new Set(p.stages.map((s) => s.id)).size).toBe(4);
    const moved = moveStage(p, p.stages[0].id, 1);
    expect(moved.stages[1].type).toBe("deinterlace");
    expect(moveStage(p, p.stages[0].id, -1)).toBe(p);
    expect(removeStage(p, p.stages[0].id).stages).toHaveLength(3);
  });

  it("computes required models like the Rust planner", () => {
    const p: Pipeline = {
      stages: [
        { id: "a", enabled: true, type: "denoise", settings: { ...D.denoise.settings } },
        { id: "b", enabled: true, type: "denoise", settings: { ...D.denoise.settings, engine: "hqdn3d" } },
        { id: "c", enabled: true, type: "interpolate", settings: { ...D.interpolate.settings, engine: "mci" } },
        { id: "d", enabled: false, type: "enhance", settings: D.enhance.settings },
        { id: "e", enabled: true, type: "face_restore", settings: D.face_restore.settings },
      ],
    };
    expect(requiredModels(p)).toEqual(["kair.ffdnet-color", "tencentarc.gfpgan-v1.4"]);
  });

  it("every built-in preset only uses catalog models and known operations", () => {
    const ids = new Set((models as unknown as ModelCard[]).map((m) => m.manifest!.id));
    for (const pr of presets as any[]) {
      for (const id of requiredModels(pr.pipeline)) expect(ids.has(id), `${pr.id} needs ${id}`).toBe(true);
      for (const s of pr.pipeline.stages) expect(stageSummary(s, models as unknown as ModelCard[])).not.toBe("");
    }
  });

  it("describes every operation", () => {
    for (const o of OPS) expect(opInfo(o.type).description.length).toBeGreaterThan(20);
  });
});

describe("IPC fixtures match the TypeScript types", () => {
  it("settings", () => {
    const s = settings as unknown as Settings;
    const typed: string[] = ["theme", "motion", "ui_scale", "output_dir", "naming_template", "preview_seconds", "cache_limit_mb", "cache_max_age_days", "low_disk_warning_mb", "performance", "notifications", "onboarding_complete", "inspector_mode_advanced", "allow_catalog_updates"];
    for (const k of typed) expect(s, k).toHaveProperty(k);
    for (const k of ["mode", "cpu_threads", "reserve_cpu_threads", "priority", "memory_preset", "memory_budget_mb", "memory_pressure", "max_concurrent_jobs", "device", "vram_budget_mb", "oom_tile_fallback"]) {
      expect(s.performance, k).toHaveProperty(k);
    }
    expect(["balanced", "high_performance", "maximum", "custom"]).toContain(s.performance.mode);
  });
  it("model cards", () => {
    for (const c of models as unknown as ModelCard[]) {
      expect(c).toHaveProperty("state");
      expect(c).toHaveProperty("capability.level");
      expect(c).toHaveProperty("in_progress");
      expect(c.manifest!.artifacts[0].sha256).toMatch(/^[0-9a-f]{64}$/);
      expect(c.manifest!.weights_license.length).toBeGreaterThan(0);
    }
  });
});

import { explainError } from "../views/Panels";

describe("explainError", () => {
  it("turns the GPU driver reset into plain words with a next step", () => {
    const e = explainError("Enhance (xinntao.realesr-animevideo-v3): GPU: GPU did not finish: Tried to wait using a submission index (3) that has not been returned by a successful submission (last successful submission: 1)");
    expect(e.title).toMatch(/graphics driver/);
    expect(e.hint).toMatch(/CPU/);
  });
  it("recognises memory, model and disk problems and keeps unknown text", () => {
    expect(explainError("GPU out of memory").title).toMatch(/memory/);
    expect(explainError("Model x is missing or damaged (p). Repair it").title).toMatch(/model/i);
    expect(explainError("No space left on device").title).toMatch(/disk space/);
    expect(explainError("something odd").title).toBe("Something odd");
  });
  it("turns common Windows and file problems into plain words, without paths or codes", () => {
    expect(explainError("The process cannot access the file because it is being used by another process. (os error 32)").title).toMatch(/open in another program/);
    expect(explainError("Access is denied. (os error 5)").title).toMatch(/did not allow access/);
    expect(explainError("Invalid data found when processing input").title).toMatch(/could not be read/);
    expect(explainError("error sending request: dns error: failed to lookup address").title).toMatch(/internet/);
    const unknown = explainError("Failed: weird thing at C:\\Users\\Ann\\Videos\\x.mp4 (os error 999)");
    expect(unknown.title).not.toMatch(/Ann|os error/);
  });
});
