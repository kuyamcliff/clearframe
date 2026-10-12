import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, describe, expect, it, vi } from "vitest";

import { NumberInput } from "../components/ui";
import { StageForm } from "../components/StageForm";
import defaults from "../lib/fixtures/operation_defaults.json";
import type { JobRecord } from "../lib/types";
import { boot, handleEvent, redo, setPipeline, undo, useStore } from "../state/store";
import { UpdateBanner } from "../components/UpdateBanner";
import { api } from "../lib/ipc";
import { ModelsDialog } from "../views/Models";
import { Sidebar } from "../views/Sidebar";

const D = defaults as any;

beforeAll(async () => {
  window.matchMedia = window.matchMedia ?? ((q: string) => ({ matches: false, media: q, addEventListener() {}, removeEventListener() {} }) as any);
  await boot();
});

describe("NumberInput", () => {
  it("only commits values inside the allowed range", async () => {
    const onChange = vi.fn();
    render(<NumberInput label="Tile" value={10} min={0} max={100} onChange={onChange} />);
    const input = screen.getByLabelText("Tile");
    await userEvent.clear(input);
    await userEvent.type(input, "250");
    expect(input).toHaveAttribute("aria-invalid", "true");
    expect(onChange).not.toHaveBeenCalledWith(250);
    await userEvent.clear(input);
    await userEvent.type(input, "42");
    expect(onChange).toHaveBeenLastCalledWith(42);
    fireEvent.keyDown(input, { key: "ArrowUp" });
  });
});

describe("store", () => {
  it("boots from the backend and keeps undo history", () => {
    const s = useStore.getState();
    expect(s.ready).toBe(true);
    expect(s.bootError).toBeNull();
    expect(s.models.length).toBeGreaterThan(5);
    expect(s.output?.container).toBe("mp4");
    setPipeline({ stages: [{ id: "grain", enabled: true, type: "grain", settings: D.grain.settings }] });
    setPipeline({ stages: [] });
    undo();
    expect(useStore.getState().pipeline.stages).toHaveLength(1);
    redo();
    expect(useStore.getState().pipeline.stages).toHaveLength(0);
  });

  it("applies job events", () => {
    const job = { id: "j1", kind: "render", name: "a.mp4", state: "processing", source: "/a.mp4", destination: "/out/a.mp4", warnings: [], progress: 0 } as unknown as JobRecord;
    act(() => handleEvent({ type: "job_updated", job }));
    act(() => handleEvent({ type: "job_progress", id: "j1", progress: { pass: 1, passes: 1, phase: "processing", frames_done: 5, frames_total: 10, fraction: 0.5, fps: 3, eta: null } }));
    expect(useStore.getState().jobs.find((j) => j.id === "j1")?.state).toBe("processing");
    expect(useStore.getState().progress.j1.fraction).toBe(0.5);
    act(() => handleEvent({ type: "job_updated", job: { ...job, state: "completed" } }));
    expect(useStore.getState().toasts.some((t) => t.message.includes("Finished"))).toBe(true);
    act(() => handleEvent({ type: "job_removed", id: "j1" }));
    expect(useStore.getState().jobs.find((j) => j.id === "j1")).toBeUndefined();
  });
});

describe("StageForm", () => {
  it("offers an explicit install button for a missing model and never installs by itself", async () => {
    const stage = { id: "enhance", enabled: true, type: "enhance" as const, settings: D.enhance.settings };
    const onChange = vi.fn();
    render(<StageForm stage={stage} onChange={onChange} />);
    expect(screen.getByRole("button", { name: /^Install AnimeVideo v3 ·/ })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("radio", { name: /General v3/ }));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ model_id: "xinntao.realesr-general-v3" }));
    expect(useStore.getState().models.every((m) => !m.in_progress)).toBe(true);
  });

  it("lists built-in methods next to AI models", () => {
    const stage = { id: "dn", enabled: true, type: "denoise" as const, settings: D.denoise.settings };
    render(<StageForm stage={stage} onChange={() => {}} />);
    expect(screen.getByRole("radio", { name: /hqdn3d/ })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /FFDNet/ })).toHaveAttribute("aria-checked", "true");
  });

  it("reports edits through onChange", async () => {
    const onChange = vi.fn();
    const stage = { id: "s", enabled: true, type: "stabilize" as const, settings: D.stabilize.settings };
    render(<StageForm stage={stage} onChange={onChange} />);
    await userEvent.click(screen.getByRole("button", { name: "Black" }));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ mode: "black", smoothing: D.stabilize.settings.smoothing }));
  });
});

describe("Models dialog", () => {
  it("lists every catalog model with recommended hardware and its storage location", async () => {
    render(<ModelsDialog />);
    await waitFor(() => expect(screen.getByTestId("models-dir")).not.toHaveTextContent("…"));
    for (const m of useStore.getState().models) expect(screen.getByTestId(`model-${m.manifest!.id}`)).toBeInTheDocument();
    const detail = screen.getByTestId("model-detail");
    expect(detail).toHaveTextContent("Recommended hardware");
    expect(detail).toHaveTextContent("video memory");
    // Licensing and provenance details are kept out of the app's Models page.
    for (const hidden of ["License", "Weights", "Source", "Training", "github.com"]) expect(detail).not.toHaveTextContent(hidden);
    expect(screen.getByRole("button", { name: "Install" })).toBeInTheDocument();
  });
});

describe("Sidebar", () => {
  it("shows every adjustment section with a toggle", () => {
    render(<Sidebar />);
    for (const t of ["Enhancement", "Denoise", "Motion deblur", "Face recovery", "Frame interpolation", "Stabilization", "Deinterlace", "Color & HDR", "Resize", "Film grain"]) {
      expect(screen.getByRole("switch", { name: t })).toBeInTheDocument();
    }
    expect(screen.getByTestId("export-as")).toBeDisabled();
  });
});

describe("Update banner", () => {
  it("announces a new version and only downloads after the user asks", async () => {
    useStore.setState((s) => ({ settings: { ...s.settings!, check_updates_on_launch: true } }));
    vi.spyOn(api, "checkUpdate").mockResolvedValue({ current: "1.0.0", latest: "1.1.0", newer: true, url: "", notes: "", size: 5_829_038, installable: true, downloaded: false });
    const download = vi.spyOn(api, "updateDownload").mockResolvedValue(undefined);
    const install = vi.spyOn(api, "updateInstall").mockResolvedValue(undefined);
    render(<UpdateBanner />);
    const banner = await screen.findByTestId("update-banner");
    expect(banner).toHaveTextContent("ClipFX Enhance 1.1.0 is available · 5.8 MB");
    expect(download).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: /Download/ }));
    expect(download).toHaveBeenCalledOnce();
    await userEvent.click(await screen.findByRole("button", { name: /Restart to update/ }));
    expect(install).toHaveBeenCalledOnce();
  });

  it("stays silent when update checks are off", async () => {
    cleanup();
    useStore.setState((s) => ({ update: null, settings: { ...s.settings!, check_updates_on_launch: false } }));
    const check = vi.spyOn(api, "checkUpdate");
    check.mockClear();
    render(<UpdateBanner />);
    await new Promise((r) => setTimeout(r, 20));
    expect(check).not.toHaveBeenCalled();
    expect(screen.queryByTestId("update-banner")).toBeNull();
  });
});
