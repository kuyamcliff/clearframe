import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, describe, expect, it, vi } from "vitest";

import { NumberInput } from "../components/ui";
import { StageForm } from "../components/StageForm";
import defaults from "../lib/fixtures/operation_defaults.json";
import type { JobRecord } from "../lib/types";
import { boot, handleEvent, redo, setPipeline, undo, useStore } from "../state/store";
import { ModelsView } from "../views/Models";

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
  it("offers an explicit install button for a missing model and never installs by itself", () => {
    const stage = { id: "enhance", enabled: true, type: "enhance" as const, settings: D.enhance.settings };
    render(<StageForm stage={stage} advanced={false} onChange={() => {}} />);
    expect(screen.getByRole("button", { name: /Install \(/ })).toBeInTheDocument();
    expect(useStore.getState().models.every((m) => !m.in_progress)).toBe(true);
  });

  it("shows advanced-only controls in advanced mode", () => {
    const stage = { id: "face", enabled: true, type: "face_restore" as const, settings: D.face_restore.settings };
    const { rerender } = render(<StageForm stage={stage} advanced={false} onChange={() => {}} />);
    expect(screen.queryByLabelText("Maximum faces")).toBeNull();
    rerender(<StageForm stage={stage} advanced onChange={() => {}} />);
    expect(screen.getByLabelText("Maximum faces")).toBeInTheDocument();
    expect(screen.getByText(/can change how people look/)).toBeInTheDocument();
  });

  it("reports edits through onChange", async () => {
    const onChange = vi.fn();
    const stage = { id: "s", enabled: true, type: "stabilize" as const, settings: D.stabilize.settings };
    render(<StageForm stage={stage} advanced={false} onChange={onChange} />);
    await userEvent.selectOptions(screen.getByLabelText("Borders"), "black");
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ mode: "black", smoothing: D.stabilize.settings.smoothing }));
  });
});

describe("Model Library", () => {
  it("lists catalog models with license information and storage location", async () => {
    render(<ModelsView />);
    await waitFor(() => expect(screen.getByTestId("models-dir")).not.toHaveTextContent("…"));
    expect(screen.getAllByRole("article").length).toBe(useStore.getState().models.length);
    expect(screen.getAllByText("Best for, limitations, license and source").length).toBeGreaterThan(0);
    expect(screen.getAllByRole("button", { name: "Install" }).length).toBeGreaterThan(0);
  });
});
