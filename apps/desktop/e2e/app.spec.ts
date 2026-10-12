import { expect, test, type Page } from "@playwright/test";

// Behavioural UI tests against the in-browser demo backend (src/lib/mock.ts).
// They check what the interface does and promises; the engine itself is
// covered by the Rust test suites.

async function start(page: Page) {
  await page.goto("/");
  await page.getByRole("button", { name: /^Start$/ }).click();
}

async function openVideo(page: Page, path = "/home/demo/Videos/holiday clip.mp4") {
  await page.evaluate((p) => (window.__cfMockPick = [[p]]), path);
  await page.getByRole("button", { name: "Open video…" }).click();
}

const mock = (page: Page) => page.evaluate(() => JSON.parse(JSON.stringify({ ...window.__cfMock, timers: undefined })));

test("onboarding never installs models that were not ticked", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("dialog", { name: "Welcome to ClearFrame" })).toBeVisible();
  await page.getByRole("button", { name: /^Start$/ }).click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  const m = await mock(page);
  expect(m.models.every((c: any) => c.state === "not_installed" && !c.in_progress)).toBe(true);
  expect(m.settings.onboarding_complete).toBe(true);
});

test("demo mode is clearly labelled", async ({ page }) => {
  await start(page);
  await expect(page.getByRole("status").filter({ hasText: "Demo mode" })).toBeVisible();
});

test("export is blocked until the required model is installed, then queues and completes", async ({ page }) => {
  await start(page);
  await openVideo(page);
  await expect(page.getByText("1280×720").first()).toBeVisible();
  await page.getByTestId("task-enhance").getByRole("switch").click();
  await expect(page.getByTestId("plan-summary")).toContainText("is not installed");
  await expect(page.getByTestId("open-export")).toBeDisabled();
  await expect(page.getByRole("button", { name: "Preview frame" })).toBeDisabled();

  await page.getByRole("button", { name: /Install \(/ }).click();
  await expect(page.getByTestId("plan-summary")).not.toContainText("is not installed", { timeout: 10_000 });
  await expect(page.getByTestId("open-export")).toBeEnabled();

  await page.getByRole("button", { name: "Preview frame" }).click();
  await expect(page.getByAltText("Processed")).toBeVisible();

  await page.getByTestId("open-export").click();
  await expect(page.getByTestId("destination")).toContainText("holiday clip_enhanced.mp4");
  await page.getByTestId("confirm-export").click();
  await page.getByRole("button", { name: "Queue", exact: true }).click();
  const job = page.locator(".job").first();
  await expect(job).toContainText("Completed", { timeout: 15_000 });
  await expect(job.getByRole("button", { name: "Show file" })).toBeVisible();
});

test("an unreadable video shows an error instead of crashing", async ({ page }) => {
  await start(page);
  await openVideo(page, "/home/demo/missing file.mp4");
  await expect(page.getByRole("alert")).toContainText("could not be opened");
  await expect(page.getByRole("button", { name: "Choose another file…" })).toBeVisible();
});

test("model install can be paused, resumed and cancelled; uninstall explains usage", async ({ page }) => {
  await start(page);
  await page.getByRole("button", { name: "Model Library", exact: true }).click();
  const card = page.getByTestId("model-xinntao.realesrgan-x4plus");
  await card.getByRole("button", { name: "Install" }).click();
  await card.getByRole("button", { name: "Pause" }).click();
  await expect(card.getByTestId("model-state")).toHaveText("Paused");
  await card.getByRole("button", { name: "Discard" }).click();
  await expect(card.getByTestId("model-state")).toHaveText("Not installed");

  const anime = page.getByTestId("model-xinntao.realesr-animevideo-v3");
  await anime.getByRole("button", { name: "Install" }).click();
  await expect(anime.getByTestId("model-state")).toHaveText("Installed", { timeout: 10_000 });
  await anime.getByRole("button", { name: "Uninstall" }).click();
  const dlg = page.getByRole("dialog");
  await expect(dlg).toContainText("Used by");
  await expect(dlg).toContainText("Preset:");
  await dlg.getByRole("button", { name: "Keep" }).click();
  await expect(anime.getByTestId("model-state")).toHaveText("Installed");
  await anime.getByRole("button", { name: "Uninstall" }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Uninstall" }).click();
  await expect(anime.getByTestId("model-state")).toHaveText("Not installed");
});

test("changing the cache location shows a plan and progress, then switches", async ({ page }) => {
  await start(page);
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await page.getByRole("button", { name: "Storage & cache" }).click();
  await page.getByTestId("loc-cache").getByRole("button", { name: "Change…" }).click();
  await expect(page.getByTestId("start-move")).toBeDisabled();
  await page.evaluate(() => (window.__cfMockPick = [["/mnt/big disk/ClearFrame cache"]]));
  await page.getByRole("dialog").getByRole("button", { name: "Choose…" }).click();
  await expect(page.getByRole("dialog")).toContainText("to move");
  await page.getByTestId("start-move").click();
  await expect(page.getByRole("dialog")).toContainText("Moved to /mnt/big disk/ClearFrame cache", { timeout: 10_000 });
  await page.getByRole("dialog").getByRole("button", { name: "Close" }).last().click();
  await expect(page.getByTestId("loc-cache")).toContainText("/mnt/big disk/ClearFrame cache");
  await expect(page.getByTestId("loc-cache")).toContainText("custom");
});

test("a destination inside the current folder is refused", async ({ page }) => {
  await start(page);
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await page.getByRole("button", { name: "Storage & cache" }).click();
  await page.getByTestId("loc-models").getByRole("button", { name: "Change…" }).click();
  await page.evaluate(() => (window.__cfMockPick = [["/home/demo/.local/share/ClearFrame/models/sub"]]));
  await page.getByRole("dialog").getByRole("button", { name: "Choose…" }).click();
  await expect(page.getByRole("dialog")).toContainText("inside the current folder");
  await expect(page.getByTestId("start-move")).toBeDisabled();
});

test("clearing the cache keeps unfinished job data unless asked", async ({ page }) => {
  await start(page);
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await page.getByRole("button", { name: "Storage & cache" }).click();
  await page.getByRole("checkbox", { name: "Select Job work files (segments, intermediates)" }).check();
  await page.getByTestId("clear-cache").click();
  await expect(page.getByRole("status").filter({ hasText: "Freed" })).toContainText("Kept 1 item");
  const m = await mock(page);
  expect(m.cacheBytes.jobs).toBeGreaterThan(0);
  expect(m.cacheBytes.previews).toBe(0);
});

test("performance mode and memory preset persist", async ({ page }) => {
  await start(page);
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await page.getByRole("button", { name: "Performance" }).click();
  await page.getByRole("radio", { name: /Maximum/ }).click();
  await page.getByRole("group", { name: "Memory preset" }).getByRole("button", { name: "More for other apps" }).click();
  await expect(page.getByTestId("monitor")).toBeVisible();
  await expect.poll(async () => (await mock(page)).settings.performance.mode).toBe("maximum");
  await expect.poll(async () => (await mock(page)).settings.performance.memory_preset).toBe("more_for_others");
  await page.getByRole("radio", { name: /Custom/ }).click();
  await expect(page.getByLabel("CPU threads", { exact: true })).toBeVisible();
});

test("presets apply to the pipeline and user presets can be saved and deleted", async ({ page }) => {
  await start(page);
  await openVideo(page);
  await page.getByRole("button", { name: "Presets", exact: true }).click();
  await page.getByTestId("preset-builtin.stabilize").getByRole("button", { name: "Use" }).click();
  await expect(page.getByText("Based on preset: Stabilize (auto crop)")).toBeVisible();
  await expect(page.getByTestId("task-stabilize").getByRole("switch")).toHaveAttribute("aria-checked", "true");

  await page.getByRole("button", { name: "Save as preset" }).click();
  await page.getByLabel("Name").fill("My steady look");
  await page.getByRole("button", { name: "Save preset" }).click();
  await page.getByRole("button", { name: "Presets", exact: true }).click();
  const mine = page.locator(".card", { hasText: "My steady look" });
  await expect(mine).toBeVisible();
  await expect(page.getByTestId("preset-builtin.stabilize").getByRole("button", { name: "Delete" })).toHaveCount(0);
  await mine.getByRole("button", { name: "Delete" }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Delete preset" }).click();
  await expect(page.locator(".card", { hasText: "My steady look" })).toHaveCount(0);
});

test("undo and redo restore pipeline edits", async ({ page }) => {
  await start(page);
  await openVideo(page);
  const sw = page.getByTestId("task-grain").getByRole("switch");
  await sw.click();
  await expect(sw).toHaveAttribute("aria-checked", "true");
  await page.getByRole("button", { name: "Undo" }).click();
  await expect(sw).toHaveAttribute("aria-checked", "false");
  await page.keyboard.press("Control+Shift+Z");
  await expect(sw).toHaveAttribute("aria-checked", "true");
});

test("advanced pipeline editor reorders and removes stages", async ({ page }) => {
  await start(page);
  await openVideo(page);
  await page.getByRole("button", { name: "Pipeline" }).click();
  for (const label of ["Film grain", "Deinterlace"]) {
    await page.getByRole("button", { name: "Add stage" }).click();
    await page.getByRole("menuitem", { name: label }).click();
  }
  const rows = page.locator(".stage-row");
  await expect(rows).toHaveCount(2);
  await expect(rows.nth(0)).toContainText("Film grain");
  await rows.nth(1).getByRole("button", { name: "Move up" }).click();
  await expect(rows.nth(0)).toContainText("Deinterlace");
  await rows.nth(1).getByRole("button", { name: "Remove stage" }).click();
  await expect(rows).toHaveCount(1);
});

test("queue items can be paused, resumed and cancelled with confirmation", async ({ page }) => {
  await start(page);
  await openVideo(page);
  await page.getByTestId("task-grain").getByRole("switch").click();
  await page.evaluate(() => (window.__cfMock!.queuePaused = true));
  await page.getByTestId("open-export").click();
  await page.getByTestId("confirm-export").click();
  await page.getByRole("button", { name: "Queue", exact: true }).click();
  const job = page.locator(".job").first();
  await expect(job).toContainText("Queued");
  await job.getByRole("button", { name: "Pause" }).click();
  await expect(job).toContainText("Paused");
  await job.getByRole("button", { name: "Cancel" }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Cancel export" }).click();
  await expect(job).toContainText("Cancelled");
  await job.getByRole("button", { name: "Retry" }).click();
  await expect(job).toContainText("Queued");
});
