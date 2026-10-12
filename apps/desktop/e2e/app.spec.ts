import { expect, test, type Page } from "@playwright/test";

// Behavioural UI tests against the in-browser demo backend (src/lib/mock.ts).
// The engine itself is covered by the Rust test suites.

const mock = (page: Page) => page.evaluate(() => JSON.parse(JSON.stringify({ ...window.__cfMock, timers: undefined })));

async function openVideo(page: Page, path = "/home/demo/Videos/holiday clip.mp4", scratch = true) {
  await page.evaluate((p) => (window.__cfMockPick = [[p]]), path);
  await page.getByRole("button", { name: "Browse" }).click();
  if (scratch) await page.getByRole("button", { name: "Start editing" }).click();
}

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await page.getByText("Drop videos here").waitFor();
});

test("demo mode is labelled and nothing is installed on start", async ({ page }) => {
  await expect(page.getByText("Demo", { exact: true })).toBeVisible();
  const m = await mock(page);
  expect(m.models.every((c: any) => c.state === "not_installed" && !c.in_progress)).toBe(true);
});

test("opening a video offers presets or a blank start", async ({ page }) => {
  await openVideo(page, undefined, false);
  const dlg = page.getByRole("dialog", { name: "How do you want to edit?" });
  await expect(dlg).toBeVisible();
  await dlg.getByTestId("preset-builtin.stabilize").click();
  await expect(dlg).toHaveCount(0);
  await expect(page.getByTestId("fx")).toContainText("Stabilization");
  await expect(page.getByTestId("sec-stabilize").getByRole("switch")).toHaveAttribute("aria-checked", "true");
});

test("export is blocked until the model is installed, then queues and completes", async ({ page }) => {
  await openVideo(page);
  await page.getByTestId("sec-enhance").getByRole("switch").click();
  await expect(page.getByTestId("issues")).toContainText("AnimeVideo v3 is not installed");
  await expect(page.getByTestId("export-as")).toBeDisabled();
  await expect(page.getByTestId("render")).toBeEnabled();

  await page.getByRole("button", { name: /^Install AnimeVideo v3 ·/ }).click();
  await expect(page.getByTestId("issues")).toHaveCount(0, { timeout: 10_000 });
  await page.getByTestId("render").click();
  await expect(page.getByAltText("Preview")).toBeVisible();

  await page.evaluate(() => (window.__cfMockSave = ["/home/demo/Out/holiday final.mp4"]));
  await page.getByTestId("export-as").click();
  await page.getByRole("button", { name: /Export queue/ }).click();
  const queue = page.getByTestId("queue");
  await expect(queue).toContainText("holiday final.mp4");
  await expect(queue).toContainText("Done", { timeout: 15_000 });
  await expect(queue.getByRole("button", { name: "Show file" })).toBeVisible();
});

test("an unreadable video shows an error", async ({ page }) => {
  await openVideo(page, "/home/demo/missing file.mp4", false);
  await page.keyboard.press("Escape");
  await expect(page.getByRole("alert").first()).toContainText("Could not open");
  await expect(page.getByRole("button", { name: "Choose another file" })).toBeVisible();
});

test("model library: pause, discard, install and uninstall with usage check", async ({ page }) => {
  await page.getByRole("button", { name: "Models", exact: true }).click();
  const dlg = page.getByRole("dialog", { name: "Models" });
  await dlg.getByTestId("model-xinntao.realesrgan-x4plus").click();
  const detail = dlg.getByTestId("model-detail");
  await detail.getByRole("button", { name: "Install" }).click();
  await detail.getByRole("button", { name: "Pause" }).click();
  await expect(dlg.getByTestId("model-xinntao.realesrgan-x4plus").getByTestId("model-state")).toHaveText("Paused");
  await detail.getByRole("button", { name: "Discard" }).click();
  await expect(dlg.getByTestId("model-xinntao.realesrgan-x4plus").getByTestId("model-state")).toHaveText("Not installed");

  await dlg.getByTestId("model-xinntao.realesr-animevideo-v3").click();
  await detail.getByRole("button", { name: "Install" }).click();
  await expect(dlg.getByTestId("model-xinntao.realesr-animevideo-v3").getByTestId("model-state")).toHaveText("Installed", { timeout: 10_000 });
  await detail.getByRole("button", { name: "Uninstall" }).click();
  const confirm = page.getByRole("dialog", { name: /Uninstall/ });
  await expect(confirm).toContainText("Preset:");
  await confirm.getByRole("button", { name: "Uninstall" }).click();
  await expect(dlg.getByTestId("model-xinntao.realesr-animevideo-v3").getByTestId("model-state")).toHaveText("Not installed");
});

test("moving the cache shows a plan and progress, then switches", async ({ page }) => {
  await page.getByRole("button", { name: "Preferences" }).click();
  await page.getByRole("navigation", { name: "Preferences sections" }).getByRole("button", { name: "Storage" }).click();
  await page.getByTestId("loc-cache").getByRole("button", { name: "Change…" }).click();
  await expect(page.getByTestId("start-move")).toBeDisabled();
  await page.evaluate(() => (window.__cfMockPick = [["/mnt/big disk/ClipFX cache"]]));
  const dlg = page.getByRole("dialog", { name: "Move cache" });
  await dlg.getByRole("button", { name: "Browse" }).click();
  await expect(dlg).toContainText("files");
  await page.getByTestId("start-move").click();
  await expect(dlg).toContainText("Moved to /mnt/big disk/ClipFX cache", { timeout: 10_000 });
  await dlg.getByRole("button", { name: "Done" }).click();
  await expect(page.getByTestId("loc-cache")).toContainText("/mnt/big disk/ClipFX cache");
  await expect(page.getByTestId("loc-cache")).toContainText("custom");
});

test("a destination inside the current folder is refused", async ({ page }) => {
  await page.getByRole("button", { name: "Preferences" }).click();
  await page.getByRole("navigation", { name: "Preferences sections" }).getByRole("button", { name: "Storage" }).click();
  await page.getByTestId("loc-models").getByRole("button", { name: "Change…" }).click();
  await page.evaluate(() => (window.__cfMockPick = [["/home/demo/.local/share/ClipFX Enhance/models/sub"]]));
  await page.getByRole("dialog", { name: "Move models" }).getByRole("button", { name: "Browse" }).click();
  await expect(page.getByRole("dialog", { name: "Move models" })).toContainText("inside the current folder");
  await expect(page.getByTestId("start-move")).toBeDisabled();
});

test("clearing the cache keeps unfinished job data unless asked", async ({ page }) => {
  await page.getByRole("button", { name: "Preferences" }).click();
  await page.getByRole("navigation", { name: "Preferences sections" }).getByRole("button", { name: "Storage" }).click();
  await page.getByTestId("cache-table").getByText("Job work files").click();
  await page.getByTestId("clear-cache").click();
  await expect(page.getByRole("status").filter({ hasText: "Freed" })).toContainText("kept 1 in use");
  const m = await mock(page);
  expect(m.cacheBytes.jobs).toBeGreaterThan(0);
  expect(m.cacheBytes.previews).toBe(0);
});

test("performance mode and memory priority persist", async ({ page }) => {
  await page.getByRole("button", { name: "Preferences" }).click();
  await page.getByRole("navigation", { name: "Preferences sections" }).getByRole("button", { name: "Performance" }).click();
  await page.getByRole("group", { name: "Performance mode" }).getByRole("button", { name: "Maximum" }).click();
  await page.getByRole("group", { name: "Memory preset" }).getByRole("button", { name: "Other apps" }).click();
  await expect(page.getByTestId("monitor")).toBeVisible();
  await expect.poll(async () => (await mock(page)).settings.performance.mode).toBe("maximum");
  await expect.poll(async () => (await mock(page)).settings.performance.memory_preset).toBe("more_for_others");
  await page.getByRole("group", { name: "Performance mode" }).getByRole("button", { name: "Custom" }).click();
  await expect(page.getByLabel("CPU threads", { exact: true })).toBeVisible();
});

test("presets popover applies, creates and deletes presets", async ({ page }) => {
  await openVideo(page);
  await page.getByRole("button", { name: "Presets", exact: true }).click();
  await page.getByTestId("preset-builtin.smooth-60").click();
  await expect(page.getByTestId("sec-interpolate").getByRole("switch")).toHaveAttribute("aria-checked", "true");

  await page.getByRole("button", { name: "Presets", exact: true }).click();
  await page.getByRole("button", { name: "Create new" }).click();
  await page.getByLabel("Name").fill("My smooth look");
  await page.getByRole("button", { name: "Save" }).click();
  const pop = page.getByRole("dialog", { name: "Presets" });
  const mine = pop.locator(".item", { hasText: "My smooth look" });
  await expect(mine).toBeVisible();
  await mine.getByRole("button", { name: /More for/ }).click();
  await page.getByRole("menuitem", { name: "Delete" }).click();
  await page.getByRole("dialog", { name: /Delete/ }).getByRole("button", { name: "Delete" }).click();
  await expect(pop.locator(".item", { hasText: "My smooth look" })).toHaveCount(0);
});

test("undo and redo restore adjustments", async ({ page }) => {
  await openVideo(page);
  const sw = page.getByTestId("sec-grain").getByRole("switch");
  await sw.click();
  await expect(sw).toHaveAttribute("aria-checked", "true");
  await page.keyboard.press("Control+z");
  await expect(sw).toHaveAttribute("aria-checked", "false");
  await page.keyboard.press("Control+Shift+Z");
  await expect(sw).toHaveAttribute("aria-checked", "true");
});

test("a second enhancement pass can be added and removed", async ({ page }) => {
  await openVideo(page);
  await page.getByTestId("sec-enhance").getByRole("switch").click();
  await page.getByRole("button", { name: "Add second enhancement" }).click();
  await expect(page.getByTestId("sec-enhance")).toContainText("Second enhancement");
  await page.getByTestId("sec-enhance").getByRole("button", { name: "Remove pass" }).click();
  await expect(page.getByTestId("sec-enhance")).not.toContainText("Second enhancement");
});

test("queue items pause, cancel with confirmation and retry", async ({ page }) => {
  await openVideo(page);
  await page.getByTestId("sec-grain").getByRole("switch").click();
  await page.evaluate(() => (window.__cfMock!.queuePaused = true));
  await page.getByTestId("quick-export").click();
  await page.getByRole("button", { name: /Export queue/ }).click();
  const q = page.getByTestId("queue");
  await expect(q).toContainText("Queued");
  await q.getByRole("button", { name: "Pause", exact: true }).click();
  await expect(q).toContainText("Paused");
  await q.getByRole("button", { name: "Cancel", exact: true }).click();
  await page.getByRole("dialog", { name: "Cancel this export?" }).getByRole("button", { name: "Cancel export" }).click();
  await expect(q).toContainText("Cancelled");
  await q.getByRole("button", { name: "Retry" }).click();
  await expect(q).toContainText("Queued");
});

test("sources panel lists opened videos and switches between them", async ({ page }) => {
  await openVideo(page);
  await page.getByRole("button", { name: /^Sources/ }).click();
  await page.evaluate(() => (window.__cfMockPick = [["/home/demo/Videos/second take.mp4"]]));
  await page.getByRole("button", { name: "Add videos" }).click();
  await page.getByRole("button", { name: "Start editing" }).click();
  const panel = page.getByTestId("sources");
  await expect(panel).toContainText("holiday clip.mp4");
  await expect(panel).toContainText("second take.mp4");
  await expect(page.locator(".crumbs")).toContainText("second take.mp4");
});
