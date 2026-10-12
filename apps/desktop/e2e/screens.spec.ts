import { expect, test } from "@playwright/test";

// Captures screenshots of every view for visual review (CF_SHOTS=dir).
test.skip(!process.env.CF_SHOTS, "screenshots only on request");

test("screens", async ({ page }) => {
  const dir = process.env.CF_SHOTS!;
  await page.goto("/");
  await page.getByRole("dialog").screenshot({ path: `${dir}/00-onboarding.png` });
  await page.getByRole("button", { name: /^Start$/ }).click();
  await page.screenshot({ path: `${dir}/01-welcome.png` });
  await page.getByRole("button", { name: /Open video…/ }).click();
  await page.getByText("1280×720").waitFor();
  await page.getByTestId("task-enhance").getByRole("switch").click();
  await page.screenshot({ path: `${dir}/02-workspace.png` });
  await page.getByRole("button", { name: "Pipeline" }).click();
  await page.screenshot({ path: `${dir}/03-pipeline.png` });
  await page.getByRole("button", { name: /Install \(/ }).click();
  await expect(page.getByRole("button", { name: /Install \(/ })).toHaveCount(0, { timeout: 10000 });
  await page.getByRole("button", { name: "Preview frame" }).click();
  await page.waitForTimeout(800);
  await page.screenshot({ path: `${dir}/03b-preview.png` });
  await page.getByTestId("open-export").click();
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${dir}/04-export.png` });
  await page.keyboard.press("Escape");
  for (const v of ["Queue", "Presets", "Model Library"]) {
    await page.getByRole("button", { name: v, exact: true }).click();
    await page.waitForTimeout(400);
    await page.screenshot({ path: `${dir}/05-${v}.png`, fullPage: true });
  }
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  for (const s of ["Storage & cache", "Performance", "General"]) {
    await page.getByRole("button", { name: s }).click();
    await page.waitForTimeout(1500);
    await page.screenshot({ path: `${dir}/06-${s.replace(/\W+/g, "_")}.png` });
  }
});
