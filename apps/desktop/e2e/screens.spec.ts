import { expect, test } from "@playwright/test";

// Captures screenshots of every surface for visual review (CF_SHOTS=dir).
test.skip(!process.env.CF_SHOTS, "screenshots only on request");
test.use({ viewport: { width: 1600, height: 960 } });

test("screens", async ({ page }) => {
  const dir = process.env.CF_SHOTS!;
  const shot = (n: string) => page.screenshot({ path: `${dir}/${n}.png` });
  await page.goto("/");
  await page.getByText("Drop videos here").waitFor();
  await shot("01-empty");
  await page.getByRole("button", { name: "Browse" }).click();
  await expect(page.getByRole("dialog", { name: "How do you want to edit?" })).toBeVisible();
  await shot("02-start-dialog");
  await page.getByRole("button", { name: "Start editing" }).click();
  await page.getByTestId("sec-enhance").getByRole("switch").click();
  await page.getByTestId("sec-interpolate").getByRole("switch").click();
  await page.waitForTimeout(400);
  await shot("03-adjustments");
  await page.getByRole("button", { name: /^Install AnimeVideo/ }).first().click();
  await page.waitForTimeout(2500);
  await page.getByTestId("sec-interpolate").getByRole("button", { name: /^Install RIFE/ }).first().click();
  await page.waitForTimeout(3500);
  await page.getByTestId("render").click();
  await page.waitForTimeout(900);
  await shot("04-split-preview");
  await page.getByRole("button", { name: "Side by side" }).click();
  await page.waitForTimeout(400);
  await shot("05-side-by-side");
  await page.getByRole("button", { name: "Split view" }).click();
  await page.getByRole("tab", { name: "Codec settings" }).click();
  await shot("06-codec");
  await page.getByRole("button", { name: "Presets" }).click();
  await shot("07-presets");
  await page.keyboard.press("Escape");
  await page.getByTestId("quick-export").click();
  await page.getByRole("button", { name: /Export queue/ }).click();
  await page.waitForTimeout(1200);
  await shot("08-queue");
  await page.getByRole("button", { name: /^Sources/ }).click();
  await page.waitForTimeout(500);
  await shot("09-sources");
  await page.getByRole("button", { name: "Close" }).click();
  await page.getByRole("button", { name: "Models", exact: true }).click();
  await page.waitForTimeout(500);
  await shot("10-models");
  await page.keyboard.press("Escape");
  await page.getByRole("button", { name: "Preferences" }).click();
  await shot("11-prefs-general");
  for (const s of ["Storage", "Performance"]) {
    await page.getByRole("navigation", { name: "Preferences sections" }).getByRole("button", { name: s }).click();
    await page.waitForTimeout(1500);
    await shot(`12-prefs-${s.toLowerCase()}`);
  }
});
