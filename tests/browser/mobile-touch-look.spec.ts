import { expect, test, type CDPSession, type Page } from "@playwright/test";
import { openRecordedOperator } from "./open-recorded-operator.js";

test.use({
  viewport: { width: 390, height: 844 },
  isMobile: true,
  hasTouch: true
});

test("移动端触摸拖拽观察真实人形世界", async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  test.skip(testInfo.project.name !== "mobile", "移动触摸回归只运行一次");
  await openRecordedOperator(page);
  const canvas = page.locator("canvas.humanoid-canvas");
  await expect(canvas).toBeVisible({ timeout: 90_000 });
  await page.waitForFunction(() => {
    const element = document.querySelector("canvas.humanoid-canvas");
    return element instanceof HTMLCanvasElement && element.width > 200 && element.height > 200;
  });
  expect(await page.evaluate(() => navigator.maxTouchPoints)).toBeGreaterThan(0);

  await expect(page.getByRole("button", { name: "跟随", exact: true }))
    .toHaveAttribute("aria-pressed", "true");
  await expect.poll(() => canvas.evaluate((element) => getComputedStyle(element).touchAction))
    .toBe("none");

  const box = await canvas.boundingBox();
  expect(box).not.toBeNull();
  await settleRendering(page);
  const viewClip = {
    x: box!.x + Math.max(0, (box!.width - Math.min(180, box!.width)) / 2),
    y: box!.y + Math.max(0, (box!.height - Math.min(120, box!.height)) / 2),
    width: Math.min(180, box!.width),
    height: Math.min(120, box!.height)
  };
  const before = await page.screenshot({ clip: viewClip, animations: "disabled" });

  const session = await page.context().newCDPSession(page);
  try {
    await dragTouch(
      session,
      { x: box!.x + box!.width * 0.62, y: box!.y + box!.height * 0.48 },
      { x: box!.x + box!.width * 0.34, y: box!.y + box!.height * 0.48 }
    );
    await settleRendering(page);
    const turned = await page.screenshot({ clip: viewClip, animations: "disabled" });
    expect(turned.equals(before)).toBe(false);
    await expect(page.locator(".graphics-error")).toHaveCount(0);
  } finally {
    await session.detach();
  }
});

async function settleRendering(page: Page): Promise<void> {
  await page.evaluate(() => new Promise<void>((resolve) => {
    window.requestAnimationFrame(() => window.requestAnimationFrame(() => resolve()));
  }));
}

async function dragTouch(
  session: CDPSession,
  from: { x: number; y: number },
  to: { x: number; y: number }
): Promise<void> {
  await session.send("Input.dispatchTouchEvent", {
    type: "touchStart",
    touchPoints: [{ ...from, id: 1 }]
  });
  for (let step = 1; step <= 6; step += 1) {
    await session.send("Input.dispatchTouchEvent", {
      type: "touchMove",
      touchPoints: [{
        x: from.x + (to.x - from.x) * step / 6,
        y: from.y + (to.y - from.y) * step / 6,
        id: 1
      }]
    });
  }
  await session.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
}
