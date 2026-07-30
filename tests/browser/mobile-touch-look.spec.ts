import { expect, test, type CDPSession, type Page } from "@playwright/test";

test.use({
  viewport: { width: 390, height: 844 },
  isMobile: true,
  hasTouch: true
});

test("移动端触摸拖拽环视且轻触仍可选取", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "mobile", "移动触摸回归只运行一次");
  await openRecordedRun(page);
  const canvas = page.locator("canvas.three-canvas");
  await expect(canvas).toBeVisible({ timeout: 25_000 });
  await page.waitForFunction(() => {
    const element = document.querySelector("canvas.three-canvas");
    return element instanceof HTMLCanvasElement && element.width > 200 && element.height > 200;
  });
  expect(await page.evaluate(() => navigator.maxTouchPoints)).toBeGreaterThan(0);

  await page.getByRole("button", { name: "第一人称视角", exact: true }).click();
  await expect(page.getByRole("button", { name: "第一人称视角", exact: true }))
    .toHaveAttribute("aria-pressed", "true");
  await expect(page.getByLabel("第一人称视角交互")).toHaveText("拖动屏幕观察");
  await expect.poll(() => canvas.evaluate((element) => getComputedStyle(element).touchAction))
    .toBe("none");

  const selection = page.getByLabel("已选择的世界目标");
  await expect(selection).toHaveCount(0);
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
    await expect(selection).toHaveCount(0);

    const candidates: Array<readonly [number, number]> = [
      [0.5, 0.7],
      [0.35, 0.72],
      [0.65, 0.72],
      [0.5, 0.58],
      [0.25, 0.8],
      [0.75, 0.8]
    ];
    for (const [x, y] of candidates) {
      await tapTouch(session, box!.x + box!.width * x, box!.y + box!.height * y);
      if (await selection.isVisible().catch(() => false)) break;
    }
    await expect(selection).toBeVisible();
    await expect(selection.locator("small")).not.toHaveText("不可用");
  } finally {
    await session.detach();
  }
});

async function openRecordedRun(page: Page): Promise<void> {
  await page.goto("/");
  const password = page.getByLabel("操作密码");
  if (await password.isVisible().catch(() => false)) {
    await password.fill(process.env.HEAR_E2E_PASSWORD ?? "hear-e2e-local");
    await page.getByRole("button", { name: /登\s*录/ }).click();
  }
  await expect(page.locator('section[aria-label="实时任务"], section[aria-label="任务回顾"]'))
    .toBeVisible({ timeout: 25_000 });
}

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

async function tapTouch(session: CDPSession, x: number, y: number): Promise<void> {
  await session.send("Input.dispatchTouchEvent", {
    type: "touchStart",
    touchPoints: [{ x, y, id: 2 }]
  });
  await session.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
}
