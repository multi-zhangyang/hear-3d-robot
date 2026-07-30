import { access } from "node:fs/promises";
import { resolve } from "node:path";
import { expect, test, type Page } from "@playwright/test";
import { E2E_RUNS_DIR } from "./e2e-runs.js";

const UPDATE_README_SCREENSHOTS = process.env.HEAR_UPDATE_SCREENSHOTS === "1";

test("渲染自主体素世界与实时机器人界面", async ({ page }, testInfo) => {
  await access(resolve(E2E_RUNS_DIR, ".operator.lock"));
  await page.goto("/");
  const passwordInput = page.getByLabel("操作密码");
  if (await passwordInput.isVisible().catch(() => false)) {
    await passwordInput.fill(process.env.HEAR_E2E_PASSWORD ?? "hear-e2e-local");
    await page.getByRole("button", { name: /登\s*录/ }).click();
  }

  const mission = page.locator('section[aria-label="实时任务"], section[aria-label="任务回顾"]');
  await expect(mission).toBeVisible({ timeout: 25_000 });
  const missionMode = await mission.getAttribute("aria-label");
  expect(["实时任务", "任务回顾"]).toContain(missionMode);
  const modelLamp = page.locator(".model-lamp");
  if (testInfo.project.name === "desktop") await expect(modelLamp).toBeVisible();
  await expect(page.getByText("模型已连接", { exact: true })).toHaveCount(0);
  if (missionMode === "任务回顾") {
    await expect(modelLamp).toHaveText("模型已响应");
    await expect(page.getByLabel("当前智能体状态")).toContainText("历史记录");
  }

  const hotbar = page.getByRole("navigation", { name: "工作区" });
  await expect(hotbar).toBeVisible();
  for (const name of ["世界", "智能体流", "行动历程", "输出"]) {
    await expect(hotbar.getByRole("button", { name, exact: true })).toBeVisible();
  }
  await expect(hotbar.getByRole("button", { name: "世界" })).toHaveAttribute("aria-current", "page");
  await expect(page.locator(".ant-pro-layout")).toHaveCount(0);

  await expect(page.getByLabel("当前智能体状态")).toBeVisible();
  await expect(page.getByLabel("当前智能体层级路径")).toBeVisible();
  await expect(page.getByLabel("机器人身体通道租约")).toBeVisible();
  await expect(page.getByLabel("世界运动状态")).toBeVisible();
  await expect(page.getByLabel("机器人朝向")).toBeVisible();
  await expect(page.locator(".world-state-strip")).toBeVisible();
  await expect(page.getByText(missionMode === "实时任务" ? "自主运行中" : "任务回顾 · 已完成", { exact: true }))
    .toBeVisible();

  const canvas = page.locator("canvas.three-canvas");
  await expect(canvas).toBeVisible();
  await expect(page.getByText("3D 场景不可用", { exact: false })).toHaveCount(0);
  await page.waitForFunction(() => {
    const element = document.querySelector("canvas.three-canvas");
    return element instanceof HTMLCanvasElement && element.width > 200 && element.height > 200;
  });

  const stageMode = missionMode === "实时任务" ? "实时机器人世界" : "机器人任务回顾";
  await expect(page.locator(`section[aria-label="${stageMode}"]`)).toBeVisible();
  await expect(page.locator(".playback-controls")).toHaveCount(0);
  await expect(page.getByLabel("上一帧")).toHaveCount(0);
  await expect(page.getByLabel("下一帧")).toHaveCount(0);

  const cameraControls = page.getByRole("group", { name: "相机模式" });
  const chase = cameraControls.getByRole("button", { name: "跟随视角", exact: true });
  const firstPerson = cameraControls.getByRole("button", { name: "第一人称视角", exact: true });
  const world = cameraControls.getByRole("button", { name: "全局视角", exact: true });
  await expect(chase).toHaveAttribute("aria-pressed", "true");
  await firstPerson.click();
  await expect(firstPerson).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator(".sensor-reticle")).toBeVisible();
  if (testInfo.project.name === "desktop") {
    await expect(page.getByLabel("第一人称视角交互")).toHaveText("单击进入视角");
    const box = await canvas.boundingBox();
    expect(box).not.toBeNull();
    await page.mouse.click(box!.x + box!.width / 2, box!.y + box!.height / 2);
    await expect(page.getByLabel("第一人称视角交互")).toHaveText("移动鼠标观察 · ESC 退出");
    await page.evaluate(() => document.exitPointerLock());
    await expect(page.getByLabel("第一人称视角交互")).toHaveText("单击进入视角");
  } else {
    await expect(page.getByLabel("第一人称视角交互")).toHaveText("拖动屏幕观察");
    await expect.poll(() => canvas.evaluate((element) => getComputedStyle(element).touchAction)).toBe("none");
  }
  await world.click();
  await expect(world).toHaveAttribute("aria-pressed", "true");
  if (testInfo.project.name === "mobile") {
    await chase.click();
    await expect(chase).toHaveAttribute("aria-pressed", "true");
  }

  const selectedTarget = page.getByLabel("已选择的世界目标");
  if (await selectedTarget.count() === 0) {
    const box = await canvas.boundingBox();
    expect(box).not.toBeNull();
    const pickCandidates: Array<readonly [number, number]> = testInfo.project.name === "mobile"
      ? [[0.5, 0.5], [0.47, 0.46], [0.53, 0.46], [0.47, 0.54], [0.53, 0.54]]
      : [[0.5, 0.56], [0.66, 0.62], [0.34, 0.68], [0.76, 0.72]];
    for (const [x, y] of pickCandidates) {
      await canvas.click({ position: { x: box!.width * x, y: box!.height * y } });
      if (await selectedTarget.count() > 0) break;
    }
  }
  await expect(selectedTarget).toBeVisible();
  await expect(selectedTarget.locator("small")).not.toHaveText("不可用");
  await page.getByRole("button", { name: "清除已选目标" }).click();
  await expect(selectedTarget).toHaveCount(0);

  if (testInfo.project.name === "desktop") await chase.click();
  await expect(chase).toHaveAttribute("aria-pressed", "true");
  await captureReadmeScreenshot(page, testInfo.project.name === "desktop" ? "mission.png" : "mobile.png");

  // Panels sit over the still-running world. Numeric shortcuts and Escape are
  // the primary game interaction, so exercise those instead of the old tabs.
  await page.keyboard.press("2");
  await expect(page.getByRole("region", { name: "智能体流面板" })).toBeVisible();
  await expect(page.getByLabel("实时层级智能体流")).toBeVisible();
  await expect(page.getByLabel("智能体执行流")).toBeVisible();
  await expect(canvas).toBeVisible();
  if (testInfo.project.name === "desktop") await captureReadmeScreenshot(page, "hierarchy.png");
  await page.keyboard.press("Escape");
  await expect(page.getByRole("region", { name: "智能体流面板" })).toHaveCount(0);
  await page.keyboard.press("3");
  await expect(page.getByRole("region", { name: "行动历程面板" })).toBeVisible();
  if (testInfo.project.name === "desktop") await captureReadmeScreenshot(page, "actions.png");
  await page.keyboard.press("Escape");
  await page.keyboard.press("4");
  await expect(page.getByRole("region", { name: "智能体输出面板" })).toBeVisible();
  await expect(page.getByLabel("模型输出")).toBeVisible();
  await expect(page.getByLabel(/\d+ 次模型调用/)).toBeVisible();
  if (missionMode === "任务回顾") {
    await expect(page.getByLabel("模型流状态")).toContainText("模型已响应");
  }
  if (testInfo.project.name === "desktop") await captureReadmeScreenshot(page, "logs.png");
  await page.keyboard.press("Escape");
  await expect(hotbar.getByRole("button", { name: "世界" })).toHaveAttribute("aria-current", "page");

  // Trigger one authoritative render, then read its WebGL buffer before the
  // browser compositor clears it. Chromium's screenshot compositor can race a
  // large SwiftShader canvas; readPixels tests the renderer itself directly.
  const painted = await canvas.evaluate((element) => new Promise<{
    colors: number;
    opaque: number;
    contextLost: boolean;
    error: number | null;
  }>((resolvePainted) => {
    if (!(element instanceof HTMLCanvasElement)) {
      resolvePainted({ colors: 0, opaque: 0, contextLost: true, error: null });
      return;
    }
    const fit = document.querySelector<HTMLButtonElement>(
      'button[aria-label="适配相机范围"]'
    );
    if (!fit) {
      resolvePainted({ colors: 0, opaque: 0, contextLost: false, error: null });
      return;
    }
    fit.addEventListener("click", () => {
      window.requestAnimationFrame(() => {
        const context = element.getContext("webgl2") ?? element.getContext("webgl");
        if (!context) {
          resolvePainted({ colors: 0, opaque: 0, contextLost: true, error: null });
          return;
        }
        context.finish();
        const bandHeight = Math.min(96, element.height);
        const bandY = Math.max(0, Math.floor((element.height - bandHeight) / 2));
        const pixels = new Uint8Array(element.width * bandHeight * 4);
        context.readPixels(
          0,
          bandY,
          element.width,
          bandHeight,
          context.RGBA,
          context.UNSIGNED_BYTE,
          pixels
        );
        const error = context.getError();
        const colors = new Set<string>();
        let opaque = 0;
        const pixelCount = element.width * bandHeight;
        const stride = Math.max(1, Math.floor(pixelCount / 12_000));
        for (let pixel = 0; pixel < pixelCount; pixel += stride) {
          const offset = pixel * 4;
          if ((pixels[offset + 3] ?? 0) > 0) opaque += 1;
          colors.add(`${pixels[offset]},${pixels[offset + 1]},${pixels[offset + 2]}`);
        }
        resolvePainted({
          colors: colors.size,
          opaque,
          contextLost: context.isContextLost(),
          error: error === context.NO_ERROR ? null : error
        });
      });
    }, { once: true });
    fit.click();
  }));
  expect(painted.contextLost).toBe(false);
  expect(painted.error).toBeNull();
  expect(painted.opaque).toBeGreaterThan(100);
  expect(painted.colors).toBeGreaterThan(8);

  const viewport = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    clientHeight: document.documentElement.clientHeight,
    scrollWidth: document.documentElement.scrollWidth,
    scrollHeight: document.documentElement.scrollHeight
  }));
  expect(viewport.scrollWidth).toBeLessThanOrEqual(viewport.clientWidth + 1);
  expect(viewport.scrollHeight).toBeLessThanOrEqual(viewport.clientHeight + 1);

  await page.screenshot({
    path: `test-results/operator-${testInfo.project.name}.png`,
    fullPage: true
  });
});

async function captureReadmeScreenshot(page: Page, name: string): Promise<void> {
  if (!UPDATE_README_SCREENSHOTS) return;
  await page.screenshot({
    path: resolve(process.cwd(), "docs", "screenshots", name),
    animations: "disabled",
    fullPage: true
  });
}
