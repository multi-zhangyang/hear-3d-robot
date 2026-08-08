import { access } from "node:fs/promises";
import { resolve } from "node:path";
import { expect, test, type Page } from "@playwright/test";
import { E2E_RUNS_DIR } from "./e2e-runs.js";
import { openRecordedOperator } from "./open-recorded-operator.js";

const UPDATE_README_SCREENSHOTS = process.env.HEAR_UPDATE_SCREENSHOTS === "1";
const DEFERRED_CHUNK = /three~|create-humanoid-stage|WorkspaceView|HumanoidMissionWorkspace|AgentFlowView|ActivityView|RobotTrailView|MissionModal/;

test("渲染自主人形世界与实时层级智能体界面", async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  const humanoidMeshes = new Set<string>();
  const deferredChunks = new Set<string>();
  page.on("request", (request) => {
    if (/\/humanoid\/g1\/meshes\/[^/]+\.STL$/i.test(request.url())) {
      humanoidMeshes.add(request.url());
    }
    const name = new URL(request.url()).pathname.split("/").at(-1) ?? "";
    if (DEFERRED_CHUNK.test(name)) deferredChunks.add(name);
  });
  await access(resolve(E2E_RUNS_DIR, ".operator.lock"));
  const mission = await openRecordedOperator(page, {
    beforeLogin: async () => expect([...deferredChunks]).toEqual([])
  });
  const label = await mission.getAttribute("aria-label");
  if (label !== "实时人形任务" && label !== "人形任务回顾") {
    throw new Error(`Unexpected humanoid mission mode: ${label ?? "missing"}`);
  }
  await assertHumanoidOperator(page, testInfo.project.name, label, humanoidMeshes, deferredChunks);
});

async function assertHumanoidOperator(
  page: Page,
  project: string,
  missionMode: "实时人形任务" | "人形任务回顾",
  meshes: ReadonlySet<string>,
  deferredChunks: ReadonlySet<string>
): Promise<void> {
  const worldMode = missionMode === "实时人形任务" ? "实时人形世界" : "人形世界回顾";
  await expect(page.locator(`section[aria-label="${worldMode}"]`)).toBeAttached();
  const canvas = page.locator("canvas.humanoid-canvas");
  await expect(canvas).toBeVisible({ timeout: 90_000 });
  await page.waitForFunction(() => {
    const element = document.querySelector("canvas.humanoid-canvas");
    return element instanceof HTMLCanvasElement && element.width > 200 && element.height > 200;
  });
  await expect.poll(() => [...deferredChunks]).toEqual(expect.arrayContaining([
    expect.stringMatching(/create-humanoid-stage/),
    expect.stringMatching(/HumanoidMissionWorkspace/),
    expect.stringMatching(/WorkspaceView/),
    expect.stringMatching(/three~/)
  ]));
  await expect.poll(() => meshes.size).toBeGreaterThan(20);

  const hierarchy = page.getByLabel("层级智能体执行状态");
  await expect(hierarchy).toBeVisible();
  const agentChain = hierarchy.locator(".humanoid-agent-chain");
  for (const name of ["自主协调智能体", "人形感知哨兵", "全身运动参考智能体", "人形物理执行智能体"]) {
    await expect(agentChain.getByText(name, { exact: true })).toBeVisible();
  }
  await expect(page.getByLabel("人形身体通道")).toContainText("双足运动");
  const physics = page.getByLabel("人形物理状态");
  if (project === "desktop") await expect(physics).toBeVisible();
  await expect(physics).toContainText("任务约束 · YAHMP · 学习控制 · MuJoCo");
  await expect(page.getByText("左脚", { exact: true })).toBeAttached();
  await expect(page.getByText("右脚", { exact: true })).toBeAttached();
  await expect(page.getByText("直立", { exact: true })).toBeAttached();
  await expect(page.locator(".graphics-error")).toHaveCount(0);

  const camera = page.getByRole("group", { name: "观察视角" });
  for (const [name, mode] of [["跟随", "follow"], ["世界", "world"], ["头部", "head"]] as const) {
    const button = camera.getByRole("button", { name, exact: true });
    await button.click();
    await expect(button).toHaveAttribute("aria-pressed", "true");
    await expect.poll(async () => (await robotProjection(page)).mode).toBe(mode);
  }
  const headProjection = await robotProjection(page);
  await camera.getByRole("button", { name: "跟随", exact: true }).click();
  await expect.poll(async () => {
    const projection = await robotProjection(page);
    return projection.mode === "follow" ? projection.revision : 0;
  }).toBeGreaterThan(headProjection.revision);
  const followProjection = await robotProjection(page);
  await page.getByRole("button", { name: "复位视角", exact: true }).click();
  await expect.poll(async () => (await robotProjection(page)).revision)
    .toBeGreaterThan(followProjection.revision);

  const hotbar = page.getByRole("navigation", { name: "工作区" });
  await expect(hotbar).toBeVisible();
  for (const name of ["世界", "智能体流", "行动历程", "输出"]) {
    await expect(hotbar.getByRole("button", { name, exact: true })).toBeVisible();
  }
  if (project === "mobile") {
    await assertMobileHudLayout(page, camera, hotbar);
  }
  await expect.poll(() => canvas.evaluate((element) => {
    if (!(element instanceof HTMLCanvasElement) || !element.dataset.robotScreenBounds) return 0;
    const bounds = JSON.parse(element.dataset.robotScreenBounds) as { top: number; bottom: number };
    return bounds.bottom - bounds.top;
  })).toBeGreaterThan(project === "desktop" ? 300 : 210);
  const [projection, canvasHeight] = await Promise.all([
    robotProjection(page),
    canvas.evaluate((element) => element.clientHeight)
  ]);
  const composition = { ...projection, canvasHeight };
  expect(composition.top).toBeGreaterThanOrEqual(-2);
  expect(composition.bottom).toBeLessThanOrEqual(composition.canvasHeight + 2);
  const [canvasLayout, hotbarLayout] = await Promise.all([
    canvas.boundingBox(),
    hotbar.boundingBox()
  ]);
  expect(canvasLayout).not.toBeNull();
  expect(hotbarLayout).not.toBeNull();
  expect(canvasLayout!.y + composition.bottom).toBeLessThan(hotbarLayout!.y - 6);
  const panels = [
    ["2", "智能体流面板", /AgentFlowView/, "实时层级智能体流", "hierarchy.png"],
    ["3", "行动历程面板", /RobotTrailView/, "机器人行动历程", "actions.png"],
    ["4", "智能体输出面板", /ActivityView/, "模型输出", "logs.png"]
  ] as const;
  for (const [key, region, deferred, content, screenshot] of panels) {
    await page.keyboard.press(key);
    await expect(page.getByRole("region", { name: region })).toBeVisible();
    await expect(page.getByLabel(content)).toBeVisible();
    await expect.poll(() => [...deferredChunks]).toEqual(expect.arrayContaining([
      expect.stringMatching(deferred)
    ]));
    if (project === "desktop") await captureReadmeScreenshot(page, screenshot);
    await page.keyboard.press("Escape");
  }
  await expect(hotbar.getByRole("button", { name: "世界" })).toHaveAttribute("aria-current", "page");
  await captureReadmeScreenshot(page, project === "desktop" ? "mission.png" : "mobile.png");

  const graphics = await canvas.evaluate((element): {
    backend: string | null;
    available: boolean;
    contextLost: boolean;
    error: number | null;
  } => {
    if (!(element instanceof HTMLCanvasElement)) {
      return { backend: null, available: false, contextLost: true, error: null };
    }
    const backend = element.dataset.renderBackend ?? null;
    if (backend === "webgpu") {
      return { backend, available: "gpu" in navigator, contextLost: false, error: null };
    }
    const context = element.getContext("webgl2") ?? element.getContext("webgl");
    if (!context) return { backend, available: false, contextLost: true, error: null };
    const error = context.getError();
    return {
      backend,
      available: true,
      contextLost: context.isContextLost(),
      error: error === context.NO_ERROR ? null : error
    };
  });
  expect(["webgpu", "webgl2"]).toContain(graphics.backend);
  expect(graphics.available).toBe(true);
  expect(graphics.contextLost).toBe(false);
  expect(graphics.error).toBeNull();

  const box = await canvas.boundingBox();
  expect(box).not.toBeNull();
  const rendered = await canvas.screenshot({ animations: "disabled" });
  const solid = await captureSolidReference(page, box!.width, box!.height);
  expect(rendered.byteLength).toBeGreaterThan(solid.byteLength * 2);
  const viewport = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    clientHeight: document.documentElement.clientHeight,
    scrollWidth: document.documentElement.scrollWidth,
    scrollHeight: document.documentElement.scrollHeight
  }));
  expect(viewport.scrollWidth).toBeLessThanOrEqual(viewport.clientWidth + 1);
  expect(viewport.scrollHeight).toBeLessThanOrEqual(viewport.clientHeight + 1);
  await page.screenshot({ path: `test-results/operator-${project}.png`, fullPage: true });
}

async function assertMobileHudLayout(
  page: Page,
  camera: ReturnType<Page["getByRole"]>,
  hotbar: ReturnType<Page["getByRole"]>
): Promise<void> {
  const goal = page.getByLabel("目标与长期记忆");
  const agent = page.getByLabel("层级智能体执行状态");
  const canvas = page.locator("canvas.humanoid-canvas");
  await expect(goal).toBeVisible();
  const boxes = await Promise.all([goal, agent, camera, hotbar, canvas]
    .map((locator) => locator.boundingBox()));
  expect(boxes.every((box) => box !== null)).toBe(true);
  const [goalBox, agentBox, cameraBox, hotbarBox, canvasBox] = boxes;
  expect(overlapArea(goalBox!, cameraBox!)).toBe(0);
  expect(overlapArea(goalBox!, hotbarBox!)).toBe(0);
  expect(overlapArea(cameraBox!, hotbarBox!)).toBe(0);
  expect(goalBox!.y).toBeGreaterThanOrEqual(0);
  expect(goalBox!.y + goalBox!.height).toBeLessThanOrEqual(
    await page.evaluate(() => document.documentElement.clientHeight)
  );
  const projection = await robotProjection(page);
  expect(canvasBox!.y + projection.top).toBeGreaterThanOrEqual(
    agentBox!.y + agentBox!.height + 6
  );
  expect(canvasBox!.y + projection.bottom).toBeLessThanOrEqual(goalBox!.y - 6);
}

interface RobotProjection {
  left: number;
  right: number;
  top: number;
  bottom: number;
  mode: "follow" | "world" | "head";
  revision: number;
}

async function robotProjection(page: Page): Promise<RobotProjection> {
  return page.locator("canvas.humanoid-canvas").evaluate(readRobotProjection);
}

function readRobotProjection(element: Element): RobotProjection {
  if (!(element instanceof HTMLCanvasElement) || !element.dataset.robotScreenBounds) {
    throw new Error("Missing projected humanoid bounds");
  }
  return JSON.parse(element.dataset.robotScreenBounds) as RobotProjection;
}

function overlapArea(
  left: { x: number; y: number; width: number; height: number },
  right: { x: number; y: number; width: number; height: number }
): number {
  const width = Math.max(0, Math.min(left.x + left.width, right.x + right.width)
    - Math.max(left.x, right.x));
  const height = Math.max(0, Math.min(left.y + left.height, right.y + right.height)
    - Math.max(left.y, right.y));
  return width * height;
}

async function captureSolidReference(page: Page, width: number, height: number): Promise<Buffer> {
  await page.evaluate(({ referenceWidth, referenceHeight }) => {
    document.querySelector("[data-e2e-solid-reference]")?.remove();
    const reference = document.createElement("div");
    reference.dataset.e2eSolidReference = "true";
    reference.setAttribute("aria-hidden", "true");
    Object.assign(reference.style, {
      position: "fixed",
      left: "0",
      top: "0",
      width: `${referenceWidth}px`,
      height: `${referenceHeight}px`,
      background: "#111821",
      zIndex: "2147483647",
      pointerEvents: "none"
    });
    document.body.append(reference);
  }, { referenceWidth: width, referenceHeight: height });
  const reference = page.locator("[data-e2e-solid-reference]");
  try {
    return await reference.screenshot({ animations: "disabled" });
  } finally {
    await reference.evaluate((element) => element.remove()).catch(() => undefined);
  }
}

async function captureReadmeScreenshot(page: Page, name: string): Promise<void> {
  if (!UPDATE_README_SCREENSHOTS) return;
  await page.screenshot({
    path: resolve(process.cwd(), "docs", "screenshots", name),
    animations: "disabled",
    fullPage: true
  });
}
