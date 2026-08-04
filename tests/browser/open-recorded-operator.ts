import { expect, type Locator, type Page, type Request } from "@playwright/test";

interface OpenRecordedOperatorOptions {
  beforeLogin?: () => Promise<void> | void;
  password?: string;
}

type EntryState = "fatal" | "loading" | "login" | "mission";

export async function openRecordedOperator(
  page: Page,
  options: OpenRecordedOperatorOptions = {}
): Promise<Locator> {
  const rendererStarted = page.waitForRequest(isHumanoidMeshRequest, { timeout: 30_000 });
  await page.goto("/");
  const mission = page.locator("section.humanoid-mission-world");
  const password = page.getByLabel("操作密码");
  const entry = await waitForEntryState(page, mission, password);
  if (entry === "fatal") throw new Error(await operatorFailure(page));
  if (entry === "login") {
    await options.beforeLogin?.();
    await password.fill(options.password ?? process.env.HEAR_E2E_PASSWORD ?? "hear-e2e-local");
    await page.getByRole("button", { name: /登\s*录/ }).click();
  }
  await rendererStarted;
  await expect(mission).toBeAttached();
  return mission;
}

async function waitForEntryState(
  page: Page,
  mission: Locator,
  password: Locator
): Promise<EntryState> {
  let observed: EntryState = "loading";
  await expect.poll(async () => {
    observed = await entryState(page, mission, password);
    return observed;
  }, {
    message: "等待操作端进入登录、任务或错误状态",
    timeout: 30_000
  }).not.toBe("loading");
  return observed;
}

async function entryState(
  page: Page,
  mission: Locator,
  password: Locator
): Promise<EntryState> {
  if (await mission.count() > 0) return "mission";
  if (await password.count() > 0) return "login";
  if (await page.locator("section.result-state").count() > 0) return "fatal";
  return "loading";
}

function isHumanoidMeshRequest(request: Request): boolean {
  return /^\/humanoid\/g1\/meshes\/[^/]+\.stl$/iu.test(new URL(request.url()).pathname);
}

async function operatorFailure(page: Page): Promise<string> {
  return await page.locator("section.result-state").innerText().catch(() => "操作端进入未知错误状态");
}
