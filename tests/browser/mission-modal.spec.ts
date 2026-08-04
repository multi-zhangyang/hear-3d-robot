import { expect, test } from "@playwright/test";

test("原生任务编辑器保持完整目标建模与键盘交互", async ({ page }, testInfo) => {
  const password = process.env.HEAR_E2E_PASSWORD ?? "hear-e2e-local";
  await page.addInitScript((value) => {
    sessionStorage.setItem("hear.password", value);
  }, password);
  await page.route("**/api/bootstrap", async (route) => {
    const response = await route.fetch();
    const body = await response.json() as Record<string, unknown>;
    await route.fulfill({
      response,
      json: {
        ...body,
        provider: {
          configured: true,
          protocol: "openai_compatible",
          model: "configured-model"
        }
      }
    });
  });

  await page.goto("/");
  const openButton = page.getByRole("button", { name: "新建任务", exact: true }).first();
  await expect(openButton).toBeEnabled();
  await openButton.click();

  const dialog = page.getByRole("dialog", { name: "新建任务" });
  await expect(dialog).toBeVisible();
  await expect(page.locator(".ant-modal, .ant-form, .ant-select")).toHaveCount(0);
  const scenario = dialog.getByLabel("世界场景");
  await expect(scenario).toBeFocused();
  await scenario.selectOption("humanoid_frontier");

  const start = dialog.getByRole("button", { name: "启动任务" });
  const confirm = dialog.getByLabel("我确认以上条件是本任务的真实验收标准");
  await expect(start).toBeDisabled();
  await expect(dialog.getByRole("heading", { name: "真实验收条件" })).toBeVisible();
  await expect(dialog.locator(".mission-contract-review li")).toHaveCount(1);
  await expect(dialog.getByRole("radio", { name: "持续运行" })).toBeChecked();
  await dialog.getByText("完成后停止", { exact: true }).click();
  await expect(dialog.getByRole("radio", { name: "完成后停止" })).toBeChecked();
  const mission = dialog.getByLabel("任务意图");
  await mission.fill("自主穿越随机方块世界并抵达信标区域");
  await expect(dialog.getByLabel("条件摘要")).not.toHaveValue("");
  await confirm.check();
  await expect(start).toBeEnabled();
  await mission.fill("自主探索方块世界并抵达信标区域");
  await expect(confirm).not.toBeChecked();
  await expect(start).toBeDisabled();
  await confirm.check();
  const summary = dialog.getByLabel("条件摘要");
  await summary.fill(`${await summary.inputValue()}并保持稳定`);
  await expect(confirm).not.toBeChecked();
  await expect(start).toBeDisabled();
  await confirm.check();
  await expect(start).toBeEnabled();

  await dialog.getByLabel("添加完成条件").selectOption("end_effector_at");
  await expect(dialog.locator(".predicate-row")).toHaveCount(2);
  await expect(dialog.getByLabel("关键部位")).toHaveValue("left_wrist");
  await expect(dialog.getByLabel("坐标系")).toHaveValue("pelvis");
  await expect(dialog.getByLabel("连续稳定帧")).toHaveValue("5");
  const poseToggle = dialog.getByLabel("限定末端姿态");
  const poseToggleLabel = dialog.getByText("限定末端姿态", { exact: true });
  await expect(poseToggle).not.toBeChecked();
  await poseToggleLabel.click();
  await expect(poseToggle).toBeChecked();
  await expect(dialog.getByLabel("侧倾")).toHaveValue("0");
  await expect(dialog.getByLabel("俯仰")).toHaveValue("0");
  await expect(dialog.getByLabel("朝向")).toHaveValue("0");
  await expect(dialog.getByLabel("姿态容差（°）")).toHaveValue("10");
  await dialog.getByLabel("朝向").fill("90");
  await poseToggleLabel.click();
  await expect(poseToggle).not.toBeChecked();
  await expect(dialog.getByLabel("朝向")).toHaveCount(0);
  await expect(start).toBeDisabled();
  await dialog.getByRole("button", { name: "删除条件 2" }).click();
  await dialog.getByLabel("添加完成条件").selectOption("object_grasped");
  await expect(dialog.locator(".predicate-row")).toHaveCount(2);
  await expect(dialog.getByLabel("抓取手")).toHaveValue("either");
  await dialog.getByLabel("目标物体").selectOption({ index: 1 });
  await dialog.getByLabel("抓取手").selectOption("right");
  await expect(dialog.getByLabel("抓取手")).toHaveValue("right");
  await dialog.getByRole("button", { name: "删除条件 2" }).click();
  await confirm.check();
  await expect(start).toBeEnabled();

  if (testInfo.project.name === "desktop") {
    await page.screenshot({
      path: "test-results/mission-composer.png",
      fullPage: true,
      animations: "disabled"
    });
  }
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
});
