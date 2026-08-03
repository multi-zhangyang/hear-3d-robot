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
          model: "configured-model",
          endpoint: "https://example.test/v1"
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
  await expect(start).toBeDisabled();
  await dialog.getByLabel("任务目标").fill("自主穿越随机方块世界并抵达信标区域");
  await expect(dialog.getByLabel("条件摘要")).not.toHaveValue("");
  await dialog.getByLabel("确认完成条件").check();
  await expect(start).toBeEnabled();

  await dialog.getByLabel("添加完成条件").selectOption("end_effector_at");
  await expect(dialog.locator(".predicate-row")).toHaveCount(2);
  await expect(dialog.getByLabel("关键部位")).toHaveValue("left_wrist");
  await expect(dialog.getByLabel("坐标系")).toHaveValue("pelvis");
  await expect(dialog.getByLabel("连续稳定帧")).toHaveValue("5");
  await expect(start).toBeDisabled();
  await dialog.getByRole("button", { name: "删除条件 2" }).click();
  await dialog.getByLabel("确认完成条件").check();
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
