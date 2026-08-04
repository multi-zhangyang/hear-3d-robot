import { expect, test } from "@playwright/test";

test("实时世界接收权威区块同步事件", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "桌面端覆盖同一实时状态链路");
  const password = process.env.HEAR_E2E_PASSWORD ?? "hear-e2e-local";
  await page.addInitScript((value) => {
    sessionStorage.setItem("hear.password", value);
  }, password);

  let synchronizedRevision: number | undefined;
  let synchronizedChunks: Record<string, unknown> | undefined;
  await page.route("**/api/runs/**", async (route) => {
    const url = new URL(route.request().url());
    if (/^\/api\/runs\/[^/]+\/events$/.test(url.pathname)) {
      if (synchronizedRevision === undefined) {
        throw new Error("Run details were not loaded before the event stream");
      }
      const runId = decodeURIComponent(url.pathname.split("/")[3]!);
      const cursor = `v1:999:${"c".repeat(64)}`;
      const event = {
        event_id: "browser-scenario-synchronized",
        run_id: runId,
        type: "humanoid_scenario_synchronized",
        at: "2026-08-04T00:00:00.000Z",
        cursor,
        data: {
          scenario_chunks: synchronizedChunks,
          synchronization: {
            changed: true,
            chunkRevision: synchronizedRevision,
            resourceRebuilt: true,
            changedDomains: ["geometry"],
            invalidatedPlanIds: []
          }
        }
      };
      await route.fulfill({
        status: 200,
        contentType: "text/event-stream",
        headers: { "cache-control": "no-cache" },
        body: `id: ${cursor}\ndata: ${JSON.stringify(event)}\n\n`
      });
      return;
    }
    if (!/^\/api\/runs\/[^/]+$/.test(url.pathname)) {
      await route.continue();
      return;
    }
    const response = await route.fetch();
    const details = await response.json() as Record<string, unknown>;
    const chunks = details.scenario_chunks as Record<string, unknown>;
    const revision = Number(chunks.revision);
    synchronizedRevision = revision + 1;
    synchronizedChunks = {
      ...chunks,
      revision: synchronizedRevision,
      changed_chunk_ids: Array.isArray(chunks.changed_chunk_ids)
        ? chunks.changed_chunk_ids
        : []
    };
    const checkpoint = details.checkpoint as Record<string, unknown>;
    await route.fulfill({
      response,
      json: {
        ...details,
        checkpoint: {
          ...checkpoint,
          status: "running",
          error: null
        }
      }
    });
  });

  await page.goto("/");
  const mission = page.locator("section.humanoid-mission-world");
  await expect(mission).toBeAttached({ timeout: 90_000 });
  await expect(mission).toHaveAttribute("aria-label", "实时人形任务");
  await expect.poll(() => synchronizedRevision ?? -1).toBeGreaterThan(0);
  const expectedRevision = synchronizedRevision!;
  await expect(
    mission.locator(".humanoid-physics-strip").getByText(
      `R${expectedRevision}`,
      { exact: true }
    )
  ).toBeVisible();
});
