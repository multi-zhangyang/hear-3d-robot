import { writeFile } from "node:fs/promises";
import { resolve } from "node:path";

import { loadEnvironment, loadProviderConfig, loadRuntimeCatalog } from "../dist/config/load.js";
import { RunManager } from "../dist/server/run-manager.js";
import { loadConfiguredHumanoidControllerSource } from
  "../dist/world/humanoid/controller-module.js";

loadEnvironment();

const scenarioId = requiredText("HEAR_LIVE_SCENARIO");
const seed = requiredSeed("HEAR_LIVE_SEED");
const runsDir = resolve(requiredText("HEAR_RUNS_DIR"));
const readyPath = resolve(requiredText("HEAR_LIVE_WORKER_READY"));
const [catalog, provider, controllerSource] = await Promise.all([
  loadRuntimeCatalog(),
  Promise.resolve(loadProviderConfig()),
  loadConfiguredHumanoidControllerSource()
]);
const scenario = catalog.materialize(scenarioId, seed);
const manager = new RunManager({
  runsDir,
  catalog,
  provider,
  ...(controllerSource ? { controllerSource } : {})
});

process.on("message", (message) => {
  if (message && typeof message === "object" && message.type === "crash") {
    process.exit(137);
  }
});

const runId = await manager.start({
  mission: "长期持续自主观察、选择目标、执行并形成可恢复记忆。",
  scenarioId,
  goal: scenario.default_goal,
  runMode: "continuous",
  seed
});
await writeFile(readyPath, `${JSON.stringify({ run_id: runId })}\n`, "utf8");
await new Promise(() => undefined);

function requiredText(name) {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`${name} is required`);
  return value;
}

function requiredSeed(name) {
  const value = Number(requiredText(name));
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new Error(`${name} must be an unsigned 32-bit integer`);
  }
  return value;
}
