import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

import { loadEnvironment, loadProviderConfig, loadRuntimeCatalog } from "../dist/config/load.js";
import { RunStore } from "../dist/persistence/run-store.js";
import { startHumanoidMission } from "../dist/runtime/humanoid-mission-runner.js";
import { drawSeed } from "../dist/world/world-generator.js";
import { inspectLiveRunEvidence } from "./live-run-evidence.mjs";

loadEnvironment();

const scenarioId = optionalText("HEAR_LIVE_SCENARIO") ?? "humanoid_courtyard";
const seed = optionalSeed("HEAR_LIVE_SEED") ?? drawSeed();
const timeoutMs = optionalPositiveInteger("HEAR_LIVE_TIMEOUT_MS")
  ?? (optionalPositiveInteger("HEAR_LIVE_TIMEOUT_MINUTES") ?? 30) * 60_000;
const runsDir = resolve(optionalText("HEAR_RUNS_DIR") ?? "runs");
const reportPath = optionalText("HEAR_LIVE_REPORT");

await mkdir(runsDir, { recursive: true });
const [catalog, provider] = await Promise.all([
  loadRuntimeCatalog(),
  Promise.resolve(loadProviderConfig())
]);
const scenario = catalog.materialize(scenarioId, seed);
const startedAt = Date.now();
const result = await startHumanoidMission({
  runsDir,
  mission: "在当前世界中自主观察、规划并以真实物理执行完成场景目标。",
  scenarioId,
  goal: scenario.default_goal,
  catalog,
  provider,
  runMode: "mission",
  seed,
  signal: AbortSignal.timeout(timeoutMs)
});

const store = await RunStore.open(result.runDir);
const evidence = await inspectLiveRunEvidence({
  store,
  scenario,
  expectedStatus: "succeeded",
  requireMissionCompletion: true
});
const report = { ...evidence, duration_ms: Date.now() - startedAt };
if (reportPath) {
  await writeFile(resolve(reportPath), `${JSON.stringify(report, null, 2)}\n`, "utf8");
}
process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);

function optionalText(name) {
  const value = process.env[name]?.trim();
  return value || undefined;
}

function optionalSeed(name) {
  const value = optionalText(name);
  if (value === undefined) return undefined;
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 0 || parsed > 0xffff_ffff) {
    throw new Error(`${name} must be an unsigned 32-bit integer`);
  }
  return parsed;
}

function optionalPositiveInteger(name) {
  const value = optionalText(name);
  if (value === undefined) return undefined;
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
  return parsed;
}
