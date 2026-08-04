import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

import { loadEnvironment, loadProviderConfig, loadRuntimeCatalog } from "../dist/config/load.js";
import { resolveRunDirectory, RunStore } from "../dist/persistence/run-store.js";
import { RunManager } from "../dist/server/run-manager.js";
import { drawSeed } from "../dist/world/world-generator.js";
import { inspectLiveRunEvidence } from "./live-run-evidence.mjs";

loadEnvironment();

const scenarioId = optionalText("HEAR_LIVE_SCENARIO") ?? "humanoid_courtyard";
const seed = optionalSeed("HEAR_LIVE_SEED") ?? drawSeed();
const timeoutMs = optionalPositiveInteger("HEAR_LIVE_TIMEOUT_MS")
  ?? (optionalPositiveInteger("HEAR_LIVE_TIMEOUT_MINUTES") ?? 45) * 60_000;
const minimumCycles = optionalPositiveInteger("HEAR_LIVE_MINIMUM_CYCLES") ?? 3;
const minimumCompletedGoals = optionalPositiveInteger(
  "HEAR_LIVE_MINIMUM_COMPLETED_GOALS"
) ?? 2;
const runsDir = resolve(optionalText("HEAR_RUNS_DIR") ?? "runs");
const reportPath = optionalText("HEAR_LIVE_REPORT");

await mkdir(runsDir, { recursive: true });
const [catalog, provider] = await Promise.all([
  loadRuntimeCatalog(),
  Promise.resolve(loadProviderConfig())
]);
const scenario = catalog.materialize(scenarioId, seed);
const manager = new RunManager({ runsDir, catalog, provider });
const startedAt = Date.now();
let runId;
try {
  runId = await manager.start({
    mission: "在当前世界持续自主观察、选择目标并通过真实物理执行推进探索。",
    scenarioId,
    goal: scenario.default_goal,
    runMode: "continuous",
    seed
  });
  const runDir = resolveRunDirectory(runsDir, runId);
  const store = await RunStore.open(runDir);
  await waitUntil(async () => {
    const checkpoint = await store.readHumanoidCheckpoint();
    if (checkpoint.status === "failed" || checkpoint.status === "interrupted") {
      throw new Error(
        `Continuous autonomy stopped before the portfolio target: ${checkpoint.error}`
      );
    }
    const completedGoals = checkpoint.goal_dag.epochs.filter((epoch) => (
      epoch.status === "completed"
    )).length;
    return checkpoint.cycle_index >= minimumCycles
      && completedGoals >= minimumCompletedGoals;
  }, timeoutMs, "Continuous autonomy did not complete enough model-selected Goals");

  manager.stop(runId, "Continuous autonomy evidence target reached");
  await waitUntil(() => !manager.isActive(runId), 30_000,
    "Continuous autonomy did not reach a durable pause");
  const evidence = await inspectLiveRunEvidence({
    store,
    scenario,
    expectedStatus: "paused",
    requireMissionCompletion: false
  });
  assert.equal(evidence.run_mode, "continuous", "Live autonomy used finite Mission mode");
  assert.ok(evidence.cycle_count >= minimumCycles,
    "Continuous autonomy report lost completed cycles");
  assert.ok(evidence.completed_goal_count >= minimumCompletedGoals,
    "Continuous autonomy stopped after only one Goal");
  const report = { ...evidence, duration_ms: Date.now() - startedAt };
  if (reportPath) {
    await writeFile(resolve(reportPath), `${JSON.stringify(report, null, 2)}\n`, "utf8");
  }
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
} finally {
  if (runId && manager.isActive(runId)) {
    manager.stop(runId, "Continuous autonomy verifier is shutting down");
  }
  await manager.drain("Continuous autonomy verifier finished");
}

async function waitUntil(predicate, timeout, message) {
  const deadline = Date.now() + timeout;
  for (;;) {
    if (await predicate()) return;
    if (Date.now() >= deadline) throw new Error(message);
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 250));
  }
}

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
