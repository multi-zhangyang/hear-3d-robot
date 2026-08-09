import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

import {
  loadEnvironment,
  loadProviderConfig,
  loadRuntimeCatalog
} from "../dist/config/load.js";
import { resolveRunDirectory, RunStore } from "../dist/persistence/run-store.js";
import { RunManager } from "../dist/server/run-manager.js";
import { loadConfiguredHumanoidControllerSource } from
  "../dist/world/humanoid/controller-module.js";
import { drawSeed } from "../dist/world/world-generator.js";
import { inspectLiveRunEvidence } from "./live-run-evidence.mjs";

loadEnvironment();

const scenarioId = optionalText("HEAR_LIVE_SCENARIO") ?? "humanoid_courtyard";
const seed = optionalSeed("HEAR_LIVE_SEED") ?? drawSeed();
const observationMs = optionalPositiveInteger("HEAR_LIVE_OBSERVATION_MS")
  ?? (optionalPositiveInteger("HEAR_LIVE_TIMEOUT_MINUTES") ?? 30) * 60_000;
const runsDir = resolve(optionalText("HEAR_RUNS_DIR") ?? "runs");
const reportPath = optionalText("HEAR_LIVE_REPORT");

await mkdir(runsDir, { recursive: true });
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
let runId;

try {
  runId = await manager.start({
    mission: "在当前世界持续自主观察、选择目标并通过真实物理执行自由活动。",
    scenarioId,
    goal: scenario.default_goal,
    runMode: "continuous",
    seed
  });
  const store = await RunStore.open(resolveRunDirectory(runsDir, runId));
  assert.equal(
    store.definition.controller_source_sha256,
    controllerSource?.sourceSha256,
    "Continuous run did not retain the configured humanoid controller source"
  );
  await observeContinuousRun(manager, store, runId, observationMs);

  manager.stop(runId, "Continuous observation window ended");
  await waitUntil(() => !manager.isActive(runId), 10 * 60_000,
    "Continuous run did not accept an operator pause");
  const report = await inspectLiveRunEvidence({
    store,
    scenario,
    expectedStatus: "paused",
    requireMissionCompletion: false
  });
  if (reportPath) {
    await writeFile(resolve(reportPath), `${JSON.stringify(report, null, 2)}\n`, "utf8");
  }
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
} finally {
  if (runId && manager.isActive(runId)) {
    manager.stop(runId, "Continuous observer is shutting down");
  }
  await manager.drain("Continuous observer finished");
}

async function observeContinuousRun(manager, store, runId, durationMs) {
  const deadline = Date.now() + durationMs;
  while (Date.now() < deadline) {
    const checkpoint = await store.readHumanoidCheckpoint();
    if (checkpoint.status !== "running") {
      throw new Error(
        `Continuous run stopped without an operator request: ${checkpoint.error ?? checkpoint.status}`
      );
    }
    if (!manager.isActive(runId)) {
      throw new Error("Continuous runtime exited while its checkpoint remained nonterminal");
    }
    await delay(Math.min(1_000, Math.max(1, deadline - Date.now())));
  }
}

async function waitUntil(predicate, timeout, message) {
  const deadline = Date.now() + timeout;
  for (;;) {
    if (await predicate()) return;
    if (Date.now() >= deadline) throw new Error(message);
    await delay(250);
  }
}

function delay(milliseconds) {
  return new Promise((resolvePromise) => setTimeout(resolvePromise, milliseconds));
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
