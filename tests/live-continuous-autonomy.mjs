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

const requestedScenarioId = optionalText("HEAR_LIVE_SCENARIO")
  ?? "humanoid_courtyard";
const requestedSeed = optionalSeed("HEAR_LIVE_SEED") ?? drawSeed();
const resumeRunId = optionalText("HEAR_LIVE_RESUME_RUN_ID");
const freshAgentEpoch = optionalBoolean("HEAR_LIVE_FRESH_AGENT_EPOCH") ?? false;
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
const manager = new RunManager({
  runsDir,
  catalog,
  provider,
  ...(controllerSource ? { controllerSource } : {})
});
let runId;
let store;
let scenario;
let retainedControllerSourceSha256;

try {
  if (resumeRunId) {
    store = await RunStore.open(resolveRunDirectory(runsDir, resumeRunId));
    assert.equal(store.definition.run_mode, "continuous",
      "Only a continuous run can be resumed by the continuous observer");
    retainedControllerSourceSha256 = store.definition.controller_source_sha256;
    scenario = structuredClone(store.definition.scenario);
    runId = await manager.resume(resumeRunId, {
      ...(freshAgentEpoch ? { freshAgentEpoch: true } : {})
    });
  } else {
    scenario = catalog.materialize(requestedScenarioId, requestedSeed);
    runId = await manager.start({
      mission: "在当前世界持续自主观察、选择目标并通过真实物理执行自由活动。",
      scenarioId: requestedScenarioId,
      goal: scenario.default_goal,
      runMode: "continuous",
      seed: requestedSeed
    });
    store = await RunStore.open(resolveRunDirectory(runsDir, runId));
  }
  if (resumeRunId) {
    assert.equal(
      store.definition.controller_source_sha256,
      retainedControllerSourceSha256,
      "Continuous resume changed the run's immutable humanoid controller source"
    );
  } else {
    assert.equal(
      store.definition.controller_source_sha256,
      controllerSource?.sourceSha256,
      "Continuous run did not retain the configured humanoid controller source"
    );
  }
  await observeContinuousRun(
    manager,
    store,
    runId,
    observationMs,
    resumeRunId !== undefined
  );

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

async function observeContinuousRun(
  manager,
  store,
  runId,
  durationMs,
  resumed
) {
  const deadline = Date.now() + durationMs;
  const startupDeadline = Math.min(deadline, Date.now() + 2 * 60_000);
  let enteredRunning = false;
  while (Date.now() < deadline) {
    const checkpoint = await store.readHumanoidCheckpoint();
    const awaitingStartup = !enteredRunning && (
      checkpoint.status === "starting"
        || resumed && (
          checkpoint.status === "paused"
            || checkpoint.status === "interrupted"
        )
    );
    if (awaitingStartup) {
      if (Date.now() >= startupDeadline) {
        throw new Error("Continuous run did not leave starting state within 2 minutes");
      }
      await delay(Math.min(250, Math.max(1, deadline - Date.now())));
      continue;
    }
    if (checkpoint.status !== "running") {
      throw new Error(
        `Continuous run stopped without an operator request: ${checkpoint.error ?? checkpoint.status}`
      );
    }
    enteredRunning = true;
    if (!manager.isActive(runId)) {
      throw new Error("Continuous runtime exited while its checkpoint remained nonterminal");
    }
    await delay(Math.min(1_000, Math.max(1, deadline - Date.now())));
  }
  if (!enteredRunning) {
    throw new Error("Continuous run never entered running state");
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

function optionalBoolean(name) {
  const value = optionalText(name)?.toLowerCase();
  if (value === undefined) return undefined;
  if (["1", "true", "yes", "on"].includes(value)) return true;
  if (["0", "false", "no", "off"].includes(value)) return false;
  throw new Error(`${name} must be a boolean`);
}
