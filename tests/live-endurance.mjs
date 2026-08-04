import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";

import { loadEnvironment, loadProviderConfig, loadRuntimeCatalog } from "../dist/config/load.js";
import { resolveRunDirectory, RunStore } from "../dist/persistence/run-store.js";
import { RunManager } from "../dist/server/run-manager.js";
import { drawSeed } from "../dist/world/world-generator.js";
import { inspectLiveRunEvidence } from "./live-run-evidence.mjs";

loadEnvironment();

const scenarioId = optionalText("HEAR_LIVE_SCENARIO") ?? "humanoid_courtyard";
const seed = optionalSeed("HEAR_LIVE_SEED") ?? drawSeed();
const timeoutMs = optionalPositiveInteger("HEAR_LIVE_TIMEOUT_MS")
  ?? (optionalPositiveInteger("HEAR_LIVE_TIMEOUT_MINUTES") ?? 90) * 60_000;
const minimumCompactionsBeforeCrash = optionalPositiveInteger(
  "HEAR_LIVE_ENDURANCE_COMPACTIONS_BEFORE_CRASH"
) ?? 1;
const minimumCyclesBeforeCrash = optionalPositiveInteger(
  "HEAR_LIVE_ENDURANCE_CYCLES_BEFORE_CRASH"
) ?? 2;
const additionalCompactionsAfterResume = optionalPositiveInteger(
  "HEAR_LIVE_ENDURANCE_COMPACTIONS_AFTER_RESUME"
) ?? 1;
const additionalCyclesAfterResume = optionalPositiveInteger(
  "HEAR_LIVE_ENDURANCE_CYCLES_AFTER_RESUME"
) ?? 2;
const baseRunsDir = resolve(optionalText("HEAR_RUNS_DIR") ?? "runs");
const runsDir = join(baseRunsDir, `endurance-${Date.now().toString(36)}`);
const readyPath = join(runsDir, "worker-ready.json");
const reportPath = optionalText("HEAR_LIVE_REPORT");

await mkdir(runsDir, { recursive: true });
const [catalog, provider] = await Promise.all([
  loadRuntimeCatalog(),
  Promise.resolve(loadProviderConfig())
]);
const scenario = catalog.materialize(scenarioId, seed);
const startedAt = Date.now();
const child = spawn(process.execPath, ["tests/live-endurance-worker.mjs"], {
  cwd: process.cwd(),
  env: {
    ...process.env,
    HEAR_LIVE_SCENARIO: scenarioId,
    HEAR_LIVE_SEED: String(seed),
    HEAR_RUNS_DIR: runsDir,
    HEAR_LIVE_WORKER_READY: readyPath
  },
  stdio: ["inherit", "inherit", "inherit", "ipc"]
});
const childExit = childExitPromise(child);
let manager;
let runId;
try {
  runId = await waitForWorkerReady(readyPath, childExit, timeoutMs);
  const store = await RunStore.open(resolveRunDirectory(runsDir, runId));
  const beforeCrash = await waitForCheckpoint(store, timeoutMs, (checkpoint) => (
    checkpoint.cycle_index >= minimumCyclesBeforeCrash
      && checkpoint.context_memory.total_compactions >= minimumCompactionsBeforeCrash
      && completedGoalCount(checkpoint) >= 1
  ), "Endurance run did not compact context and complete a Goal before process loss");
  const manifestBefore = await store.readAgentManifest();

  child.send({ type: "crash" });
  const exit = await childExit;
  assert.equal(exit.code, 137, "Endurance worker did not perform an abrupt process exit");

  manager = new RunManager({ runsDir, catalog, provider });
  const recoveredCount = await manager.recoverOrphanedRuns();
  assert.equal(recoveredCount, 1, "New operator did not recover exactly one orphaned run");
  const interrupted = await store.readHumanoidCheckpoint();
  assert.equal(interrupted.status, "interrupted",
    "Abrupt process loss was not durably distinguished from an operator pause");

  assert.equal(await manager.resume(runId), runId, "Endurance recovery changed the Run identity");
  const targetCycleCount = beforeCrash.cycle_index + additionalCyclesAfterResume;
  const targetCompactionCount = beforeCrash.context_memory.total_compactions
    + additionalCompactionsAfterResume;
  const targetCompletedGoals = completedGoalCount(beforeCrash) + 1;
  const afterResume = await waitForCheckpoint(store, timeoutMs, (checkpoint) => (
    checkpoint.cycle_index >= targetCycleCount
      && checkpoint.context_memory.total_compactions >= targetCompactionCount
      && completedGoalCount(checkpoint) >= targetCompletedGoals
  ), "Recovered endurance run did not produce new Goals and compactions");

  manager.stop(runId, "Endurance recovery evidence target reached");
  await waitUntil(() => !manager.isActive(runId), 30_000,
    "Recovered endurance run did not reach a durable pause");
  const manifestAfter = await store.readAgentManifest();
  assert.equal(manifestAfter.identity_sha256, manifestBefore.identity_sha256,
    "Endurance recovery silently changed the Agent identity");
  assert.equal(manifestAfter.epoch_id, manifestBefore.epoch_id,
    "Endurance recovery silently replaced the Agent Session epoch");

  const evidence = await inspectLiveRunEvidence({
    store,
    scenario,
    expectedStatus: "paused",
    requireMissionCompletion: false
  });
  const report = {
    ...evidence,
    duration_ms: Date.now() - startedAt,
    endurance: {
      abrupt_exit_code: exit.code,
      recovered_orphan_count: recoveredCount,
      cycle_count_before_crash: beforeCrash.cycle_index,
      cycle_count_after_resume: afterResume.cycle_index,
      compaction_count_before_crash: beforeCrash.context_memory.total_compactions,
      compaction_count_after_resume: afterResume.context_memory.total_compactions,
      completed_goal_count_before_crash: completedGoalCount(beforeCrash),
      completed_goal_count_after_resume: completedGoalCount(afterResume),
      agent_manifest_identity_preserved: true
    }
  };
  if (reportPath) {
    await writeFile(resolve(reportPath), `${JSON.stringify(report, null, 2)}\n`, "utf8");
  }
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
} finally {
  if (child.exitCode === null && child.signalCode === null) child.kill();
  if (manager) {
    if (runId && manager.isActive(runId)) {
      manager.stop(runId, "Endurance verifier is shutting down");
    }
    await manager.drain("Endurance verifier finished");
  }
}

async function waitForWorkerReady(path, childExit, timeout) {
  const deadline = Date.now() + timeout;
  for (;;) {
    try {
      const parsed = JSON.parse(await readFile(path, "utf8"));
      if (typeof parsed.run_id !== "string" || !parsed.run_id) {
        throw new Error("Endurance worker published an invalid Run identity");
      }
      return parsed.run_id;
    } catch (error) {
      if (!(error instanceof Error && "code" in error && error.code === "ENOENT")) {
        throw error;
      }
    }
    const exited = await Promise.race([
      childExit.then((value) => value),
      new Promise((resolvePromise) => setTimeout(() => resolvePromise(null), 250))
    ]);
    if (exited) {
      throw new Error(
        `Endurance worker exited before readiness: ${exited.code}/${exited.signal}`
      );
    }
    if (Date.now() >= deadline) throw new Error("Endurance worker did not become ready");
  }
}

async function waitForCheckpoint(store, timeout, predicate, message) {
  let latest;
  await waitUntil(async () => {
    latest = await store.readHumanoidCheckpoint();
    if (latest.status === "failed" || latest.status === "paused") {
      throw new Error(`${message}: ${latest.error ?? latest.status}`);
    }
    return predicate(latest);
  }, timeout, message);
  return latest;
}

async function waitUntil(predicate, timeout, message) {
  const deadline = Date.now() + timeout;
  for (;;) {
    if (await predicate()) return;
    if (Date.now() >= deadline) throw new Error(message);
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 250));
  }
}

function childExitPromise(childProcess) {
  return new Promise((resolvePromise, reject) => {
    childProcess.once("error", reject);
    childProcess.once("exit", (code, signal) => resolvePromise({ code, signal }));
  });
}

function completedGoalCount(checkpoint) {
  return checkpoint.goal_dag.epochs.filter((epoch) => epoch.status === "completed").length;
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
