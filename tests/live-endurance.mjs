import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";

import {
  loadEnvironment,
  loadProviderConfig,
  loadRuntimeCatalog
} from "../dist/config/load.js";
import { resolveRunDirectory, RunStore } from "../dist/persistence/run-store.js";
import { RunManager } from "../dist/server/run-manager.js";
import { drawSeed } from "../dist/world/world-generator.js";

loadEnvironment();

const scenarioId = optionalText("HEAR_LIVE_SCENARIO") ?? "humanoid_courtyard";
const seed = optionalSeed("HEAR_LIVE_SEED") ?? drawSeed();
const observationMs = optionalPositiveInteger("HEAR_LIVE_ENDURANCE_OBSERVATION_MS")
  ?? (optionalPositiveInteger("HEAR_LIVE_TIMEOUT_MINUTES") ?? 30) * 60_000;
const baseRunsDir = resolve(optionalText("HEAR_RUNS_DIR") ?? "runs");
const runsDir = join(baseRunsDir, `endurance-${Date.now().toString(36)}`);
const readyPath = join(runsDir, "worker-ready.json");
const reportPath = optionalText("HEAR_LIVE_REPORT");

await mkdir(runsDir, { recursive: true });
const [catalog, provider] = await Promise.all([
  loadRuntimeCatalog(),
  Promise.resolve(loadProviderConfig())
]);
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
  runId = await waitForWorkerReady(readyPath, childExit, 10 * 60_000);
  const store = await RunStore.open(resolveRunDirectory(runsDir, runId));
  await observeWorkerRun(store, childExit, observationMs);
  const manifestBefore = await store.readAgentManifest();

  child.send({ type: "crash" });
  const exit = await childExit;
  assert.equal(exit.code, 137, "Endurance worker did not perform an abrupt process exit");

  manager = new RunManager({ runsDir, catalog, provider });
  const recoveredCount = await manager.recoverOrphanedRuns();
  assert.equal(recoveredCount, 1, "New operator did not recover the orphaned run");
  const interrupted = await store.readHumanoidCheckpoint();
  assert.equal(interrupted.status, "interrupted",
    "Abrupt process loss was not distinguished from an operator pause");

  assert.equal(await manager.resume(runId), runId, "Recovery changed the Run identity");
  await observeResumedRun(manager, store, runId, observationMs);

  manager.stop(runId, "Endurance observation window ended");
  await waitUntil(() => !manager.isActive(runId), 10 * 60_000,
    "Recovered run did not accept an operator pause");
  const checkpoint = await store.readHumanoidCheckpoint();
  assert.equal(checkpoint.status, "paused", "Recovered run did not pause cleanly");
  assert.equal(checkpoint.error, null, "Recovered run recorded the operator pause as a failure");

  const manifestAfter = await store.readAgentManifest();
  assert.equal(manifestAfter.identity_sha256, manifestBefore.identity_sha256,
    "Recovery silently changed the Agent identity");
  assert.equal(manifestAfter.epoch_id, manifestBefore.epoch_id,
    "Recovery silently replaced the Agent Session epoch");

  const report = {
    version: 1,
    run_id: runId,
    scenario_id: scenarioId,
    seed,
    status: checkpoint.status,
    duration_ms: Date.now() - startedAt,
    process_loss: { exit_code: exit.code },
    recovery: {
      orphan_recovered: recoveredCount === 1,
      run_identity_preserved: true,
      agent_identity_preserved: true
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
      manager.stop(runId, "Endurance observer is shutting down");
    }
    await manager.drain("Endurance observer finished");
  }
}

async function observeWorkerRun(store, childExit, durationMs) {
  const deadline = Date.now() + durationMs;
  while (Date.now() < deadline) {
    const checkpoint = await store.readHumanoidCheckpoint();
    if (checkpoint.status !== "running") {
      throw new Error(
        `Endurance worker stopped before process loss: ${checkpoint.error ?? checkpoint.status}`
      );
    }
    const exited = await Promise.race([
      childExit,
      delay(Math.min(1_000, Math.max(1, deadline - Date.now()))).then(() => null)
    ]);
    if (exited) {
      throw new Error(`Endurance worker exited early: ${exited.code}/${exited.signal}`);
    }
  }
}

async function observeResumedRun(manager, store, runId, durationMs) {
  const deadline = Date.now() + durationMs;
  while (Date.now() < deadline) {
    const checkpoint = await store.readHumanoidCheckpoint();
    if (checkpoint.status !== "running") {
      throw new Error(
        `Recovered run stopped without an operator request: ${checkpoint.error ?? checkpoint.status}`
      );
    }
    if (!manager.isActive(runId)) {
      throw new Error("Recovered runtime exited while its checkpoint remained nonterminal");
    }
    await delay(Math.min(1_000, Math.max(1, deadline - Date.now())));
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
    const exited = await Promise.race([childExit, delay(250).then(() => null)]);
    if (exited) {
      throw new Error(`Endurance worker exited before readiness: ${exited.code}/${exited.signal}`);
    }
    if (Date.now() >= deadline) throw new Error("Endurance worker did not become ready");
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

function childExitPromise(childProcess) {
  return new Promise((resolvePromise, reject) => {
    childProcess.once("error", reject);
    childProcess.once("exit", (code, signal) => resolvePromise({ code, signal }));
  });
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
