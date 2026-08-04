import { spawn } from "node:child_process";
import { readFile, rm, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import { assertAutonomyPortfolio } from "../dist/runtime/autonomy-portfolio.js";
import { drawSeed } from "../dist/world/world-generator.js";

const runCount = optionalPositiveInteger("HEAR_LIVE_RUN_COUNT") ?? 3;
if (runCount < 2) throw new Error("HEAR_LIVE_RUN_COUNT must be at least two");
const seedStrategy = optionalText("HEAR_LIVE_SEED_STRATEGY") ?? "same";
if (seedStrategy !== "same" && seedStrategy !== "sequence") {
  throw new Error("HEAR_LIVE_SEED_STRATEGY must be same or sequence");
}
const baseSeed = optionalSeed("HEAR_LIVE_SEED") ?? drawSeed();
const reportPath = optionalText("HEAR_LIVE_REPORT");
const temporaryDirectory = await mkdtemp(join(tmpdir(), "hear-live-portfolio-"));
const startedAt = Date.now();
const reports = [];

try {
  for (let index = 0; index < runCount; index += 1) {
    const seed = seedStrategy === "same"
      ? baseSeed
      : (baseSeed + index) >>> 0;
    const childReportPath = join(temporaryDirectory, `run-${String(index + 1)}.json`);
    await runLiveMission({ seed, reportPath: childReportPath });
    reports.push(JSON.parse(await readFile(childReportPath, "utf8")));
  }
  const evaluation = assertAutonomyPortfolio(reports, {
    minimumRuns: runCount,
    requireSameInitialState: seedStrategy === "same"
  });
  const portfolio = {
    ...evaluation,
    status: "succeeded",
    base_seed: baseSeed,
    seed_strategy: seedStrategy,
    duration_ms: Date.now() - startedAt
  };
  await publishReport(portfolio);
} catch (error) {
  await publishReport({
    version: 1,
    status: "failed",
    base_seed: baseSeed,
    seed_strategy: seedStrategy,
    requested_run_count: runCount,
    completed_run_count: reports.length,
    duration_ms: Date.now() - startedAt,
    error: error instanceof Error ? error.message : String(error),
    runs: reports
  });
  throw error;
} finally {
  await rm(temporaryDirectory, { recursive: true, force: true });
}

async function runLiveMission(input) {
  await new Promise((resolvePromise, reject) => {
    const child = spawn(process.execPath, ["tests/live-continuous-autonomy.mjs"], {
      cwd: process.cwd(),
      env: {
        ...process.env,
        HEAR_LIVE_SEED: String(input.seed),
        HEAR_LIVE_REPORT: input.reportPath
      },
      stdio: "inherit"
    });
    child.once("error", reject);
    child.once("exit", (code, signal) => {
      if (code === 0) resolvePromise();
      else reject(new Error(
        `Continuous autonomy child exited with code ${String(code)} signal ${String(signal)}`
      ));
    });
  });
}

async function publishReport(report) {
  const serialized = `${JSON.stringify(report, null, 2)}\n`;
  if (reportPath) await writeFile(resolve(reportPath), serialized, "utf8");
  process.stdout.write(serialized);
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
