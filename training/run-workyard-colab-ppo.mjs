import { randomUUID } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  rmdirSync,
  writeFileSync
} from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { spawn, spawnSync } from "node:child_process";
import { create, extract } from "tar";

const REMOTE_BUNDLE = "/content/hear-workyard-baseline-bundle.tar.gz";
const REMOTE_CONFIG = "/content/hear-workyard-baseline-config.json";
const REMOTE_ARCHIVE = "/content/hear-workyard-baseline-artifacts.tar.gz";
const REMOTE_REPORT = "/content/hear-workyard-baseline-report.json";
const BUNDLE_PATHS = [
  "training/workyard_mjlab_env.py",
  "training/workyard-task-v2.json",
  "assets/humanoid/g1/g1_with_hands.xml",
  "assets/humanoid/g1/meshes"
];
const ARTIFACT_FILES = new Set([
  "hear-workyard-baseline/agent.yaml",
  "hear-workyard-baseline/env.yaml",
  "hear-workyard-baseline/training-curves.json",
  "hear-workyard-baseline/training-report.json",
  "hear-workyard-baseline/workyard_ppo_baseline.pt"
]);

const options = parseOptions(process.argv.slice(2));
const workspace = process.cwd();
const session = options.session
  ?? `hear-workyard-ppo-${randomUUID().slice(0, 8)}`;
const distro = options.distro ?? "HEAR-Linux";
const colabPath = options.colabPath ?? "/home/hear/.local/bin/colab";
const outputDirectory = resolve(
  options.output ?? `artifacts/training/workyard-baseline/${session}`
);
const localArchive = `${outputDirectory}.tar.gz`;
const localReport = `${outputDirectory}-remote-report.json`;
const temporaryDirectory = resolve(".tmp/workyard-colab");
const bundle = resolve(temporaryDirectory, `${session}.tar.gz`);
const config = resolve(temporaryDirectory, `${session}.json`);
const bootstrap = resolve("training/colab_workyard_ppo.py");

let activeSession = null;
let stopping = false;

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    emergencyStop(distro, colabPath, activeSession);
    process.exit(signal === "SIGINT" ? 130 : 143);
  });
}

await main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});

async function main() {
  assertInputs();
  for (const path of [outputDirectory, localArchive, localReport]) {
    if (existsSync(path)) throw new Error(`Training output already exists: ${path}`);
  }
  mkdirSync(dirname(outputDirectory), { recursive: true });
  mkdirSync(temporaryDirectory, { recursive: true });
  writeFileSync(config, JSON.stringify({
    iterations: options.iterations ?? 20,
    num_envs: options.numEnvs ?? 64,
    eval_envs_per_stage: options.evalEnvsPerStage ?? 8,
    eval_steps: options.evalSteps ?? 300,
    seed: options.seed ?? 42,
    archive: REMOTE_ARCHIVE
  }), "utf8");
  await create({
    cwd: workspace,
    file: bundle,
    gzip: true,
    portable: true,
    onWriteEntry(entry) {
      entry.path = entry.path.replace(
        /^assets\/humanoid\/g1\/meshes(?=\/|$)/,
        "assets/humanoid/g1/assets"
      );
    }
  }, BUNDLE_PATHS);

  let failure;
  try {
    activeSession = session;
    requireSuccess(await colab([
      "new", "--session", session, "--gpu", options.gpu ?? "L4"
    ]), "create Colab session");
    requireSuccess(await colab([
      "upload", "--session", session, toWslPath(bundle), REMOTE_BUNDLE
    ]), "upload Workyard training bundle");
    requireSuccess(await colab([
      "upload", "--session", session, toWslPath(config), REMOTE_CONFIG
    ]), "upload Workyard training configuration");
    const execution = await colab([
      "exec",
      "--session",
      session,
      "--file",
      toWslPath(bootstrap),
      "--timeout",
      String(options.timeoutSeconds ?? 3_600)
    ]);
    requireSuccess(await colab([
      "download", "--session", session, REMOTE_REPORT, toWslPath(localReport)
    ]), "download Workyard PPO report");
    validateReport(localReport);
    requireSuccess(execution, "execute Workyard PPO baseline");
    requireSuccess(await colab([
      "download", "--session", session, REMOTE_ARCHIVE, toWslPath(localArchive)
    ]), "download Workyard PPO artifacts");
    await extractArtifacts(localArchive, outputDirectory);
  } catch (error) {
    failure = error;
  } finally {
    const stop = activeSession
      ? await colab(["stop", "--session", activeSession], true)
      : 0;
    activeSession = null;
    rmSync(bundle, { force: true });
    rmSync(config, { force: true });
    try {
      rmdirSync(temporaryDirectory);
    } catch {
      // The shared ignored directory may contain another task's bundle.
    }
    if (stop !== 0) {
      const cleanupError = new Error(
        `Failed to stop Colab session ${session}; run colab stop manually`
      );
      failure = failure
        ? new AggregateError([failure, cleanupError], "PPO run and cleanup failed")
        : cleanupError;
    }
  }
  if (failure) throw failure;
  console.log(`Workyard PPO baseline: ${outputDirectory}`);
}

function assertInputs() {
  for (const path of [...BUNDLE_PATHS, relative(workspace, bootstrap)]) {
    if (!existsSync(resolve(path))) {
      throw new Error(`Workyard PPO input is missing: ${path}`);
    }
  }
  if (!/^[A-Za-z0-9._-]+$/.test(session)) {
    throw new Error(`Unsafe Colab session name: ${session}`);
  }
}

function colab(args, tolerateFailure = false) {
  console.log(`[${session}] colab ${args.join(" ")}`);
  return new Promise((resolveStatus, reject) => {
    const child = spawn(
      "wsl.exe",
      ["-d", distro, "--", colabPath, ...args],
      { cwd: workspace, shell: false, stdio: "inherit" }
    );
    child.once("error", reject);
    child.once("exit", (code, signal) => {
      const status = code ?? (signal ? 1 : 0);
      if (!tolerateFailure && signal) {
        reject(new Error(`Colab command terminated by ${signal}`));
        return;
      }
      resolveStatus(status);
    });
  });
}

function requireSuccess(status, operation) {
  if (status !== 0) throw new Error(`Failed to ${operation} (exit ${status})`);
}

function validateReport(path) {
  const report = JSON.parse(readFileSync(path, "utf8"));
  if (report.protocol !== "hear-workyard-ppo-baseline-v1") {
    throw new Error("Downloaded file is not a Workyard PPO baseline report");
  }
  if (report.ready !== true) {
    const detail = report.error?.message ?? "unknown remote failure";
    throw new Error(`Workyard PPO baseline failed: ${detail}`);
  }
  if (report.training?.environment_closed !== true
    || report.evaluation?.environment_closed !== true
    || report.evaluation?.finite !== true
    || report.evaluation?.stage_metrics?.length !== 6) {
    throw new Error("Workyard PPO report is missing required physical evidence");
  }
}

async function extractArtifacts(archive, destination) {
  mkdirSync(destination);
  await extract({
    file: archive,
    cwd: destination,
    strip: 1,
    filter: (path) => ARTIFACT_FILES.has(path)
  });
  for (const path of ARTIFACT_FILES) {
    const filename = path.slice(path.lastIndexOf("/") + 1);
    if (!existsSync(resolve(destination, filename))) {
      throw new Error(`Workyard PPO artifact is missing: ${filename}`);
    }
  }
}

function toWslPath(path) {
  const absolute = resolve(path).replaceAll("\\", "/");
  const match = /^([A-Za-z]):\/(.*)$/.exec(absolute);
  if (!match) throw new Error(`Cannot map Windows path into WSL: ${path}`);
  return `/mnt/${match[1].toLowerCase()}/${match[2]}`;
}

function emergencyStop(distroName, executable, sessionName) {
  if (!sessionName || stopping) return;
  stopping = true;
  spawnSync(
    "wsl.exe",
    ["-d", distroName, "--", executable, "stop", "--session", sessionName],
    { cwd: workspace, shell: false, stdio: "inherit", timeout: 60_000 }
  );
}

function parseOptions(args) {
  if (args[0] === "--") args = args.slice(1);
  const parsed = {};
  const names = new Map([
    ["--distro", ["distro", String]],
    ["--colab-path", ["colabPath", String]],
    ["--gpu", ["gpu", String]],
    ["--session", ["session", String]],
    ["--output", ["output", String]],
    ["--iterations", ["iterations", positiveInteger]],
    ["--num-envs", ["numEnvs", positiveInteger]],
    ["--eval-envs-per-stage", ["evalEnvsPerStage", positiveInteger]],
    ["--eval-steps", ["evalSteps", positiveInteger]],
    ["--seed", ["seed", nonNegativeInteger]],
    ["--timeout-seconds", ["timeoutSeconds", positiveInteger]]
  ]);
  for (let index = 0; index < args.length; index += 2) {
    const name = args[index];
    const entry = names.get(name);
    const value = args[index + 1];
    if (!entry || value === undefined) {
      throw new Error(`Unknown or incomplete PPO option: ${name ?? ""}`);
    }
    parsed[entry[0]] = entry[1](value);
  }
  return parsed;
}

function positiveInteger(value) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new Error(`Expected a positive integer, received: ${value}`);
  }
  return parsed;
}

function nonNegativeInteger(value) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 0) {
    throw new Error(`Expected a non-negative integer, received: ${value}`);
  }
  return parsed;
}
