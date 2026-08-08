import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { extract } from "tar";

const options = parseOptions(process.argv.slice(2));
const session = options.session ?? `hear-g1-${randomUUID().slice(0, 8)}`;
const localArchive = resolve(
  options.output ?? `artifacts/training/${session}.tar.gz`
);
const localBundle = trainingBundlePath(localArchive);
const remoteArchive = `/content/${session}.tar.gz`;
if (existsSync(localArchive) || existsSync(localBundle)) {
  throw new Error(`Training output already exists: ${localArchive}`);
}
mkdirSync(dirname(localArchive), { recursive: true });

let primaryExit = 1;
try {
  primaryExit = command([
    "run",
    "--gpu",
    options.gpu ?? "H100",
    "--keep",
    "--timeout",
    String(options.timeoutSeconds ?? 14_400),
    "--session",
    session,
    resolve("training/colab_mjlab_g1.py"),
    "--iterations",
    String(options.iterations ?? 1000),
    "--num-envs",
    String(options.numEnvs ?? 4096),
    "--eval-envs",
    String(options.evalEnvs ?? 256),
    "--eval-steps",
    String(options.evalSteps ?? 600),
    "--archive",
    remoteArchive
  ]);
  if (primaryExit !== 0) process.exitCode = primaryExit;
  if (primaryExit === 0) {
    const downloadExit = command([
      "download",
      "--session",
      session,
      remoteArchive,
      localArchive
    ]);
    if (downloadExit !== 0) process.exitCode = downloadExit;
    if (downloadExit === 0) {
      await extractTrainingBundle(localArchive, localBundle);
      console.log(`Training bundle: ${localBundle}`);
    }
  }
} finally {
  const stopExit = command(["stop", "--session", session], true);
  if (primaryExit === 0 && stopExit !== 0 && process.exitCode === undefined) {
    process.exitCode = stopExit;
  }
}

function command(args, tolerateFailure = false) {
  const result = spawnSync("colab", args, {
    cwd: process.cwd(),
    stdio: "inherit",
    shell: false
  });
  if (result.error) {
    if (tolerateFailure) return 1;
    throw result.error;
  }
  return result.status ?? 1;
}

function parseOptions(args) {
  if (args[0] === "--") args = args.slice(1);
  const parsed = {};
  const names = new Map([
    ["--gpu", ["gpu", String]],
    ["--session", ["session", String]],
    ["--output", ["output", String]],
    ["--iterations", ["iterations", positiveInteger]],
    ["--num-envs", ["numEnvs", positiveInteger]],
    ["--eval-envs", ["evalEnvs", positiveInteger]],
    ["--eval-steps", ["evalSteps", positiveInteger]],
    ["--timeout-seconds", ["timeoutSeconds", positiveInteger]]
  ]);
  for (let index = 0; index < args.length; index += 2) {
    const name = args[index];
    const entry = names.get(name);
    const value = args[index + 1];
    if (!entry || value === undefined) {
      throw new Error(`Unknown or incomplete training option: ${name ?? ""}`);
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

async function extractTrainingBundle(archive, destination) {
  const files = new Set([
    "hear-g1-training/agent.yaml",
    "hear-g1-training/env.yaml",
    "hear-g1-training/g1_velocity.onnx",
    "hear-g1-training/g1_velocity.pt",
    "hear-g1-training/training-report.json"
  ]);
  mkdirSync(destination);
  await extract({
    file: archive,
    cwd: destination,
    strip: 1,
    filter: (path) => files.has(path)
  });
  for (const path of files) {
    const filename = path.slice(path.lastIndexOf("/") + 1);
    if (!existsSync(resolve(destination, filename))) {
      throw new Error(`Training bundle is missing: ${filename}`);
    }
  }
}

function trainingBundlePath(archive) {
  return archive.endsWith(".tar.gz")
    ? archive.slice(0, -".tar.gz".length)
    : `${archive}.bundle`;
}
