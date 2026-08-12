import { randomUUID } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  rmdirSync
} from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { spawn, spawnSync } from "node:child_process";
import { create } from "tar";

const REMOTE_BUNDLE = "/content/hear-workyard-smoke-bundle.tar.gz";
const REMOTE_REPORT = "/content/hear-workyard-smoke-report.json";
const BUNDLE_PATHS = [
  "training/workyard_mjlab_env.py",
  "training/workyard-task-v2.json",
  "assets/humanoid/g1/g1_with_hands.xml",
  "assets/humanoid/g1/meshes"
];

const options = parseOptions(process.argv.slice(2));
const workspace = process.cwd();
const session = options.session
  ?? `hear-workyard-smoke-${randomUUID().slice(0, 8)}`;
const distro = options.distro ?? "HEAR-Linux";
const colabPath = options.colabPath ?? "/home/hear/.local/bin/colab";
const output = resolve(
  options.output ?? `artifacts/training/workyard-smoke/${session}.json`
);
const temporaryDirectory = resolve(".tmp/workyard-colab");
const bundle = resolve(temporaryDirectory, `${session}.tar.gz`);
const bootstrap = resolve("training/colab_workyard_smoke.py");

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
  if (existsSync(output)) {
    throw new Error(`Workyard smoke output already exists: ${output}`);
  }
  mkdirSync(dirname(output), { recursive: true });
  mkdirSync(temporaryDirectory, { recursive: true });
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
    requireSuccess(await colab(distro, colabPath, [
      "new",
      "--session",
      session,
      "--gpu",
      options.gpu ?? "L4"
    ]), "create Colab session");
    requireSuccess(await colab(distro, colabPath, [
      "upload",
      "--session",
      session,
      toWslPath(bundle),
      REMOTE_BUNDLE
    ]), "upload Workyard bundle");

    const execution = await colab(distro, colabPath, [
      "exec",
      "--session",
      session,
      "--file",
      toWslPath(bootstrap),
      "--timeout",
      String(options.timeoutSeconds ?? 1_800)
    ]);
    const download = await colab(distro, colabPath, [
      "download",
      "--session",
      session,
      REMOTE_REPORT,
      toWslPath(output)
    ]);
    requireSuccess(download, "download Workyard smoke report");
    validateReport(output);
    requireSuccess(execution, "execute Workyard smoke");
  } catch (error) {
    failure = error;
  } finally {
    const stop = activeSession
      ? await colab(
        distro,
        colabPath,
        ["stop", "--session", activeSession],
        true
      )
      : 0;
    activeSession = null;
    rmSync(bundle, { force: true });
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
        ? new AggregateError([failure, cleanupError], "Workyard smoke and cleanup failed")
        : cleanupError;
    }
  }
  if (failure) throw failure;
  console.log(`Workyard smoke report: ${output}`);
}

function assertInputs() {
  for (const path of [...BUNDLE_PATHS, relative(workspace, bootstrap)]) {
    if (!existsSync(resolve(path))) {
      throw new Error(`Workyard smoke input is missing: ${path}`);
    }
  }
  if (!/^[A-Za-z0-9._-]+$/.test(session)) {
    throw new Error(`Unsafe Colab session name: ${session}`);
  }
}

function colab(distro, executable, args, tolerateFailure = false) {
  console.log(`[${session}] colab ${args.join(" ")}`);
  return new Promise((resolveStatus, reject) => {
    const child = spawn(
      "wsl.exe",
      ["-d", distro, "--", executable, ...args],
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
  if (status !== 0) {
    throw new Error(`Failed to ${operation} (exit ${status})`);
  }
}

function validateReport(path) {
  const report = JSON.parse(readFileSync(path, "utf8"));
  if (report.protocol !== "hear-workyard-colab-smoke-v1") {
    throw new Error("Downloaded file is not a Workyard smoke report");
  }
  if (report.ready !== true) {
    const detail = report.error?.message ?? "unknown remote failure";
    throw new Error(`Workyard Colab smoke failed: ${detail}`);
  }
  if (report.observation?.shape?.[1] !== 221
    || report.action?.shape?.[1] !== 37
    || report.observation?.teacher_state_directly_exposed !== false
    || report.rollout?.environment_closed !== true) {
    throw new Error("Workyard smoke report violates the deployment contract");
  }
}

function toWslPath(path) {
  const absolute = resolve(path).replaceAll("\\", "/");
  const match = /^([A-Za-z]):\/(.*)$/.exec(absolute);
  if (!match) {
    throw new Error(`Cannot map Windows path into WSL: ${path}`);
  }
  return `/mnt/${match[1].toLowerCase()}/${match[2]}`;
}

function emergencyStop(distro, executable, sessionName) {
  if (!sessionName || stopping) return;
  stopping = true;
  spawnSync(
    "wsl.exe",
    ["-d", distro, "--", executable, "stop", "--session", sessionName],
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
    ["--timeout-seconds", ["timeoutSeconds", positiveInteger]]
  ]);
  for (let index = 0; index < args.length; index += 2) {
    const name = args[index];
    const entry = names.get(name);
    const value = args[index + 1];
    if (!entry || value === undefined) {
      throw new Error(`Unknown or incomplete smoke option: ${name ?? ""}`);
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
