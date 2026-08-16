import { spawn } from "node:child_process";
import {
  closeSync,
  existsSync,
  mkdirSync,
  openSync,
  readFileSync,
  rmSync,
  writeFileSync
} from "node:fs";
import { dirname, relative, resolve } from "node:path";

const options = parseOptions(process.argv.slice(2));
const workspace = process.cwd();
const upstreamStatus = options.upstreamStatus
  ? insideWorkspace(options.upstreamStatus)
  : undefined;
const statusPath = insideWorkspace(
  options.status ?? ".tmp/g1-getup-pipeline/status.json"
);
const lockPath = insideWorkspace(".tmp/g1-getup-pipeline/active.lock");
const output = insideWorkspace(
  options.output ?? "artifacts/training/g1-getup/formal-v1"
);
const session = options.session ?? "hear-g1-getup-formal-v1";
const desktopDriveRoot = options.desktopDriveRoot
  ?? process.env.HEAR_G1_GETUP_DESKTOP_DRIVE_ROOT?.trim();
const waitTimeoutMs = (options.waitTimeoutSeconds ?? 64_800) * 1000;
const pollMs = (options.pollSeconds ?? 120) * 1000;
const node = process.execPath;
const tsx = insideWorkspace("node_modules/tsx/dist/cli.mjs");

let activeChild;
let interrupted = false;
let lockOwned = false;

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    interrupted = true;
    activeChild?.kill(signal);
  });
}

await main().catch((error) => {
  writeStatus("failed", {
    error: error instanceof Error ? error.message : String(error)
  });
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = interrupted ? 130 : 1;
}).finally(releaseLock);

async function main() {
  acquireLock();
  if (!/^[A-Za-z0-9._-]+$/.test(session)) {
    throw new Error(`Unsafe G1 get-up pipeline session: ${session}`);
  }
  if (upstreamStatus) {
    writeStatus("waiting_for_colab_training_lane", {
      upstream_status: repositoryRelative(upstreamStatus)
    });
    await waitForUpstreamPipeline(upstreamStatus);
  } else {
    writeStatus("colab_training_lane_available", {
      lane: "dedicated_g1_session"
    });
  }

  const reportPath = resolve(output, "getup-policy-report.json");
  if (!qualifiedOutput(output)) {
    if (existsSync(output) || existsSync(`${output}.tar.gz`)) {
      throw new Error(`Incomplete G1 get-up output already exists: ${output}`);
    }
    writeStatus("getup_training", { session });
    await run(node, [
      "training/run-g1-getup-colab.mjs",
      "--mode", "train",
      "--session", session,
      "--output", repositoryRelative(output),
      "--drive-directory", `HEAR/g1-getup/${session}`,
      "--gpu", options.gpu ?? "L4",
      "--iterations", String(options.iterations ?? 5000),
      "--num-envs", String(options.numEnvs ?? 2048),
      "--eval-envs", String(options.evalEnvs ?? 500),
      "--eval-steps", String(options.evalSteps ?? 750),
      "--timeout-seconds", String(options.trainingTimeoutSeconds ?? 64_800),
      ...(desktopDriveRoot
        ? ["--desktop-drive-root", desktopDriveRoot]
        : [])
    ]);
    if (!qualifiedOutput(output)) {
      throw new Error("G1 get-up trainer returned without a qualified policy");
    }
  } else {
    writeStatus("getup_training_already_complete");
  }

  const runtimeQualification = resolve(
    output, "runtime-deployment-report.json"
  );
  writeStatus("getup_runtime_qualification", {
    runtime: "typescript-mujoco-onnxruntime-web",
    report: repositoryRelative(runtimeQualification)
  });
  await run(node, [
    repositoryRelative(tsx),
    "src/training/g1-getup-deployment-cli.ts",
    "--output", repositoryRelative(runtimeQualification),
    "--policy-directory", repositoryRelative(output)
  ]);
  const qualification = readJson(runtimeQualification);
  if (qualification?.protocol
      !== "hear-typescript-mujoco-g1-getup-deployment-gate-v1"
    || qualification?.accepted !== true
    || qualification?.summary?.recovered_count !== 4) {
    throw new Error("G1 get-up policy failed its production runtime qualification");
  }
  writeStatus("getup_policy_install", {
    source: repositoryRelative(output)
  });
  await run(node, [
    "training/install-g1-getup-policy-assets.mjs",
    "--source-root", repositoryRelative(output)
  ]);
  if (!existsSync(resolve(
    workspace, "assets/humanoid/controllers/g1-getup/g1_getup.onnx"
  ))) {
    throw new Error("G1 get-up policy installation produced no runtime asset");
  }
  writeStatus("completed", {
    source: repositoryRelative(output),
    report: repositoryRelative(reportPath),
    runtime_qualification: repositoryRelative(runtimeQualification),
    installed: "assets/humanoid/controllers/g1-getup"
  });
}

async function waitForUpstreamPipeline(path) {
  const started = Date.now();
  while (true) {
    if (interrupted) throw new Error("G1 get-up pipeline interrupted");
    const status = readJson(path);
    if (status?.protocol === "hear-workyard-autonomy-pipeline-status-v1"
      && (status.stage === "completed" || status.stage === "failed")) {
      writeStatus("colab_training_lane_available", {
        upstream_stage: status.stage,
        upstream_updated_at: status.updated_at
      });
      return;
    }
    if (Date.now() - started >= waitTimeoutMs) {
      throw new Error(
        `Workyard Colab pipeline did not release the lane within ${waitTimeoutMs / 1000}s`
      );
    }
    await delay(pollMs);
  }
}

function qualifiedOutput(root) {
  const report = readJson(resolve(root, "getup-policy-report.json"));
  const onnx = resolve(root, "g1_getup.onnx");
  return existsSync(onnx)
    && report?.protocol === "hear-g1-getup-policy-deployment-v1"
    && report?.evaluation?.deployment_accepted === true
    && report?.evaluation?.episode_count >= 500
    && report?.evaluation?.overall_success_rate >= 0.80
    && report?.evaluation?.prone_success_rate >= 0.75
    && report?.evaluation?.supine_success_rate >= 0.75
    && report?.evaluation?.side_success_rate >= 0.75
    && report?.evaluation?.non_finite_action_count === 0;
}

function run(command, args) {
  return new Promise((resolvePromise, reject) => {
    activeChild = spawn(command, args, {
      cwd: workspace,
      stdio: "inherit",
      shell: false,
      windowsHide: true
    });
    activeChild.once("error", reject);
    activeChild.once("exit", (code, signal) => {
      activeChild = undefined;
      if (code === 0) resolvePromise();
      else reject(new Error(
        `${command} ${args[0] ?? ""} exited ${code ?? signal ?? "unknown"}`
      ));
    });
  });
}

function acquireLock() {
  mkdirSync(dirname(lockPath), { recursive: true });
  let descriptor;
  try {
    descriptor = openSync(lockPath, "wx");
  } catch (error) {
    if (error?.code === "EEXIST") {
      throw new Error(`G1 get-up pipeline lock already exists: ${lockPath}`);
    }
    throw error;
  }
  writeFileSync(descriptor, `${process.pid}\n`, "utf8");
  closeSync(descriptor);
  lockOwned = true;
}

function releaseLock() {
  if (!lockOwned) return;
  rmSync(lockPath, { force: true });
  lockOwned = false;
}

function writeStatus(stage, details = {}) {
  mkdirSync(dirname(statusPath), { recursive: true });
  writeFileSync(statusPath, `${JSON.stringify({
    protocol: "hear-g1-getup-pipeline-status-v1",
    pid: process.pid,
    stage,
    updated_at: new Date().toISOString(),
    ...details
  }, null, 2)}\n`, "utf8");
}

function readJson(path) {
  try { return JSON.parse(readFileSync(path, "utf8")); } catch { return null; }
}

function insideWorkspace(path) {
  const absolute = resolve(path);
  const child = relative(workspace, absolute);
  if (!child || child === ".." || child.startsWith("../") || child.startsWith("..\\")) {
    throw new Error(`Path must be inside the workspace: ${path}`);
  }
  return absolute;
}

function repositoryRelative(path) {
  return relative(workspace, path).replaceAll("\\", "/");
}

function delay(milliseconds) {
  return new Promise((resolvePromise) => setTimeout(resolvePromise, milliseconds));
}

function parseOptions(args) {
  if (args[0] === "--") args = args.slice(1);
  const parsed = {};
  const names = new Map([
    ["--upstream-status", ["upstreamStatus", String]],
    ["--status", ["status", String]],
    ["--output", ["output", String]],
    ["--session", ["session", String]],
    ["--gpu", ["gpu", String]],
    ["--iterations", ["iterations", positiveInteger]],
    ["--num-envs", ["numEnvs", positiveInteger]],
    ["--eval-envs", ["evalEnvs", positiveInteger]],
    ["--eval-steps", ["evalSteps", positiveInteger]],
    ["--wait-timeout-seconds", ["waitTimeoutSeconds", positiveInteger]],
    ["--poll-seconds", ["pollSeconds", positiveInteger]],
    ["--training-timeout-seconds", ["trainingTimeoutSeconds", positiveInteger]],
    ["--desktop-drive-root", ["desktopDriveRoot", String]]
  ]);
  for (let index = 0; index < args.length; index += 2) {
    const entry = names.get(args[index]);
    const value = args[index + 1];
    if (!entry || value === undefined) {
      throw new Error(`Unknown or incomplete G1 pipeline option: ${args[index] ?? ""}`);
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
