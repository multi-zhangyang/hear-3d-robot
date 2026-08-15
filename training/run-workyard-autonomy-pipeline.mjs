import { createHash, randomUUID } from "node:crypto";
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
const reachSource = insideWorkspace(
  options.reachSource
    ?? "artifacts/training/workyard-residual/whole-body-v5-formal-20260815"
);
const reachCandidate = insideWorkspace(
  options.reachCandidate ?? "artifacts/training/workyard-reach-candidate-v3"
);
const reachDeployment = insideWorkspace(
  options.reachDeployment ?? "artifacts/training/workyard-reach-deployment-v3"
);
const contactPilot = insideWorkspace(
  options.contactPilot ?? "artifacts/training/workyard-contact-pilot/autonomy-v2"
);
const contactTraining = insideWorkspace(
  options.contactTraining ?? "artifacts/training/workyard-contact/formal-v3"
);
const contactDeployment = insideWorkspace(
  options.contactDeployment
    ?? "artifacts/training/workyard-contact-deployment/formal-v3"
);
const contractPath = insideWorkspace("training/workyard-contact-task-v1.json");
const statusPath = insideWorkspace(
  options.status ?? ".tmp/workyard-autonomy-pipeline/status.json"
);
const lockPath = insideWorkspace(".tmp/workyard-autonomy-pipeline/active.lock");
const sessionPrefix = options.sessionPrefix ?? `hear-autonomy-${dateStamp()}`;
const reachTrainingTimeoutSeconds = options.reachTrainingTimeoutSeconds
  ?? options.waitTimeoutSeconds
  ?? 21_600;
const reachDriveDirectory = options.reachDriveDirectory
  ?? "HEAR/workyard-residual/whole-body-v5-formal";
const driveDirectory = options.driveDirectory
  ?? "HEAR/workyard-contact/formal-v3";
const driveLocalRoot = options.driveLocalRoot
  ? resolve(options.driveLocalRoot)
  : undefined;
const pnpm = process.platform === "win32" ? "pnpm.cmd" : "pnpm";

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
  validateConfiguration();
  const reachFormalArgs = [
    "training/run-workyard-colab-residual.mjs",
    "--mode", "train",
    "--session", `${sessionPrefix}-reach-formal`,
    "--output", repositoryRelative(reachSource),
    "--drive-directory", reachDriveDirectory,
    "--gpu", "L4",
    "--iterations", "1000",
    "--dagger-steps", "400",
    "--dagger-learning-rate", "0.0003",
    "--dagger-beta-initial", "1",
    "--dagger-beta-final", "0.1",
    "--ppo-retention-mode", "critic_warmup_rollout_teacher",
    "--critic-warmup-iterations", "5",
    "--ppo-actor-learning-rate", "0.0001",
    "--ppo-critic-learning-rate", "0.0003",
    "--rollout-teacher-coefficient", "1",
    "--teacher-maximum-action-std", "0.15",
    "--teacher-dispersion-coefficient", "1",
    "--num-envs", "2048",
    "--rollout-steps", "64",
    "--eval-envs", "500",
    "--eval-steps", "600",
    "--seed", "42",
    "--timeout-seconds", String(reachTrainingTimeoutSeconds)
  ];
  if (driveLocalRoot) {
    reachFormalArgs.push("--drive-local-root", driveLocalRoot);
  }
  await stage("reach_formal_training", reachTrainingReady, [
    reachSource,
    `${reachSource}-remote-report.json`,
    `${reachSource}.tar.gz`
  ], () => run(process.execPath, reachFormalArgs));

  await stage("reach_export", reachCandidateReady, [reachCandidate], () => run(
    process.execPath,
    [
      "training/run-workyard-reach-export-colab.mjs",
      "--source-root", repositoryRelative(reachSource),
      "--output", repositoryRelative(reachCandidate),
      "--session", `${sessionPrefix}-reach-export`
    ]
  ));

  await stage("reach_deployment_qualification", reachDeploymentReady,
    [reachDeployment], () => run(pnpm, [
      "exec", "tsx", "src/training/workyard-reach-deployment-cli.ts",
      "--candidate-root", repositoryRelative(reachCandidate),
      "--output", repositoryRelative(reachDeployment)
    ]));

  writeStatus("pinning_contact_to_reach", {
    reach_deployment: repositoryRelative(reachDeployment)
  });
  await run(pnpm, [
    "exec", "tsx", "src/training/workyard-contact-pin-cli.ts",
    "--reach-root", repositoryRelative(reachDeployment)
  ]);
  const preflightReport = contactPreflightPath();

  await stage("contact_teacher_preflight", () => contactTeacherReady(preflightReport),
    [preflightReport], () => run(process.execPath, [
      "training/run-workyard-colab-contact.mjs",
      "--mode", "teacher",
      "--reach-root", repositoryRelative(reachDeployment),
      "--preflight-report", repositoryRelative(preflightReport),
      "--output", repositoryRelative(preflightReport),
      "--session", `${sessionPrefix}-contact-teacher`,
      "--gpu", "L4",
      "--timeout-seconds", "1800"
    ]));

  await stage("contact_pilot", () => contactTrainingReady(contactPilot, "pilot"),
    [contactPilot], () => run(process.execPath, [
      "training/run-workyard-colab-contact.mjs",
      "--mode", "pilot",
      "--reach-root", repositoryRelative(reachDeployment),
      "--preflight-report", repositoryRelative(preflightReport),
      "--output", repositoryRelative(contactPilot),
      "--session", `${sessionPrefix}-contact-pilot`,
      "--gpu", "L4",
      "--artifact-stream", "on",
      "--timeout-seconds", "3600"
    ]));

  const formalArgs = [
    "training/run-workyard-colab-contact.mjs",
    "--mode", "train",
    "--reach-root", repositoryRelative(reachDeployment),
    "--preflight-report", repositoryRelative(preflightReport),
    "--output", repositoryRelative(contactTraining),
    "--session", `${sessionPrefix}-contact-formal`,
    "--gpu", "L4",
    "--artifact-stream", "on",
    "--drive-checkpoints", "mount",
    "--drive-directory", driveDirectory,
    "--timeout-seconds", "21600"
  ];
  if (driveLocalRoot) {
    formalArgs.push("--drive-local-root", driveLocalRoot);
  }
  await stage("contact_formal_training",
    () => contactTrainingReady(contactTraining, "train"),
    [contactTraining], () => run(process.execPath, formalArgs));

  await stage("contact_export", contactDeploymentReady, [contactDeployment],
    () => run(process.execPath, [
      "training/run-workyard-contact-export-colab.mjs",
      "--source-root", repositoryRelative(contactTraining),
      "--output", repositoryRelative(contactDeployment),
      "--session", `${sessionPrefix}-contact-export`
    ]));

  await stage("policy_install", installedPoliciesReady, [], () => run(
    process.execPath,
    [
      "training/install-workyard-policy-assets.mjs",
      "--reach-root", repositoryRelative(reachDeployment),
      "--contact-root", repositoryRelative(contactDeployment)
    ]
  ));

  writeStatus("completed", {
    reach_deployment: repositoryRelative(reachDeployment),
    contact_deployment: repositoryRelative(contactDeployment),
    installed_controller_root: "assets/humanoid/controllers"
  });
}

async function stage(name, ready, ownedOutputs, execute) {
  if (ready()) {
    writeStatus(`${name}_already_complete`);
    return;
  }
  const existing = ownedOutputs.filter(existsSync);
  if (existing.length > 0) {
    throw new Error(
      `${name} has incomplete or incompatible existing output: ${existing.join(", ")}`
    );
  }
  writeStatus(name);
  await execute();
  if (!ready()) throw new Error(`${name} returned without qualified output`);
  writeStatus(`${name}_complete`);
}

function reachTrainingReady() {
  const checkpoint = resolve(reachSource, "workyard_reach_selected.pt");
  const reportPath = resolve(reachSource, "training-report.json");
  const report = readJson(reportPath);
  const selected = report?.training?.checkpoint_selection?.selected_checkpoint;
  return existsSync(checkpoint)
    && report?.protocol === "hear-workyard-residual-run-v4"
    && report?.mode === "train"
    && report?.ready === true
    && report?.acceptance?.phase_one_accepted === true
    && report?.acceptance?.deployment_distribution_covered === true
    && report?.acceptance?.deployment_accepted === false
    && selected?.file === "workyard_reach_selected.pt"
    && selected?.bytes === readFileSync(checkpoint).byteLength
    && selected?.sha256 === sha256(checkpoint);
}

function reachCandidateReady() {
  const report = readJson(resolve(reachCandidate, "reach-policy-report.json"));
  return report?.protocol === "hear-whole-body-reach-policy-candidate-v3"
    && artifactMatches(reachCandidate, report?.policy)
    && artifactMatches(reachCandidate, report?.onnx);
}

function reachDeploymentReady() {
  const report = readJson(resolve(reachDeployment, "reach-policy-report.json"));
  return report?.protocol === "hear-whole-body-reach-policy-deployment-v3"
    && report?.deployment?.accepted === true
    && artifactMatches(reachDeployment, report?.policy)
    && artifactMatches(reachDeployment, report?.onnx);
}

function contactPreflightPath() {
  const contract = readJson(contractPath);
  const value = contract?.qualified_inputs?.analytic_teacher_preflight?.report;
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error("Pinned Contact contract has no teacher preflight report path");
  }
  return insideWorkspace(value);
}

function contactTeacherReady(path) {
  const report = readJson(path);
  return report?.protocol === "hear-workyard-contact-analytic-teacher-preflight-v1"
    && report?.passed === true;
}

function contactTrainingReady(root, mode) {
  const report = readJson(resolve(root, "training-report.json"));
  return report?.protocol === "hear-workyard-contact-run-v2"
    && report?.mode === mode
    && report?.ready === true
    && report?.acceptance?.final_gate?.passed === true
    && existsSync(resolve(root, "workyard_contact_selected.pt"));
}

function contactDeploymentReady() {
  const report = readJson(resolve(contactDeployment, "contact-policy-report.json"));
  return report?.protocol === "hear-frozen-contact-policy-export-v2"
    && artifactMatches(contactDeployment, report?.policy?.onnx);
}

function installedPoliciesReady() {
  const reachReport = readJson(resolve(
    "assets/humanoid/controllers/workyard-reach/reach-policy-report.json"
  ));
  const contactReport = readJson(resolve(
    "assets/humanoid/controllers/workyard-contact/contact-policy-report.json"
  ));
  const sourceReach = readJson(resolve(reachDeployment, "reach-policy-report.json"));
  const sourceContact = readJson(resolve(
    contactDeployment, "contact-policy-report.json"
  ));
  return reachReport?.onnx?.sha256 === sourceReach?.onnx?.sha256
    && contactReport?.policy?.onnx?.sha256 === sourceContact?.policy?.onnx?.sha256
    && artifactMatches(
      resolve("assets/humanoid/controllers/workyard-reach"),
      sourceReach?.onnx
    )
    && artifactMatches(
      resolve("assets/humanoid/controllers/workyard-contact"),
      sourceContact?.policy?.onnx
    );
}

function artifactMatches(root, artifact) {
  if (!artifact || typeof artifact.file !== "string"
    || typeof artifact.bytes !== "number" || typeof artifact.sha256 !== "string") {
    return false;
  }
  const path = resolve(root, artifact.file);
  return existsSync(path)
    && readFileSync(path).byteLength === artifact.bytes
    && sha256(path) === artifact.sha256;
}

function validateConfiguration() {
  if (!/^[A-Za-z0-9._-]+$/u.test(sessionPrefix)) {
    throw new Error(`Unsafe autonomy pipeline session prefix: ${sessionPrefix}`);
  }
  if (!Number.isSafeInteger(reachTrainingTimeoutSeconds)
    || reachTrainingTimeoutSeconds < 60) {
    throw new Error("Reach training timeout must be at least 60 seconds");
  }
  for (const [label, directory] of [
    ["Reach", reachDriveDirectory],
    ["Contact", driveDirectory]
  ]) {
    const driveSegments = directory.split("/");
    if (driveSegments.some((segment) => !segment || segment === "." || segment === "..")) {
      throw new Error(`Unsafe ${label} Drive directory: ${directory}`);
    }
  }
  if (driveLocalRoot && !existsSync(driveLocalRoot)) {
    throw new Error(`Google Drive local root is unavailable: ${driveLocalRoot}`);
  }
  const outputs = [
    reachSource,
    reachCandidate,
    reachDeployment,
    contactPilot,
    contactTraining,
    contactDeployment,
    statusPath,
    lockPath
  ];
  for (const path of outputs) insideWorkspace(path);
}

function acquireLock() {
  mkdirSync(dirname(lockPath), { recursive: true });
  if (existsSync(lockPath)) {
    const previous = readJson(lockPath);
    const pid = previous?.pid;
    if (typeof pid === "number" && processAlive(pid)) {
      throw new Error(`Another autonomy pipeline is active with PID ${pid}`);
    }
    rmSync(lockPath, { force: true });
  }
  const descriptor = openSync(lockPath, "wx");
  try {
    writeFileSync(descriptor, JSON.stringify({
      protocol: "hear-workyard-autonomy-pipeline-lock-v1",
      pid: process.pid,
      started_at: new Date().toISOString()
    }));
  } finally {
    closeSync(descriptor);
  }
  lockOwned = true;
}

function releaseLock() {
  if (!lockOwned) return;
  rmSync(lockPath, { force: true });
  lockOwned = false;
}

function processAlive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function run(command, args) {
  if (interrupted) return Promise.reject(new Error("Autonomy pipeline interrupted"));
  return new Promise((resolvePromise, rejectPromise) => {
    const child = spawn(command, args, {
      cwd: workspace,
      stdio: "inherit",
      windowsHide: true
    });
    activeChild = child;
    child.once("error", rejectPromise);
    child.once("exit", (code, signal) => {
      activeChild = undefined;
      if (code === 0) resolvePromise();
      else rejectPromise(new Error(
        `${command} ${args.join(" ")} exited with ${code ?? signal ?? "unknown"}`
      ));
    });
  });
}

function writeStatus(stageName, detail = {}) {
  mkdirSync(dirname(statusPath), { recursive: true });
  writeFileSync(statusPath, `${JSON.stringify({
    protocol: "hear-workyard-autonomy-pipeline-status-v1",
    pipeline_id: sessionPrefix,
    pid: process.pid,
    stage: stageName,
    updated_at: new Date().toISOString(),
    ...detail
  }, null, 2)}\n`, "utf8");
  console.log(`[${sessionPrefix}] ${stageName}`);
}

function readJson(path) {
  if (!existsSync(path)) return undefined;
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch {
    return undefined;
  }
}

function sha256(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function insideWorkspace(path) {
  const absolute = resolve(path);
  const child = relative(workspace, absolute);
  if (!child || child === ".." || child.startsWith("../") || child.startsWith("..\\")) {
    throw new Error(`Autonomy pipeline path leaves the workspace: ${path}`);
  }
  return absolute;
}

function repositoryRelative(path) {
  return relative(workspace, insideWorkspace(path)).replaceAll("\\", "/");
}

function dateStamp() {
  return new Date().toISOString().replace(/[-:]/gu, "").slice(0, 13);
}

function parseOptions(args) {
  if (args[0] === "--") args = args.slice(1);
  const parsed = {};
  const names = new Map([
    ["--reach-source", ["reachSource", String]],
    ["--reach-candidate", ["reachCandidate", String]],
    ["--reach-deployment", ["reachDeployment", String]],
    ["--contact-pilot", ["contactPilot", String]],
    ["--contact-training", ["contactTraining", String]],
    ["--contact-deployment", ["contactDeployment", String]],
    ["--reach-drive-directory", ["reachDriveDirectory", String]],
    ["--drive-directory", ["driveDirectory", String]],
    ["--drive-local-root", ["driveLocalRoot", String]],
    ["--session-prefix", ["sessionPrefix", String]],
    ["--status", ["status", String]],
    ["--reach-training-timeout-seconds", ["reachTrainingTimeoutSeconds", positiveInteger]],
    // Backward-compatible alias for the former passive wait timeout.
    ["--wait-timeout-seconds", ["waitTimeoutSeconds", positiveInteger]]
  ]);
  for (let index = 0; index < args.length; index += 2) {
    const [target, parse] = names.get(args[index]) ?? [];
    const value = args[index + 1];
    if (!target || value === undefined) {
      throw new Error(`Unknown autonomy pipeline option: ${args[index] ?? ""}`);
    }
    parsed[target] = parse(value);
  }
  return parsed;
}

function positiveInteger(value) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1) {
    throw new Error(`Expected positive integer, received: ${value}`);
  }
  return parsed;
}
