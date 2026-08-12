import { createHash, randomUUID } from "node:crypto";
import {
  copyFileSync,
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

const REMOTE_BUNDLE = "/content/hear-workyard-contact-bundle.tar.gz";
const REMOTE_CONFIG = "/content/hear-workyard-contact-config.json";
const REMOTE_ARCHIVE = "/content/hear-workyard-contact-artifacts.tar.gz";
const REMOTE_REPORT = "/content/hear-workyard-contact-report.json";
const BUNDLE_UPLOAD_CHUNK_BYTES = 4 * 1024 * 1024;
const FIXED_BUNDLE_PATHS = [
  "training/hear_retention_ppo.py",
  "training/workyard_mjlab_env.py",
  "training/workyard_residual_mjlab_env.py",
  "training/workyard_contact_mjlab_env.py",
  "training/workyard_observation_parity.py",
  "training/workyard-contact-task-v1.json",
  "assets/humanoid/g1/g1_with_hands.xml",
  "assets/humanoid/g1/meshes"
];
const TRAINING_ARTIFACTS = new Set([
  "hear-workyard-contact/agent.yaml",
  "hear-workyard-contact/env.yaml",
  "hear-workyard-contact/dagger-curves.json",
  "hear-workyard-contact/training-curves.json",
  "hear-workyard-contact/workyard_contact_dagger_warm_start.pt",
  "hear-workyard-contact/workyard_contact_dagger_ppo.pt",
  "hear-workyard-contact/workyard_contact_selected.pt"
]);

const options = parseOptions(process.argv.slice(2));
const mode = options.mode ?? "smoke";
const backend = options.backend ?? "colab";
if (backend === "local" && (mode === "pilot" || mode === "train")) {
  throw new Error(
    "Formal contact training is Colab-only; the RTX 3050 Ti profile is for smoke/preflight"
  );
}
const executionProfile = options.profile ?? (
  backend === "local" ? "local-rtx3050ti-smoke-v1" : "colab-pro-l4-formal-v1"
);
const workspace = process.cwd();
const contractPath = resolve("training/workyard-contact-task-v1.json");
const defaultContract = JSON.parse(readFileSync(contractPath, "utf8"));
const session = options.session
  ?? `hear-workyard-contact-${backend}-${mode}-${randomUUID().slice(0, 8)}`;
const distro = options.distro ?? "HEAR-Linux";
const colabPath = options.colabPath ?? "/home/hear/.local/bin/colab";
const localPython = options.localPython ?? "/home/hear/.venvs/hear-mjlab/bin/python";
const locomotionRoot = resolve(
  options.locomotionRoot ?? "artifacts/training/g1-residual-teacher"
);
const reachRoot = resolve(
  options.reachRoot ?? "artifacts/training/workyard-reach-frozen-v15"
);
const preflightReport = resolve(
  options.preflightReport
    ?? defaultContract.qualified_inputs?.analytic_teacher_preflight?.report
    ?? "artifacts/training/workyard-contact-teacher/local-v59.json"
);
const output = resolve(options.output ?? (
  mode === "train" || mode === "pilot"
    ? `artifacts/training/workyard-contact-${mode}/${session}`
    : `artifacts/training/workyard-contact-${mode}/${session}.json`
));
const producesTrainingArtifacts = mode === "train" || mode === "pilot";
const localReport = producesTrainingArtifacts ? `${output}-remote-report.json` : output;
const localArchive = producesTrainingArtifacts ? `${output}.tar.gz` : null;
const temporaryDirectory = resolve(`.tmp/workyard-contact-${backend}`);
const bundle = resolve(temporaryDirectory, `${session}.tar.gz`);
const config = resolve(temporaryDirectory, `${session}.json`);
const bootstrap = resolve("training/colab_workyard_contact.py");
const localLauncher = resolve("training/run-workyard-local-residual.sh");
const localRuntimeDirectory = resolve(temporaryDirectory, `${session}-runtime`);
const localSourceRoot = resolve(localRuntimeDirectory, "source");
const localExecutionReport = resolve(localRuntimeDirectory, "report.json");
const localExecutionArchive = resolve(localRuntimeDirectory, "artifacts.tar.gz");
const localPidFile = resolve(localRuntimeDirectory, "runner.pid");
const bundleAssemblyScript = resolve(temporaryDirectory, `${session}-assemble-bundle.py`);
const bundleChunkPrefix = resolve(temporaryDirectory, `${session}-bundle-part`);

let activeSession = null;
let activeLocalPidFile = null;
let stopping = false;

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    emergencyStopColab(distro, colabPath, activeSession);
    emergencyStopLocal(distro, activeLocalPidFile);
    process.exit(signal === "SIGINT" ? 130 : 143);
  });
}

await main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});

async function main() {
  const inputs = assertInputs();
  for (const path of [output, localReport, localArchive].filter(Boolean)) {
    if (existsSync(path)) throw new Error(`Contact output already exists: ${path}`);
  }
  mkdirSync(dirname(output), { recursive: true });
  mkdirSync(temporaryDirectory, { recursive: true });
  const configValue = buildConfiguration(inputs.contract);
  writeFileSync(config, JSON.stringify(configValue), "utf8");
  await createBundle(inputs);
  if (backend === "local") {
    await runLocal();
    return;
  }

  let failure;
  try {
    activeSession = session;
    requireSuccess(await colab([
      "new", "--session", session, "--gpu", options.gpu ?? "L4"
    ]), "create Colab contact session");
    await uploadBundleInVerifiedChunks();
    requireSuccess(await retryColab([
      "upload", "--session", session, toWslPath(config), REMOTE_CONFIG
    ], "upload contact configuration", null, 5), "upload contact configuration");
    const remoteTimeoutSeconds = options.timeoutSeconds
      ?? (mode === "train" ? 21_600 : mode === "pilot" ? 3_600 : 1_800);
    const execution = await colab([
      "exec",
      "--session",
      session,
      "--file",
      toWslPath(bootstrap),
      "--timeout",
      String(remoteTimeoutSeconds)
    ], false, (remoteTimeoutSeconds + 300) * 1_000);
    requireSuccess(await retryColab([
      "download", "--session", session, REMOTE_REPORT, toWslPath(localReport)
    ], "download contact report", localReport), "download contact report");
    const report = validateReport(localReport);
    requireSuccess(execution, `execute contact Workyard ${mode}`);
    if (producesTrainingArtifacts) {
      requireSuccess(await retryColab([
        "download", "--session", session, REMOTE_ARCHIVE, toWslPath(localArchive)
      ], "download contact artifacts", localArchive), "download contact artifacts");
      await extractArtifacts(localArchive, output);
      copyFileSync(localReport, resolve(output, "training-report.json"));
      console.log(
        `Contact ${mode}: ${output} (independent gate: ${report.acceptance.final_gate.passed})`
      );
    } else {
      console.log(`Contact Workyard ${mode}: ${output}`);
    }
  } catch (error) {
    failure = error;
  } finally {
    const stop = activeSession
      ? await retryColab(
          ["stop", "--session", activeSession],
          "stop Colab contact session",
          null,
          3,
          120_000
        )
      : 0;
    activeSession = null;
    cleanupTemporaryInputs();
    if (stop !== 0) {
      const cleanupError = new Error(
        `Failed to stop Colab session ${session}; run colab stop manually`
      );
      failure = failure
        ? new AggregateError([failure, cleanupError], "Contact run and cleanup failed")
        : cleanupError;
    }
  }
  if (failure) throw failure;
}

function buildConfiguration(contract) {
  const training = contract.training;
  const selection = training.checkpoint_selection;
  const evaluation = contract.evaluation;
  const pilot = mode === "pilot";
  const value = {
    mode,
    iterations: options.iterations ?? (pilot ? 2 : training.ppo.iterations),
    dagger_steps: options.daggerSteps ?? (pilot ? 200 : training.dagger.steps),
    dagger_learning_rate: options.daggerLearningRate ?? training.dagger.learning_rate,
    dagger_beta_initial: options.daggerBetaInitial ?? training.dagger.teacher_beta_initial,
    dagger_beta_final: options.daggerBetaFinal ?? training.dagger.teacher_beta_final,
    critic_warmup_iterations: options.criticWarmupIterations
      ?? (pilot ? 1 : training.ppo.critic_warmup_iterations),
    ppo_actor_learning_rate: options.ppoActorLearningRate
      ?? training.ppo.actor_learning_rate,
    ppo_critic_learning_rate: options.ppoCriticLearningRate
      ?? training.ppo.critic_learning_rate,
    rollout_teacher_coefficient: options.rolloutTeacherCoefficient
      ?? training.ppo.rollout_teacher_coefficient,
    teacher_maximum_action_std: options.teacherMaximumActionStd
      ?? training.dagger.maximum_action_std,
    teacher_dispersion_coefficient: options.teacherDispersionCoefficient ?? 1,
    num_envs: options.numEnvs ?? (
      backend === "local" ? (mode === "teacher" ? 16 : 8) : mode === "train"
        ? training.ppo.environments : pilot ? 32 : 16
    ),
    rollout_steps: options.rolloutSteps ?? (mode === "teacher" ? 400 : 64),
    comparison_envs: options.comparisonEnvs
      ?? (pilot ? 16 : selection.comparison_episodes),
    comparison_steps: options.comparisonSteps ?? selection.control_steps_per_episode,
    final_eval_envs: options.finalEvalEnvs ?? (pilot ? 16 : evaluation.episodes),
    final_eval_steps: options.finalEvalSteps ?? evaluation.control_steps_per_episode,
    seed: options.seed ?? 42,
    execution_profile: executionProfile,
    archive: backend === "local" ? toWslPath(localExecutionArchive) : REMOTE_ARCHIVE
  };
  if (mode === "train") {
    const formal = {
      iterations: training.ppo.iterations,
      dagger_steps: training.dagger.steps,
      num_envs: training.ppo.environments,
      comparison_envs: selection.comparison_episodes,
      comparison_steps: selection.control_steps_per_episode,
      final_eval_envs: evaluation.episodes,
      final_eval_steps: evaluation.control_steps_per_episode
    };
    const drifted = Object.entries(formal)
      .filter(([name, expected]) => value[name] !== expected)
      .map(([name]) => name);
    if (drifted.length > 0) {
      throw new Error(`Formal contact configuration drifted: ${drifted.join(", ")}`);
    }
  }
  return value;
}

async function runLocal() {
  if (existsSync(localRuntimeDirectory)) {
    throw new Error(`Local contact runtime already exists: ${localRuntimeDirectory}`);
  }
  mkdirSync(localRuntimeDirectory);
  const timeoutSeconds = options.timeoutSeconds ?? (mode === "teacher" ? 1_800 : 1_200);
  let failure;
  try {
    activeLocalPidFile = toWslPath(localPidFile);
    const status = await local(timeoutSeconds).finally(() => {
      activeLocalPidFile = null;
    });
    if (!existsSync(localExecutionReport)) {
      requireSuccess(status, `execute local contact Workyard ${mode}`);
      throw new Error("Local contact run produced no report");
    }
    copyFileSync(localExecutionReport, localReport);
    validateReport(localReport);
    requireSuccess(status, `execute local contact Workyard ${mode}`);
    console.log(`Contact Workyard local ${mode}: ${output}`);
  } catch (error) {
    failure = error;
  } finally {
    if (activeLocalPidFile) {
      emergencyStopLocal(distro, activeLocalPidFile);
      activeLocalPidFile = null;
    }
    assertTemporaryChild(localRuntimeDirectory);
    rmSync(localRuntimeDirectory, { recursive: true, force: true });
    cleanupTemporaryInputs();
  }
  if (failure) throw failure;
}

function assertInputs() {
  const contract = JSON.parse(readFileSync(contractPath, "utf8"));
  if (contract.protocol !== "hear-workyard-contact-training-contract-v1"
    || contract.learner?.observation?.size !== 247
    || contract.learner?.action?.size !== 8
    || contract.composition?.logical_composed_action_size !== 22) {
    throw new Error("Contact training contract is invalid");
  }
  if (!/^[A-Za-z0-9._-]+$/.test(session)) {
    throw new Error(`Unsafe contact run identity: ${session}`);
  }
  const runnerInputs = [
    ...FIXED_BUNDLE_PATHS,
    relative(workspace, bootstrap),
    ...(backend === "local" ? [relative(workspace, localLauncher)] : [])
  ];
  for (const path of runnerInputs) {
    if (!existsSync(resolve(path))) {
      throw new Error(`Contact Workyard input is missing: ${path}`);
    }
  }
  const roots = [
    ["locomotion", locomotionRoot],
    ["reach", reachRoot],
    ["preflight", preflightReport]
  ];
  const relatives = {};
  for (const [name, path] of roots) {
    const value = workspaceRelative(path);
    relatives[name] = value;
  }
  const locomotionJit = resolve(locomotionRoot, "g1_velocity_teacher.jit.pt");
  const locomotionReport = resolve(locomotionRoot, "training-report.json");
  const reachJit = resolve(reachRoot, "workyard_reach_selected.jit.pt");
  const reachReport = resolve(reachRoot, "reach-policy-report.json");
  for (const path of [
    locomotionJit, locomotionReport, reachJit, reachReport, preflightReport
  ]) {
    if (!existsSync(path)) throw new Error(`Qualified contact artifact is missing: ${path}`);
  }
  const locomotionIdentity = JSON.parse(readFileSync(locomotionReport, "utf8")).teacher_jit;
  if (!locomotionIdentity
    || locomotionIdentity.sha256 !== sha256(locomotionJit)
    || locomotionIdentity.bytes !== readFileSync(locomotionJit).byteLength
    || locomotionIdentity.input_size !== 99
    || locomotionIdentity.output_size !== 29
    || locomotionIdentity.batch_dynamic !== true) {
    throw new Error("Frozen locomotion JIT does not match its report");
  }
  const reachReportValue = JSON.parse(readFileSync(reachReport, "utf8"));
  if (reachReportValue.policy?.sha256 !== sha256(reachJit)
    || reachReportValue.policy?.sha256 !== contract.qualified_inputs.reach_policy.jit_sha256
    || reachReportValue.policy?.bytes !== readFileSync(reachJit).byteLength
    || reachReportValue.policy?.input_size !== 231
    || reachReportValue.policy?.output_size !== 14
    || reachReportValue.policy?.gradient_parameter_count !== 0) {
    throw new Error("Frozen v15 reach JIT does not match its report");
  }
  const preflight = JSON.parse(readFileSync(preflightReport, "utf8"));
  if (preflight.gate?.protocol
      !== "hear-workyard-contact-analytic-teacher-preflight-v1"
    || preflight.gate?.passed !== true
    || !Object.values(preflight.gate?.checks ?? {}).every(Boolean)) {
    throw new Error("Analytic contact preflight is not qualified");
  }
  return {
    contract,
    bundlePaths: [
      ...FIXED_BUNDLE_PATHS,
      relatives.locomotion,
      relatives.reach,
      relatives.preflight
    ]
  };
}

function workspaceRelative(path) {
  const value = relative(workspace, path).replaceAll("\\", "/");
  if (!value || value === "." || value === ".." || value.startsWith("../")) {
    throw new Error(`Contact input must remain inside the workspace: ${path}`);
  }
  return value;
}

async function createBundle(inputs) {
  await create({
    cwd: workspace,
    file: bundle,
    gzip: true,
    portable: true
  }, [...new Set(inputs.bundlePaths)]);
}

async function uploadBundleInVerifiedChunks() {
  const value = readFileSync(bundle);
  const expectedSha256 = createHash("sha256").update(value).digest("hex");
  const chunks = [];
  for (let offset = 0, index = 0; offset < value.byteLength; index += 1) {
    const end = Math.min(offset + BUNDLE_UPLOAD_CHUNK_BYTES, value.byteLength);
    const path = `${bundleChunkPrefix}-${String(index).padStart(3, "0")}`;
    writeFileSync(path, value.subarray(offset, end));
    chunks.push(path);
    offset = end;
  }
  if (chunks.length === 0) throw new Error("Contact bundle is empty");

  const remoteParts = chunks.map((_, index) => (
    `${REMOTE_BUNDLE}.part-${String(index).padStart(3, "0")}`
  ));
  writeFileSync(bundleAssemblyScript, [
    "from hashlib import sha256",
    "from pathlib import Path",
    `target = Path(${JSON.stringify(REMOTE_BUNDLE)})`,
    `parts = [Path(value) for value in ${JSON.stringify(remoteParts)}]`,
    `expected_bytes = ${value.byteLength}`,
    `expected_sha256 = ${JSON.stringify(expectedSha256)}`,
    "valid_target = (",
    "    target.is_file()",
    "    and target.stat().st_size == expected_bytes",
    "    and sha256(target.read_bytes()).hexdigest() == expected_sha256",
    ")",
    "if not valid_target:",
    "    missing = [str(part) for part in parts if not part.is_file()]",
    "    if missing:",
    "        raise RuntimeError(f'missing bundle chunks: {missing}')",
    "    with target.open('wb') as output:",
    "        for part in parts:",
    "            output.write(part.read_bytes())",
    `if target.stat().st_size != ${value.byteLength}:`,
    "    raise RuntimeError(f'bundle byte count mismatch: {target.stat().st_size}')",
    `if sha256(target.read_bytes()).hexdigest() != ${JSON.stringify(expectedSha256)}:`,
    "    raise RuntimeError('bundle sha256 mismatch')",
    "for part in parts:",
    "    part.unlink(missing_ok=True)",
    "print(f'[hear] verified contact bundle: {target.stat().st_size} bytes')",
    ""
  ].join("\n"), "utf8");

  console.log(
    `[${session}] uploading ${value.byteLength} byte contact bundle in ${chunks.length} verified chunks`
  );
  for (let index = 0; index < chunks.length; index += 1) {
    requireSuccess(await retryColab([
      "upload", "--session", session, toWslPath(chunks[index]), remoteParts[index]
    ], `upload contact bundle chunk ${index + 1}/${chunks.length}`, null, 5),
    `upload contact bundle chunk ${index + 1}/${chunks.length}`);
  }
  requireSuccess(await retryColab([
    "exec", "--session", session, "--file", toWslPath(bundleAssemblyScript), "--timeout", "300"
  ], "verify and assemble contact bundle", null, 5, 360_000),
  "verify and assemble contact bundle");
}

function local(timeoutSeconds) {
  const pidFile = toWslPath(localPidFile);
  const command = [
    "env",
    `HEAR_WORKYARD_EXECUTION_ROOT=${toWslPath(localRuntimeDirectory)}`,
    `HEAR_WORKYARD_BUNDLE=${toWslPath(bundle)}`,
    `HEAR_WORKYARD_CONFIG=${toWslPath(config)}`,
    `HEAR_WORKYARD_SOURCE_ROOT=${toWslPath(localSourceRoot)}`,
    `HEAR_WORKYARD_REPORT=${toWslPath(localExecutionReport)}`,
    `HEAR_WORKYARD_PID_FILE=${pidFile}`,
    `HEAR_WORKYARD_TIMEOUT_SECONDS=${timeoutSeconds}`,
    "HEAR_WORKYARD_SKIP_DEPENDENCY_INSTALL=1",
    "MUJOCO_GL=egl",
    "WANDB_MODE=disabled",
    "PYTHONUNBUFFERED=1",
    "bash",
    toWslPath(localLauncher),
    localPython,
    toWslPath(bootstrap)
  ];
  console.log(
    `[${session}] local ${executionProfile} (${timeoutSeconds}s timeout, ${localPython})`
  );
  return spawnPromise(
    "wsl.exe",
    ["-d", distro, "--", ...command],
    (timeoutSeconds + 60) * 1_000,
    () => emergencyStopLocal(distro, pidFile),
    `Local contact run exceeded its host watchdog`
  );
}

function colab(args, tolerateFailure = false, watchdogMs = 600_000) {
  console.log(`[${session}] colab ${args.join(" ")}`);
  return spawnPromise(
    "wsl.exe",
    ["-d", distro, "--", colabPath, ...args],
    watchdogMs,
    () => {},
    `Colab command exceeded its host watchdog: ${args[0]}`,
    tolerateFailure
  );
}

async function retryColab(
  args,
  operation,
  partialDownload = null,
  attempts = 3,
  watchdogMs = 600_000
) {
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    if (partialDownload) rmSync(partialDownload, { force: true });
    const status = await colab(args, true, watchdogMs);
    if (status === 0) return status;
    if (attempt === attempts) return status;
    const delayMs = 15_000 * attempt;
    console.warn(
      `[${session}] ${operation} failed (attempt ${attempt}/${attempts}); retrying in ${delayMs / 1000}s`
    );
    await new Promise((resolveDelay) => setTimeout(resolveDelay, delayMs));
  }
  return 1;
}

function spawnPromise(command, args, watchdogMs, beforeKill, timeoutMessage, tolerateSignal = false) {
  return new Promise((resolveStatus, reject) => {
    let settled = false;
    let timedOut = false;
    const child = spawn(command, args, {
      cwd: workspace,
      shell: false,
      stdio: "inherit",
      windowsHide: true
    });
    const watchdog = setTimeout(() => {
      if (settled) return;
      timedOut = true;
      beforeKill();
      terminateProcessTree(child.pid);
    }, watchdogMs);
    child.once("error", (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(watchdog);
      reject(error);
    });
    child.once("exit", (code, signal) => {
      if (settled) return;
      settled = true;
      clearTimeout(watchdog);
      if (timedOut) return reject(new Error(timeoutMessage));
      if (signal && !tolerateSignal) {
        return reject(new Error(`${command} terminated by ${signal}`));
      }
      resolveStatus(code ?? (signal ? 1 : 0));
    });
  });
}

function validateReport(path) {
  const report = JSON.parse(readFileSync(path, "utf8"));
  if (report.protocol !== "hear-workyard-contact-run-v1") {
    throw new Error("Downloaded file is not a contact Workyard report");
  }
  if (report.ready !== true) {
    throw new Error(`Contact Workyard failed: ${report.error?.message ?? "unknown failure"}`);
  }
  const evaluation = report.evaluation;
  if (report.mode !== mode
    || report.execution?.profile !== executionProfile
    || report.contract?.observation_size !== 247
    || report.contract?.action_size !== 8
    || report.contract?.logical_composed_action_size !== 22
    || report.contract?.terminal_stage !== "grasp"
    || JSON.stringify(report.contract?.excluded_stages) !== JSON.stringify(["lift", "carry", "place"])
    || report.contract?.teacher_state_directly_exposed !== false
    || report.contract?.cpu_round_trip_per_control_step !== false
    || report.contract?.hand_max_closing_joint_lead_rad !== 0.25
    || report.contract?.opposing_support_coordination !== 0.4
    || evaluation?.frozen_locomotion?.gradient_parameter_count !== 0
    || evaluation?.frozen_reach?.gradient_parameter_count !== 0
    || evaluation?.maximum_unauthorized_hand_action !== 0
    || evaluation?.maximum_inactive_hand_coordination !== 0
    || evaluation?.contact_base_assist_environment_count !== 0
    || evaluation?.numerical_instability_count !== 0
    || evaluation?.finite !== true
    || evaluation?.environment_closed !== true
    || report.acceptance?.structural_invariants_passed !== true
    || report.acceptance?.deployment_accepted !== false
    || report.acceptance?.lift_carry_place_authorized !== false) {
    throw new Error("Contact report violates the 8D authority contract");
  }
  if (mode === "teacher" && report.fresh_analytic_preflight?.passed !== true) {
    throw new Error("Fresh analytic contact preflight failed");
  }
  if ((mode === "pilot" || mode === "train") && (
    report.fresh_analytic_preflight?.passed !== true
    || report.training?.numerical_recovery?.within_budget !== true
    || report.training?.dagger?.protocol
      !== "hear-contact-online-dagger-warm-start-v1"
    || report.training?.dagger?.label_coverage !== 1
    || report.training?.dagger?.authorized_state_count <= 0
    || report.training?.ppo_retention?.protocol
      !== "hear-contact-ppo-retention-v1"
    || report.training?.ppo_retention?.actor_distribution?.structurally_bounded
      !== true
    || report.training?.ppo_retention?.rollout_teacher_label_coverage !== 1
    || report.training?.checkpoint_selection?.selected_checkpoint?.file
      !== "workyard_contact_selected.pt"
    || report.acceptance?.final_gate?.protocol
      !== "hear-workyard-contact-independent-500-gate-v1"
    || report.acceptance?.verified_grasp_policy_accepted
      !== report.acceptance?.final_gate?.passed
  )) {
    throw new Error("Contact training report lacks DAgger/retention/gate evidence");
  }
  return report;
}

async function extractArtifacts(archive, destination) {
  mkdirSync(destination);
  await extract({
    file: archive,
    cwd: destination,
    strip: 1,
    filter: (path) => TRAINING_ARTIFACTS.has(path)
  });
  for (const path of TRAINING_ARTIFACTS) {
    const filename = path.slice(path.lastIndexOf("/") + 1);
    if (!existsSync(resolve(destination, filename))) {
      throw new Error(`Contact training artifact is missing: ${filename}`);
    }
  }
}

function cleanupTemporaryInputs() {
  rmSync(bundle, { force: true });
  rmSync(config, { force: true });
  rmSync(bundleAssemblyScript, { force: true });
  for (let index = 0; index < 10_000; index += 1) {
    const path = `${bundleChunkPrefix}-${String(index).padStart(3, "0")}`;
    if (!existsSync(path)) break;
    rmSync(path, { force: true });
  }
  try {
    rmdirSync(temporaryDirectory);
  } catch {
    // A separate ignored runner may still own the shared temporary directory.
  }
}

function assertTemporaryChild(path) {
  const child = relative(temporaryDirectory, path);
  if (!child || child === ".." || child.startsWith(`..${process.platform === "win32" ? "\\" : "/"}`)) {
    throw new Error(`Refusing to clean an unsafe contact runtime path: ${path}`);
  }
}

function terminateProcessTree(pid) {
  if (!Number.isSafeInteger(pid) || pid <= 0) return;
  if (process.platform === "win32") {
    spawnSync("taskkill.exe", ["/PID", String(pid), "/T", "/F"], {
      cwd: workspace,
      shell: false,
      stdio: "ignore",
      timeout: 30_000,
      windowsHide: true
    });
    return;
  }
  try {
    process.kill(pid, "SIGTERM");
  } catch {
    // Process may have exited between watchdog and cleanup.
  }
}

function emergencyStopColab(distroName, executable, sessionName) {
  if (!sessionName || stopping) return;
  stopping = true;
  spawnSync(
    "wsl.exe",
    ["-d", distroName, "--", executable, "stop", "--session", sessionName],
    { cwd: workspace, shell: false, stdio: "inherit", timeout: 60_000, windowsHide: true }
  );
  stopping = false;
}

function emergencyStopLocal(distroName, pidFile) {
  if (!pidFile || stopping) return;
  stopping = true;
  const terminate = [
    'pid=$(cat "$1" 2>/dev/null) || exit 0',
    'case "$pid" in ""|*[!0-9]*) exit 0;; esac',
    'kill -TERM "$pid" 2>/dev/null || exit 0',
    'sleep 2',
    'kill -KILL "$pid" 2>/dev/null || true'
  ].join("; ");
  spawnSync(
    "wsl.exe",
    ["-d", distroName, "--", "sh", "-c", terminate, "hear-contact-stop", pidFile],
    { cwd: workspace, shell: false, stdio: "inherit", timeout: 30_000, windowsHide: true }
  );
  stopping = false;
}

function requireSuccess(status, operation) {
  if (status !== 0) throw new Error(`Failed to ${operation} (exit ${status})`);
}

function sha256(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function toWslPath(path) {
  const absolute = resolve(path).replaceAll("\\", "/");
  const match = /^([A-Za-z]):\/(.*)$/.exec(absolute);
  if (!match) throw new Error(`Cannot map Windows path into WSL: ${path}`);
  return `/mnt/${match[1].toLowerCase()}/${match[2]}`;
}

function parseOptions(args) {
  if (args[0] === "--") args = args.slice(1);
  const parsed = {};
  const names = new Map([
    ["--mode", ["mode", trainingMode]],
    ["--backend", ["backend", executionBackend]],
    ["--profile", ["profile", nonEmptyString]],
    ["--distro", ["distro", String]],
    ["--colab-path", ["colabPath", String]],
    ["--local-python", ["localPython", nonEmptyString]],
    ["--gpu", ["gpu", String]],
    ["--session", ["session", String]],
    ["--locomotion-root", ["locomotionRoot", String]],
    ["--reach-root", ["reachRoot", String]],
    ["--preflight-report", ["preflightReport", String]],
    ["--output", ["output", String]],
    ["--iterations", ["iterations", positiveInteger]],
    ["--dagger-steps", ["daggerSteps", positiveInteger]],
    ["--dagger-learning-rate", ["daggerLearningRate", positiveNumber]],
    ["--dagger-beta-initial", ["daggerBetaInitial", unitRate]],
    ["--dagger-beta-final", ["daggerBetaFinal", unitRate]],
    ["--critic-warmup-iterations", ["criticWarmupIterations", nonNegativeInteger]],
    ["--ppo-actor-learning-rate", ["ppoActorLearningRate", positiveNumber]],
    ["--ppo-critic-learning-rate", ["ppoCriticLearningRate", positiveNumber]],
    ["--rollout-teacher-coefficient", ["rolloutTeacherCoefficient", positiveNumber]],
    ["--teacher-maximum-action-std", ["teacherMaximumActionStd", unitRateExclusiveZero]],
    ["--teacher-dispersion-coefficient", ["teacherDispersionCoefficient", positiveNumber]],
    ["--num-envs", ["numEnvs", positiveInteger]],
    ["--rollout-steps", ["rolloutSteps", positiveInteger]],
    ["--comparison-envs", ["comparisonEnvs", positiveInteger]],
    ["--comparison-steps", ["comparisonSteps", positiveInteger]],
    ["--final-eval-envs", ["finalEvalEnvs", positiveInteger]],
    ["--final-eval-steps", ["finalEvalSteps", positiveInteger]],
    ["--seed", ["seed", nonNegativeInteger]],
    ["--timeout-seconds", ["timeoutSeconds", positiveInteger]]
  ]);
  for (let index = 0; index < args.length; index += 2) {
    const name = args[index];
    const entry = names.get(name);
    const value = args[index + 1];
    if (!entry || value === undefined) {
      throw new Error(`Unknown or incomplete contact option: ${name ?? ""}`);
    }
    parsed[entry[0]] = entry[1](value);
  }
  return parsed;
}

function trainingMode(value) {
  if (value !== "smoke" && value !== "teacher" && value !== "pilot" && value !== "train") {
    throw new Error(`Expected smoke, teacher, pilot, or train, received: ${value}`);
  }
  return value;
}

function executionBackend(value) {
  if (value !== "local" && value !== "colab") {
    throw new Error(`Expected local or colab backend, received: ${value}`);
  }
  return value;
}

function nonEmptyString(value) {
  if (!String(value).trim()) throw new Error("Expected a non-empty value");
  return String(value);
}

function positiveNumber(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`Expected a positive number, received: ${value}`);
  }
  return parsed;
}

function unitRate(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0 || parsed > 1) {
    throw new Error(`Expected a number inside [0, 1], received: ${value}`);
  }
  return parsed;
}

function unitRateExclusiveZero(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0 || parsed > 1) {
    throw new Error(`Expected a number inside (0, 1], received: ${value}`);
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
