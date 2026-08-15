import { createHash, randomUUID } from "node:crypto";
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  rmdirSync,
  statSync,
  writeFileSync
} from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { spawn, spawnSync } from "node:child_process";
import { create, extract } from "tar";

const REMOTE_BUNDLE = "/content/hear-workyard-residual-bundle.tar.gz";
const REMOTE_CONFIG = "/content/hear-workyard-residual-config.json";
const REMOTE_ARCHIVE = "/content/hear-workyard-residual-artifacts.tar.gz";
const REMOTE_REPORT = "/content/hear-workyard-residual-report.json";
const FIXED_BUNDLE_PATHS = [
  "training/hear_retention_ppo.py",
  "training/workyard_mjlab_env.py",
  "training/workyard_residual_mjlab_env.py",
  "training/workyard-task-v4.json",
  "assets/humanoid/g1/g1_with_hands.xml",
  "assets/humanoid/g1/meshes"
];
const TRAINING_ARTIFACTS = new Set([
  "hear-workyard-residual/agent.yaml",
  "hear-workyard-residual/env.yaml",
  "hear-workyard-residual/dagger-curves.json",
  "hear-workyard-residual/training-curves.json",
  "hear-workyard-residual/workyard_reach_dagger_warm_start.pt",
  "hear-workyard-residual/workyard_reach_dagger_ppo.pt",
  "hear-workyard-residual/workyard_reach_selected.pt"
]);

const options = parseOptions(process.argv.slice(2));
const mode = options.mode ?? "smoke";
const backend = options.backend ?? "colab";
if (backend === "local" && mode === "train") {
  throw new Error(
    "The local RTX 3050 Ti profile is bounded to smoke/teacher runs; use Colab for training"
  );
}
const executionProfile = options.profile ?? (
  backend === "local" ? "local-rtx3050ti-smoke-v1" : "colab-pro-l4-formal-v1"
);
const retentionMode = options.ppoRetentionMode ?? "critic_warmup_rollout_teacher";
const retentionCoefficient = options.rolloutTeacherCoefficient ?? 1;
const teacherMaximumActionStd = options.teacherMaximumActionStd ?? 0.15;
const teacherDispersionCoefficient = options.teacherDispersionCoefficient ?? 1;
const workspace = process.cwd();
const session = options.session
  ?? `hear-workyard-residual-${backend}-${mode}-${randomUUID().slice(0, 8)}`;
const distro = options.distro ?? "HEAR-Linux";
const colabPath = options.colabPath ?? "/home/hear/.local/bin/colab";
const localPython = options.localPython ?? "/home/hear/.venvs/hear-mjlab/bin/python";
const teacherRoot = resolve(
  options.teacherRoot ?? "artifacts/training/g1-residual-teacher"
);
const teacherJit = resolve(teacherRoot, "g1_velocity_teacher.jit.pt");
const teacherReport = resolve(teacherRoot, "training-report.json");
const output = resolve(options.output ?? (
  mode === "train"
    ? `artifacts/training/workyard-residual/${session}`
    : `artifacts/training/workyard-residual-${mode}/${session}.json`
));
const localReport = mode === "train" ? `${output}-remote-report.json` : output;
const localArchive = mode === "train" ? `${output}.tar.gz` : null;
const temporaryDirectory = resolve(`.tmp/workyard-${backend}`);
const bundle = resolve(temporaryDirectory, `${session}.tar.gz`);
const config = resolve(temporaryDirectory, `${session}.json`);
const driveBackup = resolve(temporaryDirectory, `${session}-drive-backup.py`);
const bootstrap = resolve("training/colab_workyard_residual.py");
const localLauncher = resolve("training/run-workyard-local-residual.sh");
const localRuntimeDirectory = resolve(temporaryDirectory, `${session}-runtime`);
const localSourceRoot = resolve(localRuntimeDirectory, "source");
const localExecutionReport = resolve(localRuntimeDirectory, "report.json");
const localExecutionArchive = resolve(localRuntimeDirectory, "artifacts.tar.gz");
const localPidFile = resolve(localRuntimeDirectory, "runner.pid");
const driveDirectory = options.driveDirectory
  ?? `HEAR/workyard-residual/${session}`;
const driveLocalRoot = options.driveLocalRoot
  ? resolve(options.driveLocalRoot)
  : null;
const persistCheckpointsToDrive = backend === "colab" && mode === "train";
const remoteDriveOutputRoot = `/content/drive/MyDrive/${driveDirectory}`;

let activeSession = null;
let activeLocalPidFile = null;
let stopping = false;

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    emergencyStop(distro, colabPath, activeSession);
    emergencyStopLocal(distro, activeLocalPidFile);
    process.exit(signal === "SIGINT" ? 130 : 143);
  });
}

await main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});

async function main() {
  const teacherPaths = assertInputs();
  for (const path of [output, localReport, localArchive].filter(Boolean)) {
    if (existsSync(path)) throw new Error(`Residual output already exists: ${path}`);
  }
  mkdirSync(dirname(output), { recursive: true });
  mkdirSync(temporaryDirectory, { recursive: true });
  assertDriveDirectory();
  writeFileSync(config, JSON.stringify({
    mode,
    iterations: options.iterations ?? (mode === "train" ? 1000 : 1),
    dagger_steps: options.daggerSteps ?? (mode === "train" ? 400 : 1),
    dagger_learning_rate: options.daggerLearningRate ?? 3e-4,
    dagger_beta_initial: options.daggerBetaInitial ?? 1,
    dagger_beta_final: options.daggerBetaFinal ?? 0.1,
    ppo_retention_mode: retentionMode,
    critic_warmup_iterations: options.criticWarmupIterations ?? 5,
    ppo_actor_learning_rate: options.ppoActorLearningRate ?? 1e-5,
    ppo_critic_learning_rate: options.ppoCriticLearningRate ?? 3e-4,
    rollout_teacher_coefficient: retentionCoefficient,
    teacher_maximum_action_std: teacherMaximumActionStd,
    teacher_dispersion_coefficient: teacherDispersionCoefficient,
    num_envs: options.numEnvs ?? (
      backend === "local" ? (mode === "teacher" ? 16 : 8) : mode === "train" ? 2048 : 16
    ),
    rollout_steps: options.rolloutSteps ?? (
      backend === "local" && mode === "teacher" ? 200 : 64
    ),
    eval_envs: options.evalEnvs ?? (
      backend === "local" ? 16 : mode === "train" ? 500 : 128
    ),
    eval_steps: options.evalSteps ?? 600,
    seed: options.seed ?? 42,
    execution_profile: executionProfile,
    archive: backend === "local" ? toWslPath(localExecutionArchive) : REMOTE_ARCHIVE,
    ...(persistCheckpointsToDrive
      ? { output_root: remoteDriveOutputRoot }
      : {})
  }), "utf8");
  await createBundle(teacherPaths);
  if (backend === "local") {
    await runLocal();
    return;
  }

  let failure;
  try {
    activeSession = session;
    requireSuccess(await colab([
      "new", "--session", session, "--gpu", options.gpu ?? "L4"
    ]), "create Colab session");
    requireSuccess(await colab([
      "upload", "--session", session, toWslPath(bundle), REMOTE_BUNDLE
    ]), "upload residual Workyard bundle");
    requireSuccess(await colab([
      "upload", "--session", session, toWslPath(config), REMOTE_CONFIG
    ]), "upload residual Workyard configuration");
    if (persistCheckpointsToDrive) {
      requireSuccess(await colab([
        "drivemount", "--session", session, "/content/drive"
      ], false, 300_000), "mount Google Drive for residual checkpoints");
      writeDriveBackup();
    }
    const remoteTimeoutSeconds = options.timeoutSeconds
      ?? (mode === "train" ? 14_400 : 1_800);
    const execution = await colab([
      "exec",
      "--session",
      session,
      "--file",
      toWslPath(bootstrap),
      "--timeout",
      String(remoteTimeoutSeconds)
    ], false, (remoteTimeoutSeconds + 300) * 1_000);
    requireSuccess(await colab([
      "download", "--session", session, REMOTE_REPORT, toWslPath(localReport)
    ]), "download residual Workyard report");
    if (mode === "train") {
      requireSuccess(await colab([
        "download",
        "--session",
        session,
        REMOTE_ARCHIVE,
        toWslPath(localArchive)
      ]), "download residual Workyard artifacts");
      await extractArtifacts(localArchive, output);
      copyFileSync(localReport, resolve(output, "training-report.json"));
      if (persistCheckpointsToDrive) {
        requireSuccess(await colab([
          "exec", "--session", session, "--file", toWslPath(driveBackup),
          "--timeout", "1200"
        ], false, 1_260_000), "persist residual training artifacts to Google Drive");
      }
    }
    const report = validateReport(localReport);
    requireSuccess(execution, `execute residual Workyard ${mode}`);
    if (mode === "train") {
      console.log(`Residual Workyard training: ${output}`);
      console.log(`Residual Workyard Drive backup: ${
        `MyDrive/${driveDirectory}`
      }${driveLocalRoot
        ? ` (desktop mirror: ${resolve(driveLocalRoot, ...driveDirectory.split("/"))})`
        : ""
      }`);
    } else {
      console.log(`Residual Workyard ${mode}: ${output}`);
    }
    if (report.acceptance.deployment_accepted !== false) {
      throw new Error("A smoke/warm-start run cannot claim deployment acceptance");
    }
  } catch (error) {
    failure = error;
  } finally {
    const stop = activeSession
      ? await colab(["stop", "--session", activeSession], true, 120_000)
      : 0;
    activeSession = null;
    rmSync(bundle, { force: true });
    rmSync(config, { force: true });
    rmSync(driveBackup, { force: true });
    try {
      rmdirSync(temporaryDirectory);
    } catch {
      // Another ignored Colab bundle may still be using the shared directory.
    }
    if (stop !== 0) {
      const cleanupError = new Error(
        `Failed to stop Colab session ${session}; run colab stop manually`
      );
      failure = failure
        ? new AggregateError([failure, cleanupError], "Residual run and cleanup failed")
        : cleanupError;
    }
  }
  if (failure) throw failure;
}

async function runLocal() {
  if (existsSync(localRuntimeDirectory)) {
    throw new Error(`Local residual runtime already exists: ${localRuntimeDirectory}`);
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
      requireSuccess(status, `execute local residual Workyard ${mode}`);
      throw new Error("Local residual Workyard produced no report");
    }
    copyFileSync(localExecutionReport, localReport);
    const report = validateReport(localReport);
    requireSuccess(status, `execute local residual Workyard ${mode}`);
    if (report.acceptance.deployment_accepted !== false) {
      throw new Error("A local smoke/teacher run cannot claim deployment acceptance");
    }
    console.log(`Residual Workyard local ${mode}: ${output}`);
  } catch (error) {
    failure = error;
  } finally {
    if (activeLocalPidFile) {
      emergencyStopLocal(distro, activeLocalPidFile);
      activeLocalPidFile = null;
    }
    assertTemporaryChild(localRuntimeDirectory);
    rmSync(localRuntimeDirectory, { recursive: true, force: true });
    rmSync(bundle, { force: true });
    rmSync(config, { force: true });
    try {
      rmdirSync(temporaryDirectory);
    } catch {
      // Another ignored local run may still be using the shared directory.
    }
  }
  if (failure) throw failure;
}

function assertInputs() {
  const runnerInputs = [
    ...FIXED_BUNDLE_PATHS,
    relative(workspace, bootstrap),
    ...(backend === "local" ? [relative(workspace, localLauncher)] : [])
  ];
  for (const path of runnerInputs) {
    if (!existsSync(resolve(path))) {
      throw new Error(`Residual Workyard input is missing: ${path}`);
    }
  }
  if (!/^[A-Za-z0-9._-]+$/.test(session)) {
    throw new Error(`Unsafe residual run identity: ${session}`);
  }
  const teacherRelative = relative(workspace, teacherRoot).replaceAll("\\", "/");
  if (!teacherRelative || teacherRelative === "."
    || teacherRelative === ".." || teacherRelative.startsWith("../")) {
    throw new Error("Teacher artifacts must remain inside the HEAR workspace");
  }
  if (!existsSync(teacherJit) || !existsSync(teacherReport)) {
    throw new Error(
      `Validated teacher JIT/report are missing under ${teacherRoot}`
    );
  }
  const report = JSON.parse(readFileSync(teacherReport, "utf8"));
  const identity = report.teacher_jit;
  if (!identity
    || identity.file !== "g1_velocity_teacher.jit.pt"
    || identity.bytes !== readFileSync(teacherJit).byteLength
    || identity.sha256 !== sha256(teacherJit)
    || identity.input_size !== 99
    || identity.output_size !== 29
    || identity.batch_dynamic !== true
    || identity.runtime !== "torchscript_cuda") {
    throw new Error("Teacher JIT does not match its identity report");
  }
  return {
    teacherRelative,
    jit: `${teacherRelative}/g1_velocity_teacher.jit.pt`,
    report: `${teacherRelative}/training-report.json`
  };
}

function assertDriveDirectory() {
  const segments = driveDirectory.split("/");
  if (segments.length === 0 || segments.some((segment) => (
    !segment || segment === "." || segment === ".." || !/^[\w.-]+$/.test(segment)
  ))) {
    throw new Error(`Unsafe residual Drive directory: ${driveDirectory}`);
  }
  if (driveLocalRoot && (
    !existsSync(driveLocalRoot) || !statSync(driveLocalRoot).isDirectory()
  )) {
    throw new Error(`Residual local Drive root is unavailable: ${driveLocalRoot}`);
  }
}

function writeDriveBackup() {
  writeFileSync(driveBackup, [
    "import shutil",
    "from hashlib import sha256",
    "from pathlib import Path",
    "drive_root = Path('/content/drive/MyDrive').resolve()",
    `target = (drive_root / ${JSON.stringify(driveDirectory)}).resolve()`,
    "if drive_root not in target.parents: raise RuntimeError(f'unsafe Drive target: {target}')",
    "target.mkdir(parents=True, exist_ok=True)",
    "sources = [",
    `  Path(${JSON.stringify(REMOTE_ARCHIVE)}),`,
    `  Path(${JSON.stringify(REMOTE_REPORT)}),`,
    "]",
    "temporaries = []",
    "try:",
    "  for source in sources:",
    "    if not source.is_file(): raise RuntimeError(f'missing training artifact: {source}')",
    "    destination = target / source.name",
    "    temporary = target / (source.name + '.partial')",
    "    temporary.unlink(missing_ok=True)",
    "    temporaries.append(temporary)",
    "    shutil.copy2(source, temporary)",
    "    if temporary.stat().st_size != source.stat().st_size: raise RuntimeError('Drive byte count mismatch')",
    "    if sha256(temporary.read_bytes()).hexdigest() != sha256(source.read_bytes()).hexdigest(): raise RuntimeError('Drive sha256 mismatch')",
    "    temporary.replace(destination)",
    "except Exception:",
    "  for temporary in temporaries: temporary.unlink(missing_ok=True)",
    "  raise",
    "print(f'[hear] residual training persisted to {target}')",
    ""
  ].join("\n"), "utf8");
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
  return new Promise((resolveStatus, reject) => {
    let settled = false;
    let timedOut = false;
    const child = spawn(
      "wsl.exe",
      ["-d", distro, "--", ...command],
      {
        cwd: workspace,
        shell: false,
        stdio: "inherit",
        windowsHide: true
      }
    );
    const watchdogMs = (timeoutSeconds + 60) * 1_000;
    const watchdog = setTimeout(() => {
      if (settled) return;
      timedOut = true;
      emergencyStopLocal(distro, pidFile);
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
      if (timedOut) {
        reject(new Error(
          `Local residual run exceeded its ${watchdogMs}ms host watchdog`
        ));
        return;
      }
      resolveStatus(code ?? (signal ? 1 : 0));
    });
  });
}

function assertTemporaryChild(path) {
  const child = relative(temporaryDirectory, path);
  if (!child || child === ".." || child.startsWith(`..${process.platform === "win32" ? "\\" : "/"}`)) {
    throw new Error(`Refusing to clean an unsafe local runtime path: ${path}`);
  }
}

async function createBundle(teacherPaths) {
  const paths = [...FIXED_BUNDLE_PATHS, teacherPaths.jit, teacherPaths.report];
  await create({
    cwd: workspace,
    file: bundle,
    gzip: true,
    portable: true,
    onWriteEntry(entry) {
      if (entry.path === teacherPaths.jit) {
        entry.path = "assets/humanoid/controllers/mjlab-g1-velocity/"
          + "g1_velocity_teacher.jit.pt";
      } else if (entry.path === teacherPaths.report) {
        entry.path = "assets/humanoid/controllers/mjlab-g1-velocity/training-report.json";
      }
    }
  }, paths);
}

function colab(args, tolerateFailure = false, watchdogMs = 600_000) {
  console.log(`[${session}] colab ${args.join(" ")}`);
  return new Promise((resolveStatus, reject) => {
    let settled = false;
    let timedOut = false;
    const child = spawn(
      "wsl.exe",
      ["-d", distro, "--", colabPath, ...args],
      {
        cwd: workspace,
        shell: false,
        stdio: "inherit",
        windowsHide: true
      }
    );
    const watchdog = setTimeout(() => {
      if (settled) return;
      timedOut = true;
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
      if (timedOut) {
        reject(new Error(
          `Colab command exceeded its ${watchdogMs}ms host watchdog: ${args[0]}`
        ));
        return;
      }
      const status = code ?? (signal ? 1 : 0);
      if (!tolerateFailure && signal) {
        reject(new Error(`Colab command terminated by ${signal}`));
        return;
      }
      resolveStatus(status);
    });
  });
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
    // The process may have exited between the watchdog and termination.
  }
}

function requireSuccess(status, operation) {
  if (status !== 0) throw new Error(`Failed to ${operation} (exit ${status})`);
}

function validateReport(path) {
  const report = JSON.parse(readFileSync(path, "utf8"));
  if (report.protocol !== "hear-workyard-residual-run-v4") {
    throw new Error("Downloaded file is not a residual Workyard report");
  }
  if (report.ready !== true) {
    const detail = report.error?.message ?? "unknown remote failure";
    throw new Error(`Residual Workyard ${mode} failed: ${detail}`);
  }
  const evaluation = report.evaluation;
  if (report.mode !== mode
    || report.execution?.profile !== executionProfile
    || report.contract?.observation_size !== 246
    || report.contract?.action_size !== 29
    || report.contract?.teacher_state_directly_exposed !== false
    || report.contract?.cpu_round_trip_per_control_step !== false
    || evaluation?.teacher?.gradient_parameter_count !== 0
    || !String(evaluation?.teacher?.device ?? "").startsWith("cuda")
    || evaluation?.composition?.maximum_balance_composition_error > 1e-6
    || evaluation?.composition?.maximum_upper_body_command_error > 1e-6
    || evaluation?.composition?.maximum_fixed_open_hand_target_error_rad > 1e-6
    || evaluation?.reach_teacher?.label_coverage !== 1
    || evaluation?.reach_teacher?.actor_observation_exposure !== false
    || evaluation?.reach_teacher?.execution_authority !== "none"
    || evaluation?.dynamic_com?.protocol !== "hear-support-relative-dynamic-com-v1"
    || evaluation?.attribution?.teacher_frame_ratio !== 1
    || evaluation?.finite !== true
    || evaluation?.environment_closed !== true
    || report.acceptance?.structural_invariants_passed !== true) {
    throw new Error("Residual Workyard report violates the phase-one contract");
  }
  if (mode === "train" && (
    report.training?.ppo_retention?.protocol !== "hear-ppo-retention-v2"
    || report.training?.ppo_retention?.mode !== retentionMode
    || report.training?.ppo_retention?.actor_distribution?.class_name
      !== "BetaDistribution"
    || report.training?.ppo_retention?.actor_distribution?.structurally_bounded
      !== true
    || report.training?.ppo_retention?.rollout_teacher_label_coverage !== 1
    || report.training?.checkpoint_selection?.selected_checkpoint?.file
      !== "workyard_reach_selected.pt"
    || report.evaluation?.wrist_position_error_m?.initial_maximum < 0.35
    || report.acceptance?.deployment_distribution_covered !== true
    || report.acceptance?.phase_one_accepted !== true
    || (retentionMode === "critic_warmup_rollout_teacher" && (
      report.training?.ppo_retention?.rollout_teacher_loss_coefficient
        !== retentionCoefficient
      || report.training?.ppo_retention?.teacher_maximum_action_std
        !== teacherMaximumActionStd
      || report.training?.ppo_retention?.teacher_dispersion_coefficient
        !== teacherDispersionCoefficient
      || !Number.isFinite(
        report.training?.ppo_retention?.final_mean_policy_action_std
      )
    ))
  )) {
    throw new Error("Residual training report violates retention-v2 evidence");
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
      throw new Error(`Residual Workyard artifact is missing: ${filename}`);
    }
  }
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

function emergencyStop(distroName, executable, sessionName) {
  if (!sessionName || stopping) return;
  stopping = true;
  spawnSync(
    "wsl.exe",
    ["-d", distroName, "--", executable, "stop", "--session", sessionName],
    {
      cwd: workspace,
      shell: false,
      stdio: "inherit",
      timeout: 60_000,
      windowsHide: true
    }
  );
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
    ["-d", distroName, "--", "sh", "-c", terminate, "hear-local-stop", pidFile],
    {
      cwd: workspace,
      shell: false,
      stdio: "inherit",
      timeout: 30_000,
      windowsHide: true
    }
  );
  stopping = false;
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
    ["--teacher-root", ["teacherRoot", String]],
    ["--output", ["output", String]],
    ["--drive-directory", ["driveDirectory", nonEmptyString]],
    ["--drive-local-root", ["driveLocalRoot", nonEmptyString]],
    ["--iterations", ["iterations", positiveInteger]],
    ["--dagger-steps", ["daggerSteps", positiveInteger]],
    ["--dagger-learning-rate", ["daggerLearningRate", positiveNumber]],
    ["--dagger-beta-initial", ["daggerBetaInitial", unitRate]],
    ["--dagger-beta-final", ["daggerBetaFinal", unitRate]],
    ["--ppo-retention-mode", ["ppoRetentionMode", ppoRetentionMode]],
    ["--critic-warmup-iterations", ["criticWarmupIterations", nonNegativeInteger]],
    ["--ppo-actor-learning-rate", ["ppoActorLearningRate", positiveNumber]],
    ["--ppo-critic-learning-rate", ["ppoCriticLearningRate", positiveNumber]],
    ["--rollout-teacher-coefficient", ["rolloutTeacherCoefficient", positiveNumber]],
    ["--teacher-maximum-action-std", ["teacherMaximumActionStd", unitRateExclusiveZero]],
    ["--teacher-dispersion-coefficient", ["teacherDispersionCoefficient", positiveNumber]],
    ["--num-envs", ["numEnvs", positiveInteger]],
    ["--rollout-steps", ["rolloutSteps", positiveInteger]],
    ["--eval-envs", ["evalEnvs", positiveInteger]],
    ["--eval-steps", ["evalSteps", positiveInteger]],
    ["--seed", ["seed", nonNegativeInteger]],
    ["--timeout-seconds", ["timeoutSeconds", positiveInteger]]
  ]);
  for (let index = 0; index < args.length; index += 2) {
    const name = args[index];
    const entry = names.get(name);
    const value = args[index + 1];
    if (!entry || value === undefined) {
      throw new Error(`Unknown or incomplete residual option: ${name ?? ""}`);
    }
    parsed[entry[0]] = entry[1](value);
  }
  return parsed;
}

function trainingMode(value) {
  if (value !== "smoke" && value !== "teacher" && value !== "train") {
    throw new Error(`Expected smoke, teacher, or train, received: ${value}`);
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

function ppoRetentionMode(value) {
  if (value !== "baseline" && value !== "critic_warmup_rollout_teacher") {
    throw new Error(`Unknown PPO retention mode: ${value}`);
  }
  return value;
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
