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
import { spawnSync } from "node:child_process";
import { create, extract } from "tar";

const REMOTE_BUNDLE = "/content/hear-g1-getup-bundle.tar.gz";
const REMOTE_CONFIG = "/content/hear-g1-getup-config.json";
const REMOTE_ARCHIVE = "/content/hear-g1-getup-artifacts.tar.gz";
const REMOTE_REPORT = "/content/hear-g1-getup-report.json";
const REMOTE_ROOT = "/content/hear-g1-getup-source";
const BUNDLE_PATHS = [
  "training/colab_g1_getup.py",
  "training/g1_getup_mjlab_env.py",
  "training/g1-getup-task-v1.json",
  "training/workyard_mjlab_env.py",
  "assets/humanoid/g1/g1_with_hands.xml",
  "assets/humanoid/g1/meshes",
  "assets/humanoid/controllers/mjlab-g1-velocity/training-report.json"
];
const ARTIFACT_FILES = new Set([
  "hear-g1-getup/agent.yaml",
  "hear-g1-getup/env.yaml",
  "hear-g1-getup/g1_getup.pt",
  "hear-g1-getup/g1_getup.onnx",
  "hear-g1-getup/getup-policy-report.json"
]);

const options = parseOptions(process.argv.slice(2));
const mode = options.mode ?? "train";
const workspace = process.cwd();
const session = options.session
  ?? `hear-g1-getup-${mode}-${randomUUID().slice(0, 8)}`;
const distro = options.distro ?? "HEAR-Linux";
const colabPath = options.colabPath ?? "/home/hear/.local/bin/colab";
const output = resolve(options.output ?? (
  mode !== "smoke"
    ? `artifacts/training/g1-getup/${session}`
    : `artifacts/training/g1-getup-smoke/${session}`
));
const archive = `${output}.tar.gz`;
const report = `${output}-remote-report.json`;
const temporaryDirectory = resolve(".tmp/g1-getup-colab");
const bundle = resolve(temporaryDirectory, `${session}.tar.gz`);
const config = resolve(temporaryDirectory, `${session}.json`);
const driveBackup = resolve(temporaryDirectory, `${session}-drive-backup.py`);
const evaluationInput = resolve(temporaryDirectory, `${session}-input`);
const evaluationInputRelative = relative(workspace, evaluationInput)
  .replaceAll("\\", "/");
const remoteEvaluationInput = `${REMOTE_ROOT}/${evaluationInputRelative}`;
const driveDirectory = options.driveDirectory ?? `HEAR/g1-getup/${session}`;
const remoteDriveOutputRoot = `/content/drive/MyDrive/${driveDirectory}`;
const desktopDriveRoot = options.desktopDriveRoot
  ?? process.env.HEAR_G1_GETUP_DESKTOP_DRIVE_ROOT?.trim();
const remoteTrainingOutputRoot = remoteDriveOutputRoot;
const reuseSession = options.reuseSession ?? false;

let activeSession = null;
for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    emergencyStop();
    process.exit(signal === "SIGINT" ? 130 : 143);
  });
}

await main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});

async function main() {
  assertInputs();
  for (const path of [output, archive, report, bundle, config]) {
    if (existsSync(path)) throw new Error(`G1 get-up output already exists: ${path}`);
  }
  mkdirSync(dirname(output), { recursive: true });
  mkdirSync(temporaryDirectory, { recursive: true });
  if (mode === "evaluate") prepareEvaluationInput();
  writeFileSync(config, JSON.stringify({
    mode,
    iterations: options.iterations ?? (mode === "smoke" ? 2 : 5000),
    num_envs: options.numEnvs ?? (mode === "smoke" ? 16 : 2048),
    eval_envs: options.evalEnvs ?? (mode === "smoke" ? 32 : 500),
    eval_batch_size: options.evalBatchSize ?? (mode === "smoke" ? 32 : 64),
    eval_steps: options.evalSteps ?? (mode === "smoke" ? 20 : 750),
    seed: options.seed ?? 42,
    archive: REMOTE_ARCHIVE,
    ...(mode === "train" ? { output_root: remoteTrainingOutputRoot } : {}),
    ...(mode === "evaluate" ? { output_root: remoteEvaluationInput } : {})
  }), "utf8");
  await create({ gzip: true, file: bundle, cwd: workspace }, [
    ...BUNDLE_PATHS,
    ...(mode === "evaluate" ? [evaluationInputRelative] : [])
  ]);
  writeDriveBackup();

  let failure;
  let executionStatus = 1;
  try {
    activeSession = session;
    if (reuseSession) {
      requireSuccess(colab([
        "status", "--session", session
      ]), "attach to the existing G1 get-up Colab session");
    } else {
      requireSuccess(colab([
        "new", "--session", session, "--gpu", options.gpu ?? "L4"
      ]), "create G1 get-up Colab session");
    }
    requireSuccess(colab([
      "upload", "--session", session, toWslPath(bundle), REMOTE_BUNDLE
    ]), "upload G1 get-up bundle");
    requireSuccess(colab([
      "upload", "--session", session, toWslPath(config), REMOTE_CONFIG
    ]), "upload G1 get-up configuration");
    if (mode === "train" && !reuseSession) {
      requireSuccess(colab([
        "drivemount", "--session", session, "/content/drive"
      ], 900_000), "mount Google Drive for G1 get-up checkpoints");
    }
    const timeoutSeconds = options.timeoutSeconds
      ?? (mode === "smoke" ? 1_800 : 64_800);
    executionStatus = colab([
      "exec", "--session", session,
      "--file", toWslPath(resolve("training/colab_g1_getup.py")),
      "--timeout", String(timeoutSeconds)
    ], (timeoutSeconds + 300) * 1000);

    requireSuccess(colab([
      "download", "--session", session, REMOTE_REPORT, toWslPath(report)
    ]), "download G1 get-up report");
    requireSuccess(colab([
      "download", "--session", session, REMOTE_ARCHIVE, toWslPath(archive)
    ]), "download G1 get-up artifacts");
    await extractArtifacts();
    const parsed = validateReport();
    if (mode === "evaluate") persistDesktopDriveEvaluation(parsed);
    if (executionStatus !== 0
      || (mode !== "smoke" && parsed.evaluation.deployment_accepted !== true)) {
      throw new Error(
        `G1 get-up ${mode} completed but did not pass deployment: ${report}`
      );
    }
    if (mode === "train") {
      const backupStatus = colab([
        "exec", "--session", session, "--file", toWslPath(driveBackup),
        "--timeout", "1200"
      ], 1_260_000);
      requireSuccess(backupStatus, "persist G1 get-up artifacts to Google Drive");
    }
    console.log(`G1 get-up policy: ${output}`);
    console.log(`G1 get-up Drive backup: MyDrive/${driveDirectory}`);
    if (desktopDriveRoot) {
      console.log(`G1 get-up desktop Drive sync: ${resolve(
        desktopDriveRoot, ...driveDirectory.split("/")
      )}`);
    }
  } catch (error) {
    failure = error;
  } finally {
    const stopStatus = activeSession
      ? colab(["stop", "--session", activeSession], 120_000)
      : 0;
    activeSession = null;
    rmSync(bundle, { force: true });
    rmSync(config, { force: true });
    rmSync(driveBackup, { force: true });
    rmSync(evaluationInput, { recursive: true, force: true });
    try { rmdirSync(temporaryDirectory); } catch {}
    if (stopStatus !== 0) {
      const cleanup = new Error(`Failed to stop Colab session ${session}`);
      failure = failure
        ? new AggregateError([failure, cleanup], "G1 get-up run and cleanup failed")
        : cleanup;
    }
  }
  if (failure) throw failure;
}

function prepareEvaluationInput() {
  const source = options.evaluationSource
    ? resolve(options.evaluationSource)
    : desktopDriveRoot
      ? resolve(desktopDriveRoot, ...driveDirectory.split("/"))
      : undefined;
  if (!source) {
    throw new Error(
      "G1 evaluation requires --evaluation-source or --desktop-drive-root"
    );
  }
  const files = [
    "resume-identity.json",
    "artifacts/agent.yaml",
    "artifacts/env.yaml",
    "artifacts/g1_getup.pt",
    "artifacts/g1_getup.onnx"
  ];
  mkdirSync(resolve(evaluationInput, "artifacts"), { recursive: true });
  for (const file of files) {
    const input = resolve(source, file);
    if (!existsSync(input)) {
      throw new Error(`G1 evaluation input is missing: ${input}`);
    }
    copyFileSync(input, resolve(evaluationInput, file));
  }
}

function persistDesktopDriveEvaluation(parsed) {
  if (!desktopDriveRoot) return;
  const destination = resolve(
    desktopDriveRoot, ...driveDirectory.split("/")
  );
  const artifacts = resolve(destination, "artifacts");
  mkdirSync(artifacts, { recursive: true });
  copyFileSync(
    resolve(output, "getup-policy-report.json"),
    resolve(artifacts, "getup-policy-report.json")
  );
  copyFileSync(report, resolve(destination, "hear-g1-getup-report.json"));
  copyFileSync(archive, resolve(destination, "hear-g1-getup-artifacts.tar.gz"));
  writeFileSync(resolve(destination, "training-state.json"), `${JSON.stringify({
    protocol: "hear-g1-getup-training-state-v1",
    stage: parsed.evaluation.deployment_accepted ? "completed" : "rejected",
    checkpoint_iteration: parsed.training.iterations - 1,
    target_iterations: parsed.training.iterations,
    deployment_accepted: parsed.evaluation.deployment_accepted,
    updated_at: new Date().toISOString()
  }, null, 2)}\n`, "utf8");
}

function assertInputs() {
  if (!/^[A-Za-z0-9._-]+$/.test(session)) {
    throw new Error(`Unsafe G1 get-up session identity: ${session}`);
  }
  for (const path of BUNDLE_PATHS) {
    if (!existsSync(resolve(path))) {
      throw new Error(`G1 get-up bundle input is missing: ${path}`);
    }
  }
  const segments = driveDirectory.split("/");
  if (segments.some((segment) => (
    !segment || segment === "." || segment === ".." || !/^[\w.-]+$/.test(segment)
  ))) {
    throw new Error(`Unsafe G1 get-up Drive directory: ${driveDirectory}`);
  }
}

async function extractArtifacts() {
  mkdirSync(output);
  try {
    await extract({
      file: archive,
      cwd: output,
      strip: 1,
      filter: (path) => ARTIFACT_FILES.has(path)
    });
    for (const path of ARTIFACT_FILES) {
      const filename = path.slice(path.lastIndexOf("/") + 1);
      if (!existsSync(resolve(output, filename))) {
        throw new Error(`G1 get-up archive is missing: ${filename}`);
      }
    }
    copyFileSync(report, resolve(output, "remote-report.json"));
  } catch (error) {
    rmSync(output, { recursive: true, force: true });
    throw error;
  }
}

function validateReport() {
  const parsed = JSON.parse(readFileSync(report, "utf8"));
  if (parsed.protocol !== "hear-g1-getup-policy-deployment-v1"
    || parsed.framework?.task_id !== "Hear-G1-Getup-v1"
    || parsed.policy?.input_protocol !== "hear-g1-getup-proprioception-v1"
    || parsed.policy?.input_size !== 99
    || parsed.policy?.output_protocol !== "hear-g1-getup-joint-target-v1"
    || parsed.policy?.output_size !== 29
    || parsed.policy?.onnx?.sha256 !== sha256(resolve(output, "g1_getup.onnx"))) {
    throw new Error("Downloaded G1 get-up report is incompatible");
  }
  return parsed;
}

function writeDriveBackup() {
  writeFileSync(driveBackup, [
    "import shutil",
    "from hashlib import sha256",
    "from pathlib import Path",
    "root = Path('/content/drive/MyDrive').resolve()",
    `target = (root / ${JSON.stringify(driveDirectory)}).resolve()`,
    "if root not in target.parents: raise RuntimeError(f'unsafe Drive target: {target}')",
    "target.mkdir(parents=True, exist_ok=True)",
    `sources = [Path(${JSON.stringify(REMOTE_ARCHIVE)}), Path(${JSON.stringify(REMOTE_REPORT)})]`,
    "temporaries = []",
    "try:",
    "  for source in sources:",
    "    if not source.is_file(): raise RuntimeError(f'missing artifact: {source}')",
    "    destination = target / source.name",
    "    temporary = target / (source.name + '.partial')",
    "    temporary.unlink(missing_ok=True)",
    "    temporaries.append(temporary)",
    "    shutil.copy2(source, temporary)",
    "    if temporary.stat().st_size != source.stat().st_size: raise RuntimeError('Drive byte count mismatch')",
    "    if sha256(temporary.read_bytes()).hexdigest() != sha256(source.read_bytes()).hexdigest():",
    "      raise RuntimeError('Drive SHA256 mismatch')",
    "    temporary.replace(destination)",
    "except Exception:",
    "  for temporary in temporaries: temporary.unlink(missing_ok=True)",
    "  raise",
    "print(f'[hear] G1 get-up artifacts persisted to {target}')",
    ""
  ].join("\n"), "utf8");
}

function colab(args, timeoutMs = 600_000) {
  const result = spawnSync("wsl.exe", ["-d", distro, "--", colabPath, ...args], {
    cwd: workspace,
    stdio: "inherit",
    shell: false,
    timeout: timeoutMs,
    windowsHide: true
  });
  if (result.error) throw result.error;
  return result.status ?? 1;
}

function requireSuccess(status, label) {
  if (status !== 0) throw new Error(`Unable to ${label}; exit code ${status}`);
}

function emergencyStop() {
  if (!activeSession) return;
  try { colab(["stop", "--session", activeSession], 120_000); } catch {}
  activeSession = null;
}

function parseOptions(args) {
  if (args[0] === "--") args = args.slice(1);
  const parsed = {};
  const names = new Map([
    ["--mode", ["mode", enumValue(["smoke", "train", "evaluate"])]],
    ["--distro", ["distro", String]],
    ["--colab-path", ["colabPath", String]],
    ["--gpu", ["gpu", String]],
    ["--session", ["session", String]],
    ["--output", ["output", String]],
    ["--drive-directory", ["driveDirectory", String]],
    ["--desktop-drive-root", ["desktopDriveRoot", String]],
    ["--evaluation-source", ["evaluationSource", String]],
    ["--iterations", ["iterations", positiveInteger]],
    ["--num-envs", ["numEnvs", positiveInteger]],
    ["--eval-envs", ["evalEnvs", positiveInteger]],
    ["--eval-batch-size", ["evalBatchSize", positiveInteger]],
    ["--eval-steps", ["evalSteps", positiveInteger]],
    ["--seed", ["seed", nonnegativeInteger]],
    ["--timeout-seconds", ["timeoutSeconds", positiveInteger]],
    ["--reuse-session", ["reuseSession", booleanValue]]
  ]);
  for (let index = 0; index < args.length; index += 2) {
    const entry = names.get(args[index]);
    const value = args[index + 1];
    if (!entry || value === undefined) {
      throw new Error(`Unknown or incomplete G1 get-up option: ${args[index] ?? ""}`);
    }
    parsed[entry[0]] = entry[1](value);
  }
  return parsed;
}

function enumValue(values) {
  return (value) => {
    if (!values.includes(value)) throw new Error(`Expected one of ${values.join(", ")}`);
    return value;
  };
}

function positiveInteger(value) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new Error(`Expected a positive integer, received: ${value}`);
  }
  return parsed;
}

function nonnegativeInteger(value) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 0) {
    throw new Error(`Expected a non-negative integer, received: ${value}`);
  }
  return parsed;
}

function booleanValue(value) {
  if (value === "true") return true;
  if (value === "false") return false;
  throw new Error(`Expected true or false, received: ${value}`);
}

function toWslPath(path) {
  const absolute = resolve(path).replaceAll("\\", "/");
  const match = /^([A-Za-z]):\/(.*)$/.exec(absolute);
  if (!match) throw new Error(`Cannot map Windows path into WSL: ${path}`);
  return `/mnt/${match[1].toLowerCase()}/${match[2]}`;
}

function sha256(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}
