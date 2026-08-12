import { createHash, randomUUID } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync
} from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { spawn, spawnSync } from "node:child_process";
import { create } from "tar";

const options = parseOptions(process.argv.slice(2));
const workspace = process.cwd();
const sourceRoot = resolve(options.sourceRoot
  ?? "artifacts/training/workyard-residual/hear-colab-reach-v15-qualification-20260811");
const checkpoint = resolve(sourceRoot, "workyard_reach_selected.pt");
const trainingReport = resolve(sourceRoot, "training-report.json");
const exporter = resolve("training/export_workyard_reach_policy.py");
const output = resolve(options.output
  ?? "artifacts/training/workyard-reach-deployment-v15");
const session = options.session
  ?? `hear-workyard-reach-export-${randomUUID().slice(0, 8)}`;
const distro = options.distro ?? "HEAR-Linux";
const colabPath = options.colabPath ?? "/home/hear/.local/bin/colab";
const temporaryRoot = resolve(".tmp/workyard-reach-export", session);
const bundle = resolve(temporaryRoot, "bundle.tar.gz");
const driver = resolve(temporaryRoot, "export.py");
const assembler = resolve(temporaryRoot, "assemble.py");
const partPrefix = resolve(temporaryRoot, "bundle-part");
const remoteBundle = "/content/hear-workyard-reach-export.tar.gz";
const remoteRoot = "/content/hear-workyard-reach-export";
const outputs = {
  jit: resolve(output, "workyard_reach.jit.pt"),
  onnx: resolve(output, "workyard_reach.onnx"),
  report: resolve(output, "reach-policy-report.json")
};

let activeSession = null;
let outputCreated = false;
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
  if (existsSync(temporaryRoot)) {
    throw new Error(`Reach export temporary directory already exists: ${temporaryRoot}`);
  }
  mkdirSync(temporaryRoot, { recursive: true });
  let failure;
  try {
    await create({
      cwd: workspace,
      file: bundle,
      gzip: true,
      portable: true
    }, [
      workspaceRelative(exporter),
      workspaceRelative(checkpoint),
      workspaceRelative(trainingReport)
    ]);
    writeDriver();
    activeSession = session;
    requireSuccess(await colab(["new", "--session", session, "--gpu", "L4"]),
      "create Colab reach export session");
    await uploadBundle();
    requireSuccess(await colab([
      "exec", "--session", session, "--file", toWslPath(driver), "--timeout", "1200"
    ], false, 1_260_000), "export accepted reach policy");
    mkdirSync(output, { recursive: true });
    outputCreated = true;
    for (const [name, local] of Object.entries(outputs)) {
      requireSuccess(await retryColab([
        "download", "--session", session, `${remoteRoot}/${local.split(/[\\/]/).at(-1)}`,
        toWslPath(local)
      ], `download reach ${name}`, local, 5), `download reach ${name}`);
    }
    validateOutputs();
    console.log(`Workyard reach deployment export: ${output}`);
  } catch (error) {
    failure = error;
  } finally {
    if (activeSession) {
      const stop = await retryColab(
        ["stop", "--session", activeSession],
        "stop Colab reach export session",
        null,
        3,
        120_000
      );
      activeSession = null;
      if (stop !== 0) {
        const stopError = new Error(`Failed to stop Colab session ${session}`);
        failure = failure
          ? new AggregateError([failure, stopError], "Reach export and cleanup failed")
          : stopError;
      }
    }
    if (failure && outputCreated) {
      rmSync(output, { recursive: true, force: true });
    }
    rmSync(temporaryRoot, { recursive: true, force: true });
  }
  if (failure) throw failure;
}

function assertInputs() {
  workspaceRelative(output);
  if (!/^[A-Za-z0-9._-]+$/.test(session)) {
    throw new Error(`Unsafe reach export session: ${session}`);
  }
  for (const path of [checkpoint, trainingReport, exporter]) {
    if (!existsSync(path)) throw new Error(`Reach export input is missing: ${path}`);
  }
  if (existsSync(output)) throw new Error(`Reach export output already exists: ${output}`);
  const report = JSON.parse(readFileSync(trainingReport, "utf8"));
  const selected = report.training?.checkpoint_selection?.selected_checkpoint;
  if (report.protocol !== "hear-workyard-residual-run-v4"
    || report.mode !== "train"
    || report.acceptance?.phase_one_accepted !== true
    || selected?.file !== "workyard_reach_selected.pt"
    || selected?.bytes !== readFileSync(checkpoint).byteLength
    || selected?.sha256 !== sha256(checkpoint)) {
    throw new Error("Reach export source is not the accepted v15 checkpoint");
  }
}

function writeDriver() {
  const checkpointRelative = workspaceRelative(checkpoint);
  const reportRelative = workspaceRelative(trainingReport);
  const exporterRelative = workspaceRelative(exporter);
  writeFileSync(driver, [
    "import subprocess, sys, tarfile",
    "from pathlib import Path",
    `bundle = Path(${JSON.stringify(remoteBundle)})`,
    `root = Path(${JSON.stringify(remoteRoot)})`,
    "root.mkdir()",
    "with tarfile.open(bundle, 'r:gz') as archive:",
    "  resolved = root.resolve()",
    "  for member in archive.getmembers():",
    "    target = (root / member.name).resolve()",
    "    if target != resolved and resolved not in target.parents:",
    "      raise RuntimeError(f'unsafe bundle member: {member.name}')",
    "  archive.extractall(root, filter='data')",
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check', '--no-input', 'onnx==1.18.0'], check=True)",
    "subprocess.run([",
    "  sys.executable, str(root / " + JSON.stringify(exporterRelative) + "),",
    "  '--checkpoint', str(root / " + JSON.stringify(checkpointRelative) + "),",
    "  '--training-report', str(root / " + JSON.stringify(reportRelative) + "),",
    "  '--output', str(root / 'workyard_reach.jit.pt'),",
    "  '--onnx-output', str(root / 'workyard_reach.onnx'),",
    "  '--report', str(root / 'reach-policy-report.json'),",
    "], check=True)",
    ""
  ].join("\n"), "utf8");
}

async function uploadBundle() {
  const value = readFileSync(bundle);
  const parts = [];
  for (let offset = 0, index = 0; offset < value.length; index += 1) {
    const path = `${partPrefix}-${String(index).padStart(3, "0")}`;
    writeFileSync(path, value.subarray(offset, Math.min(offset + 4 * 1024 * 1024, value.length)));
    parts.push(path);
    offset += 4 * 1024 * 1024;
  }
  const remoteParts = parts.map((_, index) => (
    `${remoteBundle}.part-${String(index).padStart(3, "0")}`
  ));
  writeFileSync(assembler, [
    "from hashlib import sha256",
    "from pathlib import Path",
    `target = Path(${JSON.stringify(remoteBundle)})`,
    `parts = [Path(value) for value in ${JSON.stringify(remoteParts)}]`,
    `expected_bytes = ${value.length}`,
    `expected_sha256 = ${JSON.stringify(sha256(bundle))}`,
    "valid_target = (target.is_file() and target.stat().st_size == expected_bytes",
    "  and sha256(target.read_bytes()).hexdigest() == expected_sha256)",
    "if not valid_target:",
    "  missing = [str(part) for part in parts if not part.is_file()]",
    "  if missing: raise RuntimeError(f'missing reach bundle parts: {missing}')",
    "  with target.open('wb') as output:",
    "    for part in parts: output.write(part.read_bytes())",
    "assert target.stat().st_size == expected_bytes",
    "assert sha256(target.read_bytes()).hexdigest() == expected_sha256",
    "for part in parts: part.unlink(missing_ok=True)",
    ""
  ].join("\n"), "utf8");
  for (let index = 0; index < parts.length; index += 1) {
    requireSuccess(await retryColab([
      "upload", "--session", session, toWslPath(parts[index]), remoteParts[index]
    ], `upload reach bundle part ${index + 1}/${parts.length}`, null, 5),
    `upload reach bundle part ${index + 1}/${parts.length}`);
  }
  requireSuccess(await retryColab([
    "exec", "--session", session, "--file", toWslPath(assembler), "--timeout", "300"
  ], "assemble reach export bundle", null, 5, 360_000),
  "assemble reach export bundle");
}

function validateOutputs() {
  const report = JSON.parse(readFileSync(outputs.report, "utf8"));
  if (report.protocol !== "hear-frozen-reach-policy-export-v1"
    || report.source?.checkpoint_sha256 !== sha256(checkpoint)
    || report.source?.held_out_environment_count !== 500
    || report.source?.held_out_success_rate < 0.99
    || report.policy?.file !== "workyard_reach.jit.pt"
    || report.policy?.sha256 !== sha256(outputs.jit)
    || report.onnx?.file !== "workyard_reach.onnx"
    || report.onnx?.sha256 !== sha256(outputs.onnx)
    || report.onnx?.opset !== 17
    || report.validation?.onnx_checker_full !== true
    || report.validation?.maximum_onnx_error > 1e-5) {
    throw new Error("Downloaded reach deployment export failed validation");
  }
}

function colab(args, tolerateFailure = false, timeoutMs = 600_000) {
  console.log(`[${session}] colab ${args.join(" ")}`);
  return new Promise((resolveStatus, reject) => {
    const child = spawn("wsl.exe", ["-d", distro, "--", colabPath, ...args], {
      cwd: workspace,
      shell: false,
      stdio: "inherit",
      windowsHide: true
    });
    const watchdog = setTimeout(() => {
      child.kill();
      reject(new Error(`Colab reach export command timed out: ${args[0]}`));
    }, timeoutMs);
    child.once("error", (error) => {
      clearTimeout(watchdog);
      reject(error);
    });
    child.once("exit", (code, signal) => {
      clearTimeout(watchdog);
      if (signal && !tolerateFailure) {
        reject(new Error(`Colab reach export command terminated by ${signal}`));
      } else {
        resolveStatus(code ?? 1);
      }
    });
  });
}

async function retryColab(
  args,
  operation,
  partialDownload = null,
  attempts = 3,
  timeoutMs = 600_000
) {
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    if (partialDownload) rmSync(partialDownload, { force: true });
    const status = await colab(args, true, timeoutMs);
    if (status === 0) return status;
    if (attempt === attempts) return status;
    const delayMs = attempt * 15_000;
    console.warn(
      `[${session}] ${operation} failed (attempt ${attempt}/${attempts}); `
      + `retrying in ${delayMs / 1_000}s`
    );
    await new Promise((resolveDelay) => setTimeout(resolveDelay, delayMs));
  }
  return 1;
}

function emergencyStop() {
  if (!activeSession) return;
  spawnSync("wsl.exe", [
    "-d", distro, "--", colabPath, "stop", "--session", activeSession
  ], { cwd: workspace, shell: false, stdio: "inherit", timeout: 60_000, windowsHide: true });
  activeSession = null;
}

function workspaceRelative(path) {
  const value = relative(workspace, path).replaceAll("\\", "/");
  if (!value || value === ".." || value.startsWith("../")) {
    throw new Error(`Reach export input escaped the workspace: ${path}`);
  }
  return value;
}

function toWslPath(path) {
  const absolute = resolve(path).replaceAll("\\", "/");
  const match = /^([A-Za-z]):\/(.*)$/.exec(absolute);
  if (!match) throw new Error(`Cannot map path to WSL: ${path}`);
  return `/mnt/${match[1].toLowerCase()}/${match[2]}`;
}

function sha256(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function requireSuccess(status, operation) {
  if (status !== 0) throw new Error(`Failed to ${operation} (exit ${status})`);
}

function parseOptions(args) {
  if (args[0] === "--") args = args.slice(1);
  const parsed = {};
  const names = new Map([
    ["--source-root", "sourceRoot"],
    ["--output", "output"],
    ["--session", "session"],
    ["--distro", "distro"],
    ["--colab-path", "colabPath"]
  ]);
  for (let index = 0; index < args.length; index += 2) {
    const target = names.get(args[index]);
    const value = args[index + 1];
    if (!target || !value) throw new Error(`Unknown reach export option: ${args[index] ?? ""}`);
    parsed[target] = value;
  }
  return parsed;
}
