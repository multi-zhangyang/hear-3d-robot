import { createHash, randomUUID } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync
} from "node:fs";
import { relative, resolve } from "node:path";
import { spawn, spawnSync } from "node:child_process";
import { create } from "tar";

const REMOTE_BUNDLE = "/content/hear-workyard-contact-export.tar.gz";
const REMOTE_ROOT = "/content/hear-workyard-contact-export";
const CHUNK_BYTES = 4 * 1024 * 1024;

const options = parseOptions(process.argv.slice(2));
const workspace = process.cwd();
const sourceRoot = resolve(
  options.sourceRoot ?? "artifacts/training/workyard-contact/formal-v2"
);
const checkpoint = resolve(sourceRoot, "workyard_contact_selected.pt");
const trainingReport = resolve(sourceRoot, "training-report.json");
const exporter = resolve("training/export_workyard_contact_policy.py");
const trainingContract = resolve("training/workyard-contact-task-v1.json");
const trainingEnvironment = resolve("training/workyard_contact_mjlab_env.py");
const plantXml = resolve("assets/humanoid/g1/g1_with_hands.xml");
const output = resolve(
  options.output ?? "artifacts/training/workyard-contact-deployment/formal-v2"
);
const session = options.session
  ?? `hear-workyard-contact-export-${randomUUID().slice(0, 8)}`;
const distro = options.distro ?? "HEAR-Linux";
const colabPath = options.colabPath ?? "/home/hear/.local/bin/colab";
const temporaryRoot = resolve(".tmp/workyard-contact-export", session);
const bundle = resolve(temporaryRoot, "bundle.tar.gz");
const driver = resolve(temporaryRoot, "export.py");
const assembler = resolve(temporaryRoot, "assemble.py");
const partPrefix = resolve(temporaryRoot, "bundle-part");
const outputs = {
  torchscript: resolve(output, "workyard_contact.jit.pt"),
  onnx: resolve(output, "workyard_contact.onnx"),
  report: resolve(output, "contact-policy-report.json")
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
    throw new Error(
      `Contact export temporary directory already exists: ${temporaryRoot}`
    );
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
      workspaceRelative(trainingContract),
      workspaceRelative(trainingEnvironment),
      workspaceRelative(plantXml),
      workspaceRelative(checkpoint),
      workspaceRelative(trainingReport)
    ]);
    writeDriver();
    activeSession = session;
    requireSuccess(
      await colab(["new", "--session", session, "--gpu", "L4"]),
      "create Colab contact export session"
    );
    await uploadBundle();
    requireSuccess(await colab([
      "exec", "--session", session, "--file", toWslPath(driver),
      "--timeout", "1200"
    ], false, 1_260_000), "export accepted contact policy");
    mkdirSync(output, { recursive: true });
    outputCreated = true;
    for (const [name, local] of Object.entries(outputs)) {
      const remote = `${REMOTE_ROOT}/${local.split(/[\\/]/).at(-1)}`;
      requireSuccess(await retryColab([
        "download", "--session", session, remote, toWslPath(local)
      ], `download contact ${name}`, local, 5), `download contact ${name}`);
    }
    validateOutputs();
    console.log(`Workyard contact deployment export: ${output}`);
  } catch (error) {
    failure = error;
  } finally {
    if (activeSession) {
      const stop = await retryColab(
        ["stop", "--session", activeSession],
        "stop Colab contact export session",
        null,
        3,
        120_000
      );
      activeSession = null;
      if (stop !== 0) {
        const stopError = new Error(`Failed to stop Colab session ${session}`);
        failure = failure
          ? new AggregateError(
              [failure, stopError],
              "Contact export and cleanup failed"
            )
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
    throw new Error(`Unsafe contact export session: ${session}`);
  }
  for (const path of [
    checkpoint,
    trainingReport,
    exporter,
    trainingContract,
    trainingEnvironment,
    plantXml
  ]) {
    if (!existsSync(path)) {
      throw new Error(`Contact export input is missing: ${path}`);
    }
  }
  if (existsSync(output)) {
    throw new Error(`Contact export output already exists: ${output}`);
  }
  const report = JSON.parse(readFileSync(trainingReport, "utf8"));
  const finalGate = report.acceptance?.final_gate;
  const selected = report.training?.checkpoint_selection?.selected_checkpoint;
  const reportedLead = report.contract?.hand_max_closing_joint_lead_rad;
  const reportedSupport = report.contract?.opposing_support_coordination;
  if (report.protocol !== "hear-workyard-contact-run-v1"
    || report.mode !== "train"
    || report.ready !== true
    || report.acceptance?.verified_grasp_policy_accepted !== true
    || finalGate?.protocol !== "hear-workyard-contact-independent-500-gate-v1"
    || finalGate?.passed !== true
    || !Object.values(finalGate?.checks ?? {}).every((value) => value === true)
    || report.evaluation?.episode_count !== 500
    || selected?.file !== "workyard_contact_selected.pt"
    || selected?.bytes !== readFileSync(checkpoint).byteLength
    || selected?.sha256 !== sha256(checkpoint)
    || report.bundle?.contract_sha256 !== sha256(trainingContract)
    || report.bundle?.environment_sha256 !== sha256(trainingEnvironment)
    || (reportedLead !== undefined && reportedLead !== 0.25)
    || (reportedSupport !== undefined && reportedSupport !== 0.4)) {
    throw new Error(
      "Contact export source is not backed by the accepted formal 500-seed gate"
    );
  }
}

function writeDriver() {
  const checkpointRelative = workspaceRelative(checkpoint);
  const reportRelative = workspaceRelative(trainingReport);
  const exporterRelative = workspaceRelative(exporter);
  const trainingContractRelative = workspaceRelative(trainingContract);
  const trainingEnvironmentRelative = workspaceRelative(trainingEnvironment);
  const plantXmlRelative = workspaceRelative(plantXml);
  writeFileSync(driver, [
    "import subprocess, sys, tarfile",
    "from pathlib import Path",
    `bundle = Path(${JSON.stringify(REMOTE_BUNDLE)})`,
    `root = Path(${JSON.stringify(REMOTE_ROOT)})`,
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
    "  '--training-contract', str(root / " + JSON.stringify(trainingContractRelative) + "),",
    "  '--training-environment', str(root / " + JSON.stringify(trainingEnvironmentRelative) + "),",
    "  '--plant-xml', str(root / " + JSON.stringify(plantXmlRelative) + "),",
    "  '--torchscript-output', str(root / 'workyard_contact.jit.pt'),",
    "  '--onnx-output', str(root / 'workyard_contact.onnx'),",
    "  '--report', str(root / 'contact-policy-report.json'),",
    "], check=True)",
    ""
  ].join("\n"), "utf8");
}

async function uploadBundle() {
  const value = readFileSync(bundle);
  const parts = [];
  for (let offset = 0, index = 0; offset < value.length; index += 1) {
    const path = `${partPrefix}-${String(index).padStart(3, "0")}`;
    writeFileSync(path, value.subarray(offset, Math.min(offset + CHUNK_BYTES, value.length)));
    parts.push(path);
    offset += CHUNK_BYTES;
  }
  const remoteParts = parts.map((_, index) => (
    `${REMOTE_BUNDLE}.part-${String(index).padStart(3, "0")}`
  ));
  writeFileSync(assembler, [
    "from hashlib import sha256",
    "from pathlib import Path",
    `target = Path(${JSON.stringify(REMOTE_BUNDLE)})`,
    `parts = [Path(value) for value in ${JSON.stringify(remoteParts)}]`,
    `expected_bytes = ${value.length}`,
    `expected_sha256 = ${JSON.stringify(sha256(bundle))}`,
    "valid_target = (target.is_file() and target.stat().st_size == expected_bytes",
    "  and sha256(target.read_bytes()).hexdigest() == expected_sha256)",
    "if not valid_target:",
    "  missing = [str(part) for part in parts if not part.is_file()]",
    "  if missing: raise RuntimeError(f'missing contact bundle parts: {missing}')",
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
    ], `upload contact bundle part ${index + 1}/${parts.length}`, null, 5),
    `upload contact bundle part ${index + 1}/${parts.length}`);
  }
  requireSuccess(await retryColab([
    "exec", "--session", session, "--file", toWslPath(assembler),
    "--timeout", "300"
  ], "assemble contact export bundle", null, 5, 360_000),
  "assemble contact export bundle");
}

function validateOutputs() {
  const report = JSON.parse(readFileSync(outputs.report, "utf8"));
  if (report.protocol !== "hear-frozen-contact-policy-export-v1"
    || report.source?.checkpoint_sha256 !== sha256(checkpoint)
    || report.source?.training_contract_sha256 !== sha256(trainingContract)
    || report.source?.training_environment_sha256 !== sha256(trainingEnvironment)
    || report.source?.formal_gate_passed !== true
    || report.source?.held_out_episode_count !== 500
    || report.source?.held_out_success_rate < 0.75
    || report.source?.maximum_active_hand_force_n > 30
    || report.plant?.protocol !== "hear-workyard-contact-deployment-plant-v1"
    || report.plant?.g1_xml?.sha256 !== sha256(plantXml)
    || report.plant?.g1_xml?.bytes !== readFileSync(plantXml).byteLength
    || report.plant?.hand_joint_count !== 14
    || report.plant?.hand_position_kp !== 2.5
    || report.plant?.hand_velocity_damping !== 0.3
    || report.plant?.workyard_rod?.shape !== "cylinder"
    || report.plant?.workyard_rod?.radius_m !== 0.03
    || report.plant?.workyard_rod?.half_height_m !== 0.11
    || report.plant?.workyard_rod?.mass_kg !== 0.35
    || report.policy?.torchscript?.file !== "workyard_contact.jit.pt"
    || report.policy?.torchscript?.sha256 !== sha256(outputs.torchscript)
    || report.policy?.onnx?.file !== "workyard_contact.onnx"
    || report.policy?.onnx?.sha256 !== sha256(outputs.onnx)
    || report.policy?.onnx?.opset !== 17
    || report.policy?.input_size !== 247
    || report.policy?.output_size !== 8
    || report.harness?.maximum_closing_joint_lead_rad !== 0.25
    || report.validation?.finite !== true
    || report.validation?.onnx_checker_full !== true
    || report.validation?.maximum_onnx_error > 1e-5) {
    throw new Error("Downloaded contact deployment export failed validation");
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
      reject(new Error(`Colab contact export command timed out: ${args[0]}`));
    }, timeoutMs);
    child.once("error", (error) => {
      clearTimeout(watchdog);
      reject(error);
    });
    child.once("exit", (code, signal) => {
      clearTimeout(watchdog);
      if (signal && !tolerateFailure) {
        reject(new Error(`Colab contact export command terminated by ${signal}`));
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
  ], {
    cwd: workspace,
    shell: false,
    stdio: "inherit",
    timeout: 60_000,
    windowsHide: true
  });
  activeSession = null;
}

function workspaceRelative(path) {
  const value = relative(workspace, path).replaceAll("\\", "/");
  if (!value || value === ".." || value.startsWith("../")) {
    throw new Error(`Contact export input escaped the workspace: ${path}`);
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
    if (!target || !value) {
      throw new Error(`Unknown contact export option: ${args[index] ?? ""}`);
    }
    parsed[target] = value;
  }
  return parsed;
}
