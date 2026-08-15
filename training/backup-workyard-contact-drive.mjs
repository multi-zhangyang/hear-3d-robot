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

const REMOTE_BUNDLE = "/content/hear-workyard-contact-drive.tar.gz";
const CHUNK_BYTES = 4 * 1024 * 1024;
const options = parseOptions(process.argv.slice(2));
const workspace = process.cwd();
const sourceRoot = resolve(
  options.sourceRoot ?? "artifacts/training/workyard-contact/formal-v2"
);
const deploymentRoot = resolve(
  options.deploymentRoot
    ?? "artifacts/training/workyard-contact-deployment/formal-v2"
);
const sourceArchive = resolve(options.sourceArchive ?? `${sourceRoot}.tar.gz`);
const session = options.session
  ?? `hear-contact-drive-${randomUUID().slice(0, 8)}`;
const driveDirectory = options.driveDirectory ?? "HEAR/workyard-contact/formal-v2";
const distro = options.distro ?? "HEAR-Linux";
const colabPath = options.colabPath ?? "/home/hear/.local/bin/colab";
const temporaryRoot = resolve(".tmp/workyard-contact-drive", session);
const bundle = resolve(temporaryRoot, "backup.tar.gz");
const manifest = resolve(temporaryRoot, "drive-manifest.json");
const assembler = resolve(temporaryRoot, "assemble.py");
const installer = resolve(temporaryRoot, "install.py");
const partPrefix = resolve(temporaryRoot, "bundle-part");

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
  if (existsSync(temporaryRoot)) {
    throw new Error(`Drive backup temporary directory exists: ${temporaryRoot}`);
  }
  mkdirSync(temporaryRoot, { recursive: true });
  let failure;
  try {
    const included = [sourceRoot, deploymentRoot];
    if (existsSync(sourceArchive)) included.push(sourceArchive);
    writeFileSync(manifest, `${JSON.stringify({
      protocol: "hear-workyard-contact-drive-backup-v1",
      created_at: new Date().toISOString(),
      drive_directory: driveDirectory,
      included: included.map((path) => ({
        path: workspaceRelative(path),
        kind: path === sourceArchive ? "training_archive" : "directory"
      }))
    }, null, 2)}\n`, "utf8");
    await create({
      cwd: workspace,
      file: bundle,
      gzip: true,
      portable: true
    }, [
      ...included.map(workspaceRelative),
      workspaceRelative(manifest)
    ]);
    activeSession = session;
    requireSuccess(
      await colab(["new", "--session", session]),
      "create Colab Drive backup session"
    );
    await uploadBundle();
    requireSuccess(await colab([
      "drivemount", "--session", session, "/content/drive"
    ], false, 300_000), "mount Google Drive");
    writeInstaller();
    requireSuccess(await colab([
      "exec", "--session", session, "--file", toWslPath(installer),
      "--timeout", "1200"
    ], false, 1_260_000), "write verified Workyard backup to Drive");
    console.log(`Workyard contact backup: MyDrive/${driveDirectory}`);
  } catch (error) {
    failure = error;
  } finally {
    if (activeSession) {
      const stop = await retryColab(
        ["stop", "--session", activeSession],
        "stop Drive backup session",
        3,
        120_000
      );
      activeSession = null;
      if (stop !== 0) {
        const cleanup = new Error(`Failed to stop Colab session ${session}`);
        failure = failure
          ? new AggregateError([failure, cleanup], "Drive backup cleanup failed")
          : cleanup;
      }
    }
    rmSync(temporaryRoot, { recursive: true, force: true });
  }
  if (failure) throw failure;
}

function assertInputs() {
  if (!/^[A-Za-z0-9._-]+$/.test(session)) {
    throw new Error(`Unsafe Drive backup session: ${session}`);
  }
  const segments = driveDirectory.split("/");
  if (segments.length === 0 || segments.some((segment) => (
    !segment || segment === "." || segment === ".." || !/^[\w.-]+$/.test(segment)
  ))) {
    throw new Error(`Unsafe Drive backup directory: ${driveDirectory}`);
  }
  for (const path of [sourceRoot, deploymentRoot]) {
    workspaceRelative(path);
    if (!existsSync(path)) throw new Error(`Drive backup input is missing: ${path}`);
  }
  workspaceRelative(sourceArchive);
  workspaceRelative(temporaryRoot);
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
  if (parts.length === 0) throw new Error("Drive backup bundle is empty");
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
    "with target.open('wb') as output:",
    "  for part in parts:",
    "    if not part.is_file(): raise RuntimeError(f'missing backup part: {part}')",
    "    output.write(part.read_bytes())",
    "if target.stat().st_size != expected_bytes: raise RuntimeError('backup byte count mismatch')",
    "if sha256(target.read_bytes()).hexdigest() != expected_sha256: raise RuntimeError('backup sha256 mismatch')",
    "for part in parts: part.unlink(missing_ok=True)",
    ""
  ].join("\n"), "utf8");
  for (let index = 0; index < parts.length; index += 1) {
    requireSuccess(await retryColab([
      "upload", "--session", session, toWslPath(parts[index]), remoteParts[index]
    ], `upload Drive backup part ${index + 1}/${parts.length}`, 5),
    `upload Drive backup part ${index + 1}/${parts.length}`);
  }
  requireSuccess(await retryColab([
    "exec", "--session", session, "--file", toWslPath(assembler),
    "--timeout", "300"
  ], "assemble Drive backup", 5, 360_000), "assemble Drive backup");
}

function writeInstaller() {
  writeFileSync(installer, [
    "import shutil, tarfile",
    "from pathlib import Path",
    `bundle = Path(${JSON.stringify(REMOTE_BUNDLE)})`,
    "drive_root = Path('/content/drive/MyDrive').resolve()",
    `target = (drive_root / ${JSON.stringify(driveDirectory)}).resolve()`,
    "if drive_root not in target.parents: raise RuntimeError(f'unsafe Drive target: {target}')",
    "if target.exists(): raise RuntimeError(f'Drive backup target already exists: {target}')",
    "target.mkdir(parents=True)",
    "try:",
    "  with tarfile.open(bundle, 'r:gz') as archive:",
    "    for member in archive.getmembers():",
    "      resolved = (target / member.name).resolve()",
    "      if resolved != target and target not in resolved.parents:",
    "        raise RuntimeError(f'unsafe backup member: {member.name}')",
    "    archive.extractall(target, filter='data')",
    "except Exception:",
    "  shutil.rmtree(target, ignore_errors=True)",
    "  raise",
    "print(f'[hear] Drive backup installed: {target}')",
    ""
  ].join("\n"), "utf8");
}

function sha256(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function workspaceRelative(path) {
  const value = relative(workspace, resolve(path)).replaceAll("\\", "/");
  if (!value || value === "." || value === ".." || value.startsWith("../")) {
    throw new Error(`Drive backup path must stay in the workspace: ${path}`);
  }
  return value;
}

function toWslPath(path) {
  const absolute = resolve(path);
  const match = /^([A-Za-z]):[\\/](.*)$/.exec(absolute);
  if (!match) throw new Error(`Cannot convert path to WSL: ${absolute}`);
  return `/mnt/${match[1].toLowerCase()}/${match[2].replaceAll("\\", "/")}`;
}

function colab(args, tolerateFailure = false, timeoutMs = 600_000) {
  console.log(`[${session}] colab ${args.join(" ")}`);
  return new Promise((resolveStatus, reject) => {
    const child = spawn("wsl.exe", ["-d", distro, "--", colabPath, ...args], {
      cwd: workspace,
      shell: false,
      windowsHide: true,
      stdio: "inherit"
    });
    const watchdog = setTimeout(() => {
      terminateProcessTree(child.pid);
      reject(new Error(`Colab Drive command timed out: ${args[0]}`));
    }, timeoutMs);
    child.once("error", (error) => {
      clearTimeout(watchdog);
      reject(error);
    });
    child.once("exit", (code, signal) => {
      clearTimeout(watchdog);
      if (signal && !tolerateFailure) {
        return reject(new Error(`Colab Drive command terminated by ${signal}`));
      }
      resolveStatus(code ?? (signal ? 1 : 0));
    });
  });
}

async function retryColab(args, operation, attempts = 3, timeoutMs = 600_000) {
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    const status = await colab(args, true, timeoutMs);
    if (status === 0) return 0;
    if (attempt === attempts) return status;
    console.warn(`[${session}] ${operation} failed (${attempt}/${attempts})`);
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 15_000 * attempt));
  }
  return 1;
}

function requireSuccess(status, operation) {
  if (status !== 0) throw new Error(`Failed to ${operation} (exit ${status})`);
}

function emergencyStop() {
  if (!activeSession) return;
  spawnSync("wsl.exe", [
    "-d", distro, "--", colabPath, "stop", "--session", activeSession
  ], { cwd: workspace, shell: false, windowsHide: true, stdio: "ignore" });
  activeSession = null;
}

function terminateProcessTree(pid) {
  if (!Number.isSafeInteger(pid) || pid <= 0) return;
  spawnSync("taskkill.exe", ["/PID", String(pid), "/T", "/F"], {
    cwd: workspace,
    shell: false,
    windowsHide: true,
    stdio: "ignore"
  });
}

function parseOptions(args) {
  if (args[0] === "--") args = args.slice(1);
  const parsed = {};
  const names = new Map([
    ["--source-root", "sourceRoot"],
    ["--deployment-root", "deploymentRoot"],
    ["--source-archive", "sourceArchive"],
    ["--drive-directory", "driveDirectory"],
    ["--session", "session"],
    ["--distro", "distro"],
    ["--colab-path", "colabPath"]
  ]);
  for (let index = 0; index < args.length; index += 2) {
    const target = names.get(args[index]);
    const value = args[index + 1];
    if (!target || !value) throw new Error(`Unknown Drive backup option: ${args[index] ?? ""}`);
    parsed[target] = value;
  }
  return parsed;
}
