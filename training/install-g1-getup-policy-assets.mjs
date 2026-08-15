import { createHash, randomUUID } from "node:crypto";
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  rmSync
} from "node:fs";
import { relative, resolve } from "node:path";

const options = parseOptions(process.argv.slice(2));
const workspace = process.cwd();
const sourceRoot = resolve(options.sourceRoot);
const destination = resolve(
  options.destination ?? "assets/humanoid/controllers/g1-getup"
);
const source = {
  onnx: resolve(sourceRoot, "g1_getup.onnx"),
  report: resolve(sourceRoot, "getup-policy-report.json")
};

assertWorkspaceChild(sourceRoot, "G1 get-up source");
assertWorkspaceChild(destination, "G1 get-up destination");
for (const path of Object.values(source)) {
  if (!existsSync(path)) throw new Error(`G1 get-up deployment input is missing: ${path}`);
}
const report = JSON.parse(readFileSync(source.report, "utf8"));
validateReport(report);

const parent = resolve(destination, "..");
mkdirSync(parent, { recursive: true });
const transaction = resolve(parent, `.g1-getup-install-${randomUUID()}`);
const staged = resolve(transaction, "next");
const previous = resolve(transaction, "previous");
assertWorkspaceChild(transaction, "G1 get-up install transaction");
mkdirSync(staged, { recursive: true });
let backedUp = false;
let installed = false;
try {
  copyFileSync(source.onnx, resolve(staged, "g1_getup.onnx"));
  copyFileSync(source.report, resolve(staged, "getup-policy-report.json"));
  assertInstalled(staged);
  if (existsSync(destination)) {
    renameSync(destination, previous);
    backedUp = true;
  }
  renameSync(staged, destination);
  installed = true;
  assertInstalled(destination);
  rmSync(transaction, { recursive: true, force: true });
} catch (error) {
  if (installed && existsSync(destination)) {
    rmSync(destination, { recursive: true, force: true });
  }
  if (backedUp && existsSync(previous)) renameSync(previous, destination);
  if (existsSync(transaction)) rmSync(transaction, { recursive: true, force: true });
  throw error;
}

console.log(JSON.stringify({
  protocol: "hear-g1-getup-policy-install-v1",
  destination: relative(workspace, destination).replaceAll("\\", "/"),
  onnx_sha256: sha256(source.onnx),
  evaluation: report.evaluation
}, null, 2));

function validateReport(value) {
  const policy = value.policy;
  const evaluation = value.evaluation;
  if (value.protocol !== "hear-g1-getup-policy-deployment-v1"
    || value.framework?.task_id !== "Hear-G1-Getup-v1"
    || policy?.onnx?.file !== "g1_getup.onnx"
    || policy?.onnx?.bytes !== readFileSync(source.onnx).byteLength
    || policy?.onnx?.sha256 !== sha256(source.onnx)
    || policy?.input !== "obs"
    || policy?.input_protocol !== "hear-g1-getup-proprioception-v1"
    || policy?.input_size !== 99
    || policy?.output !== "actions"
    || policy?.output_protocol !== "hear-g1-getup-joint-target-v1"
    || policy?.output_size !== 29
    || policy?.actor_inputs !== "proprioception_only_no_reference_phase"
    || policy?.action_mapping !== "neutral_piecewise_soft_joint_limits"
    || !Array.isArray(policy?.joint_names) || policy.joint_names.length !== 29
    || !Array.isArray(policy?.default_joint_positions)
      || policy.default_joint_positions.length !== 29
    || !Array.isArray(policy?.joint_lower_limits)
      || policy.joint_lower_limits.length !== 29
    || !Array.isArray(policy?.joint_upper_limits)
      || policy.joint_upper_limits.length !== 29
    || !Array.isArray(policy?.stiffness) || policy.stiffness.length !== 29
    || !Array.isArray(policy?.damping) || policy.damping.length !== 29
    || evaluation?.deployment_accepted !== true
    || evaluation?.episode_count < 500
    || evaluation?.overall_success_rate < 0.80
    || evaluation?.prone_success_rate < 0.75
    || evaluation?.supine_success_rate < 0.75
    || evaluation?.side_success_rate < 0.75
    || evaluation?.stable_exit_rate < 0.75
    || evaluation?.non_finite_action_count !== 0) {
    throw new Error("G1 get-up deployment export is not qualified for installation");
  }
}

function assertInstalled(root) {
  if (sha256(resolve(root, "g1_getup.onnx")) !== sha256(source.onnx)
    || sha256(resolve(root, "getup-policy-report.json")) !== sha256(source.report)) {
    throw new Error("Installed G1 get-up policy changed during copy");
  }
}

function assertWorkspaceChild(path, label) {
  const child = relative(workspace, resolve(path));
  if (!child || child === ".." || child.startsWith("../") || child.startsWith("..\\")) {
    throw new Error(`Unsafe ${label}: ${path}`);
  }
}

function sha256(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function parseOptions(args) {
  if (args[0] === "--") args = args.slice(1);
  const parsed = {};
  const names = new Map([
    ["--source-root", "sourceRoot"],
    ["--destination", "destination"]
  ]);
  for (let index = 0; index < args.length; index += 2) {
    const target = names.get(args[index]);
    const value = args[index + 1];
    if (!target || !value) {
      throw new Error(`Unknown G1 get-up install option: ${args[index] ?? ""}`);
    }
    parsed[target] = value;
  }
  if (!parsed.sourceRoot) throw new Error("--source-root is required");
  return parsed;
}
