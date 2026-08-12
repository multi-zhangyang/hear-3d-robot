import { createHash } from "node:crypto";
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync
} from "node:fs";
import { relative, resolve } from "node:path";

const options = parseOptions(process.argv.slice(2));
const workspace = process.cwd();
const reachRoot = resolve(
  options.reachRoot ?? "artifacts/training/workyard-reach-deployment-v15"
);
const contactRoot = resolve(
  options.contactRoot ?? "artifacts/training/workyard-contact-deployment/formal-v2"
);
const destinationRoot = resolve(
  options.destinationRoot ?? "assets/humanoid/controllers"
);
const reachDestination = resolve(destinationRoot, "workyard-reach");
const contactDestination = resolve(destinationRoot, "workyard-contact");

const reach = {
  onnx: resolve(reachRoot, "workyard_reach.onnx"),
  report: resolve(reachRoot, "reach-policy-report.json")
};
const contact = {
  onnx: resolve(contactRoot, "workyard_contact.onnx"),
  report: resolve(contactRoot, "contact-policy-report.json")
};

main();

function main() {
  assertWorkspaceChild(destinationRoot, "deployment asset root");
  for (const path of [reach.onnx, reach.report, contact.onnx, contact.report]) {
    if (!existsSync(path)) throw new Error(`Policy deployment input is missing: ${path}`);
  }
  for (const path of [reachDestination, contactDestination]) {
    if (existsSync(path)) throw new Error(`Policy asset destination already exists: ${path}`);
  }
  const reachReport = JSON.parse(readFileSync(reach.report, "utf8"));
  const contactReport = JSON.parse(readFileSync(contact.report, "utf8"));
  validateReach(reachReport);
  validateContact(contactReport);

  const created = [];
  try {
    for (const path of [reachDestination, contactDestination]) {
      mkdirSync(path, { recursive: true });
      created.push(path);
    }
    copyFileSync(reach.onnx, resolve(reachDestination, "workyard_reach.onnx"));
    copyFileSync(reach.report, resolve(reachDestination, "reach-policy-report.json"));
    copyFileSync(contact.onnx, resolve(contactDestination, "workyard_contact.onnx"));
    copyFileSync(
      contact.report,
      resolve(contactDestination, "contact-policy-report.json")
    );
    assertInstalled(reachDestination, reach, reachReport, "reach");
    assertInstalled(contactDestination, contact, contactReport, "contact");
  } catch (error) {
    for (const path of created.reverse()) {
      assertWorkspaceChild(path, "partial deployment asset");
      rmSync(path, { recursive: true, force: true });
    }
    throw error;
  }
  console.log(JSON.stringify({
    protocol: "hear-workyard-policy-assets-install-v1",
    reach: {
      destination: relative(workspace, reachDestination).replaceAll("\\", "/"),
      onnx_sha256: sha256(reach.onnx)
    },
    contact: {
      destination: relative(workspace, contactDestination).replaceAll("\\", "/"),
      onnx_sha256: sha256(contact.onnx)
    }
  }, null, 2));
}

function validateReach(report) {
  if (report.protocol !== "hear-frozen-reach-policy-export-v1"
    || report.source?.held_out_environment_count !== 500
    || report.source?.held_out_success_rate < 0.99
    || report.onnx?.file !== "workyard_reach.onnx"
    || report.onnx?.bytes !== readFileSync(reach.onnx).byteLength
    || report.onnx?.sha256 !== sha256(reach.onnx)
    || report.onnx?.opset !== 17
    || report.onnx?.input_size !== 231
    || report.onnx?.output_size !== 14
    || report.validation?.finite !== true
    || report.validation?.onnx_checker_full !== true
    || report.validation?.maximum_onnx_error > 1e-5) {
    throw new Error("Reach deployment export is not qualified for installation");
  }
}

function validateContact(report) {
  const onnx = report.policy?.onnx;
  if (report.protocol !== "hear-frozen-contact-policy-export-v1"
    || report.source?.formal_gate_passed !== true
    || report.source?.held_out_episode_count !== 500
    || report.source?.held_out_success_rate < 0.75
    || report.source?.maximum_active_hand_force_n > 30
    || onnx?.file !== "workyard_contact.onnx"
    || onnx?.bytes !== readFileSync(contact.onnx).byteLength
    || onnx?.sha256 !== sha256(contact.onnx)
    || onnx?.opset !== 17
    || report.policy?.input_size !== 247
    || report.policy?.output_size !== 8
    || report.policy?.gradient_parameter_count !== 0
    || report.harness?.coordination_step !== 0.0075
    || report.harness?.maximum_closing_joint_lead_rad !== 0.25
    || report.harness?.force_release_threshold_n !== 6
    || report.harness?.emergency_force_release_threshold_n !== 12
    || report.validation?.finite !== true
    || report.validation?.onnx_checker_full !== true
    || report.validation?.maximum_onnx_error > 1e-5) {
    throw new Error("Contact deployment export is not qualified for installation");
  }
}

function assertInstalled(destination, source, report, kind) {
  const onnxName = kind === "reach" ? "workyard_reach.onnx" : "workyard_contact.onnx";
  const reportName = kind === "reach"
    ? "reach-policy-report.json"
    : "contact-policy-report.json";
  const installedOnnx = resolve(destination, onnxName);
  const installedReport = resolve(destination, reportName);
  if (sha256(installedOnnx) !== sha256(source.onnx)
    || sha256(installedReport) !== sha256(source.report)) {
    throw new Error(`Installed ${kind} policy assets changed during copy`);
  }
  const installedValue = JSON.parse(readFileSync(installedReport, "utf8"));
  if (JSON.stringify(installedValue) !== JSON.stringify(report)) {
    throw new Error(`Installed ${kind} policy report changed during copy`);
  }
}

function assertWorkspaceChild(path, label) {
  const value = relative(workspace, resolve(path));
  if (!value || value === ".." || value.startsWith("../") || value.startsWith("..\\")) {
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
    ["--reach-root", "reachRoot"],
    ["--contact-root", "contactRoot"],
    ["--destination-root", "destinationRoot"]
  ]);
  for (let index = 0; index < args.length; index += 2) {
    const target = names.get(args[index]);
    const value = args[index + 1];
    if (!target || !value) {
      throw new Error(`Unknown policy install option: ${args[index] ?? ""}`);
    }
    parsed[target] = value;
  }
  return parsed;
}
