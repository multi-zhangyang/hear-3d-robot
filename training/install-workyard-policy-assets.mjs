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
const reachRoot = resolve(
  options.reachRoot ?? "artifacts/training/workyard-reach-deployment-v3"
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
const contactPlant = resolve("assets/humanoid/g1/g1_with_hands.xml");

main();

function main() {
  assertWorkspaceChild(destinationRoot, "deployment asset root");
  for (const path of [
    reach.onnx,
    reach.report,
    contact.onnx,
    contact.report,
    contactPlant
  ]) {
    if (!existsSync(path)) throw new Error(`Policy deployment input is missing: ${path}`);
  }
  const reachReport = JSON.parse(readFileSync(reach.report, "utf8"));
  const contactReport = JSON.parse(readFileSync(contact.report, "utf8"));
  validateReach(reachReport);
  validateContact(contactReport);

  mkdirSync(destinationRoot, { recursive: true });
  const transactionRoot = resolve(
    destinationRoot,
    `.workyard-policy-install-${randomUUID()}`
  );
  const stagedReach = resolve(transactionRoot, "next-reach");
  const stagedContact = resolve(transactionRoot, "next-contact");
  const previousReach = resolve(transactionRoot, "previous-reach");
  const previousContact = resolve(transactionRoot, "previous-contact");
  assertWorkspaceChild(transactionRoot, "policy install transaction");
  mkdirSync(stagedReach, { recursive: true });
  mkdirSync(stagedContact, { recursive: true });
  try {
    stagePolicyAssetSet(stagedReach, reach, reachReport, "reach");
    stagePolicyAssetSet(stagedContact, contact, contactReport, "contact");
    replacePolicyAssetPair({
      stagedReach,
      stagedContact,
      previousReach,
      previousContact,
      reachReport,
      contactReport,
      transactionRoot
    });
  } catch (error) {
    if (!error?.preservePolicyInstallTransaction && existsSync(transactionRoot)) {
      assertWorkspaceChild(transactionRoot, "failed policy install transaction");
      rmSync(transactionRoot, { recursive: true, force: true });
    }
    throw error;
  }
  console.log(JSON.stringify({
    protocol: "hear-workyard-policy-assets-install-v2",
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

function stagePolicyAssetSet(destination, source, report, kind) {
  const onnxName = kind === "reach" ? "workyard_reach.onnx" : "workyard_contact.onnx";
  const reportName = kind === "reach"
    ? "reach-policy-report.json"
    : "contact-policy-report.json";
  copyFileSync(source.onnx, resolve(destination, onnxName));
  copyFileSync(source.report, resolve(destination, reportName));
  assertInstalled(destination, source, report, kind);
}

function replacePolicyAssetPair(input) {
  let reachBackedUp = false;
  let contactBackedUp = false;
  let reachInstalled = false;
  let contactInstalled = false;
  try {
    if (existsSync(reachDestination)) {
      renameSync(reachDestination, input.previousReach);
      reachBackedUp = true;
    }
    if (existsSync(contactDestination)) {
      renameSync(contactDestination, input.previousContact);
      contactBackedUp = true;
    }
    renameSync(input.stagedReach, reachDestination);
    reachInstalled = true;
    renameSync(input.stagedContact, contactDestination);
    contactInstalled = true;
    assertInstalled(reachDestination, reach, input.reachReport, "reach");
    assertInstalled(contactDestination, contact, input.contactReport, "contact");
  } catch (error) {
    const rollbackErrors = [];
    for (const [installed, destination, previous, backedUp] of [
      [contactInstalled, contactDestination, input.previousContact, contactBackedUp],
      [reachInstalled, reachDestination, input.previousReach, reachBackedUp]
    ]) {
      try {
        if (installed && existsSync(destination)) {
          assertWorkspaceChild(destination, "new policy asset during rollback");
          rmSync(destination, { recursive: true, force: true });
        }
        if (backedUp && existsSync(previous)) renameSync(previous, destination);
      } catch (rollbackError) {
        rollbackErrors.push(rollbackError);
      }
    }
    if (rollbackErrors.length > 0) {
      const rollbackFailure = new AggregateError(
        [error, ...rollbackErrors],
        `Policy install failed and rollback is incomplete; recovery assets remain at ${input.transactionRoot}`
      );
      rollbackFailure.preservePolicyInstallTransaction = true;
      throw rollbackFailure;
    }
    throw error;
  }
  assertWorkspaceChild(input.transactionRoot, "completed policy install transaction");
  rmSync(input.transactionRoot, { recursive: true, force: true });
}

function validateReach(report) {
  if (report.protocol !== "hear-whole-body-reach-policy-deployment-v3"
    || report.source?.deployment_distribution_covered !== true
    || report.source?.training_deployment_accepted !== false
    || !(report.source?.initial_wrist_error_maximum_m >= 0.35)
    || report.source?.target_protocol
      !== "typescript-pregrasp-geometry-top-wrist-target-v1"
    || report.deployment?.protocol
      !== "hear-typescript-mujoco-reach-deployment-gate-v1"
    || report.deployment?.accepted !== true
    || report.deployment?.runtime !== "typescript-mujoco-onnxruntime-web"
    || report.deployment?.scenario_id !== "humanoid_workyard"
    || report.deployment?.controller_mode !== "learned_policy_only"
    || report.deployment?.target_protocol
      !== "typescript-pregrasp-geometry-top-wrist-target-v1"
    || !(report.deployment?.case_count >= 12)
    || !(report.deployment?.success_rate >= 0.9)
    || !(report.deployment?.initial_wrist_error_maximum_m >= 0.35)
    || !(report.deployment?.terminal_wrist_error_maximum_m <= 0.06)
    || report.deployment?.fall_count !== 0
    || report.deployment?.unauthorized_collision_count !== 0
    || report.deployment?.terminal_assistance_step_count !== 0
    || !(report.deployment?.minimum_support_margin_m >= -0.04)
    || !(report.deployment?.maximum_foot_planar_displacement_m <= 0.08)
    || !(report.deployment?.maximum_foot_slip_speed_m_s <= 0.20)
    || !(report.deployment?.double_support_loss_rate_maximum <= 0.10)
    || !(report.deployment?.no_foot_contact_rate_maximum <= 0.01)
    || report.source?.held_out_environment_count !== 500
    || !(report.source?.held_out_success_rate >= 0.85)
    || report.onnx?.file !== "workyard_reach.onnx"
    || report.onnx?.bytes !== readFileSync(reach.onnx).byteLength
    || report.onnx?.sha256 !== sha256(reach.onnx)
    || report.onnx?.opset !== 17
    || report.onnx?.input_protocol
      !== "hear-workyard-whole-body-reach-observation-v5"
    || report.onnx?.input_size !== 246
    || report.onnx?.output_protocol !== "bounded-whole-body-reach-mean"
    || report.onnx?.output_size !== 29
    || report.validation?.finite !== true
    || report.validation?.onnx_checker_full !== true
    || !(report.validation?.maximum_onnx_error <= 1e-5)) {
    throw new Error("Reach deployment export is not qualified for installation");
  }
}

function validateContact(report) {
  const onnx = report.policy?.onnx;
  const reachReport = JSON.parse(readFileSync(reach.report, "utf8"));
  const frozenReach = report.source?.frozen_reach;
  if (report.protocol !== "hear-frozen-contact-policy-export-v2"
    || report.source?.formal_gate_passed !== true
    || report.source?.held_out_episode_count !== 500
    || report.source?.held_out_success_rate < 0.75
    || report.source?.maximum_active_hand_force_n > 30
    || frozenReach?.protocol !== "hear-frozen-contact-reach-binding-v1"
    || frozenReach?.runtime_protocol
      !== "hear-frozen-qualified-whole-body-reach-runtime-v2"
    || frozenReach?.source_checkpoint_sha256
      !== reachReport.source?.checkpoint_sha256
    || frozenReach?.jit_sha256 !== reachReport.policy?.sha256
    || frozenReach?.report_sha256 !== sha256(reach.report)
    || report.plant?.protocol !== "hear-workyard-contact-deployment-plant-v1"
    || report.plant?.g1_xml?.file !== "g1_with_hands.xml"
    || report.plant?.g1_xml?.bytes !== readFileSync(contactPlant).byteLength
    || report.plant?.g1_xml?.sha256 !== sha256(contactPlant)
    || report.plant?.g1_xml?.hand_contact_collision_count !== 14
    || report.plant?.g1_xml?.hand_contact_priority !== 2
    || report.plant?.g1_xml?.hand_contact_solref_time_constant_s !== 0.04
    || report.plant?.g1_xml?.hand_contact_solref_damping_ratio !== 1
    || onnx?.file !== "workyard_contact.onnx"
    || onnx?.bytes !== readFileSync(contact.onnx).byteLength
    || onnx?.sha256 !== sha256(contact.onnx)
    || onnx?.opset !== 17
    || report.policy?.input_protocol !== "hear-workyard-contact-observation-v2"
    || report.policy?.input_size !== 262
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
