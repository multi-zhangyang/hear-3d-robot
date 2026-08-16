import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type {
  HumanoidControllerModuleContext
} from "../world/humanoid/controller-module.js";
import type {
  HumanoidWholeBodyController
} from "../world/humanoid/whole-body-controller.js";
import {
  G1_GETUP_POLICY_ASSET_ID,
  G1_GETUP_REPORT_ASSET_ID,
  createG1GetupController
} from "./g1-getup-controller.js";
import { G1RecoveryGatedController } from "./g1-recovery-gated-controller.js";

export const G1_GETUP_POLICY_DIRECTORY_ENV =
  "HEAR_G1_GETUP_POLICY_DIRECTORY";
export const G1_GETUP_QUALIFICATION_MODE_ENV =
  "HEAR_G1_GETUP_QUALIFICATION_MODE";
export const G1_GETUP_QUALIFICATION_ASSET_ID =
  "getup_runtime_qualification";

interface ControllerAssetDeclaration {
  readonly id: string;
  readonly path: string | URL;
  readonly sourceIdentity?: boolean;
}

/**
 * Declares the recovery expert as part of a controller module's immutable
 * source. A trained policy is optional before installation, but a configured
 * directory is authoritative and therefore fails during module loading if its
 * deployment bundle is incomplete.
 */
export function g1RecoveryControllerAssets(
  environment: NodeJS.ProcessEnv = process.env
): ControllerAssetDeclaration[] {
  const configuredDirectory = environment[G1_GETUP_POLICY_DIRECTORY_ENV]?.trim();
  const bundledPolicy = new URL(
    "../../assets/humanoid/controllers/g1-getup/g1_getup.onnx",
    import.meta.url
  );
  const bundledReport = new URL(
    "../../assets/humanoid/controllers/g1-getup/getup-policy-report.json",
    import.meta.url
  );
  const bundledQualification = new URL(
    "../../assets/humanoid/controllers/g1-getup/runtime-deployment-report.json",
    import.meta.url
  );
  const qualificationMode = Boolean(configuredDirectory)
    && environment[G1_GETUP_QUALIFICATION_MODE_ENV] === "1";
  const declarations: ControllerAssetDeclaration[] = [
    {
      id: G1_GETUP_POLICY_ASSET_ID,
      path: configuredDirectory
        ? resolve(configuredDirectory, "g1_getup.onnx")
        : bundledPolicy
    },
    {
      id: G1_GETUP_REPORT_ASSET_ID,
      path: configuredDirectory
        ? resolve(configuredDirectory, "getup-policy-report.json")
        : bundledReport
    },
    {
      id: G1_GETUP_QUALIFICATION_ASSET_ID,
      sourceIdentity: false,
      path: configuredDirectory
        ? resolve(configuredDirectory, "runtime-deployment-report.json")
        : bundledQualification
    }
  ];
  if (qualificationMode) return declarations.slice(0, 2);
  if (configuredDirectory) return declarations;
  const existing = [bundledPolicy, bundledReport, bundledQualification]
    .map((path) => existsSync(fileURLToPath(path)));
  if (!existing.some(Boolean)) return [];
  // Once any bundled recovery file exists, the directory is an asserted
  // deployment bundle. Returning every declaration makes partial installs
  // fail closed in the controller loader instead of silently losing recovery.
  return declarations;
}

/**
 * Adds the low-level recovery expert to any body controller while preserving
 * one owner for disposal and controller state. The semantic Harness remains
 * responsible for authorizing stabilize.recover_support; this function only
 * makes that authorized Skill physically executable.
 */
export async function attachG1RecoveryExpert(
  body: HumanoidWholeBodyController,
  context: HumanoidControllerModuleContext
): Promise<HumanoidWholeBodyController> {
  const hasPolicy = context.assets.some(
    ({ id }) => id === G1_GETUP_POLICY_ASSET_ID
  );
  const hasReport = context.assets.some(
    ({ id }) => id === G1_GETUP_REPORT_ASSET_ID
  );
  const hasQualification = context.assets.some(
    ({ id }) => id === G1_GETUP_QUALIFICATION_ASSET_ID
  );
  if (!hasPolicy && !hasReport && !hasQualification) return body;
  const qualificationMode = process.env[G1_GETUP_QUALIFICATION_MODE_ENV] === "1";
  if (!hasPolicy || !hasReport
    || (!hasQualification && !qualificationMode)) {
    await body.dispose();
    throw new Error("G1 recovery expert deployment bundle is incomplete");
  }
  let recovery: HumanoidWholeBodyController | undefined;
  try {
    if (hasQualification) assertRuntimeQualification(context);
    recovery = await createG1GetupController({ assets: context.assets });
    return new G1RecoveryGatedController(body, recovery);
  } catch (error) {
    await Promise.allSettled([
      body.dispose(),
      ...(recovery ? [recovery.dispose()] : [])
    ]);
    throw error;
  }
}

function assertRuntimeQualification(
  context: HumanoidControllerModuleContext
): void {
  const assets = context.assets;
  const policy = assets.find(({ id }) => id === G1_GETUP_POLICY_ASSET_ID);
  const deployment = assets.find(({ id }) => id === G1_GETUP_REPORT_ASSET_ID);
  const qualification = assets.find(
    ({ id }) => id === G1_GETUP_QUALIFICATION_ASSET_ID
  );
  if (!policy || !deployment || !qualification) {
    throw new Error("G1 recovery expert deployment bundle is incomplete");
  }
  let report: unknown;
  try {
    report = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(
      qualification.bytes
    ));
  } catch (error) {
    throw new Error("G1 runtime qualification is not valid UTF-8 JSON", {
      cause: error
    });
  }
  if (!isRecord(report)
    || report.protocol !== "hear-typescript-mujoco-g1-getup-deployment-gate-v1"
    || report.accepted !== true
    || report.controller_source_sha256 !== context.sourceSha256
    || report.policy_sha256 !== policy.sha256
    || report.deployment_report_sha256 !== deployment.sha256
    || !isRecord(report.summary)
    || report.summary.recovered_count !== 4) {
    throw new Error("G1 recovery expert lacks a matching runtime qualification");
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
