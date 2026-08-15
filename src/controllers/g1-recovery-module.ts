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

interface ControllerAssetDeclaration {
  readonly id: string;
  readonly path: string | URL;
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
  const available = Boolean(configuredDirectory)
    || (existsSync(fileURLToPath(bundledPolicy))
      && existsSync(fileURLToPath(bundledReport)));
  if (!available) return [];
  return [
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
    }
  ];
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
  if (!hasPolicy && !hasReport) return body;
  if (!hasPolicy || !hasReport) {
    await body.dispose();
    throw new Error("G1 recovery expert deployment bundle is incomplete");
  }
  let recovery: HumanoidWholeBodyController | undefined;
  try {
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
