import { resolve } from "node:path";
import type {
  HumanoidControllerModuleAsset,
  HumanoidControllerModuleContext
} from "../world/humanoid/controller-module.js";
import type {
  HumanoidWholeBodyController
} from "../world/humanoid/whole-body-controller.js";
import {
  G1_GETUP_POLICY_DIRECTORY_ENV,
  attachG1RecoveryExpert,
  g1RecoveryControllerAssets
} from "./g1-recovery-module.js";
import {
  createMjlabG1VelocityController,
  parseMjlabG1VelocityTrainingBundle
} from "./mjlab-g1-velocity-controller.js";
import { createWorkyardReachController } from "./workyard-reach-controller.js";

export const WORKYARD_CONTACT_TARGET_ZONE_ENV =
  "HEAR_WORKYARD_CONTACT_TARGET_ZONE_ID";
export const WORKYARD_REACH_POLICY_DIRECTORY_ENV =
  "HEAR_WORKYARD_REACH_POLICY_DIRECTORY";
export { G1_GETUP_POLICY_DIRECTORY_ENV };
const BODY_POLICY_DIRECTORY_ENV = "HEAR_MJLAB_G1_POLICY_DIRECTORY";

export function humanoidControllerAssets(
  environment: NodeJS.ProcessEnv = process.env
): Array<{ id: string; path: string | URL }> {
  const bodyDirectory = environment[BODY_POLICY_DIRECTORY_ENV]?.trim();
  const reachDirectory = environment[WORKYARD_REACH_POLICY_DIRECTORY_ENV]
    ?.trim();
  return [
    {
      id: "body_policy",
      path: bodyDirectory
        ? resolve(bodyDirectory, "g1_velocity.onnx")
        : new URL(
            "../../assets/humanoid/controllers/mjlab-g1-velocity/g1_velocity.onnx",
            import.meta.url
          )
    },
    {
      id: "body_training_report",
      path: bodyDirectory
        ? resolve(bodyDirectory, "training-report.json")
        : new URL(
            "../../assets/humanoid/controllers/mjlab-g1-velocity/training-report.json",
            import.meta.url
          )
    },
    {
      id: "reach_policy",
      path: reachDirectory
        ? resolve(reachDirectory, "workyard_reach.onnx")
        : new URL(
            "../../assets/humanoid/controllers/workyard-reach/workyard_reach.onnx",
            import.meta.url
          )
    },
    {
      id: "reach_policy_report",
      path: reachDirectory
        ? resolve(reachDirectory, "reach-policy-report.json")
        : new URL(
            "../../assets/humanoid/controllers/workyard-reach/reach-policy-report.json",
            import.meta.url
          )
    },
    ...g1RecoveryControllerAssets(environment)
  ];
}

export function createHumanoidWholeBodyController(
  context: HumanoidControllerModuleContext
): Promise<HumanoidWholeBodyController> {
  return createWorkyardReachControllerFromModuleContext(context);
}

export async function createWorkyardReachControllerFromModuleContext(
  context: HumanoidControllerModuleContext
): Promise<HumanoidWholeBodyController> {
  const bodyPolicyAsset = remapAsset(context, "body_policy", "policy");
  const bodyReportAsset = remapAsset(
    context,
    "body_training_report",
    "training_report"
  );
  const bodyContext: HumanoidControllerModuleContext = Object.freeze({
    protocol: "hear-humanoid-controller-module-v1",
    sourceSha256: context.sourceSha256,
    assets: Object.freeze([bodyPolicyAsset, bodyReportAsset])
  });
  const bodyMetadata = parseMjlabG1VelocityTrainingBundle(
    bodyPolicyAsset,
    bodyReportAsset
  );
  const body = await createMjlabG1VelocityController(bodyContext);
  let reach: Awaited<ReturnType<typeof createWorkyardReachController>>;
  try {
    reach = await createWorkyardReachController({
      assets: context.assets,
      body,
      bodyPolicy: bodyMetadata,
      targetZoneId: process.env[WORKYARD_CONTACT_TARGET_ZONE_ENV]?.trim()
        || "assembly_bay"
    });
  } catch (error) {
    await body.dispose();
    throw error;
  }
  return attachG1RecoveryExpert(reach, context);
}

function remapAsset(
  context: HumanoidControllerModuleContext,
  sourceId: string,
  targetId: string
): HumanoidControllerModuleAsset {
  if (context.protocol !== "hear-humanoid-controller-module-v1") {
    throw new Error("Invalid Workyard reach controller module context");
  }
  const source = context.assets.find(({ id }) => id === sourceId);
  if (!source) {
    throw new Error(`Workyard reach controller asset is missing: ${sourceId}`);
  }
  return Object.freeze({
    id: targetId,
    sha256: source.sha256,
    bytes: source.bytes.slice()
  });
}
