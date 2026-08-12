import { resolve } from "node:path";
import type {
  HumanoidControllerModuleAsset,
  HumanoidControllerModuleContext
} from "../world/humanoid/controller-module.js";
import {
  HumanoidHandSynergyOverlayController
} from "../world/humanoid/hand-synergy-overlay-controller.js";
import {
  createMjlabG1VelocityController,
  parseMjlabG1VelocityTrainingBundle
} from "./mjlab-g1-velocity-controller.js";
import {
  createWorkyardContactHandPolicy
} from "./workyard-contact-hand-policy.js";
import {
  createWorkyardReachController
} from "./workyard-reach-controller.js";

export const WORKYARD_CONTACT_POLICY_DIRECTORY_ENV =
  "HEAR_WORKYARD_CONTACT_POLICY_DIRECTORY";
export const WORKYARD_CONTACT_TARGET_ZONE_ENV =
  "HEAR_WORKYARD_CONTACT_TARGET_ZONE_ID";
export const WORKYARD_REACH_POLICY_DIRECTORY_ENV =
  "HEAR_WORKYARD_REACH_POLICY_DIRECTORY";
const BODY_POLICY_DIRECTORY_ENV = "HEAR_MJLAB_G1_POLICY_DIRECTORY";

export function humanoidControllerAssets(
  environment: NodeJS.ProcessEnv = process.env
): Array<{ id: string; path: string | URL }> {
  const bodyDirectory = environment[BODY_POLICY_DIRECTORY_ENV]?.trim();
  const contactDirectory = environment[WORKYARD_CONTACT_POLICY_DIRECTORY_ENV]
    ?.trim();
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
    {
      id: "contact_policy",
      path: contactDirectory
        ? resolve(contactDirectory, "workyard_contact.onnx")
        : new URL(
            "../../assets/humanoid/controllers/workyard-contact/workyard_contact.onnx",
            import.meta.url
          )
    },
    {
      id: "contact_policy_report",
      path: contactDirectory
        ? resolve(contactDirectory, "contact-policy-report.json")
        : new URL(
            "../../assets/humanoid/controllers/workyard-contact/contact-policy-report.json",
            import.meta.url
          )
    },
    {
      id: "contact_plant",
      path: new URL(
        "../../assets/humanoid/g1/g1_with_hands.xml",
        import.meta.url
      )
    }
  ];
}

export async function createHumanoidWholeBodyController(
  context: HumanoidControllerModuleContext
): Promise<HumanoidHandSynergyOverlayController> {
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
  let reach;
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
  try {
    const hand = await createWorkyardContactHandPolicy({
      assets: context.assets
    });
    try {
      return new HumanoidHandSynergyOverlayController(reach, hand);
    } catch (error) {
      await hand.dispose();
      throw error;
    }
  } catch (error) {
    await reach.dispose();
    throw error;
  }
}

function remapAsset(
  context: HumanoidControllerModuleContext,
  sourceId: string,
  targetId: string
): HumanoidControllerModuleAsset {
  if (context.protocol !== "hear-humanoid-controller-module-v1") {
    throw new Error("Invalid Workyard contact controller module context");
  }
  const source = context.assets.find(({ id }) => id === sourceId);
  if (!source) {
    throw new Error(`Workyard contact controller asset is missing: ${sourceId}`);
  }
  return Object.freeze({
    id: targetId,
    sha256: source.sha256,
    bytes: source.bytes.slice()
  });
}
