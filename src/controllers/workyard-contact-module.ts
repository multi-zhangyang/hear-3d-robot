import { resolve } from "node:path";
import type {
  HumanoidControllerModuleContext
} from "../world/humanoid/controller-module.js";
import {
  HumanoidHandSynergyOverlayController
} from "../world/humanoid/hand-synergy-overlay-controller.js";
import {
  createWorkyardContactHandPolicy
} from "./workyard-contact-hand-policy.js";
import {
  WORKYARD_CONTACT_TARGET_ZONE_ENV,
  G1_GETUP_POLICY_DIRECTORY_ENV,
  WORKYARD_REACH_POLICY_DIRECTORY_ENV,
  createWorkyardReachControllerFromModuleContext,
  humanoidControllerAssets as reachControllerAssets
} from "./workyard-reach-module.js";

export {
  WORKYARD_CONTACT_TARGET_ZONE_ENV,
  G1_GETUP_POLICY_DIRECTORY_ENV,
  WORKYARD_REACH_POLICY_DIRECTORY_ENV
};

export const WORKYARD_CONTACT_POLICY_DIRECTORY_ENV =
  "HEAR_WORKYARD_CONTACT_POLICY_DIRECTORY";
export function humanoidControllerAssets(
  environment: NodeJS.ProcessEnv = process.env
): Array<{ id: string; path: string | URL }> {
  const contactDirectory = environment[WORKYARD_CONTACT_POLICY_DIRECTORY_ENV]
    ?.trim();
  return [
    ...reachControllerAssets(environment),
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
  const reach = await createWorkyardReachControllerFromModuleContext(context);
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
