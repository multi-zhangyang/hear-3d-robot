import { YahmpController } from "../world/humanoid/yahmp-controller.js";
import type {
  HumanoidControllerModuleContext
} from "../world/humanoid/controller-module.js";
import type {
  HumanoidWholeBodyController
} from "../world/humanoid/whole-body-controller.js";
import {
  attachG1RecoveryExpert,
  g1RecoveryControllerAssets
} from "./g1-recovery-module.js";

const YAHMP_POLICY_ASSET_ID = "yahmp_policy";

export function humanoidControllerAssets(
  environment: NodeJS.ProcessEnv = process.env
): Array<{ id: string; path: string | URL }> {
  return [
    {
      id: YAHMP_POLICY_ASSET_ID,
      path: new URL(
        "../../assets/humanoid/controllers/g1_yahmp.onnx",
        import.meta.url
      )
    },
    ...g1RecoveryControllerAssets(environment)
  ];
}

export async function createHumanoidWholeBodyController(
  context: HumanoidControllerModuleContext
): Promise<HumanoidWholeBodyController> {
  const policy = context.assets.find(({ id }) => id === YAHMP_POLICY_ASSET_ID);
  if (!policy) throw new Error("YAHMP controller policy asset is missing");
  const body = await YahmpController.create(policy.bytes);
  return attachG1RecoveryExpert(body, context);
}
