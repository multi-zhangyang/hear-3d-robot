import { YahmpController } from "../world/humanoid/yahmp-controller.js";

export function createHumanoidWholeBodyController(): Promise<YahmpController> {
  return YahmpController.create();
}
