import { resolve } from "node:path";
import {
  createMjlabG1VelocityController
} from "./mjlab-g1-velocity-controller.js";
import type {
  HumanoidControllerModuleContext
} from "../world/humanoid/controller-module.js";

export const MJLAB_G1_POLICY_DIRECTORY_ENV =
  "HEAR_MJLAB_G1_POLICY_DIRECTORY";

export function humanoidControllerAssets(
  environment: NodeJS.ProcessEnv = process.env
): Array<{
  id: string;
  path: string | URL;
}> {
  const directory = environment[MJLAB_G1_POLICY_DIRECTORY_ENV]?.trim();
  const policy = directory
    ? resolve(directory, "g1_velocity.onnx")
    : new URL(
        "../../assets/humanoid/controllers/mjlab-g1-velocity/g1_velocity.onnx",
        import.meta.url
      );
  const report = directory
    ? resolve(directory, "training-report.json")
    : new URL(
        "../../assets/humanoid/controllers/mjlab-g1-velocity/training-report.json",
        import.meta.url
      );
  return [
    { id: "policy", path: policy },
    { id: "training_report", path: report }
  ];
}

export function createHumanoidWholeBodyController(
  context: HumanoidControllerModuleContext
) {
  return createMjlabG1VelocityController(context);
}
