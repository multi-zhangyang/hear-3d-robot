import { z } from "zod";

const HUMANOID_RUN_MODES = ["mission", "continuous"] as const;

export const HumanoidRunModeSchema = z.enum(HUMANOID_RUN_MODES);

export type HumanoidRunMode = z.infer<typeof HumanoidRunModeSchema>;

export function humanoidRunShouldFinish(input: {
  mode: HumanoidRunMode;
  activeGoalCompleted: boolean;
  missionGoalCompleted: boolean;
}): boolean {
  return input.mode === "mission"
    && input.activeGoalCompleted
    && input.missionGoalCompleted;
}
