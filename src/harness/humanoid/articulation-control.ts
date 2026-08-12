import { z } from "zod";
import type { HumanoidSkillInvocation } from "../../domain/humanoid-skill.js";
import type {
  HumanoidObjectWorldModelEntry
} from "../../world/humanoid/object-world-model.js";

type Articulation = NonNullable<HumanoidObjectWorldModelEntry["articulation"]>;
type ArticulationInvocation = Extract<HumanoidSkillInvocation, {
  skill: "open" | "close" | "turn";
}>;

export const HUMANOID_ARTICULATION_HORIZON = Object.freeze({
  maximum_task_path_m: 0.14,
  minimum_segment_progress_ratio: 0.65,
  minimum_hinge_progress_rad: 0.025,
  minimum_slide_progress_m: 0.006,
  maximum_segments: 32,
  // One immutable semantic window spans all deterministic 50 Hz motion
  // segments. Individual articulation candidates are capped below 8 seconds.
  maximum_control_steps: 32 * 400
});

const HumanoidArticulationGoalSchema = z.object({
  joint_id: z.string().trim().min(1),
  origin_position: z.number().finite(),
  target_position: z.number().finite(),
  direction: z.enum(["increasing", "decreasing"])
}).strict();

export type HumanoidArticulationGoal = z.infer<
  typeof HumanoidArticulationGoalSchema
>;

export function humanoidArticulationGoal(input: {
  invocation: ArticulationInvocation;
  articulation: Articulation;
  targetPosition?: number;
}): HumanoidArticulationGoal {
  const current = input.articulation.position;
  if (current === null) {
    throw new Error("Articulation control requires an observed joint position");
  }
  if (input.articulation.joint_id !== input.invocation.joint_id) {
    throw new Error("Articulation control joint does not match the Skill invocation");
  }
  const target = input.targetPosition ?? targetPosition(
    input.invocation,
    input.articulation,
    current
  );
  if (target < input.articulation.range.minimum - 1e-9
    || target > input.articulation.range.maximum + 1e-9) {
    throw new Error("Articulation control target is outside the observed joint range");
  }
  if (Math.abs(target - current) <= 1e-9) {
    throw new Error("Articulation control target is already satisfied");
  }
  const direction = target > current ? "increasing" : "decreasing";
  if (input.invocation.skill === "turn"
    && direction !== input.invocation.direction) {
    throw new Error("Articulation continuation reversed the model-selected direction");
  }
  return HumanoidArticulationGoalSchema.parse({
    joint_id: input.articulation.joint_id,
    origin_position: input.targetPosition === undefined ? current : targetOrigin(
      input.invocation,
      target
    ),
    target_position: target,
    direction
  });
}

export function humanoidArticulationGoalSatisfied(
  goal: HumanoidArticulationGoal,
  articulation: Articulation | null | undefined
): boolean {
  if (!articulation || articulation.position === null
    || articulation.joint_id !== goal.joint_id) return false;
  const tolerance = 1e-4;
  return goal.direction === "increasing"
    ? articulation.position >= goal.target_position - tolerance
    : articulation.position <= goal.target_position + tolerance;
}

export function humanoidArticulationSegmentBudgetExhausted(
  completedSegments: number,
  currentProcessSegments = 0
): boolean {
  if (!Number.isSafeInteger(completedSegments) || completedSegments < 0
    || !Number.isSafeInteger(currentProcessSegments)
    || currentProcessSegments < 0) {
    throw new Error("Articulation segment progress must be a non-negative integer");
  }
  return completedSegments + currentProcessSegments
    >= HUMANOID_ARTICULATION_HORIZON.maximum_segments;
}

export function humanoidArticulationSegmentMinimumDelta(input: {
  articulation: Articulation;
  segmentDelta: number;
}): number {
  const magnitude = Math.abs(input.segmentDelta);
  const physicalFloor = input.articulation.type === "hinge"
    ? HUMANOID_ARTICULATION_HORIZON.minimum_hinge_progress_rad
    : HUMANOID_ARTICULATION_HORIZON.minimum_slide_progress_m;
  return Math.min(
    magnitude,
    Math.max(
      physicalFloor,
      magnitude * HUMANOID_ARTICULATION_HORIZON.minimum_segment_progress_ratio
    )
  );
}

function targetPosition(
  invocation: ArticulationInvocation,
  articulation: Articulation,
  current: number
): number {
  if (invocation.skill === "open") {
    return articulation.closed_position
      + (articulation.open_position - articulation.closed_position)
        * invocation.minimum_open_fraction;
  }
  if (invocation.skill === "close") {
    return articulation.closed_position
      + (articulation.open_position - articulation.closed_position)
        * invocation.maximum_open_fraction;
  }
  return current + (invocation.direction === "increasing" ? 1 : -1)
    * invocation.rotation_radians;
}

function targetOrigin(
  invocation: ArticulationInvocation,
  target: number
): number {
  if (invocation.skill !== "turn") return target;
  return target - (invocation.direction === "increasing" ? 1 : -1)
    * invocation.rotation_radians;
}
