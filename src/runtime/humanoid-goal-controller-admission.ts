import type { HumanoidLearnedPolicyCapability } from
  "../domain/humanoid-policy.js";
import type { Goal } from "../domain/schema.js";
import type { HumanoidControllerDescriptor } from
  "../world/humanoid/whole-body-controller.js";

type HumanoidGoalPredicateType = Goal["predicates"][number]["type"];

const EMBODIMENT_GOAL_PREDICATES = [
  "robot_at",
  "robot_in_zone",
  "end_effector_at"
] as const satisfies readonly HumanoidGoalPredicateType[];

const MANIPULABLE_OBJECT_GOAL_PREDICATES = [
  "object_grasped",
  "object_at",
  "object_in_zone",
  "object_placed",
  "object_inside",
  "object_on"
] as const satisfies readonly HumanoidGoalPredicateType[];

const ARTICULATED_OBJECT_GOAL_PREDICATES = [
  "articulation_state"
] as const satisfies readonly HumanoidGoalPredicateType[];

const STATIC_SOLID_GOAL_PREDICATES = [
  "block_removed"
] as const satisfies readonly HumanoidGoalPredicateType[];

const CONTACT_GOAL_PREDICATES = new Set<HumanoidGoalPredicateType>([
  ...MANIPULABLE_OBJECT_GOAL_PREDICATES,
  ...ARTICULATED_OBJECT_GOAL_PREDICATES,
  ...STATIC_SOLID_GOAL_PREDICATES
]);

export function humanoidGoalControllerCapabilitySurface(
  capabilities: readonly HumanoidLearnedPolicyCapability[] | undefined
) {
  const contactRichManipulation = capabilities?.includes(
    "contact_rich_manipulation"
  ) ?? false;
  return {
    embodiment_predicates: [...EMBODIMENT_GOAL_PREDICATES],
    manipulable_object_predicates: contactRichManipulation
      ? [...MANIPULABLE_OBJECT_GOAL_PREDICATES]
      : [],
    articulated_object_predicates: contactRichManipulation
      ? [...ARTICULATED_OBJECT_GOAL_PREDICATES]
      : [],
    static_solid_predicates: contactRichManipulation
      ? [...STATIC_SOLID_GOAL_PREDICATES]
      : []
  };
}

/**
 * Rejects a permanently impossible controller/Goal pairing before the first
 * model episode. Per-Skill admission remains authoritative for every concrete
 * action; this only covers capabilities that no reference-control route may
 * claim on behalf of an untrained contact policy.
 */
export function assertHumanoidGoalControllerAdmission(
  goal: Goal,
  controller: HumanoidControllerDescriptor
): void {
  const required = requiredPermanentCapabilities(goal);
  if (required.length === 0) return;
  const available = new Set(controller.learnedPolicy?.capabilities ?? []);
  const missing = required.filter((capability) => !available.has(capability));
  if (missing.length === 0) return;
  throw new Error(
    `Humanoid Goal requires trained controller capabilities that are not installed: `
      + `${missing.join(", ")}. Controller ${controller.implementation} declares: `
      + `${[...available].join(", ") || "none"}`
  );
}

function requiredPermanentCapabilities(
  goal: Goal
): HumanoidLearnedPolicyCapability[] {
  return goal.predicates.some((predicate) => (
    CONTACT_GOAL_PREDICATES.has(predicate.type)
  ))
    ? ["contact_rich_manipulation"]
    : [];
}
