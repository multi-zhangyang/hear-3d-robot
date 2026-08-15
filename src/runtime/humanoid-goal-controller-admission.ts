import type { HumanoidLearnedPolicyCapability } from
  "../domain/humanoid-policy.js";
import type { Goal } from "../domain/schema.js";
import type { HumanoidControllerDescriptor } from
  "../world/humanoid/whole-body-controller.js";

const CONTACT_GOAL_PREDICATES = new Set<Goal["predicates"][number]["type"]>([
  "block_removed",
  "object_in_zone",
  "object_placed",
  "object_at",
  "object_grasped",
  "object_inside",
  "object_on",
  "articulation_state"
]);

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
