import type { Vec3 } from "../../domain/schema.js";
import type {
  HumanoidWorld,
  NavigationPlanReceipt
} from "../../world/humanoid/world.js";
import type { ActiveHumanoidSkillBinding } from "./skill-binding.js";
import { humanoidEmbodiedSkillIdentity } from "./skill-binding.js";
import type { AutonomousHumanoidSkillPlan } from
  "./autonomous-skill-planner.js";

export interface AutonomousHumanoidNavigationAttempt {
  target: Vec3;
  accepted_position_tolerance_m: number | null;
  score: number;
  accepted: boolean;
  reason: string | null;
  blocking_contacts?: NonNullable<NavigationPlanReceipt["blockingContacts"]>;
}

export interface AutonomousHumanoidNavigationPlanningResult {
  selected: NavigationPlanReceipt | null;
  attempts: AutonomousHumanoidNavigationAttempt[];
}

/**
 * Deterministically validates the semantic planner's ranked navigation
 * candidates. This is shared by the first Agent-requested plan and every
 * continuation inside the same autonomous Skill horizon.
 */
export async function planAutonomousHumanoidNavigation(input: {
  world: HumanoidWorld;
  binding: ActiveHumanoidSkillBinding;
  plan: Extract<AutonomousHumanoidSkillPlan, { kind: "navigation" }>;
}): Promise<AutonomousHumanoidNavigationPlanningResult> {
  const attempts: AutonomousHumanoidNavigationAttempt[] = [];
  let selected: NavigationPlanReceipt | null = null;
  for (const candidate of input.plan.targets.slice(0, 8)) {
    const result = await input.world.planNavigation(
      candidate.target,
      candidate.arrivalHeading,
      candidate.acceptedPositionToleranceMeters ?? null,
      { skillCallIdentity: humanoidEmbodiedSkillIdentity(input.binding) }
    );
    attempts.push({
      target: { ...candidate.target },
      accepted_position_tolerance_m:
        candidate.acceptedPositionToleranceMeters ?? null,
      score: candidate.score,
      accepted: result.accepted,
      reason: result.reason ?? null,
      ...(result.blockingContacts
        ? { blocking_contacts: structuredClone(result.blockingContacts) }
        : {})
    });
    if (result.accepted) {
      selected = result;
      break;
    }
  }
  return { selected, attempts };
}
