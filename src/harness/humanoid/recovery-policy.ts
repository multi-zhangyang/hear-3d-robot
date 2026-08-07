import { z } from "zod";
import {
  HUMANOID_SKILL_FAILURE_CODES,
  HUMANOID_SKILL_IDS,
  HUMANOID_SKILL_CONTRACTS,
  type HumanoidSkillFailureCode,
  type HumanoidSkillId,
  type HumanoidSkillInvocation
} from "../../domain/humanoid-skill.js";
import type { ActiveHumanoidSkillBinding } from "./skill-binding.js";

export interface HumanoidRecoveryPolicyState {
  protocol: "humanoid-recovery-policy-v1";
  source_execution_transaction_id: string;
  source_planning_transaction_id: string;
  source_skill_transaction_id: string;
  source_skill: HumanoidSkillId;
  source_phase: string;
  physical_failure_code: string;
  failure_reason: HumanoidSkillFailureCode | null;
  world_revision: number;
  candidate_skills: HumanoidSkillId[];
  requires_model_selection: true;
  automatic_actuation: false;
}

export const HumanoidRecoveryPolicyStateSchema: z.ZodType<HumanoidRecoveryPolicyState> =
  z.object({
    protocol: z.literal("humanoid-recovery-policy-v1"),
    source_execution_transaction_id: z.string().trim().min(1),
    source_planning_transaction_id: z.string().trim().min(1),
    source_skill_transaction_id: z.string().trim().min(1),
    source_skill: z.enum(HUMANOID_SKILL_IDS),
    source_phase: z.string().trim().min(1),
    physical_failure_code: z.string().trim().min(1),
    failure_reason: z.enum(HUMANOID_SKILL_FAILURE_CODES).nullable(),
    world_revision: z.number().int().nonnegative(),
    candidate_skills: z.array(z.enum(HUMANOID_SKILL_IDS)).min(1),
    requires_model_selection: z.literal(true),
    automatic_actuation: z.literal(false)
  }).strict();

export function createHumanoidRecoveryPolicy(input: {
  executionTransactionId: string;
  planningTransactionId: string;
  physicalFailureCode: string;
  worldRevision: number;
  binding: ActiveHumanoidSkillBinding;
}): HumanoidRecoveryPolicyState {
  const contract = HUMANOID_SKILL_CONTRACTS[input.binding.invocation.skill];
  return {
    protocol: "humanoid-recovery-policy-v1",
    source_execution_transaction_id: input.executionTransactionId,
    source_planning_transaction_id: input.planningTransactionId,
    source_skill_transaction_id: input.binding.transaction_id,
    source_skill: input.binding.invocation.skill,
    source_phase: input.binding.phase,
    physical_failure_code: input.physicalFailureCode,
    failure_reason: physicalFailureReason(input.physicalFailureCode),
    world_revision: input.worldRevision,
    candidate_skills: [...contract.recovery_entry],
    requires_model_selection: true,
    automatic_actuation: false
  };
}

export function humanoidRecoverySelectionAccepted(
  policy: HumanoidRecoveryPolicyState,
  invocation: HumanoidSkillInvocation
): boolean {
  return policy.candidate_skills.includes(invocation.skill);
}

function physicalFailureReason(code: string): HumanoidSkillFailureCode | null {
  const normalized = code.toLowerCase();
  if (normalized.includes("slip")) return "object_slipped";
  if (normalized.includes("grasp")) return "grasp_unstable";
  if (normalized.includes("contact")) return "contact_missing";
  if (normalized.includes("articulation") || normalized.includes("joint_stall")) {
    return "articulation_stalled";
  }
  if (normalized.includes("place") || normalized.includes("settle")) {
    return "placement_misaligned";
  }
  if (normalized.includes("route") || normalized.includes("path")
    || normalized.includes("navigation")) return "path_blocked";
  if (normalized.includes("fall") || normalized.includes("balance")) {
    return "balance_lost";
  }
  if (normalized.includes("collision") || normalized.includes("unauthorized_contact")) {
    return "collision_risk";
  }
  if (normalized.includes("stale") || normalized.includes("drift")
    || normalized.includes("world_revision")) return "unexpected_world_change";
  if (normalized.includes("unreachable") || normalized.includes("ik")
    || normalized.includes("invalid_reference")) return "unreachable";
  return null;
}
