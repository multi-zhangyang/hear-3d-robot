import { z } from "zod";
import {
  HUMANOID_SKILL_FAILURE_CODES,
  HUMANOID_SKILL_IDS,
  HUMANOID_SKILL_CONTRACTS,
  HumanoidSkillInvocationSchema,
  type HumanoidSkillFailureCode,
  type HumanoidSkillId,
  type HumanoidSkillInvocation
} from "../../domain/humanoid-skill.js";
import type { ActiveHumanoidSkillBinding } from "./skill-binding.js";
import { modelPayloadSha256 } from "../../domain/model-call-authority.js";

export interface HumanoidRecoveryPolicyState {
  protocol: "humanoid-recovery-policy-v1";
  source_execution_transaction_id: string;
  source_planning_transaction_id: string;
  source_skill_transaction_id: string;
  source_skill: HumanoidSkillId;
  source_invocation?: HumanoidSkillInvocation | undefined;
  source_invocation_sha256?: string | undefined;
  source_phase: string;
  physical_failure_code: string;
  failure_reason: HumanoidSkillFailureCode | null;
  world_revision: number;
  candidate_skills: HumanoidSkillId[];
  adaptation_requirements?: string[] | undefined;
  excluded_interaction_point_ids?: string[] | undefined;
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
    source_invocation: HumanoidSkillInvocationSchema.optional(),
    source_invocation_sha256: z.string().regex(/^[a-f0-9]{64}$/).optional(),
    source_phase: z.string().trim().min(1),
    physical_failure_code: z.string().trim().min(1),
    failure_reason: z.enum(HUMANOID_SKILL_FAILURE_CODES).nullable(),
    world_revision: z.number().int().nonnegative(),
    candidate_skills: z.array(z.enum(HUMANOID_SKILL_IDS)).min(1),
    adaptation_requirements: z.array(z.string().trim().min(1)).min(1).optional(),
    excluded_interaction_point_ids: z.array(z.string().trim().min(1)).optional(),
    requires_model_selection: z.literal(true),
    automatic_actuation: z.literal(false)
  }).strict().superRefine((policy, context) => {
    if ((policy.source_invocation === undefined)
      !== (policy.source_invocation_sha256 === undefined)) {
      context.addIssue({
        code: "custom",
        path: [policy.source_invocation === undefined
          ? "source_invocation"
          : "source_invocation_sha256"],
        message: "Recovery source invocation and identity must be provided together"
      });
    }
    if (policy.source_invocation
      && policy.source_invocation_sha256 !== modelPayloadSha256(
        policy.source_invocation
      )) {
      context.addIssue({
        code: "custom",
        path: ["source_invocation_sha256"],
        message: "Recovery source invocation identity is invalid"
      });
    }
  });

export function createHumanoidRecoveryPolicy(input: {
  executionTransactionId: string;
  planningTransactionId: string;
  physicalFailureCode: string;
  worldRevision: number;
  binding: ActiveHumanoidSkillBinding;
}): HumanoidRecoveryPolicyState {
  const contract = HUMANOID_SKILL_CONTRACTS[input.binding.invocation.skill];
  const failureReason = physicalFailureReason(
    input.physicalFailureCode,
    input.binding
  );
  const sourceInvocation = structuredClone(input.binding.invocation);
  const excludedInteractionPointIds = failedInteractionPointIds(
    sourceInvocation
  );
  return {
    protocol: "humanoid-recovery-policy-v1",
    source_execution_transaction_id: input.executionTransactionId,
    source_planning_transaction_id: input.planningTransactionId,
    source_skill_transaction_id: input.binding.transaction_id,
    source_skill: input.binding.invocation.skill,
    source_invocation: sourceInvocation,
    source_invocation_sha256: modelPayloadSha256(sourceInvocation),
    source_phase: input.binding.phase,
    physical_failure_code: input.physicalFailureCode,
    failure_reason: failureReason,
    world_revision: input.worldRevision,
    candidate_skills: recoveryCandidates(
      failureReason,
      contract.recovery_entry
    ),
    adaptation_requirements: adaptationRequirements(failureReason),
    ...(excludedInteractionPointIds.length > 0
      ? { excluded_interaction_point_ids: excludedInteractionPointIds }
      : {}),
    requires_model_selection: true,
    automatic_actuation: false
  };
}

function recoveryCandidates(
  reason: HumanoidSkillFailureCode | null,
  contractCandidates: readonly HumanoidSkillId[]
): HumanoidSkillId[] {
  const failureCandidates: Partial<Record<
    HumanoidSkillFailureCode,
    HumanoidSkillId[]
  >> = {
    balance_lost: ["stabilize", "retreat"],
    collision_risk: ["retreat", "stabilize", "approach"],
    object_slipped: ["regrasp", "bimanual_support", "place", "stabilize"],
    grasp_unstable: ["regrasp", "reach", "bimanual_support", "place"],
    contact_missing: ["reach", "regrasp", "approach", "stabilize"],
    articulation_stalled: ["regrasp", "approach", "stabilize", "pull", "push", "retreat"],
    placement_misaligned: ["regrasp", "place", "bimanual_support", "retreat"],
    path_blocked: ["retreat", "approach", "explore", "stabilize"],
    unexpected_world_change: ["explore", "approach", "retreat", "stabilize"],
    unreachable: ["approach", "regrasp", "bimanual_support", "retreat"],
    interaction_point_missing: ["regrasp", "approach", "explore"],
    target_unobserved: ["explore", "approach", "retreat"],
    affordance_missing: ["explore", "retreat"],
    precondition_failed: ["approach", "reach", "stabilize"]
  };
  return [...new Set([
    ...(reason ? failureCandidates[reason] ?? [] : []),
    ...contractCandidates
  ])];
}

export function humanoidRecoverySelectionAccepted(
  policy: HumanoidRecoveryPolicyState,
  invocation: HumanoidSkillInvocation
): boolean {
  if (!policy.candidate_skills.includes(invocation.skill)) return false;
  if (policy.source_invocation_sha256
    && policy.source_invocation_sha256 === modelPayloadSha256(invocation)) {
    return false;
  }
  if (invocation.skill === "regrasp") {
    const excluded = new Set(policy.excluded_interaction_point_ids ?? []);
    if (excluded.has(invocation.interaction_point_id)) return false;
    if ([...excluded].some((pointId) => (
      !invocation.excluded_interaction_point_ids.includes(pointId)
    ))) return false;
  }
  return true;
}

function physicalFailureReason(
  code: string,
  binding: ActiveHumanoidSkillBinding
): HumanoidSkillFailureCode | null {
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
  if (binding.invocation.skill === "grasp" || binding.invocation.skill === "regrasp") {
    return "grasp_unstable";
  }
  if (binding.invocation.skill === "lift" || binding.invocation.skill === "carry"
    || binding.invocation.skill === "bimanual_carry") return "object_slipped";
  if (binding.invocation.skill === "place") return "placement_misaligned";
  if (["open", "close", "turn", "press", "push", "pull"].includes(
    binding.invocation.skill
  )) return "articulation_stalled";
  if (binding.phase_authority === "navigation") return "path_blocked";
  if (binding.invocation.skill === "stabilize") return "balance_lost";
  return null;
}

function failedInteractionPointIds(
  invocation: HumanoidSkillInvocation
): string[] {
  const selected = "interaction_point_id" in invocation
    && invocation.interaction_point_id !== null
    ? [invocation.interaction_point_id]
    : invocation.skill === "bimanual_support"
      ? [
          invocation.left_interaction_point_id,
          invocation.right_interaction_point_id
        ]
      : [];
  const existing = invocation.skill === "regrasp"
    ? invocation.excluded_interaction_point_ids
    : [];
  return [...new Set([...existing, ...selected])].sort();
}

function adaptationRequirements(
  reason: HumanoidSkillFailureCode | null
): string[] {
  const common = [
    "reobserve_current_world",
    "preserve_active_goal",
    "do_not_repeat_identical_invocation"
  ];
  const specific: Partial<Record<HumanoidSkillFailureCode, string[]>> = {
    grasp_unstable: ["change_grasp_point_or_hand"],
    contact_missing: ["change_contact_alignment_or_approach"],
    object_slipped: ["relocalize_object_before_regrasp"],
    path_blocked: ["plan_from_current_pose_with_updated_obstacles"],
    articulation_stalled: ["change_stance_contact_point_or_force_direction"],
    placement_misaligned: ["reobserve_destination_and_realign_pose"],
    unexpected_world_change: ["discard_stale_skill_dag"],
    balance_lost: ["stabilize_or_retreat_before_task_progress"],
    collision_risk: ["increase_clearance_before_retry"],
    interaction_point_missing: ["select_a_currently_observed_interaction_point"],
    target_unobserved: ["reobserve_or_explore_before_manipulation"],
    unreachable: ["change_base_placement_hand_or_interaction_point"],
    affordance_missing: ["select_an_object_with_required_affordance"],
    precondition_failed: ["satisfy_missing_skill_preconditions"]
  };
  return [...common, ...(reason ? specific[reason] ?? [] : [])];
}
