import {
  createHumanoidGroundingObligation,
  createHumanoidGroundingReceipt,
  type HumanoidGroundingObligation,
  type HumanoidGroundingReceipt
} from "../../domain/humanoid-grounding.js";
import { modelPayloadSha256 } from "../../domain/model-call-authority.js";
import type { NeuralSafetyInterrupt } from "../../domain/neural-hierarchy.js";
import type { Goal, JsonValue, Vec3 } from "../../domain/schema.js";
import type { HumanoidWorldObservation } from
  "../../world/humanoid/world.js";
import {
  ActiveHumanoidSkillBindingSchema,
  bindHumanoidSkill,
  humanoidEmbodiedSkillIdentity,
  type ActiveHumanoidSkillBinding
} from "./skill-binding.js";
import {
  humanoidRecoverySafetyInterruptIsCurrent
} from "./recovery-safety-authority.js";

const TARGET_POSITION_TOLERANCE_METERS = 0.015;
const ARTICULATION_POSITION_TOLERANCE = 0.015;

interface GroundingPlanningReceipt {
  transactionId: string;
  accepted: boolean;
  worldAfterRevision: number;
  detail: JsonValue;
}

interface GroundingExecutionIntent {
  transactionId: string;
  planningTransactionId: string;
  planId: string;
}

export function groundHumanoidPhysicalExecution(input: {
  planningReceipt: GroundingPlanningReceipt;
  intent: GroundingExecutionIntent;
  observation: HumanoidWorldObservation;
  authorityStateSha256: string;
  activeGoal?: Goal;
  recoveryInterrupts?: readonly NeuralSafetyInterrupt[];
}): HumanoidGroundingReceipt {
  const detail = record(input.planningReceipt.detail);
  const parsedBinding = ActiveHumanoidSkillBindingSchema.safeParse(
    detail?.skill_binding
  );
  const bindingPresent = detail?.skill_binding !== undefined;
  const binding = parsedBinding.success ? parsedBinding.data : null;
  const currentGoalSha256 = input.activeGoal
    ? modelPayloadSha256(input.activeGoal)
    : null;
  const recoveryInterrupt = binding?.recovery_interrupt_id
    ? input.recoveryInterrupts?.find((interrupt) => (
        interrupt.interrupt_id === binding.recovery_interrupt_id
      ))
    : undefined;
  const rebound = binding
    ? bindHumanoidSkill({
        transactionId: binding.transaction_id,
        agentId: binding.agent_id,
        request: {
          skill_plan_transaction_id: binding.skill_plan_transaction_id,
          skill_node_id: binding.skill_node_id,
          invocation: binding.invocation,
          phase: binding.phase
        },
        observation: input.observation,
        ...(input.activeGoal ? { activeGoal: input.activeGoal } : {}),
        ...(binding.recovery_authorized ? { recoveryAuthorized: true } : {}),
        ...(recoveryInterrupt ? { recoveryInterrupt } : {})
      })
    : null;
  const reboundBinding = rebound?.accepted ? rebound.binding : null;
  const obligations: HumanoidGroundingObligation[] = [
    obligation({
      id: "planning_plan",
      scope: "plan",
      required: true,
      satisfied: input.planningReceipt.accepted
        && input.planningReceipt.transactionId
          === input.intent.planningTransactionId
        && detail?.plan_id === input.intent.planId,
      successCode: "planning_plan_bound",
      failureCode: "planning_plan_mismatch",
      detail: {
        planning_transaction_id: input.planningReceipt.transactionId,
        requested_planning_transaction_id: input.intent.planningTransactionId,
        planned_plan_id: typeof detail?.plan_id === "string"
          ? detail.plan_id
          : null,
        requested_plan_id: input.intent.planId,
        planning_accepted: input.planningReceipt.accepted
      }
    }),
    obligation({
      id: "world_authority",
      scope: "world",
      required: true,
      satisfied: input.planningReceipt.worldAfterRevision
        <= input.observation.worldRevision,
      successCode: input.planningReceipt.worldAfterRevision
        === input.observation.worldRevision
        ? "world_authority_current"
        : "world_authority_requires_plan_revalidation",
      failureCode: "world_authority_regressed",
      detail: {
        planned_world_revision: input.planningReceipt.worldAfterRevision,
        current_world_frame: input.observation.frame,
        current_world_revision: input.observation.worldRevision,
        authority_state_sha256: input.authorityStateSha256
      }
    }),
    skillBindingObligation({
      binding,
      bindingPresent,
      parsedBinding,
      observation: input.observation,
      semanticRegrounded: rebound?.accepted === true
    }),
    recoveryInterruptObligation(
      binding,
      recoveryInterrupt,
      input.observation
    ),
    activeGoalObligation(binding, currentGoalSha256),
    semanticPreconditionsObligation(bindingPresent, rebound),
    targetEvidenceObligation(binding, reboundBinding),
    interactionEvidenceObligation(binding, reboundBinding)
  ];
  return createHumanoidGroundingReceipt({
    protocol: "humanoid-grounding-receipt-v1",
    receipt_id: [
      "grounding",
      input.intent.transactionId,
      input.observation.worldRevision
    ].join(":"),
    transaction_id: input.intent.transactionId,
    planning_transaction_id: input.intent.planningTransactionId,
    plan_id: input.intent.planId,
    call_id: binding ? humanoidEmbodiedSkillIdentity(binding).callId : null,
    world_frame: input.observation.frame,
    world_revision: input.observation.worldRevision,
    authority_state_sha256: input.authorityStateSha256,
    obligations
  });
}

function recoveryInterruptObligation(
  binding: ActiveHumanoidSkillBinding | null,
  interrupt: NeuralSafetyInterrupt | undefined,
  observation: HumanoidWorldObservation
): HumanoidGroundingObligation {
  const expectedId = binding?.recovery_interrupt_id;
  if (!expectedId) {
    return createHumanoidGroundingObligation({
      id: "recovery_interrupt",
      scope: "skill",
      required: false,
      status: "not_applicable",
      code: "recovery_interrupt_not_required",
      detail: null
    });
  }
  const satisfied = binding?.recovery_authorized === true
    && binding.invocation.skill === "stabilize"
    && binding.phase === "recover_support"
    && humanoidRecoverySafetyInterruptIsCurrent(interrupt, {
      worldRevision: observation.worldRevision,
      interruptId: expectedId
    });
  return obligation({
    id: "recovery_interrupt",
    scope: "skill",
    required: true,
    satisfied,
    successCode: "recovery_interrupt_current",
    failureCode: "recovery_interrupt_invalid",
    detail: {
      expected_interrupt_id: expectedId,
      current_interrupt_id: interrupt?.interrupt_id ?? null,
      interrupt_status: interrupt?.status ?? null,
      fallen: observation.robot.fallen,
      current_world_revision: observation.worldRevision
    }
  });
}

function skillBindingObligation(input: {
  binding: ActiveHumanoidSkillBinding | null;
  bindingPresent: boolean;
  parsedBinding: ReturnType<typeof ActiveHumanoidSkillBindingSchema.safeParse>;
  observation: HumanoidWorldObservation;
  semanticRegrounded: boolean;
}): HumanoidGroundingObligation {
  if (!input.bindingPresent) {
    return createHumanoidGroundingObligation({
      id: "skill_binding",
      scope: "skill",
      required: false,
      status: "not_applicable",
      code: "legacy_plan_without_skill_binding",
      detail: null
    });
  }
  const binding = input.binding;
  return obligation({
    id: "skill_binding",
    scope: "skill",
    required: true,
    satisfied: binding !== null
      && ((binding.observed_frame === input.observation.frame
        && binding.observed_world_revision === input.observation.worldRevision)
        || (binding.observed_frame <= input.observation.frame
          && binding.observed_world_revision <= input.observation.worldRevision
          && input.semanticRegrounded)),
    successCode: binding
      && (binding.observed_frame !== input.observation.frame
        || binding.observed_world_revision !== input.observation.worldRevision)
      ? "skill_binding_regrounded"
      : "skill_binding_current",
    failureCode: input.parsedBinding.success
      ? "skill_binding_stale"
      : "skill_binding_invalid",
    detail: binding
      ? {
          binding_transaction_id: binding.transaction_id,
          invocation_sha256: binding.invocation_sha256,
          skill_catalog_sha256: binding.skill_catalog_sha256,
          bound_world_frame: binding.observed_frame,
          bound_world_revision: binding.observed_world_revision,
          current_world_frame: input.observation.frame,
          current_world_revision: input.observation.worldRevision
        }
      : {
          binding_present: true,
          parse_error: input.parsedBinding.success
            ? null
            : input.parsedBinding.error.issues.map(({ path, message }) => ({
                path: path.join("."),
                message
              }))
        }
  });
}

function semanticPreconditionsObligation(
  bindingPresent: boolean,
  rebound: ReturnType<typeof bindHumanoidSkill> | null
): HumanoidGroundingObligation {
  if (!bindingPresent) {
    return createHumanoidGroundingObligation({
      id: "semantic_preconditions",
      scope: "skill",
      required: false,
      status: "not_applicable",
      code: "legacy_plan_without_semantic_preconditions",
      detail: null
    });
  }
  return obligation({
    id: "semantic_preconditions",
    scope: "skill",
    required: true,
    satisfied: rebound?.accepted === true,
    successCode: "semantic_preconditions_current",
    failureCode: rebound && !rebound.accepted
      ? rebound.code
      : "semantic_binding_unavailable",
    detail: rebound?.accepted
      ? {
          skill: rebound.binding.invocation.skill,
          phase: rebound.binding.phase,
          control_mode: rebound.binding.control_mode,
          required_capabilities:
            rebound.binding.learned_policy_required_capabilities,
          missing_capabilities:
            rebound.binding.learned_policy_missing_capabilities
        }
      : rebound?.detail ?? { binding_available: false }
  });
}

function activeGoalObligation(
  binding: ActiveHumanoidSkillBinding | null,
  currentGoalSha256: string | null
): HumanoidGroundingObligation {
  const expected = binding?.active_goal_sha256 ?? null;
  if (expected === null) {
    return createHumanoidGroundingObligation({
      id: "active_goal",
      scope: "goal",
      required: false,
      status: "not_applicable",
      code: "active_goal_not_bound",
      detail: { current_goal_sha256: currentGoalSha256 }
    });
  }
  return obligation({
    id: "active_goal",
    scope: "goal",
    required: true,
    satisfied: expected === currentGoalSha256,
    successCode: "active_goal_current",
    failureCode: "active_goal_changed",
    detail: {
      bound_goal_sha256: expected,
      current_goal_sha256: currentGoalSha256
    }
  });
}

function targetEvidenceObligation(
  binding: ActiveHumanoidSkillBinding | null,
  rebound: ActiveHumanoidSkillBinding | null
): HumanoidGroundingObligation {
  if (!binding || (binding.target_evidence_position === null
    && binding.target_solid === null
    && binding.target_articulation === null)) {
    return createHumanoidGroundingObligation({
      id: "target_evidence",
      scope: "object",
      required: false,
      status: "not_applicable",
      code: "target_evidence_not_required",
      detail: null
    });
  }
  const positionDrift = distanceNullable(
    binding.target_evidence_position,
    rebound?.target_evidence_position ?? null
  );
  const solidMatches = sameSolid(binding.target_solid, rebound?.target_solid ?? null);
  const articulationMatches = sameArticulation(
    binding.target_articulation,
    rebound?.target_articulation ?? null
  );
  return obligation({
    id: "target_evidence",
    scope: "object",
    required: true,
    satisfied: rebound !== null
      && positionDrift <= TARGET_POSITION_TOLERANCE_METERS
      && solidMatches
      && articulationMatches,
    successCode: "target_evidence_current",
    failureCode: "target_evidence_changed",
    detail: {
      position_drift_m: Number.isFinite(positionDrift) ? positionDrift : null,
      maximum_position_drift_m: TARGET_POSITION_TOLERANCE_METERS,
      solid_matches: solidMatches,
      articulation_matches: articulationMatches,
      bound_target_position: binding.target_evidence_position,
      current_target_position: rebound?.target_evidence_position ?? null
    }
  });
}

function interactionEvidenceObligation(
  binding: ActiveHumanoidSkillBinding | null,
  rebound: ActiveHumanoidSkillBinding | null
): HumanoidGroundingObligation {
  const expected = binding?.eligible_interaction_points ?? [];
  if (expected.length === 0) {
    return createHumanoidGroundingObligation({
      id: "interaction_evidence",
      scope: "object",
      required: false,
      status: "not_applicable",
      code: "interaction_evidence_not_required",
      detail: null
    });
  }
  const actual = new Map(
    rebound?.eligible_interaction_points.map((point) => [point.id, point]) ?? []
  );
  const changed = expected.flatMap((point) => {
    const current = actual.get(point.id);
    const drift = current
      ? distance(point.world_position, current.world_position)
      : Number.POSITIVE_INFINITY;
    return !current
      || current.kind !== point.kind
      || current.compatible_hands !== point.compatible_hands
      || drift > TARGET_POSITION_TOLERANCE_METERS
      ? [{ id: point.id, drift_m: Number.isFinite(drift) ? drift : null }]
      : [];
  });
  return obligation({
    id: "interaction_evidence",
    scope: "object",
    required: true,
    satisfied: rebound !== null && changed.length === 0,
    successCode: "interaction_evidence_current",
    failureCode: "interaction_evidence_changed",
    detail: {
      expected_interaction_point_ids: expected.map(({ id }) => id),
      current_interaction_point_ids: [...actual.keys()].sort(compareCodePoints),
      changed
    }
  });
}

function obligation(input: {
  id: HumanoidGroundingObligation["id"];
  scope: HumanoidGroundingObligation["scope"];
  required: boolean;
  satisfied: boolean;
  successCode: string;
  failureCode: string;
  detail: JsonValue;
}): HumanoidGroundingObligation {
  return createHumanoidGroundingObligation({
    id: input.id,
    scope: input.scope,
    required: input.required,
    status: input.satisfied ? "satisfied" : "failed",
    code: input.satisfied ? input.successCode : input.failureCode,
    detail: input.detail
  });
}

function sameSolid(
  left: ActiveHumanoidSkillBinding["target_solid"],
  right: ActiveHumanoidSkillBinding["target_solid"]
): boolean {
  if (left === null || right === null) return left === right;
  return left.id === right.id
    && left.sourceId === right.sourceId
    && left.kind === right.kind
    && distance(left.center, right.center) <= 1e-9
    && distance(left.size, right.size) <= 1e-9;
}

function sameArticulation(
  left: ActiveHumanoidSkillBinding["target_articulation"],
  right: ActiveHumanoidSkillBinding["target_articulation"]
): boolean {
  if (left === null || right === null) return left === right;
  const positionDrift = left.position === null || right.position === null
    ? left.position === right.position ? 0 : Number.POSITIVE_INFINITY
    : Math.abs(left.position - right.position);
  return left.joint_id === right.joint_id
    && left.type === right.type
    && left.semantic === right.semantic
    && positionDrift <= ARTICULATION_POSITION_TOLERANCE;
}

function distanceNullable(left: Vec3 | null, right: Vec3 | null): number {
  if (left === null || right === null) {
    return left === right ? 0 : Number.POSITIVE_INFINITY;
  }
  return distance(left, right);
}

function distance(left: Vec3, right: Vec3): number {
  return Math.hypot(
    left.x - right.x,
    left.y - right.y,
    left.z - right.z
  );
}

function record(value: JsonValue | undefined): Record<string, JsonValue> | null {
  return value !== undefined && value !== null && typeof value === "object"
    && !Array.isArray(value)
    ? value
    : null;
}

function compareCodePoints(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}
