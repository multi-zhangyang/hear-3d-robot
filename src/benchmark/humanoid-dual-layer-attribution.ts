import { z } from "zod";
import {
  HumanoidGroundingReceiptSchema,
  type HumanoidGroundingReceipt
} from "../domain/humanoid-grounding.js";
import {
  HumanoidReplanModelCallSchema,
  type HumanoidReplanModelCall
} from "../domain/humanoid-replan-budget.js";
import {
  PersistedHumanoidActionReceiptSchema
} from "../domain/humanoid-run.js";
import { RunStatusSchema } from "../domain/schema.js";
import {
  ActiveHumanoidSkillBindingSchema,
  humanoidEmbodiedSkillIdentity
} from "../harness/humanoid/skill-binding.js";

const PlanningActions = new Set([
  "plan_humanoid_skill",
  "plan_whole_body_motion",
  "plan_whole_body_motion_candidates",
  "plan_humanoid_navigation"
]);

const ExecutionActions = new Set([
  "execute_humanoid_skill",
  "execute_whole_body_motion",
  "execute_humanoid_navigation"
]);

const SuccessfulExecutionCodes = new Set([
  "motion_completed",
  "navigation_completed",
  "motion_option_succeeded"
]);

const AuthorityRejectionCodes = new Set([
  "autonomous_skill_authority_missing",
  "planning_receipt_action_mismatch",
  "planning_receipt_missing",
  "planning_skill_authority_missing",
  "skill_phase_authority_mismatch",
  "skill_plan_dependencies_incomplete",
  "unauthorized_model_source"
]);

const PlanLifecycleRejectionCodes = new Set([
  "plan_revalidation_failed",
  "plan_stale",
  "skill_observation_changed",
  "skill_world_revision_stale"
]);

const BoundaryObligations = new Set([
  "planning_plan",
  "world_authority",
  "skill_binding",
  "active_goal"
]);

const SemanticObligations = new Set([
  "semantic_preconditions",
  "target_evidence",
  "interaction_evidence"
]);

type Receipt = z.infer<typeof PersistedHumanoidActionReceiptSchema>;
type RunStatus = z.infer<typeof RunStatusSchema>;

export type HumanoidFailureLayer =
  | "semantic_planner"
  | "harness_gate"
  | "controller"
  | "none"
  | "unattributed";

interface HumanoidExecutionAttribution {
  transaction_id: string;
  planning_transaction_id: string | null;
  semantic_skill_call_id: string | null;
  semantic_skill: boolean;
  code: string;
  frame_count: number;
  harness: {
    decision: "admitted" | "rejected";
    rejection_class: "grounding" | "authority" | "plan_lifecycle"
      | "pre_actuation_other" | null;
    grounding_receipt_id: string | null;
    grounding_accepted: boolean | null;
    failed_obligation_ids: string[];
    automatic_actuation: boolean;
  };
  controller: {
    outcome: "succeeded" | "failed" | "not_started" | "not_required";
    route: "primary" | "fallback" | "upper_body_overlay" | "unavailable";
    primary_steps: number;
    fallback_steps: number;
    upper_body_overlay_steps: number;
    memory_bridge_steps: number;
    failure_class: "safety" | "execution_drift" | "contract_unmet"
      | "interrupted" | "other" | null;
  };
  primary_failure_layer: HumanoidFailureLayer;
}

export interface HumanoidDualLayerRunAttribution {
  protocol: "humanoid-dual-layer-attribution-v1";
  coverage: {
    semantic_skill_call_count: number;
    semantic_execution_count: number;
    grounded_semantic_execution_count: number;
    routed_execution_count: number;
    fully_attributed_semantic_execution_count: number;
    full_attribution_rate: number | null;
  };
  harness: {
    semantic_planning: {
      skill_call_count: number;
      accepted_skill_call_count: number;
      rejected_skill_call_count: number;
      success_rate: number | null;
    };
    skill_use: {
      trigger_expected: number;
      trigger_observed: number;
      trigger_missed: number;
      trigger_pending: number;
      trigger_rate: number | null;
      compliance_evaluated: number;
      compliant: number;
      compliance_violations: number;
      compliance_rate: number | null;
      failed_obligation_counts: Record<string, number>;
      boundary_attempts: number;
      boundary_contained: number;
      boundary_escaped: number;
      boundary_containment_rate: number | null;
    };
    dispatch: {
      attempts: number;
      admitted: number;
      rejected_before_actuation: number;
      grounding_rejections: number;
      authority_rejections: number;
      plan_lifecycle_rejections: number;
      other_rejections: number;
    };
    recovery: {
      model_call_evidence_available: boolean;
      compact_replan_decisions: number;
      specialist_replan_calls: number;
      goal_reevaluation_decisions: number;
      goal_reevaluation_calls: number;
      completed_model_calls: number;
      failed_model_calls: number;
      in_flight_model_calls: number;
      model_call_slo_violations: number;
      recovery_deadline_exceeded: boolean;
      local_navigation_replans: number;
      local_navigation_replans_accepted: number;
      local_navigation_budget_exhaustions: number;
      local_navigation_model_calls: number;
    };
  };
  controller: {
    admitted_execution_count: number;
    physical_execution_count: number;
    succeeded_execution_count: number;
    failed_execution_count: number;
    success_rate_after_admission: number | null;
    not_required_execution_count: number;
    route_counts: {
      primary: number;
      fallback: number;
      upper_body_overlay: number;
      unavailable: number;
    };
    attributed_steps: {
      primary: number;
      fallback: number;
      upper_body_overlay: number;
      memory_bridge: number;
    };
    primary_only_successes: number;
    fallback_rescued_successes: number;
    upper_body_overlay_successes: number;
    failure_class_counts: {
      safety: number;
      execution_drift: number;
      contract_unmet: number;
      interrupted: number;
      other: number;
    };
  };
  joint: {
    successful_semantic_skills: number;
    failed_semantic_skills: number;
    semantic_skill_success_rate: number | null;
    harness_rejected_no_actuation: number;
    harness_admitted_controller_succeeded: number;
    harness_admitted_controller_failed: number;
    harness_admitted_controller_not_required: number;
    primary_failure_layer_counts: Record<HumanoidFailureLayer, number>;
    mission_failure_unattributed: boolean;
  };
  executions: HumanoidExecutionAttribution[];
}

export function attributeHumanoidDualLayerRun(input: {
  status: RunStatus;
  actions: readonly unknown[];
  events?: readonly unknown[];
}): HumanoidDualLayerRunAttribution {
  const actions = input.actions.map(parseJournalAction);
  const planning = actions.filter((action) => PlanningActions.has(action.action));
  const executions = actions.filter((action) => ExecutionActions.has(action.action));
  const semanticGroups = groupSemanticSkillPlans(planning);
  const executionsByPlanningId = new Map(executions.flatMap((execution) => {
    const planningId = nonEmptyString(record(execution.input)?.planning_transaction_id);
    return planningId ? [[planningId, execution] as const] : [];
  }));
  const terminal = ["succeeded", "failed", "interrupted"].includes(input.status);
  let triggerExpected = 0;
  let triggerObserved = 0;
  let triggerMissed = 0;
  let triggerPending = 0;
  let acceptedSkillCalls = 0;
  for (const group of semanticGroups.values()) {
    const acceptedPlans = group.plans.filter((plan) => plan.accepted);
    if (acceptedPlans.length === 0) continue;
    acceptedSkillCalls += 1;
    triggerExpected += 1;
    const observed = acceptedPlans.some((plan) => (
      executionsByPlanningId.has(plan.transactionId)
    ));
    if (observed) triggerObserved += 1;
    else if (terminal) triggerMissed += 1;
    else triggerPending += 1;
  }

  const callIdByPlanningId = new Map<string, string>();
  for (const group of semanticGroups.values()) {
    for (const plan of group.plans) callIdByPlanningId.set(plan.transactionId, group.callId);
  }
  const attributions = executions.map((execution) => executionAttribution(
    execution,
    callIdByPlanningId
  ));
  const semanticAttributions = attributions.filter((entry) => entry.semantic_skill);
  const groundingReceipts = executions.flatMap((execution) => {
    const grounding = groundingReceipt(execution);
    return grounding ? [grounding] : [];
  });
  const failedObligations = groundingReceipts.flatMap((receipt) => (
    receipt.obligations.filter((obligation) => (
      obligation.required && obligation.status === "failed"
    )).map((obligation) => obligation.id)
  ));
  const boundaryEntries = attributions.filter((entry) => boundaryAttempt(entry));
  const controllerAdmitted = attributions.filter(
    (entry) => entry.harness.decision === "admitted"
  );
  const physicalController = controllerAdmitted.filter(
    (entry) => entry.controller.outcome !== "not_required"
  );
  const controllerSucceeded = physicalController.filter(
    (entry) => entry.controller.outcome === "succeeded"
  );
  const controllerFailed = physicalController.filter(
    (entry) => entry.controller.outcome === "failed"
  );
  const recovery = recoveryMetrics(input.events ?? [], executions);
  const planningFailures = [...semanticGroups.values()].filter(
    (group) => !group.plans.some((plan) => plan.accepted)
  ).length;
  const failureLayerCounts: Record<HumanoidFailureLayer, number> = {
    semantic_planner: planningFailures,
    harness_gate: triggerMissed,
    controller: 0,
    none: 0,
    unattributed: 0
  };
  for (const attribution of attributions) {
    failureLayerCounts[attribution.primary_failure_layer] += 1;
  }
  const successfulSemanticSkills = semanticAttributions.filter(
    (entry) => entry.controller.outcome === "succeeded"
      || entry.controller.outcome === "not_required"
  ).length;
  const failedSemanticSkills = semanticAttributions.length
    - successfulSemanticSkills;
  const attributedFailureCount = failureLayerCounts.semantic_planner
    + failureLayerCounts.harness_gate
    + failureLayerCounts.controller;
  const missionFailureUnattributed = terminal
    && input.status !== "succeeded"
    && attributedFailureCount === 0;
  if (missionFailureUnattributed) failureLayerCounts.unattributed += 1;

  const fullyAttributedSemantic = semanticAttributions.filter((entry) => (
    entry.harness.grounding_receipt_id !== null
      && entry.controller.route !== "unavailable"
  )).length;
  const complianceReceipts = groundingReceipts.filter((receipt) => (
    receipt.obligations.some((obligation) => (
      obligation.required && SemanticObligations.has(obligation.id)
    ))
  ));
  const groundingCompliance = complianceReceipts.filter((receipt) => (
    !receipt.obligations.some((obligation) => (
      obligation.required
        && SemanticObligations.has(obligation.id)
        && obligation.status === "failed"
    ))
  ));
  const routeCounts = countRoutes(attributions);
  const failureClassCounts = countControllerFailureClasses(controllerFailed);
  const primaryOnlySuccesses = controllerSucceeded.filter((entry) => (
    entry.controller.primary_steps > 0
      && entry.controller.fallback_steps === 0
      && entry.controller.upper_body_overlay_steps === 0
      && entry.controller.memory_bridge_steps === 0
  )).length;
  const fallbackRescuedSuccesses = controllerSucceeded.filter((entry) => (
    entry.controller.fallback_steps > 0
      || entry.controller.memory_bridge_steps > 0
  )).length;
  const upperBodyOverlaySuccesses = controllerSucceeded.filter(
    (entry) => entry.controller.upper_body_overlay_steps > 0
  ).length;

  return {
    protocol: "humanoid-dual-layer-attribution-v1",
    coverage: {
      semantic_skill_call_count: semanticGroups.size,
      semantic_execution_count: semanticAttributions.length,
      grounded_semantic_execution_count: semanticAttributions.filter(
        (entry) => entry.harness.grounding_receipt_id !== null
      ).length,
      routed_execution_count: attributions.filter(
        (entry) => entry.controller.route !== "unavailable"
      ).length,
      fully_attributed_semantic_execution_count: fullyAttributedSemantic,
      full_attribution_rate: ratio(
        fullyAttributedSemantic,
        semanticAttributions.length
      )
    },
    harness: {
      semantic_planning: {
        skill_call_count: semanticGroups.size,
        accepted_skill_call_count: acceptedSkillCalls,
        rejected_skill_call_count: semanticGroups.size - acceptedSkillCalls,
        success_rate: ratio(acceptedSkillCalls, semanticGroups.size)
      },
      skill_use: {
        trigger_expected: triggerExpected,
        trigger_observed: triggerObserved,
        trigger_missed: triggerMissed,
        trigger_pending: triggerPending,
        trigger_rate: ratio(triggerObserved, triggerExpected),
        compliance_evaluated: complianceReceipts.length,
        compliant: groundingCompliance.length,
        compliance_violations: complianceReceipts.length - groundingCompliance.length,
        compliance_rate: ratio(groundingCompliance.length, complianceReceipts.length),
        failed_obligation_counts: countStrings(failedObligations),
        boundary_attempts: boundaryEntries.length,
        boundary_contained: boundaryEntries.filter(
          (entry) => !entry.harness.automatic_actuation
        ).length,
        boundary_escaped: boundaryEntries.filter(
          (entry) => entry.harness.automatic_actuation
        ).length,
        boundary_containment_rate: ratio(
          boundaryEntries.filter((entry) => !entry.harness.automatic_actuation).length,
          boundaryEntries.length
        )
      },
      dispatch: {
        attempts: attributions.length,
        admitted: controllerAdmitted.length,
        rejected_before_actuation: attributions.filter(
          (entry) => entry.harness.decision === "rejected"
            && !entry.harness.automatic_actuation
        ).length,
        grounding_rejections: countRejectionClass(attributions, "grounding"),
        authority_rejections: countRejectionClass(attributions, "authority"),
        plan_lifecycle_rejections: countRejectionClass(
          attributions,
          "plan_lifecycle"
        ),
        other_rejections: countRejectionClass(
          attributions,
          "pre_actuation_other"
        )
      },
      recovery
    },
    controller: {
      admitted_execution_count: controllerAdmitted.length,
      physical_execution_count: physicalController.length,
      succeeded_execution_count: controllerSucceeded.length,
      failed_execution_count: controllerFailed.length,
      success_rate_after_admission: ratio(
        controllerSucceeded.length,
        physicalController.length
      ),
      not_required_execution_count: controllerAdmitted.filter(
        (entry) => entry.controller.outcome === "not_required"
      ).length,
      route_counts: routeCounts,
      attributed_steps: {
        primary: sum(attributions.map((entry) => entry.controller.primary_steps)),
        fallback: sum(attributions.map((entry) => entry.controller.fallback_steps)),
        upper_body_overlay: sum(attributions.map(
          (entry) => entry.controller.upper_body_overlay_steps
        )),
        memory_bridge: sum(attributions.map(
          (entry) => entry.controller.memory_bridge_steps
        ))
      },
      primary_only_successes: primaryOnlySuccesses,
      fallback_rescued_successes: fallbackRescuedSuccesses,
      upper_body_overlay_successes: upperBodyOverlaySuccesses,
      failure_class_counts: failureClassCounts
    },
    joint: {
      successful_semantic_skills: successfulSemanticSkills,
      failed_semantic_skills: failedSemanticSkills,
      semantic_skill_success_rate: ratio(
        successfulSemanticSkills,
        semanticAttributions.length
      ),
      harness_rejected_no_actuation: attributions.filter(
        (entry) => entry.harness.decision === "rejected"
          && !entry.harness.automatic_actuation
      ).length,
      harness_admitted_controller_succeeded: controllerSucceeded.length,
      harness_admitted_controller_failed: controllerFailed.length,
      harness_admitted_controller_not_required: controllerAdmitted.filter(
        (entry) => entry.controller.outcome === "not_required"
      ).length,
      primary_failure_layer_counts: failureLayerCounts,
      mission_failure_unattributed: missionFailureUnattributed
    },
    executions: attributions
  };
}

function executionAttribution(
  execution: Receipt,
  callIdByPlanningId: ReadonlyMap<string, string>
): HumanoidExecutionAttribution {
  const planningId = nonEmptyString(record(execution.input)?.planning_transaction_id);
  const semantic = execution.action === "execute_humanoid_skill";
  const grounding = groundingReceipt(execution);
  const rejected = !execution.accepted && execution.frameCount === 0;
  const rejectionClass = rejected
    ? harnessRejectionClass(execution, grounding)
    : null;
  const route = controllerRoute(execution);
  const success = execution.accepted && SuccessfulExecutionCodes.has(execution.code);
  const notRequired = execution.accepted && execution.frameCount === 0;
  const controllerOutcome = rejected
    ? "not_started" as const
    : notRequired
      ? "not_required" as const
      : success
        ? "succeeded" as const
        : "failed" as const;
  const failureLayer: HumanoidFailureLayer = rejected
    ? "harness_gate"
    : controllerOutcome === "failed"
      ? "controller"
      : controllerOutcome === "succeeded" || controllerOutcome === "not_required"
        ? "none"
        : "unattributed";
  return {
    transaction_id: execution.transactionId,
    planning_transaction_id: planningId ?? null,
    semantic_skill_call_id: planningId
      ? callIdByPlanningId.get(planningId) ?? null
      : null,
    semantic_skill: semantic,
    code: execution.code,
    frame_count: execution.frameCount,
    harness: {
      decision: rejected ? "rejected" : "admitted",
      rejection_class: rejectionClass,
      grounding_receipt_id: grounding?.receipt_id ?? null,
      grounding_accepted: grounding?.accepted ?? null,
      failed_obligation_ids: grounding?.failed_obligation_ids ?? [],
      automatic_actuation: execution.frameCount > 0
    },
    controller: {
      outcome: controllerOutcome,
      route: route.route,
      primary_steps: route.primarySteps,
      fallback_steps: route.fallbackSteps,
      upper_body_overlay_steps: route.upperBodyOverlaySteps,
      memory_bridge_steps: route.memoryBridgeSteps,
      failure_class: controllerOutcome === "failed"
        ? controllerFailureClass(execution)
        : null
    },
    primary_failure_layer: failureLayer
  };
}

function groupSemanticSkillPlans(
  planning: readonly Receipt[]
): Map<string, { callId: string; plans: Receipt[] }> {
  const groups = new Map<string, { callId: string; plans: Receipt[] }>();
  for (const plan of planning) {
    if (plan.action !== "plan_humanoid_skill") continue;
    const detail = record(plan.detail);
    const binding = ActiveHumanoidSkillBindingSchema.safeParse(detail?.skill_binding);
    const callId = binding.success
      ? humanoidEmbodiedSkillIdentity(binding.data).callId
      : `unbound-plan:${plan.transactionId}`;
    const group = groups.get(callId) ?? { callId, plans: [] };
    group.plans.push(plan);
    groups.set(callId, group);
  }
  return groups;
}

function harnessRejectionClass(
  execution: Receipt,
  grounding: HumanoidGroundingReceipt | undefined
): HumanoidExecutionAttribution["harness"]["rejection_class"] {
  if (execution.code === "execution_grounding_rejected" || grounding?.accepted === false) {
    return "grounding";
  }
  if (AuthorityRejectionCodes.has(execution.code)) return "authority";
  if (PlanLifecycleRejectionCodes.has(execution.code)) return "plan_lifecycle";
  return "pre_actuation_other";
}

function boundaryAttempt(entry: HumanoidExecutionAttribution): boolean {
  if (entry.harness.rejection_class === "authority") return true;
  return entry.harness.failed_obligation_ids.some((id) => BoundaryObligations.has(id));
}

function groundingReceipt(execution: Receipt): HumanoidGroundingReceipt | undefined {
  const detail = record(execution.detail);
  const parsed = HumanoidGroundingReceiptSchema.safeParse(detail?.grounding_receipt);
  return parsed.success ? parsed.data : undefined;
}

function controllerRoute(execution: Receipt): {
  route: HumanoidExecutionAttribution["controller"]["route"];
  primarySteps: number;
  fallbackSteps: number;
  upperBodyOverlaySteps: number;
  memoryBridgeSteps: number;
} {
  const detail = record(execution.detail);
  const result = record(detail?.result);
  const routing = record(result?.controller_routing ?? detail?.controller_routing);
  const routedExecution = record(routing?.execution);
  const attribution = record(routedExecution?.attribution);
  const route = routedExecution?.route;
  return {
    route: route === "primary" || route === "fallback" || route === "upper_body_overlay"
      ? route
      : "unavailable",
    primarySteps: nonnegativeInteger(attribution?.primarySteps),
    fallbackSteps: nonnegativeInteger(attribution?.fallbackSteps),
    upperBodyOverlaySteps: nonnegativeInteger(attribution?.upperBodyOverlaySteps),
    memoryBridgeSteps: nonnegativeInteger(attribution?.memoryBridgeSteps)
  };
}

function controllerFailureClass(
  execution: Receipt
): NonNullable<HumanoidExecutionAttribution["controller"]["failure_class"]> {
  const detail = record(execution.detail);
  const result = record(detail?.result);
  const reason = [detail?.reason, result?.reason].find(
    (candidate): candidate is string => typeof candidate === "string"
  ) ?? "";
  if (execution.code === "motion_execution_drifted"
    || reason.includes("execution_drift")) return "execution_drift";
  if (execution.code === "execution_interrupted") return "interrupted";
  if (execution.code === "motion_constraint_violated"
    || reason.includes("fallen")
    || reason.includes("unauthorized_contact")
    || reason.includes("carried_object_collision")
    || reason.includes("carried_grasp_lost")) return "safety";
  if (execution.code === "motion_goal_unmet"
    || execution.code === "motion_goal_uncertain"
    || execution.code === "motion_failed"
    || execution.code === "navigation_blocked") return "contract_unmet";
  return "other";
}

function recoveryMetrics(
  events: readonly unknown[],
  executions: readonly Receipt[]
): HumanoidDualLayerRunAttribution["harness"]["recovery"] {
  const calls = new Map<string, HumanoidReplanModelCall>();
  let deadlineExceeded = false;
  for (const rawEvent of events) {
    const event = record(rawEvent);
    const data = record(event?.data) ?? event;
    const evidence = record(data?.replan_budget_evidence);
    const parsed = HumanoidReplanModelCallSchema.safeParse(evidence?.call);
    if (parsed.success) {
      const current = calls.get(parsed.data.model_call_id);
      if (!current || current.status === "started") {
        calls.set(parsed.data.model_call_id, parsed.data);
      }
    }
    const authority = record(evidence?.authority);
    if (authority?.recovery_deadline_exceeded === true) deadlineExceeded = true;
  }
  const localBudgets = executions.flatMap((execution) => {
    const detail = record(execution.detail);
    const result = record(detail?.result);
    const budget = record(result?.online_replan_budget);
    return budget ? [budget] : [];
  });
  const localReplans = executions.flatMap((execution) => {
    const detail = record(execution.detail);
    const result = record(detail?.result);
    return Array.isArray(result?.online_replans)
      ? result.online_replans.flatMap((candidate) => {
          const parsed = record(candidate);
          return parsed ? [parsed] : [];
        })
      : [];
  });
  const modelCalls = [...calls.values()];
  return {
    model_call_evidence_available: modelCalls.length > 0,
    compact_replan_decisions: modelCalls.filter(
      (call) => call.role === "replan_decision"
    ).length,
    specialist_replan_calls: modelCalls.filter(
      (call) => call.role === "specialist_replan"
    ).length,
    goal_reevaluation_decisions: modelCalls.filter(
      (call) => call.role === "goal_re_evaluation_decision"
    ).length,
    goal_reevaluation_calls: modelCalls.filter(
      (call) => call.role === "goal_re_evaluation"
    ).length,
    completed_model_calls: modelCalls.filter((call) => call.status === "completed").length,
    failed_model_calls: modelCalls.filter((call) => call.status === "failed").length,
    in_flight_model_calls: modelCalls.filter((call) => call.status === "started").length,
    model_call_slo_violations: modelCalls.filter(
      (call) => call.slo_violated === true
    ).length,
    recovery_deadline_exceeded: deadlineExceeded,
    local_navigation_replans: localReplans.length,
    local_navigation_replans_accepted: localReplans.filter(
      (replan) => replan.accepted === true
    ).length,
    local_navigation_budget_exhaustions: localBudgets.filter(
      (budget) => budget.terminal_failure_class === "budget_exhausted"
    ).length,
    local_navigation_model_calls: sum(localBudgets.map(
      (budget) => nonnegativeInteger(budget.model_calls_consumed)
    ))
  };
}

function countRoutes(
  entries: readonly HumanoidExecutionAttribution[]
): HumanoidDualLayerRunAttribution["controller"]["route_counts"] {
  return {
    primary: entries.filter((entry) => entry.controller.route === "primary").length,
    fallback: entries.filter((entry) => entry.controller.route === "fallback").length,
    upper_body_overlay: entries.filter(
      (entry) => entry.controller.route === "upper_body_overlay"
    ).length,
    unavailable: entries.filter(
      (entry) => entry.controller.route === "unavailable"
    ).length
  };
}

function countControllerFailureClasses(
  entries: readonly HumanoidExecutionAttribution[]
): HumanoidDualLayerRunAttribution["controller"]["failure_class_counts"] {
  return {
    safety: entries.filter((entry) => entry.controller.failure_class === "safety").length,
    execution_drift: entries.filter(
      (entry) => entry.controller.failure_class === "execution_drift"
    ).length,
    contract_unmet: entries.filter(
      (entry) => entry.controller.failure_class === "contract_unmet"
    ).length,
    interrupted: entries.filter(
      (entry) => entry.controller.failure_class === "interrupted"
    ).length,
    other: entries.filter((entry) => entry.controller.failure_class === "other").length
  };
}

function countRejectionClass(
  entries: readonly HumanoidExecutionAttribution[],
  value: NonNullable<HumanoidExecutionAttribution["harness"]["rejection_class"]>
): number {
  return entries.filter((entry) => entry.harness.rejection_class === value).length;
}

function parseJournalAction(value: unknown): Receipt {
  const action = record(value);
  if (!action) return PersistedHumanoidActionReceiptSchema.parse(value);
  const { runtime_event_id: _runtimeEventId, ...receipt } = action;
  return PersistedHumanoidActionReceiptSchema.parse(receipt);
}

function nonEmptyString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}

function nonnegativeInteger(value: unknown): number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0
    ? value
    : 0;
}

function countStrings(values: readonly string[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const value of values) counts[value] = (counts[value] ?? 0) + 1;
  return Object.fromEntries(Object.entries(counts).sort(([left], [right]) => (
    left < right ? -1 : left > right ? 1 : 0
  )));
}

function ratio(numerator: number, denominator: number): number | null {
  return denominator === 0
    ? null
    : Math.round(numerator / denominator * 1_000_000) / 1_000_000;
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}
