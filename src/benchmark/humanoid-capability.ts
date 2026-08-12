import { z } from "zod";
import {
  PersistedHumanoidActionReceiptSchema
} from "../domain/humanoid-run.js";
import {
  EmptyModelUsageState,
  ModelUsageStateSchema
} from "../domain/model-usage.js";
import {
  PhysicalTrajectorySummarySchema,
  type PhysicalTrajectorySummary
} from "../domain/physical-trajectory.js";
import { RunStatusSchema } from "../domain/schema.js";
import {
  HumanoidPhysicalSafetyEvidenceSchema,
  type HumanoidPhysicalSafetyEvidence
} from "../world/humanoid/physical-safety.js";
import { HumanoidPolicyAdmissionAssessmentSchema } from
  "../world/humanoid/policy-capability-evidence.js";
import {
  attributeHumanoidDualLayerRun,
  type HumanoidDualLayerRunAttribution,
  type HumanoidFailureLayer
} from "./humanoid-dual-layer-attribution.js";

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

const CapabilityPosteriorSchema = z.object({
  outcomes: z.number().int().nonnegative(),
  successes: z.number().int().nonnegative(),
  failures: z.number().int().nonnegative(),
  posteriorMean: z.number().finite().min(0).max(1),
  lowerBound: z.number().finite().min(0).max(1),
  upperBound: z.number().finite().min(0).max(1),
  recentSuccessRate: z.number().finite().min(0).max(1).nullable(),
  transitionAttempts: z.number().int().nonnegative(),
  transitionSuccesses: z.number().int().nonnegative()
}).strict();

const ControllerMemoryBridgeReceiptSchema = z.object({
  protocol: z.literal("humanoid-policy-memory-bridge-v1"),
  phase: z.enum(["guiding", "completed", "timed_out", "aborted"]),
  trigger: z.literal("entry_state_ood"),
  completedSteps: z.number().int().nonnegative(),
  maximumSteps: z.number().int().positive(),
  stableSteps: z.number().int().nonnegative(),
  requiredStableSteps: z.number().int().positive(),
  progress: z.number().finite().min(0).max(1),
  entryStateOodScore: z.number().finite().nonnegative(),
  jointPrototypeRmsError: z.number().finite().nonnegative(),
  maximumJointVelocity: z.number().finite().nonnegative()
}).strict();

const ControllerRoutingReceiptSchema = z.object({
  execution: z.object({
    callId: z.string().trim().min(1),
    route: z.enum(["primary", "fallback", "upper_body_overlay"]),
    assessment: HumanoidPolicyAdmissionAssessmentSchema,
    attribution: z.object({
      primarySteps: z.number().int().nonnegative(),
      fallbackSteps: z.number().int().nonnegative(),
      upperBodyOverlaySteps: z.number().int().nonnegative(),
      memoryBridgeSteps: z.number().int().nonnegative().default(0)
    }).strict(),
    memoryBridge: ControllerMemoryBridgeReceiptSchema.nullable().default(null)
  }).strict().nullable(),
  capability_evidence: z.array(z.object({
    implementation: z.string().trim().min(1),
    skillFamily: z.string().trim().min(1),
    posterior: CapabilityPosteriorSchema
  }).strict())
}).strict();

const BenchmarkRunDefinitionSchema = z.object({
  version: z.literal(1),
  run_id: z.string().trim().min(1),
  scenario_id: z.string().trim().min(1),
  run_mode: z.enum(["mission", "continuous"]).default("mission"),
  controller_source_sha256: z.string().regex(/^[a-f0-9]{64}$/).optional(),
  created_at: z.string().datetime()
}).passthrough();

const BenchmarkCheckpointSchema = z.object({
  runtime: z.literal("humanoid_g1"),
  run_id: z.string().trim().min(1),
  scenario_id: z.string().trim().min(1),
  status: RunStatusSchema,
  world: z.object({
    frame: z.number().int().nonnegative(),
    worldRevision: z.number().int().nonnegative(),
    robot: z.object({
      fallen: z.boolean(),
      controller: z.object({
        implementation: z.string().trim().min(1),
        learnedPolicy: z.object({
          capabilities: z.array(z.string().trim().min(1))
        }).passthrough().nullish()
      }).passthrough(),
      controllerExecution: z.object({
        mode: z.enum(["learned_policy", "reference_control", "hybrid_control"]),
        activeImplementation: z.string().trim().min(1)
      }).passthrough().optional()
    }).passthrough()
  }).passthrough(),
  total_model_calls: z.number().int().nonnegative(),
  model_usage: ModelUsageStateSchema.optional(),
  checker: z.object({ success: z.boolean() }).passthrough().nullable(),
  created_at: z.string().datetime(),
  updated_at: z.string().datetime()
}).passthrough();

export interface HumanoidCapabilityRunInput {
  definition: unknown;
  checkpoint: unknown;
  actions: readonly unknown[];
  events?: readonly unknown[];
  modelUsage?: unknown;
}

export interface HumanoidCapabilityRunMetrics {
  version: 1;
  run_id: string;
  scenario_id: string;
  run_mode: "mission" | "continuous";
  status: z.infer<typeof RunStatusSchema>;
  mission_success: boolean;
  final_fallen: boolean;
  duration_ms: number;
  world_frame: number;
  world_revision: number;
  controller: {
    identity: string;
    source_sha256: string | null;
    implementation: string;
    active_implementation: string;
    learned_capabilities: string[];
  };
  model: {
    calls: number;
    usage_available: boolean;
    requests: number | null;
    input_tokens: number | null;
    output_tokens: number | null;
    total_tokens: number | null;
    cached_input_tokens: number | null;
    reasoning_tokens: number | null;
    by_agent_requests: Record<string, number>;
  };
  actions: {
    total: number;
    accepted: number;
    rejected: number;
    planning_total: number;
    planning_accepted: number;
    planning_rejected: number;
    execution_total: number;
    execution_succeeded: number;
    execution_failed: number;
    execution_frame_count: number;
    online_replans: number;
    online_replans_accepted: number;
    blocking_contact_count: number;
    code_counts: Record<string, number>;
    achieved_predicate_counts: Record<string, number>;
  };
  motion: {
    trajectory_count: number;
    complete_trajectory_count: number;
    observed_frame_count: number;
    root_planar_path_length_m: number;
    controller_usage: {
      available: boolean;
      complete_from_admission: boolean;
      observed_frame_count: number;
      mode_frame_counts: {
        learned_policy: number;
        reference_control: number;
        hybrid_control: number;
      };
      implementation_frame_counts: Record<string, number>;
      transition_frame_count: number;
      learned_policy_frame_ratio: number | null;
      reference_control_frame_ratio: number | null;
      hybrid_control_frame_ratio: number | null;
    };
    routing: {
      decision_count: number;
      admitted_count: number;
      rejected_count: number;
      cold_start_count: number;
      rejection_reason_counts: {
        insufficient_success_posterior: number;
        entry_state_ood: number;
        command_ood: number;
        memory_bridge_timeout: number;
      };
      memory_bridge_attempt_count: number;
      memory_bridge_completed_count: number;
      memory_bridge_timeout_count: number;
      memory_bridge_aborted_count: number;
      memory_bridge_completion_rate: number | null;
      transition_attempts: number;
      transition_successes: number;
      transition_success_rate: number | null;
      skill_families: Record<string, {
        implementation: string;
        outcomes: number;
        successes: number;
        posterior_mean: number;
        lower_bound: number;
        upper_bound: number;
      }>;
    };
  };
  safety: {
    evidence_execution_count: number;
    minimum_support_margin_m: number | null;
    maximum_foot_slip_mps: number | null;
    minimum_joint_limit_margin_rad: number | null;
    maximum_joint_velocity_rad_s: number | null;
    maximum_requested_effort_utilization: number | null;
    saturated_execution_count: number;
    peak_contact_normal_force_n: number | null;
    peak_total_normal_force_n: number | null;
    peak_total_force_rise_rate_nps: number | null;
  };
  attribution: HumanoidDualLayerRunAttribution;
}

interface HumanoidCapabilityBenchmarkSummary {
  run_count: number;
  succeeded_run_count: number;
  success_rate: number | null;
  fallen_run_count: number;
  fall_rate: number | null;
  duration_ms: Distribution;
  model_calls: Distribution;
  total_model_tokens: number | null;
  model_usage_complete: boolean;
  execution_success_rate: number | null;
  planning_rejection_rate: number | null;
  controller_mode_frame_counts: {
    learned_policy: number;
    reference_control: number;
    hybrid_control: number;
  };
  learned_policy_frame_ratio: number | null;
  reference_control_frame_ratio: number | null;
  hybrid_control_frame_ratio: number | null;
  controller_usage_complete: boolean;
  routing_decision_count: number;
  routing_rejection_rate: number | null;
  routing_cold_start_count: number;
  routing_rejection_reason_counts: {
    insufficient_success_posterior: number;
    entry_state_ood: number;
    command_ood: number;
    memory_bridge_timeout: number;
  };
  memory_bridge_attempt_count: number;
  memory_bridge_completion_rate: number | null;
  memory_bridge_timeout_count: number;
  memory_bridge_aborted_count: number;
  transition_success_rate: number | null;
  dual_layer: {
    full_attribution_rate: number | null;
    skill_trigger_rate: number | null;
    skill_compliance_rate: number | null;
    boundary_containment_rate: number | null;
    controller_success_rate_after_admission: number | null;
    semantic_skill_success_rate: number | null;
    primary_only_successes: number;
    fallback_rescued_successes: number;
    upper_body_overlay_successes: number;
    primary_failure_layer_counts: Record<HumanoidFailureLayer, number>;
    mission_failure_unattributed_count: number;
    compact_replan_decisions: number;
    goal_reevaluation_decisions: number;
    replan_model_call_slo_violations: number;
    recovery_deadline_exceeded_run_count: number;
    local_navigation_replans: number;
    local_navigation_model_calls: number;
  };
  worst_safety: HumanoidCapabilityRunMetrics["safety"];
}

interface Distribution {
  minimum: number | null;
  median: number | null;
  p95: number | null;
  maximum: number | null;
  mean: number | null;
}

export interface HumanoidCapabilityBenchmarkReport {
  version: 1;
  generated_at: string;
  run_count: number;
  summary: HumanoidCapabilityBenchmarkSummary;
  groups: Array<{
    scenario_id: string;
    controller_identity: string;
    summary: HumanoidCapabilityBenchmarkSummary;
  }>;
  runs: HumanoidCapabilityRunMetrics[];
}

export function measureHumanoidCapabilityRun(
  input: HumanoidCapabilityRunInput
): HumanoidCapabilityRunMetrics {
  const definition = BenchmarkRunDefinitionSchema.parse(input.definition);
  const checkpoint = BenchmarkCheckpointSchema.parse(input.checkpoint);
  if (definition.run_id !== checkpoint.run_id
    || definition.scenario_id !== checkpoint.scenario_id) {
    throw new Error("Benchmark run definition and checkpoint identities do not match");
  }
  const actions = input.actions.map(parseJournalAction);
  const planning = actions.filter((action) => PlanningActions.has(action.action));
  const executions = actions.filter((action) => ExecutionActions.has(action.action));
  const successfulExecutions = executions.filter((action) => (
    action.accepted && SuccessfulExecutionCodes.has(action.code)
  ));
  const planningById = new Map(planning.map((action) => [action.transactionId, action]));
  const trajectories = executions.flatMap((action) => {
    const candidate = record(action.detail)?.physical_trajectory;
    return candidate === undefined
      ? []
      : [PhysicalTrajectorySummarySchema.parse(candidate)];
  });
  const safetyEvidence = executions.flatMap((action) => {
    const detail = record(action.detail);
    const candidate = record(detail?.result)?.physical_safety
      ?? detail?.physical_safety;
    return candidate === undefined
      ? []
      : [HumanoidPhysicalSafetyEvidenceSchema.parse(candidate)];
  });
  const replans = executions.flatMap((action) => {
    const result = record(record(action.detail)?.result);
    return Array.isArray(result?.online_replans)
      ? result.online_replans.flatMap((candidate) => {
          const parsed = record(candidate);
          return parsed ? [parsed] : [];
        })
      : [];
  });
  const blockingContacts = actions.reduce((total, action) => {
    const detail = record(action.detail);
    const result = record(detail?.result);
    return total + arrayLength(detail?.blocking_contacts)
      + arrayLength(result?.blocking_contacts);
  }, 0);
  const controllerUsage = aggregateControllerUsage(trajectories);
  const controllerRouting = aggregateControllerRouting(executions);
  const controller = checkpoint.world.robot.controller;
  const activeImplementation = checkpoint.world.robot.controllerExecution
    ?.activeImplementation ?? controller.implementation;
  const sourceSha256 = definition.controller_source_sha256 ?? null;
  const implementation = controller.implementation;
  const identity = sourceSha256 ?? implementation;
  const createdAt = Date.parse(checkpoint.created_at);
  const updatedAt = Date.parse(checkpoint.updated_at);
  if (!Number.isFinite(createdAt) || !Number.isFinite(updatedAt) || updatedAt < createdAt) {
    throw new Error("Benchmark checkpoint has an invalid run duration");
  }
  const suppliedUsage = input.modelUsage === undefined
    ? undefined
    : ModelUsageStateSchema.parse(input.modelUsage);
  const modelUsage = checkpoint.model_usage ?? suppliedUsage;
  const usage = modelUsage?.total ?? EmptyModelUsageState.total;
  return {
    version: 1,
    run_id: checkpoint.run_id,
    scenario_id: checkpoint.scenario_id,
    run_mode: definition.run_mode,
    status: checkpoint.status,
    mission_success: checkpoint.status === "succeeded"
      || checkpoint.checker?.success === true,
    final_fallen: checkpoint.world.robot.fallen,
    duration_ms: updatedAt - createdAt,
    world_frame: checkpoint.world.frame,
    world_revision: checkpoint.world.worldRevision,
    controller: {
      identity,
      source_sha256: sourceSha256,
      implementation,
      active_implementation: activeImplementation,
      learned_capabilities: [...new Set(
        controller.learnedPolicy?.capabilities ?? []
      )].sort(compareCodePoints)
    },
    model: {
      calls: checkpoint.total_model_calls,
      usage_available: modelUsage !== undefined,
      requests: modelUsage ? usage.requests : null,
      input_tokens: modelUsage ? usage.input_tokens : null,
      output_tokens: modelUsage ? usage.output_tokens : null,
      total_tokens: modelUsage ? usage.total_tokens : null,
      cached_input_tokens: modelUsage ? usage.cached_input_tokens : null,
      reasoning_tokens: modelUsage ? usage.reasoning_tokens : null,
      by_agent_requests: Object.fromEntries(Object.entries(
        modelUsage?.by_agent ?? {}
      ).map(([agentId, agentUsage]) => [agentId, agentUsage.requests]))
    },
    actions: {
      total: actions.length,
      accepted: actions.filter((action) => action.accepted).length,
      rejected: actions.filter((action) => !action.accepted).length,
      planning_total: planning.length,
      planning_accepted: planning.filter((action) => action.accepted).length,
      planning_rejected: planning.filter((action) => !action.accepted).length,
      execution_total: executions.length,
      execution_succeeded: successfulExecutions.length,
      execution_failed: executions.length - successfulExecutions.length,
      execution_frame_count: executions.reduce(
        (total, action) => total + action.frameCount,
        0
      ),
      online_replans: replans.length,
      online_replans_accepted: replans.filter((replan) => replan.accepted === true).length,
      blocking_contact_count: blockingContacts,
      code_counts: countStrings(actions.map((action) => action.code)),
      achieved_predicate_counts: achievedPredicates(
        successfulExecutions,
        planningById
      )
    },
    motion: {
      trajectory_count: trajectories.length,
      complete_trajectory_count: trajectories.filter(
        (trajectory) => trajectory.complete_from_admission
      ).length,
      observed_frame_count: trajectories.reduce(
        (total, trajectory) => total + trajectory.observed_frame_count,
        0
      ),
      root_planar_path_length_m: rounded(trajectories.reduce(
        (total, trajectory) => total + trajectory.root_planar_path_length_m,
        0
      )),
      controller_usage: controllerUsage,
      routing: controllerRouting
    },
    safety: aggregateSafety(safetyEvidence),
    attribution: attributeHumanoidDualLayerRun({
      status: checkpoint.status,
      actions: input.actions,
      events: input.events ?? []
    })
  };
}

function parseJournalAction(
  value: unknown
): z.infer<typeof PersistedHumanoidActionReceiptSchema> {
  const action = record(value);
  if (!action) return PersistedHumanoidActionReceiptSchema.parse(value);
  const { runtime_event_id: _runtimeEventId, ...receipt } = action;
  return PersistedHumanoidActionReceiptSchema.parse(receipt);
}

export function createHumanoidCapabilityBenchmarkReport(
  runs: readonly HumanoidCapabilityRunMetrics[],
  generatedAt = new Date().toISOString()
): HumanoidCapabilityBenchmarkReport {
  if (runs.length === 0) {
    throw new Error("Capability benchmark requires at least one humanoid run");
  }
  const generated = new Date(generatedAt);
  if (!Number.isFinite(generated.getTime())) {
    throw new Error("Capability benchmark generation time is invalid");
  }
  const sortedRuns = [...runs].sort((left, right) => (
    compareCodePoints(left.run_id, right.run_id)
  ));
  const grouped = new Map<string, HumanoidCapabilityRunMetrics[]>();
  for (const run of sortedRuns) {
    const key = `${run.scenario_id}\0${run.controller.identity}`;
    const current = grouped.get(key) ?? [];
    current.push(run);
    grouped.set(key, current);
  }
  return {
    version: 1,
    generated_at: generated.toISOString(),
    run_count: sortedRuns.length,
    summary: summarizeRuns(sortedRuns),
    groups: [...grouped.values()].map((groupRuns) => ({
      scenario_id: groupRuns[0]!.scenario_id,
      controller_identity: groupRuns[0]!.controller.identity,
      summary: summarizeRuns(groupRuns)
    })).sort((left, right) => compareCodePoints(
      `${left.scenario_id}\0${left.controller_identity}`,
      `${right.scenario_id}\0${right.controller_identity}`
    )),
    runs: sortedRuns
  };
}

function summarizeRuns(
  runs: readonly HumanoidCapabilityRunMetrics[]
): HumanoidCapabilityBenchmarkSummary {
  const succeeded = runs.filter((run) => run.mission_success).length;
  const fallen = runs.filter((run) => run.final_fallen).length;
  const planningTotal = sum(runs.map((run) => run.actions.planning_total));
  const planningRejected = sum(runs.map((run) => run.actions.planning_rejected));
  const executionTotal = sum(runs.map((run) => run.actions.execution_total));
  const executionSucceeded = sum(runs.map((run) => run.actions.execution_succeeded));
  const modes = {
    learned_policy: sum(runs.map(
      (run) => run.motion.controller_usage.mode_frame_counts.learned_policy
    )),
    reference_control: sum(runs.map(
      (run) => run.motion.controller_usage.mode_frame_counts.reference_control
    )),
    hybrid_control: sum(runs.map(
      (run) => run.motion.controller_usage.mode_frame_counts.hybrid_control
    ))
  };
  const controllerFrames = sum(Object.values(modes));
  const usageAvailable = runs.every((run) => run.motion.controller_usage.available);
  const modelUsageComplete = runs.every((run) => run.model.usage_available);
  const routingDecisions = sum(runs.map(
    (run) => run.motion.routing.decision_count
  ));
  const routingRejected = sum(runs.map(
    (run) => run.motion.routing.rejected_count
  ));
  const transitionAttempts = sum(runs.map(
    (run) => run.motion.routing.transition_attempts
  ));
  const transitionSuccesses = sum(runs.map(
    (run) => run.motion.routing.transition_successes
  ));
  const semanticExecutions = sum(runs.map(
    (run) => run.attribution.coverage.semantic_execution_count
  ));
  const fullyAttributedSemanticExecutions = sum(runs.map(
    (run) => run.attribution.coverage.fully_attributed_semantic_execution_count
  ));
  const triggerExpected = sum(runs.map(
    (run) => run.attribution.harness.skill_use.trigger_expected
  ));
  const triggerObserved = sum(runs.map(
    (run) => run.attribution.harness.skill_use.trigger_observed
  ));
  const complianceEvaluated = sum(runs.map(
    (run) => run.attribution.harness.skill_use.compliance_evaluated
  ));
  const compliant = sum(runs.map(
    (run) => run.attribution.harness.skill_use.compliant
  ));
  const boundaryAttempts = sum(runs.map(
    (run) => run.attribution.harness.skill_use.boundary_attempts
  ));
  const boundaryContained = sum(runs.map(
    (run) => run.attribution.harness.skill_use.boundary_contained
  ));
  const physicalExecutions = sum(runs.map(
    (run) => run.attribution.controller.physical_execution_count
  ));
  const controllerSucceeded = sum(runs.map(
    (run) => run.attribution.controller.succeeded_execution_count
  ));
  const semanticSkillOutcomes = sum(runs.map((run) => (
    run.attribution.joint.successful_semantic_skills
      + run.attribution.joint.failed_semantic_skills
  )));
  const semanticSkillSuccesses = sum(runs.map(
    (run) => run.attribution.joint.successful_semantic_skills
  ));
  return {
    run_count: runs.length,
    succeeded_run_count: succeeded,
    success_rate: ratio(succeeded, runs.length),
    fallen_run_count: fallen,
    fall_rate: ratio(fallen, runs.length),
    duration_ms: distribution(runs.map((run) => run.duration_ms)),
    model_calls: distribution(runs.map((run) => run.model.calls)),
    total_model_tokens: modelUsageComplete
      ? sum(runs.map((run) => run.model.total_tokens ?? 0))
      : null,
    model_usage_complete: modelUsageComplete,
    execution_success_rate: ratio(executionSucceeded, executionTotal),
    planning_rejection_rate: ratio(planningRejected, planningTotal),
    controller_mode_frame_counts: modes,
    learned_policy_frame_ratio: usageAvailable
      ? ratio(modes.learned_policy, controllerFrames)
      : null,
    reference_control_frame_ratio: usageAvailable
      ? ratio(modes.reference_control, controllerFrames)
      : null,
    hybrid_control_frame_ratio: usageAvailable
      ? ratio(modes.hybrid_control, controllerFrames)
      : null,
    controller_usage_complete: usageAvailable && runs.every(
      (run) => run.motion.controller_usage.complete_from_admission
    ),
    routing_decision_count: routingDecisions,
    routing_rejection_rate: ratio(routingRejected, routingDecisions),
    routing_cold_start_count: sum(runs.map(
      (run) => run.motion.routing.cold_start_count
    )),
    routing_rejection_reason_counts: {
      insufficient_success_posterior: sum(runs.map(
        (run) => run.motion.routing.rejection_reason_counts
          .insufficient_success_posterior
      )),
      entry_state_ood: sum(runs.map(
        (run) => run.motion.routing.rejection_reason_counts.entry_state_ood
      )),
      command_ood: sum(runs.map(
        (run) => run.motion.routing.rejection_reason_counts.command_ood
      )),
      memory_bridge_timeout: sum(runs.map(
        (run) => run.motion.routing.rejection_reason_counts
          .memory_bridge_timeout
      ))
    },
    memory_bridge_attempt_count: sum(runs.map(
      (run) => run.motion.routing.memory_bridge_attempt_count
    )),
    memory_bridge_completion_rate: ratio(
      sum(runs.map((run) => run.motion.routing.memory_bridge_completed_count)),
      sum(runs.map((run) => run.motion.routing.memory_bridge_attempt_count))
    ),
    memory_bridge_timeout_count: sum(runs.map(
      (run) => run.motion.routing.memory_bridge_timeout_count
    )),
    memory_bridge_aborted_count: sum(runs.map(
      (run) => run.motion.routing.memory_bridge_aborted_count
    )),
    transition_success_rate: ratio(transitionSuccesses, transitionAttempts),
    dual_layer: {
      full_attribution_rate: ratio(
        fullyAttributedSemanticExecutions,
        semanticExecutions
      ),
      skill_trigger_rate: ratio(triggerObserved, triggerExpected),
      skill_compliance_rate: ratio(compliant, complianceEvaluated),
      boundary_containment_rate: ratio(boundaryContained, boundaryAttempts),
      controller_success_rate_after_admission: ratio(
        controllerSucceeded,
        physicalExecutions
      ),
      semantic_skill_success_rate: ratio(
        semanticSkillSuccesses,
        semanticSkillOutcomes
      ),
      primary_only_successes: sum(runs.map(
        (run) => run.attribution.controller.primary_only_successes
      )),
      fallback_rescued_successes: sum(runs.map(
        (run) => run.attribution.controller.fallback_rescued_successes
      )),
      upper_body_overlay_successes: sum(runs.map(
        (run) => run.attribution.controller.upper_body_overlay_successes
      )),
      primary_failure_layer_counts: {
        semantic_planner: sum(runs.map(
          (run) => run.attribution.joint.primary_failure_layer_counts.semantic_planner
        )),
        harness_gate: sum(runs.map(
          (run) => run.attribution.joint.primary_failure_layer_counts.harness_gate
        )),
        controller: sum(runs.map(
          (run) => run.attribution.joint.primary_failure_layer_counts.controller
        )),
        none: sum(runs.map(
          (run) => run.attribution.joint.primary_failure_layer_counts.none
        )),
        unattributed: sum(runs.map(
          (run) => run.attribution.joint.primary_failure_layer_counts.unattributed
        ))
      },
      mission_failure_unattributed_count: runs.filter(
        (run) => run.attribution.joint.mission_failure_unattributed
      ).length,
      compact_replan_decisions: sum(runs.map(
        (run) => run.attribution.harness.recovery.compact_replan_decisions
      )),
      goal_reevaluation_decisions: sum(runs.map(
        (run) => run.attribution.harness.recovery.goal_reevaluation_decisions
      )),
      replan_model_call_slo_violations: sum(runs.map(
        (run) => run.attribution.harness.recovery.model_call_slo_violations
      )),
      recovery_deadline_exceeded_run_count: runs.filter(
        (run) => run.attribution.harness.recovery.recovery_deadline_exceeded
      ).length,
      local_navigation_replans: sum(runs.map(
        (run) => run.attribution.harness.recovery.local_navigation_replans
      )),
      local_navigation_model_calls: sum(runs.map(
        (run) => run.attribution.harness.recovery.local_navigation_model_calls
      ))
    },
    worst_safety: aggregateWorstSafety(runs.map((run) => run.safety))
  };
}

function aggregateControllerUsage(
  trajectories: readonly PhysicalTrajectorySummary[]
): HumanoidCapabilityRunMetrics["motion"]["controller_usage"] {
  const usages = trajectories.flatMap((trajectory) => (
    trajectory.controller_usage ? [trajectory.controller_usage] : []
  ));
  const modes = {
    learned_policy: sum(usages.map((usage) => usage.mode_frame_counts.learned_policy)),
    reference_control: sum(usages.map(
      (usage) => usage.mode_frame_counts.reference_control
    )),
    hybrid_control: sum(usages.map((usage) => usage.mode_frame_counts.hybrid_control))
  };
  const observed = sum(usages.map((usage) => usage.observed_frame_count));
  const implementations: Record<string, number> = {};
  for (const usage of usages) {
    for (const [implementation, count] of Object.entries(
      usage.implementation_frame_counts
    )) {
      implementations[implementation] = (implementations[implementation] ?? 0) + count;
    }
  }
  const available = trajectories.length > 0 && usages.length === trajectories.length;
  return {
    available,
    complete_from_admission: available && usages.every(
      (usage) => usage.complete_from_admission
    ),
    observed_frame_count: observed,
    mode_frame_counts: modes,
    implementation_frame_counts: sortedRecord(implementations),
    transition_frame_count: sum(usages.map((usage) => usage.transition_frame_count)),
    learned_policy_frame_ratio: available ? ratio(modes.learned_policy, observed) : null,
    reference_control_frame_ratio: available
      ? ratio(modes.reference_control, observed)
      : null,
    hybrid_control_frame_ratio: available ? ratio(modes.hybrid_control, observed) : null
  };
}

function aggregateControllerRouting(
  executions: readonly ReturnType<typeof parseJournalAction>[]
): HumanoidCapabilityRunMetrics["motion"]["routing"] {
  const receipts = executions.flatMap((action) => {
    const detail = record(action.detail);
    const result = record(detail?.result);
    const candidate = result?.controller_routing ?? detail?.controller_routing;
    return candidate === undefined
      ? []
      : [ControllerRoutingReceiptSchema.parse(candidate)];
  });
  const decisions = receipts.flatMap((receipt) => (
    receipt.execution ? [receipt.execution.assessment] : []
  ));
  const memoryBridges = receipts.flatMap((receipt) => (
    receipt.execution?.memoryBridge ? [receipt.execution.memoryBridge] : []
  ));
  const evidence = new Map<
    string,
    z.infer<typeof ControllerRoutingReceiptSchema>["capability_evidence"][number]
  >();
  for (const receipt of receipts) {
    for (const candidate of receipt.capability_evidence) {
      const key = `${candidate.implementation}\0${candidate.skillFamily}`;
      const current = evidence.get(key);
      if (!current
        || candidate.posterior.outcomes > current.posterior.outcomes
        || (candidate.posterior.outcomes === current.posterior.outcomes
          && candidate.posterior.transitionAttempts
            > current.posterior.transitionAttempts)) {
        evidence.set(key, candidate);
      }
    }
  }
  const rejected = decisions.filter((decision) => !decision.admitted);
  const transitionAttempts = sum([...evidence.values()].map(
    (entry) => entry.posterior.transitionAttempts
  ));
  const transitionSuccesses = sum([...evidence.values()].map(
    (entry) => entry.posterior.transitionSuccesses
  ));
  return {
    decision_count: decisions.length,
    admitted_count: decisions.filter((decision) => decision.admitted).length,
    rejected_count: rejected.length,
    cold_start_count: decisions.filter((decision) => decision.coldStart).length,
    rejection_reason_counts: {
      insufficient_success_posterior: rejected.filter(
        (decision) => decision.reason === "insufficient_success_posterior"
      ).length,
      entry_state_ood: rejected.filter(
        (decision) => decision.reason === "entry_state_ood"
      ).length,
      command_ood: rejected.filter(
        (decision) => decision.reason === "command_ood"
      ).length,
      memory_bridge_timeout: rejected.filter(
        (decision) => decision.reason === "memory_bridge_timeout"
      ).length
    },
    memory_bridge_attempt_count: memoryBridges.length,
    memory_bridge_completed_count: memoryBridges.filter(
      (bridge) => bridge.phase === "completed"
    ).length,
    memory_bridge_timeout_count: memoryBridges.filter(
      (bridge) => bridge.phase === "timed_out"
    ).length,
    memory_bridge_aborted_count: memoryBridges.filter(
      (bridge) => bridge.phase === "aborted"
    ).length,
    memory_bridge_completion_rate: ratio(
      memoryBridges.filter((bridge) => bridge.phase === "completed").length,
      memoryBridges.length
    ),
    transition_attempts: transitionAttempts,
    transition_successes: transitionSuccesses,
    transition_success_rate: ratio(transitionSuccesses, transitionAttempts),
    skill_families: sortedRecord(Object.fromEntries([...evidence.values()].map(
      (entry) => [
        `${entry.implementation}/${entry.skillFamily}`,
        {
          implementation: entry.implementation,
          outcomes: entry.posterior.outcomes,
          successes: entry.posterior.successes,
          posterior_mean: entry.posterior.posteriorMean,
          lower_bound: entry.posterior.lowerBound,
          upper_bound: entry.posterior.upperBound
        }
      ]
    )))
  };
}

function aggregateSafety(
  evidence: readonly HumanoidPhysicalSafetyEvidence[]
): HumanoidCapabilityRunMetrics["safety"] {
  return {
    evidence_execution_count: evidence.length,
    minimum_support_margin_m: minimum(evidence.flatMap((item) => (
      item.minimum_signed_support_margin
        ? [item.minimum_signed_support_margin.signed_margin_m]
        : []
    ))),
    maximum_foot_slip_mps: maximum(evidence.flatMap((item) => (
      item.maximum_foot_tangential_speed
        ? [item.maximum_foot_tangential_speed.tangential_speed_mps]
        : []
    ))),
    minimum_joint_limit_margin_rad: minimum(evidence.map(
      (item) => item.minimum_joint_limit_margin.margin_rad
    )),
    maximum_joint_velocity_rad_s: maximum(evidence.map(
      (item) => item.maximum_joint_velocity.absolute_velocity_rad_s
    )),
    maximum_requested_effort_utilization: maximum(evidence.flatMap((item) => (
      item.maximum_actuator_effort_utilization
        ? [item.maximum_actuator_effort_utilization.requested_utilization]
        : []
    ))),
    saturated_execution_count: evidence.filter(
      (item) => item.maximum_actuator_effort_utilization?.saturated === true
    ).length,
    peak_contact_normal_force_n: maximum(evidence.flatMap((item) => (
      item.peak_contact_normal_force
        ? [item.peak_contact_normal_force.contact.normal_force_n]
        : []
    ))),
    peak_total_normal_force_n: maximum(evidence.map(
      (item) => item.peak_total_normal_force.total_normal_force_n
    )),
    peak_total_force_rise_rate_nps: maximum(evidence.flatMap((item) => (
      item.peak_total_normal_force_rise_rate
        ? [item.peak_total_normal_force_rise_rate.rise_rate_nps]
        : []
    )))
  };
}

function aggregateWorstSafety(
  evidence: readonly HumanoidCapabilityRunMetrics["safety"][]
): HumanoidCapabilityRunMetrics["safety"] {
  return {
    evidence_execution_count: sum(evidence.map((item) => item.evidence_execution_count)),
    minimum_support_margin_m: minimum(nonNull(evidence.map(
      (item) => item.minimum_support_margin_m
    ))),
    maximum_foot_slip_mps: maximum(nonNull(evidence.map(
      (item) => item.maximum_foot_slip_mps
    ))),
    minimum_joint_limit_margin_rad: minimum(nonNull(evidence.map(
      (item) => item.minimum_joint_limit_margin_rad
    ))),
    maximum_joint_velocity_rad_s: maximum(nonNull(evidence.map(
      (item) => item.maximum_joint_velocity_rad_s
    ))),
    maximum_requested_effort_utilization: maximum(nonNull(evidence.map(
      (item) => item.maximum_requested_effort_utilization
    ))),
    saturated_execution_count: sum(evidence.map(
      (item) => item.saturated_execution_count
    )),
    peak_contact_normal_force_n: maximum(nonNull(evidence.map(
      (item) => item.peak_contact_normal_force_n
    ))),
    peak_total_normal_force_n: maximum(nonNull(evidence.map(
      (item) => item.peak_total_normal_force_n
    ))),
    peak_total_force_rise_rate_nps: maximum(nonNull(evidence.map(
      (item) => item.peak_total_force_rise_rate_nps
    )))
  };
}

function achievedPredicates(
  executions: readonly z.infer<typeof PersistedHumanoidActionReceiptSchema>[],
  planningById: ReadonlyMap<
    string,
    z.infer<typeof PersistedHumanoidActionReceiptSchema>
  >
): Record<string, number> {
  const types: string[] = [];
  for (const execution of executions) {
    const planningId = record(execution.input)?.planning_transaction_id;
    if (typeof planningId !== "string") continue;
    const planning = planningById.get(planningId);
    if (!planning) continue;
    for (const predicate of terminationPredicates(planning.detail)) {
      if (typeof predicate.type === "string" && predicate.type) {
        types.push(predicate.type);
      }
    }
  }
  return countStrings(types);
}

function terminationPredicates(value: unknown): Record<string, unknown>[] {
  const detail = record(value);
  const direct = record(detail?.termination);
  const option = record(detail?.option);
  const contract = record(option?.contract);
  const optionTermination = record(contract?.termination);
  for (const candidate of [direct?.predicates, optionTermination?.predicates]) {
    if (Array.isArray(candidate)) return candidate.flatMap((entry) => {
      const parsed = record(entry);
      return parsed ? [parsed] : [];
    });
  }
  return [];
}

function distribution(values: readonly number[]): Distribution {
  if (values.length === 0) {
    return { minimum: null, median: null, p95: null, maximum: null, mean: null };
  }
  const sorted = [...values].sort((left, right) => left - right);
  return {
    minimum: sorted[0]!,
    median: percentile(sorted, 0.5),
    p95: percentile(sorted, 0.95),
    maximum: sorted.at(-1)!,
    mean: rounded(sum(sorted) / sorted.length)
  };
}

function percentile(sorted: readonly number[], probability: number): number {
  const index = Math.max(0, Math.ceil(sorted.length * probability) - 1);
  return sorted[index]!;
}

function countStrings(values: readonly string[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const value of values) counts[value] = (counts[value] ?? 0) + 1;
  return sortedRecord(counts);
}

function sortedRecord<T>(values: Record<string, T>): Record<string, T> {
  return Object.fromEntries(Object.entries(values).sort(([left], [right]) => (
    compareCodePoints(left, right)
  )));
}

function ratio(numerator: number, denominator: number): number | null {
  return denominator === 0 ? null : rounded(numerator / denominator);
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function minimum(values: readonly number[]): number | null {
  return values.length === 0 ? null : Math.min(...values);
}

function maximum(values: readonly number[]): number | null {
  return values.length === 0 ? null : Math.max(...values);
}

function nonNull(values: readonly (number | null)[]): number[] {
  return values.filter((value): value is number => value !== null);
}

function arrayLength(value: unknown): number {
  return Array.isArray(value) ? value.length : 0;
}

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

function rounded(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}

function compareCodePoints(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}
