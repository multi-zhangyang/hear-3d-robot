import type { Goal, JsonValue, Vec3 } from "../../domain/schema.js";
import type {
  HumanoidExecutionOptions,
  HumanoidExecutionReceipt,
  HumanoidFrameSink,
  HumanoidWorld,
  HumanoidWorldObservation,
  HumanoidWorldSnapshot,
  NavigationPlanReceipt
} from "../../world/humanoid/world.js";
import {
  HumanoidEmbodiedSkillStatusSchema,
  type HumanoidEmbodiedSkillStatus
} from "../../world/humanoid/embodied-skill-call.js";
import type { HumanoidSkillEventStream } from
  "../../world/humanoid/skill-event-stream.js";
import { humanoidCarriedObjectBindingSetSha256 } from
  "../../world/humanoid/carried-object-binding.js";
import { planAutonomousHumanoidSkill } from "./autonomous-skill-planner.js";
import { planAutonomousHumanoidNavigation } from
  "./autonomous-navigation-planning.js";
import {
  HUMANOID_NAVIGATION_HORIZON,
  humanoidNavigationSegmentBudgetExhausted
} from "./navigation-control.js";
import {
  bindHumanoidSkill,
  humanoidEmbodiedSkillIdentity,
  type ActiveHumanoidSkillBinding
} from "./skill-binding.js";

interface HumanoidNavigationHorizonSegment {
  index: number;
  plan_id: string;
  world_revision_before: number;
  world_revision_after: number;
  root_position_before: Vec3;
  root_position_after: Vec3;
  requested_target: Vec3;
  chunk_target: Vec3;
  remaining_distance_before_m: number;
  accepted: boolean;
  code: string;
  frames: number;
  terminal_result_sha256: string | null;
}

export interface HumanoidNavigationHorizonResult {
  accepted: boolean;
  code: string;
  frames: number;
  finalSnapshot: HumanoidWorldSnapshot;
  terminalResultSha256?: string;
  detail: JsonValue;
}

export function isHumanoidNavigationSkill(
  binding: ActiveHumanoidSkillBinding | undefined
): binding is ActiveHumanoidSkillBinding {
  return binding?.phase_authority === "navigation";
}

/**
 * Owns a complete semantic navigation phase after one Agent decision and one
 * Execution Gate admission. Individual 3 m routes remain independently
 * planned, previewed, persisted, and safety-checked, but they no longer leak
 * into the model loop as requests to repeat the same Skill decision.
 */
export async function executeHumanoidNavigationHorizon(input: {
  world: HumanoidWorld;
  binding: ActiveHumanoidSkillBinding;
  initialPlan: NavigationPlanReceipt | null;
  initialExecution: HumanoidExecutionReceipt;
  initialRootPosition: Vec3;
  skillEventStream: HumanoidSkillEventStream;
  initialCommittedFrames?: number;
  initialCompletedSegments?: number;
  frameSink?: HumanoidFrameSink;
  executionOptions?: HumanoidExecutionOptions;
  activeGoal?: Goal;
  recoveryAuthorized?: boolean;
}): Promise<HumanoidNavigationHorizonResult> {
  let execution = input.initialExecution;
  let totalFrames = (input.initialCommittedFrames ?? 0) + execution.frames;
  const completedSegmentPrefix = input.initialCompletedSegments ?? 0;
  let observation = input.world.observe();
  let latestPlan = input.initialPlan;
  const segments: HumanoidNavigationHorizonSegment[] = latestPlan
    ? [navigationSegment({
        index: completedSegmentPrefix + 1,
        plan: latestPlan,
        beforeRevision: input.binding.observed_world_revision,
        beforePosition: input.initialRootPosition,
        observation,
        execution
      })]
    : [];

  if (!execution.accepted) {
    return finishNavigationHorizon(input, aggregateNavigationExecution({
      execution,
      totalFrames,
      completedSegmentPrefix,
      segments,
      observation,
      completed: false
    }), observation, input.initialRootPosition, latestPlan?.target ?? null);
  }

  while (!navigationPhaseSatisfied(input.binding, observation, latestPlan)) {
    if (humanoidNavigationSegmentBudgetExhausted(
      completedSegmentPrefix,
      segments.length
    )) {
      return finishNavigationHorizon(input, navigationFailure({
        execution,
        code: "navigation_horizon_exhausted",
        reason: "The autonomous navigation Skill exhausted its bounded route-segment horizon",
        totalFrames,
        completedSegmentPrefix,
        segments,
        observation
      }), observation, input.initialRootPosition, latestPlan?.target ?? null);
    }

    const rebound = bindHumanoidSkill({
      transactionId: input.binding.transaction_id,
      agentId: input.binding.agent_id,
      request: {
        skill_plan_transaction_id: input.binding.skill_plan_transaction_id,
        skill_node_id: input.binding.skill_node_id,
        invocation: input.binding.invocation,
        phase: input.binding.phase
      },
      observation,
      ...(input.activeGoal ? { activeGoal: input.activeGoal } : {}),
      ...(input.recoveryAuthorized ? { recoveryAuthorized: true } : {})
    });
    if (!rebound.accepted) {
      return finishNavigationHorizon(input, navigationFailure({
        execution,
        code: "navigation_horizon_rebind_failed",
        reason: jsonString(rebound.detail),
        totalFrames,
        completedSegmentPrefix,
        segments,
        observation
      }), observation, input.initialRootPosition, latestPlan?.target ?? null);
    }

    let continuationPlanId = input.world.pendingNavigationPlanIdForSkillCall(
      humanoidEmbodiedSkillIdentity(input.binding).callId
    );
    let continuationPlan: NavigationPlanReceipt | null = null;
    if (!continuationPlanId) {
      let semanticPlan;
      try {
        semanticPlan = planAutonomousHumanoidSkill({
          binding: rebound.binding,
          observation,
          ...(input.activeGoal ? { activeGoal: input.activeGoal } : {}),
          ...(input.recoveryAuthorized ? { recoveryAuthorized: true } : {})
        });
      } catch (error) {
        return finishNavigationHorizon(input, navigationFailure({
          execution,
          code: "navigation_horizon_solver_failed",
          reason: error instanceof Error ? error.message : String(error),
          totalFrames,
          completedSegmentPrefix,
          segments,
          observation
        }), observation, input.initialRootPosition, latestPlan?.target ?? null);
      }
      if (semanticPlan.kind !== "navigation") {
        return finishNavigationHorizon(input, navigationFailure({
          execution,
          code: "navigation_horizon_plan_kind_invalid",
          reason: "Navigation continuation produced a non-navigation plan",
          totalFrames,
          completedSegmentPrefix,
          segments,
          observation
        }), observation, input.initialRootPosition, latestPlan?.target ?? null);
      }
      const planned = await planAutonomousHumanoidNavigation({
        world: input.world,
        binding: rebound.binding,
        plan: semanticPlan
      });
      if (!planned.selected) {
        return finishNavigationHorizon(input, navigationFailure({
          execution,
          code: "navigation_horizon_planning_failed",
          reason: JSON.stringify(planned.attempts),
          totalFrames,
          completedSegmentPrefix,
          segments,
          observation
        }), observation, input.initialRootPosition, latestPlan?.target ?? null);
      }
      continuationPlan = planned.selected;
      continuationPlanId = planned.selected.planId;
    }

    const beforeObservation = observation;
    execution = await input.world.executeNavigation(
      continuationPlanId,
      input.frameSink,
      {
        ...input.executionOptions,
        skillWindow: {
          maximumSteps: HUMANOID_NAVIGATION_HORIZON.maximum_control_steps,
          stepOffset: totalFrames
        }
      }
    );
    totalFrames += execution.frames;
    observation = input.world.observe();
    latestPlan = continuationPlan ?? navigationPlanForHorizon(
      input.world,
      continuationPlanId
    );
    if (!latestPlan) {
      return finishNavigationHorizon(input, navigationFailure({
        execution,
        code: "navigation_horizon_plan_state_missing",
        reason: `Navigation continuation ${continuationPlanId} has no recoverable plan state`,
        totalFrames,
        completedSegmentPrefix,
        segments,
        observation
      }), observation, input.initialRootPosition, null);
    }
    segments.push(navigationSegment({
      index: completedSegmentPrefix + segments.length + 1,
      plan: latestPlan,
      beforeRevision: beforeObservation.worldRevision,
      beforePosition: beforeObservation.robot.rootPosition,
      observation,
      execution
    }));
    const completed = navigationPhaseSatisfied(
      input.binding,
      observation,
      latestPlan
    );
    if (execution.accepted && !completed && execution.frames <= 0) {
      return finishNavigationHorizon(input, navigationFailure({
        execution,
        code: "navigation_horizon_no_progress",
        reason: "A navigation route completed without a committed physical frame or semantic completion",
        totalFrames,
        completedSegmentPrefix,
        segments,
        observation
      }), observation, input.initialRootPosition, latestPlan.target);
    }
    await input.skillEventStream.progress(navigationHorizonStatus({
      binding: input.binding,
      result: aggregateNavigationExecution({
        execution,
        totalFrames,
        completedSegmentPrefix,
        segments,
        observation,
        completed
      }),
      observation,
      initialRootPosition: input.initialRootPosition,
      target: latestPlan.target,
      terminal: false
    }));
    if (!execution.accepted) {
      return finishNavigationHorizon(input, aggregateNavigationExecution({
        execution,
        totalFrames,
        completedSegmentPrefix,
        segments,
        observation,
        completed: false
      }), observation, input.initialRootPosition, latestPlan.target);
    }
  }

  return finishNavigationHorizon(input, aggregateNavigationExecution({
    execution,
    totalFrames,
    completedSegmentPrefix,
    segments,
    observation,
    completed: true
  }), observation, input.initialRootPosition, latestPlan?.target ?? null);
}

async function finishNavigationHorizon(
  input: Pick<
    Parameters<typeof executeHumanoidNavigationHorizon>[0],
    "world" | "binding" | "skillEventStream" | "executionOptions"
  >,
  result: HumanoidNavigationHorizonResult,
  observation: HumanoidWorldObservation,
  initialRootPosition: Vec3,
  target: Vec3 | null
): Promise<HumanoidNavigationHorizonResult> {
  const status = navigationHorizonStatus({
    binding: input.binding,
    result,
    observation,
    initialRootPosition,
    target,
    terminal: true
  });
  input.world.recordSkillOutcome({
    protocol: "humanoid-controller-skill-outcome-v1",
    identity: humanoidEmbodiedSkillIdentity(input.binding),
    outcome: status.state === "succeeded" ? "succeeded" : "failed",
    terminalReason: result.code
  });
  // Capability posterior updates are controller authority, even though they
  // do not advance MuJoCo by another frame. Commit that terminal-boundary
  // state through the same physical ledger before returning the Skill result.
  await input.executionOptions?.persistenceSink?.(
    await input.world.capturePersistenceState()
  );
  await input.skillEventStream.terminal(
    status.state === "succeeded" ? "succeeded" : "failed",
    status
  );
  return {
    ...result,
    detail: jsonValue({
      ...jsonRecord(result.detail),
      skill_status: status
    })
  };
}

function navigationHorizonStatus(input: {
  binding: ActiveHumanoidSkillBinding;
  result: HumanoidNavigationHorizonResult;
  observation: HumanoidWorldObservation;
  initialRootPosition: Vec3;
  target: Vec3 | null;
  terminal: boolean;
}): HumanoidEmbodiedSkillStatus {
  const completion = input.target
    ? directedNavigationCompletion(
        input.initialRootPosition,
        input.observation.robot.rootPosition,
        input.target
      )
    : 0;
  const succeeded = input.terminal && input.result.accepted;
  const controller = input.result.finalSnapshot.robot.controllerExecution;
  return HumanoidEmbodiedSkillStatusSchema.parse({
    protocol: "humanoid-embodied-skill-status-v1",
    callId: humanoidEmbodiedSkillIdentity(input.binding).callId,
    state: input.terminal ? succeeded ? "succeeded" : "failed" : "executing",
    progress: {
      elapsedRatio: Math.min(
        1,
        input.result.frames / HUMANOID_NAVIGATION_HORIZON.maximum_control_steps
      ),
      physicalCompletionRatio: succeeded ? 1 : completion,
      satisfiedPredicateRatio: succeeded ? 1 : 0,
      stableSteps: succeeded ? 1 : 0,
      requiredStableSteps: 1
    },
    confidence: { value: 1, basis: "observable_contract_evidence" },
    failure: input.terminal && !succeeded ? {
      code: input.result.code,
      detail: failureReason(input.result.detail)
    } : null,
    recoverability: input.terminal && !succeeded ? "replan" : "not_applicable",
    worldFrame: input.result.finalSnapshot.frame,
    worldRevision: input.result.finalSnapshot.worldRevision,
    controller: controller ? {
      mode: controller.mode,
      implementation: controller.activeImplementation
    } : null
  });
}

function navigationPhaseSatisfied(
  binding: ActiveHumanoidSkillBinding,
  observation: HumanoidWorldObservation,
  latestPlan: NavigationPlanReceipt | null
): boolean {
  const invocation = binding.invocation;
  if (invocation.skill === "navigate_to_zone") {
    return observation.interaction.zones.find(
      ({ zone_id: zoneId }) => zoneId === invocation.zone_id
    )?.robot_inside_horizontal === true;
  }
  if (invocation.skill === "carry_to_zone") {
    const carried = observation.interaction.carrying.bindings.some(
      ({ object_id: objectId }) => objectId === invocation.object_id
    );
    const relation = observation.interaction.manipulable_objects
      .find(({ object_id: objectId }) => objectId === invocation.object_id)
      ?.zone_relations.find(({ zone_id: zoneId }) => zoneId === invocation.zone_id);
    return carried && relation !== undefined
      && relation.minimum_horizontal_clearance_m + invocation.tolerance_m >= 0
      && Math.abs(relation.support_height_error_m)
        <= Math.max(invocation.tolerance_m, 0.025);
  }
  return latestPlan !== null && latestPlan.remainingDistance <= 1e-9;
}

export function navigationPlanForHorizon(
  world: HumanoidWorld,
  planId: string
): NavigationPlanReceipt | null {
  const route = world.checkpoint().routes.find(({ id }) => id === planId);
  if (!route || !route.validatedStateSha256 || route.expiresRevision === undefined
    || !route.intentSha256
    || !route.carriedObjectBindings) return null;
  const remainingDistance = route.remainingDistance ?? Math.hypot(
    route.requestedTarget.x - route.plan.resolvedTarget.x,
    route.requestedTarget.z - route.plan.resolvedTarget.z
  );
  const requestedTarget = { ...route.requestedTarget };
  const chunkTarget = { ...route.plan.resolvedTarget };
  return {
    accepted: true,
    planId: route.id,
    createdRevision: route.createdRevision,
    validatedStateSha256: route.validatedStateSha256,
    expiresRevision: route.expiresRevision,
    intentSha256: route.intentSha256,
    target: requestedTarget,
    chunkTarget,
    requestedArrivalHeading: route.requestedArrivalHeading
      ? structuredClone(route.requestedArrivalHeading)
      : null,
    arrivalHeading: route.arrivalHeading
      ? structuredClone(route.arrivalHeading)
      : null,
    acceptedPositionToleranceMeters: route.acceptedPositionToleranceMeters,
    waypoints: route.plan.waypoints.map((point) => ({ ...point })),
    distance: route.plan.distance,
    remainingDistance,
    carry: {
      binding_set_sha256: humanoidCarriedObjectBindingSetSha256(
        route.carriedObjectBindings
      ),
      bindings: route.carriedObjectBindings.bindings.map(({ object_id, hand }) => ({
        object_id,
        hand
      }))
    }
  };
}

function navigationSegment(input: {
  index: number;
  plan: NavigationPlanReceipt;
  beforeRevision: number;
  beforePosition: Vec3;
  observation: HumanoidWorldObservation;
  execution: HumanoidExecutionReceipt;
}): HumanoidNavigationHorizonSegment {
  return {
    index: input.index,
    plan_id: input.plan.planId,
    world_revision_before: input.beforeRevision,
    world_revision_after: input.observation.worldRevision,
    root_position_before: { ...input.beforePosition },
    root_position_after: { ...input.observation.robot.rootPosition },
    requested_target: { ...input.plan.target },
    chunk_target: { ...input.plan.chunkTarget },
    remaining_distance_before_m: input.plan.remainingDistance,
    accepted: input.execution.accepted,
    code: input.execution.code,
    frames: input.execution.frames,
    terminal_result_sha256: input.execution.terminalResultSha256 ?? null
  };
}

function aggregateNavigationExecution(input: {
  execution: HumanoidExecutionReceipt;
  totalFrames: number;
  completedSegmentPrefix: number;
  segments: HumanoidNavigationHorizonSegment[];
  observation: HumanoidWorldObservation;
  completed: boolean;
}): HumanoidNavigationHorizonResult {
  return {
    accepted: input.execution.accepted && input.completed,
    code: input.execution.accepted && input.completed
      ? "navigation_completed"
      : input.execution.code,
    frames: input.totalFrames,
    finalSnapshot: input.execution.finalSnapshot,
    ...(input.execution.terminalResultSha256
      ? { terminalResultSha256: input.execution.terminalResultSha256 }
      : {}),
    detail: jsonValue({
      ...input.execution.detail,
      navigation_horizon: {
        protocol: "humanoid-navigation-horizon-v1",
        completed: input.completed,
        segment_count: input.completedSegmentPrefix + input.segments.length,
        completed_segment_prefix_count: input.completedSegmentPrefix,
        final_world_revision: input.observation.worldRevision,
        segments: input.segments
      }
    })
  };
}

function navigationFailure(input: {
  execution: HumanoidExecutionReceipt;
  code: string;
  reason: string;
  totalFrames: number;
  completedSegmentPrefix: number;
  segments: HumanoidNavigationHorizonSegment[];
  observation: HumanoidWorldObservation;
}): HumanoidNavigationHorizonResult {
  const result = aggregateNavigationExecution({
    execution: input.execution,
    totalFrames: input.totalFrames,
    completedSegmentPrefix: input.completedSegmentPrefix,
    segments: input.segments,
    observation: input.observation,
    completed: false
  });
  return {
    ...result,
    accepted: false,
    code: input.code,
    detail: jsonValue({ ...jsonRecord(result.detail), reason: input.reason })
  };
}

function directedNavigationCompletion(start: Vec3, current: Vec3, target: Vec3): number {
  const initial = planarDistance(start, target);
  if (initial <= 1e-9) return planarDistance(current, target) <= 1e-9 ? 1 : 0;
  return Math.max(0, Math.min(1, 1 - planarDistance(current, target) / initial));
}

function planarDistance(left: Vec3, right: Vec3): number {
  return Math.hypot(left.x - right.x, left.z - right.z);
}

function failureReason(value: JsonValue): string | null {
  const reason = jsonRecord(value).reason;
  return typeof reason === "string" && reason.trim() ? reason : null;
}

function jsonRecord(value: JsonValue): Record<string, JsonValue> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value
    : {};
}

function jsonString(value: JsonValue): string {
  return typeof value === "string" ? value : JSON.stringify(value);
}

function jsonValue(value: unknown): JsonValue {
  return JSON.parse(JSON.stringify(value)) as JsonValue;
}
