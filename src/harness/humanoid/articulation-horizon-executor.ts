import type { JsonValue } from "../../domain/schema.js";
import type {
  HumanoidExecutionOptions,
  HumanoidExecutionReceipt,
  HumanoidFrameSink,
  HumanoidWorld,
  HumanoidWorldObservation,
  HumanoidWorldSnapshot
} from "../../world/humanoid/world.js";
import {
  HumanoidEmbodiedSkillStatusSchema,
  type HumanoidEmbodiedSkillStatus
} from "../../world/humanoid/embodied-skill-call.js";
import type { HumanoidSkillEventStream } from
  "../../world/humanoid/skill-event-stream.js";
import {
  HUMANOID_ARTICULATION_HORIZON,
  humanoidArticulationGoal,
  humanoidArticulationGoalSatisfied,
  humanoidArticulationSegmentBudgetExhausted,
  type HumanoidArticulationGoal
} from "./articulation-control.js";
import { planAutonomousHumanoidSkill } from "./autonomous-skill-planner.js";
import {
  bindHumanoidSkill,
  humanoidEmbodiedSkillIdentity,
  type ActiveHumanoidSkillBinding
} from "./skill-binding.js";

interface HumanoidArticulationHorizonSegment {
  index: number;
  plan_id: string;
  world_revision_before: number;
  world_revision_after: number;
  joint_position_before: number | null;
  joint_position_after: number | null;
  accepted: boolean;
  code: string;
  frames: number;
  terminal_result_sha256: string | null;
}

export interface HumanoidArticulationHorizonResult {
  accepted: boolean;
  code: string;
  frames: number;
  finalSnapshot: HumanoidWorldSnapshot;
  terminalResultSha256?: string;
  detail: JsonValue;
}

export function isHumanoidArticulationActuation(
  binding: ActiveHumanoidSkillBinding | undefined
): binding is ActiveHumanoidSkillBinding & {
  invocation: Extract<ActiveHumanoidSkillBinding["invocation"], {
    skill: "open" | "close" | "turn";
  }>;
} {
  return binding?.phase === "actuate_joint"
    && (binding.invocation.skill === "open"
      || binding.invocation.skill === "close"
      || binding.invocation.skill === "turn");
}

export async function executeHumanoidArticulationHorizon(input: {
  world: HumanoidWorld;
  binding: ActiveHumanoidSkillBinding & {
    invocation: Extract<ActiveHumanoidSkillBinding["invocation"], {
      skill: "open" | "close" | "turn";
    }>;
  };
  initialPlanId: string | null;
  initialExecution: HumanoidExecutionReceipt;
  skillEventStream: HumanoidSkillEventStream;
  initialCommittedFrames?: number;
  initialCompletedSegments?: number;
  frameSink?: HumanoidFrameSink;
  executionOptions?: HumanoidExecutionOptions;
}): Promise<HumanoidArticulationHorizonResult> {
  const initialArticulation = input.binding.target_articulation;
  if (!initialArticulation) {
    return finishHorizon(input, failureFromExecution(
      input.initialExecution,
      "articulation_horizon_authority_missing",
      [],
      null,
      "The bound Skill does not contain live articulation authority"
    ), null, input.world.observe());
  }
  const goal = humanoidArticulationGoal({
    invocation: input.binding.invocation,
    articulation: initialArticulation
  });
  let execution = input.initialExecution;
  let totalFrames = (input.initialCommittedFrames ?? 0) + execution.frames;
  const completedSegmentPrefix = input.initialCompletedSegments ?? 0;
  let observation = input.world.observe();
  const segments: HumanoidArticulationHorizonSegment[] = input.initialPlanId
    ? [segmentRecord({
    index: completedSegmentPrefix + 1,
    planId: input.initialPlanId,
    jointId: goal.joint_id,
    beforeRevision: input.binding.observed_world_revision,
    beforePosition: initialArticulation.position,
    observation,
    execution
      })]
    : [];
  if (!execution.accepted) {
    return finishHorizon(
      input,
      aggregateExecution(
        execution,
        totalFrames,
        goal,
        segments,
        observation,
        completedSegmentPrefix
      ),
      goal,
      observation
    );
  }

  while (!goalSatisfied(input.binding, observation, goal)) {
    if (humanoidArticulationSegmentBudgetExhausted(
      completedSegmentPrefix,
      segments.length
    )) {
      return finishHorizon(input, failureFromExecution(
        execution,
        "articulation_horizon_exhausted",
        segments,
        goal,
        "The rolling articulation horizon exhausted its bounded segment budget",
        totalFrames,
        completedSegmentPrefix
      ), goal, observation);
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
      articulationGoal: goal
    });
    if (!rebound.accepted) {
      return finishHorizon(input, failureFromExecution(
        execution,
        "articulation_horizon_rebind_failed",
        segments,
        goal,
        jsonString(rebound.detail),
        totalFrames,
        completedSegmentPrefix
      ), goal, observation);
    }
    let continuationPlanId = input.world.pendingWholeBodyMotionPlanIdForSkillCall(
      humanoidEmbodiedSkillIdentity(input.binding).callId
    );
    if (!continuationPlanId) {
      const plan = planAutonomousHumanoidSkill({
        binding: rebound.binding,
        observation,
        articulationGoal: goal
      });
      if (plan.kind !== "motion") {
        return finishHorizon(input, failureFromExecution(
          execution,
          "articulation_horizon_plan_kind_invalid",
          segments,
          goal,
          "Articulation continuation produced a non-motion plan",
          totalFrames,
          completedSegmentPrefix
        ), goal, observation);
      }
      const planned = await input.world.planWholeBodyMotionCandidates(plan.batch, {
        skillCallIdentity: humanoidEmbodiedSkillIdentity(input.binding)
      });
      if (!planned.accepted) {
        return finishHorizon(input, failureFromExecution(
          execution,
          "articulation_horizon_planning_failed",
          segments,
          goal,
          JSON.stringify(planned.candidates.map((candidate) => ({
            rank: candidate.rank,
            failures: candidate.validation.failures
          }))),
          totalFrames,
          completedSegmentPrefix
        ), goal, observation);
      }
      continuationPlanId = planned.planId;
    }
    const beforeObservation = observation;
    execution = await input.world.executeWholeBodyMotion(
      continuationPlanId,
      input.frameSink,
      {
        ...input.executionOptions,
        skillWindow: {
          maximumSteps: HUMANOID_ARTICULATION_HORIZON.maximum_control_steps,
          stepOffset: totalFrames
        }
      }
    );
    totalFrames += execution.frames;
    observation = input.world.observe();
    segments.push(segmentRecord({
      index: completedSegmentPrefix + segments.length + 1,
      planId: continuationPlanId,
      jointId: goal.joint_id,
      beforeRevision: beforeObservation.worldRevision,
      beforePosition: articulationPosition(input.binding, beforeObservation),
      observation,
      execution
    }));
    await input.skillEventStream.progress(articulationHorizonStatus(
      input.binding,
      aggregateExecution(
        execution,
        totalFrames,
        goal,
        segments,
        observation,
        completedSegmentPrefix
      ),
      goal,
      observation,
      false
    ));
    if (!execution.accepted) {
      return finishHorizon(
        input,
        aggregateExecution(
          execution,
          totalFrames,
          goal,
          segments,
          observation,
          completedSegmentPrefix
        ),
        goal,
        observation
      );
    }
  }
  return finishHorizon(
    input,
    aggregateExecution(
      execution,
      totalFrames,
      goal,
      segments,
      observation,
      completedSegmentPrefix
    ),
    goal,
    observation
  );
}

async function finishHorizon(
  input: Pick<
    Parameters<typeof executeHumanoidArticulationHorizon>[0],
    "world" | "binding" | "skillEventStream"
  >,
  result: HumanoidArticulationHorizonResult,
  goal: HumanoidArticulationGoal | null,
  observation: HumanoidWorldObservation
): Promise<HumanoidArticulationHorizonResult> {
  const status = articulationHorizonStatus(
    input.binding,
    result,
    goal,
    observation
  );
  input.world.recordSkillOutcome({
    protocol: "humanoid-controller-skill-outcome-v1",
    identity: humanoidEmbodiedSkillIdentity(input.binding),
    outcome: status.state === "succeeded" ? "succeeded" : "failed",
    terminalReason: result.code
  });
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

function articulationHorizonStatus(
  binding: ActiveHumanoidSkillBinding,
  result: HumanoidArticulationHorizonResult,
  goal: HumanoidArticulationGoal | null,
  observation: HumanoidWorldObservation,
  terminal = true
): HumanoidEmbodiedSkillStatus {
  const articulation = goal ? articulationForGoal(observation, goal) : null;
  const completion = goal && articulation && articulation.position !== null
    ? directedCompletion(goal, articulation.position)
    : 0;
  const succeeded = terminal && result.accepted
    && goal !== null
    && humanoidArticulationGoalSatisfied(goal, articulation);
  const controller = result.finalSnapshot.robot.controllerExecution;
  return HumanoidEmbodiedSkillStatusSchema.parse({
    protocol: "humanoid-embodied-skill-status-v1",
    callId: humanoidEmbodiedSkillIdentity(binding).callId,
    state: terminal ? succeeded ? "succeeded" : "failed" : "executing",
    progress: {
      elapsedRatio: Math.min(
        1,
        result.frames / HUMANOID_ARTICULATION_HORIZON.maximum_control_steps
      ),
      physicalCompletionRatio: succeeded ? 1 : completion,
      satisfiedPredicateRatio: succeeded ? 1 : 0,
      stableSteps: succeeded ? 1 : 0,
      requiredStableSteps: 1
    },
    confidence: {
      value: articulation?.position === null || articulation === null ? 0 : 1,
      basis: "observable_contract_evidence"
    },
    failure: terminal && !succeeded ? {
      code: result.code,
      detail: failureReason(result.detail)
    } : null,
    recoverability: terminal && !succeeded ? "replan" : "not_applicable",
    worldFrame: result.finalSnapshot.frame,
    worldRevision: result.finalSnapshot.worldRevision,
    controller: controller ? {
      mode: controller.mode,
      implementation: controller.activeImplementation
    } : null
  });
}

function articulationForGoal(
  observation: HumanoidWorldObservation,
  goal: HumanoidArticulationGoal
) {
  return observation.interaction.object_world_model.objects
    .flatMap(({ articulation }) => articulation ? [articulation] : [])
    .find(({ joint_id }) => joint_id === goal.joint_id) ?? null;
}

function directedCompletion(
  goal: HumanoidArticulationGoal,
  position: number
): number {
  const span = goal.target_position - goal.origin_position;
  if (Math.abs(span) <= 1e-12) return 0;
  return Math.max(0, Math.min(1, (position - goal.origin_position) / span));
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

function goalSatisfied(
  binding: ActiveHumanoidSkillBinding,
  observation: HumanoidWorldObservation,
  goal: HumanoidArticulationGoal
): boolean {
  return humanoidArticulationGoalSatisfied(
    goal,
    articulation(binding, observation)
  );
}

function articulation(
  binding: ActiveHumanoidSkillBinding,
  observation: HumanoidWorldObservation
) {
  if (!("object_id" in binding.invocation)) return null;
  const objectId = binding.invocation.object_id;
  return observation.interaction.object_world_model.objects.find(
    ({ id }) => id === objectId
  )?.articulation ?? null;
}

function articulationPosition(
  binding: ActiveHumanoidSkillBinding,
  observation: HumanoidWorldObservation
): number | null {
  return articulation(binding, observation)?.position ?? null;
}

function segmentRecord(input: {
  index: number;
  planId: string;
  jointId: string;
  beforeRevision: number;
  beforePosition: number | null;
  observation: HumanoidWorldObservation;
  execution: HumanoidExecutionReceipt;
}): HumanoidArticulationHorizonSegment {
  const worldModel = input.observation.interaction.object_world_model;
  const articulationPositionAfter = worldModel.objects
    .flatMap(({ articulation }) => articulation ? [articulation] : [])
    .find((candidate) => candidate.joint_id === input.jointId)
    ?.position ?? null;
  return {
    index: input.index,
    plan_id: input.planId,
    world_revision_before: input.beforeRevision,
    world_revision_after: input.observation.worldRevision,
    joint_position_before: input.beforePosition,
    joint_position_after: articulationPositionAfter,
    accepted: input.execution.accepted,
    code: input.execution.code,
    frames: input.execution.frames,
    terminal_result_sha256: input.execution.terminalResultSha256 ?? null
  };
}

function aggregateExecution(
  execution: HumanoidExecutionReceipt,
  totalFrames: number,
  goal: HumanoidArticulationGoal,
  segments: HumanoidArticulationHorizonSegment[],
  observation: HumanoidWorldObservation,
  completedSegmentPrefix = 0
): HumanoidArticulationHorizonResult {
  return {
    accepted: execution.accepted
      && humanoidArticulationGoalSatisfied(
        goal,
        observation.interaction.object_world_model.objects
          .flatMap(({ articulation }) => articulation ? [articulation] : [])
          .find(({ joint_id }) => joint_id === goal.joint_id)
      ),
    code: execution.accepted ? "motion_option_succeeded" : execution.code,
    frames: totalFrames,
    finalSnapshot: execution.finalSnapshot,
    ...(execution.terminalResultSha256
      ? { terminalResultSha256: execution.terminalResultSha256 }
      : {}),
    detail: jsonValue({
      ...execution.detail,
      articulation_horizon: {
        protocol: "humanoid-articulation-horizon-v1",
        goal,
        completed: humanoidArticulationGoalSatisfied(
          goal,
          observation.interaction.object_world_model.objects
            .flatMap(({ articulation }) => articulation ? [articulation] : [])
            .find(({ joint_id }) => joint_id === goal.joint_id)
        ),
        segment_count: completedSegmentPrefix + segments.length,
        completed_segment_prefix_count: completedSegmentPrefix,
        segments
      }
    })
  };
}

function failureFromExecution(
  execution: HumanoidExecutionReceipt,
  code: string,
  segments: HumanoidArticulationHorizonSegment[],
  goal: HumanoidArticulationGoal | null,
  reason: string,
  totalFrames = execution.frames,
  completedSegmentPrefix = 0
): HumanoidArticulationHorizonResult {
  return {
    accepted: false,
    code,
    frames: totalFrames,
    finalSnapshot: execution.finalSnapshot,
    ...(execution.terminalResultSha256
      ? { terminalResultSha256: execution.terminalResultSha256 }
      : {}),
    detail: jsonValue({
      ...execution.detail,
      reason,
      articulation_horizon: {
        protocol: "humanoid-articulation-horizon-v1",
        goal,
        completed: false,
        segment_count: completedSegmentPrefix + segments.length,
        completed_segment_prefix_count: completedSegmentPrefix,
        segments
      }
    })
  };
}

function jsonString(value: JsonValue): string {
  return typeof value === "string" ? value : JSON.stringify(value);
}

function jsonValue(value: unknown): JsonValue {
  return JSON.parse(JSON.stringify(value)) as JsonValue;
}
