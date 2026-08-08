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
  HUMANOID_ARTICULATION_HORIZON,
  humanoidArticulationGoal,
  humanoidArticulationGoalSatisfied,
  type HumanoidArticulationGoal
} from "./articulation-control.js";
import { planAutonomousHumanoidSkill } from "./autonomous-skill-planner.js";
import {
  bindHumanoidSkill,
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
  initialPlanId: string;
  initialExecution: HumanoidExecutionReceipt;
  frameSink?: HumanoidFrameSink;
  executionOptions?: HumanoidExecutionOptions;
}): Promise<HumanoidArticulationHorizonResult> {
  const initialArticulation = input.binding.target_articulation;
  if (!initialArticulation) {
    return failureFromExecution(
      input.initialExecution,
      "articulation_horizon_authority_missing",
      [],
      null,
      "The bound Skill does not contain live articulation authority"
    );
  }
  const goal = humanoidArticulationGoal({
    invocation: input.binding.invocation,
    articulation: initialArticulation
  });
  let execution = input.initialExecution;
  let totalFrames = execution.frames;
  let observation = input.world.observe();
  const segments: HumanoidArticulationHorizonSegment[] = [segmentRecord({
    index: 1,
    planId: input.initialPlanId,
    jointId: goal.joint_id,
    beforeRevision: input.binding.observed_world_revision,
    beforePosition: initialArticulation.position,
    observation,
    execution
  })];
  if (!execution.accepted) {
    return aggregateExecution(execution, totalFrames, goal, segments, observation);
  }

  while (!goalSatisfied(input.binding, observation, goal)) {
    if (segments.length >= HUMANOID_ARTICULATION_HORIZON.maximum_segments) {
      return failureFromExecution(
        execution,
        "articulation_horizon_exhausted",
        segments,
        goal,
        "The rolling articulation horizon exhausted its bounded segment budget",
        totalFrames
      );
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
      return failureFromExecution(
        execution,
        "articulation_horizon_rebind_failed",
        segments,
        goal,
        jsonString(rebound.detail),
        totalFrames
      );
    }
    const plan = planAutonomousHumanoidSkill({
      binding: rebound.binding,
      observation,
      articulationGoal: goal
    });
    if (plan.kind !== "motion") {
      return failureFromExecution(
        execution,
        "articulation_horizon_plan_kind_invalid",
        segments,
        goal,
        "Articulation continuation produced a non-motion plan",
        totalFrames
      );
    }
    const planned = await input.world.planWholeBodyMotionCandidates(plan.batch);
    if (!planned.accepted) {
      return failureFromExecution(
        execution,
        "articulation_horizon_planning_failed",
        segments,
        goal,
        JSON.stringify(planned.candidates.map((candidate) => ({
          rank: candidate.rank,
          failures: candidate.validation.failures
        }))),
        totalFrames
      );
    }
    const beforeObservation = observation;
    execution = await input.world.executeWholeBodyMotion(
      planned.planId,
      input.frameSink,
      input.executionOptions
    );
    totalFrames += execution.frames;
    observation = input.world.observe();
    segments.push(segmentRecord({
      index: segments.length + 1,
      planId: planned.planId,
      jointId: goal.joint_id,
      beforeRevision: beforeObservation.worldRevision,
      beforePosition: articulationPosition(input.binding, beforeObservation),
      observation,
      execution
    }));
    if (!execution.accepted) {
      return aggregateExecution(execution, totalFrames, goal, segments, observation);
    }
  }
  return aggregateExecution(execution, totalFrames, goal, segments, observation);
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
  observation: HumanoidWorldObservation
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
        segment_count: segments.length,
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
  totalFrames = execution.frames
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
        segment_count: segments.length,
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
