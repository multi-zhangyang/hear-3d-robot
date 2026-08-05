import type {
  Goal,
  JsonValue,
  Scenario,
  TaskNode
} from "../../domain/schema.js";
import type { HumanoidRunMode } from "../../domain/run-mode.js";
import type { HumanoidRunCheckpoint } from "../../domain/humanoid-run.js";
import { sameAutonomousCycle } from "../../domain/autonomous-cycle.js";
import { inspectHumanoidGoal } from "../../runtime/humanoid-checker.js";
import type {
  HumanoidWorldObservation,
  HumanoidWorldSnapshot
} from "../../world/humanoid/world.js";
import {
  recentEmbodiedEpisodes,
  recentEmbodiedExperiences
} from "./embodied-memory.js";
import {
  createWorldGoalEvidence,
  type GoalEvidenceArtifact
} from "./goal-evidence.js";
import type { HumanoidCycleCompletionReadiness } from "./cycle-causal-evidence.js";
import type { HumanoidCoordinatorPhase } from "./run-runtime.js";
import { createHumanoidAutonomyContext } from "./autonomy-context.js";
import { goalDAGContextView } from "./goal-dag-context.js";

export interface HumanoidContextAnchorResult {
  anchor: JsonValue;
  worldEvidence: GoalEvidenceArtifact;
}

export function createHumanoidContextAnchor(input: {
  mission: string;
  runMode: HumanoidRunMode;
  scenarioId: string;
  scenario: Scenario;
  missionGoal: Goal;
  checkpoint: HumanoidRunCheckpoint;
  activeGoal?: Goal;
  node: TaskNode;
  observation: HumanoidWorldObservation;
  world: HumanoidWorldSnapshot;
  cycleCompletion: HumanoidCycleCompletionReadiness;
  coordinatorPhase: HumanoidCoordinatorPhase;
}): HumanoidContextAnchorResult {
  const checker = input.activeGoal && input.checkpoint.goal_progress
    ? inspectHumanoidGoal(
        input.activeGoal,
        input.scenario,
        input.world,
        input.checkpoint.goal_progress
      )
    : null;
  const worldEvidence = createWorldGoalEvidence({
    world: input.world,
    observation: input.observation,
    scenario: input.scenario
  });
  const recentReceipts = Object.values(input.checkpoint.committed_actions)
    .slice(-16)
    .map((receipt) => ({
      transaction_id: receipt.transactionId,
      agent_id: receipt.agentId,
      action: receipt.action,
      accepted: receipt.accepted,
      code: receipt.code,
      world_before_revision: receipt.worldBeforeRevision,
      world_after_revision: receipt.worldAfterRevision,
      frame_count: receipt.frameCount
    }));
  const executionAuthority = pendingExecutionAuthority(input);
  return {
    worldEvidence,
    anchor: json({
      mission: input.mission,
      run_mode: input.runMode,
      scenario_id: input.scenarioId,
      mission_goal: input.missionGoal,
      goal_dag: goalDAGContextView(input.checkpoint.goal_dag),
      active_goal: input.activeGoal ?? null,
      active_cycle: input.checkpoint.active_cycle ?? null,
      cycle_completion: input.cycleCompletion,
      coordinator_phase: input.coordinatorPhase,
      execution_authority: executionAuthority,
      goal_context: {
        evidence_ref: worldEvidence.evidence.ref,
        evidence: worldEvidence.evidence,
        observation: worldEvidence.observation,
        autonomy: createHumanoidAutonomyContext({
          goalDAG: input.checkpoint.goal_dag,
          worldEvidence
        })
      },
      cycle_index: input.checkpoint.cycle_index,
      world_frame: input.world.frame,
      world_revision: input.world.worldRevision,
      robot: {
        root_position: input.world.robot.rootPosition,
        root_rotation: input.world.robot.rootRotation,
        fallen: input.world.robot.fallen,
        balance: input.world.robot.balance,
        feet: input.world.robot.feet,
        navigation: input.world.navigation
      },
      interaction: input.observation.interaction,
      goal_state: checker,
      recent_physical_episodes: recentEmbodiedEpisodes(
        input.checkpoint.embodied_memory
      ).map(({ model_summary: _modelSummary, ...episode }) => ({
        ...episode,
        historical_only: true,
        model_narrative_omitted: true
      })),
      embodied_experience_memory: {
        total: input.checkpoint.embodied_memory.total_experiences,
        pruned: input.checkpoint.embodied_memory.pruned_experiences,
        outcomes: input.checkpoint.embodied_memory.outcome_counts,
        predicate_outcomes: input.checkpoint.embodied_memory.predicate_outcome_counts,
        object_outcomes: input.checkpoint.embodied_memory.object_outcome_counts,
        zone_outcomes: input.checkpoint.embodied_memory.zone_outcome_counts,
        recent: recentEmbodiedExperiences(
          input.checkpoint.embodied_memory
        ).map((experience) => ({
          ...experience,
          historical_only: true
        }))
      },
      active_agent: {
        id: input.node.id,
        name: input.node.name,
        objective: input.node.objective,
        capabilities: input.node.capabilities,
        status: input.node.status
      },
      recent_receipts: recentReceipts
    })
  };
}

function pendingExecutionAuthority(input: {
  checkpoint: HumanoidRunCheckpoint;
  world: HumanoidWorldSnapshot;
  coordinatorPhase: HumanoidCoordinatorPhase;
}): JsonValue {
  const cycle = input.checkpoint.active_cycle;
  if (input.coordinatorPhase !== "execute_plan" || !cycle) return null;
  const receipt = Object.values(input.checkpoint.committed_actions).findLast((candidate) => (
    candidate.accepted
      && sameAutonomousCycle(candidate.cycle, cycle)
      && (candidate.action === "plan_whole_body_motion"
        || candidate.action === "plan_whole_body_motion_candidates"
        || candidate.action === "plan_humanoid_navigation")
  ));
  if (!receipt) return null;
  const executorAction = receipt.action === "plan_humanoid_navigation"
    ? "execute_humanoid_navigation"
    : "execute_whole_body_motion";
  const detail = record(receipt.detail);
  const expiresRevision = typeof detail?.expires_revision === "number"
    && Number.isSafeInteger(detail.expires_revision)
    ? detail.expires_revision
    : null;
  return json({
    task: "execute_plan",
    planning_action: receipt.action,
    planning_transaction_id: receipt.transactionId,
    executor_action: executorAction,
    accepted_world_revision: receipt.worldAfterRevision,
    expires_world_revision: expiresRevision,
    remaining_lease_revisions: expiresRevision === null
      ? null
      : Math.max(0, expiresRevision - input.world.worldRevision)
  });
}

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

function json(value: unknown): JsonValue {
  return JSON.parse(JSON.stringify(value)) as JsonValue;
}
