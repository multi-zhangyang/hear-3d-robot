import type {
  Goal,
  JsonValue,
  Scenario,
  TaskNode
} from "../../domain/schema.js";
import type { HumanoidRunMode } from "../../domain/run-mode.js";
import {
  humanoidActionReceiptsInCommitOrder,
  type HumanoidRunCheckpoint
} from "../../domain/humanoid-run.js";
import { sameAutonomousCycle } from "../../domain/autonomous-cycle.js";
import { inspectHumanoidGoal } from "../../runtime/humanoid-checker.js";
import { yawFromQuaternion } from "../../world/geometry.js";
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
import type { HumanoidAutonomyReadiness } from "./run-runtime.js";
import { humanoidReplanBudgetAuthority } from "../../domain/humanoid-replan-budget.js";
import { createHumanoidAutonomyContext } from "./autonomy-context.js";
import { goalDAGContextView } from "./goal-dag-context.js";
import { recentReceiptContext } from "./receipt-context.js";

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
  autonomyReadiness: HumanoidAutonomyReadiness;
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
  const recentReceipts = humanoidActionReceiptsInCommitOrder(
    input.checkpoint.committed_actions
  )
    .slice(-16)
    .map(recentReceiptContext);
  const executionAuthority = pendingExecutionAuthority(input);
  const recoveryAuthority = input.checkpoint.active_cycle
    ? humanoidReplanBudgetAuthority(input.checkpoint.active_cycle.replan_budget)
    : null;
  const rootYaw = yawFromQuaternion(input.world.robot.rootRotation);
  return {
    worldEvidence,
    anchor: json({
      mission: input.mission,
      run_mode: input.runMode,
      scenario_id: input.scenarioId,
      mission_goal: input.missionGoal,
      goal_dag: goalDAGContextView(input.checkpoint.goal_dag),
      active_goal: input.activeGoal ?? null,
      active_cycle: input.checkpoint.active_cycle
        ? {
            ...input.checkpoint.active_cycle,
            replan_budget: recoveryAuthority
          }
        : null,
      cycle_completion: input.cycleCompletion,
      autonomy_readiness: input.autonomyReadiness,
      execution_authority: executionAuthority,
      recovery_authority: recoveryAuthority,
      goal_context: {
        evidence_ref: worldEvidence.evidence.ref,
        evidence: worldEvidence.evidence,
        observation: worldEvidence.observation,
        autonomy: createHumanoidAutonomyContext({
          goalDAG: input.checkpoint.goal_dag,
          worldEvidence,
          runMode: input.runMode,
          missionGoal: input.missionGoal,
          spatialBelief: input.observation.spatialBelief,
          controllerCapabilities:
            input.world.robot.controller.learnedPolicy?.capabilities
        })
      },
      cycle_index: input.checkpoint.cycle_index,
      previous_cycle_transition: previousCycleTransitionContext(
        input.checkpoint.last_cycle
      ),
      world_frame: input.world.frame,
      world_revision: input.world.worldRevision,
      robot: {
        root_position: input.world.robot.rootPosition,
        root_rotation: input.world.robot.rootRotation,
        root_heading: {
          yaw_radians: rootYaw,
          forward_world: {
            x: Math.sin(rootYaw),
            y: 0,
            z: Math.cos(rootYaw)
          },
          left_world: {
            x: Math.cos(rootYaw),
            y: 0,
            z: -Math.sin(rootYaw)
          }
        },
        fallen: input.world.robot.fallen,
        balance: input.world.robot.balance,
        feet: input.world.robot.feet,
        navigation: input.world.navigation
      },
      interaction: input.observation.interaction,
      spatial_belief: input.observation.spatialBelief,
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
        solid_outcomes: input.checkpoint.embodied_memory.solid_outcome_counts,
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
  autonomyReadiness: HumanoidAutonomyReadiness;
}): JsonValue {
  const cycle = input.checkpoint.active_cycle;
  if (input.autonomyReadiness !== "execute_plan" || !cycle) return null;
  const receipt = humanoidActionReceiptsInCommitOrder(
    input.checkpoint.committed_actions
  ).findLast((candidate) => (
    candidate.accepted
      && sameAutonomousCycle(candidate.cycle, cycle)
      && (candidate.action === "plan_humanoid_skill"
        || candidate.action === "plan_whole_body_motion"
        || candidate.action === "plan_whole_body_motion_candidates"
        || candidate.action === "plan_humanoid_navigation")
  ));
  if (!receipt) return null;
  const executorAction = receipt.action === "plan_humanoid_skill"
    ? "execute_humanoid_skill"
    : receipt.action === "plan_humanoid_navigation"
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

/**
 * A prior model output is historical data, not authority for a successor
 * Cycle. In particular, Goal valuation prose must not smuggle a hand,
 * interaction point, coordinate or Skill through the Executive into a fresh
 * perception/action-selection branch. Current observations and receipts carry
 * the physical facts; only the durable transition identity crosses the cut.
 */
function previousCycleTransitionContext(value: JsonValue | null): JsonValue {
  const source = record(value);
  if (!source || typeof source.status !== "string") return null;
  const cycle = record(source.cycle);
  return json({
    status: source.status,
    cycle: cycle
      ? {
          cycle_id: typeof cycle.cycle_id === "string" ? cycle.cycle_id : null,
          cycle_index: typeof cycle.cycle_index === "number" ? cycle.cycle_index : null,
          goal_epoch_id: typeof cycle.goal_epoch_id === "string"
            ? cycle.goal_epoch_id
            : null
        }
      : null,
    evidence_ref: typeof source.evidence_ref === "string"
      ? source.evidence_ref
      : null,
    world_revision: typeof source.world_revision === "number"
      ? source.world_revision
      : null,
    retirement_status: typeof source.retirement_status === "string"
      ? source.retirement_status
      : null
  });
}

function json(value: unknown): JsonValue {
  return JSON.parse(JSON.stringify(value)) as JsonValue;
}
