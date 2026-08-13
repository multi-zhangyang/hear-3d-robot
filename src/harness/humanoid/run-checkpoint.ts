import type { Goal } from "../../domain/schema.js";
import { EmptyContextMemoryState } from "../../domain/schema.js";
import {
  EmptyHumanoidEmbodiedMemoryState,
  humanoidActionReceiptEntriesInCommitOrder,
  type HumanoidRunCheckpoint
} from "../../domain/humanoid-run.js";
import type { RunStore } from "../../persistence/run-store.js";
import type { HumanoidWorld } from "../../world/humanoid/world.js";
import { createGoalDAG } from "../../domain/goal-epoch.js";
import { EmptyActionCommitOutbox } from "../../domain/action-commit-outbox.js";
import { EmptyActionExecutionLedger } from "../../domain/action-execution-ledger.js";
import { EmptyModelUsageState } from "../../domain/model-usage.js";
import {
  createNeuralHierarchyState
} from "../../domain/neural-hierarchy.js";
import {
  HUMANOID_NEURAL_AGENT_IDS,
  HUMANOID_NEURAL_RUNTIME_CAPABILITIES,
  humanoidNeuralHierarchyNodes,
  reconcileHumanoidNeuralHierarchyNodes
} from "./neural-hierarchy-contract.js";

export function createHumanoidRunCheckpoint(input: {
  store: RunStore;
  goal: Goal;
  world: HumanoidWorld;
}): HumanoidRunCheckpoint {
  const at = new Date().toISOString();
  const rootId = HUMANOID_NEURAL_AGENT_IDS.executive;
  const world = input.world.snapshot();
  return {
    version: 6,
    runtime: "humanoid_g1",
    run_id: input.store.definition.run_id,
    scenario_id: input.store.definition.scenario_id,
    mission_goal: structuredClone(input.goal),
    capability_catalog: [...HUMANOID_NEURAL_RUNTIME_CAPABILITIES],
    status: "starting",
    root_id: rootId,
    active_agent_id: rootId,
    active_agent_ids: [rootId],
    nodes: humanoidNeuralHierarchyNodes({
      mission: input.store.definition.mission,
      goalSummary: input.goal.summary,
      predicateIndexes: input.goal.predicates.map((_, index) => index),
      at
    }),
    world,
    world_checkpoint: input.world.checkpoint(),
    physical_state_anchor: null,
    goal_state_anchor: null,
    embodied_memory_state_anchor: null,
    context_memory_state_anchor: null,
    execution_ledger_state_anchor: null,
    goal_dag: createGoalDAG(),
    goal_progress: null,
    active_cycle: null,
    action_commit_outbox: structuredClone(EmptyActionCommitOutbox),
    action_execution_ledger: structuredClone(EmptyActionExecutionLedger),
    committed_actions: {},
    action_runtime_state: null,
    context_memory: structuredClone(EmptyContextMemoryState),
    embodied_memory: structuredClone(EmptyHumanoidEmbodiedMemoryState),
    neural_hierarchy_state: createNeuralHierarchyState(at),
    pending_lifecycle_events: [],
    cycle_index: 0,
    total_model_calls: 0,
    model_usage: structuredClone(EmptyModelUsageState),
    checker: null,
    last_cycle: null,
    final_output: null,
    error: null,
    created_at: at,
    updated_at: at
  };
}

export function reconcileHumanoidHierarchyCapabilities(
  source: HumanoidRunCheckpoint
): HumanoidRunCheckpoint {
  const checkpoint = structuredClone(source);
  const actionEntries = humanoidActionReceiptEntriesInCommitOrder(
    checkpoint.committed_actions
  );
  if (actionEntries.length > 0
    && actionEntries.every(([, receipt]) => receipt.commitSequence === undefined)) {
    for (const [index, [, receipt]] of actionEntries.entries()) {
      receipt.commitSequence = index + 1;
    }
  }
  const previousRootObjective = checkpoint.nodes[checkpoint.root_id]?.objective
    ?? checkpoint.mission_goal.summary;
  checkpoint.capability_catalog = [...HUMANOID_NEURAL_RUNTIME_CAPABILITIES];
  checkpoint.root_id = HUMANOID_NEURAL_AGENT_IDS.executive;
  checkpoint.nodes = reconcileHumanoidNeuralHierarchyNodes({
    nodes: checkpoint.nodes,
    mission: previousRootObjective,
    goalSummary: checkpoint.mission_goal.summary,
    predicateIndexes: checkpoint.mission_goal.predicates.map((_, index) => index),
    at: checkpoint.updated_at
  });
  const activeIds = checkpoint.active_agent_ids.filter(
    (agentId) => checkpoint.nodes[agentId] !== undefined
  );
  checkpoint.active_agent_id = checkpoint.active_agent_id
    && checkpoint.nodes[checkpoint.active_agent_id]
    ? checkpoint.active_agent_id
    : HUMANOID_NEURAL_AGENT_IDS.executive;
  checkpoint.active_agent_ids = activeIds.length > 0
    ? activeIds
    : [checkpoint.active_agent_id];
  return checkpoint;
}
