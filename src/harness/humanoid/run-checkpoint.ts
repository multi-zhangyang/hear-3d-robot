import type { Goal, TaskNode } from "../../domain/schema.js";
import { EmptyContextMemoryState } from "../../domain/schema.js";
import {
  EmptyHumanoidEmbodiedMemoryState,
  type HumanoidRunCheckpoint
} from "../../domain/humanoid-run.js";
import type { RunStore } from "../../persistence/run-store.js";
import type { HumanoidWorld } from "../../world/humanoid/world.js";
import { createHumanoidGoalProgress } from "../../runtime/humanoid-checker.js";
import {
  HUMANOID_AGENT_IDS,
  HUMANOID_CAPABILITIES
} from "./agents.js";

export function createHumanoidRunCheckpoint(input: {
  store: RunStore;
  goal: Goal;
  world: HumanoidWorld;
}): HumanoidRunCheckpoint {
  const at = new Date().toISOString();
  const rootId = HUMANOID_AGENT_IDS.coordinator;
  const world = input.world.snapshot();
  return {
    version: 5,
    runtime: "humanoid_g1",
    run_id: input.store.definition.run_id,
    scenario_id: input.store.definition.scenario_id,
    goal: structuredClone(input.goal),
    capability_catalog: [...HUMANOID_CAPABILITIES],
    status: "starting",
    root_id: rootId,
    active_agent_id: rootId,
    active_agent_ids: [rootId],
    nodes: hierarchyNodes(input.store.definition.mission, input.goal, at),
    world,
    world_checkpoint: input.world.checkpoint(),
    goal_progress: createHumanoidGoalProgress(input.goal, world),
    committed_actions: {},
    context_memory: structuredClone(EmptyContextMemoryState),
    embodied_memory: structuredClone(EmptyHumanoidEmbodiedMemoryState),
    pending_lifecycle_events: [],
    cycle_index: 0,
    total_model_calls: 0,
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
  checkpoint.capability_catalog = [...HUMANOID_CAPABILITIES];
  const coordinator = checkpoint.nodes[HUMANOID_AGENT_IDS.coordinator];
  if (coordinator) coordinator.capabilities = [...HUMANOID_CAPABILITIES];
  const sentry = checkpoint.nodes[HUMANOID_AGENT_IDS.sentry];
  if (sentry) sentry.capabilities = ["observe_humanoid"];
  const motion = checkpoint.nodes[HUMANOID_AGENT_IDS.motion];
  if (motion) motion.capabilities = [
    "observe_humanoid",
    "recall_embodied_history",
    "plan_whole_body_motion_candidates",
    "plan_humanoid_navigation"
  ];
  const executor = checkpoint.nodes[HUMANOID_AGENT_IDS.executor];
  if (executor) executor.capabilities = [
    "execute_whole_body_motion",
    "execute_humanoid_navigation"
  ];
  return checkpoint;
}

function hierarchyNodes(mission: string, goal: Goal, at: string): Record<string, TaskNode> {
  const coordinator = node({
    id: HUMANOID_AGENT_IDS.coordinator,
    name: "人形自主协调智能体",
    parentId: null,
    childIds: [
      HUMANOID_AGENT_IDS.sentry,
      HUMANOID_AGENT_IDS.motion,
      HUMANOID_AGENT_IDS.executor
    ],
    objective: mission,
    criteria: [goal.summary],
    capabilities: [...HUMANOID_CAPABILITIES],
    mayDelegate: true,
    status: "active",
    predicateIndexes: goal.predicates.map((_, index) => index),
    at
  });
  const sentry = node({
    id: HUMANOID_AGENT_IDS.sentry,
    name: "人形感知哨兵",
    parentId: coordinator.id,
    childIds: [],
    objective: "从当前头部传感器、本体感觉和接触状态建立受限观察。",
    criteria: ["返回当前世界版本的传感器权威回执。"],
    capabilities: ["observe_humanoid"],
    mayDelegate: false,
    status: "ready",
    predicateIndexes: [],
    at
  });
  const motion = node({
    id: HUMANOID_AGENT_IDS.motion,
    name: "全身运动参考智能体",
    parentId: coordinator.id,
    childIds: [],
    objective: "根据实时状态生成连续全身参考或双足路线，并通过完整物理预演。",
    criteria: ["产生 accepted 规划回执或带物理证据的拒绝回执。"],
    capabilities: [
      "observe_humanoid",
      "recall_embodied_history",
      "plan_whole_body_motion_candidates",
      "plan_humanoid_navigation"
    ],
    mayDelegate: false,
    status: "ready",
    predicateIndexes: [],
    at
  });
  const executor = node({
    id: HUMANOID_AGENT_IDS.executor,
    name: "人形物理执行智能体",
    parentId: coordinator.id,
    childIds: [],
    objective: "消费已接受规划回执并在 YAHMP 与 MuJoCo 闭环中执行。",
    criteria: ["返回包含世界版本增长与物理终态的执行回执。"],
    capabilities: [
      "execute_whole_body_motion",
      "execute_humanoid_navigation"
    ],
    mayDelegate: false,
    status: "ready",
    predicateIndexes: [],
    at
  });
  return Object.fromEntries([coordinator, sentry, motion, executor].map((entry) => [entry.id, entry]));
}

function node(input: {
  id: string;
  name: string;
  parentId: string | null;
  childIds: string[];
  objective: string;
  criteria: string[];
  capabilities: string[];
  mayDelegate: boolean;
  status: TaskNode["status"];
  predicateIndexes: number[];
  at: string;
}): TaskNode {
  return {
    id: input.id,
    name: input.name,
    parent_id: input.parentId,
    child_ids: input.childIds,
    objective: input.objective,
    success_criteria: input.criteria,
    evidence_requirements: [],
    goal_predicate_indexes: input.predicateIndexes,
    capabilities: input.capabilities,
    may_delegate: input.mayDelegate,
    references: [],
    depth: input.parentId === null ? 0 : 1,
    status: input.status,
    steps_used: 0,
    model_calls_used: 0,
    created_at: input.at,
    updated_at: input.at
  };
}
