import type { Goal, TaskNode } from "../../domain/schema.js";
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
    version: 6,
    runtime: "humanoid_g1",
    run_id: input.store.definition.run_id,
    scenario_id: input.store.definition.scenario_id,
    mission_goal: structuredClone(input.goal),
    capability_catalog: [...HUMANOID_CAPABILITIES],
    status: "starting",
    root_id: rootId,
    active_agent_id: rootId,
    active_agent_ids: [rootId],
    nodes: hierarchyNodes(input.store.definition.mission, input.goal, at),
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
  checkpoint.capability_catalog = [...HUMANOID_CAPABILITIES];
  const coordinator = checkpoint.nodes[HUMANOID_AGENT_IDS.coordinator];
  if (coordinator) coordinator.capabilities = [...HUMANOID_CAPABILITIES];
  const goalManager = checkpoint.nodes[HUMANOID_AGENT_IDS.goalManager]
    ?? node({
      id: HUMANOID_AGENT_IDS.goalManager,
      name: "自主目标管理智能体",
      parentId: HUMANOID_AGENT_IDS.coordinator,
      childIds: [],
      objective: "根据长期任务、Goal DAG 和当前物理证据提出并显式选择下一阶段目标。",
      criteria: ["候选和选择均绑定真实模型调用、物理证据与恢复身份。"],
      capabilities: [],
      mayDelegate: false,
      status: "ready",
      predicateIndexes: [],
      at: checkpoint.updated_at
    });
  goalManager.capabilities = [
    "recall_goal_history",
    "submit_goal_candidates",
    "select_goal_candidate",
    "retire_goal_epoch",
    "continue_goal_epoch"
  ];
  checkpoint.nodes[goalManager.id] = goalManager;
  if (coordinator && !coordinator.child_ids.includes(goalManager.id)) {
    coordinator.child_ids = [goalManager.id, ...coordinator.child_ids];
  }
  const sentry = checkpoint.nodes[HUMANOID_AGENT_IDS.sentry];
  if (sentry) {
    sentry.name = "异步物理 Grounding Monitor";
    sentry.capabilities = ["observe_humanoid"];
  }
  const motion = checkpoint.nodes[HUMANOID_AGENT_IDS.motion];
  if (motion) {
    motion.name = "全身运动动作智能体";
    motion.parent_id = HUMANOID_AGENT_IDS.motionPlanner;
    motion.capabilities = [
    "submit_humanoid_skill_plan",
    "begin_humanoid_skill",
    "plan_whole_body_motion_candidates",
    "plan_humanoid_navigation"
    ];
  }
  const motionPlanner = checkpoint.nodes[HUMANOID_AGENT_IDS.motionPlanner]
    ?? node({
      id: HUMANOID_AGENT_IDS.motionPlanner,
      name: "全身运动规划智能体",
      parentId: HUMANOID_AGENT_IDS.coordinator,
      childIds: [HUMANOID_AGENT_IDS.motion],
      objective: "基于实时状态和长期具身经验产生本轮有界语义运动计划。",
      criteria: ["输出一个不含执行权限的 Motion Plan Artifact。"],
      capabilities: ["recall_embodied_history"],
      mayDelegate: true,
      status: "ready",
      predicateIndexes: [],
      at: checkpoint.updated_at
    });
  motionPlanner.child_ids = [HUMANOID_AGENT_IDS.motion];
  motionPlanner.capabilities = ["recall_embodied_history"];
  checkpoint.nodes[motionPlanner.id] = motionPlanner;
  if (coordinator) {
    coordinator.child_ids = coordinator.child_ids
      .filter((id) => id !== HUMANOID_AGENT_IDS.motion);
    if (!coordinator.child_ids.includes(motionPlanner.id)) {
      coordinator.child_ids.push(motionPlanner.id);
    }
  }
  const executor = checkpoint.nodes[HUMANOID_AGENT_IDS.executor];
  if (executor) {
    executor.name = "确定性物理 Execution Gate";
    executor.capabilities = [
      "execute_humanoid_skill",
      "execute_whole_body_motion",
      "execute_humanoid_navigation",
      "remove_world_block"
    ];
  }
  return checkpoint;
}

function hierarchyNodes(mission: string, goal: Goal, at: string): Record<string, TaskNode> {
  const coordinator = node({
    id: HUMANOID_AGENT_IDS.coordinator,
    name: "人形自主协调智能体",
    parentId: null,
    childIds: [
      HUMANOID_AGENT_IDS.goalManager,
      HUMANOID_AGENT_IDS.sentry,
      HUMANOID_AGENT_IDS.motionPlanner,
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
  const goalManager = node({
    id: HUMANOID_AGENT_IDS.goalManager,
    name: "自主目标管理智能体",
    parentId: coordinator.id,
    childIds: [],
    objective: "根据长期任务、Goal DAG 和当前物理证据提出并显式选择下一阶段目标。",
    criteria: ["候选和选择均绑定真实模型调用、物理证据与恢复身份。"],
    capabilities: [
      "recall_goal_history",
      "submit_goal_candidates",
      "select_goal_candidate",
      "retire_goal_epoch",
      "continue_goal_epoch"
    ],
    mayDelegate: false,
    status: "ready",
    predicateIndexes: [],
    at
  });
  const sentry = node({
    id: HUMANOID_AGENT_IDS.sentry,
    name: "异步物理 Grounding Monitor",
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
  const motionPlanner = node({
    id: HUMANOID_AGENT_IDS.motionPlanner,
    name: "全身运动规划智能体",
    parentId: coordinator.id,
    childIds: [HUMANOID_AGENT_IDS.motion],
    objective: "基于实时状态和长期具身经验产生本轮有界语义运动计划。",
    criteria: ["输出一个不含执行权限的 Motion Plan Artifact。"],
    capabilities: ["recall_embodied_history"],
    mayDelegate: true,
    status: "ready",
    predicateIndexes: [],
    at
  });
  const motion = node({
    id: HUMANOID_AGENT_IDS.motion,
    name: "全身运动动作智能体",
    parentId: motionPlanner.id,
    childIds: [],
    objective: "把本轮有界语义计划和最新权威状态落实为唯一正式规划工具调用。",
    criteria: ["产生 accepted 规划回执或带物理证据的拒绝回执。"],
    capabilities: [
      "submit_humanoid_skill_plan",
      "begin_humanoid_skill",
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
    name: "确定性物理 Execution Gate",
    parentId: coordinator.id,
    childIds: [],
    objective: "消费已接受规划回执并在 YAHMP 与 MuJoCo 闭环中执行。",
    criteria: ["返回包含世界版本增长与物理终态的执行回执。"],
    capabilities: [
      "execute_humanoid_skill",
      "execute_whole_body_motion",
      "execute_humanoid_navigation",
      "remove_world_block"
    ],
    mayDelegate: false,
    status: "ready",
    predicateIndexes: [],
    at
  });
  return Object.fromEntries(
    [coordinator, goalManager, sentry, motionPlanner, motion, executor]
      .map((entry) => [entry.id, entry])
  );
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
