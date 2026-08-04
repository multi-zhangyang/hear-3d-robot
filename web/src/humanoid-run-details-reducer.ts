import type {
  HumanoidActionReceipt,
  HumanoidRunDetails,
  ScenarioChunkDeltaState,
  HumanoidWorldSnapshot,
  RuntimeEvent,
  TaskNode
} from "./types";
import {
  activeAutonomousCycleFrom,
  asRecord,
  contextMemoryFrom,
  embodiedMemoryFrom,
  failOpenNodes,
  goalDAGFrom,
  humanoidActionReceiptFrom,
  humanoidCheckerFrom,
  humanoidGoalProgressFrom,
  modelUsageFrom,
  taskNodesFrom,
  upsertHumanoidAction,
  upsertRuntimeJournalEntry
} from "./stream-state";

interface HumanoidReducerInput {
  details: HumanoidRunDetails;
  event: RuntimeEvent;
  worlds: HumanoidWorldSnapshot[];
  historical: boolean;
  limits: { actions: number; provider: number };
}

export function reduceHumanoidRunDetails(input: HumanoidReducerInput): HumanoidRunDetails {
  const { details, event } = input;
  if (details.definition.run_id !== event.run_id) return details;
  const data = asRecord(event.data);
  if (event.type === "provider_event") {
    if (input.historical) return details;
    const modelUsage = modelUsageFrom(data?.model_usage);
    return {
      ...details,
      provider: upsertRuntimeJournalEntry(details.provider, event.data, input.limits.provider),
      ...(modelUsage ? {
        checkpoint: {
          ...details.checkpoint,
          model_usage: modelUsage,
          updated_at: event.at
        }
      } : {})
    };
  }
  if (input.historical) return details;

  let next = details;
  const candidate = input.worlds.at(-1);
  if (candidate
    && candidate.frame >= next.checkpoint.world.frame
    && candidate.robot.simulatedTime >= next.checkpoint.world.robot.simulatedTime) {
    next = {
      ...next,
      checkpoint: { ...next.checkpoint, world: candidate, updated_at: event.at }
    };
  }

  switch (event.type) {
    case "run_started":
    case "run_resumed":
      return {
        ...next,
        checkpoint: {
          ...next.checkpoint,
          status: "running",
          active_agent_id: next.checkpoint.active_agent_id ?? next.checkpoint.root_id,
          active_agent_ids: next.checkpoint.active_agent_ids.length > 0
            ? next.checkpoint.active_agent_ids
            : [next.checkpoint.root_id],
          error: null,
          updated_at: event.at
        }
      };

    case "hierarchy_focus_changed": {
      const nodes = taskNodesFrom(data?.nodes);
      const activeId = typeof data?.active_agent_id === "string"
        ? data.active_agent_id
        : next.checkpoint.active_agent_id;
      if (!nodes && activeId === next.checkpoint.active_agent_id) return next;
      return {
        ...next,
        checkpoint: {
          ...next.checkpoint,
          ...(nodes ? { nodes } : {}),
          active_agent_id: activeId,
          active_agent_ids: activeId ? [activeId] : [],
          updated_at: event.at
        }
      };
    }

    case "model_request_started":
    case "model_requests_reconciled": {
      const agentId = typeof data?.agent_id === "string" ? data.agent_id : null;
      const calls = typeof data?.node_model_calls === "number" ? data.node_model_calls : null;
      const total = typeof data?.total_model_calls === "number" ? data.total_model_calls : null;
      if (!agentId || calls === null) return next;
      const node = next.checkpoint.nodes[agentId];
      if (!node) return next;
      return {
        ...next,
        checkpoint: {
          ...next.checkpoint,
          total_model_calls: total ?? next.checkpoint.total_model_calls,
          active_agent_id: agentId,
          active_agent_ids: [agentId],
          nodes: {
            ...next.checkpoint.nodes,
            [agentId]: { ...node, model_calls_used: calls, updated_at: event.at }
          },
          updated_at: event.at
        }
      };
    }

    case "context_memory_updated": {
      const memory = contextMemoryFrom(data?.context_memory);
      return memory ? {
        ...next,
        checkpoint: { ...next.checkpoint, context_memory: memory, updated_at: event.at }
      } : next;
    }

    case "humanoid_goal_state_updated": {
      const goalDAG = goalDAGFrom(data?.goal_dag);
      if (!goalDAG || next.checkpoint.version !== 6) return next;
      const worldFrame = typeof data?.world_frame === "number" ? data.world_frame : null;
      const worldRevision = typeof data?.world_revision === "number" ? data.world_revision : null;
      const aligned = worldFrame === next.checkpoint.world.frame
        && worldRevision === next.checkpoint.world.worldRevision;
      const progress = data?.goal_progress === null
        ? null
        : aligned ? humanoidGoalProgressFrom(data?.goal_progress) : null;
      const checker = data?.checker === null
        ? null
        : aligned ? humanoidCheckerFrom(data?.checker) : null;
      const activeCycle = data?.active_cycle === null
        ? null
        : activeAutonomousCycleFrom(data?.active_cycle);
      return {
        ...next,
        checkpoint: {
          ...next.checkpoint,
          goal_dag: goalDAG,
          ...(data && Object.hasOwn(data, "active_cycle")
            ? { active_cycle: activeCycle }
            : {}),
          goal_progress: progress,
          checker,
          updated_at: event.at
        }
      };
    }

    case "autonomous_cycle_started": {
      const activeCycle = activeAutonomousCycleFrom(data?.cycle);
      return activeCycle ? {
        ...next,
        checkpoint: {
          ...next.checkpoint,
          active_cycle: activeCycle,
          updated_at: event.at
        }
      } : next;
    }

    case "autonomous_cycle_interrupted":
      return {
        ...next,
        checkpoint: {
          ...next.checkpoint,
          active_cycle: null,
          updated_at: event.at
        }
      };

    case "embodied_episode_recorded": {
      const memory = embodiedMemoryFrom(data?.embodied_memory);
      return memory ? {
        ...next,
        checkpoint: { ...next.checkpoint, embodied_memory: memory, updated_at: event.at }
      } : next;
    }

    case "humanoid_world_frame": {
      const checker = humanoidCheckerFrom(data?.checker);
      const progress = humanoidGoalProgressFrom(data?.goal_progress);
      const frame = next.checkpoint.world.frame;
      const revision = next.checkpoint.world.worldRevision;
      const alignedChecker = checker
        && checker.worldFrame === frame
        && checker.worldRevision === revision
        ? checker
        : null;
      const alignedProgress = progress
        && progress.last_world_frame === frame
        && progress.last_world_revision === revision
        ? progress
        : null;
      if (!alignedChecker && !alignedProgress) return next;
      return {
        ...next,
        checkpoint: {
          ...next.checkpoint,
          checker: alignedChecker ?? next.checkpoint.checker,
          ...(alignedProgress ? { goal_progress: alignedProgress } : {}),
          updated_at: event.at
        }
      };
    }

    case "humanoid_action_committed": {
      const receipt = humanoidActionReceiptFrom(data?.receipt);
      const checker = humanoidCheckerFrom(data?.checker);
      if (!receipt) return next;
      const committed = applyReceipt(next, receipt, checker, event.at, input.limits.actions);
      const scenarioChunks = nextScenarioChunks(
        committed.scenario_chunks,
        data?.scenario_chunks
      );
      return scenarioChunks
        ? { ...committed, scenario_chunks: scenarioChunks }
        : committed;
    }

    case "humanoid_scenario_synchronized": {
      const scenarioChunks = nextScenarioChunks(
        next.scenario_chunks,
        data?.scenario_chunks
      );
      return scenarioChunks
        ? { ...next, scenario_chunks: scenarioChunks }
        : next;
    }

    case "autonomous_cycle_completed": {
      const checker = humanoidCheckerFrom(data?.checker);
      const cycleIndex = typeof data?.cycle_index === "number"
        ? data.cycle_index
        : next.checkpoint.cycle_index;
      return {
        ...next,
        checkpoint: {
          ...next.checkpoint,
          cycle_index: cycleIndex,
          active_cycle: null,
          last_cycle: data?.output ?? next.checkpoint.last_cycle,
          checker: checker ?? next.checkpoint.checker,
          updated_at: event.at
        }
      };
    }

    case "run_succeeded": {
      const output = typeof data?.output === "string"
        ? data.output
        : typeof data?.final_output === "string" ? data.final_output : next.checkpoint.final_output;
      return {
        ...next,
        checkpoint: {
          ...next.checkpoint,
          status: "succeeded",
          active_agent_id: null,
          active_agent_ids: [],
          nodes: finishRoot(next.checkpoint.nodes, next.checkpoint.root_id, "completed", output, event.at),
          final_output: output,
          error: null,
          updated_at: event.at
        }
      };
    }

    case "run_failed":
    case "run_paused":
    case "run_interrupted": {
      const status = event.type === "run_failed"
        ? "failed" as const
        : event.type === "run_paused" ? "paused" as const : "interrupted" as const;
      const reason = typeof data?.error === "string"
        ? data.error
        : typeof data?.reason === "string" ? data.reason : "运行已结束，但服务端未提供原因";
      return {
        ...next,
        checkpoint: {
          ...next.checkpoint,
          status,
          active_agent_id: null,
          active_agent_ids: [],
          ...(status === "failed" ? { nodes: failOpenNodes(next.checkpoint.nodes, reason, event.at) } : {}),
          error: status === "paused" ? null : reason,
          updated_at: event.at
        }
      };
    }

    default:
      return next;
  }
}

function scenarioChunksFrom(value: unknown): ScenarioChunkDeltaState | null {
  const record = asRecord(value);
  if (!record || record.version !== 1
    || typeof record.scenario_seed !== "number"
    || !Number.isSafeInteger(record.scenario_seed)
    || record.scenario_seed < 0
    || typeof record.scenario_sha256 !== "string"
    || !/^[a-f0-9]{64}$/.test(record.scenario_sha256)
    || record.manifest_version !== 1
    || typeof record.revision !== "number"
    || !Number.isSafeInteger(record.revision)
    || record.revision < 0
    || !Array.isArray(record.changed_chunk_ids)
    || !record.changed_chunk_ids.every((id) => typeof id === "string" && id.length > 0)
    || new Set(record.changed_chunk_ids).size !== record.changed_chunk_ids.length
    || !Array.isArray(record.chunks)) return null;
  return structuredClone(value) as ScenarioChunkDeltaState;
}

function nextScenarioChunks(
  current: ScenarioChunkDeltaState,
  value: unknown
): ScenarioChunkDeltaState | null {
  const candidate = scenarioChunksFrom(value);
  if (!candidate
    || candidate.scenario_seed !== current.scenario_seed
    || candidate.scenario_sha256 !== current.scenario_sha256
    || candidate.manifest_version !== current.manifest_version
    || candidate.revision < current.revision) return null;
  if (candidate.revision === current.revision) {
    return JSON.stringify(candidate) === JSON.stringify(current) ? current : null;
  }
  return candidate;
}

function applyReceipt(
  details: HumanoidRunDetails,
  receipt: HumanoidActionReceipt,
  checker: HumanoidRunDetails["checkpoint"]["checker"],
  at: string,
  limit: number
): HumanoidRunDetails {
  const actions = upsertHumanoidAction(details.actions, receipt, limit);
  const existed = details.checkpoint.committed_actions[receipt.transactionId] !== undefined;
  const node = details.checkpoint.nodes[receipt.agentId];
  const nodes = !existed && node ? {
    ...details.checkpoint.nodes,
    [receipt.agentId]: {
      ...node,
      steps_used: node.steps_used + 1,
      last_result: receipt,
      updated_at: receipt.committedAt
    }
  } : details.checkpoint.nodes;
  return {
    ...details,
    actions,
    checkpoint: {
      ...details.checkpoint,
      nodes,
      committed_actions: {
        ...details.checkpoint.committed_actions,
        [receipt.transactionId]: receipt
      },
      checker: checker ?? details.checkpoint.checker,
      updated_at: at
    }
  };
}

function finishRoot(
  nodes: Record<string, TaskNode>,
  rootId: string,
  status: "completed" | "failed",
  result: string | null,
  at: string
): Record<string, TaskNode> {
  const root = nodes[rootId];
  if (!root) return nodes;
  return {
    ...nodes,
    [rootId]: {
      ...root,
      status,
      ...(result ? { last_result: { output: result } } : {}),
      updated_at: at
    }
  };
}
