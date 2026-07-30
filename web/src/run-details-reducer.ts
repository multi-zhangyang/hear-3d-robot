/**
 * Folds one runtime event into the run details the operator UI renders.
 *
 * This is the entire live view of a mission. Events arrive over SSE from a run
 * that cannot be replayed, possibly out of order, sometimes replayed from a
 * cursor after a reconnect — so every case here is written to be idempotent and
 * to ignore anything addressed to a different run. Keeping it a pure function
 * of (details, event) is what makes that testable at all: the React layer only
 * has to decide *when* to apply it, not what it means.
 */
import type { ActionReceipt, RunDetails, RuntimeEvent, WorldSnapshot } from "./types";
import {
  actionReceiptFrom,
  asRecord,
  checkerFrom,
  completeRootNode,
  contextMemoryFrom,
  failOpenNodes,
  taskNodesFrom,
  upsertRuntimeJournalEntry,
  upsertAction
} from "./stream-state";

interface ReducerLimits {
  actions: number;
  provider: number;
}

export interface ReducerInput {
  details: RunDetails;
  event: RuntimeEvent;
  /** World frames already parsed out of this event, oldest first. */
  worlds: WorldSnapshot[];
  /**
   * True when the event predates the snapshot the details were loaded from.
   * Such an event may still carry journal lines worth keeping, but it must not
   * move status or world state backwards.
   */
  historical: boolean;
  limits: ReducerLimits;
}

/**
 * Returns the updated details, or the same object when the event changes
 * nothing — so React can skip the re-render by reference.
 */
export function reduceRunDetails(input: ReducerInput): RunDetails {
  const { event, details } = input;
  if (details.definition.run_id !== event.run_id) return details;
  const data = asRecord(event.data);

  if (event.type === "provider_event") {
    // Provider activity is an append-only journal, so a replayed event is
    // harmless, but a historical one must not overwrite newer status.
    if (input.historical) return details;
    return {
      ...details,
      provider: upsertRuntimeJournalEntry(details.provider, event.data, input.limits.provider)
    };
  }
  if (input.historical) return details;

  let next = details;
  const candidateWorld = input.worlds.at(-1);
  const latestWorld = candidateWorld
    && candidateWorld.frame >= next.checkpoint.world.frame
    && candidateWorld.simulated_time >= next.checkpoint.world.simulated_time
    ? candidateWorld
    : undefined;
  if (latestWorld) {
    next = {
      ...next,
      checkpoint: { ...next.checkpoint, world: latestWorld, updated_at: event.at }
    };
  }

  switch (event.type) {
    case "run_started":
    case "run_resumed": {
      const nodes = taskNodesFrom(data?.nodes);
      const rootId = typeof data?.root_id === "string" ? data.root_id : undefined;
      const activeId = typeof data?.active_agent_id === "string" || data?.active_agent_id === null
        ? data.active_agent_id
        : undefined;
      const activeIds = stringArray(data?.active_agent_ids);
      return {
        ...next,
        checkpoint: {
          ...next.checkpoint,
          status: "running",
          ...(nodes ? { nodes } : {}),
          ...(rootId !== undefined ? { root_id: rootId } : {}),
          ...(activeId !== undefined ? { active_agent_id: activeId } : {}),
          ...(activeIds ? { active_agent_ids: activeIds } : {}),
          error: null,
          updated_at: event.at
        }
      };
    }

    case "hierarchy_changed": {
      const nodes = taskNodesFrom(data?.nodes);
      if (!nodes) return next;
      const activeIds = stringArray(data?.active_agent_ids)
        ?? Object.values(nodes).filter((node) => node.status === "active").map((node) => node.id);
      const eventFocus = typeof data?.active_agent_id === "string" ? data.active_agent_id : null;
      const focus = eventFocus && activeIds.includes(eventFocus)
        ? eventFocus
        : activeIds.at(-1) ?? null;
      return {
        ...next,
        checkpoint: {
          ...next.checkpoint,
          nodes,
          active_agent_id: focus,
          active_agent_ids: activeIds,
          updated_at: event.at
        }
      };
    }

    case "model_request_started": {
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
          nodes: {
            ...next.checkpoint.nodes,
            [agentId]: { ...node, model_calls_used: calls, updated_at: event.at }
          },
          updated_at: event.at
        }
      };
    }

    case "context_memory_updated": {
      const contextMemory = contextMemoryFrom(data?.context_memory);
      if (!contextMemory) return next;
      return {
        ...next,
        checkpoint: {
          ...next.checkpoint,
          context_memory: contextMemory,
          updated_at: event.at
        }
      };
    }

    case "action_rejected":
    case "action_committed":
    case "action_reused": {
      const receipt = actionReceiptFrom(data?.receipt);
      return receipt ? applyActionReceipt(next, event.type, receipt, event.at, input.limits) : next;
    }

    case "run_succeeded": {
      const checker = checkerFrom(data?.checker);
      const finalOutput = typeof data?.final_output === "string"
        ? data.final_output
        : next.checkpoint.final_output;
      return {
        ...next,
        checkpoint: {
          ...next.checkpoint,
          status: "succeeded",
          active_agent_id: null,
          active_agent_ids: [],
          nodes: completeRootNode(
            next.checkpoint.nodes,
            next.checkpoint.root_id,
            finalOutput,
            checker ?? next.checkpoint.checker,
            event.at
          ),
          checker: checker ?? next.checkpoint.checker,
          final_output: finalOutput,
          error: null,
          updated_at: event.at
        }
      };
    }

    case "run_failed":
    case "run_interrupted": {
      const status = event.type === "run_failed" ? "failed" as const : "interrupted" as const;
      const error = typeof data?.error === "string"
        ? data.error
        : typeof data?.reason === "string" ? data.reason : "运行已结束，但服务端未提供原因";
      return {
        ...next,
        checkpoint: {
          ...next.checkpoint,
          status,
          // An interrupted run can be resumed, so its agents stay as they were.
          // A failed one cannot, and leaving nodes "active" would misreport a
          // dead hierarchy as still working.
          ...(status === "failed"
            ? {
                active_agent_id: null,
                active_agent_ids: [],
                nodes: failOpenNodes(next.checkpoint.nodes, error, event.at)
              }
            : {}),
          error,
          updated_at: event.at
        }
      };
    }

    default:
      return next;
  }
}

/**
 * Applies a receipt to the action list and the agent that produced it. A
 * `reused` receipt is a cache hit replayed for display: it updates the record
 * but must not count as new work, or a resumed run inflates every step counter.
 */
function applyActionReceipt(
  details: RunDetails,
  eventType: string,
  receipt: ActionReceipt,
  at: string,
  limits: ReducerLimits
): RunDetails {
  const actions = upsertAction(details.actions, receipt, limits.actions);
  const alreadyCommitted = details.checkpoint.committed_actions[receipt.transaction_id] !== undefined;
  const node = details.checkpoint.nodes[receipt.agent_id];
  const nodes = eventType !== "action_reused" && !alreadyCommitted && node
    ? {
        ...details.checkpoint.nodes,
        [receipt.agent_id]: {
          ...node,
          steps_used: node.steps_used + 1,
          last_result: receipt,
          updated_at: receipt.committed_at
        }
      }
    : details.checkpoint.nodes;
  const committedActions = Object.fromEntries(actions.map((action) => [
    action.transaction_id,
    action.transaction_id === receipt.transaction_id
      ? receipt
      : details.checkpoint.committed_actions[action.transaction_id] ?? action
  ]));
  const inflightActions = { ...(details.checkpoint.inflight_actions ?? {}) };
  delete inflightActions[receipt.transaction_id];
  const inflight = Object.values(inflightActions).at(-1) ?? null;

  return {
    ...details,
    actions,
    checkpoint: {
      ...details.checkpoint,
      nodes,
      inflight_action: inflight,
      inflight_actions: inflightActions,
      committed_actions: committedActions,
      ...(receipt.kind === "checker"
        ? { checker: checkerFrom(receipt.detail) ?? details.checkpoint.checker }
        : {}),
      updated_at: at
    }
  };
}

function stringArray(value: unknown): string[] | undefined {
  return Array.isArray(value) && value.every((entry) => typeof entry === "string")
    ? value
    : undefined;
}
