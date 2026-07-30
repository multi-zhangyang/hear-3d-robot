import { describe, expect, it } from "vitest";
import { reduceRunDetails } from "./run-details-reducer";
import type { ActionReceipt, RunDetails, RuntimeEvent, TaskNode, WorldSnapshot } from "./types";

/**
 * The reducer is the operator UI's entire understanding of a live mission. Events
 * reach it over SSE from a run that cannot be replayed, so the cases that matter
 * are the awkward ones: a reconnect replaying events already folded in, a
 * receipt arriving twice, an event for a run the operator has since switched
 * away from. Each of those must leave the view exactly as correct as before.
 */

const LIMITS = { actions: 100, provider: 100 };

function details(overrides: Partial<RunDetails["checkpoint"]> = {}): RunDetails {
  return {
    definition: { run_id: "run_1" },
    actions: [],
    provider: [],
    framework: [],
    event_cursor: null,
    checkpoint: {
      run_id: "run_1",
      status: "running",
      root_id: "root",
      active_agent_id: "root",
      nodes: { root: node("root", "active") },
      committed_actions: {},
      inflight_action: null,
      checker: null,
      final_output: null,
      error: null,
      total_model_calls: 0,
      updated_at: "2026-07-26T12:00:00.000Z",
      world: frame(0),
      ...overrides
    }
  } as unknown as RunDetails;
}

function node(id: string, status: TaskNode["status"]): TaskNode {
  return { id, name: id, status, child_ids: [], steps_used: 0, model_calls_used: 0 } as unknown as TaskNode;
}

function frame(index: number): WorldSnapshot {
  return { frame: index, simulated_time: index / 60 } as unknown as WorldSnapshot;
}

function receipt(transactionId: string, accepted: boolean): ActionReceipt {
  return {
    transaction_id: transactionId,
    agent_id: "root",
    agent_name: "root",
    kind: "skill",
    name: "execute_base_plan",
    accepted,
    code: accepted ? "base_plan_completed" : "base_path_unavailable",
    detail: {},
    frame_count: 3,
    committed_at: "2026-07-26T12:00:05.000Z"
  } as unknown as ActionReceipt;
}

function event(type: string, data: unknown, at = "2026-07-26T12:00:05.000Z"): RuntimeEvent {
  return { event_id: `${type}_${at}`, run_id: "run_1", type, at, data };
}

function reduce(
  current: RunDetails,
  runtimeEvent: RuntimeEvent,
  options: { worlds?: WorldSnapshot[]; historical?: boolean } = {}
): RunDetails {
  return reduceRunDetails({
    details: current,
    event: runtimeEvent,
    worlds: options.worlds ?? [],
    historical: options.historical ?? false,
    limits: LIMITS
  });
}

describe("reduceRunDetails", () => {
  it("ignores an event addressed to a different run", () => {
    const current = details();
    const foreign = { ...event("run_failed", { error: "boom" }), run_id: "run_2" };
    expect(reduce(current, foreign)).toBe(current);
  });

  it("drops historical events so a reconnect cannot rewind the view", () => {
    const current = details({ status: "succeeded" } as Partial<RunDetails["checkpoint"]>);
    const replayed = reduce(current, event("run_started", { nodes: {} }), { historical: true });
    expect(replayed.checkpoint.status).toBe("succeeded");
  });

  it("still records provider activity that is not historical", () => {
    const next = reduce(details(), event("provider_event", { status: "contacted" }));
    expect(next.provider).toHaveLength(1);
  });

  it("advances the world from the latest frame in the event", () => {
    const next = reduce(details(), event("world_frames", {}), { worlds: [frame(4), frame(9)] });
    expect(next.checkpoint.world.frame).toBe(9);
  });

  it("counts a committed action once even when the receipt is delivered twice", () => {
    const first = reduce(details(), event("action_committed", { receipt: receipt("tx_1", true) }));
    expect(first.actions).toHaveLength(1);
    expect(first.checkpoint.nodes.root?.steps_used).toBe(1);

    // The same receipt again — a reconnect replay, or the server re-sending.
    const second = reduce(first, event("action_committed", { receipt: receipt("tx_1", true) }));
    expect(second.actions).toHaveLength(1);
    expect(second.checkpoint.nodes.root?.steps_used).toBe(1);
  });

  it("records a reused action without charging the agent another step", () => {
    const next = reduce(details(), event("action_reused", { receipt: receipt("tx_2", true) }));
    expect(next.actions).toHaveLength(1);
    expect(next.checkpoint.nodes.root?.steps_used).toBe(0);
  });

  it("keeps a denied receipt, because denials are the primary debugging signal", () => {
    const next = reduce(details(), event("action_rejected", { receipt: receipt("tx_3", false) }));
    expect(next.actions[0]).toMatchObject({ accepted: false, code: "base_path_unavailable" });
  });

  it("marks open nodes failed on run_failed but leaves them alone on interrupt", () => {
    const failed = reduce(details(), event("run_failed", { error: "provider down" }));
    expect(failed.checkpoint.status).toBe("failed");
    expect(failed.checkpoint.nodes.root?.status).toBe("failed");
    expect(failed.checkpoint.active_agent_id).toBeNull();

    // An interrupted run is resumable, so its hierarchy must survive intact.
    const interrupted = reduce(details(), event("run_interrupted", { reason: "operator stop" }));
    expect(interrupted.checkpoint.status).toBe("interrupted");
    expect(interrupted.checkpoint.nodes.root?.status).toBe("active");
  });

  it("supplies a reason when a terminal event carries none", () => {
    const next = reduce(details(), event("run_failed", {}));
    expect(next.checkpoint.error).toBe("运行已结束，但服务端未提供原因");
  });

  it("tracks model call counts onto the agent that made them", () => {
    const next = reduce(details(), event("model_request_started", {
      agent_id: "root",
      node_model_calls: 4,
      total_model_calls: 7
    }));
    expect(next.checkpoint.nodes.root?.model_calls_used).toBe(4);
    expect(next.checkpoint.total_model_calls).toBe(7);
  });

  it("updates the live context-memory meter from a compaction event", () => {
    const context_memory = {
      version: 1 as const,
      context_window_tokens: 65536,
      compact_trigger_tokens: 36000,
      compact_recent_model_turns: 4,
      compact_max_output_tokens: 2048,
      active_scope_id: "root",
      active_estimated_tokens: 18400,
      total_compactions: 2,
      last_compacted_at: "2026-07-27T00:00:00.000Z",
      scopes: {}
    };
    const next = reduce(details(), event("context_memory_updated", { context_memory }));
    expect(next.checkpoint.context_memory).toEqual(context_memory);
  });

  it("ignores model call counts for an agent the UI has never seen", () => {
    const current = details();
    expect(reduce(current, event("model_request_started", {
      agent_id: "ghost",
      node_model_calls: 2
    }))).toBe(current);
  });

  it("derives the active agent from the hierarchy it is given", () => {
    const next = reduce(details(), event("hierarchy_changed", {
      nodes: { root: node("root", "waiting"), child: node("child", "active") }
    }));
    expect(next.checkpoint.active_agent_id).toBe("child");
  });

  it("preserves multiple active siblings while model telemetry changes the UI focus", () => {
    const root = node("root", "waiting");
    const first = node("first", "active");
    const second = node("second", "active");
    const hierarchy = reduce(details(), event("hierarchy_changed", {
      nodes: { root, first, second },
      active_agent_id: "first",
      active_agent_ids: ["first", "second"]
    }));
    expect(hierarchy.checkpoint.active_agent_ids).toEqual(["first", "second"]);
    expect(hierarchy.checkpoint.active_agent_id).toBe("first");

    const focused = reduce(hierarchy, event("model_request_started", {
      agent_id: "second",
      node_model_calls: 1,
      total_model_calls: 2
    }));
    expect(focused.checkpoint.active_agent_id).toBe("second");
    expect(focused.checkpoint.active_agent_ids).toEqual(["first", "second"]);
  });

  it("clears only the matching inflight command when siblings finish out of order", () => {
    const first = receipt("tx_1", true);
    const second = receipt("tx_2", true);
    const current = details({
      inflight_action: second as never,
      inflight_actions: {
        tx_1: { ...first, kind: "skill", started_at: first.committed_at },
        tx_2: { ...second, kind: "skill", started_at: second.committed_at }
      } as never
    });
    const next = reduce(current, event("action_committed", { receipt: first }));
    expect(Object.keys(next.checkpoint.inflight_actions ?? {})).toEqual(["tx_2"]);
    expect(next.checkpoint.inflight_action).toMatchObject({ transaction_id: "tx_2" });
  });

  it("records the checker result that decides whether the mission succeeded", () => {
    const next = reduce(details(), event("run_succeeded", {
      final_output: "done",
      checker: { success: true, checks: [] }
    }));
    expect(next.checkpoint.status).toBe("succeeded");
    expect(next.checkpoint.checker).toMatchObject({ success: true });
    expect(next.checkpoint.final_output).toBe("done");
    expect(next.checkpoint.active_agent_id).toBeNull();
  });

  it("returns the same object for an event type it does not handle", () => {
    const current = details();
    expect(reduce(current, event("framework_event", { anything: true }))).toBe(current);
  });
});
