import { describe, expect, it } from "vitest";
import { reduceHumanoidRunDetails } from "./humanoid-run-details-reducer";
import type {
  HumanoidActionReceipt,
  HumanoidRunDetails,
  HumanoidWorldSnapshot,
  RuntimeEvent,
  TaskNode
} from "./types";

const LIMITS = { actions: 20, provider: 20 };

describe("reduceHumanoidRunDetails", () => {
  it("ignores foreign and historical events", () => {
    const current = details();
    const foreign = { ...event("run_failed", { error: "boom" }), run_id: "other" };
    expect(reduce(current, foreign)).toBe(current);
    expect(reduce(current, event("run_failed", { error: "boom" }), { historical: true }))
      .toBe(current);
  });

  it("advances only to a monotonic authoritative humanoid frame", () => {
    const advanced = reduce(details(), event("humanoid_world_frame", {}), {
      worlds: [frame(2), frame(4)]
    });
    expect(advanced.checkpoint.world.frame).toBe(4);

    const stale = reduce(advanced, event("humanoid_action_committed", {}), {
      worlds: [frame(3)]
    });
    expect(stale.checkpoint.world.frame).toBe(4);
  });

  it("projects live physical goal stability from the matching world frame", () => {
    const world = frame(3);
    const goal_progress = {
      version: 1 as const,
      goal_sha256: "a".repeat(64),
      predicate_count: 1,
      last_world_frame: 3,
      last_world_revision: 3,
      predicate_streaks: [2]
    };
    const checker = {
      success: false,
      worldFrame: 3,
      worldRevision: 3,
      checks: [{ name: "1:end_effector_at", passed: false, actual: {} }],
      checkedAt: "2026-08-02T00:00:01.000Z"
    };
    const next = reduce(details(), event("humanoid_world_frame", {
      world,
      checker,
      goal_progress
    }), { worlds: [world] });

    expect(next.checkpoint.world.frame).toBe(3);
    expect(next.checkpoint.goal_progress).toEqual(goal_progress);
    expect(next.checkpoint.checker).toEqual(checker);

    const stale = reduce(next, event("humanoid_world_frame", {
      world: frame(2),
      checker: { ...checker, worldFrame: 2, worldRevision: 2 },
      goal_progress: {
        ...goal_progress,
        last_world_frame: 2,
        last_world_revision: 2,
        predicate_streaks: [1]
      }
    }), { worlds: [frame(2)] });
    expect(stale.checkpoint.goal_progress).toEqual(goal_progress);
  });

  it("commits one receipt exactly once", () => {
    const committed = event("humanoid_action_committed", { receipt: receipt("tx-1") });
    const first = reduce(details(), committed);
    const replayed = reduce(first, committed);

    expect(replayed.actions).toHaveLength(1);
    expect(replayed.checkpoint.nodes.root?.steps_used).toBe(1);
    expect(replayed.checkpoint.committed_actions["tx-1"]).toMatchObject({
      code: "humanoid_observed"
    });
  });

  it("projects model ownership and context pressure onto the correct agent", () => {
    const focused = reduce(details(), event("model_request_started", {
      agent_id: "motion",
      node_model_calls: 3,
      total_model_calls: 8
    }));
    expect(focused.checkpoint.active_agent_id).toBe("motion");
    expect(focused.checkpoint.nodes.motion?.model_calls_used).toBe(3);
    expect(focused.checkpoint.total_model_calls).toBe(8);

    const context_memory = {
      ...focused.checkpoint.context_memory,
      active_estimated_tokens: 12_400,
      total_compactions: 2
    };
    const compacted = reduce(focused, event("context_memory_updated", { context_memory }));
    expect(compacted.checkpoint.context_memory).toEqual(context_memory);
  });

  it("projects newly recorded embodied experience into the live checkpoint", () => {
    const embodied_memory = {
      version: 1 as const,
      total_episodes: 1,
      pruned_episodes: 0,
      recent_episodes: [{
        sequence: 1,
        transaction_id: "execution-1",
        action: "execute_whole_body_motion" as const,
        code: "motion_completed",
        model_summary: "机器人完成一次真实全身动作。",
        world_before_revision: 0,
        world_after_revision: 20,
        frame_count: 20,
        result_frame: 100,
        result_root_position: { x: 1, y: 0.8, z: 2 },
        fallen: false,
        support: "double" as const,
        upright: 0.99,
        goal_success: false,
        recorded_at: "2026-08-02T00:00:01.000Z"
      }]
    };
    const next = reduce(details(), event("embodied_episode_recorded", { embodied_memory }));
    expect(next.checkpoint.embodied_memory).toEqual(embodied_memory);
  });

  it("records terminal success without fabricating an active agent", () => {
    const next = reduce(details(), event("run_succeeded", { output: "已完成" }));
    expect(next.checkpoint.status).toBe("succeeded");
    expect(next.checkpoint.active_agent_id).toBeNull();
    expect(next.checkpoint.active_agent_ids).toEqual([]);
    expect(next.checkpoint.nodes.root?.status).toBe("completed");
    expect(next.checkpoint.final_output).toBe("已完成");
  });
});

function reduce(
  current: HumanoidRunDetails,
  runtimeEvent: RuntimeEvent,
  options: { worlds?: HumanoidWorldSnapshot[]; historical?: boolean } = {}
): HumanoidRunDetails {
  return reduceHumanoidRunDetails({
    details: current,
    event: runtimeEvent,
    worlds: options.worlds ?? [],
    historical: options.historical ?? false,
    limits: LIMITS
  });
}

function details(): HumanoidRunDetails {
  const root = node("root", "active");
  const motion = node("motion", "ready");
  return {
    definition: { run_id: "humanoid-run", runtime: "humanoid_g1" },
    actions: [],
    provider: [],
    framework: [],
    event_cursor: null,
    checkpoint: {
      version: 4,
      runtime: "humanoid_g1",
      run_id: "humanoid-run",
      status: "running",
      root_id: "root",
      active_agent_id: "root",
      active_agent_ids: ["root"],
      nodes: { root, motion },
      committed_actions: {},
      world: frame(0),
      context_memory: {
        version: 1,
        context_window_tokens: 65_536,
        compact_trigger_tokens: 36_000,
        compact_recent_model_turns: 4,
        compact_max_output_tokens: 2_048,
        active_scope_id: "root",
        active_estimated_tokens: 0,
        total_compactions: 0,
        last_compacted_at: null,
        scopes: {}
      },
      embodied_memory: {
        version: 1,
        total_episodes: 0,
        pruned_episodes: 0,
        recent_episodes: []
      },
      total_model_calls: 0,
      cycle_index: 0,
      checker: null,
      final_output: null,
      error: null,
      updated_at: "2026-08-02T00:00:00.000Z"
    }
  } as unknown as HumanoidRunDetails;
}

function node(id: string, status: TaskNode["status"]): TaskNode {
  return {
    id,
    name: id,
    status,
    child_ids: [],
    steps_used: 0,
    model_calls_used: 0,
    created_at: "2026-08-02T00:00:00.000Z",
    updated_at: "2026-08-02T00:00:00.000Z"
  } as unknown as TaskNode;
}

function frame(index: number): HumanoidWorldSnapshot {
  return {
    frame: index,
    worldRevision: index,
    robot: { simulatedTime: index / 50 }
  } as unknown as HumanoidWorldSnapshot;
}

function receipt(transactionId: string): HumanoidActionReceipt {
  return {
    transactionId,
    agentId: "root",
    action: "observe_humanoid",
    input: {},
    fingerprint: "fingerprint",
    accepted: true,
    code: "humanoid_observed",
    worldBeforeRevision: 0,
    worldAfterRevision: 0,
    frameCount: 0,
    channels: [],
    detail: {},
    committedAt: "2026-08-02T00:00:01.000Z"
  };
}

function event(type: string, data: unknown): RuntimeEvent {
  return {
    event_id: `${type}-event`,
    run_id: "humanoid-run",
    type,
    at: "2026-08-02T00:00:01.000Z",
    data
  };
}
