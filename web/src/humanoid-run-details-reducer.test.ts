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

  it("applies the live V2 Goal DAG stream and rejects obsolete V1 events", () => {
    const current = details();
    const goalDAG = streamedGoalDAG("live-v2-state");
    const updated = reduce(current, event("humanoid_goal_state_updated", {
      goal_dag: goalDAG,
      goal_progress: null,
      checker: null,
      world_frame: 0,
      world_revision: 0
    }));

    expect(updated.checkpoint.goal_dag).toEqual(goalDAG);
    expect(updated.checkpoint.goal_dag?.archive.summary).toMatchObject({
      archived_epoch_count: 3,
      outcomes: { selected: { total: 3, completed: 2 } }
    });

    const obsolete = reduce(updated, event("humanoid_goal_state_updated", {
      goal_dag: { ...streamedGoalDAG("obsolete-v1-state"), version: 1 },
      goal_progress: null,
      checker: null,
      world_frame: 0,
      world_revision: 0
    }));
    expect(obsolete).toBe(updated);
  });

  it("commits one receipt exactly once", () => {
    const scenarioChunks = {
      ...emptyScenarioChunks(),
      revision: 1,
      changed_chunk_ids: ["chunk_0_0"]
    };
    const committed = event("humanoid_action_committed", {
      receipt: receipt("tx-1"),
      scenario_chunks: scenarioChunks
    });
    const first = reduce(details(), committed);
    const replayed = reduce(first, committed);

    expect(replayed.actions).toHaveLength(1);
    expect(replayed.checkpoint.nodes.root?.steps_used).toBe(1);
    expect(replayed.checkpoint.committed_actions["tx-1"]).toMatchObject({
      code: "humanoid_observed"
    });
    expect(replayed.scenario_chunks).toEqual(scenarioChunks);
  });

  it("applies authoritative scenario synchronization monotonically", () => {
    const revisionOne = {
      ...emptyScenarioChunks(),
      revision: 1,
      changed_chunk_ids: ["chunk_0_0"]
    };
    const synchronized = reduce(details(), event("humanoid_scenario_synchronized", {
      scenario_chunks: revisionOne,
      synchronization: { changed: true }
    }));
    expect(synchronized.scenario_chunks).toEqual(revisionOne);

    const stale = reduce(synchronized, event("humanoid_scenario_synchronized", {
      scenario_chunks: emptyScenarioChunks(),
      synchronization: { changed: false }
    }));
    expect(stale).toBe(synchronized);

    const conflictingIdentity = reduce(synchronized, event(
      "humanoid_scenario_synchronized",
      {
        scenario_chunks: {
          ...revisionOne,
          revision: 2,
          scenario_sha256: "b".repeat(64)
        }
      }
    ));
    expect(conflictingIdentity).toBe(synchronized);
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

    const model_usage = {
      version: 1 as const,
      total: usageTotals(600),
      by_agent: { motion: usageTotals(600) },
      updated_at: "2026-08-02T00:00:03.000Z"
    };
    const metered = reduce(compacted, event("provider_event", {
      status: "usable_stream",
      usage: { inputTokens: 480, outputTokens: 120, totalTokens: 600 },
      model_usage
    }));
    expect(metered.checkpoint.model_usage).toEqual(model_usage);
    expect(metered.provider).toHaveLength(1);
  });

  it("projects newly recorded embodied experience into the live checkpoint", () => {
    const embodied_memory = {
      version: 2 as const,
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
      }],
      total_experiences: 1,
      pruned_experiences: 0,
      recent_experiences: [{
        sequence: 1,
        source_ref: "action:execution-1",
        transaction_id: "execution-1",
        cycle: {
          cycle_id: "autonomous-cycle:00000000-0000-4000-8000-000000000001",
          cycle_index: 1,
          goal_epoch_id: `goal-epoch:${"a".repeat(64)}`
        },
        action: "execute_whole_body_motion" as const,
        accepted: true,
        code: "motion_completed",
        outcome: "succeeded" as const,
        world_before_revision: 0,
        world_after_revision: 20,
        frame_count: 20,
        goal_content_sha256: "b".repeat(64),
        goal_summary: "移动到目标位置",
        predicate_types: ["robot_at"],
        object_ids: [],
        solid_ids: [],
        zone_ids: [],
        recorded_at: "2026-08-02T00:00:01.000Z"
      }],
      outcome_counts: { succeeded: 1, rejected: 0, physically_failed: 0 },
      predicate_outcome_counts: {
        robot_at: { succeeded: 1, rejected: 0, physically_failed: 0 }
      },
      object_outcome_counts: {},
      zone_outcome_counts: {}
    };
    const next = reduce(details(), event("embodied_episode_recorded", { embodied_memory }));
    expect(next.checkpoint.embodied_memory).toEqual(embodied_memory);
  });

  it("projects the explicit autonomous cycle lifecycle without time inference", () => {
    const cycle = {
      cycle_id: "autonomous-cycle:00000000-0000-4000-8000-000000000001",
      cycle_index: 1,
      goal_epoch_id: `goal-epoch:${"a".repeat(64)}`,
      started_world_frame: 0,
      started_world_revision: 0,
      started_at: "2026-08-03T00:00:00.000Z"
    };
    const active = reduce(details(), event("autonomous_cycle_started", { cycle }));
    expect(active.checkpoint.active_cycle).toEqual(cycle);

    const completed = reduce(active, event("autonomous_cycle_completed", {
      cycle,
      cycle_index: 1,
      output: { status: "cycle_completed" }
    }));
    expect(completed.checkpoint.active_cycle).toBeNull();
    expect(completed.checkpoint.cycle_index).toBe(1);
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
    scenario_chunks: emptyScenarioChunks(),
    event_cursor: null,
    checkpoint: {
      version: 6,
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
        version: 2,
        total_episodes: 0,
        pruned_episodes: 0,
        recent_episodes: [],
        total_experiences: 0,
        pruned_experiences: 0,
        recent_experiences: [],
        outcome_counts: { succeeded: 0, rejected: 0, physically_failed: 0 },
        predicate_outcome_counts: {},
        object_outcome_counts: {},
        zone_outcome_counts: {}
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

function streamedGoalDAG(stateSha256: string) {
  return {
    version: 2 as const,
    status: "awaiting_model_selection" as const,
    candidates: {},
    candidate_sequences: {},
    next_candidate_sequence: 4,
    epochs: [],
    current_epoch_id: null,
    next_epoch_index: 3,
    evidence: {},
    archive: {
      record_count: 3,
      last_record_sha256: "a".repeat(64),
      last_epoch_id: `goal-epoch:${"b".repeat(64)}`,
      retained_candidate_ids: [],
      summary: {
        version: 1 as const,
        archived_epoch_count: 3,
        last_record_sha256: "a".repeat(64),
        records_without_alternate_history: 1,
        outcomes: {
          selected: {
            total: 3,
            completed: 2,
            blocked: 1,
            abandoned: 0,
            superseded: 0,
            expired: 0
          },
          not_selected: 4,
          predicate_outcomes: [],
          entity_outcomes: []
        }
      }
    },
    state_sha256: stateSha256
  };
}

function emptyScenarioChunks() {
  return {
    version: 1 as const,
    scenario_seed: 1,
    scenario_sha256: "a".repeat(64),
    manifest_version: 1 as const,
    revision: 0,
    changed_chunk_ids: [],
    chunks: []
  };
}

function usageTotals(totalTokens: number) {
  return {
    requests: 1,
    reported_requests: 1,
    input_tokens: Math.max(0, totalTokens - 120),
    output_tokens: Math.min(120, totalTokens),
    total_tokens: totalTokens,
    cached_input_tokens: 0,
    reasoning_tokens: 0
  };
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
