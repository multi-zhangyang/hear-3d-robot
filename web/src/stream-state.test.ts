import { describe, expect, it } from "vitest";
import {
  appendRecent,
  contextMemoryFrom,
  failOpenNodes,
  humanoidActionReceiptFrom,
  humanoidCheckerFrom,
  humanoidGoalProgressFrom,
  humanoidWorldSnapshotsFrom,
  latestProviderActivity,
  nextRuntimeEventCursor,
  taskNodesFrom,
  updateRunListStatus,
  upsertHumanoidAction,
  upsertRuntimeJournalEntry
} from "./stream-state";
import type {
  HumanoidActionReceipt,
  HumanoidWorldSnapshot,
  RuntimeEvent,
  TaskNode
} from "./types";

describe("runtime event cursor", () => {
  it("does not advance past a live-only physics frame", () => {
    const event = runtimeEvent("world_frames", { frames: [] }, false);
    expect(nextRuntimeEventCursor("durable-before", event)).toBe("durable-before");
  });

  it("prefers a versioned cursor and supports older durable records", () => {
    const current = runtimeEvent("run_started", {}, true);
    expect(nextRuntimeEventCursor(undefined, current)).toBe(current.event_id);
    expect(nextRuntimeEventCursor(undefined, { ...current, cursor: "v1:0:proof" }))
      .toBe("v1:0:proof");
  });
});

describe("bounded live state", () => {
  it("replaces a humanoid receipt with the same transaction", () => {
    const first = receipt("tx-1", "motion_planned");
    const revised = receipt("tx-1", "motion_completed");
    const next = upsertHumanoidAction([first], revised, 10);
    expect(next).toEqual([revised]);
    expect(next).not.toBe([first]);
  });

  it("bounds append-only activity", () => {
    expect(appendRecent([1, 2], 3, 2)).toEqual([2, 3]);
    expect(appendRecent([1], 2, 3)).toEqual([1, 2]);
  });

  it("deduplicates a details tail and its matching SSE domain event", () => {
    const before: unknown[] = [
      { status: "contacted", runtime_event_id: "event-1" }
    ];
    const update = { status: "usable_stream", runtime_event_id: "event-1" };
    expect(upsertRuntimeJournalEntry(before, update, 10)).toEqual([update]);
    expect(upsertRuntimeJournalEntry(before, { status: "other" }, 2)).toHaveLength(2);
  });
});

describe("run projections", () => {
  it("updates only the addressed run", () => {
    const runs = [{
      run_id: "run-1",
      scenario_id: "humanoid_courtyard",
      mission: "walk",
      status: "running" as const,
      created_at: null,
      updated_at: null,
      error: null
    }];
    expect(updateRunListStatus(runs, "run-1", "succeeded", null, "now")[0])
      .toMatchObject({ status: "succeeded", updated_at: "now" });
    expect(updateRunListStatus(runs, "other", "failed", "error", "now")).toEqual(runs);
  });

  it("fails open hierarchy nodes without rewriting settled nodes", () => {
    const active = node("active", "active");
    const complete = node("complete", "completed");
    const next = failOpenNodes({ active, complete }, "physics stopped", "now");
    expect(next.active).toMatchObject({ status: "failed", last_result: { error: "physics stopped" } });
    expect(next.complete).toBe(complete);
  });
});

describe("humanoid payload guards", () => {
  it("finds the newest provider activity", () => {
    expect(latestProviderActivity([{}, { status: "contacted", at: "earlier" }, {
      status: "usable_stream",
      at: "latest",
      source: "model"
    }])).toEqual({ status: "usable_stream", at: "latest", source: "model" });
  });

  it("extracts only authoritative humanoid snapshots", () => {
    const first = frame(1);
    const second = frame(2);
    expect(humanoidWorldSnapshotsFrom({ frames: [first, { frame: 3 }, second] }))
      .toEqual([first, second]);
    expect(humanoidWorldSnapshotsFrom({ snapshot: first })).toEqual([first]);
  });

  it("rejects malformed hierarchy, action and checker records", () => {
    expect(taskNodesFrom({ root: node("root", "active") })).not.toBeNull();
    expect(taskNodesFrom({ root: { id: "root" } })).toBeNull();
    expect(humanoidActionReceiptFrom(receipt("tx", "humanoid_observed"))).not.toBeNull();
    expect(humanoidActionReceiptFrom({ transactionId: "tx" })).toBeNull();
    expect(humanoidCheckerFrom({ success: true, worldFrame: 1, worldRevision: 1, checks: [] }))
      .not.toBeNull();
    expect(humanoidCheckerFrom({ success: true, checks: [] })).toBeNull();
  });

  it("accepts only aligned nonnegative goal stability arrays", () => {
    const progress = {
      version: 1,
      goal_sha256: "a".repeat(64),
      predicate_count: 2,
      last_world_frame: 4,
      last_world_revision: 4,
      predicate_streaks: [0, 3]
    };
    expect(humanoidGoalProgressFrom(progress)).toEqual(progress);
    expect(humanoidGoalProgressFrom({ ...progress, predicate_streaks: [0] }))
      .toBeNull();
    expect(humanoidGoalProgressFrom({ ...progress, predicate_streaks: [0, -1] }))
      .toBeNull();
  });

  it("accepts only a structured context memory envelope", () => {
    const memory = {
      version: 1,
      context_window_tokens: 65_536,
      compact_trigger_tokens: 40_000,
      active_estimated_tokens: 12_000,
      total_compactions: 2,
      scopes: {}
    };
    expect(contextMemoryFrom(memory)).toEqual(memory);
    expect(contextMemoryFrom({ ...memory, scopes: [] })).toBeNull();
  });
});

function frame(index: number): HumanoidWorldSnapshot {
  return {
    frame: index,
    worldRevision: index,
    robot: {
      simulatedTime: index / 50,
      rootPosition: { x: 0, y: 0, z: 0 },
      links: {},
      objects: {}
    },
    navigation: {}
  } as unknown as HumanoidWorldSnapshot;
}

function receipt(transactionId: string, code: string): HumanoidActionReceipt {
  return {
    transactionId,
    agentId: "humanoid-executor",
    action: "execute_whole_body_motion",
    input: {},
    fingerprint: "fingerprint",
    accepted: true,
    code,
    worldBeforeRevision: 0,
    worldAfterRevision: 1,
    frameCount: 10,
    channels: ["locomotion"],
    detail: {},
    committedAt: "2026-08-02T00:00:00.000Z"
  };
}

function node(id: string, status: TaskNode["status"]): TaskNode {
  return {
    id,
    name: id,
    parent_id: null,
    child_ids: [],
    objective: "execute",
    success_criteria: [],
    evidence_requirements: [],
    goal_predicate_indexes: [],
    capabilities: [],
    may_delegate: false,
    references: [],
    depth: 0,
    status,
    steps_used: 0,
    model_calls_used: 0,
    created_at: "2026-08-02T00:00:00.000Z",
    updated_at: "2026-08-02T00:00:00.000Z"
  };
}

function runtimeEvent(type: string, data: unknown, durable?: boolean): RuntimeEvent {
  return {
    event_id: `${type}-event`,
    run_id: "humanoid-run",
    type,
    at: "2026-08-02T00:00:00.000Z",
    data,
    ...(durable === undefined ? {} : { durable })
  };
}
