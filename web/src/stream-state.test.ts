import { describe, expect, it } from "vitest";
import {
  actionReceiptFrom,
  appendRecent,
  checkerFrom,
  completeRootNode,
  failOpenNodes,
  isWorldSnapshot,
  latestProviderActivity,
  mergeWorldFrames,
  taskNodesFrom,
  updateRunListStatus,
  upsertAction,
  worldSnapshotsFrom
} from "./stream-state";
import type { ActionReceipt, TaskNode, WorldSnapshot } from "./types";

/**
 * These reducers are the whole SSE ingest path: every frame, receipt, and node
 * the console shows passes through them, out of order and possibly duplicated,
 * from a live run that cannot be replayed. A silent mistake here shows up as a
 * console that disagrees with the run journal.
 */

function frame(index: number): WorldSnapshot {
  return {
    frame: index,
    simulated_time: index / 60,
    robot: {},
    objects: [],
    zones: [],
    obstacles: []
  } as unknown as WorldSnapshot;
}

function receipt(transactionId: string, code: string): ActionReceipt {
  return {
    transaction_id: transactionId,
    agent_id: "agent_1",
    agent_name: "Motion controller",
    name: "execute_base_plan",
    accepted: code === "base_plan_completed",
    code
  } as unknown as ActionReceipt;
}

function node(id: string, status: TaskNode["status"]): TaskNode {
  return { id, name: id, status, child_ids: [] } as unknown as TaskNode;
}

describe("mergeWorldFrames", () => {
  it("appends in-order frames without copying the array when nothing changes", () => {
    const current = [frame(1), frame(2)];
    expect(mergeWorldFrames(current, [], 10)).toBe(current);
  });

  it("orders late frames by frame number rather than arrival order", () => {
    const merged = mergeWorldFrames([frame(1), frame(4)], [frame(2), frame(3)], 10);
    expect(merged.map((entry) => entry.frame)).toEqual([1, 2, 3, 4]);
  });

  it("replaces a frame that arrives twice instead of duplicating it", () => {
    const revised = { ...frame(2), simulated_time: 99 };
    const merged = mergeWorldFrames([frame(1), frame(2), frame(3)], [revised], 10);
    expect(merged.map((entry) => entry.frame)).toEqual([1, 2, 3]);
    expect(merged[1]?.simulated_time).toBe(99);
  });

  it("replaces the newest frame when the same frame number is re-sent", () => {
    const revised = { ...frame(3), simulated_time: 42 };
    const merged = mergeWorldFrames([frame(1), frame(3)], [revised], 10);
    expect(merged).toHaveLength(2);
    expect(merged.at(-1)?.simulated_time).toBe(42);
  });

  it("keeps only the most recent frames once the limit is exceeded", () => {
    const merged = mergeWorldFrames([], [frame(1), frame(2), frame(3), frame(4)], 2);
    expect(merged.map((entry) => entry.frame)).toEqual([3, 4]);
  });

  it("does not mutate the array it was given", () => {
    const current = [frame(1), frame(3)];
    mergeWorldFrames(current, [frame(2)], 10);
    expect(current.map((entry) => entry.frame)).toEqual([1, 3]);
  });
});

describe("upsertAction", () => {
  it("replaces a receipt in place when the same transaction is revised", () => {
    const actions = [receipt("t1", "base_path_planned"), receipt("t2", "base_path_planned")];
    const next = upsertAction(actions, receipt("t1", "base_plan_completed"), 10);
    expect(next).toHaveLength(2);
    expect(next[0]).toMatchObject({ transaction_id: "t1", code: "base_plan_completed" });
    expect(actions[0]?.code).toBe("base_path_planned");
  });

  it("appends an unseen transaction", () => {
    const next = upsertAction([receipt("t1", "base_path_planned")], receipt("t2", "denied"), 10);
    expect(next.map((entry) => entry.transaction_id)).toEqual(["t1", "t2"]);
  });
});

describe("appendRecent", () => {
  it("drops the oldest entry once the limit is reached", () => {
    expect(appendRecent([1, 2, 3], 4, 3)).toEqual([2, 3, 4]);
  });

  it("grows until the limit", () => {
    expect(appendRecent([1], 2, 3)).toEqual([1, 2]);
  });
});

describe("run and node reducers", () => {
  it("updates only the addressed run", () => {
    const runs = [
      { run_id: "a", status: "running", error: null, updated_at: "t0" },
      { run_id: "b", status: "running", error: null, updated_at: "t0" }
    ] as Parameters<typeof updateRunListStatus>[0];
    const next = updateRunListStatus(runs, "b", "failed", "boom", "t1");
    expect(next[0]).toMatchObject({ run_id: "a", status: "running" });
    expect(next[1]).toMatchObject({ run_id: "b", status: "failed", error: "boom", updated_at: "t1" });
  });

  it("attaches the final result to the root node only when both output and checker exist", () => {
    const nodes = { root: node("root", "active") };
    const withResult = completeRootNode(nodes, "root", "done", { success: true, checks: [] } as never, "t1");
    expect(withResult.root).toMatchObject({ status: "completed", updated_at: "t1" });
    expect(withResult.root?.last_result).toMatchObject({ output: "done" });

    const withoutChecker = completeRootNode(nodes, "root", "done", null, "t1");
    expect(withoutChecker.root?.status).toBe("completed");
    expect(withoutChecker.root?.last_result).toBeUndefined();
  });

  it("leaves the map untouched when the root id is unknown", () => {
    const nodes = { root: node("root", "active") };
    expect(completeRootNode(nodes, "missing", "done", null, "t1")).toBe(nodes);
  });

  it("fails every still-open node and leaves settled ones alone", () => {
    const nodes = {
      a: node("a", "ready"),
      b: node("b", "active"),
      c: node("c", "waiting"),
      d: node("d", "completed")
    };
    const failed = failOpenNodes(nodes, "run aborted", "t2");
    expect(failed.a?.status).toBe("failed");
    expect(failed.b?.status).toBe("failed");
    expect(failed.c?.status).toBe("failed");
    expect(failed.d).toBe(nodes.d);
    expect(failed.a?.last_result).toMatchObject({ error: "run aborted" });
  });
});

describe("payload guards", () => {
  it("finds the newest provider activity, ignoring unrelated entries", () => {
    expect(latestProviderActivity([
      { status: "ok", at: "t0", source: "model" },
      { unrelated: true },
      { status: "error", at: "t1", source: "model" },
      "not an object"
    ])).toMatchObject({ status: "error", at: "t1" });
    expect(latestProviderActivity([{ unrelated: true }])).toBeNull();
  });

  it("extracts world snapshots from every shape the stream sends", () => {
    expect(worldSnapshotsFrom(frame(1))).toHaveLength(1);
    expect(worldSnapshotsFrom({ world: frame(1) })).toHaveLength(1);
    expect(worldSnapshotsFrom({ snapshot: frame(1) })).toHaveLength(1);
    expect(worldSnapshotsFrom({ frames: [frame(1), frame(2), { bogus: true }] })).toHaveLength(2);
    expect(worldSnapshotsFrom("nonsense")).toEqual([]);
  });

  it("rejects a snapshot missing any required field", () => {
    expect(isWorldSnapshot(frame(1))).toBe(true);
    expect(isWorldSnapshot({ ...frame(1), objects: undefined })).toBe(false);
    expect(isWorldSnapshot({ ...frame(1), robot: null })).toBe(false);
    expect(isWorldSnapshot(null)).toBe(false);
  });

  it("rejects a node map when any node is malformed", () => {
    expect(taskNodesFrom({ a: node("a", "active") })).not.toBeNull();
    expect(taskNodesFrom({ a: { id: "a", name: "a", status: "active" } })).toBeNull();
    expect(taskNodesFrom([node("a", "active")])).toBeNull();
  });

  it("rejects a receipt or checker payload missing its identifying fields", () => {
    expect(actionReceiptFrom(receipt("t1", "ok"))).not.toBeNull();
    expect(actionReceiptFrom({ ...receipt("t1", "ok"), accepted: "yes" })).toBeNull();
    expect(actionReceiptFrom({ ...receipt("t1", "ok"), transaction_id: undefined })).toBeNull();
    expect(checkerFrom({ success: true, checks: [] })).not.toBeNull();
    expect(checkerFrom({ success: true })).toBeNull();
  });
});
