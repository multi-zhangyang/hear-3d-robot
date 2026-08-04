import { describe, expect, it } from "vitest";
import type { GoalDAG } from "../../domain/goal-epoch.js";
import { recallGoalHistory } from "./goal-history.js";

describe("Goal history recall", () => {
  it("recalls bounded semantic pages from the complete durable DAG", () => {
    const dag = goalDAG();
    const first = recallGoalHistory({
      goalDAG: dag,
      currentWorldRevision: 31,
      request: {
        statuses: ["completed"],
        zone_ids: ["workshop"],
        limit: 1
      }
    }) as Record<string, unknown>;

    expect(first).toMatchObject({
      historical_only: true,
      current_world_revision: 31,
      goal_dag_state_sha256: "state-hash",
      total_candidate_count: 3,
      total_matches: 2,
      returned: 1,
      next_before_candidate_sequence: 3,
      candidates: [{
        sequence: 3,
        candidate_id: "candidate-3",
        status: "completed",
        goal: { summary: "放置物体" },
        epoch: { epoch_index: 2, status: "completed" }
      }]
    });

    const next = recallGoalHistory({
      goalDAG: dag,
      currentWorldRevision: 31,
      request: {
        before_candidate_sequence: 3,
        statuses: ["completed"],
        zone_ids: ["workshop"],
        limit: 1
      }
    }) as Record<string, unknown>;
    expect(next).toMatchObject({
      total_matches: 1,
      next_before_candidate_sequence: null,
      candidates: [{ sequence: 1, candidate_id: "candidate-1" }]
    });
  });

  it("reports missing exact identities without substituting candidates", () => {
    const recalled = recallGoalHistory({
      goalDAG: goalDAG(),
      currentWorldRevision: 31,
      request: {
        candidate_ids: ["candidate-2", "missing"],
        limit: 2
      }
    });
    expect(recalled).toMatchObject({
      returned: 1,
      missing_candidate_ids: ["missing"],
      next_before_candidate_sequence: null,
      candidates: [{ candidate_id: "candidate-2", status: "blocked" }]
    });
  });
});

function goalDAG(): GoalDAG {
  const candidates = [
    candidate("candidate-1", "completed", {
      summary: "进入区域",
      predicates: [{ type: "robot_in_zone", zone_id: "workshop", tolerance: 0.2 }]
    }),
    candidate("candidate-2", "blocked", {
      summary: "拆除方块",
      predicates: [{ type: "block_removed", block_id: "column" }]
    }),
    candidate("candidate-3", "completed", {
      summary: "放置物体",
      predicates: [{
        type: "object_placed",
        object_id: "crate",
        zone_id: "workshop",
        tolerance: 0.2
      }]
    })
  ];
  return {
    version: 1,
    status: "awaiting_model_selection",
    candidates: Object.fromEntries(candidates.map((entry) => [entry.candidate_id, entry])),
    epochs: candidates.map((entry, index) => ({
      epoch_id: `epoch-${index + 1}`,
      epoch_index: index,
      candidate_id: entry.candidate_id,
      status: entry.status,
      retirement_reason: entry.status === "blocked" ? "路径受阻" : null,
      created_world_revision: index * 10,
      resolved_world_revision: index * 10 + 5
    })),
    current_epoch_id: null,
    next_epoch_index: 3,
    evidence: {},
    state_sha256: "state-hash"
  } as unknown as GoalDAG;
}

function candidate(
  id: string,
  status: "completed" | "blocked",
  goal: {
    summary: string;
    predicates: Array<Record<string, unknown>>;
  }
) {
  return {
    candidate_id: id,
    status,
    goal,
    mission_link: "推进任务",
    dependency_candidate_ids: [],
    created_world_revision: 0,
    resolved_world_revision: 5
  };
}
