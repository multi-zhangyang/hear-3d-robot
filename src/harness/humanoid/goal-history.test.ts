import { describe, expect, it } from "vitest";
import {
  applyGoalHistoryArchiveRecord,
  createGoalHistoryArchiveRecord,
  type GoalHistoryArchiveRecord
} from "../../domain/goal-history-archive.js";
import { createCompletedGoalDAG } from "../../domain/goal-history.test-support.js";
import type { GoalDAG } from "../../domain/goal-epoch.js";
import { recallGoalHistory } from "./goal-history.js";

describe("Goal history recall", () => {
  it("recalls bounded semantic pages from the complete durable DAG", async () => {
    const dag = goalDAG();
    const first = await recallGoalHistory({
      goalDAG: dag,
      journal: emptyJournal(),
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

    const next = await recallGoalHistory({
      goalDAG: dag,
      journal: emptyJournal(),
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

  it("reports missing exact identities without substituting candidates", async () => {
    const recalled = await recallGoalHistory({
      goalDAG: goalDAG(),
      journal: emptyJournal(),
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

  it("pages exact and semantic history across the archive and working DAG", async () => {
    let dag = createCompletedGoalDAG(15);
    const records: GoalHistoryArchiveRecord[] = [];
    while (dag.epochs.length > 12) {
      const record = createGoalHistoryArchiveRecord(dag);
      records.push(record);
      dag = applyGoalHistoryArchiveRecord(dag, record);
    }
    const journal = archiveJournal(records);

    await expect(recallGoalHistory({
      goalDAG: dag,
      journal,
      currentWorldRevision: 45,
      request: {
        candidate_ids: [records[0]!.candidate.candidate_id],
        limit: 1
      }
    })).resolves.toMatchObject({
      total_candidate_count: 15,
      total_matches: 1,
      candidates: [{ sequence: 1, epoch: { epoch_index: 0 } }],
      goal_history_archive_sha256: records.at(-1)!.record_sha256
    });

    const page = await recallGoalHistory({
      goalDAG: dag,
      journal,
      currentWorldRevision: 45,
      request: {
        before_candidate_sequence: 4,
        statuses: ["completed"],
        predicate_types: ["robot_at"],
        limit: 2
      }
    });
    expect(page).toMatchObject({
      total_matches: 3,
      candidates: [{ sequence: 3 }, { sequence: 2 }],
      next_before_candidate_sequence: 2
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
    candidate_sequences: Object.fromEntries(candidates.map((entry, index) => [
      entry.candidate_id,
      index + 1
    ])),
    next_candidate_sequence: 4,
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
    archive: {
      record_count: 0,
      last_record_sha256: null,
      last_epoch_id: null,
      retained_candidate_ids: []
    },
    state_sha256: "state-hash"
  } as unknown as GoalDAG;
}

function emptyJournal() {
  return {
    readJournalTail: async () => ({ entries: [], next: null, total: 0 }),
    readJournalPage: async () => ({ entries: [], next: null, total: 0 })
  };
}

function archiveJournal(records: GoalHistoryArchiveRecord[]) {
  return {
    readJournalTail: async (_name: "goal_history", limit: number) => ({
      entries: records.slice(-limit),
      next: null,
      total: records.length
    }),
    readJournalPage: async (
      _name: "goal_history",
      from: number,
      limit: number
    ) => ({
      entries: records.slice(from, from + limit),
      next: from + limit < records.length ? from + limit : null,
      total: records.length
    })
  };
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
