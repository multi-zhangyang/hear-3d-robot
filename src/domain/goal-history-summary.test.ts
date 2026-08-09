import { describe, expect, it } from "vitest";
import type { GoalPredicate } from "./schema.js";
import type { GoalCandidate, GoalDAG, GoalEpoch } from "./goal-epoch.js";
import {
  appendGoalHistorySummary,
  createEmptyGoalHistorySummary,
  goalHistoryLifetimeProjection
} from "./goal-history-summary.js";

describe("Goal history lifetime summary", () => {
  it("aggregates every terminal result by predicate and referenced entity", () => {
    let summary = createEmptyGoalHistorySummary();
    const records: Array<{
      status: GoalEpoch["status"];
      predicate: GoalPredicate;
    }> = [
      {
        status: "completed",
        predicate: {
          type: "object_inside",
          object_id: "crate",
          container_id: "cabinet",
          expected: true,
          tolerance: 0.1
        }
      },
      {
        status: "blocked",
        predicate: { type: "block_removed", block_id: "stone-column" }
      },
      {
        status: "abandoned",
        predicate: { type: "robot_in_zone", zone_id: "workshop", tolerance: 0.2 }
      },
      {
        status: "superseded",
        predicate: {
          type: "articulation_state",
          object_id: "cabinet",
          joint_id: "door-hinge",
          state: "open",
          tolerance: 0.1
        }
      },
      {
        status: "expired",
        predicate: {
          type: "end_effector_at",
          end_effector: "left_wrist",
          target: { x: 1, y: 1, z: 1 },
          tolerance: 0.1,
          frame: "world",
          stable_frames: 3
        }
      }
    ];

    for (const [index, record] of records.entries()) {
      const sequence = index + 1;
      const selected = candidate(`selected-${sequence}`, record.status, record.predicate);
      summary = appendGoalHistorySummary(summary, {
        sequence,
        recordSha256: sequence.toString(16).padStart(64, "0"),
        candidate: selected,
        epoch: epoch(sequence, selected, record.status),
        alternateCandidates: sequence === 1
          ? [
              candidate("alternate-zone", "expired", {
                type: "object_placed",
                object_id: "crate",
                zone_id: "delivery",
                tolerance: 0.1
              }),
              candidate("alternate-object", "expired", {
                type: "object_at",
                object_id: "spare-part",
                target: { x: 2, y: 1, z: 3 },
                tolerance: 0.1
              })
            ]
          : [],
        alternateHistoryComplete: true
      });
    }

    expect(summary).toMatchObject({
      archived_epoch_count: 5,
      records_without_alternate_history: 0,
      outcomes: {
        selected: {
          total: 5,
          completed: 1,
          blocked: 1,
          abandoned: 1,
          superseded: 1,
          expired: 1
        },
        not_selected: 2
      }
    });
    expect(summary.outcomes.predicate_outcomes.map((entry) => entry.predicate_type))
      .toEqual([
        "articulation_state",
        "block_removed",
        "end_effector_at",
        "object_at",
        "object_inside",
        "object_placed",
        "robot_in_zone"
      ]);
    expect(summary.outcomes.entity_outcomes).toEqual(expect.arrayContaining([
      expect.objectContaining({
        entity_kind: "object",
        entity_id: "crate",
        selected: expect.objectContaining({ total: 1, completed: 1 }),
        not_selected: 1
      }),
      expect.objectContaining({
        entity_kind: "object",
        entity_id: "cabinet",
        selected: expect.objectContaining({ total: 2, completed: 1, superseded: 1 })
      }),
      expect.objectContaining({ entity_kind: "solid", entity_id: "stone-column" }),
      expect.objectContaining({ entity_kind: "zone", entity_id: "workshop" }),
      expect.objectContaining({ entity_kind: "zone", entity_id: "delivery", not_selected: 1 }),
      expect.objectContaining({ entity_kind: "end_effector", entity_id: "left_wrist" })
    ]));
  });

  it("marks legacy records whose unselected slate was not preserved", () => {
    const selected = candidate("legacy-selected", "completed", {
      type: "robot_at",
      target: { x: 1, y: 0, z: 1 },
      tolerance: 0.2
    });
    const summary = appendGoalHistorySummary(createEmptyGoalHistorySummary(), {
      sequence: 1,
      recordSha256: "f".repeat(64),
      candidate: selected,
      epoch: epoch(1, selected, "completed"),
      alternateCandidates: [],
      alternateHistoryComplete: false
    });

    expect(summary.records_without_alternate_history).toBe(1);
    expect(summary.outcomes.not_selected).toBe(0);
  });

  it("counts an active slate's already rejected alternatives without resolving its Goal", () => {
    const source = { model_call_id: "shared-proposal-call" };
    const selected = {
      ...candidate("selected-active", "active", {
        type: "robot_at",
        target: { x: 2, y: 0, z: 2 },
        tolerance: 0.2
      }),
      source,
      physical_evidence_refs: { proposal: ["observation"], resolution: [] },
      resolved_world_revision: null
    } as GoalCandidate;
    const alternates = ["east", "west"].map((direction) => ({
      ...candidate(`alternate-${direction}`, "expired", {
        type: "robot_at",
        target: { x: direction === "east" ? 3 : -3, y: 0, z: 2 },
        tolerance: 0.2
      }),
      source,
      physical_evidence_refs: {
        proposal: ["observation"],
        resolution: ["selection"]
      },
      resolved_world_revision: 7
    } as GoalCandidate));
    const dag = {
      version: 2,
      status: "active",
      candidates: Object.fromEntries([selected, ...alternates].map((entry) => (
        [entry.candidate_id, entry]
      ))),
      candidate_sequences: {
        [selected.candidate_id]: 1,
        [alternates[0]!.candidate_id]: 2,
        [alternates[1]!.candidate_id]: 3
      },
      next_candidate_sequence: 4,
      epochs: [{
        epoch_id: "active-epoch",
        epoch_index: 0,
        candidate_id: selected.candidate_id,
        status: "active",
        created_world_revision: 7,
        resolved_world_revision: null,
        physical_evidence_refs: { selection: ["selection"], resolution: [] }
      }],
      current_epoch_id: "active-epoch",
      next_epoch_index: 1,
      evidence: {},
      archive: {
        record_count: 0,
        last_record_sha256: null,
        last_epoch_id: null,
        retained_candidate_ids: [],
        summary: createEmptyGoalHistorySummary()
      },
      state_sha256: "unused"
    } as unknown as GoalDAG;

    expect(goalHistoryLifetimeProjection(dag)).toMatchObject({
      total_selected_epoch_count: 1,
      resolved_selected_goal_count: 0,
      active_selected_goal_count: 1,
      selected: { total: 0 },
      not_selected: 2
    });
  });
});

function candidate(
  id: string,
  status: GoalCandidate["status"],
  predicate: GoalPredicate
): GoalCandidate {
  return {
    candidate_id: id,
    status,
    goal: { summary: id, predicates: [predicate] }
  } as unknown as GoalCandidate;
}

function epoch(
  sequence: number,
  selected: GoalCandidate,
  status: GoalEpoch["status"]
): GoalEpoch {
  return {
    epoch_id: `epoch-${sequence}`,
    epoch_index: sequence - 1,
    candidate_id: selected.candidate_id,
    status,
    created_world_revision: sequence * 2 - 1,
    resolved_world_revision: sequence * 2
  } as unknown as GoalEpoch;
}
