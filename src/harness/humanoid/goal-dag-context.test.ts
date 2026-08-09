import { describe, expect, it } from "vitest";
import type { GoalDAG } from "../../domain/goal-epoch.js";
import {
  CONTEXT_GOAL_EPOCH_LIMIT,
  CONTEXT_PROPOSED_GOAL_LIMIT,
  goalDAGContextView
} from "./goal-dag-context.js";

describe("Goal DAG context view", () => {
  it("bounds historical epochs and proposals without changing durable identity", () => {
    const epochCount = CONTEXT_GOAL_EPOCH_LIMIT + 5;
    const proposalCount = CONTEXT_PROPOSED_GOAL_LIMIT + 4;
    const candidates = Object.fromEntries([
      ...Array.from({ length: epochCount }, (_, index) => candidate(index, "completed")),
      ...Array.from({ length: proposalCount }, (_, index) => (
        candidate(epochCount + index, "proposed")
      ))
    ].map((entry) => [entry.candidate_id, entry]));
    const dependentCandidateId = `candidate-${epochCount + proposalCount - 1}`;
    candidates[dependentCandidateId]!.dependency_candidate_ids = ["candidate-0"];
    const epochs = Array.from({ length: epochCount }, (_, index) => ({
      epoch_id: `epoch-${index}`,
      candidate_id: `candidate-${index}`,
      physical_evidence_refs: {
        selection: [`selection-${index}`],
        resolution: [`resolution-${index}`]
      }
    }));
    const evidence = Object.fromEntries(epochs.flatMap((epoch) => [
      [epoch.physical_evidence_refs.selection[0]!, { ref: epoch.physical_evidence_refs.selection[0] }],
      [epoch.physical_evidence_refs.resolution[0]!, { ref: epoch.physical_evidence_refs.resolution[0] }]
    ]));
    const candidateSequences = Object.fromEntries(
      Object.keys(candidates).map((candidateId, index) => [candidateId, index + 1])
    );
    const view = goalDAGContextView({
      version: 2,
      status: "awaiting_model_selection",
      candidates,
      candidate_sequences: candidateSequences,
      next_candidate_sequence: epochCount + proposalCount + 1,
      epochs,
      current_epoch_id: null,
      next_epoch_index: epochCount,
      evidence,
      archive: {
        record_count: 0,
        last_record_sha256: null,
        last_epoch_id: null,
        retained_candidate_ids: [],
        summary: {
          version: 1,
          archived_epoch_count: 0,
          last_record_sha256: null,
          records_without_alternate_history: 0,
          outcomes: {
            selected: {
              total: 0,
              completed: 0,
              blocked: 0,
              abandoned: 0,
              superseded: 0,
              expired: 0
            },
            not_selected: 0,
            predicate_outcomes: [],
            entity_outcomes: []
          }
        }
      },
      state_sha256: "durable-state"
    } as unknown as GoalDAG);

    expect(view.state_sha256).toBe("durable-state");
    expect(view.epochs).toHaveLength(CONTEXT_GOAL_EPOCH_LIMIT);
    expect(Object.values(view.candidates).filter(
      (entry) => entry.status === "proposed"
    )).toHaveLength(CONTEXT_PROPOSED_GOAL_LIMIT);
    expect(Object.values(view.candidates).every(
      (entry) => Number.isSafeInteger(entry.candidate_sequence)
    )).toBe(true);
    expect(view.candidate_sequences).toEqual(Object.fromEntries(
      Object.keys(view.candidates).map((candidateId) => (
        [candidateId, candidateSequences[candidateId]]
      ))
    ));
    expect(view.next_candidate_sequence).toBe(epochCount + proposalCount + 1);
    expect(view.archive.summary?.archived_epoch_count).toBe(0);
    expect(view.candidates[dependentCandidateId]?.dependency_candidates).toEqual([{
      candidate_id: "candidate-0",
      candidate_sequence: 1,
      status: "completed"
    }]);
    expect(view.context_projection).toMatchObject({
      total_candidate_count: epochCount + proposalCount,
      visible_candidate_count: CONTEXT_GOAL_EPOCH_LIMIT + CONTEXT_PROPOSED_GOAL_LIMIT,
      total_epoch_count: epochCount,
      visible_epoch_count: CONTEXT_GOAL_EPOCH_LIMIT,
      history_truncated: true
    });
  });
});

function candidate(index: number, status: "completed" | "proposed") {
  return {
    candidate_id: `candidate-${index}`,
    status,
    dependency_candidate_ids: [] as string[],
    physical_evidence_refs: {
      proposal: [`proposal-${index}`],
      resolution: status === "completed" ? [`resolution-${index}`] : []
    }
  };
}
