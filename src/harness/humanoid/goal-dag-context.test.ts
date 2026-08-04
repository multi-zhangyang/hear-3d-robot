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
    const view = goalDAGContextView({
      version: 1,
      status: "awaiting_model_selection",
      candidates,
      epochs,
      current_epoch_id: null,
      next_epoch_index: epochCount,
      evidence,
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
    physical_evidence_refs: {
      proposal: [`proposal-${index}`],
      resolution: status === "completed" ? [`resolution-${index}`] : []
    }
  };
}
