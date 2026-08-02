import { describe, expect, it } from "vitest";
import type { ActionReceipt } from "../domain/schema.js";
import {
  frontierCycleActionEnabled,
  frontierCycleState
} from "./frontier-cycle.js";

const surveyTransaction = "agent_a:survey_1";

describe("frontier cycle projection", () => {
  it("exposes observation before a current survey and selection after it", () => {
    const initial = cycle([]);
    expect(initial).toMatchObject({
      phase: "survey_required",
      enabled_action: "survey_terrain",
      current_survey_transaction_id: null,
      available_choice_ids: []
    });
    expect(frontierCycleActionEnabled("survey_terrain", initial)).toBe(true);
    expect(frontierCycleActionEnabled("navigate_frontier", initial)).toBe(false);

    const current = cycle([surveyReceipt()]);
    expect(current).toMatchObject({
      phase: "choice_required",
      enabled_action: "navigate_frontier",
      current_survey_transaction_id: surveyTransaction,
      available_choice_ids: ["frontier_a", "frontier_b"],
      decision_owner: "model",
      automatic_actuation: false
    });
    expect(frontierCycleActionEnabled("survey_terrain", current)).toBe(false);
    expect(frontierCycleActionEnabled("navigate_frontier", current)).toBe(true);
    expect(frontierCycleActionEnabled("complete_assignment", current)).toBe(true);
  });

  it("requires a new survey after the world revision changes", () => {
    expect(cycle([surveyReceipt()], 8)).toMatchObject({
      phase: "survey_required",
      reason: "no_current_survey",
      current_world_revision: 8
    });
  });

  it("accepts an explicitly granted current survey from another agent", () => {
    const state = cycle([surveyReceipt()], 7, "agent_b", [{
      name: "survey_terrain",
      transaction_id: surveyTransaction
    }]);
    expect(state).toMatchObject({
      phase: "choice_required",
      current_survey_transaction_id: surveyTransaction
    });
  });

  it("does not expose navigation from an unrelated agent's survey", () => {
    expect(cycle([surveyReceipt()], 7, "agent_b")).toMatchObject({
      phase: "survey_required",
      current_survey_transaction_id: null
    });
  });

  it("allows a different survey when the current radius found no choices", () => {
    const empty = surveyReceipt();
    empty.detail = { frontier: [] };
    expect(cycle([empty])).toMatchObject({
      phase: "survey_required",
      reason: "no_reachable_choices",
      current_survey_transaction_id: surveyTransaction
    });
  });

  it("does not constrain agents lacking the paired frontier protocol", () => {
    expect(frontierCycleState({
      agent: { id: "agent_a", capabilities: ["survey_terrain"], references: [] },
      receipts: [surveyReceipt()],
      currentWorldRevision: 7
    })).toBeNull();
  });
});

function cycle(
  receipts: ActionReceipt[],
  currentWorldRevision = 7,
  agentId = "agent_a",
  references: Array<{ name: string; transaction_id: string }> = []
) {
  return frontierCycleState({
    agent: {
      id: agentId,
      capabilities: ["survey_terrain", "navigate_frontier"],
      references
    },
    receipts,
    currentWorldRevision
  });
}

function surveyReceipt(): ActionReceipt {
  return {
    transaction_id: surveyTransaction,
    agent_id: "agent_a",
    agent_name: "Survey agent",
    kind: "tool",
    name: "survey_terrain",
    input: { radius_cells: 12 },
    accepted: true,
    code: "terrain_survey",
    detail: {
      frontier: [
        {
          choice_id: "frontier_a",
          target: { x: 1, y: 0.38, z: 2 },
          face_point: { x: 2, y: 0.38, z: 2 }
        },
        {
          choice_id: "frontier_b",
          target: { x: -2, y: 0.38, z: 3 },
          face_point: { x: -3, y: 0.38, z: 3 }
        }
      ]
    },
    world_before_frame: 10,
    world_before_revision: 7,
    world_after_frame: 10,
    frame_count: 0,
    world_revision: 7,
    channels: [],
    gates: [],
    committed_at: "2026-08-02T00:00:00.000Z"
  };
}
