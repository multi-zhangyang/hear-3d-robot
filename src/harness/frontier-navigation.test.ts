import { describe, expect, it } from "vitest";
import type { ActionReceipt } from "../domain/schema.js";
import { resolveFrontierNavigation } from "./frontier-navigation.js";

const transactionId = "agent_surveyor:survey_1";
const choice = {
  choice_id: "frontier_selected",
  target: { x: 4, y: 0.38, z: -2 },
  face_point: { x: 5, y: 0.38, z: -2 },
  travel_distance: 4.4,
  unseen_neighbours: 3,
  turn_degrees: 42,
  motion_style: "probe"
} as const;

describe("frontier navigation provenance", () => {
  it("resolves only the exact model-selected choice from a current owned survey", () => {
    const result = resolve(receipt(), {
      survey_transaction_id: transactionId,
      choice_id: choice.choice_id
    });
    expect(result).toEqual({
      ok: true,
      input: {
        survey_transaction_id: transactionId,
        survey_world_revision: 7,
        choice_id: choice.choice_id,
        target: choice.target,
        face_point: choice.face_point
      }
    });
  });

  it("does not substitute another candidate for an unknown model choice", () => {
    const result = resolve(receipt(), {
      survey_transaction_id: transactionId,
      choice_id: "frontier_invented"
    });
    expect(result).toMatchObject({
      ok: false,
      result: {
        accepted: false,
        code: "unknown_frontier_choice",
        detail: { available_choice_ids: [choice.choice_id] }
      }
    });
  });

  it("rejects a survey after any body revision change", () => {
    const result = resolve(receipt(), {
      survey_transaction_id: transactionId,
      choice_id: choice.choice_id
    }, 8);
    expect(result).toMatchObject({
      ok: false,
      result: {
        code: "stale_survey_revision",
        detail: { surveyed_world_revision: 7, current_world_revision: 8 }
      }
    });
  });

  it("requires another agent's survey to be explicitly granted", () => {
    const result = resolve(receipt(), {
      survey_transaction_id: transactionId,
      choice_id: choice.choice_id
    }, 7, "agent_executor");
    expect(result).toMatchObject({
      ok: false,
      result: { code: "survey_transaction_not_granted" }
    });
  });
});

function resolve(
  source: ActionReceipt,
  rawInput: unknown,
  currentWorldRevision = 7,
  agentId = "agent_surveyor"
) {
  return resolveFrontierNavigation({
    rawInput,
    agent: { id: agentId, references: [] },
    currentWorldRevision,
    lookupReceipt: (id) => id === transactionId ? source : undefined
  });
}

function receipt(): ActionReceipt {
  return {
    transaction_id: transactionId,
    agent_id: "agent_surveyor",
    agent_name: "Surveyor",
    kind: "tool",
    name: "survey_terrain",
    input: { radius_cells: 12 },
    accepted: true,
    code: "terrain_survey",
    detail: { frontier: [choice] },
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
