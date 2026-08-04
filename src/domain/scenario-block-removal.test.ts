import { describe, expect, it } from "vitest";
import { ScenarioSchema } from "./schema.js";
import {
  applyScenarioChunkDeltaMutation,
  applyScenarioChunkDeltaMutations,
  materializeScenarioChunkDeltaState,
  resolveScenarioChunkDeltaEntity
} from "./scenario-chunk-delta.js";
import { createScenarioChunkDeltaState } from "./scenario-chunk-delta-schema.js";
import {
  ScenarioBlockRemovalTransactionSchema,
  assertScenarioBlockRemovalApplied,
  createScenarioBlockRemovalTransaction,
  scenarioBlockRemovalMutations
} from "./scenario-block-removal.js";

const scenario = ScenarioSchema.parse({
  title: "Authorized block-removal fixture",
  seed: 83,
  bounds: { width: 24, depth: 24 },
  visibility_radius: 10,
  robot: { x: 2, z: 2, yaw: 0 },
  obstacles: [{
    id: "stone-4-3",
    center: { x: 4, y: 0.5, z: 3 },
    size: { x: 1, y: 1, z: 1 }
  }],
  objects: [{
    id: "pedestal",
    kind: "fixture",
    color: "#888888",
    position: { x: 8, y: 0.5, z: 8 },
    size: { x: 1, y: 1, z: 1 },
    portable: false
  }],
  zones: [],
  default_goal: {
    summary: "Reach the block.",
    predicates: [{
      type: "robot_at",
      target: { x: 3, y: 0, z: 3 },
      tolerance: 0.3
    }]
  }
});

describe("evidence-authorized scenario block removal", () => {
  it("atomically persists a provenance-bound tombstone and reconciles idempotently", () => {
    const initial = createScenarioChunkDeltaState(scenario);
    const transaction = removal(initial);
    const committed = applyScenarioChunkDeltaMutations(
      scenario,
      initial,
      scenarioBlockRemovalMutations(scenario, initial, transaction)
    );

    expect(committed.revision).toBe(1);
    expect(resolveScenarioChunkDeltaEntity(scenario, committed, "stone-4-3"))
      .toMatchObject({
        category: "block",
        state: {
          present: false,
          properties: {
            hear_block_removal_v1: {
              transaction_id: "remove-call-1",
              execution_transaction_id: "execute-call-1",
              source_world_revision: 140
            }
          }
        }
      });
    expect(materializeScenarioChunkDeltaState(scenario, committed).obstacles).toEqual([]);
    expect(() => assertScenarioBlockRemovalApplied(scenario, committed, transaction))
      .not.toThrow();
    expect(scenarioBlockRemovalMutations(scenario, committed, transaction)).toEqual([]);
  });

  it("rejects a changed base state and removal of a fixed object", () => {
    const initial = createScenarioChunkDeltaState(scenario);
    const transaction = removal(initial);
    const changed = applyScenarioChunkDeltaMutation(scenario, initial, {
      type: "put_block",
      block: {
        id: "stone-4-3",
        center: { x: 4, y: 0.5, z: 3 },
        size: { x: 1, y: 1, z: 1 },
        material: "stone",
        properties: { weathered: true }
      }
    });

    expect(() => scenarioBlockRemovalMutations(scenario, changed, transaction))
      .toThrow("base chunk state changed");
    expect(() => createScenarioBlockRemovalTransaction({
      scenario,
      chunks: initial,
      transactionId: "remove-fixture",
      solidId: "object-pedestal",
      executionTransactionId: "execute-fixture",
      planningTransactionId: "plan-fixture",
      sourceWorldFrame: 140,
      sourceWorldRevision: 140,
      contactEvidence: contactEvidence()
    })).toThrow(/Unknown scenario entity|not a removable block/);
  });

  it("rejects tampered force evidence and transaction projections", () => {
    const transaction = removal(createScenarioChunkDeltaState(scenario));
    const weak = structuredClone(transaction);
    weak.contact_evidence.observed_maximum_normal_force_n = 1;
    expect(ScenarioBlockRemovalTransactionSchema.safeParse(weak).success).toBe(false);

    const tampered = structuredClone(transaction);
    tampered.projected_chunk_state_sha256 = "0".repeat(64);
    expect(ScenarioBlockRemovalTransactionSchema.safeParse(tampered).success).toBe(false);
  });
});

function removal(chunks: ReturnType<typeof createScenarioChunkDeltaState>) {
  return createScenarioBlockRemovalTransaction({
    scenario,
    chunks,
    transactionId: "remove-call-1",
    solidId: "stone-4-3",
    executionTransactionId: "execute-call-1",
    planningTransactionId: "plan-call-1",
    sourceWorldFrame: 140,
    sourceWorldRevision: 140,
    contactEvidence: contactEvidence()
  });
}

function contactEvidence() {
  return {
    predicate_index: 0,
    predicate_type: "hand_contact_solid" as const,
    surface_kind: "hand_surface" as const,
    surface: "left_hand_palm_link",
    planned_stable_frames: 10,
    observed_stable_frames: 10,
    planned_minimum_normal_force_n: 7,
    observed_maximum_normal_force_n: 11
  };
}
