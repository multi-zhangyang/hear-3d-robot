import { describe, expect, it } from "vitest";
import { ScenarioSchema } from "./schema.js";
import {
  applyScenarioChunkDeltaMutations,
  resolveScenarioChunkDeltaEntity
} from "./scenario-chunk-delta.js";
import { createScenarioChunkDeltaState } from "./scenario-chunk-delta-schema.js";
import {
  ScenarioPhysicalWorldDeltaSchema,
  assertScenarioPhysicalWorldDeltaApplied,
  createScenarioPhysicalWorldDelta,
  scenarioPhysicalWorldDeltaMutations
} from "./scenario-physical-delta.js";

const scenario = ScenarioSchema.parse({
  title: "Physical chunk commit fixture",
  seed: 71,
  bounds: { width: 30, depth: 20 },
  visibility_radius: 8,
  robot: { x: 2, z: 2, yaw: 0 },
  obstacles: [],
  objects: [{
    id: "crate",
    kind: "crate",
    color: "#b47742",
    position: { x: 3, y: 0.25, z: 3 },
    size: { x: 0.5, y: 0.5, z: 0.5 },
    portable: true
  }],
  zones: [],
  default_goal: {
    summary: "Move the crate.",
    predicates: [{
      type: "object_at",
      object_id: "crate",
      target: { x: 3, y: 0.25, z: 3 },
      tolerance: 0.2
    }]
  }
});

describe("physical world chunk commits", () => {
  it("moves a MuJoCo body with action and frame provenance at one chunk revision", () => {
    const initial = createScenarioChunkDeltaState(scenario);
    const record = physicalRecord("action-1", 120, 120, initial.revision, 24, 14);
    const state = applyScenarioChunkDeltaMutations(
      scenario,
      initial,
      scenarioPhysicalWorldDeltaMutations(scenario, initial, record)
    );

    expect(state.revision).toBe(1);
    expect(state.changed_chunk_ids).toEqual(["chunk_0_0", "chunk_2_1"]);
    expect(resolveScenarioChunkDeltaEntity(scenario, state, "crate")).toMatchObject({
      chunk_id: "chunk_2_1",
      state: {
        position: { x: 24, y: 0.25, z: 14 },
        physical_authority: {
          source: "humanoid_mujoco",
          transaction_id: "action-1",
          world_frame: 120,
          world_revision: 120
        }
      }
    });
    expect(() => assertScenarioPhysicalWorldDeltaApplied(scenario, state, record))
      .not.toThrow();
    expect(scenarioPhysicalWorldDeltaMutations(scenario, state, record)).toEqual([]);
  });

  it("never lets an older recovered action overwrite a newer physical body state", () => {
    const initial = createScenarioChunkDeltaState(scenario);
    const older = physicalRecord("action-1", 120, 120, initial.revision, 12, 8);
    const first = applyScenarioChunkDeltaMutations(
      scenario,
      initial,
      scenarioPhysicalWorldDeltaMutations(scenario, initial, older)
    );
    const newer = physicalRecord("action-2", 180, 180, first.revision, 24, 14);
    const second = applyScenarioChunkDeltaMutations(
      scenario,
      first,
      scenarioPhysicalWorldDeltaMutations(scenario, first, newer)
    );

    expect(scenarioPhysicalWorldDeltaMutations(scenario, second, older)).toEqual([]);
    expect(() => assertScenarioPhysicalWorldDeltaApplied(scenario, second, older))
      .not.toThrow();
    expect(resolveScenarioChunkDeltaEntity(scenario, second, "crate")).toMatchObject({
      state: {
        position: { x: 24, z: 14 },
        physical_authority: { transaction_id: "action-2", world_revision: 180 }
      }
    });
  });

  it("rejects conflicting physical identities and tampered records", () => {
    const initial = createScenarioChunkDeltaState(scenario);
    const firstRecord = physicalRecord("action-1", 120, 120, initial.revision, 12, 8);
    const first = applyScenarioChunkDeltaMutations(
      scenario,
      initial,
      scenarioPhysicalWorldDeltaMutations(scenario, initial, firstRecord)
    );
    const conflict = physicalRecord("action-conflict", 120, 120, initial.revision, 13, 8);
    expect(() => scenarioPhysicalWorldDeltaMutations(scenario, first, conflict)).toThrow(
      "conflicts at world revision 120"
    );

    const tampered = structuredClone(firstRecord);
    tampered.entities[0]!.position.x += 1;
    expect(() => ScenarioPhysicalWorldDeltaSchema.parse(tampered)).toThrow(
      "integrity hash does not match"
    );
  });
});

function physicalRecord(
  transactionId: string,
  worldFrame: number,
  worldRevision: number,
  baseChunkRevision: number,
  x: number,
  z: number
) {
  return createScenarioPhysicalWorldDelta({
    version: 1,
    transaction_id: transactionId,
    source_world_frame: worldFrame,
    source_world_revision: worldRevision,
    base_chunk_revision: baseChunkRevision,
    entities: [{
      id: "crate",
      position: { x, y: 0.25, z },
      rotation: { x: 0, y: 0, z: 0, w: 1 },
      linear_velocity: { x: 0, y: 0, z: 0 },
      angular_velocity: { x: 0, y: 0, z: 0 },
      physical_authority: {
        source: "humanoid_mujoco",
        transaction_id: transactionId,
        world_frame: worldFrame,
        world_revision: worldRevision
      }
    }]
  });
}
