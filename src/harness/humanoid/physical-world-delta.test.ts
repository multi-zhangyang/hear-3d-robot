import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { ScenarioSchema } from "../../domain/schema.js";
import { resolveScenarioChunkDeltaEntity } from "../../domain/scenario-chunk-delta.js";
import { createScenarioChunkDeltaState } from "../../domain/scenario-chunk-delta-schema.js";
import { RunStore } from "../../persistence/run-store.js";
import {
  assertHumanoidPhysicalWorldDeltaRecovery,
  captureHumanoidPhysicalWorldDelta,
  projectHumanoidPhysicalWorldDelta,
  reconcileHumanoidPhysicalWorldDelta
} from "./physical-world-delta.js";

const scenario = ScenarioSchema.parse({
  title: "Humanoid physical persistence fixture",
  seed: 91,
  bounds: { width: 20, depth: 20 },
  visibility_radius: 8,
  robot: { x: 2, z: 2, yaw: 0 },
  obstacles: [],
  objects: [{
    id: "crate",
    kind: "crate",
    color: "#bb7744",
    position: { x: 3, y: 0.25, z: 3 },
    size: { x: 0.5, y: 0.5, z: 0.5 },
    portable: true
  }, {
    id: "cabinet-frame",
    kind: "cabinet",
    color: "#59645d",
    position: { x: 8, y: 1, z: 8 },
    size: { x: 1.5, y: 1.7, z: 0.72 },
    portable: false,
    capability: {
      shape: "box",
      mass_kg: 18,
      affordances: ["container"],
      interaction_points: [],
      container: {
        interior_center: { x: 0, y: 0, z: 0 },
        interior_size: { x: 1.4, y: 1.6, z: 0.65 },
        opening_direction: { x: 0, y: 0, z: -1 },
        wall_thickness_m: 0.045
      }
    }
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

describe("humanoid physical world delta", () => {
  it("captures authoritative MuJoCo kinematics and durably reconciles them", async () => {
    const chunks = createScenarioChunkDeltaState(scenario);
    const record = captureHumanoidPhysicalWorldDelta({
      scenario,
      chunks,
      transactionId: "execute-physical-1",
      world: physicalFrame(84, 84, 14, 13)
    });
    expect(record).toMatchObject({
      transaction_id: "execute-physical-1",
      source_world_frame: 84,
      source_world_revision: 84,
      base_chunk_revision: 0,
      entities: [{
        id: "crate",
        position: { x: 14, y: 0.25, z: 13 },
        physical_authority: { source: "humanoid_mujoco" }
      }]
    });
    const projected = projectHumanoidPhysicalWorldDelta(scenario, chunks, record!);

    const runsDir = await mkdtemp(join(tmpdir(), "hear-physical-delta-"));
    const store = await RunStore.create(runsDir, {
      mission: "Persist the real object state",
      scenarioId: "physical_fixture",
      scenario,
      goal: scenario.default_goal
    });
    const committed = await reconcileHumanoidPhysicalWorldDelta(store, record!);
    const replayed = await reconcileHumanoidPhysicalWorldDelta(store, record!);

    expect(committed.revision).toBe(1);
    expect(committed).toEqual(projected);
    expect(replayed).toEqual(committed);
    expect(resolveScenarioChunkDeltaEntity(scenario, committed, "crate")).toMatchObject({
      chunk_id: "chunk_1_1",
      state: {
        position: { x: 14, y: 0.25, z: 13 },
        linear_velocity: { x: 0.1, y: 0, z: -0.05 },
        physical_authority: {
          transaction_id: "execute-physical-1",
          world_revision: 84
        }
      }
    });
    expect(() => assertHumanoidPhysicalWorldDeltaRecovery({
      scenario,
      chunks: committed,
      world: physicalFrame(84, 84, 14, 13)
    })).not.toThrow();
    expect(() => assertHumanoidPhysicalWorldDeltaRecovery({
      scenario,
      chunks: committed,
      world: physicalFrame(83, 83, 14, 13)
    })).toThrow("ahead of the physical checkpoint");
    expect(() => assertHumanoidPhysicalWorldDeltaRecovery({
      scenario,
      chunks: committed,
      world: physicalFrame(84, 84, 13, 13)
    })).toThrow("conflicts with the physical checkpoint");
  });

  it("does not create a world mutation when physical kinematics are unchanged", () => {
    expect(captureHumanoidPhysicalWorldDelta({
      scenario,
      chunks: createScenarioChunkDeltaState(scenario),
      transactionId: "execute-stationary",
      world: physicalFrame(10, 10, 3, 3, true)
    })).toBeUndefined();
  });
});

function physicalFrame(
  frame: number,
  worldRevision: number,
  x: number,
  z: number,
  stationary = false
) {
  return {
    frame,
    worldRevision,
    robot: {
      objects: {
        crate: {
          id: "crate",
          position: { x, y: 0.25, z },
          rotation: { x: 0, y: 0, z: 0, w: 1 },
          linearVelocity: stationary
            ? { x: 0, y: 0, z: 0 }
            : { x: 0.1, y: 0, z: -0.05 },
          angularVelocity: { x: 0, y: 0, z: 0 }
        },
        "cabinet-frame": {
          id: "cabinet-frame",
          position: { x: 8, y: 1, z: 8 },
          rotation: { x: 0, y: 0, z: 0, w: 1 },
          linearVelocity: { x: 0, y: 0, z: 0 },
          angularVelocity: { x: 0, y: 0, z: 0 }
        }
      }
    }
  };
}
