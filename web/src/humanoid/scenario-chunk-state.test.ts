import { describe, expect, it } from "vitest";
import type { ScenarioChunkDeltaState, ScenarioDefinition } from "../types";
import {
  changedScenarioVisualChunkIds,
  resolveScenarioWorldVisualState
} from "./scenario-chunk-state";

const scenario: ScenarioDefinition = {
  title: "Chunk visual fixture",
  seed: 7,
  bounds: { width: 20, depth: 10 },
  visibility_radius: 8,
  robot: { x: 2, z: 2, yaw: 0 },
  obstacles: [{
    id: "wall",
    center: { x: 3, y: 0.5, z: 3 },
    size: { x: 1, y: 1, z: 1 }
  }],
  objects: [{
    id: "crate",
    kind: "crate",
    color: "#bb7744",
    position: { x: 4, y: 0.25, z: 4 },
    size: { x: 0.5, y: 0.5, z: 0.5 },
    portable: true
  }],
  zones: [{
    id: "target",
    color: "#55aa88",
    center: { x: 14, y: 0.01, z: 4 },
    size: { x: 2, y: 0.02, z: 2 }
  }],
  default_goal: { summary: "Reach target", predicates: [] },
  chunk_manifest: {
    version: 1,
    chunk_size: 10,
    grid: { columns: 2, rows: 1 },
    chunks: [{
      id: "chunk_0_0",
      coordinate: { column: 0, row: 0 },
      bounds: { minimum: { x: 0, z: 0 }, maximum: { x: 10, z: 10 } },
      entity_ids: { obstacles: ["wall"], objects: ["crate"], zones: [] }
    }, {
      id: "chunk_1_0",
      coordinate: { column: 1, row: 0 },
      bounds: { minimum: { x: 10, z: 0 }, maximum: { x: 20, z: 10 } },
      entity_ids: { obstacles: [], objects: [], zones: ["target"] }
    }]
  }
};

describe("scenario chunk visual state", () => {
  it("projects created, removed and migrated entities into their current chunks", () => {
    const state = resolveScenarioWorldVisualState(scenario, deltas());

    expect(state.revision).toBe(3);
    expect(state.chunks.get("chunk_0_0")).toMatchObject({
      blocks: [{ id: "placed:block" }],
      zones: [],
      objects: []
    });
    expect(state.chunks.get("chunk_1_0")).toMatchObject({
      blocks: [],
      zones: [],
      objects: [{ id: "crate", position: { x: 15, z: 4 } }]
    });
    expect(state.objects.get("crate")).toMatchObject({ position: { x: 15, z: 4 } });
  });

  it("rejects a delta stream from another world seed", () => {
    expect(() => resolveScenarioWorldVisualState(scenario, {
      ...deltas(),
      scenario_seed: 8
    })).toThrow("世界种子不一致");
  });

  it("identifies only chunks whose rendered contents changed", () => {
    const initial = resolveScenarioWorldVisualState(scenario, {
      ...deltas(),
      revision: 0,
      changed_chunk_ids: [],
      chunks: []
    });
    const next = resolveScenarioWorldVisualState(scenario, deltas());

    expect(changedScenarioVisualChunkIds(initial, next)).toEqual([
      "chunk_0_0",
      "chunk_1_0"
    ]);
    expect(changedScenarioVisualChunkIds(next, {
      ...next,
      revision: next.revision + 1
    })).toEqual([]);
  });
});

function deltas(): ScenarioChunkDeltaState {
  return {
    version: 1,
    scenario_seed: 7,
    scenario_sha256: "a".repeat(64),
    manifest_version: 1,
    revision: 3,
    changed_chunk_ids: ["chunk_0_0", "chunk_1_0"],
    chunks: [{
      chunk_id: "chunk_0_0",
      revision: 3,
      blocks: [{
        id: "wall",
        origin: "scenario",
        present: false,
        center: { x: 3, y: 0.5, z: 3 },
        size: { x: 1, y: 1, z: 1 },
        material: "solid",
        properties: {}
      }, {
        id: "placed:block",
        origin: "created",
        present: true,
        center: { x: 6, y: 0.5, z: 6 },
        size: { x: 1, y: 1, z: 1 },
        material: "stone",
        properties: {}
      }],
      zones: [],
      dynamic_entities: []
    }, {
      chunk_id: "chunk_1_0",
      revision: 3,
      blocks: [],
      zones: [{
        id: "target",
        origin: "scenario",
        present: true,
        color: "#55aa88",
        center: { x: 14, y: 0.01, z: 4 },
        size: { x: 2, y: 0.02, z: 2 },
        enabled: false,
        properties: {}
      }],
      dynamic_entities: [{
        id: "crate",
        origin: "scenario",
        present: true,
        kind: "crate",
        color: "#bb7744",
        position: { x: 15, y: 0.25, z: 4 },
        rotation: { x: 0, y: 0, z: 0, w: 1 },
        linear_velocity: { x: 0, y: 0, z: 0 },
        angular_velocity: { x: 0, y: 0, z: 0 },
        size: { x: 0.5, y: 0.5, z: 0.5 },
        portable: true,
        properties: {}
      }]
    }]
  };
}
