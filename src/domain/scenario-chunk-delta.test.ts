import { describe, expect, it } from "vitest";
import { ScenarioSchema } from "./schema.js";
import {
  applyScenarioChunkDeltaMutation,
  applyScenarioChunkDeltaMutations,
  materializeScenarioChunkDeltaState,
  resolveScenarioChunkDeltaContents,
  resolveScenarioChunkDeltaEntity,
  type ResolvedScenarioChunkContents
} from "./scenario-chunk-delta.js";
import {
  createScenarioChunkDeltaState,
  restoreScenarioChunkDeltaState
} from "./scenario-chunk-delta-schema.js";

const scenario = ScenarioSchema.parse({
  title: "Persistent chunk delta fixture",
  seed: 41,
  bounds: { width: 30, depth: 24 },
  visibility_radius: 8,
  robot: { x: 2, z: 2, yaw: 0 },
  obstacles: [{
    id: "north_block",
    center: { x: 4, y: 1, z: 18 },
    size: { x: 1, y: 2, z: 1 }
  }],
  objects: [{
    id: "rover",
    kind: "crate",
    color: "#a46d3c",
    position: { x: 4, y: 0.25, z: 4 },
    size: { x: 0.5, y: 0.5, z: 0.5 },
    portable: true
  }],
  zones: [{
    id: "east_zone",
    color: "#55aa88",
    center: { x: 20, y: 0.01, z: 4 },
    size: { x: 2, y: 0.02, z: 2 }
  }],
  default_goal: {
    summary: "Reach the east zone.",
    predicates: [{ type: "robot_in_zone", zone_id: "east_zone", tolerance: 0.2 }]
  }
});

describe("persistent scenario chunk deltas", () => {
  it("starts as a versioned sparse overlay without enumerating the world grid", () => {
    const state = createScenarioChunkDeltaState(scenario);

    expect(scenario.chunk_manifest.chunks).toHaveLength(6);
    expect(state).toMatchObject({
      version: 1,
      manifest_version: 1,
      scenario_seed: 41,
      revision: 0,
      changed_chunk_ids: [],
      chunks: []
    });
    expect(state.scenario_sha256).toMatch(/^[a-f0-9]{64}$/);
  });

  it("marks both owners at one revision when migration vacates the old chunk", () => {
    const state = applyScenarioChunkDeltaMutation(
      scenario,
      createScenarioChunkDeltaState(scenario),
      { type: "put_dynamic_entity", entity: dynamic("rover", 26, 18) }
    );

    expect(state.revision).toBe(1);
    expect(state.changed_chunk_ids).toEqual(["chunk_0_0", "chunk_2_1"]);
    expect(state.chunks.map(({ chunk_id, revision }) => ({ chunk_id, revision }))).toEqual([
      { chunk_id: "chunk_2_1", revision: 1 }
    ]);
    expect(resolveScenarioChunkDeltaContents(scenario, state, "chunk_0_0").dynamic_entities)
      .toEqual([]);
    expect(resolveScenarioChunkDeltaContents(scenario, state, "chunk_2_1").dynamic_entities)
      .toMatchObject([{ id: "rover" }]);
  });

  it("moves a dynamic entity across chunks with one stable ID and one atomic owner", () => {
    let state = createScenarioChunkDeltaState(scenario);
    state = applyScenarioChunkDeltaMutation(scenario, state, {
      type: "create_block",
      block: block("placed:block-1", 3, 3)
    });
    state = applyScenarioChunkDeltaMutation(scenario, state, {
      type: "put_dynamic_entity",
      entity: dynamic("rover", 26, 18)
    });

    expect(state.revision).toBe(2);
    expect(state.changed_chunk_ids).toEqual(["chunk_0_0", "chunk_2_1"]);
    expect(state.chunks.map(({ chunk_id, revision }) => ({ chunk_id, revision }))).toEqual([
      { chunk_id: "chunk_0_0", revision: 2 },
      { chunk_id: "chunk_2_1", revision: 2 }
    ]);
    expect(state.chunks.flatMap(({ dynamic_entities }) => dynamic_entities))
      .toHaveLength(1);
    expect(resolveScenarioChunkDeltaEntity(scenario, state, "rover")).toMatchObject({
      category: "dynamic_entity",
      chunk_id: "chunk_2_1",
      state: { id: "rover", origin: "scenario", present: true, position: { x: 26, z: 18 } }
    });
    const origin: ResolvedScenarioChunkContents = resolveScenarioChunkDeltaContents(
      scenario,
      state,
      "chunk_0_0"
    );
    expect(origin).toMatchObject({
      blocks: [{ id: "placed:block-1" }],
      dynamic_entities: []
    });
    expect(resolveScenarioChunkDeltaContents(scenario, state, "chunk_2_1")).toMatchObject({
      dynamic_entities: [{ id: "rover", position: { x: 26, z: 18 } }]
    });
  });

  it("publishes a multi-entity world transaction at one shared revision", () => {
    const state = applyScenarioChunkDeltaMutations(
      scenario,
      createScenarioChunkDeltaState(scenario),
      [
        { type: "create_block", block: block("placed:bridge", 3, 3) },
        { type: "put_dynamic_entity", entity: dynamic("rover", 26, 18) },
        {
          type: "put_zone",
          zone: {
            id: "east_zone",
            color: "#55aa88",
            center: { x: 20, y: 0.01, z: 4 },
            size: { x: 2, y: 0.02, z: 2 },
            enabled: false,
            properties: {}
          }
        }
      ]
    );

    expect(state.revision).toBe(1);
    expect(state.changed_chunk_ids).toEqual([
      "chunk_0_0",
      "chunk_1_0",
      "chunk_2_1"
    ]);
    expect(state.chunks.every(({ revision }) => revision === 1)).toBe(true);
    expect(resolveScenarioChunkDeltaEntity(scenario, state, "rover")).toMatchObject({
      chunk_id: "chunk_2_1",
      state: { position: { x: 26, z: 18 } }
    });
    expect(resolveScenarioChunkDeltaEntity(scenario, state, "east_zone")).toMatchObject({
      chunk_id: "chunk_1_0",
      state: { enabled: false }
    });
  });

  it("persists block removal, zone state and created-entity tombstones", () => {
    let state = createScenarioChunkDeltaState(scenario);
    state = applyScenarioChunkDeltaMutation(scenario, state, {
      type: "remove_block",
      entity_id: "north_block"
    });
    state = applyScenarioChunkDeltaMutation(scenario, state, {
      type: "put_zone",
      zone: {
        id: "east_zone",
        color: "#aa7755",
        center: { x: 20, y: 0.01, z: 4 },
        size: { x: 2, y: 0.02, z: 2 },
        enabled: false,
        properties: { reason: "temporarily_closed" }
      }
    });
    state = applyScenarioChunkDeltaMutation(scenario, state, {
      type: "create_dynamic_entity",
      entity: dynamic("spawned:carrier-1", 8, 18)
    });
    state = applyScenarioChunkDeltaMutation(scenario, state, {
      type: "remove_dynamic_entity",
      entity_id: "spawned:carrier-1"
    });

    expect(resolveScenarioChunkDeltaEntity(scenario, state, "north_block").state)
      .toMatchObject({ origin: "scenario", present: false });
    expect(resolveScenarioChunkDeltaEntity(scenario, state, "east_zone").state)
      .toMatchObject({ origin: "scenario", present: true, enabled: false });
    expect(resolveScenarioChunkDeltaEntity(scenario, state, "spawned:carrier-1").state)
      .toMatchObject({ origin: "created", present: false });
    expect(() => applyScenarioChunkDeltaMutation(scenario, state, {
      type: "create_block",
      block: block("spawned:carrier-1", 2, 2)
    })).toThrow("already exists");
  });

  it("materializes the current sparse overlay for physics and navigation", () => {
    const state = applyScenarioChunkDeltaMutations(
      scenario,
      createScenarioChunkDeltaState(scenario),
      [
        { type: "remove_block", entity_id: "north_block" },
        { type: "create_block", block: block("placed:ramp", 12, 10) },
        { type: "put_dynamic_entity", entity: dynamic("rover", 16, 12) },
        {
          type: "put_zone",
          zone: {
            id: "east_zone",
            color: "#55aa88",
            center: { x: 20, y: 0.01, z: 4 },
            size: { x: 2, y: 0.02, z: 2 },
            enabled: false,
            properties: {}
          }
        }
      ]
    );
    const current = materializeScenarioChunkDeltaState(scenario, state);

    expect(current.obstacles.map(({ id }) => id)).toEqual(["placed:ramp"]);
    expect(current.objects).toMatchObject([{
      id: "rover",
      position: { x: 16, y: 0.25, z: 12 }
    }]);
    expect(current.zones).toEqual([]);
    expect(current.chunk_manifest.chunks.flatMap(({ entity_ids }) => entity_ids.obstacles))
      .toEqual(["placed:ramp"]);
  });

  it("rejects unknown updates and cross-category identity reuse", () => {
    const state = createScenarioChunkDeltaState(scenario);

    expect(() => applyScenarioChunkDeltaMutation(scenario, state, {
      type: "remove_dynamic_entity",
      entity_id: "missing"
    })).toThrow("Unknown scenario entity");
    expect(() => applyScenarioChunkDeltaMutation(scenario, state, {
      type: "put_block",
      block: block("rover", 2, 2)
    })).toThrow("is dynamic_entity, not block");
    expect(() => applyScenarioChunkDeltaMutation(scenario, state, {
      type: "create_zone",
      zone: {
        id: "north_block",
        color: "#fff",
        center: { x: 2, y: 0.01, z: 2 },
        size: { x: 1, y: 0.02, z: 1 },
        enabled: true,
        properties: {}
      }
    })).toThrow("already exists");
  });

  it("bounds per-entity property size and nesting depth", () => {
    const state = createScenarioChunkDeltaState(scenario);
    expect(() => applyScenarioChunkDeltaMutation(scenario, state, {
      type: "create_block",
      block: {
        ...block("placed:oversized", 2, 2),
        properties: { payload: "x".repeat(16 * 1024) }
      }
    })).toThrow("cannot exceed 16384 UTF-8 bytes");

    let nested: unknown = true;
    for (let depth = 0; depth < 9; depth += 1) nested = { next: nested };
    expect(() => applyScenarioChunkDeltaMutation(scenario, state, {
      type: "create_block",
      block: {
        ...block("placed:too-deep", 2, 2),
        properties: { nested } as never
      }
    })).toThrow("cannot exceed nesting depth 8");
  });

  it("hard-rejects duplicated, unknown and wrongly owned persisted records", () => {
    const valid = applyScenarioChunkDeltaMutation(
      scenario,
      createScenarioChunkDeltaState(scenario),
      { type: "create_block", block: block("placed:block-1", 3, 3) }
    );
    const duplicated = structuredClone(valid);
    duplicated.chunks[0]!.blocks.push(structuredClone(duplicated.chunks[0]!.blocks[0]!));
    expect(() => restoreScenarioChunkDeltaState(scenario, duplicated)).toThrow(
      "entity records must be unique and sorted"
    );

    const unknown = structuredClone(valid);
    unknown.chunks[0]!.blocks[0]!.origin = "scenario";
    expect(() => restoreScenarioChunkDeltaState(scenario, unknown)).toThrow(
      "references unknown block entity"
    );

    const wrongOwner = structuredClone(valid);
    wrongOwner.chunks[0]!.chunk_id = "chunk_1_0";
    expect(() => restoreScenarioChunkDeltaState(scenario, wrongOwner)).toThrow(
      "must belong to chunk_0_0"
    );
  });

  it("binds persisted deltas to the exact scenario baseline", () => {
    const state = createScenarioChunkDeltaState(scenario);
    const changed = ScenarioSchema.parse({
      ...scenario,
      chunk_manifest: undefined,
      seed: 42
    });

    expect(() => restoreScenarioChunkDeltaState(changed, state)).toThrow(
      "scenario seed does not match"
    );
  });
});

function block(id: string, x: number, z: number) {
  return {
    id,
    center: { x, y: 0.5, z },
    size: { x: 1, y: 1, z: 1 },
    material: "stone",
    properties: {}
  };
}

function dynamic(id: string, x: number, z: number) {
  return {
    id,
    kind: "crate",
    color: "#a46d3c",
    position: { x, y: 0.25, z },
    rotation: { x: 0, y: 0, z: 0, w: 1 },
    linear_velocity: { x: 0, y: 0, z: 0 },
    angular_velocity: { x: 0, y: 0, z: 0 },
    size: { x: 0.5, y: 0.5, z: 0.5 },
    portable: true,
    properties: {}
  };
}
