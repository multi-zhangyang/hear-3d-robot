import { describe, expect, it } from "vitest";
import { ScenarioSchema } from "../domain/schema.js";
import {
  queryScenarioChunksInCircle,
  queryScenarioChunksInRectangle
} from "./scenario-chunks.js";

const scenario = ScenarioSchema.parse({
  title: "Active region fixture",
  seed: 11,
  bounds: { width: 36, depth: 36 },
  visibility_radius: 8,
  robot: { x: 2, z: 2, yaw: 0 },
  obstacles: [
    {
      id: "origin_block",
      center: { x: 2, y: 1, z: 2 },
      size: { x: 1, y: 2, z: 1 }
    },
    {
      id: "north_block",
      center: { x: 18, y: 1, z: 4 },
      size: { x: 1, y: 2, z: 1 }
    }
  ],
  objects: [{
    id: "center_crate",
    kind: "crate",
    color: "#a46d3c",
    position: { x: 18, y: 0.25, z: 18 },
    size: { x: 0.5, y: 0.5, z: 0.5 },
    portable: true
  }],
  zones: [{
    id: "far_zone",
    color: "#55aa88",
    center: { x: 34, y: 0.01, z: 34 },
    size: { x: 1, y: 0.02, z: 1 }
  }],
  default_goal: {
    summary: "Reach the far zone.",
    predicates: [{
      type: "robot_in_zone",
      zone_id: "far_zone",
      tolerance: 0.2
    }]
  }
});

describe("active scenario chunk queries", () => {
  it("returns rectangle-intersecting chunks with their uniquely owned entities", () => {
    const active = queryScenarioChunksInRectangle(scenario, {
      minimum: { x: 12.1, z: 0 },
      maximum: { x: 23.9, z: 11.9 }
    });

    expect(active.chunks.map(({ chunk }) => chunk.id)).toEqual(["chunk_1_0"]);
    expect(active.obstacles.map(({ id }) => id)).toEqual(["north_block"]);
    expect(active.objects).toEqual([]);
    expect(active.zones).toEqual([]);
    expect(active.chunks[0]).toMatchObject({
      obstacles: [{ id: "north_block" }],
      objects: [],
      zones: []
    });
  });

  it("uses exact circle-to-chunk intersection rather than a square approximation", () => {
    const active = queryScenarioChunksInCircle(scenario, {
      center: { x: 18, z: 18 },
      radius: 7
    });

    expect(active.chunks.map(({ chunk }) => chunk.id)).toEqual([
      "chunk_1_0",
      "chunk_0_1",
      "chunk_1_1",
      "chunk_2_1",
      "chunk_1_2"
    ]);
    expect(active.obstacles.map(({ id }) => id)).toEqual(["north_block"]);
    expect(active.objects.map(({ id }) => id)).toEqual(["center_crate"]);
    expect(active.zones).toEqual([]);
  });

  it("returns an empty selection when the active region misses the world", () => {
    const rectangle = queryScenarioChunksInRectangle(scenario, {
      minimum: { x: -20, z: -20 },
      maximum: { x: -10, z: -10 }
    });
    const circle = queryScenarioChunksInCircle(scenario, {
      center: { x: 100, z: 100 },
      radius: 2
    });

    expect(rectangle).toEqual({ chunks: [], obstacles: [], objects: [], zones: [] });
    expect(circle).toEqual({ chunks: [], obstacles: [], objects: [], zones: [] });
  });

  it("rejects malformed active regions", () => {
    expect(() => queryScenarioChunksInRectangle(scenario, {
      minimum: { x: 5, z: 0 },
      maximum: { x: 4, z: 1 }
    })).toThrow("Rectangle minimum");
    expect(() => queryScenarioChunksInCircle(scenario, {
      center: { x: 0, z: 0 },
      radius: -1
    })).toThrow("Circle radius");
    expect(() => queryScenarioChunksInCircle(scenario, {
      center: { x: Number.NaN, z: 0 },
      radius: 1
    })).toThrow("Circle center");
  });
});
