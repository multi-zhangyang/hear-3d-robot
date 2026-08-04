import { describe, expect, it } from "vitest";
import type { ScenarioDefinition } from "../types";
import { scenarioChunkAt, visibleScenarioChunkIds } from "./chunk-visibility";

describe("humanoid world chunk visibility", () => {
  it("selects only chunks intersecting the active circle", () => {
    const scenario = chunkScenario();
    expect([...visibleScenarioChunkIds(scenario, { x: 2, z: 2 }, 3)])
      .toEqual(["chunk_0_0"]);
    expect([...visibleScenarioChunkIds(scenario, { x: 9, z: 9 }, 3)].sort())
      .toEqual(["chunk_0_0", "chunk_0_1", "chunk_1_0", "chunk_1_1"]);
  });

  it("assigns shared edges canonically and includes the outer world edge", () => {
    const scenario = chunkScenario();
    expect(scenarioChunkAt(scenario, { x: 10, z: 4 })?.id).toBe("chunk_1_0");
    expect(scenarioChunkAt(scenario, { x: 20, z: 20 })?.id).toBe("chunk_1_1");
    expect(scenarioChunkAt(scenario, { x: -0.01, z: 2 })).toBeUndefined();
  });

  it("rejects invalid visibility radii", () => {
    expect(() => visibleScenarioChunkIds(chunkScenario(), { x: 0, z: 0 }, -1))
      .toThrow(/radius/i);
  });
});

function chunkScenario(): ScenarioDefinition {
  const chunks = [];
  for (let row = 0; row < 2; row += 1) {
    for (let column = 0; column < 2; column += 1) {
      chunks.push({
        id: `chunk_${column}_${row}`,
        coordinate: { column, row },
        bounds: {
          minimum: { x: column * 10, z: row * 10 },
          maximum: { x: (column + 1) * 10, z: (row + 1) * 10 }
        },
        entity_ids: { obstacles: [], objects: [], zones: [] }
      });
    }
  }
  return {
    title: "Chunk visibility",
    seed: 1,
    bounds: { width: 20, depth: 20 },
    visibility_radius: 4,
    robot: { x: 2, z: 2, yaw: 0 },
    obstacles: [],
    objects: [],
    zones: [],
    default_goal: {
      summary: "Remain in the first chunk",
      predicates: [{
        type: "robot_at",
        target: { x: 2, y: 0, z: 2 },
        tolerance: 0.5
      }]
    },
    chunk_manifest: {
      version: 1,
      chunk_size: 10,
      grid: { columns: 2, rows: 2 },
      chunks
    }
  };
}
