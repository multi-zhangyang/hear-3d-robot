import { describe, expect, it } from "vitest";
import type { Terrain } from "../domain/schema.js";
import { VoxelEditError, VoxelStore } from "./voxel-store.js";

const terrain: Terrain = {
  cell: 1,
  block: 1,
  columns: 32,
  rows: 32,
  chunk_size: 16,
  maximum_height: 12,
  heights: Array.from({ length: 32 * 32 }, (_, index) => {
    const column = index % 32;
    const row = Math.floor(index / 32);
    return column === 0 || row === 0 || column === 31 || row === 31 ? 3 : 0;
  })
};

describe("VoxelStore", () => {
  it("persists real break and place mutations independently of the generated baseline", () => {
    const store = new VoxelStore(terrain);
    store.setLoadedChunks([{ column: 0, row: 0 }]);

    const placed = store.placeBlock(
      { column: 4, level: 0, row: 5 },
      "placed",
      { commandId: "place_1", agentId: "builder" }
    );
    expect(placed.mutation).toMatchObject({ before: null, after: "placed", revision: 1 });
    expect(store.materialAt({ column: 4, level: 0, row: 5 })).toBe("placed");

    const broken = store.breakBlock(
      { column: 4, level: 0, row: 5 },
      { commandId: "break_1", agentId: "builder" }
    );
    expect(broken.mutation).toMatchObject({ before: "placed", after: null, revision: 2 });
    expect(store.inventory().placed).toBe(8);

    const restored = new VoxelStore(terrain, store.snapshot(2));
    expect(restored.materialAt({ column: 4, level: 0, row: 5 })).toBeNull();
    expect(restored.revision).toBe(2);
  });

  it("loads only the chunk neighbourhood around the robot", () => {
    const store = new VoxelStore(terrain);
    const desired = store.desiredChunksAround({ x: 2, z: 2 }, 1);
    expect(desired).toEqual([
      { column: 0, row: 0 },
      { column: 1, row: 0 },
      { column: 0, row: 1 },
      { column: 1, row: 1 }
    ]);
    store.setLoadedChunks(desired);
    expect(store.isLoaded({ column: 20, level: 0, row: 20 })).toBe(true);
  });

  it("rejects unsupported construction and edits in unloaded chunks", () => {
    const store = new VoxelStore(terrain);
    store.setLoadedChunks([{ column: 0, row: 0 }]);
    expect(() => store.placeBlock(
      { column: 5, level: 3, row: 5 },
      "placed",
      { commandId: "place_floating", agentId: "builder" }
    )).toThrowError(VoxelEditError);
    expect(() => store.placeBlock(
      { column: 20, level: 0, row: 20 },
      "placed",
      { commandId: "place_unloaded", agentId: "builder" }
    )).toThrowError(VoxelEditError);
  });

  it("merges occupied cells into chunk-local collision prisms", () => {
    const store = new VoxelStore(terrain);
    store.setLoadedChunks([{ column: 0, row: 0 }]);
    store.placeBlock(
      { column: 4, level: 0, row: 5 },
      "placed",
      { commandId: "place_a", agentId: "builder" }
    );
    store.placeBlock(
      { column: 5, level: 0, row: 5 },
      "placed",
      { commandId: "place_b", agentId: "builder" }
    );
    expect(store.boxesInChunk({ column: 0, row: 0 })).toEqual(expect.arrayContaining([
      expect.objectContaining({
        center: { x: 5, y: 0.5, z: 5.5 },
        size: { x: 2, y: 1, z: 1 }
      })
    ]));
  });
});
