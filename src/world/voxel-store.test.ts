import { describe, expect, it, vi } from "vitest";
import type { Terrain, VoxelWorldState } from "../domain/schema.js";
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

  it("keeps one authoritative overlay record per coordinate across long edit histories", () => {
    const store = new VoxelStore(terrain);
    const coordinate = { column: 4, level: 0, row: 5 };
    store.setLoadedChunks([{ column: 0, row: 0 }]);

    for (let cycle = 0; cycle < 40; cycle += 1) {
      store.placeBlock(coordinate, "placed", {
        commandId: `place_${cycle}`,
        agentId: "builder"
      });
      store.breakBlock(coordinate, {
        commandId: `break_${cycle}`,
        agentId: "builder"
      });
    }

    const snapshot = store.snapshot(2);
    expect(snapshot.revision).toBe(80);
    expect(snapshot.mutations).toEqual([
      expect.objectContaining({
        coordinate,
        before: "placed",
        after: null,
        revision: 80,
        source_command_id: "break_39"
      })
    ]);
    expect(snapshot.inventory.placed).toBe(8);

    const restored = new VoxelStore(terrain, snapshot);
    expect(restored.materialAt(coordinate)).toBeNull();
    expect(restored.snapshot(2)).toEqual(snapshot);
  });

  it("compacts legacy duplicate overlays by highest revision during restore", () => {
    const coordinate = { column: 4, level: 0, row: 5 };
    const legacy: VoxelWorldState = {
      version: 1,
      revision: 3,
      chunk_size: 16,
      load_radius_chunks: 2,
      loaded_chunks: [{ column: 0, row: 0 }],
      mutations: [
        {
          coordinate,
          before: null,
          after: "placed",
          revision: 1,
          source_command_id: "place_old",
          source_agent_id: "builder"
        },
        {
          coordinate,
          before: "placed",
          after: null,
          revision: 2,
          source_command_id: "break_old",
          source_agent_id: "builder"
        },
        {
          coordinate,
          before: null,
          after: "placed",
          revision: 3,
          source_command_id: "place_latest",
          source_agent_id: "builder"
        }
      ],
      inventory: { grass: 0, dirt: 0, stone: 0, sand: 0, placed: 7 }
    };

    const restored = new VoxelStore(terrain, legacy);
    expect(restored.materialAt(coordinate)).toBe("placed");
    expect(restored.snapshot(2).mutations).toEqual([
      expect.objectContaining({ revision: 3, source_command_id: "place_latest" })
    ]);
  });

  it("rejects a restored overlay newer than its declared world revision", () => {
    const coordinate = { column: 4, level: 0, row: 5 };
    const invalid: VoxelWorldState = {
      version: 1,
      revision: 1,
      chunk_size: 16,
      load_radius_chunks: 2,
      loaded_chunks: [],
      mutations: [{
        coordinate,
        before: null,
        after: "placed",
        revision: 2,
        source_command_id: "future_edit",
        source_agent_id: "builder"
      }],
      inventory: { grass: 0, dirt: 0, stone: 0, sand: 0, placed: 7 }
    };

    expect(() => new VoxelStore(terrain, invalid)).toThrow(
      "Voxel mutation revision 2 exceeds world revision 1"
    );
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

  it("maintains the projected column height incrementally across edits and restore", () => {
    const store = new VoxelStore(terrain);
    store.setLoadedChunks([{ column: 0, row: 0 }]);
    for (let level = 0; level < 3; level += 1) {
      store.placeBlock(
        { column: 4, level, row: 5 },
        "placed",
        { commandId: `place_${level}`, agentId: "builder" }
      );
    }
    expect(store.heightAt(4, 5)).toBe(3);
    expect(store.projectedTerrain().heights[5 * terrain.columns + 4]).toBe(3);

    store.breakBlock(
      { column: 4, level: 2, row: 5 },
      { commandId: "break_top", agentId: "builder" }
    );
    expect(store.heightAt(4, 5)).toBe(2);

    const restored = new VoxelStore(terrain, store.snapshot(2));
    expect(restored.heightAt(4, 5)).toBe(2);
    expect(restored.projectedTerrain().heights).toEqual(store.projectedTerrain().heights);
  });

  it("projects a large terrain without rescanning every vertical voxel", () => {
    const large: Terrain = {
      ...terrain,
      columns: 384,
      rows: 384,
      maximum_height: 64,
      heights: new Array<number>(384 * 384).fill(24)
    };
    const store = new VoxelStore(large);
    const reads = vi.spyOn(store, "materialAt");

    const projected = store.projectedTerrain();

    expect(projected.heights).toHaveLength(384 * 384);
    expect(projected.heights[projected.heights.length - 1]).toBe(24);
    expect(reads).not.toHaveBeenCalled();
  });

  it("caps legacy authored heights at the authoritative vertical voxel bound", () => {
    const bounded: Terrain = {
      ...terrain,
      maximum_height: 4,
      heights: terrain.heights.map((height, index) => index === 0 ? 12 : height)
    };
    const store = new VoxelStore(bounded);

    expect(store.heightAt(0, 0)).toBe(4);
    expect(store.projectedTerrain().heights[0]).toBe(4);
    expect(store.materialAt({ column: 0, level: 4, row: 0 })).toBeNull();
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
