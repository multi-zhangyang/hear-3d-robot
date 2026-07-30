import RAPIER from "@dimforge/rapier3d-compat";
import { beforeAll, describe, expect, it } from "vitest";
import type { Terrain } from "../domain/schema.js";
import { VoxelChunkPhysics } from "./voxel-chunk-physics.js";
import { VoxelStore } from "./voxel-store.js";

beforeAll(async () => {
  await RAPIER.init();
});

describe("VoxelChunkPhysics", () => {
  it("retains remote entity chunks without widening the local navigation scope", () => {
    const world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
    const store = new VoxelStore(flatTerrain(96, 16));
    const chunks = new VoxelChunkPhysics(world, store, 2);
    try {
      const synchronized = chunks.synchronize(
        { x: 8.5, z: 8.5 },
        [{
          minimum: { x: 80.2, z: 80.2 },
          maximum: { x: 80.8, z: 80.8 }
        }]
      );

      expect(synchronized.active).toContainEqual({ column: 5, row: 5 });
      expect(synchronized.navigation).not.toContainEqual({ column: 5, row: 5 });
      expect(synchronized.region).toEqual({
        minimum: { x: 0, y: 0, z: 0 },
        maximum: { x: 48, y: 24, z: 48 }
      });

      const shifted = chunks.synchronize(
        { x: 8.5, z: 56.5 },
        [{
          minimum: { x: 80.2, z: 80.2 },
          maximum: { x: 80.8, z: 80.8 }
        }]
      );
      expect(shifted.active).toContainEqual({ column: 5, row: 5 });
      expect(shifted.navigation).not.toContainEqual({ column: 5, row: 5 });
      expect(shifted.navigation).toEqual(expect.arrayContaining([
        { column: 0, row: 1 },
        { column: 2, row: 5 }
      ]));
      expect(shifted.region).toEqual({
        minimum: { x: 0, y: 0, z: 16 },
        maximum: { x: 48, y: 24, z: 96 }
      });
    } finally {
      chunks.dispose();
      world.free();
    }
  });
});

function flatTerrain(size: number, chunkSize: number): Terrain {
  return {
    cell: 1,
    columns: size,
    rows: size,
    block: 1,
    chunk_size: chunkSize,
    maximum_height: 24,
    heights: new Array<number>(size * size).fill(0)
  };
}
