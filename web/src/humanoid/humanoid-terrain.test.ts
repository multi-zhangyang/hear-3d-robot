import { describe, expect, it } from "vitest";
import { HumanoidTerrain, type HumanoidTerrainChunk } from "./humanoid-terrain";

describe("HumanoidTerrain", () => {
  it("keeps tile meshes only for resident chunks", () => {
    const terrain = new HumanoidTerrain({ width: 20, depth: 10 }, 19);
    const left = chunk("chunk_0_0", 0, 10);
    const right = chunk("chunk_1_0", 10, 20);

    expect(terrain.root.userData.tileCount).toBe(0);
    terrain.update([left]);
    const first = terrain.root.getObjectByName("humanoid-terrain-chunk_0_0");
    expect(first).toBeDefined();
    expect(terrain.root.getObjectByName("humanoid-terrain-chunk_1_0")).toBeUndefined();
    expect(terrain.root.userData.tileCount).toBe(100);
    expect(terrain.root.userData.residentChunkCount).toBe(1);

    terrain.update([right]);
    expect(terrain.root.getObjectByName("humanoid-terrain-chunk_0_0")).toBeUndefined();
    expect(terrain.root.getObjectByName("humanoid-terrain-chunk_1_0")).toBeDefined();
    expect(terrain.root.userData.tileCount).toBe(100);
  });

  it("rejects duplicate residency entries and invalid chunk bounds", () => {
    const terrain = new HumanoidTerrain({ width: 20, depth: 10 }, 3);
    const left = chunk("chunk_0_0", 0, 10);
    expect(() => terrain.update([left, left])).toThrow(/unique/i);
    expect(() => terrain.update([{
      id: "broken",
      bounds: {
        minimum: { x: 5, z: 0 },
        maximum: { x: 5, z: 10 }
      }
    }])).toThrow(/bounds/i);
  });
});

function chunk(id: string, minimumX: number, maximumX: number): HumanoidTerrainChunk {
  return {
    id,
    bounds: {
      minimum: { x: minimumX, z: 0 },
      maximum: { x: maximumX, z: 10 }
    }
  };
}
