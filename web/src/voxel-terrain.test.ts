import * as THREE from "three";
import { describe, expect, it } from "vitest";
import type { TerrainDefinition } from "./types";
import { VoxelTerrain } from "./voxel-terrain";

const TERRAIN: TerrainDefinition = {
  cell: 1,
  block: 1,
  columns: 8,
  rows: 8,
  chunk_size: 8,
  maximum_height: 4,
  heights: new Array<number>(64).fill(1)
};

const WIDE_TERRAIN: TerrainDefinition = {
  ...TERRAIN,
  columns: 16,
  heights: new Array<number>(128).fill(1)
};

describe("VoxelTerrain", () => {
  it("does not upload unchanged instance colours on every world frame", () => {
    const terrain = new VoxelTerrain(TERRAIN, 17);
    terrain.update(undefined, "");
    const mesh = terrain.root.children.find((child) => child instanceof THREE.InstancedMesh);
    expect(mesh).toBeInstanceOf(THREE.InstancedMesh);
    if (!(mesh instanceof THREE.InstancedMesh) || !mesh.instanceColor) {
      throw new Error("Expected an instanced terrain mesh with per-voxel colours");
    }
    const firstVersion = mesh.instanceColor.version;

    terrain.update(undefined, "");

    expect(mesh.instanceColor.version).toBe(firstVersion);
  });

  it("keeps an explored chunk visible after the physics window moves away", () => {
    const terrain = new VoxelTerrain(WIDE_TERRAIN, 17);
    const exploration = new Uint8Array(16);
    exploration[0] = 1;
    terrain.update({
      version: 1,
      revision: 0,
      chunk_size: 8,
      load_radius_chunks: 1,
      loaded_chunks: [{ column: 0, row: 0 }],
      mutations: [],
      inventory: { grass: 0, dirt: 0, stone: 0, sand: 0, placed: 0 }
    }, bytesToBase64(exploration));

    terrain.update({
      version: 1,
      revision: 0,
      chunk_size: 8,
      load_radius_chunks: 1,
      loaded_chunks: [{ column: 1, row: 0 }],
      mutations: [],
      inventory: { grass: 0, dirt: 0, stone: 0, sand: 0, placed: 0 }
    }, bytesToBase64(exploration));

    const meshes = terrain.root.children.filter(
      (child): child is THREE.InstancedMesh => child instanceof THREE.InstancedMesh
    );
    expect(meshes.map((mesh) => mesh.name).sort()).toEqual([
      "terrain-chunk-0:0",
      "terrain-chunk-1:0"
    ]);
    expect(meshes.find((mesh) => mesh.name === "terrain-chunk-0:0")?.userData)
      .toMatchObject({ loaded_in_physics: false });
    expect(meshes.find((mesh) => mesh.name === "terrain-chunk-1:0")?.userData)
      .toMatchObject({ loaded_in_physics: true });
  });

  it("resolves an instanced ray hit to its authoritative voxel coordinate and material", () => {
    const terrain = new VoxelTerrain(TERRAIN, 17);
    terrain.update({
      version: 1,
      revision: 1,
      chunk_size: 8,
      load_radius_chunks: 1,
      loaded_chunks: [{ column: 0, row: 0 }],
      mutations: [{
        coordinate: { column: 0, level: 0, row: 0 },
        before: "grass",
        after: "stone",
        revision: 1,
        source_command_id: "command-1",
        source_agent_id: "agent-1"
      }],
      inventory: { grass: 0, dirt: 0, stone: 0, sand: 0, placed: 0 }
    }, "");
    const mesh = terrain.root.children.find((child) => child instanceof THREE.InstancedMesh);
    expect(mesh).toBeInstanceOf(THREE.InstancedMesh);
    if (!(mesh instanceof THREE.InstancedMesh)) throw new Error("Expected voxel mesh");

    const resolved = terrain.resolveIntersection({ object: mesh, instanceId: 0 });

    expect(resolved?.selection).toEqual({
      kind: "voxel",
      coordinate: { column: 0, level: 0, row: 0 },
      material: "stone"
    });
    expect(resolved?.box.getCenter(new THREE.Vector3()).toArray()).toEqual([0.5, 0.5, 0.5]);
    expect(terrain.resolveSelection(resolved!.selection)?.selection.material).toBe("stone");
  });
});

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return globalThis.btoa(binary);
}
