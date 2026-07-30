import * as THREE from "three";
import { describe, expect, it, vi } from "vitest";
import type { TerrainDefinition, VoxelMutation, VoxelWorldState } from "./types";
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
  it("compacts a maximum-sized legacy mutation history without argument overflow", () => {
    const terrain = new VoxelTerrain(TERRAIN, 17);
    const mutationCount = 150_000;
    const mutations = Array.from({ length: mutationCount }, (_, index): VoxelMutation => ({
      coordinate: { column: 0, level: 0, row: 0 },
      before: index === 0 ? "grass" : index % 2 === 0 ? "stone" : "dirt",
      after: index === mutationCount - 1
        ? "placed"
        : index % 2 === 0 ? "dirt" : "stone",
      revision: index + 1,
      source_command_id: `command-${index + 1}`,
      source_agent_id: "agent-1"
    }));

    terrain.update(voxelState(
      mutationCount,
      [{ column: 0, row: 0 }],
      mutations
    ), "");

    expect(terrain.resolveSelection({
      kind: "voxel",
      coordinate: { column: 0, level: 0, row: 0 },
      material: "placed"
    })?.selection.material).toBe("placed");
    expect(chunkMesh(terrain, "terrain-chunk-0:0").count).toBe(64);
  });

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

  it("refreshes only the chunk containing a changed exploration bit", () => {
    const terrain = new VoxelTerrain(WIDE_TERRAIN, 17);
    const loaded = [{ column: 0, row: 0 }, { column: 1, row: 0 }];
    terrain.update(voxelState(0, loaded), "");
    const left = chunkMesh(terrain, "terrain-chunk-0:0");
    const right = chunkMesh(terrain, "terrain-chunk-1:0");
    if (!left.instanceColor || !right.instanceColor) {
      throw new Error("Expected per-voxel colours on both terrain chunks");
    }
    const leftVersion = left.instanceColor.version;
    const rightVersion = right.instanceColor.version;
    const leftColors = vi.spyOn(left, "setColorAt");
    const rightColors = vi.spyOn(right, "setColorAt");
    const exploration = new Uint8Array(16);
    exploration[0] = 1;

    terrain.update(voxelState(0, loaded), bytesToBase64(exploration));

    expect(leftColors).toHaveBeenCalledTimes(left.count);
    expect(rightColors).not.toHaveBeenCalled();
    expect(left.instanceColor.version).toBe(leftVersion + 1);
    expect(right.instanceColor.version).toBe(rightVersion);
  });

  it("disposes an unloaded chunk mesh without disposing its shared resources", () => {
    const terrain = new VoxelTerrain(WIDE_TERRAIN, 17);
    terrain.update(voxelState(0, [{ column: 0, row: 0 }]), "");
    const removed = chunkMesh(terrain, "terrain-chunk-0:0");
    const geometry = removed.geometry;
    const material = singleMaterial(removed);
    const meshDispose = vi.spyOn(removed, "dispose");
    const geometryDispose = vi.spyOn(geometry, "dispose");
    const materialDispose = vi.spyOn(material, "dispose");

    terrain.update(voxelState(0, [{ column: 1, row: 0 }]), "");

    const retained = chunkMesh(terrain, "terrain-chunk-1:0");
    expect(meshDispose).toHaveBeenCalledTimes(1);
    expect(geometryDispose).not.toHaveBeenCalled();
    expect(materialDispose).not.toHaveBeenCalled();
    expect(removed.parent).toBeNull();
    expect(terrain.resolveIntersection({ object: removed, instanceId: 0 })).toBeNull();
    expect(retained.geometry).toBe(geometry);
    expect(retained.material).toBe(material);
    expect(retained.count).toBe(64);
  });

  it("disposes the replaced chunk mesh while reusing geometry and material", () => {
    const terrain = new VoxelTerrain(TERRAIN, 17);
    const loaded = [{ column: 0, row: 0 }];
    terrain.update(voxelState(0, loaded), "");
    const replaced = chunkMesh(terrain, "terrain-chunk-0:0");
    const geometry = replaced.geometry;
    const material = singleMaterial(replaced);
    const meshDispose = vi.spyOn(replaced, "dispose");
    const geometryDispose = vi.spyOn(geometry, "dispose");
    const materialDispose = vi.spyOn(material, "dispose");
    const mutation: VoxelMutation = {
      coordinate: { column: 0, level: 0, row: 0 },
      before: "grass",
      after: "stone",
      revision: 1,
      source_command_id: "command-1",
      source_agent_id: "agent-1"
    };

    terrain.update(voxelState(1, loaded, [mutation]), "");

    const rebuilt = chunkMesh(terrain, "terrain-chunk-0:0");
    expect(rebuilt).not.toBe(replaced);
    expect(meshDispose).toHaveBeenCalledTimes(1);
    expect(geometryDispose).not.toHaveBeenCalled();
    expect(materialDispose).not.toHaveBeenCalled();
    expect(replaced.parent).toBeNull();
    expect(rebuilt.geometry).toBe(geometry);
    expect(rebuilt.material).toBe(material);
    expect(terrain.resolveIntersection({ object: rebuilt, instanceId: 0 })?.selection.material)
      .toBe("stone");
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

function voxelState(
  revision: number,
  loadedChunks: VoxelWorldState["loaded_chunks"],
  mutations: VoxelMutation[] = []
): VoxelWorldState {
  return {
    version: 1,
    revision,
    chunk_size: 8,
    load_radius_chunks: 1,
    loaded_chunks: loadedChunks,
    mutations,
    inventory: { grass: 0, dirt: 0, stone: 0, sand: 0, placed: 0 }
  };
}

function chunkMesh(terrain: VoxelTerrain, name: string): THREE.InstancedMesh {
  const mesh = terrain.root.children.find((child) => child.name === name);
  if (!(mesh instanceof THREE.InstancedMesh)) throw new Error(`Expected terrain mesh ${name}`);
  return mesh;
}

function singleMaterial(mesh: THREE.InstancedMesh): THREE.Material {
  if (Array.isArray(mesh.material)) throw new Error("Expected one shared terrain material");
  return mesh.material;
}
