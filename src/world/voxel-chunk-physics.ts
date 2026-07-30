import RAPIER from "@dimforge/rapier3d-compat";
import type { Vec3 } from "../domain/schema.js";
import type { TerrainBox } from "./terrain.js";
import {
  VoxelStore,
  chunkKey,
  type VoxelChunkReference
} from "./voxel-store.js";

type World = InstanceType<typeof RAPIER.World>;
type RigidBody = InstanceType<typeof RAPIER.RigidBody>;

export interface ChunkRegion {
  minimum: Vec3;
  maximum: Vec3;
}

export interface ChunkSynchronization {
  changed: boolean;
  loaded: VoxelChunkReference[];
  unloaded: VoxelChunkReference[];
  active: VoxelChunkReference[];
  navigation: VoxelChunkReference[];
  region: ChunkRegion;
}

/**
 * Planar bounds of a dynamic entity whose physical support must remain live.
 *
 * These regions affect Rapier residency only. Navigation remains scoped to the
 * robot's bounded neighbourhood, otherwise one remote payload would stretch a
 * local Recast build across the whole world.
 */
export interface ChunkResidentRegion {
  minimum: Pick<Vec3, "x" | "z">;
  maximum: Pick<Vec3, "x" | "z">;
}

/** Owns the bounded robot neighbourhood plus terrain supporting dynamic bodies. */
export class VoxelChunkPhysics {
  readonly #world: World;
  readonly #store: VoxelStore;
  readonly #loadRadiusChunks: number;
  readonly #bodies = new Map<string, RigidBody[]>();
  #navigationChunks: VoxelChunkReference[] = [];

  constructor(world: World, store: VoxelStore, loadRadiusChunks = 2) {
    this.#world = world;
    this.#store = store;
    this.#loadRadiusChunks = loadRadiusChunks;
  }

  get loadRadiusChunks(): number {
    return this.#loadRadiusChunks;
  }

  synchronize(
    point: Pick<Vec3, "x" | "z">,
    residents: readonly ChunkResidentRegion[] = []
  ): ChunkSynchronization {
    const navigation = this.#store.desiredChunksAround(point, this.#loadRadiusChunks);
    this.#navigationChunks = navigation;
    const desiredByKey = new Map(navigation.map((chunk) => [chunkKey(chunk), chunk]));
    for (const resident of residents) {
      for (const chunk of this.#chunksOverlapping(resident)) {
        desiredByKey.set(chunkKey(chunk), chunk);
      }
    }
    const desired = [...desiredByKey.values()].sort(chunkOrder);
    const desiredKeys = new Set(desired.map(chunkKey));
    const loaded: VoxelChunkReference[] = [];
    const unloaded: VoxelChunkReference[] = [];

    for (const key of [...this.#bodies.keys()]) {
      if (desiredKeys.has(key)) continue;
      const chunk = parseChunkKey(key);
      this.#unload(key);
      unloaded.push(chunk);
    }
    for (const chunk of desired) {
      const key = chunkKey(chunk);
      if (this.#bodies.has(key)) continue;
      this.#load(chunk);
      loaded.push(chunk);
    }
    this.#store.setLoadedChunks(desired);
    return {
      changed: loaded.length > 0 || unloaded.length > 0,
      loaded,
      unloaded,
      active: this.#store.loadedChunks(),
      navigation: this.navigationChunks(),
      region: this.region()
    };
  }

  rebuild(chunk: VoxelChunkReference): void {
    const key = chunkKey(chunk);
    if (!this.#bodies.has(key)) return;
    this.#unload(key);
    this.#load(chunk);
  }

  activeSolids(): TerrainBox[] {
    return this.#navigationChunks.flatMap((chunk) => this.#store.boxesInChunk(chunk));
  }

  navigationChunks(): VoxelChunkReference[] {
    return this.#navigationChunks.map((chunk) => ({ ...chunk }));
  }

  region(): ChunkRegion {
    return this.#store.regionForChunks(this.#navigationChunks);
  }

  dispose(): void {
    for (const key of [...this.#bodies.keys()]) this.#unload(key);
    this.#navigationChunks = [];
    this.#store.setLoadedChunks([]);
  }

  #chunksOverlapping(region: ChunkResidentRegion): VoxelChunkReference[] {
    const terrain = this.#store.terrain;
    if (!finiteBounds(region)
      || region.minimum.x > region.maximum.x
      || region.minimum.z > region.maximum.z) {
      throw new Error("Invalid dynamic voxel resident bounds");
    }
    const width = terrain.columns * terrain.cell;
    const depth = terrain.rows * terrain.cell;
    if (region.maximum.x < 0 || region.maximum.z < 0
      || region.minimum.x > width || region.minimum.z > depth) return [];

    const minimumCellColumn = clamp(
      Math.floor(region.minimum.x / terrain.cell),
      0,
      terrain.columns - 1
    );
    const maximumCellColumn = clamp(
      Math.floor(region.maximum.x / terrain.cell),
      0,
      terrain.columns - 1
    );
    const minimumCellRow = clamp(
      Math.floor(region.minimum.z / terrain.cell),
      0,
      terrain.rows - 1
    );
    const maximumCellRow = clamp(
      Math.floor(region.maximum.z / terrain.cell),
      0,
      terrain.rows - 1
    );
    const minimumChunkColumn = Math.floor(minimumCellColumn / terrain.chunk_size);
    const maximumChunkColumn = Math.floor(maximumCellColumn / terrain.chunk_size);
    const minimumChunkRow = Math.floor(minimumCellRow / terrain.chunk_size);
    const maximumChunkRow = Math.floor(maximumCellRow / terrain.chunk_size);
    const chunks: VoxelChunkReference[] = [];
    for (let row = minimumChunkRow; row <= maximumChunkRow; row += 1) {
      for (let column = minimumChunkColumn; column <= maximumChunkColumn; column += 1) {
        chunks.push({ column, row });
      }
    }
    return chunks;
  }

  #load(chunk: VoxelChunkReference): void {
    const bodies = this.#store.boxesInChunk(chunk).map((box) => {
      const body = this.#world.createRigidBody(
        RAPIER.RigidBodyDesc.fixed()
          .setTranslation(box.center.x, box.center.y, box.center.z)
          .setUserData({
            kind: "voxel",
            id: box.id,
            chunk_column: chunk.column,
            chunk_row: chunk.row
          })
      );
      this.#world.createCollider(
        RAPIER.ColliderDesc.cuboid(box.size.x / 2, box.size.y / 2, box.size.z / 2)
          .setFriction(0.86),
        body
      );
      return body;
    });
    this.#bodies.set(chunkKey(chunk), bodies);
  }

  #unload(key: string): void {
    const bodies = this.#bodies.get(key);
    if (!bodies) return;
    for (const body of bodies) this.#world.removeRigidBody(body);
    this.#bodies.delete(key);
  }
}

function parseChunkKey(key: string): VoxelChunkReference {
  const [column, row] = key.split(":").map(Number);
  if (column === undefined || row === undefined) throw new Error(`Invalid chunk key ${key}`);
  return { column, row };
}

function chunkOrder(left: VoxelChunkReference, right: VoxelChunkReference): number {
  return left.row - right.row || left.column - right.column;
}

function finiteBounds(region: ChunkResidentRegion): boolean {
  return Number.isFinite(region.minimum.x)
    && Number.isFinite(region.minimum.z)
    && Number.isFinite(region.maximum.x)
    && Number.isFinite(region.maximum.z);
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}
