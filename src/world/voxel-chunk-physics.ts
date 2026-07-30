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
  region: ChunkRegion;
}

/**
 * Owns only the terrain bodies currently close enough to affect the robot.
 * Scenario objects and authored obstacles remain owned by the scene builder.
 */
export class VoxelChunkPhysics {
  readonly #world: World;
  readonly #store: VoxelStore;
  readonly #loadRadiusChunks: number;
  readonly #bodies = new Map<string, RigidBody[]>();

  constructor(world: World, store: VoxelStore, loadRadiusChunks = 2) {
    this.#world = world;
    this.#store = store;
    this.#loadRadiusChunks = loadRadiusChunks;
  }

  get loadRadiusChunks(): number {
    return this.#loadRadiusChunks;
  }

  synchronize(point: Pick<Vec3, "x" | "z">): ChunkSynchronization {
    const desired = this.#store.desiredChunksAround(point, this.#loadRadiusChunks);
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
    return this.#store.loadedChunks().flatMap((chunk) => this.#store.boxesInChunk(chunk));
  }

  region(): ChunkRegion {
    return this.#store.loadedRegion();
  }

  dispose(): void {
    for (const key of [...this.#bodies.keys()]) this.#unload(key);
    this.#store.setLoadedChunks([]);
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
