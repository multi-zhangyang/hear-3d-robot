import * as THREE from "three";
import type {
  TerrainDefinition,
  VoxelMaterial,
  VoxelMutation,
  VoxelWorldState
} from "./types";
import type { WorldSelection } from "./stage/world-selection";

type RenderMaterial = VoxelMaterial | "ground";

interface BlockInstance {
  cell: number;
  level: number;
  top: boolean;
  material: RenderMaterial;
}

interface RenderedChunk {
  mesh: THREE.InstancedMesh;
  instances: BlockInstance[];
  mutationSignature: string;
}

interface ChunkReference {
  column: number;
  row: number;
}

export interface ResolvedVoxelSelection {
  selection: Extract<WorldSelection, { kind: "voxel" }>;
  box: THREE.Box3;
}

/**
 * Chunked visual projection of the authoritative voxel baseline + mutations.
 *
 * The backend decides which chunks are active in physics. Visually, explored
 * chunks remain resident after they leave that moving physics window so the
 * world does not disappear behind the robot. Geometry is rebuilt only when a
 * mutation changes; exploration updates colours without rebuilding it.
 */
export class VoxelTerrain {
  readonly root = new THREE.Group();
  readonly #terrain: TerrainDefinition;
  readonly #worldSeed: number;
  readonly #geometry: THREE.BoxGeometry;
  readonly #material: THREE.MeshStandardMaterial;
  readonly #chunks = new Map<string, RenderedChunk>();
  readonly #chunksByMesh = new Map<THREE.Object3D, RenderedChunk>();
  readonly #exploredChunkCounts = new Map<string, number>();
  readonly #exploredChunks = new Map<string, ChunkReference>();
  #mutations = new Map<string, VoxelMaterial | null>();
  #mutationLists = new Map<string, VoxelMutation[]>();
  #explorationKey = "";
  #explorationBits: Uint8Array<ArrayBufferLike> = new Uint8Array();
  #stateRevision = -1;

  constructor(terrain: TerrainDefinition, worldSeed: number) {
    this.root.name = "voxel-terrain";
    this.#terrain = terrain;
    this.#worldSeed = worldSeed;
    this.#geometry = new THREE.BoxGeometry(
      terrain.cell * 0.96,
      terrain.block * 0.96,
      terrain.cell * 0.96
    );
    this.#material = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.92,
      metalness: 0.02,
      emissive: 0x20262b,
      emissiveIntensity: 0.72
    });
  }

  update(state: VoxelWorldState | null | undefined, encodedExploration: string): void {
    const dirtyExplorationChunks = this.#updateExploration(encodedExploration);
    if (state && state.revision !== this.#stateRevision) {
      this.#stateRevision = state.revision;
      this.#indexMutations(state.mutations);
    }

    // A legacy checkpoint has no backend chunk set. Rendering every chunk keeps
    // old real-run artifacts viewable; current runs always take the streamed set.
    const loaded = state?.loaded_chunks ?? allChunks(this.#terrain);
    const desired = new Map<string, { reference: ChunkReference; physical: boolean }>();
    for (const reference of this.#exploredChunks.values()) {
      desired.set(chunkKey(reference), { reference, physical: false });
    }
    for (const reference of loaded) {
      desired.set(chunkKey(reference), { reference, physical: true });
    }
    const desiredKeys = new Set(desired.keys());
    for (const [key, chunk] of this.#chunks) {
      if (desiredKeys.has(key)) continue;
      this.root.remove(chunk.mesh);
      this.#chunksByMesh.delete(chunk.mesh);
      this.#chunks.delete(key);
      chunk.mesh.dispose();
    }
    for (const { reference, physical } of desired.values()) {
      this.#synchronizeChunk(
        reference,
        physical,
        dirtyExplorationChunks.has(chunkKey(reference))
      );
    }
  }

  resolveIntersection(
    intersection: Pick<THREE.Intersection, "object" | "instanceId">
  ): ResolvedVoxelSelection | null {
    if (intersection.instanceId === undefined) return null;
    const chunk = this.#chunksByMesh.get(intersection.object);
    const block = chunk?.instances[intersection.instanceId];
    return block ? this.#resolveBlock(block) : null;
  }

  pickables(): THREE.InstancedMesh[] {
    return [...this.#chunks.values()].map((chunk) => chunk.mesh);
  }

  resolveSelection(
    selection: Extract<WorldSelection, { kind: "voxel" }>
  ): ResolvedVoxelSelection | null {
    const reference = {
      column: Math.floor(selection.coordinate.column / this.#terrain.chunk_size),
      row: Math.floor(selection.coordinate.row / this.#terrain.chunk_size)
    };
    const chunk = this.#chunks.get(chunkKey(reference));
    if (!chunk) return null;
    const cell = selection.coordinate.row * this.#terrain.columns + selection.coordinate.column;
    const block = chunk.instances.find((candidate) =>
      candidate.cell === cell && candidate.level === selection.coordinate.level);
    return block ? this.#resolveBlock(block) : null;
  }

  #resolveBlock(block: BlockInstance): ResolvedVoxelSelection {
    const column = block.cell % this.#terrain.columns;
    const row = (block.cell - column) / this.#terrain.columns;
    const center = new THREE.Vector3(
      (column + 0.5) * this.#terrain.cell,
      (block.level + 0.5) * this.#terrain.block,
      (row + 0.5) * this.#terrain.cell
    );
    const size = new THREE.Vector3(
      this.#terrain.cell * 0.96,
      this.#terrain.block * 0.96,
      this.#terrain.cell * 0.96
    );
    return {
      selection: {
        kind: "voxel",
        coordinate: { column, level: block.level, row },
        material: block.material
      },
      box: new THREE.Box3().setFromCenterAndSize(center, size)
    };
  }

  #indexMutations(mutations: VoxelMutation[]): void {
    this.#mutations = new Map();
    this.#mutationLists = new Map();
    for (const mutation of mutations) {
      this.#mutations.set(coordinateKey(mutation.coordinate), mutation.after);
      const key = chunkKey({
        column: Math.floor(mutation.coordinate.column / this.#terrain.chunk_size),
        row: Math.floor(mutation.coordinate.row / this.#terrain.chunk_size)
      });
      const entries = this.#mutationLists.get(key) ?? [];
      entries.push(mutation);
      this.#mutationLists.set(key, entries);
    }
  }

  #synchronizeChunk(
    reference: ChunkReference,
    physical: boolean,
    explorationDirty: boolean
  ): void {
    const key = chunkKey(reference);
    const signature = (this.#mutationLists.get(key) ?? [])
      .map((mutation) => `${coordinateKey(mutation.coordinate)}=${mutation.after ?? "air"}@${mutation.revision}`)
      .join("|");
    const existing = this.#chunks.get(key);
    if (existing?.mutationSignature === signature) {
      existing.mesh.userData.loaded_in_physics = physical;
      existing.mesh.receiveShadow = physical;
      if (explorationDirty) this.#updateChunkColors(existing);
      return;
    }
    if (existing) {
      this.root.remove(existing.mesh);
      this.#chunksByMesh.delete(existing.mesh);
      this.#chunks.delete(key);
      existing.mesh.dispose();
    }

    const instances = this.#instancesInChunk(reference);
    if (instances.length === 0) return;
    const mesh = new THREE.InstancedMesh(this.#geometry, this.#material, instances.length);
    mesh.name = `terrain-chunk-${key}`;
    mesh.userData = {
      kind: "voxel_chunk",
      column: reference.column,
      row: reference.row,
      loaded_in_physics: physical
    };
    mesh.receiveShadow = physical;
    mesh.frustumCulled = true;
    const matrix = new THREE.Matrix4();
    for (const [instance, block] of instances.entries()) {
      const column = block.cell % this.#terrain.columns;
      const row = (block.cell - column) / this.#terrain.columns;
      matrix.makeTranslation(
        (column + 0.5) * this.#terrain.cell,
        (block.level + 0.5) * this.#terrain.block,
        (row + 0.5) * this.#terrain.cell
      );
      mesh.setMatrixAt(instance, matrix);
    }
    mesh.instanceMatrix.setUsage(THREE.StaticDrawUsage);
    mesh.instanceMatrix.needsUpdate = true;
    const rendered = {
      mesh,
      instances,
      mutationSignature: signature
    };
    this.#updateChunkColors(rendered);
    this.#chunks.set(key, rendered);
    this.#chunksByMesh.set(mesh, rendered);
    this.root.add(mesh);
  }

  #instancesInChunk(reference: ChunkReference): BlockInstance[] {
    const startColumn = reference.column * this.#terrain.chunk_size;
    const startRow = reference.row * this.#terrain.chunk_size;
    const endColumn = Math.min(this.#terrain.columns, startColumn + this.#terrain.chunk_size);
    const endRow = Math.min(this.#terrain.rows, startRow + this.#terrain.chunk_size);
    const instances: BlockInstance[] = [];
    for (let row = startRow; row < endRow; row += 1) {
      for (let column = startColumn; column < endColumn; column += 1) {
        const cell = row * this.#terrain.columns + column;
        const baselineHeight = this.#terrain.heights[cell] ?? 0;
        const mutatedLevels = this.#mutationLists.get(chunkKey(reference))
          ?.filter((mutation) => mutation.coordinate.column === column
            && mutation.coordinate.row === row)
          .map((mutation) => mutation.coordinate.level) ?? [];
        const highestLevel = Math.max(baselineHeight - 1, ...mutatedLevels);
        const occupied: Array<{ level: number; material: VoxelMaterial }> = [];
        for (let level = 0; level <= highestLevel; level += 1) {
          const material = this.#materialAt(column, level, row);
          if (material) occupied.push({ level, material });
        }
        if (occupied.length === 0) {
          instances.push({ cell, level: -1, top: true, material: "ground" });
          continue;
        }
        const topLevel = Math.max(...occupied.map((block) => block.level));
        for (const block of occupied) {
          instances.push({
            cell,
            level: block.level,
            top: block.level === topLevel,
            material: block.material
          });
        }
      }
    }
    return instances;
  }

  #materialAt(column: number, level: number, row: number): VoxelMaterial | null {
    const key = coordinateKey({ column, level, row });
    if (this.#mutations.has(key)) return this.#mutations.get(key) ?? null;
    const height = this.#terrain.heights[row * this.#terrain.columns + column] ?? 0;
    if (level >= height) return null;
    if (level === height - 1) return "grass";
    return level === 0 && height >= 3 ? "stone" : "dirt";
  }

  #updateExploration(encoded: string): Set<string> {
    const dirtyChunks = new Set<string>();
    if (encoded === this.#explorationKey) return dirtyChunks;
    const nextBits = decodeBase64(encoded);
    const byteLength = Math.max(this.#explorationBits.length, nextBits.length);
    const cellCount = this.#terrain.columns * this.#terrain.rows;
    for (let byteIndex = 0; byteIndex < byteLength; byteIndex += 1) {
      const previousByte = this.#explorationBits[byteIndex] ?? 0;
      const nextByte = nextBits[byteIndex] ?? 0;
      const changed = previousByte ^ nextByte;
      if (changed === 0) continue;
      for (let bit = 0; bit < 8; bit += 1) {
        const mask = 1 << bit;
        if ((changed & mask) === 0) continue;
        const cell = byteIndex * 8 + bit;
        if (cell >= cellCount) break;
        const column = cell % this.#terrain.columns;
        const row = (cell - column) / this.#terrain.columns;
        const reference = {
          column: Math.floor(column / this.#terrain.chunk_size),
          row: Math.floor(row / this.#terrain.chunk_size)
        };
        const key = chunkKey(reference);
        const count = this.#exploredChunkCounts.get(key) ?? 0;
        const nextCount = count + ((nextByte & mask) === 0 ? -1 : 1);
        if (nextCount > 0) {
          this.#exploredChunkCounts.set(key, nextCount);
          this.#exploredChunks.set(key, reference);
        } else {
          this.#exploredChunkCounts.delete(key);
          this.#exploredChunks.delete(key);
        }
        dirtyChunks.add(key);
      }
    }
    this.#explorationKey = encoded;
    this.#explorationBits = nextBits;
    return dirtyChunks;
  }

  #updateChunkColors(chunk: RenderedChunk): void {
    for (const [instance, block] of chunk.instances.entries()) {
      const seen = ((this.#explorationBits[block.cell >> 3] ?? 0)
        & (1 << (block.cell & 7))) !== 0;
      chunk.mesh.setColorAt(instance, blockColor(
        block,
        seen,
        this.#terrain,
        this.#worldSeed
      ));
    }
    if (chunk.mesh.instanceColor) chunk.mesh.instanceColor.needsUpdate = true;
  }
}

function allChunks(terrain: TerrainDefinition): ChunkReference[] {
  const chunks: ChunkReference[] = [];
  const columns = Math.ceil(terrain.columns / terrain.chunk_size);
  const rows = Math.ceil(terrain.rows / terrain.chunk_size);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) chunks.push({ column, row });
  }
  return chunks;
}

function chunkKey(chunk: ChunkReference): string {
  return `${chunk.column}:${chunk.row}`;
}

function coordinateKey(coordinate: { column: number; level: number; row: number }): string {
  return `${coordinate.column}:${coordinate.level}:${coordinate.row}`;
}

function blockColor(
  block: BlockInstance,
  seen: boolean,
  terrain: TerrainDefinition,
  worldSeed: number
): THREE.Color {
  if (!seen) {
    const shade = 0x555e61 + Math.min(3, Math.max(0, block.level)) * 0x030303;
    return new THREE.Color(shade);
  }
  const color = new THREE.Color(materialColor(block, terrain, worldSeed));
  const hash = Math.imul(block.cell + 1, 0x45d9f3b) >>> 0;
  color.offsetHSL(0, 0, ((hash & 0xff) / 255 - 0.5) * 0.08);
  return color;
}

function materialColor(
  block: BlockInstance,
  terrain: TerrainDefinition,
  worldSeed: number
): number {
  if (block.material === "placed") return 0x35c7b0;
  if (block.material === "sand") return 0xc9b56b;
  if (block.material === "stone") return block.top ? 0x959b98 : 0x6c716f;
  if (block.material === "dirt") return block.top ? 0x7c6544 : 0x5d4937;
  if (block.material === "ground") return topColor(block, terrain, worldSeed);
  return block.top ? topColor(block, terrain, worldSeed) : 0x5d4937;
}

function topColor(
  block: BlockInstance,
  terrain: TerrainDefinition,
  worldSeed: number
): number {
  const column = block.cell % terrain.columns;
  const row = (block.cell - column) / terrain.columns;
  const patchX = Math.floor(column / 7);
  const patchZ = Math.floor(row / 7);
  const biome = Math.imul(
    (worldSeed ^ Math.imul(patchX + 17, 0x45d9f3b)
      ^ Math.imul(patchZ + 31, 0x119de1f3)) >>> 0,
    0x27d4eb2d
  ) >>> 0;
  if (block.level <= 0) {
    const band = biome % 10;
    if (band <= 1) return 0xc6b36d;
    if (band <= 3) return 0x4e8660;
    return 0x70a766;
  }
  if (block.level === 1) return biome % 4 === 0 ? 0x6f815a : 0x849b67;
  if (block.level === 2) return 0x989c92;
  return 0xd5dbd7;
}

function decodeBase64(value: string): Uint8Array {
  if (!value) return new Uint8Array();
  const binary = globalThis.atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}
