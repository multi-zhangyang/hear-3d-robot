import type {
  Terrain,
  Vec3,
  VoxelCoordinate,
  VoxelMaterial,
  VoxelMutation,
  VoxelWorldState
} from "../domain/schema.js";
import type { TerrainBox } from "./terrain.js";

export interface VoxelChunkReference {
  column: number;
  row: number;
}

export interface VoxelEditSource {
  commandId: string;
  agentId: string;
}

export interface VoxelEditResult {
  mutation: VoxelMutation;
  chunk: VoxelChunkReference;
}

export interface VoxelBlock {
  coordinate: VoxelCoordinate;
  material: VoxelMaterial;
  center: Vec3;
}

export interface VoxelFace {
  normal: Vec3;
  point: Vec3;
}

export interface VoxelChunkRegion {
  minimum: Vec3;
  maximum: Vec3;
}

export type VoxelEditability =
  | { editable: true }
  | { editable: false; code: string; detail: Record<string, unknown> };

const EMPTY_INVENTORY: Record<VoxelMaterial, number> = {
  grass: 0,
  dirt: 0,
  stone: 0,
  sand: 0,
  placed: 8
};

const NEIGHBOURS: ReadonlyArray<readonly [number, number, number]> = [
  [1, 0, 0],
  [-1, 0, 0],
  [0, 1, 0],
  [0, -1, 0],
  [0, 0, 1],
  [0, 0, -1]
];

/**
 * Authoritative mutable voxel overlay.
 *
 * The generated height field remains the compact immutable baseline. Only
 * changes are stored, so a large world does not become a repeated cube dump in
 * every checkpoint. Reads always compose baseline + latest mutation.
 */
export class VoxelStore {
  readonly terrain: Terrain;
  readonly #overlay = new Map<string, VoxelMaterial | null>();
  readonly #mutations: VoxelMutation[] = [];
  readonly #loaded = new Set<string>();
  readonly #inventory: Record<VoxelMaterial, number>;
  #revision = 0;

  constructor(terrain: Terrain, restore?: VoxelWorldState | null) {
    this.terrain = terrain;
    this.#inventory = { ...EMPTY_INVENTORY };
    if (!restore) return;
    if (restore.chunk_size !== terrain.chunk_size) {
      throw new Error(
        `Voxel chunk size changed from ${restore.chunk_size} to ${terrain.chunk_size}`
      );
    }
    this.#revision = restore.revision;
    Object.assign(this.#inventory, restore.inventory);
    for (const mutation of restore.mutations) {
      this.#assertCoordinate(mutation.coordinate);
      this.#overlay.set(coordinateKey(mutation.coordinate), mutation.after);
      this.#mutations.push(structuredClone(mutation));
    }
    for (const chunk of restore.loaded_chunks) {
      if (this.#validChunk(chunk)) this.#loaded.add(chunkKey(chunk));
    }
  }

  get revision(): number {
    return this.#revision;
  }

  materialAt(coordinate: VoxelCoordinate): VoxelMaterial | null {
    if (!this.contains(coordinate)) return null;
    const key = coordinateKey(coordinate);
    if (this.#overlay.has(key)) return this.#overlay.get(key) ?? null;
    return baselineMaterial(this.terrain, coordinate);
  }

  contains(coordinate: VoxelCoordinate): boolean {
    return Number.isInteger(coordinate.column)
      && Number.isInteger(coordinate.level)
      && Number.isInteger(coordinate.row)
      && coordinate.column >= 0
      && coordinate.column < this.terrain.columns
      && coordinate.row >= 0
      && coordinate.row < this.terrain.rows
      && coordinate.level >= 0
      && coordinate.level < this.terrain.maximum_height;
  }

  coordinateAt(point: Vec3): VoxelCoordinate | null {
    const coordinate = {
      column: Math.floor(point.x / this.terrain.cell),
      level: Math.floor(point.y / this.terrain.block),
      row: Math.floor(point.z / this.terrain.cell)
    };
    return this.contains(coordinate) ? coordinate : null;
  }

  centerOf(coordinate: VoxelCoordinate): Vec3 {
    this.#assertCoordinate(coordinate);
    return {
      x: (coordinate.column + 0.5) * this.terrain.cell,
      y: (coordinate.level + 0.5) * this.terrain.block,
      z: (coordinate.row + 0.5) * this.terrain.cell
    };
  }

  heightAt(column: number, row: number): number {
    if (column < 0 || row < 0 || column >= this.terrain.columns || row >= this.terrain.rows) {
      return 0;
    }
    for (let level = this.terrain.maximum_height - 1; level >= 0; level -= 1) {
      if (this.materialAt({ column, level, row })) return level + 1;
    }
    return 0;
  }

  /** Height projection used only for local frontier candidates and compact maps. */
  projectedTerrain(): Terrain {
    const heights = new Array<number>(this.terrain.columns * this.terrain.rows);
    for (let row = 0; row < this.terrain.rows; row += 1) {
      for (let column = 0; column < this.terrain.columns; column += 1) {
        heights[row * this.terrain.columns + column] = this.heightAt(column, row);
      }
    }
    return { ...this.terrain, heights };
  }

  chunkAt(coordinate: Pick<VoxelCoordinate, "column" | "row">): VoxelChunkReference {
    return {
      column: Math.floor(coordinate.column / this.terrain.chunk_size),
      row: Math.floor(coordinate.row / this.terrain.chunk_size)
    };
  }

  chunkAtPoint(point: Pick<Vec3, "x" | "z">): VoxelChunkReference {
    return this.chunkAt({
      column: clamp(Math.floor(point.x / this.terrain.cell), 0, this.terrain.columns - 1),
      row: clamp(Math.floor(point.z / this.terrain.cell), 0, this.terrain.rows - 1)
    });
  }

  desiredChunksAround(point: Pick<Vec3, "x" | "z">, radius: number): VoxelChunkReference[] {
    const centre = this.chunkAtPoint(point);
    const columns = Math.ceil(this.terrain.columns / this.terrain.chunk_size);
    const rows = Math.ceil(this.terrain.rows / this.terrain.chunk_size);
    const chunks: VoxelChunkReference[] = [];
    for (let row = Math.max(0, centre.row - radius); row <= Math.min(rows - 1, centre.row + radius); row += 1) {
      for (let column = Math.max(0, centre.column - radius); column <= Math.min(columns - 1, centre.column + radius); column += 1) {
        chunks.push({ column, row });
      }
    }
    return chunks;
  }

  setLoadedChunks(chunks: readonly VoxelChunkReference[]): boolean {
    const next = new Set(chunks.filter((chunk) => this.#validChunk(chunk)).map(chunkKey));
    if (sameSet(this.#loaded, next)) return false;
    this.#loaded.clear();
    for (const key of next) this.#loaded.add(key);
    return true;
  }

  loadedChunks(): VoxelChunkReference[] {
    return [...this.#loaded].map(parseChunkKey).sort(chunkOrder);
  }

  loadedRegion(): VoxelChunkRegion {
    const chunks = this.loadedChunks();
    if (chunks.length === 0) {
      return {
        minimum: { x: 0, y: 0, z: 0 },
        maximum: {
          x: this.terrain.columns * this.terrain.cell,
          y: this.terrain.maximum_height * this.terrain.block,
          z: this.terrain.rows * this.terrain.cell
        }
      };
    }
    const chunkMetres = this.terrain.chunk_size * this.terrain.cell;
    const minimumColumn = Math.min(...chunks.map((chunk) => chunk.column));
    const maximumColumn = Math.max(...chunks.map((chunk) => chunk.column));
    const minimumRow = Math.min(...chunks.map((chunk) => chunk.row));
    const maximumRow = Math.max(...chunks.map((chunk) => chunk.row));
    return {
      minimum: {
        x: Math.max(0, minimumColumn * chunkMetres),
        y: 0,
        z: Math.max(0, minimumRow * chunkMetres)
      },
      maximum: {
        x: Math.min(this.terrain.columns * this.terrain.cell, (maximumColumn + 1) * chunkMetres),
        y: this.terrain.maximum_height * this.terrain.block,
        z: Math.min(this.terrain.rows * this.terrain.cell, (maximumRow + 1) * chunkMetres)
      }
    };
  }

  isLoaded(coordinate: VoxelCoordinate): boolean {
    return this.#loaded.has(chunkKey(this.chunkAt(coordinate)));
  }

  blocksInChunk(chunk: VoxelChunkReference): VoxelBlock[] {
    if (!this.#validChunk(chunk)) return [];
    const bounds = this.#chunkCellBounds(chunk);
    const blocks: VoxelBlock[] = [];
    for (let row = bounds.rowStart; row < bounds.rowEnd; row += 1) {
      for (let column = bounds.columnStart; column < bounds.columnEnd; column += 1) {
        for (let level = 0; level < this.terrain.maximum_height; level += 1) {
          const coordinate = { column, level, row };
          const material = this.materialAt(coordinate);
          if (material) blocks.push({ coordinate, material, center: this.centerOf(coordinate) });
        }
      }
    }
    return blocks;
  }

  /** Collision/navigation prisms for one chunk, merged across equal vertical runs. */
  boxesInChunk(chunk: VoxelChunkReference): TerrainBox[] {
    const bounds = this.#chunkCellBounds(chunk);
    const runs = new Map<string, Set<string>>();
    for (let row = bounds.rowStart; row < bounds.rowEnd; row += 1) {
      for (let column = bounds.columnStart; column < bounds.columnEnd; column += 1) {
        for (const [start, end] of this.#verticalRuns(column, row)) {
          const signature = `${start}:${end}`;
          const cells = runs.get(signature) ?? new Set<string>();
          cells.add(`${column}:${row}`);
          runs.set(signature, cells);
        }
      }
    }

    const boxes: TerrainBox[] = [];
    for (const [signature, cells] of runs) {
      const [startText, endText] = signature.split(":");
      const start = Number(startText);
      const end = Number(endText);
      while (cells.size > 0) {
        const first = cells.values().next().value as string;
        const [columnText, rowText] = first.split(":");
        const column = Number(columnText);
        const row = Number(rowText);
        let width = 1;
        while (cells.has(`${column + width}:${row}`)) width += 1;
        let depth = 1;
        while (rectangleRowPresent(cells, column, row + depth, width)) depth += 1;
        for (let innerRow = row; innerRow < row + depth; innerRow += 1) {
          for (let innerColumn = column; innerColumn < column + width; innerColumn += 1) {
            cells.delete(`${innerColumn}:${innerRow}`);
          }
        }
        const levels = end - start;
        boxes.push({
          id: `voxel_${chunk.column}_${chunk.row}_${start}_${column}_${row}`,
          center: {
            x: (column + width / 2) * this.terrain.cell,
            y: (start + levels / 2) * this.terrain.block,
            z: (row + depth / 2) * this.terrain.cell
          },
          size: {
            x: width * this.terrain.cell,
            y: levels * this.terrain.block,
            z: depth * this.terrain.cell
          }
        });
      }
    }
    return boxes;
  }

  visibleSurfaceBlocks(origin: Vec3, radius: number, limit: number): VoxelBlock[] {
    const radiusSquared = radius * radius;
    const found: VoxelBlock[] = [];
    for (const chunk of this.loadedChunks()) {
      for (const block of this.blocksInChunk(chunk)) {
        const delta = subtract(block.center, origin);
        if (dot(delta, delta) > radiusSquared || !this.#isSurface(block.coordinate)) continue;
        found.push(block);
      }
    }
    return found
      .sort((left, right) => distanceSquared(left.center, origin) - distanceSquared(right.center, origin))
      .slice(0, limit);
  }

  exposedFaces(coordinate: VoxelCoordinate): VoxelFace[] {
    if (!this.materialAt(coordinate)) return [];
    const center = this.centerOf(coordinate);
    return NEIGHBOURS.flatMap(([dx, dy, dz]) => {
      const neighbour = {
        column: coordinate.column + dx,
        level: coordinate.level + dy,
        row: coordinate.row + dz
      };
      if (this.materialAt(neighbour) !== null) return [];
      const normal = { x: dx, y: dy, z: dz };
      return [{
        normal,
        point: {
          x: center.x + dx * this.terrain.cell / 2,
          y: center.y + dy * this.terrain.block / 2,
          z: center.z + dz * this.terrain.cell / 2
        }
      }];
    });
  }

  placementSupported(coordinate: VoxelCoordinate): boolean {
    return this.contains(coordinate) && this.#hasSupport(coordinate);
  }

  editability(coordinate: VoxelCoordinate): VoxelEditability {
    if (!this.contains(coordinate)) {
      return {
        editable: false,
        code: "voxel_out_of_bounds",
        detail: {
          coordinate,
          bounds: {
            columns: this.terrain.columns,
            rows: this.terrain.rows,
            levels: this.terrain.maximum_height
          }
        }
      };
    }
    if (!this.isLoaded(coordinate)) {
      return {
        editable: false,
        code: "voxel_chunk_unloaded",
        detail: {
          coordinate,
          chunk: this.chunkAt(coordinate),
          loaded_chunks: this.loadedChunks()
        }
      };
    }
    const boundary = coordinate.column === 0
      || coordinate.row === 0
      || coordinate.column === this.terrain.columns - 1
      || coordinate.row === this.terrain.rows - 1;
    if (boundary) {
      return {
        editable: false,
        code: "voxel_boundary_protected",
        detail: { coordinate }
      };
    }
    return { editable: true };
  }

  breakBlock(coordinate: VoxelCoordinate, source: VoxelEditSource): VoxelEditResult {
    this.#assertEditable(coordinate);
    const before = this.materialAt(coordinate);
    if (!before) throw new VoxelEditError("voxel_empty", { coordinate });
    const mutation = this.#record(coordinate, before, null, source);
    this.#inventory[before] += 1;
    return { mutation, chunk: this.chunkAt(coordinate) };
  }

  placeBlock(
    coordinate: VoxelCoordinate,
    material: VoxelMaterial,
    source: VoxelEditSource
  ): VoxelEditResult {
    this.#assertEditable(coordinate);
    if (this.materialAt(coordinate)) {
      throw new VoxelEditError("voxel_occupied", { coordinate, material: this.materialAt(coordinate) });
    }
    if (!this.#hasSupport(coordinate)) {
      throw new VoxelEditError("voxel_unsupported", {
        coordinate,
        recovery: "Choose an empty cell touching the ground or an existing block."
      });
    }
    if (this.#inventory[material] <= 0) {
      throw new VoxelEditError("voxel_material_unavailable", {
        material,
        inventory: this.inventory()
      });
    }
    const mutation = this.#record(coordinate, null, material, source);
    this.#inventory[material] -= 1;
    return { mutation, chunk: this.chunkAt(coordinate) };
  }

  inventory(): Record<VoxelMaterial, number> {
    return { ...this.#inventory };
  }

  snapshot(loadRadiusChunks: number): VoxelWorldState {
    return {
      version: 1,
      revision: this.#revision,
      chunk_size: this.terrain.chunk_size,
      load_radius_chunks: loadRadiusChunks,
      loaded_chunks: this.loadedChunks(),
      mutations: structuredClone(this.#mutations),
      inventory: this.inventory()
    };
  }

  #record(
    coordinate: VoxelCoordinate,
    before: VoxelMaterial | null,
    after: VoxelMaterial | null,
    source: VoxelEditSource
  ): VoxelMutation {
    this.#revision += 1;
    const mutation: VoxelMutation = {
      coordinate: { ...coordinate },
      before,
      after,
      revision: this.#revision,
      source_command_id: source.commandId,
      source_agent_id: source.agentId
    };
    this.#overlay.set(coordinateKey(coordinate), after);
    this.#mutations.push(mutation);
    return structuredClone(mutation);
  }

  #verticalRuns(column: number, row: number): Array<readonly [number, number]> {
    const runs: Array<readonly [number, number]> = [];
    let start: number | null = null;
    for (let level = 0; level <= this.terrain.maximum_height; level += 1) {
      const occupied = level < this.terrain.maximum_height
        && this.materialAt({ column, level, row }) !== null;
      if (occupied && start === null) start = level;
      if (!occupied && start !== null) {
        runs.push([start, level]);
        start = null;
      }
    }
    return runs;
  }

  #isSurface(coordinate: VoxelCoordinate): boolean {
    return NEIGHBOURS.some(([dx, dy, dz]) => this.materialAt({
      column: coordinate.column + dx,
      level: coordinate.level + dy,
      row: coordinate.row + dz
    }) === null);
  }

  #hasSupport(coordinate: VoxelCoordinate): boolean {
    if (coordinate.level === 0) return true;
    return NEIGHBOURS.some(([dx, dy, dz]) => this.materialAt({
      column: coordinate.column + dx,
      level: coordinate.level + dy,
      row: coordinate.row + dz
    }) !== null);
  }

  #assertEditable(coordinate: VoxelCoordinate): void {
    const editability = this.editability(coordinate);
    if (!editability.editable) {
      throw new VoxelEditError(editability.code, editability.detail);
    }
  }

  #assertCoordinate(coordinate: VoxelCoordinate): void {
    if (!this.contains(coordinate)) {
      throw new VoxelEditError("voxel_out_of_bounds", {
        coordinate,
        bounds: {
          columns: this.terrain.columns,
          rows: this.terrain.rows,
          levels: this.terrain.maximum_height
        }
      });
    }
  }

  #validChunk(chunk: VoxelChunkReference): boolean {
    return Number.isInteger(chunk.column)
      && Number.isInteger(chunk.row)
      && chunk.column >= 0
      && chunk.row >= 0
      && chunk.column < Math.ceil(this.terrain.columns / this.terrain.chunk_size)
      && chunk.row < Math.ceil(this.terrain.rows / this.terrain.chunk_size);
  }

  #chunkCellBounds(chunk: VoxelChunkReference): {
    columnStart: number;
    columnEnd: number;
    rowStart: number;
    rowEnd: number;
  } {
    if (!this.#validChunk(chunk)) throw new Error(`Invalid voxel chunk ${chunkKey(chunk)}`);
    const columnStart = chunk.column * this.terrain.chunk_size;
    const rowStart = chunk.row * this.terrain.chunk_size;
    return {
      columnStart,
      columnEnd: Math.min(this.terrain.columns, columnStart + this.terrain.chunk_size),
      rowStart,
      rowEnd: Math.min(this.terrain.rows, rowStart + this.terrain.chunk_size)
    };
  }
}

export class VoxelEditError extends Error {
  readonly code: string;
  readonly detail: Record<string, unknown>;

  constructor(code: string, detail: Record<string, unknown>) {
    super(code);
    this.name = "VoxelEditError";
    this.code = code;
    this.detail = detail;
  }
}

export function chunkKey(chunk: VoxelChunkReference): string {
  return `${chunk.column}:${chunk.row}`;
}

function parseChunkKey(key: string): VoxelChunkReference {
  const [column, row] = key.split(":").map(Number);
  if (column === undefined || row === undefined) throw new Error(`Invalid chunk key ${key}`);
  return { column, row };
}

function coordinateKey(coordinate: VoxelCoordinate): string {
  return `${coordinate.column}:${coordinate.level}:${coordinate.row}`;
}

function baselineMaterial(terrain: Terrain, coordinate: VoxelCoordinate): VoxelMaterial | null {
  const height = terrain.heights[coordinate.row * terrain.columns + coordinate.column] ?? 0;
  if (coordinate.level >= height) return null;
  if (coordinate.level === height - 1) return "grass";
  return coordinate.level === 0 && height >= 3 ? "stone" : "dirt";
}

function rectangleRowPresent(cells: Set<string>, column: number, row: number, width: number): boolean {
  for (let offset = 0; offset < width; offset += 1) {
    if (!cells.has(`${column + offset}:${row}`)) return false;
  }
  return true;
}

function chunkOrder(left: VoxelChunkReference, right: VoxelChunkReference): number {
  return left.row - right.row || left.column - right.column;
}

function sameSet(left: Set<string>, right: Set<string>): boolean {
  return left.size === right.size && [...left].every((value) => right.has(value));
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function subtract(left: Vec3, right: Vec3): Vec3 {
  return { x: left.x - right.x, y: left.y - right.y, z: left.z - right.z };
}

function dot(left: Vec3, right: Vec3): number {
  return left.x * right.x + left.y * right.y + left.z * right.z;
}

function distanceSquared(left: Vec3, right: Vec3): number {
  return dot(subtract(left, right), subtract(left, right));
}
