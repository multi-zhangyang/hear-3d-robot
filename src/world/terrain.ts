/**
 * Voxel terrain: a square grid of unit columns, each raised by a whole number
 * of blocks.
 *
 * The point of expressing terrain this way rather than as a list of obstacle
 * boxes is that it is compact and it is generated. A run stores its scenario,
 * and a grid of small integers stays small enough to store whole — the same
 * scene written as one box per column would be an order of magnitude larger and
 * would have to be repeated in every world frame the console streams.
 *
 * A column at level 0 is floor the base can drive across. Any column above that
 * is solid from the ground up, so it blocks both the navmesh and the rig. There
 * are no overhangs and no gaps underneath, which is what lets the whole surface
 * be described by one height per column.
 */
import type { Scenario, Terrain, Vec3 } from "../domain/schema.js";
import { createRandom, deriveSeed, randomBetween } from "./random.js";

export type { Terrain };

export interface TerrainBox {
  id: string;
  center: Vec3;
  size: Vec3;
}

/**
 * Every static solid the physics scene and the navmesh must both account for:
 * the scenario's authored obstacles plus, when the world is voxel terrain, the
 * merged terrain boxes.
 *
 * Terrain is deliberately not stored in `obstacles`. That list is part of every
 * world snapshot, and a few hundred merged boxes repeated in every streamed
 * frame would dominate the run journal. The grid travels once with the scenario
 * and is expanded here, where it is needed.
 *
 * The merge is memoised against the terrain object because both callers ask for
 * it while building and the grid never changes afterwards.
 */
export function staticSolids(scenario: Scenario): TerrainBox[] {
  if (!scenario.terrain) return scenario.obstacles.map(cloneBox);
  const terrain = scenario.terrain;
  let boxes = mergedBoxes.get(terrain);
  if (!boxes) {
    boxes = terrainBoxes(terrain);
    mergedBoxes.set(terrain, boxes);
  }
  return [...scenario.obstacles.map(cloneBox), ...boxes];
}

const mergedBoxes = new WeakMap<Terrain, TerrainBox[]>();

function cloneBox(box: TerrainBox): TerrainBox {
  return { id: box.id, center: { ...box.center }, size: { ...box.size } };
}

/** Grid coordinates of one column. */
export interface Cell {
  column: number;
  row: number;
}

export interface TerrainShape {
  /** Columns per side. */
  size: number;
  /** Side length of one column, in metres. */
  cell: number;
  /** Height of one block level, in metres. */
  block: number;
  /** Side length of one independently loaded physics/render chunk, in cells. */
  chunk_size: number;
  /** Highest editable voxel level, exclusive. */
  maximum_height: number;
  /** Highest level the interior may reach. */
  relief: number;
  /**
   * Fraction of interior columns that end up raised, before connectivity is
   * enforced. Higher values give a denser, more maze-like world.
   */
  density: number;
}

export function terrainHeight(terrain: Terrain, cell: Cell): number {
  return terrain.heights[cell.row * terrain.columns + cell.column] ?? 0;
}

/** World-space centre of a column's floor footprint. */
export function cellCenter(terrain: Terrain, cell: Cell): Vec3 {
  return {
    x: (cell.column + 0.5) * terrain.cell,
    y: 0,
    z: (cell.row + 0.5) * terrain.cell
  };
}

/** The column containing a world-space point, clamped to the grid. */
export function cellAt(terrain: Terrain, point: Vec3): Cell {
  return {
    column: clampIndex(Math.floor(point.x / terrain.cell), terrain.columns),
    row: clampIndex(Math.floor(point.z / terrain.cell), terrain.rows)
  };
}

/**
 * Generates terrain from a seed.
 *
 * Relief comes from value noise on a coarse lattice rather than from
 * independent per-column draws: independent draws produce a field of isolated
 * pillars, which is noise, not landscape. Interpolating a coarse lattice gives
 * ridges and basins with a recognisable shape, and quantising the result to
 * whole levels is what makes it read as blocks.
 *
 * The outer ring is always raised, so the world is walled rather than ending at
 * an invisible boundary the base is simply refused passage through.
 */
export function generateTerrain(shape: TerrainShape, seed: number): Terrain {
  const random = createRandom(deriveSeed(seed, "terrain"));
  const lattice = Math.max(3, Math.round(shape.size / 4));
  const field = valueNoiseField(lattice, random);
  const heights: number[] = new Array<number>(shape.size * shape.size).fill(0);

  // The threshold is chosen so that the requested fraction of columns clears it,
  // measured against the field that was actually generated. Comparing against a
  // fixed constant would let one seed produce an empty plain and the next a
  // solid wall, because value noise does not span the same range every time.
  const samples: number[] = [];
  for (let row = 0; row < shape.size; row += 1) {
    for (let column = 0; column < shape.size; column += 1) {
      samples.push(sampleField(field, lattice, column / shape.size, row / shape.size));
    }
  }
  const threshold = quantile(samples, 1 - shape.density);

  for (let row = 0; row < shape.size; row += 1) {
    for (let column = 0; column < shape.size; column += 1) {
      const index = row * shape.size + column;
      if (row === 0 || column === 0 || row === shape.size - 1 || column === shape.size - 1) {
        heights[index] = shape.relief;
        continue;
      }
      const value = samples[index]!;
      if (value <= threshold) continue;
      const span = Math.max(1e-6, 1 - threshold);
      heights[index] = Math.min(
        shape.relief,
        1 + Math.floor(((value - threshold) / span) * shape.relief)
      );
    }
  }

  const terrain: Terrain = {
    cell: shape.cell,
    columns: shape.size,
    rows: shape.size,
    block: shape.block,
    chunk_size: shape.chunk_size,
    maximum_height: shape.maximum_height,
    heights
  };
  openLargestRegion(terrain);
  return terrain;
}

/**
 * Keeps the largest connected patch of floor and fills in every smaller one.
 *
 * Noise routinely leaves a few floor cells walled off from the rest. They are
 * not reachable, so leaving them open would mean the world advertises standing
 * room the base can never occupy — and a generated object placed in one would
 * make the mission impossible rather than hard.
 */
function openLargestRegion(terrain: Terrain): void {
  const regions = new Int32Array(terrain.columns * terrain.rows).fill(-1);
  const sizes: number[] = [];
  for (let row = 0; row < terrain.rows; row += 1) {
    for (let column = 0; column < terrain.columns; column += 1) {
      const index = row * terrain.columns + column;
      if (terrain.heights[index] !== 0 || regions[index] !== -1) continue;
      const region = sizes.length;
      let size = 0;
      const queue: number[] = [index];
      regions[index] = region;
      while (queue.length > 0) {
        const current = queue.pop()!;
        size += 1;
        const currentColumn = current % terrain.columns;
        const currentRow = (current - currentColumn) / terrain.columns;
        for (const [dx, dz] of NEIGHBOURS) {
          const nextColumn = currentColumn + dx;
          const nextRow = currentRow + dz;
          if (nextColumn < 0 || nextRow < 0
            || nextColumn >= terrain.columns || nextRow >= terrain.rows) continue;
          const next = nextRow * terrain.columns + nextColumn;
          if (terrain.heights[next] !== 0 || regions[next] !== -1) continue;
          regions[next] = region;
          queue.push(next);
        }
      }
      sizes.push(size);
    }
  }
  if (sizes.length === 0) return;
  let largest = 0;
  for (const [region, size] of sizes.entries()) {
    if (size > sizes[largest]!) largest = region;
  }
  for (let index = 0; index < regions.length; index += 1) {
    if (terrain.heights[index] === 0 && regions[index] !== largest) terrain.heights[index] = 1;
  }
}

const NEIGHBOURS: ReadonlyArray<readonly [number, number]> = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1]
];

/** Every column the base could stand on, in row-major order. */
export function walkableCells(terrain: Terrain): Cell[] {
  const cells: Cell[] = [];
  for (let row = 0; row < terrain.rows; row += 1) {
    for (let column = 0; column < terrain.columns; column += 1) {
      if (terrain.heights[row * terrain.columns + column] === 0) cells.push({ column, row });
    }
  }
  return cells;
}

/**
 * Whether a disc of the given radius centred on a column sits entirely on
 * floor. Placing the base or an object needs clearance in metres, not one free
 * column, because both are wider than a column at the sizes this world uses.
 */
export function cellClearance(terrain: Terrain, cell: Cell, radius: number): boolean {
  const reach = Math.ceil(radius / terrain.cell);
  const centre = cellCenter(terrain, cell);
  for (let row = cell.row - reach; row <= cell.row + reach; row += 1) {
    for (let column = cell.column - reach; column <= cell.column + reach; column += 1) {
      if (column < 0 || row < 0 || column >= terrain.columns || row >= terrain.rows) return false;
      if (terrain.heights[row * terrain.columns + column] === 0) continue;
      // Nearest point of the raised column's square footprint to the centre.
      const minimumX = column * terrain.cell;
      const minimumZ = row * terrain.cell;
      const nearestX = Math.min(Math.max(centre.x, minimumX), minimumX + terrain.cell);
      const nearestZ = Math.min(Math.max(centre.z, minimumZ), minimumZ + terrain.cell);
      if (Math.hypot(nearestX - centre.x, nearestZ - centre.z) < radius) return false;
    }
  }
  return true;
}

/**
 * Merges the raised columns into as few boxes as possible.
 *
 * One box per column is correct and unusably slow: the navmesh triangulates
 * every face and the physics world creates a collider for each, so a few
 * hundred columns cost a few hundred of both. Columns of equal height that sit
 * next to each other describe exactly the same solid as one larger box, so the
 * grid is decomposed greedily into maximal rectangles of constant height.
 */
export function terrainBoxes(terrain: Terrain): TerrainBox[] {
  const consumed = new Uint8Array(terrain.columns * terrain.rows);
  const boxes: TerrainBox[] = [];
  for (let row = 0; row < terrain.rows; row += 1) {
    for (let column = 0; column < terrain.columns; column += 1) {
      const index = row * terrain.columns + column;
      const height = terrain.heights[index]!;
      if (height === 0 || consumed[index] === 1) continue;

      let width = 1;
      while (column + width < terrain.columns
        && consumed[index + width] === 0
        && terrain.heights[index + width] === height) width += 1;

      let depth = 1;
      while (row + depth < terrain.rows && rowMatches(terrain, consumed, column, row + depth, width, height)) {
        depth += 1;
      }

      for (let innerRow = row; innerRow < row + depth; innerRow += 1) {
        consumed.fill(1, innerRow * terrain.columns + column, innerRow * terrain.columns + column + width);
      }

      const solidHeight = height * terrain.block;
      boxes.push({
        id: `terrain_${boxes.length}`,
        center: {
          x: (column + width / 2) * terrain.cell,
          y: solidHeight / 2,
          z: (row + depth / 2) * terrain.cell
        },
        size: {
          x: width * terrain.cell,
          y: solidHeight,
          z: depth * terrain.cell
        }
      });
    }
  }
  return boxes;
}

function rowMatches(
  terrain: Terrain,
  consumed: Uint8Array,
  column: number,
  row: number,
  width: number,
  height: number
): boolean {
  const start = row * terrain.columns + column;
  for (let offset = 0; offset < width; offset += 1) {
    if (consumed[start + offset] === 1 || terrain.heights[start + offset] !== height) return false;
  }
  return true;
}

/** Bilinearly interpolated value noise on a `lattice`×`lattice` grid of corners. */
function valueNoiseField(lattice: number, random: () => number): number[] {
  const field: number[] = [];
  for (let index = 0; index < lattice * lattice; index += 1) field.push(random());
  return field;
}

function sampleField(field: number[], lattice: number, u: number, v: number): number {
  const x = u * (lattice - 1);
  const z = v * (lattice - 1);
  const x0 = Math.min(lattice - 1, Math.floor(x));
  const z0 = Math.min(lattice - 1, Math.floor(z));
  const x1 = Math.min(lattice - 1, x0 + 1);
  const z1 = Math.min(lattice - 1, z0 + 1);
  const tx = smoothstep(x - x0);
  const tz = smoothstep(z - z0);
  const top = mix(field[z0 * lattice + x0]!, field[z0 * lattice + x1]!, tx);
  const bottom = mix(field[z1 * lattice + x0]!, field[z1 * lattice + x1]!, tx);
  return mix(top, bottom, tz);
}

function smoothstep(t: number): number {
  return t * t * (3 - 2 * t);
}

function mix(left: number, right: number, t: number): number {
  return left + (right - left) * t;
}

function quantile(values: number[], fraction: number): number {
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.min(
    sorted.length - 1,
    Math.max(0, Math.round(fraction * (sorted.length - 1)))
  );
  return sorted[index]!;
}

function clampIndex(value: number, limit: number): number {
  return Math.min(limit - 1, Math.max(0, value));
}

/** A point uniformly inside a column, kept clear of the seams with its neighbours. */
export function pointInCell(terrain: Terrain, cell: Cell, random: () => number, inset: number): Vec3 {
  const centre = cellCenter(terrain, cell);
  const range = Math.max(0, terrain.cell / 2 - inset);
  return {
    x: centre.x + randomBetween(random, -range, range),
    y: 0,
    z: centre.z + randomBetween(random, -range, range)
  };
}
