import type { Scenario, Vec3 } from "../domain/schema.js";
import {
  NAVIGATION_TILE_WORLD_SIZE,
  type NavigationBuildScope,
  type NavigationPlanarRegion
} from "./navigation.js";

const CORRIDOR_EXPANSION_TILES = [1, 2, 4] as const;

interface TileCoordinate {
  column: number;
  row: number;
}

export interface NavigationCorridorBuildStage {
  key: string;
  expansionTiles: number | null;
  selectedTileCount: number;
  totalTileCount: number;
  fullWorld: boolean;
  scope: NavigationBuildScope;
}

export function navigationCorridorBuildStages(
  scenario: Scenario,
  start: Pick<Vec3, "x" | "z">,
  target: Pick<Vec3, "x" | "z">
): NavigationCorridorBuildStage[] {
  assertPoint(start, "Navigation corridor start");
  assertPoint(target, "Navigation corridor target");
  const tileSize = NAVIGATION_TILE_WORLD_SIZE;
  const columns = Math.ceil(scenario.bounds.width / tileSize);
  const rows = Math.ceil(scenario.bounds.depth / tileSize);
  const totalTileCount = columns * rows;
  const route = segmentTiles(
    clampedPoint(start, scenario),
    clampedPoint(target, scenario),
    tileSize,
    columns,
    rows
  );
  const stages: NavigationCorridorBuildStage[] = [];
  const seen = new Set<string>();
  for (const expansionTiles of CORRIDOR_EXPANSION_TILES) {
    const selected = expandTiles(route, expansionTiles, columns, rows);
    appendStage(
      stages,
      seen,
      scenario,
      selected,
      totalTileCount,
      expansionTiles,
      tileSize,
      columns,
      rows
    );
    if (selected.size === totalTileCount) return stages;
  }
  appendStage(
    stages,
    seen,
    scenario,
    allTiles(columns, rows),
    totalTileCount,
    null,
    tileSize,
    columns,
    rows
  );
  return stages;
}

function appendStage(
  stages: NavigationCorridorBuildStage[],
  seen: Set<string>,
  scenario: Scenario,
  selected: ReadonlySet<string>,
  totalTileCount: number,
  expansionTiles: number | null,
  tileSize: number,
  columns: number,
  rows: number
): void {
  const keys = [...selected].sort(compareTileKeys);
  const key = keys.join("|");
  if (seen.has(key)) return;
  seen.add(key);
  const regions = keys.map((entry) => tileRegion(
    parseTileKey(entry),
    scenario,
    tileSize,
    columns,
    rows
  ));
  const region = enclosingRegion(regions);
  stages.push({
    key,
    expansionTiles,
    selectedTileCount: selected.size,
    totalTileCount,
    fullWorld: selected.size === totalTileCount,
    scope: {
      region,
      walkableRegions: regions,
      terrainSolids: []
    }
  });
}

function segmentTiles(
  start: Pick<Vec3, "x" | "z">,
  target: Pick<Vec3, "x" | "z">,
  tileSize: number,
  columns: number,
  rows: number
): Set<string> {
  let current = coordinateForPoint(start, tileSize, columns, rows);
  const end = coordinateForPoint(target, tileSize, columns, rows);
  const selected = new Set<string>([tileKey(current)]);
  const dx = target.x - start.x;
  const dz = target.z - start.z;
  const stepX = Math.sign(dx);
  const stepZ = Math.sign(dz);
  const tDeltaX = stepX === 0 ? Number.POSITIVE_INFINITY : tileSize / Math.abs(dx);
  const tDeltaZ = stepZ === 0 ? Number.POSITIVE_INFINITY : tileSize / Math.abs(dz);
  let tMaxX = stepX === 0
    ? Number.POSITIVE_INFINITY
    : ((stepX > 0 ? (current.column + 1) * tileSize : current.column * tileSize)
      - start.x) / dx;
  let tMaxZ = stepZ === 0
    ? Number.POSITIVE_INFINITY
    : ((stepZ > 0 ? (current.row + 1) * tileSize : current.row * tileSize)
      - start.z) / dz;
  const maximumSteps = columns * rows * 3;
  for (let step = 0;
    (current.column !== end.column || current.row !== end.row) && step < maximumSteps;
    step += 1) {
    if (Math.abs(tMaxX - tMaxZ) <= 1e-12) {
      const horizontal = {
        column: current.column + stepX,
        row: current.row
      };
      const vertical = {
        column: current.column,
        row: current.row + stepZ
      };
      addInBounds(selected, horizontal, columns, rows);
      addInBounds(selected, vertical, columns, rows);
      current = {
        column: current.column + stepX,
        row: current.row + stepZ
      };
      tMaxX += tDeltaX;
      tMaxZ += tDeltaZ;
    } else if (tMaxX < tMaxZ) {
      current = { column: current.column + stepX, row: current.row };
      tMaxX += tDeltaX;
    } else {
      current = { column: current.column, row: current.row + stepZ };
      tMaxZ += tDeltaZ;
    }
    addInBounds(selected, current, columns, rows);
  }
  if (current.column !== end.column || current.row !== end.row) {
    throw new Error("Navigation corridor traversal exceeded its bounded tile grid");
  }
  selected.add(tileKey(end));
  return selected;
}

function expandTiles(
  source: ReadonlySet<string>,
  radius: number,
  columns: number,
  rows: number
): Set<string> {
  const selected = new Set<string>();
  for (const entry of source) {
    const center = parseTileKey(entry);
    for (let row = center.row - radius; row <= center.row + radius; row += 1) {
      for (let column = center.column - radius;
        column <= center.column + radius;
        column += 1) {
        addInBounds(selected, { column, row }, columns, rows);
      }
    }
  }
  return selected;
}

function allTiles(columns: number, rows: number): Set<string> {
  const selected = new Set<string>();
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      selected.add(tileKey({ column, row }));
    }
  }
  return selected;
}

function tileRegion(
  coordinate: TileCoordinate,
  scenario: Scenario,
  tileSize: number,
  columns: number,
  rows: number
): NavigationPlanarRegion {
  if (!inBounds(coordinate, columns, rows)) {
    throw new Error(`Navigation tile is outside the world: ${tileKey(coordinate)}`);
  }
  return {
    minimum: {
      x: coordinate.column * tileSize,
      z: coordinate.row * tileSize
    },
    maximum: {
      x: Math.min((coordinate.column + 1) * tileSize, scenario.bounds.width),
      z: Math.min((coordinate.row + 1) * tileSize, scenario.bounds.depth)
    }
  };
}

function enclosingRegion(regions: readonly NavigationPlanarRegion[]): NavigationPlanarRegion {
  if (regions.length === 0) throw new Error("Navigation corridor cannot be empty");
  return {
    minimum: {
      x: Math.min(...regions.map((region) => region.minimum.x)),
      z: Math.min(...regions.map((region) => region.minimum.z))
    },
    maximum: {
      x: Math.max(...regions.map((region) => region.maximum.x)),
      z: Math.max(...regions.map((region) => region.maximum.z))
    }
  };
}

function coordinateForPoint(
  point: Pick<Vec3, "x" | "z">,
  tileSize: number,
  columns: number,
  rows: number
): TileCoordinate {
  return {
    column: Math.min(columns - 1, Math.max(0, Math.floor(point.x / tileSize))),
    row: Math.min(rows - 1, Math.max(0, Math.floor(point.z / tileSize)))
  };
}

function clampedPoint(
  point: Pick<Vec3, "x" | "z">,
  scenario: Scenario
): Pick<Vec3, "x" | "z"> {
  return {
    x: Math.min(scenario.bounds.width, Math.max(0, point.x)),
    z: Math.min(scenario.bounds.depth, Math.max(0, point.z))
  };
}

function addInBounds(
  selected: Set<string>,
  coordinate: TileCoordinate,
  columns: number,
  rows: number
): void {
  if (inBounds(coordinate, columns, rows)) selected.add(tileKey(coordinate));
}

function inBounds(
  coordinate: TileCoordinate,
  columns: number,
  rows: number
): boolean {
  return coordinate.column >= 0 && coordinate.column < columns
    && coordinate.row >= 0 && coordinate.row < rows;
}

function tileKey(coordinate: TileCoordinate): string {
  return `${String(coordinate.column)}:${String(coordinate.row)}`;
}

function parseTileKey(value: string): TileCoordinate {
  const [column, row, extra] = value.split(":");
  if (extra !== undefined) throw new Error(`Invalid navigation tile key: ${value}`);
  const parsed = { column: Number(column), row: Number(row) };
  if (!Number.isSafeInteger(parsed.column) || !Number.isSafeInteger(parsed.row)) {
    throw new Error(`Invalid navigation tile key: ${value}`);
  }
  return parsed;
}

function compareTileKeys(left: string, right: string): number {
  const leftCoordinate = parseTileKey(left);
  const rightCoordinate = parseTileKey(right);
  return leftCoordinate.row - rightCoordinate.row
    || leftCoordinate.column - rightCoordinate.column;
}

function assertPoint(point: Pick<Vec3, "x" | "z">, label: string): void {
  if (!Number.isFinite(point.x) || !Number.isFinite(point.z)) {
    throw new Error(`${label} must be finite`);
  }
}
