import type { ScenarioDefinition, Vec3 } from "../types";

type ScenarioChunk = ScenarioDefinition["chunk_manifest"]["chunks"][number];

export function visibleScenarioChunkIds(
  scenario: ScenarioDefinition,
  center: Pick<Vec3, "x" | "z">,
  radius = scenario.visibility_radius + scenario.chunk_manifest.chunk_size / 2
): Set<string> {
  if (!Number.isFinite(radius) || radius < 0) {
    throw new Error("Scenario chunk visibility radius must be finite and non-negative");
  }
  const manifest = scenario.chunk_manifest;
  const minimumColumn = boundedGridIndex(
    center.x - radius,
    manifest.chunk_size,
    manifest.grid.columns
  );
  const maximumColumn = boundedGridIndex(
    center.x + radius,
    manifest.chunk_size,
    manifest.grid.columns
  );
  const minimumRow = boundedGridIndex(
    center.z - radius,
    manifest.chunk_size,
    manifest.grid.rows
  );
  const maximumRow = boundedGridIndex(
    center.z + radius,
    manifest.chunk_size,
    manifest.grid.rows
  );
  if (center.x + radius < 0 || center.z + radius < 0
    || center.x - radius > scenario.bounds.width
    || center.z - radius > scenario.bounds.depth) return new Set();

  const visible = new Set<string>();
  for (let row = minimumRow; row <= maximumRow; row += 1) {
    for (let column = minimumColumn; column <= maximumColumn; column += 1) {
      const chunk = canonicalChunk(scenario, column, row);
      if (circleIntersectsChunk(center, radius, chunk)) visible.add(chunk.id);
    }
  }
  return visible;
}

export function scenarioChunkAt(
  scenario: ScenarioDefinition,
  point: Pick<Vec3, "x" | "z">
): ScenarioChunk | undefined {
  if (!Number.isFinite(point.x) || !Number.isFinite(point.z)) return undefined;
  if (point.x < 0 || point.z < 0
    || point.x > scenario.bounds.width || point.z > scenario.bounds.depth) return undefined;
  const manifest = scenario.chunk_manifest;
  const column = boundedGridIndex(point.x, manifest.chunk_size, manifest.grid.columns);
  const row = boundedGridIndex(point.z, manifest.chunk_size, manifest.grid.rows);
  return canonicalChunk(scenario, column, row);
}

function canonicalChunk(
  scenario: ScenarioDefinition,
  column: number,
  row: number
): ScenarioChunk {
  const manifest = scenario.chunk_manifest;
  const chunk = manifest.chunks[row * manifest.grid.columns + column];
  if (!chunk || chunk.coordinate.column !== column || chunk.coordinate.row !== row) {
    throw new Error(`Scenario chunk manifest is not in canonical grid order: ${column},${row}`);
  }
  return chunk;
}

function boundedGridIndex(value: number, chunkSize: number, count: number): number {
  return Math.min(count - 1, Math.max(0, Math.floor(value / chunkSize)));
}

function circleIntersectsChunk(
  center: Pick<Vec3, "x" | "z">,
  radius: number,
  chunk: ScenarioChunk
): boolean {
  const nearestX = clamp(center.x, chunk.bounds.minimum.x, chunk.bounds.maximum.x);
  const nearestZ = clamp(center.z, chunk.bounds.minimum.z, chunk.bounds.maximum.z);
  return Math.hypot(center.x - nearestX, center.z - nearestZ) <= radius;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}
