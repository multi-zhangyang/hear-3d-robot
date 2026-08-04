import type { Scenario, Vec3 } from "../../domain/schema.js";

const PHYSICAL_REGION_NEIGHBOURHOOD_CHUNKS = 1;

export interface HumanoidPhysicalRegion {
  key: string;
  anchorChunkIds: readonly string[];
  chunkIds: readonly string[];
  bounds: ReadonlyArray<{
    minimum: Pick<Vec3, "x" | "z">;
    maximum: Pick<Vec3, "x" | "z">;
  }>;
}

export function humanoidPhysicalRegion(
  scenario: Scenario,
  anchors: readonly Pick<Vec3, "x" | "z">[]
): HumanoidPhysicalRegion {
  if (anchors.length === 0) {
    throw new Error("Humanoid physical region requires at least one anchor");
  }
  const chunksByCoordinate = new Map(scenario.chunk_manifest.chunks.map((chunk) => (
    [`${String(chunk.coordinate.column)}:${String(chunk.coordinate.row)}`, chunk]
  )));
  const anchorChunks = anchors.map((anchor, index) => {
    assertFinitePoint(anchor, `Humanoid physical anchor ${String(index)}`);
    const column = Math.min(
      scenario.chunk_manifest.grid.columns - 1,
      Math.max(0, Math.floor(clamp(anchor.x, 0, scenario.bounds.width)
        / scenario.chunk_manifest.chunk_size))
    );
    const row = Math.min(
      scenario.chunk_manifest.grid.rows - 1,
      Math.max(0, Math.floor(clamp(anchor.z, 0, scenario.bounds.depth)
        / scenario.chunk_manifest.chunk_size))
    );
    const chunk = chunksByCoordinate.get(`${String(column)}:${String(row)}`);
    if (!chunk) throw new Error(`Scenario physical anchor has no chunk: ${column}:${row}`);
    return chunk;
  });
  const selected = new Map<string, typeof scenario.chunk_manifest.chunks[number]>();
  for (const anchorChunk of anchorChunks) {
    for (let row = anchorChunk.coordinate.row - PHYSICAL_REGION_NEIGHBOURHOOD_CHUNKS;
      row <= anchorChunk.coordinate.row + PHYSICAL_REGION_NEIGHBOURHOOD_CHUNKS;
      row += 1) {
      for (let column = anchorChunk.coordinate.column
        - PHYSICAL_REGION_NEIGHBOURHOOD_CHUNKS;
        column <= anchorChunk.coordinate.column
          + PHYSICAL_REGION_NEIGHBOURHOOD_CHUNKS;
        column += 1) {
        const chunk = chunksByCoordinate.get(`${String(column)}:${String(row)}`);
        if (chunk) selected.set(chunk.id, chunk);
      }
    }
  }
  const chunks = [...selected.values()].sort((left, right) => (
    left.coordinate.row - right.coordinate.row
      || left.coordinate.column - right.coordinate.column
  ));
  const anchorChunkIds = [...new Set(anchorChunks.map(({ id }) => id))].sort();
  const chunkIds = chunks.map(({ id }) => id);
  return {
    key: chunkIds.join("|"),
    anchorChunkIds,
    chunkIds,
    bounds: chunks.map(({ bounds }) => structuredClone(bounds))
  };
}

export function humanoidPhysicalRegionIncludesBox(
  region: HumanoidPhysicalRegion,
  box: { center: Vec3; size: Vec3 }
): boolean {
  const values = [
    box.center.x,
    box.center.z,
    box.size.x,
    box.size.z
  ];
  if (!values.every(Number.isFinite) || box.size.x <= 0 || box.size.z <= 0) {
    throw new Error("Humanoid physical region box must have finite positive planar extents");
  }
  return region.bounds.some((bounds) => (
    box.center.x + box.size.x / 2 >= bounds.minimum.x
      && box.center.x - box.size.x / 2 <= bounds.maximum.x
      && box.center.z + box.size.z / 2 >= bounds.minimum.z
      && box.center.z - box.size.z / 2 <= bounds.maximum.z
  ));
}

function assertFinitePoint(point: Pick<Vec3, "x" | "z">, label: string): void {
  if (!Number.isFinite(point.x) || !Number.isFinite(point.z)) {
    throw new Error(`${label} must be finite`);
  }
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}
