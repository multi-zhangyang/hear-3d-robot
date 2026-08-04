import { z } from "zod";

const SCENARIO_CHUNK_MANIFEST_VERSION = 1 as const;
export const DEFAULT_SCENARIO_CHUNK_SIZE = 12;

const Point2Schema = z.object({
  x: z.number().finite(),
  z: z.number().finite()
}).strict();

const ScenarioChunkSchema = z.object({
  id: z.string().trim().min(1),
  coordinate: z.object({
    column: z.number().int().nonnegative(),
    row: z.number().int().nonnegative()
  }).strict(),
  bounds: z.object({
    minimum: Point2Schema,
    maximum: Point2Schema
  }).strict(),
  entity_ids: z.object({
    obstacles: z.array(z.string().trim().min(1)),
    objects: z.array(z.string().trim().min(1)),
    zones: z.array(z.string().trim().min(1))
  }).strict()
}).strict();

export type ScenarioChunk = z.infer<typeof ScenarioChunkSchema>;

export const ScenarioChunkManifestSchema = z.object({
  version: z.literal(SCENARIO_CHUNK_MANIFEST_VERSION),
  chunk_size: z.number().finite().positive(),
  grid: z.object({
    columns: z.number().int().positive(),
    rows: z.number().int().positive()
  }).strict(),
  chunks: z.array(ScenarioChunkSchema).min(1)
}).strict();

export type ScenarioChunkManifest = z.infer<typeof ScenarioChunkManifestSchema>;

export interface ScenarioChunkSource {
  bounds: { width: number; depth: number };
  obstacles: ReadonlyArray<{ id: string; center: { x: number; z: number } }>;
  objects: ReadonlyArray<{ id: string; position: { x: number; z: number } }>;
  zones: ReadonlyArray<{ id: string; center: { x: number; z: number } }>;
}

type EntityCategory = keyof ScenarioChunk["entity_ids"];

interface SourceEntity {
  id: string;
  point: { x: number; z: number };
}

const ENTITY_CATEGORIES: readonly EntityCategory[] = ["obstacles", "objects", "zones"];

function scenarioChunkId(column: number, row: number): string {
  if (!Number.isSafeInteger(column) || column < 0
    || !Number.isSafeInteger(row) || row < 0) {
    throw new Error("Scenario chunk coordinates must be nonnegative integers");
  }
  return `chunk_${column}_${row}`;
}

export function buildScenarioChunkManifest(
  source: ScenarioChunkSource,
  chunkSize = DEFAULT_SCENARIO_CHUNK_SIZE
): ScenarioChunkManifest {
  if (!Number.isFinite(chunkSize) || chunkSize <= 0) {
    throw new Error("Scenario chunk size must be positive and finite");
  }
  if (!Number.isFinite(source.bounds.width) || source.bounds.width <= 0
    || !Number.isFinite(source.bounds.depth) || source.bounds.depth <= 0) {
    throw new Error("Scenario bounds must be positive and finite");
  }

  const columns = Math.ceil(source.bounds.width / chunkSize);
  const rows = Math.ceil(source.bounds.depth / chunkSize);
  const chunks: ScenarioChunk[] = [];
  const chunksById = new Map<string, ScenarioChunk>();
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const id = scenarioChunkId(column, row);
      const chunk: ScenarioChunk = {
        id,
        coordinate: { column, row },
        bounds: {
          minimum: { x: column * chunkSize, z: row * chunkSize },
          maximum: {
            x: Math.min((column + 1) * chunkSize, source.bounds.width),
            z: Math.min((row + 1) * chunkSize, source.bounds.depth)
          }
        },
        entity_ids: { obstacles: [], objects: [], zones: [] }
      };
      chunks.push(chunk);
      chunksById.set(id, chunk);
    }
  }

  for (const category of ENTITY_CATEGORIES) {
    for (const entity of sourceEntities(source, category)) {
      const coordinate = coordinateForPoint(entity.point, source.bounds, chunkSize, columns, rows);
      const owner = chunksById.get(scenarioChunkId(coordinate.column, coordinate.row));
      if (!owner) throw new Error(`Unable to resolve chunk owner for ${category}:${entity.id}`);
      owner.entity_ids[category].push(entity.id);
    }
  }
  for (const chunk of chunks) {
    for (const category of ENTITY_CATEGORIES) {
      chunk.entity_ids[category].sort(compareCodePoints);
    }
  }

  return ScenarioChunkManifestSchema.parse({
    version: SCENARIO_CHUNK_MANIFEST_VERSION,
    chunk_size: chunkSize,
    grid: { columns, rows },
    chunks
  });
}

export function rebuildScenarioChunkManifest<
  T extends ScenarioChunkSource & { chunk_manifest?: ScenarioChunkManifest }
>(source: T, chunkSize = source.chunk_manifest?.chunk_size ?? DEFAULT_SCENARIO_CHUNK_SIZE): T & {
  chunk_manifest: ScenarioChunkManifest;
} {
  return {
    ...source,
    chunk_manifest: buildScenarioChunkManifest(source, chunkSize)
  };
}

export function scenarioChunkIntegrityIssues(
  source: ScenarioChunkSource,
  manifest: ScenarioChunkManifest
): string[] {
  const issues: string[] = [];
  const expectedColumns = Math.ceil(source.bounds.width / manifest.chunk_size);
  const expectedRows = Math.ceil(source.bounds.depth / manifest.chunk_size);
  if (manifest.grid.columns !== expectedColumns || manifest.grid.rows !== expectedRows) {
    issues.push(
      `chunk grid must be ${expectedColumns}x${expectedRows} for the world bounds and chunk size`
    );
  }

  const expectedChunkCount = expectedColumns * expectedRows;
  if (manifest.chunks.length !== expectedChunkCount) {
    issues.push(`chunk manifest must contain ${expectedChunkCount} chunks`);
  }

  const chunksById = new Map<string, ScenarioChunk>();
  const coordinates = new Set<string>();
  for (const [index, chunk] of manifest.chunks.entries()) {
    if (chunksById.has(chunk.id)) issues.push(`duplicate chunk ID: ${chunk.id}`);
    else chunksById.set(chunk.id, chunk);

    const coordinateKey = `${chunk.coordinate.column}:${chunk.coordinate.row}`;
    if (coordinates.has(coordinateKey)) issues.push(`duplicate chunk coordinate: ${coordinateKey}`);
    else coordinates.add(coordinateKey);

    const expectedColumn = index % expectedColumns;
    const expectedRow = Math.floor(index / expectedColumns);
    if (expectedRow < expectedRows
      && (chunk.coordinate.column !== expectedColumn || chunk.coordinate.row !== expectedRow)) {
      issues.push(`chunks must use canonical row-major order at index ${index}`);
    }

    if (chunk.coordinate.column >= expectedColumns || chunk.coordinate.row >= expectedRows) {
      issues.push(`chunk ${chunk.id} coordinate is outside the world grid`);
      continue;
    }
    const expectedId = scenarioChunkId(chunk.coordinate.column, chunk.coordinate.row);
    if (chunk.id !== expectedId) {
      issues.push(`chunk at ${coordinateKey} must use stable ID ${expectedId}`);
    }
    const expectedBounds = chunkBounds(
      chunk.coordinate.column,
      chunk.coordinate.row,
      manifest.chunk_size,
      source.bounds
    );
    if (!sameBounds(chunk.bounds, expectedBounds)) {
      issues.push(`chunk ${chunk.id} bounds do not match its coordinate and world bounds`);
    }
    for (const category of ENTITY_CATEGORIES) {
      if (!isCanonicalOrder(chunk.entity_ids[category])) {
        issues.push(`${chunk.id} ${category} IDs must be unique and sorted`);
      }
    }
  }

  for (let row = 0; row < expectedRows; row += 1) {
    for (let column = 0; column < expectedColumns; column += 1) {
      const id = scenarioChunkId(column, row);
      if (!chunksById.has(id)) issues.push(`missing chunk: ${id}`);
    }
  }

  const globalSourceOwners = new Map<string, string>();
  for (const category of ENTITY_CATEGORIES) {
    for (const entity of sourceEntities(source, category)) {
      const previous = globalSourceOwners.get(entity.id);
      if (previous) issues.push(`source entity ID ${entity.id} is shared by ${previous} and ${category}`);
      else globalSourceOwners.set(entity.id, category);
    }
  }

  for (const category of ENTITY_CATEGORIES) {
    const expectedEntities = sourceEntities(source, category);
    const expectedIds = new Set(expectedEntities.map(({ id }) => id));
    const assignments = new Map<string, string[]>();
    for (const chunk of manifest.chunks) {
      for (const id of chunk.entity_ids[category]) {
        const owners = assignments.get(id) ?? [];
        owners.push(chunk.id);
        assignments.set(id, owners);
        if (!expectedIds.has(id)) issues.push(`${chunk.id} references unknown ${category} entity: ${id}`);
      }
    }

    for (const entity of expectedEntities) {
      const owners = assignments.get(entity.id) ?? [];
      if (owners.length === 0) {
        issues.push(`${category} entity ${entity.id} is missing from the chunk manifest`);
        continue;
      }
      if (owners.length !== 1) {
        issues.push(`${category} entity ${entity.id} must belong to exactly one chunk`);
        continue;
      }
      if (!pointInBounds(entity.point, source.bounds)) {
        issues.push(`${category} entity ${entity.id} anchor is outside the world bounds`);
        continue;
      }
      const coordinate = coordinateForPoint(
        entity.point,
        source.bounds,
        manifest.chunk_size,
        expectedColumns,
        expectedRows
      );
      const expectedOwner = scenarioChunkId(coordinate.column, coordinate.row);
      if (owners[0] !== expectedOwner) {
        issues.push(`${category} entity ${entity.id} must belong to ${expectedOwner}`);
      }
    }
  }

  return issues;
}

export function assertScenarioChunkIntegrity(
  source: ScenarioChunkSource,
  manifest: ScenarioChunkManifest
): void {
  const issues = scenarioChunkIntegrityIssues(source, manifest);
  if (issues.length > 0) throw new Error(`Invalid scenario chunk manifest: ${issues.join("; ")}`);
}

export function scenarioChunkIdForPoint(
  source: Pick<ScenarioChunkSource, "bounds">,
  manifest: ScenarioChunkManifest,
  point: { x: number; z: number }
): string {
  if (!pointInBounds(point, source.bounds)) {
    throw new Error("Scenario entity anchor is outside the world bounds");
  }
  const coordinate = coordinateForPoint(
    point,
    source.bounds,
    manifest.chunk_size,
    manifest.grid.columns,
    manifest.grid.rows
  );
  const id = scenarioChunkId(coordinate.column, coordinate.row);
  if (!manifest.chunks.some((chunk) => chunk.id === id)) {
    throw new Error(`Scenario chunk manifest is missing owner ${id}`);
  }
  return id;
}

function sourceEntities(source: ScenarioChunkSource, category: EntityCategory): SourceEntity[] {
  if (category === "objects") {
    return source.objects.map(({ id, position }) => ({ id, point: position }));
  }
  return source[category].map(({ id, center }) => ({ id, point: center }));
}

function coordinateForPoint(
  point: { x: number; z: number },
  bounds: ScenarioChunkSource["bounds"],
  chunkSize: number,
  columns: number,
  rows: number
): { column: number; row: number } {
  const x = Math.min(bounds.width, Math.max(0, point.x));
  const z = Math.min(bounds.depth, Math.max(0, point.z));
  return {
    column: Math.min(columns - 1, Math.floor(x / chunkSize)),
    row: Math.min(rows - 1, Math.floor(z / chunkSize))
  };
}

function chunkBounds(
  column: number,
  row: number,
  chunkSize: number,
  bounds: ScenarioChunkSource["bounds"]
): ScenarioChunk["bounds"] {
  return {
    minimum: { x: column * chunkSize, z: row * chunkSize },
    maximum: {
      x: Math.min((column + 1) * chunkSize, bounds.width),
      z: Math.min((row + 1) * chunkSize, bounds.depth)
    }
  };
}

function sameBounds(left: ScenarioChunk["bounds"], right: ScenarioChunk["bounds"]): boolean {
  return nearlyEqual(left.minimum.x, right.minimum.x)
    && nearlyEqual(left.minimum.z, right.minimum.z)
    && nearlyEqual(left.maximum.x, right.maximum.x)
    && nearlyEqual(left.maximum.z, right.maximum.z);
}

function nearlyEqual(left: number, right: number): boolean {
  return Math.abs(left - right) <= 1e-9;
}

function pointInBounds(
  point: { x: number; z: number },
  bounds: ScenarioChunkSource["bounds"]
): boolean {
  return point.x >= 0 && point.x <= bounds.width && point.z >= 0 && point.z <= bounds.depth;
}

function isCanonicalOrder(ids: readonly string[]): boolean {
  return ids.every((id, index) => index === 0 || compareCodePoints(ids[index - 1]!, id) < 0);
}

function compareCodePoints(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}
