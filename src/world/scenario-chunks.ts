import type {
  Scenario,
  Vec3
} from "../domain/schema.js";
import type { ScenarioChunk } from "../domain/scenario-chunk.js";

export interface ScenarioRectangleRegion {
  minimum: Pick<Vec3, "x" | "z">;
  maximum: Pick<Vec3, "x" | "z">;
}

export interface ScenarioCircleRegion {
  center: Pick<Vec3, "x" | "z">;
  radius: number;
}

interface ActiveScenarioChunk {
  chunk: ScenarioChunk;
  obstacles: ReadonlyArray<Scenario["obstacles"][number]>;
  objects: ReadonlyArray<Scenario["objects"][number]>;
  zones: ReadonlyArray<Scenario["zones"][number]>;
}

export interface ActiveScenarioRegion {
  chunks: ReadonlyArray<ActiveScenarioChunk>;
  obstacles: ReadonlyArray<Scenario["obstacles"][number]>;
  objects: ReadonlyArray<Scenario["objects"][number]>;
  zones: ReadonlyArray<Scenario["zones"][number]>;
}

export function queryScenarioChunksInRectangle(
  scenario: Scenario,
  region: ScenarioRectangleRegion
): ActiveScenarioRegion {
  assertFinitePoint(region.minimum, "Rectangle minimum");
  assertFinitePoint(region.maximum, "Rectangle maximum");
  if (region.minimum.x > region.maximum.x || region.minimum.z > region.maximum.z) {
    throw new Error("Rectangle minimum must not exceed its maximum");
  }
  return hydrateChunks(scenario, scenario.chunk_manifest.chunks.filter(({ bounds }) => (
    bounds.maximum.x >= region.minimum.x
      && bounds.minimum.x <= region.maximum.x
      && bounds.maximum.z >= region.minimum.z
      && bounds.minimum.z <= region.maximum.z
  )));
}

export function queryScenarioChunksInCircle(
  scenario: Scenario,
  region: ScenarioCircleRegion
): ActiveScenarioRegion {
  assertFinitePoint(region.center, "Circle center");
  if (!Number.isFinite(region.radius) || region.radius < 0) {
    throw new Error("Circle radius must be finite and nonnegative");
  }
  const radiusSquared = region.radius * region.radius;
  return hydrateChunks(scenario, scenario.chunk_manifest.chunks.filter(({ bounds }) => {
    const nearestX = Math.max(bounds.minimum.x, Math.min(region.center.x, bounds.maximum.x));
    const nearestZ = Math.max(bounds.minimum.z, Math.min(region.center.z, bounds.maximum.z));
    const dx = region.center.x - nearestX;
    const dz = region.center.z - nearestZ;
    return dx * dx + dz * dz <= radiusSquared;
  }));
}

function hydrateChunks(
  scenario: Scenario,
  selected: readonly ScenarioChunk[]
): ActiveScenarioRegion {
  const obstacles = new Map(scenario.obstacles.map((entity) => [entity.id, entity]));
  const objects = new Map(scenario.objects.map((entity) => [entity.id, entity]));
  const zones = new Map(scenario.zones.map((entity) => [entity.id, entity]));
  const groups = selected.map((chunk): ActiveScenarioChunk => ({
    chunk,
    obstacles: resolveEntities(chunk.entity_ids.obstacles, obstacles, "obstacle"),
    objects: resolveEntities(chunk.entity_ids.objects, objects, "object"),
    zones: resolveEntities(chunk.entity_ids.zones, zones, "zone")
  }));
  return {
    chunks: groups,
    obstacles: groups.flatMap((group) => group.obstacles),
    objects: groups.flatMap((group) => group.objects),
    zones: groups.flatMap((group) => group.zones)
  };
}

function resolveEntities<T>(
  ids: readonly string[],
  entities: ReadonlyMap<string, T>,
  kind: string
): T[] {
  return ids.map((id) => {
    const entity = entities.get(id);
    if (!entity) throw new Error(`Chunk manifest references unknown ${kind}: ${id}`);
    return entity;
  });
}

function assertFinitePoint(point: { x: number; z: number }, label: string): void {
  if (!Number.isFinite(point.x) || !Number.isFinite(point.z)) {
    throw new Error(`${label} must be finite`);
  }
}
