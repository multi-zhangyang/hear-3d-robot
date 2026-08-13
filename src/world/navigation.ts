import * as RecastNavigation from "recast-navigation";
import * as RecastGenerators from "recast-navigation/generators";
import type { Scenario, Vec3 } from "../domain/schema.js";

interface RecastNavMesh {
  destroy(): void;
}

type RecastTileCacheObstacle = object;

interface RecastTileCache {
  addBoxObstacle(position: Vec3, halfExtents: Vec3, angle: number):
    | { success: true; status: number; obstacle: RecastTileCacheObstacle }
    | { success: false; status: number; obstacle?: RecastTileCacheObstacle };
  removeObstacle(obstacle: RecastTileCacheObstacle): {
    success: boolean;
    status: number;
  };
  update(navMesh: RecastNavMesh): {
    success: boolean;
    status: number;
    upToDate: boolean;
  };
  destroy(): void;
}

interface RecastNavMeshQuery {
  defaultQueryHalfExtents: Vec3;
  findClosestPoint(position: Vec3): {
    success: boolean;
    point: Vec3;
  };
  computePath(start: Vec3, target: Vec3, options: {
    maxPathPolys: number;
    maxStraightPathPoints: number;
  }): {
    success: boolean;
    error?: { name: string };
    path: Vec3[];
  };
  destroy(): void;
}

interface RecastApi {
  init(): Promise<void>;
  NavMeshQuery: new (
    navMesh: RecastNavMesh,
    options: { maxNodes: number }
  ) => RecastNavMeshQuery;
}

interface TileCacheNavMeshConfig {
  cs: number;
  ch: number;
  tileSize: number;
  walkableSlopeAngle: number;
  walkableHeight: number;
  walkableClimb: number;
  walkableRadius: number;
  maxSimplificationError: number;
  mergeRegionArea: number;
  maxVertsPerPoly: number;
  detailSampleDist: number;
  detailSampleMaxError: number;
  expectedLayersPerTile: number;
  maxObstacles: number;
  bounds: [[number, number, number], [number, number, number]];
}

interface RecastGeneratorsApi {
  generateTileCache(
    positions: ArrayLike<number>,
    indices: ArrayLike<number>,
    config: TileCacheNavMeshConfig
  ):
    | { success: true; navMesh: RecastNavMesh; tileCache: RecastTileCache }
    | { success: false; error: string; navMesh?: undefined; tileCache?: undefined };
}

const recastApi = RecastNavigation as unknown as RecastApi;
const recastGeneratorsApi = RecastGenerators as unknown as RecastGeneratorsApi;

export interface NavigationPlan {
  waypoints: Vec3[];
  distance: number;
  resolvedTarget: Vec3;
  projectionDistance: number;
}

export type NavigationPlanningFailureCode =
  | "start_off_mesh"
  | "target_off_mesh"
  | "target_projection_exceeded"
  | "path_not_found";

/**
 * A valid navigation mesh can still reject a route. Callers may retry these
 * failures with a wider build scope; mesh generation and tile-cache failures
 * deliberately remain ordinary errors and must surface immediately.
 */
export class NavigationPlanningError extends Error {
  readonly code: NavigationPlanningFailureCode;
  readonly projectedTarget: Vec3 | undefined;
  readonly projectionDistance: number | undefined;

  constructor(
    code: NavigationPlanningFailureCode,
    message: string,
    evidence: {
      projectedTarget?: Vec3;
      projectionDistance?: number;
    } = {}
  ) {
    super(message);
    this.name = "NavigationPlanningError";
    this.code = code;
    this.projectedTarget = evidence.projectedTarget
      ? { ...evidence.projectedTarget }
      : undefined;
    this.projectionDistance = evidence.projectionDistance;
  }
}

export interface StandoffPose {
  position: Vec3;
  radius: number;
  distance: number;
}

export interface NavigationObstacle {
  id: string;
  center: Vec3;
  halfExtents: Vec3;
  yaw: number;
}

export interface NavigationPlanarRegion {
  minimum: Pick<Vec3, "x" | "z">;
  maximum: Pick<Vec3, "x" | "z">;
}

export interface NavigationBuildScope {
  region: NavigationPlanarRegion;
  walkableRegions?: NavigationPlanarRegion[];
  terrainSolids: NavigationSolid[];
}

interface NavigationSolid {
  center: Vec3;
  size: Vec3;
}

export interface NavigationAgentProfile {
  radius: number;
  height: number;
  maximumClimb: number;
  maximumSlopeDegrees: number;
  maximumTargetProjection: number;
}

let recastInitialization: Promise<void> | undefined;

const NAVIGATION_CELL_SIZE = 0.1;
const NAVIGATION_CELL_HEIGHT = 0.05;
const NAVIGATION_TILE_SIZE = 32;
const NAVIGATION_OBSTACLE_SKIN = NAVIGATION_CELL_SIZE;
const NAVIGATION_MAXIMUM_QUERY_NODES = 8_192;
const NAVIGATION_MAXIMUM_PATH_POLYGONS = 4_096;
const NAVIGATION_MAXIMUM_STRAIGHT_PATH_POINTS = 4_096;
const NAVIGATION_PATH_ENDPOINT_TOLERANCE = NAVIGATION_CELL_SIZE / 2;
export const NAVIGATION_TILE_WORLD_SIZE = NAVIGATION_CELL_SIZE * NAVIGATION_TILE_SIZE;

export class NavigationMesh {
  readonly #navMesh: RecastNavMesh;
  readonly #tileCache: RecastTileCache;
  readonly #query: RecastNavMeshQuery;
  readonly #profile: NavigationAgentProfile;
  readonly #walkableRegions: readonly NavigationPlanarRegion[];
  readonly #obstacles = new Map<string, {
    descriptor: NavigationObstacle;
    handle: RecastTileCacheObstacle;
  }>();

  static async create(
    scenario: Scenario,
    scope?: NavigationBuildScope,
    profile?: NavigationAgentProfile
  ): Promise<NavigationMesh> {
    if (!profile) throw new Error("Navigation agent profile is required");
    recastInitialization ??= recastApi.init();
    await recastInitialization;
    const solids = [...scenario.obstacles, ...(scope?.terrainSolids ?? [])];
    const region = scope?.region ?? {
      minimum: { x: 0, z: 0 },
      maximum: { x: scenario.bounds.width, z: scenario.bounds.depth }
    };
    const walkableRegions = validatedWalkableRegions(
      scope?.walkableRegions ?? [region],
      region
    );
    const geometry = worldGeometry(scenario, solids, walkableRegions);
    const maximumGeometryHeight = worldGeometryHeight(
      scenario,
      solids,
      walkableRegions
    );
    const navigationProfile = validatedProfile(profile);
    const generated = recastGeneratorsApi.generateTileCache(
      geometry.positions,
      geometry.indices,
      {
        cs: NAVIGATION_CELL_SIZE,
        ch: NAVIGATION_CELL_HEIGHT,
        tileSize: NAVIGATION_TILE_SIZE,
        walkableSlopeAngle: navigationProfile.maximumSlopeDegrees,
        walkableHeight: Math.ceil(navigationProfile.height / NAVIGATION_CELL_HEIGHT),
        walkableClimb: Math.floor(navigationProfile.maximumClimb / NAVIGATION_CELL_HEIGHT),
        walkableRadius: Math.ceil(navigationProfile.radius / NAVIGATION_CELL_SIZE),
        maxSimplificationError: 1.1,
        mergeRegionArea: 12,
        maxVertsPerPoly: 6,
        detailSampleDist: 4,
        detailSampleMaxError: 0.5,
        expectedLayersPerTile: 4,
        maxObstacles: Math.max(128, scenario.objects.length * 2),
        bounds: [
          [region.minimum.x, -0.1, region.minimum.z],
          [
            region.maximum.x,
            maximumGeometryHeight + navigationProfile.height,
            region.maximum.z
          ]
        ]
      });
    if (!generated.success) {
      throw new Error(`Navigation mesh generation failed: ${generated.error}`);
    }
    try {
      return new NavigationMesh(
        generated.navMesh,
        generated.tileCache,
        navigationProfile,
        walkableRegions
      );
    } catch (error) {
      generated.tileCache.destroy();
      generated.navMesh.destroy();
      throw error;
    }
  }

  private constructor(
    navMesh: RecastNavMesh,
    tileCache: RecastTileCache,
    profile: NavigationAgentProfile,
    walkableRegions: readonly NavigationPlanarRegion[]
  ) {
    this.#navMesh = navMesh;
    this.#tileCache = tileCache;
    this.#query = new recastApi.NavMeshQuery(navMesh, {
      maxNodes: NAVIGATION_MAXIMUM_QUERY_NODES
    });
    this.#profile = profile;
    this.#walkableRegions = walkableRegions.map((region) => structuredClone(region));
    this.#query.defaultQueryHalfExtents = {
      x: Math.max(0.8, profile.radius * 2),
      y: Math.max(2.2, profile.height * 1.5),
      z: Math.max(0.8, profile.radius * 2)
    };
  }

  plan(start: Vec3, target: Vec3, obstacles: readonly NavigationObstacle[]): NavigationPlan {
    this.#synchronizeObstacles(obstacles);
    const startResult = this.#query.findClosestPoint({ x: start.x, y: 0, z: start.z });
    const targetResult = this.#query.findClosestPoint({ x: target.x, y: 0, z: target.z });
    if (!startResult.success) {
      throw new NavigationPlanningError(
        "start_off_mesh",
        "Robot base is not on the navigation mesh"
      );
    }
    if (!targetResult.success) {
      throw new NavigationPlanningError(
        "target_off_mesh",
        `Navigation target has no walkable projection: requested=${formatPoint(target)}`
      );
    }
    const projectionDistance = planarDistance(targetResult.point, target);
    if (projectionDistance > this.#profile.maximumTargetProjection) {
      throw new NavigationPlanningError(
        "target_projection_exceeded",
        `Navigation target projection exceeds ${this.#profile.maximumTargetProjection.toFixed(2)}m: `
        + `requested=${formatPoint(target)}, projected=${formatPoint(targetResult.point)}, `
        + `distance=${projectionDistance.toFixed(3)}m`,
        {
          projectedTarget: {
            x: targetResult.point.x,
            y: start.y,
            z: targetResult.point.z
          },
          projectionDistance
        }
      );
    }
    const resolvedTarget = {
      x: targetResult.point.x,
      y: start.y,
      z: targetResult.point.z
    };
    if (planarDistance(startResult.point, targetResult.point) <= 0.015) {
      const waypoints = deduplicate([{ ...start }, resolvedTarget]);
      return {
        waypoints,
        distance: pathDistance(waypoints),
        resolvedTarget,
        projectionDistance
      };
    }
    const result = this.#query.computePath(startResult.point, targetResult.point, {
      maxPathPolys: NAVIGATION_MAXIMUM_PATH_POLYGONS,
      maxStraightPathPoints: NAVIGATION_MAXIMUM_STRAIGHT_PATH_POINTS
    });
    if (!result.success || result.path.length < 2) {
      throw new NavigationPlanningError(
        "path_not_found",
        `No navigation path: ${result.error?.name ?? "empty path"}`
      );
    }
    const intersectedObstacle = pathIntersectedObstacle(
      result.path,
      [...this.#obstacles.values()].map((entry) => entry.descriptor),
      this.#profile.radius
    );
    if (intersectedObstacle) {
      throw new NavigationPlanningError(
        "path_not_found",
        `No complete navigation path: computed path intersects obstacle ${intersectedObstacle}`
      );
    }
    const computedEndpoint = result.path.at(-1)!;
    const incompleteDistance = planarDistance(computedEndpoint, targetResult.point);
    if (incompleteDistance > NAVIGATION_PATH_ENDPOINT_TOLERANCE) {
      throw new NavigationPlanningError(
        "path_not_found",
        "No complete navigation path: "
        + `partial_endpoint=${formatPoint(computedEndpoint)}, `
        + `target=${formatPoint(targetResult.point)}, `
        + `gap=${incompleteDistance.toFixed(3)}m`
      );
    }
    const resolvedStart = {
      x: startResult.point.x,
      y: start.y,
      z: startResult.point.z
    };
    const waypoints = deduplicate([
      { x: start.x, y: start.y, z: start.z },
      resolvedStart,
      ...result.path.slice(1, -1).map((point) => ({ x: point.x, y: start.y, z: point.z })),
      resolvedTarget
    ]);
    return {
      waypoints,
      distance: pathDistance(waypoints),
      resolvedTarget,
      projectionDistance
    };
  }

  /**
   * Samples standoff poses on rings around a point and keeps those the navmesh
   * actually accepts. The navmesh is eroded by the base footprint, so a pose
   * close to an object is usually unreachable; returning the surviving ring
   * points lets a caller choose a real approach pose instead of guessing one.
   */
  reachableStandoffs(
    around: Vec3,
    radii: readonly number[],
    obstacles: readonly NavigationObstacle[],
    headings = 16
  ): StandoffPose[] {
    this.#synchronizeObstacles(obstacles);
    const poses: StandoffPose[] = [];
    for (const radius of radii) {
      if (!Number.isFinite(radius) || radius <= 0) continue;
      for (let index = 0; index < headings; index += 1) {
        const angle = (index / headings) * Math.PI * 2;
        const candidate = {
          x: around.x + Math.cos(angle) * radius,
          y: around.y,
          z: around.z + Math.sin(angle) * radius
        };
        const projected = this.#query.findClosestPoint({ x: candidate.x, y: 0, z: candidate.z });
        if (!projected.success) continue;
        if (planarDistance(projected.point, candidate)
          > this.#profile.maximumTargetProjection) continue;
        poses.push({
          position: { x: projected.point.x, y: around.y, z: projected.point.z },
          radius,
          distance: planarDistance(projected.point, around)
        });
      }
    }
    return poses.sort((left, right) => left.distance - right.distance);
  }

  /**
   * Keeps the candidates the base could actually be sent to, in the order
   * given, and reports where each one lands on the mesh.
   *
   * The mesh is eroded by the base footprint, so a point that is unambiguously
   * open floor on the terrain grid can still sit inside the eroded margin and
   * be refused. Offering such a point as a destination is worse than offering
   * nothing: it looks legal, and the refusal only arrives after a planning
   * call. Filtering here means a caller can only be handed places it can go.
   *
   * Candidates are projected in the order supplied and the walk stops at
   * `limit`, so the caller controls both preference and cost.
   */
  walkableProjections(
    candidates: readonly Vec3[],
    obstacles: readonly NavigationObstacle[],
    limit: number
  ): Array<{ requested: Vec3; point: Vec3 }> {
    this.#synchronizeObstacles(obstacles);
    const found: Array<{ requested: Vec3; point: Vec3 }> = [];
    for (const candidate of candidates) {
      if (found.length >= limit) break;
      const projected = this.#query.findClosestPoint({ x: candidate.x, y: 0, z: candidate.z });
      if (!projected.success) continue;
      if (planarDistance(projected.point, candidate)
        > this.#profile.maximumTargetProjection) continue;
      found.push({
        requested: candidate,
        point: { x: projected.point.x, y: candidate.y, z: projected.point.z }
      });
    }
    return found;
  }

  #synchronizeObstacles(obstacles: readonly NavigationObstacle[]): void {
    const requested = new Map<string, NavigationObstacle>();
    for (const obstacle of [...obstacles].sort((left, right) => left.id.localeCompare(right.id))) {
      const descriptor = validatedObstacle(obstacle);
      if (!navigationObstacleOverlapsRegions(descriptor, this.#walkableRegions)) continue;
      if (requested.has(descriptor.id)) {
        throw new Error(`Duplicate navigation obstacle id: ${descriptor.id}`);
      }
      requested.set(descriptor.id, descriptor);
    }

    for (const [id, tracked] of this.#obstacles) {
      const next = requested.get(id);
      if (next && sameObstacle(tracked.descriptor, next)) continue;
      this.#removeObstacle(id, tracked.handle);
    }
    this.#flushObstacleUpdates();

    for (const [id, descriptor] of requested) {
      if (this.#obstacles.has(id)) continue;
      const navigationHalfExtents = expandedObstacleHalfExtents(
        descriptor.halfExtents,
        this.#profile.radius
      );
      let added = this.#tileCache.addBoxObstacle(
        descriptor.center,
        navigationHalfExtents,
        descriptor.yaw
      );
      if (!added.success) {
        this.#flushObstacleUpdates();
        added = this.#tileCache.addBoxObstacle(
          descriptor.center,
          navigationHalfExtents,
          descriptor.yaw
        );
      }
      if (!added.success) {
        throw new Error(`Failed to add navigation obstacle ${id}: status=${added.status}`);
      }
      this.#obstacles.set(id, { descriptor, handle: added.obstacle });
    }
    this.#flushObstacleUpdates();
  }

  #removeObstacle(id: string, handle: RecastTileCacheObstacle): void {
    let removed = this.#tileCache.removeObstacle(handle);
    if (!removed.success) {
      this.#flushObstacleUpdates();
      removed = this.#tileCache.removeObstacle(handle);
    }
    if (!removed.success) {
      throw new Error(`Failed to remove navigation obstacle ${id}: status=${removed.status}`);
    }
    this.#obstacles.delete(id);
  }

  #flushObstacleUpdates(): void {
    while (true) {
      const update = this.#tileCache.update(this.#navMesh);
      if (!update.success) {
        throw new Error(`Failed to update navigation obstacles: status=${update.status}`);
      }
      if (update.upToDate) return;
    }
  }

  dispose(): void {
    this.#query.destroy();
    this.#tileCache.destroy();
    this.#navMesh.destroy();
  }
}

interface Geometry {
  positions: number[];
  indices: number[];
}

function worldGeometry(
  scenario: Scenario,
  solids: NavigationSolid[],
  regions: readonly NavigationPlanarRegion[]
): Geometry {
  const geometry: Geometry = { positions: [], indices: [] };
  for (const region of regions) {
    appendQuad(
      geometry,
      { x: region.minimum.x, y: 0, z: region.minimum.z },
      { x: region.minimum.x, y: 0, z: region.maximum.z },
      { x: region.maximum.x, y: 0, z: region.maximum.z },
      { x: region.maximum.x, y: 0, z: region.minimum.z }
    );
  }
  for (const obstacle of solids) {
    if (overlapsAnyRegion(obstacle, regions)) {
      appendBox(geometry, obstacle.center, obstacle.size);
    }
  }
  for (const object of scenario.objects) {
    if (!object.portable
      && overlapsAnyRegion({ center: object.position, size: object.size }, regions)) {
      appendBox(geometry, object.position, object.size);
    }
  }
  return deindexGeometry(geometry);
}

function deindexGeometry(geometry: Geometry): Geometry {
  const expanded: Geometry = { positions: [], indices: [] };
  for (const sourceIndex of geometry.indices) {
    const offset = sourceIndex * 3;
    const x = geometry.positions[offset];
    const y = geometry.positions[offset + 1];
    const z = geometry.positions[offset + 2];
    if (x === undefined || y === undefined || z === undefined) {
      throw new Error(`Navigation geometry references an unknown vertex: ${sourceIndex}`);
    }
    expanded.positions.push(x, y, z);
    expanded.indices.push(expanded.indices.length);
  }
  return expanded;
}

function worldGeometryHeight(
  scenario: Scenario,
  solids: NavigationSolid[],
  regions: readonly NavigationPlanarRegion[]
): number {
  let maximum = 0;
  for (const obstacle of solids) {
    if (overlapsAnyRegion(obstacle, regions)) {
      maximum = Math.max(maximum, obstacle.center.y + obstacle.size.y / 2);
    }
  }
  for (const object of scenario.objects) {
    if (!object.portable
      && overlapsAnyRegion({ center: object.position, size: object.size }, regions)) {
      maximum = Math.max(maximum, object.position.y + object.size.y / 2);
    }
  }
  return maximum;
}

function overlapsRegion(
  box: { center: Vec3; size: Vec3 },
  region: NavigationPlanarRegion
): boolean {
  return box.center.x + box.size.x / 2 >= region.minimum.x
    && box.center.x - box.size.x / 2 <= region.maximum.x
    && box.center.z + box.size.z / 2 >= region.minimum.z
    && box.center.z - box.size.z / 2 <= region.maximum.z;
}

function overlapsAnyRegion(
  box: { center: Vec3; size: Vec3 },
  regions: readonly NavigationPlanarRegion[]
): boolean {
  return regions.some((region) => overlapsRegion(box, region));
}

function validatedWalkableRegions(
  regions: readonly NavigationPlanarRegion[],
  bounds: NavigationPlanarRegion
): NavigationPlanarRegion[] {
  if (regions.length === 0) {
    throw new Error("Navigation build scope requires at least one walkable region");
  }
  const validate = (region: NavigationPlanarRegion, label: string): void => {
    const values = [
      region.minimum.x,
      region.minimum.z,
      region.maximum.x,
      region.maximum.z
    ];
    if (!values.every(Number.isFinite)
      || region.minimum.x >= region.maximum.x
      || region.minimum.z >= region.maximum.z) {
      throw new Error(`${label} must have finite positive planar extents`);
    }
  };
  validate(bounds, "Navigation build bounds");
  return regions.map((region, index) => {
    validate(region, `Navigation walkable region ${String(index)}`);
    if (region.minimum.x < bounds.minimum.x
      || region.minimum.z < bounds.minimum.z
      || region.maximum.x > bounds.maximum.x
      || region.maximum.z > bounds.maximum.z) {
      throw new Error(`Navigation walkable region ${String(index)} exceeds build bounds`);
    }
    return structuredClone(region);
  });
}

function navigationObstacleOverlapsRegions(
  obstacle: NavigationObstacle,
  regions: readonly NavigationPlanarRegion[]
): boolean {
  const cosine = Math.abs(Math.cos(obstacle.yaw));
  const sine = Math.abs(Math.sin(obstacle.yaw));
  return overlapsAnyRegion({
    center: obstacle.center,
    size: {
      x: 2 * (cosine * obstacle.halfExtents.x + sine * obstacle.halfExtents.z),
      y: obstacle.halfExtents.y * 2,
      z: 2 * (sine * obstacle.halfExtents.x + cosine * obstacle.halfExtents.z)
    }
  }, regions);
}

function appendBox(geometry: Geometry, center: Vec3, size: Vec3): void {
  const x0 = center.x - size.x / 2;
  const x1 = center.x + size.x / 2;
  const y0 = Math.max(0, center.y - size.y / 2);
  const y1 = center.y + size.y / 2;
  const z0 = center.z - size.z / 2;
  const z1 = center.z + size.z / 2;
  const vertices: Vec3[] = [
    { x: x0, y: y0, z: z0 },
    { x: x0, y: y0, z: z1 },
    { x: x1, y: y0, z: z1 },
    { x: x1, y: y0, z: z0 },
    { x: x0, y: y1, z: z0 },
    { x: x0, y: y1, z: z1 },
    { x: x1, y: y1, z: z1 },
    { x: x1, y: y1, z: z0 }
  ];
  const offset = geometry.positions.length / 3;
  for (const vertex of vertices) geometry.positions.push(vertex.x, vertex.y, vertex.z);
  geometry.indices.push(
    offset + 4, offset + 5, offset + 6, offset + 4, offset + 6, offset + 7,
    offset + 0, offset + 4, offset + 7, offset + 0, offset + 7, offset + 3,
    offset + 3, offset + 7, offset + 6, offset + 3, offset + 6, offset + 2,
    offset + 2, offset + 6, offset + 5, offset + 2, offset + 5, offset + 1,
    offset + 1, offset + 5, offset + 4, offset + 1, offset + 4, offset + 0
  );
}

function appendQuad(geometry: Geometry, a: Vec3, b: Vec3, c: Vec3, d: Vec3): void {
  const offset = geometry.positions.length / 3;
  for (const vertex of [a, b, c, d]) geometry.positions.push(vertex.x, vertex.y, vertex.z);
  geometry.indices.push(offset, offset + 1, offset + 2, offset, offset + 2, offset + 3);
}

function deduplicate(points: Vec3[]): Vec3[] {
  const result: Vec3[] = [];
  for (const [index, point] of points.entries()) {
    const previous = result.at(-1);
    if (!previous || planarDistance(previous, point) > 0.015) {
      result.push(point);
    } else if (index === points.length - 1) {
      result[result.length - 1] = point;
    }
  }
  return result;
}

function pathDistance(waypoints: readonly Vec3[]): number {
  return waypoints.slice(1).reduce(
    (total, waypoint, index) => total + planarDistance(waypoints[index]!, waypoint),
    0
  );
}

function validatedObstacle(obstacle: NavigationObstacle): NavigationObstacle {
  if (!obstacle.id.trim()) throw new Error("Navigation obstacle id cannot be empty");
  if (!finitePoint(obstacle.center)
    || !finitePoint(obstacle.halfExtents)
    || !Number.isFinite(obstacle.yaw)
    || obstacle.halfExtents.x <= 0
    || obstacle.halfExtents.y <= 0
    || obstacle.halfExtents.z <= 0) {
    throw new Error(`Invalid navigation obstacle: ${obstacle.id}`);
  }
  return {
    id: obstacle.id,
    center: { ...obstacle.center },
    halfExtents: { ...obstacle.halfExtents },
    yaw: normalizeAngle(obstacle.yaw)
  };
}

export function navigationObstaclePlanarExpansion(agentRadius: number): number {
  if (!Number.isFinite(agentRadius) || agentRadius < 0) {
    throw new Error("Navigation agent radius must be finite and non-negative");
  }
  return agentRadius + NAVIGATION_OBSTACLE_SKIN;
}

function expandedObstacleHalfExtents(halfExtents: Vec3, agentRadius: number): Vec3 {
  const planarExpansion = navigationObstaclePlanarExpansion(agentRadius);
  return {
    x: halfExtents.x + planarExpansion,
    y: halfExtents.y,
    z: halfExtents.z + planarExpansion
  };
}

function pathIntersectedObstacle(
  path: readonly Vec3[],
  obstacles: readonly NavigationObstacle[],
  agentRadius: number
): string | null {
  for (const obstacle of obstacles) {
    const halfExtents = expandedObstacleHalfExtents(
      obstacle.halfExtents,
      agentRadius
    );
    for (let index = 1; index < path.length; index += 1) {
      if (segmentIntersectsOrientedBox(
        path[index - 1]!,
        path[index]!,
        obstacle.center,
        halfExtents,
        obstacle.yaw
      )) return obstacle.id;
    }
  }
  return null;
}

function segmentIntersectsOrientedBox(
  start: Vec3,
  end: Vec3,
  center: Vec3,
  halfExtents: Vec3,
  yaw: number
): boolean {
  const cosine = Math.cos(yaw);
  const sine = Math.sin(yaw);
  const local = (point: Vec3): readonly [number, number] => {
    const dx = point.x - center.x;
    const dz = point.z - center.z;
    return [cosine * dx + sine * dz, -sine * dx + cosine * dz];
  };
  const [startX, startZ] = local(start);
  const [endX, endZ] = local(end);
  let minimumTime = 0;
  let maximumTime = 1;
  for (const [from, to, extent] of [
    [startX, endX, halfExtents.x],
    [startZ, endZ, halfExtents.z]
  ] as const) {
    const delta = to - from;
    if (Math.abs(delta) <= 1e-12) {
      if (Math.abs(from) > extent) return false;
      continue;
    }
    const first = (-extent - from) / delta;
    const second = (extent - from) / delta;
    minimumTime = Math.max(minimumTime, Math.min(first, second));
    maximumTime = Math.min(maximumTime, Math.max(first, second));
    if (minimumTime > maximumTime) return false;
  }
  return maximumTime >= 0 && minimumTime <= 1;
}

function validatedProfile(profile: NavigationAgentProfile): NavigationAgentProfile {
  const values = [
    profile.radius,
    profile.height,
    profile.maximumClimb,
    profile.maximumSlopeDegrees,
    profile.maximumTargetProjection
  ];
  if (values.some((value) => !Number.isFinite(value))
    || profile.radius <= 0
    || profile.height <= 0
    || profile.maximumClimb < 0
    || profile.maximumSlopeDegrees <= 0
    || profile.maximumSlopeDegrees > 90
    || profile.maximumTargetProjection < 0) {
    throw new Error("Invalid navigation agent profile");
  }
  return { ...profile };
}

function sameObstacle(left: NavigationObstacle, right: NavigationObstacle): boolean {
  return samePoint(left.center, right.center)
    && samePoint(left.halfExtents, right.halfExtents)
    && Math.abs(normalizeAngle(left.yaw - right.yaw)) <= 1e-4;
}

function samePoint(left: Vec3, right: Vec3): boolean {
  return Math.abs(left.x - right.x) <= 1e-4
    && Math.abs(left.y - right.y) <= 1e-4
    && Math.abs(left.z - right.z) <= 1e-4;
}

function finitePoint(point: Vec3): boolean {
  return Number.isFinite(point.x) && Number.isFinite(point.y) && Number.isFinite(point.z);
}

function normalizeAngle(value: number): number {
  return Math.atan2(Math.sin(value), Math.cos(value));
}

function formatPoint(point: Vec3): string {
  return `(${point.x.toFixed(3)},${point.y.toFixed(3)},${point.z.toFixed(3)})`;
}

function planarDistance(left: Pick<Vec3, "x" | "z">, right: Pick<Vec3, "x" | "z">): number {
  return Math.hypot(left.x - right.x, left.z - right.z);
}
