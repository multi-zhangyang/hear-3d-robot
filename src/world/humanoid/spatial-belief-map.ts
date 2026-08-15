import { z } from "zod";
import type { Quaternion, Scenario, Vec3 } from "../../domain/schema.js";
import {
  inverseQuaternion,
  rotateVector,
  subtract,
  vectorLength
} from "../geometry.js";

const HUMANOID_SPATIAL_MAP_RESOLUTION_METERS = 0.5;
const FRONTIER_LIMIT = 24;
const INFORMATION_RADIUS_CELLS = 3;
const FRONTIER_SEPARATION_CELLS = 3;
const GROUND_OBSERVATION_HEIGHT_METERS = 0.04;
const RAY_TERMINAL_EPSILON = 1e-6;

const SpatialCellSchema = z.object({
  x: z.number().int().nonnegative(),
  z: z.number().int().nonnegative(),
  first_observed_frame: z.number().int().nonnegative(),
  last_observed_frame: z.number().int().nonnegative(),
  visit_count: z.number().int().nonnegative(),
  occupied: z.boolean()
}).strict().superRefine((cell, context) => {
  if (cell.first_observed_frame > cell.last_observed_frame) {
    context.addIssue({
      code: "custom",
      path: ["last_observed_frame"],
      message: "Spatial cell observation time cannot move backwards"
    });
  }
});
const SpatialCellIdentitySchema = z.object({
  x: z.number().int().nonnegative(),
  z: z.number().int().nonnegative()
}).strict();

export const HumanoidSpatialBeliefMapCheckpointSchema = z.object({
  version: z.literal(1),
  resolution_m: z.literal(HUMANOID_SPATIAL_MAP_RESOLUTION_METERS),
  last_updated_frame: z.number().int().min(-1),
  last_sensor_observed_frame: z.number().int().min(-1).optional(),
  last_visited_cell: SpatialCellIdentitySchema.nullable().optional(),
  cells: z.array(SpatialCellSchema)
}).strict().superRefine((checkpoint, context) => {
  if ((checkpoint.last_sensor_observed_frame ?? checkpoint.last_updated_frame)
    > checkpoint.last_updated_frame) {
    context.addIssue({
      code: "custom",
      path: ["last_sensor_observed_frame"],
      message: "Spatial sensor observation head exceeds the map update head"
    });
  }
  const identities = new Set<string>();
  checkpoint.cells.forEach((cell, index) => {
    const identity = key(cell.x, cell.z);
    if (identities.has(identity)) {
      context.addIssue({
        code: "custom",
        path: ["cells", index],
        message: `Spatial belief checkpoint repeats cell ${identity}`
      });
    }
    identities.add(identity);
    if (cell.last_observed_frame > checkpoint.last_updated_frame) {
      context.addIssue({
        code: "custom",
        path: ["cells", index, "last_observed_frame"],
        message: "Spatial cell is newer than the checkpoint observation head"
      });
    }
  });
});

export type HumanoidSpatialBeliefMapCheckpoint = z.infer<
  typeof HumanoidSpatialBeliefMapCheckpointSchema
>;

export interface HumanoidExplorationFrontier {
  id: string;
  target: Vec3;
  expected_information_gain: number;
  travel_distance_m: number;
  revisit_penalty: number;
  score: number;
}

export interface HumanoidSpatialBeliefObservation {
  protocol: "humanoid-spatial-belief-v2";
  visibility_model: "occlusion_aware_head_camera";
  frontier_model: "reachable_geodesic_diversity";
  resolution_m: number;
  observed_cell_count: number;
  free_cell_count: number;
  occupied_cell_count: number;
  visited_cell_count: number;
  reachable_free_cell_count: number;
  total_cell_count: number;
  coverage_ratio: number;
  frontiers: HumanoidExplorationFrontier[];
}

interface SpatialCell {
  x: number;
  z: number;
  firstObservedFrame: number;
  lastObservedFrame: number;
  visitCount: number;
  occupied: boolean;
}

export class HumanoidSpatialBeliefMap {
  readonly #columns: number;
  readonly #rows: number;
  readonly #cells = new Map<string, SpatialCell>();
  #lastUpdatedFrame = -1;
  #lastSensorObservedFrame = -1;
  #lastVisitedCell: { x: number; z: number } | null = null;

  constructor(
    private readonly scenario: Pick<Scenario, "bounds">,
    checkpoint?: HumanoidSpatialBeliefMapCheckpoint
  ) {
    this.#columns = Math.ceil(
      scenario.bounds.width / HUMANOID_SPATIAL_MAP_RESOLUTION_METERS
    );
    this.#rows = Math.ceil(
      scenario.bounds.depth / HUMANOID_SPATIAL_MAP_RESOLUTION_METERS
    );
    if (!checkpoint) return;
    const parsed = HumanoidSpatialBeliefMapCheckpointSchema.parse(checkpoint);
    this.#lastUpdatedFrame = parsed.last_updated_frame;
    this.#lastSensorObservedFrame = parsed.last_sensor_observed_frame
      ?? parsed.last_updated_frame;
    this.#lastVisitedCell = parsed.last_visited_cell
      ? { ...parsed.last_visited_cell }
      : null;
    for (const cell of parsed.cells) {
      if (!this.#inside(cell.x, cell.z)) {
        throw new Error(`Spatial belief cell is outside the world bounds: ${key(cell.x, cell.z)}`);
      }
      this.#cells.set(key(cell.x, cell.z), {
        x: cell.x,
        z: cell.z,
        firstObservedFrame: cell.first_observed_frame,
        lastObservedFrame: cell.last_observed_frame,
        visitCount: cell.visit_count,
        occupied: cell.occupied
      });
    }
  }

  observe(input: {
    frame: number;
    rootPosition: Vec3;
    sensor: {
      position: Vec3;
      rotation: Quaternion;
      maximumRange: number;
      horizontalFieldOfView: number;
      verticalFieldOfView: number;
    };
    visibleSolids: readonly {
      center: Vec3;
      size: Vec3;
      occupiesNavigationSpace?: boolean;
    }[];
    pointVisibility: (points: readonly Vec3[]) => readonly boolean[];
  }): void {
    if (input.frame < this.#lastUpdatedFrame) {
      throw new Error("Spatial belief observation cannot move backwards in time");
    }
    if (input.frame === this.#lastSensorObservedFrame) return;
    const inverseSensorRotation = inverseQuaternion(input.sensor.rotation);
    const candidates: Array<{ x: number; z: number; point: Vec3 }> = [];
    for (let z = 0; z < this.#rows; z += 1) {
      for (let x = 0; x < this.#columns; x += 1) {
        const center = {
          ...this.#center(x, z),
          y: GROUND_OBSERVATION_HEIGHT_METERS
        };
        if (!insideSensorFrustum(center, input.sensor, inverseSensorRotation)) continue;
        if (input.visibleSolids.some((solid) => pointInsideSolid(center, solid))) continue;
        candidates.push({ x, z, point: center });
      }
    }
    const visible = input.pointVisibility(candidates.map(({ point }) => point));
    if (visible.length !== candidates.length) {
      throw new Error("Spatial visibility result does not match its requested ground points");
    }
    candidates.forEach((candidate, index) => {
      if (visible[index] === true) {
        this.#markObserved(candidate.x, candidate.z, input.frame, false);
      }
    });
    for (const solid of input.visibleSolids.filter((candidate) => (
      candidate.occupiesNavigationSpace !== false
    ))) {
      this.#markSolid(solid, input.frame);
    }
    this.recordTraversal(input.frame, input.rootPosition);
    this.#lastSensorObservedFrame = input.frame;
  }

  recordTraversal(frame: number, rootPosition: Vec3): void {
    if (frame < this.#lastUpdatedFrame) {
      throw new Error("Spatial belief traversal cannot move backwards in time");
    }
    const root = this.#index(rootPosition);
    if (root) {
      const cell = this.#markObserved(root.x, root.z, frame, false);
      if (!this.#lastVisitedCell
        || root.x !== this.#lastVisitedCell.x
        || root.z !== this.#lastVisitedCell.z) {
        cell.visitCount += 1;
        this.#lastVisitedCell = { ...root };
      }
    }
    this.#lastUpdatedFrame = Math.max(this.#lastUpdatedFrame, frame);
  }

  observation(rootPosition: Vec3): HumanoidSpatialBeliefObservation {
    const cells = [...this.#cells.values()];
    const free = cells.filter((cell) => !cell.occupied);
    const occupied = cells.length - free.length;
    const total = this.#columns * this.#rows;
    const reachableDistances = this.#reachableDistances(rootPosition);
    return {
      protocol: "humanoid-spatial-belief-v2",
      visibility_model: "occlusion_aware_head_camera",
      frontier_model: "reachable_geodesic_diversity",
      resolution_m: HUMANOID_SPATIAL_MAP_RESOLUTION_METERS,
      observed_cell_count: cells.length,
      free_cell_count: free.length,
      occupied_cell_count: occupied,
      visited_cell_count: cells.filter((cell) => cell.visitCount > 0).length,
      reachable_free_cell_count: reachableDistances.size,
      total_cell_count: total,
      coverage_ratio: total === 0 ? 0 : cells.length / total,
      frontiers: this.#frontiers(reachableDistances)
    };
  }

  checkpoint(): HumanoidSpatialBeliefMapCheckpoint {
    return HumanoidSpatialBeliefMapCheckpointSchema.parse({
      version: 1,
      resolution_m: HUMANOID_SPATIAL_MAP_RESOLUTION_METERS,
      last_updated_frame: this.#lastUpdatedFrame,
      last_sensor_observed_frame: this.#lastSensorObservedFrame,
      last_visited_cell: this.#lastVisitedCell
        ? { ...this.#lastVisitedCell }
        : null,
      cells: [...this.#cells.values()]
        .sort((left, right) => left.z - right.z || left.x - right.x)
        .map((cell) => ({
          x: cell.x,
          z: cell.z,
          first_observed_frame: cell.firstObservedFrame,
          last_observed_frame: cell.lastObservedFrame,
          visit_count: cell.visitCount,
          occupied: cell.occupied
        }))
    });
  }

  #frontiers(reachableDistances: ReadonlyMap<string, number>): HumanoidExplorationFrontier[] {
    const ranked = [...this.#cells.values()]
      .filter((cell) => !cell.occupied
        && (reachableDistances.get(key(cell.x, cell.z)) ?? -1)
          >= HUMANOID_SPATIAL_MAP_RESOLUTION_METERS
        && this.#unknownNeighborCount(cell.x, cell.z) > 0)
      .map((cell): HumanoidExplorationFrontier => {
        const target = this.#center(cell.x, cell.z);
        const gain = this.#unknownInformationGain(cell.x, cell.z);
        const travel = reachableDistances.get(key(cell.x, cell.z))!;
        const revisit = Math.log2(cell.visitCount + 1);
        return {
          id: `frontier:${cell.x}:${cell.z}`,
          target,
          expected_information_gain: gain,
          travel_distance_m: travel,
          revisit_penalty: revisit,
          score: gain / (1 + travel) - revisit
        };
      })
      .filter((frontier) => frontier.expected_information_gain > 0)
      .sort((left, right) => right.score - left.score
        || right.expected_information_gain - left.expected_information_gain
        || left.travel_distance_m - right.travel_distance_m
        || left.id.localeCompare(right.id));
    const selected: HumanoidExplorationFrontier[] = [];
    const minimumSeparation = FRONTIER_SEPARATION_CELLS
      * HUMANOID_SPATIAL_MAP_RESOLUTION_METERS;
    for (const frontier of ranked) {
      if (selected.some((existing) => Math.hypot(
        existing.target.x - frontier.target.x,
        existing.target.z - frontier.target.z
      ) < minimumSeparation)) continue;
      selected.push(frontier);
      if (selected.length >= FRONTIER_LIMIT) break;
    }
    return selected;
  }

  #reachableDistances(rootPosition: Vec3): Map<string, number> {
    const root = this.#index(rootPosition);
    if (!root) return new Map();
    const rootCell = this.#cells.get(key(root.x, root.z));
    if (!rootCell || rootCell.occupied) return new Map();
    const distances = new Map<string, number>([[key(root.x, root.z), 0]]);
    const queue = [root];
    for (let index = 0; index < queue.length; index += 1) {
      const current = queue[index]!;
      const currentDistance = distances.get(key(current.x, current.z))!;
      for (const [dx, dz] of NEIGHBORS) {
        const next = { x: current.x + dx, z: current.z + dz };
        const identity = key(next.x, next.z);
        if (distances.has(identity)) continue;
        const cell = this.#cells.get(identity);
        if (!cell || cell.occupied) continue;
        distances.set(
          identity,
          currentDistance + HUMANOID_SPATIAL_MAP_RESOLUTION_METERS
        );
        queue.push(next);
      }
    }
    return distances;
  }

  #unknownNeighborCount(x: number, z: number): number {
    return NEIGHBORS.reduce((count, [dx, dz]) => {
      const nx = x + dx;
      const nz = z + dz;
      return count + (this.#inside(nx, nz) && !this.#cells.has(key(nx, nz)) ? 1 : 0);
    }, 0);
  }

  #unknownInformationGain(x: number, z: number): number {
    let gain = 0;
    for (let dz = -INFORMATION_RADIUS_CELLS; dz <= INFORMATION_RADIUS_CELLS; dz += 1) {
      for (let dx = -INFORMATION_RADIUS_CELLS; dx <= INFORMATION_RADIUS_CELLS; dx += 1) {
        if (dx * dx + dz * dz > INFORMATION_RADIUS_CELLS ** 2) continue;
        const nx = x + dx;
        const nz = z + dz;
        if (this.#inside(nx, nz) && !this.#cells.has(key(nx, nz))) gain += 1;
      }
    }
    return gain;
  }

  #markSolid(solid: { center: Vec3; size: Vec3 }, frame: number): void {
    const minimumX = Math.floor(
      (solid.center.x - solid.size.x / 2) / HUMANOID_SPATIAL_MAP_RESOLUTION_METERS
    );
    const maximumX = Math.ceil(
      (solid.center.x + solid.size.x / 2) / HUMANOID_SPATIAL_MAP_RESOLUTION_METERS
    ) - 1;
    const minimumZ = Math.floor(
      (solid.center.z - solid.size.z / 2) / HUMANOID_SPATIAL_MAP_RESOLUTION_METERS
    );
    const maximumZ = Math.ceil(
      (solid.center.z + solid.size.z / 2) / HUMANOID_SPATIAL_MAP_RESOLUTION_METERS
    ) - 1;
    for (let z = minimumZ; z <= maximumZ; z += 1) {
      for (let x = minimumX; x <= maximumX; x += 1) {
        if (this.#inside(x, z)) this.#markObserved(x, z, frame, true);
      }
    }
  }

  #markObserved(x: number, z: number, frame: number, occupied: boolean): SpatialCell {
    const identity = key(x, z);
    const existing = this.#cells.get(identity);
    if (existing) {
      existing.lastObservedFrame = Math.max(existing.lastObservedFrame, frame);
      existing.occupied = occupied;
      return existing;
    }
    const cell: SpatialCell = {
      x,
      z,
      firstObservedFrame: frame,
      lastObservedFrame: frame,
      visitCount: 0,
      occupied
    };
    this.#cells.set(identity, cell);
    return cell;
  }

  #center(x: number, z: number): Vec3 {
    return {
      x: Math.min(
        this.scenario.bounds.width,
        (x + 0.5) * HUMANOID_SPATIAL_MAP_RESOLUTION_METERS
      ),
      y: 0,
      z: Math.min(
        this.scenario.bounds.depth,
        (z + 0.5) * HUMANOID_SPATIAL_MAP_RESOLUTION_METERS
      )
    };
  }

  #index(position: Vec3): { x: number; z: number } | null {
    const x = Math.floor(position.x / HUMANOID_SPATIAL_MAP_RESOLUTION_METERS);
    const z = Math.floor(position.z / HUMANOID_SPATIAL_MAP_RESOLUTION_METERS);
    return this.#inside(x, z) ? { x, z } : null;
  }

  #inside(x: number, z: number): boolean {
    return x >= 0 && z >= 0 && x < this.#columns && z < this.#rows;
  }
}

const NEIGHBORS = [
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1]
] as const;

function key(x: number, z: number): string {
  return `${x}:${z}`;
}

function insideSensorFrustum(
  point: Vec3,
  sensor: {
    position: Vec3;
    maximumRange: number;
    horizontalFieldOfView: number;
    verticalFieldOfView: number;
  },
  inverseSensorRotation: Quaternion
): boolean {
  const delta = subtract(point, sensor.position);
  const distance = vectorLength(delta);
  if (distance <= RAY_TERMINAL_EPSILON || distance > sensor.maximumRange) return false;
  const local = rotateVector(inverseSensorRotation, {
    x: delta.x / distance,
    y: delta.y / distance,
    z: delta.z / distance
  });
  const horizontal = Math.atan2(local.x, local.z);
  const vertical = Math.atan2(local.y, Math.hypot(local.x, local.z));
  return local.z > 0
    && Math.abs(horizontal) <= sensor.horizontalFieldOfView / 2
    && Math.abs(vertical) <= sensor.verticalFieldOfView / 2;
}

function pointInsideSolid(
  point: Vec3,
  solid: { center: Vec3; size: Vec3 }
): boolean {
  return Math.abs(point.x - solid.center.x) <= solid.size.x / 2
    && Math.abs(point.y - solid.center.y) <= solid.size.y / 2
    && Math.abs(point.z - solid.center.z) <= solid.size.z / 2;
}
