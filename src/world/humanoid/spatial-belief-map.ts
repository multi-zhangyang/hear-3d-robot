import { z } from "zod";
import type { Quaternion, Scenario, Vec3 } from "../../domain/schema.js";
import { rotateVector } from "../geometry.js";

const HUMANOID_SPATIAL_MAP_RESOLUTION_METERS = 0.5;
const FRONTIER_LIMIT = 24;
const INFORMATION_RADIUS_CELLS = 3;

const SpatialCellSchema = z.object({
  x: z.number().int().nonnegative(),
  z: z.number().int().nonnegative(),
  first_observed_frame: z.number().int().nonnegative(),
  last_observed_frame: z.number().int().nonnegative(),
  visit_count: z.number().int().nonnegative(),
  occupied: z.boolean()
}).strict();

export const HumanoidSpatialBeliefMapCheckpointSchema = z.object({
  version: z.literal(1),
  resolution_m: z.literal(HUMANOID_SPATIAL_MAP_RESOLUTION_METERS),
  last_updated_frame: z.number().int().min(-1),
  cells: z.array(SpatialCellSchema)
}).strict();

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
  protocol: "humanoid-spatial-belief-v1";
  resolution_m: number;
  observed_cell_count: number;
  free_cell_count: number;
  occupied_cell_count: number;
  visited_cell_count: number;
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
    for (const cell of parsed.cells) {
      if (!this.#inside(cell.x, cell.z)) continue;
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
    };
    visibleSolids: readonly { center: Vec3; size: Vec3 }[];
  }): void {
    if (input.frame < this.#lastUpdatedFrame) {
      throw new Error("Spatial belief observation cannot move backwards in time");
    }
    if (input.frame === this.#lastUpdatedFrame) return;
    const forward = rotateVector(input.sensor.rotation, { x: 0, y: 0, z: 1 });
    const sensorYaw = Math.atan2(forward.x, forward.z);
    for (let z = 0; z < this.#rows; z += 1) {
      for (let x = 0; x < this.#columns; x += 1) {
        const center = this.#center(x, z);
        const dx = center.x - input.sensor.position.x;
        const dz = center.z - input.sensor.position.z;
        const distance = Math.hypot(dx, dz);
        if (distance > input.sensor.maximumRange) continue;
        const bearing = Math.atan2(dx, dz);
        if (Math.abs(wrapRadians(bearing - sensorYaw))
          > input.sensor.horizontalFieldOfView / 2) continue;
        this.#markObserved(x, z, input.frame, false);
      }
    }
    for (const solid of input.visibleSolids) {
      this.#markSolid(solid, input.frame);
    }
    const root = this.#index(input.rootPosition);
    if (root) {
      const cell = this.#markObserved(root.x, root.z, input.frame, false);
      cell.visitCount += 1;
    }
    this.#lastUpdatedFrame = input.frame;
  }

  observation(rootPosition: Vec3): HumanoidSpatialBeliefObservation {
    const cells = [...this.#cells.values()];
    const free = cells.filter((cell) => !cell.occupied);
    const occupied = cells.length - free.length;
    const total = this.#columns * this.#rows;
    return {
      protocol: "humanoid-spatial-belief-v1",
      resolution_m: HUMANOID_SPATIAL_MAP_RESOLUTION_METERS,
      observed_cell_count: cells.length,
      free_cell_count: free.length,
      occupied_cell_count: occupied,
      visited_cell_count: cells.filter((cell) => cell.visitCount > 0).length,
      total_cell_count: total,
      coverage_ratio: total === 0 ? 0 : cells.length / total,
      frontiers: this.#frontiers(rootPosition)
    };
  }

  checkpoint(): HumanoidSpatialBeliefMapCheckpoint {
    return HumanoidSpatialBeliefMapCheckpointSchema.parse({
      version: 1,
      resolution_m: HUMANOID_SPATIAL_MAP_RESOLUTION_METERS,
      last_updated_frame: this.#lastUpdatedFrame,
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

  #frontiers(rootPosition: Vec3): HumanoidExplorationFrontier[] {
    return [...this.#cells.values()]
      .filter((cell) => !cell.occupied && this.#unknownNeighborCount(cell.x, cell.z) > 0)
      .map((cell): HumanoidExplorationFrontier => {
        const target = this.#center(cell.x, cell.z);
        const gain = this.#unknownInformationGain(cell.x, cell.z);
        const travel = Math.hypot(
          target.x - rootPosition.x,
          target.z - rootPosition.z
        );
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
        || left.id.localeCompare(right.id))
      .slice(0, FRONTIER_LIMIT);
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
    const maximumX = Math.floor(
      (solid.center.x + solid.size.x / 2) / HUMANOID_SPATIAL_MAP_RESOLUTION_METERS
    );
    const minimumZ = Math.floor(
      (solid.center.z - solid.size.z / 2) / HUMANOID_SPATIAL_MAP_RESOLUTION_METERS
    );
    const maximumZ = Math.floor(
      (solid.center.z + solid.size.z / 2) / HUMANOID_SPATIAL_MAP_RESOLUTION_METERS
    );
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
      existing.occupied ||= occupied;
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

function wrapRadians(value: number): number {
  let wrapped = value;
  while (wrapped > Math.PI) wrapped -= Math.PI * 2;
  while (wrapped < -Math.PI) wrapped += Math.PI * 2;
  return wrapped;
}
