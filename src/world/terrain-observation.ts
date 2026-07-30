import type { Terrain, Vec3 } from "../domain/schema.js";
import { inverseQuaternion, rotateVector, scale, subtract, vectorLength } from "./geometry.js";
import { buildFrontierSample, pointKey } from "./frontier-sampler.js";
import { cellAt, cellCenter, terrainHeight } from "./terrain.js";

interface QuaternionLike {
  x: number;
  y: number;
  z: number;
  w: number;
}

export interface WalkableProjection {
  requested: Vec3;
  point: Vec3;
}

export interface TerrainSurveyInput {
  terrain: Terrain;
  robotPosition: Vec3;
  radiusCells: number;
  isExplored: (index: number) => boolean;
  exploredCount: number;
  exploredTotal: number;
  motionSeed: number;
  worldRevision: number;
  robotYaw: number;
  projectWalkable: (candidates: readonly Vec3[]) => WalkableProjection[];
}

export interface TerrainFrontierChoice {
  choice_id: string;
  target: Vec3;
  face_point: Vec3;
  travel_distance: number;
  unseen_neighbours: number;
  turn_degrees: number;
  motion_style: "stride" | "turn" | "probe";
}

export interface TerrainSurvey {
  cell_size: number;
  block_height: number;
  grid: { columns: number; rows: number };
  robot_cell: { column: number; row: number };
  origin_cell: { column: number; row: number };
  legend: string;
  rows: string[];
  exploration: { cells_seen: number; cells_total: number };
  movement_sampling: {
    sample_id: string;
    strategy: "entropy_ordered_reachable_frontiers";
    decision_owner: "model";
    automatic_actuation: false;
    choice_count: number;
    fallback: string;
  };
  frontier: TerrainFrontierChoice[];
}

/**
 * Builds the model-facing terrain memory without owning either navigation or
 * actuation. Recast remains the authority for reachability through the injected
 * projection, and the returned choices still require an explicit model plan and
 * execution call.
 */
export function buildTerrainSurvey(input: TerrainSurveyInput): TerrainSurvey {
  const here = cellAt(input.terrain, input.robotPosition);
  const reach = Math.max(1, Math.min(24, Math.trunc(input.radiusCells)));
  const rows: string[] = [];
  for (let row = here.row - reach; row <= here.row + reach; row += 1) {
    let line = "";
    for (let column = here.column - reach; column <= here.column + reach; column += 1) {
      if (column < 0 || row < 0
        || column >= input.terrain.columns || row >= input.terrain.rows) {
        line += "#";
        continue;
      }
      const index = row * input.terrain.columns + column;
      if (!input.isExplored(index)) {
        line += "?";
        continue;
      }
      const height = input.terrain.heights[index] ?? 0;
      line += height === 0 ? "." : String(Math.min(9, height));
    }
    rows.push(line);
  }

  const sample = buildFrontierSample({
    terrain: input.terrain,
    here,
    isExplored: input.isExplored,
    motionSeed: input.motionSeed,
    worldRevision: input.worldRevision,
    exploredCount: input.exploredCount,
    robotYaw: input.robotYaw
  });
  const byPoint = new Map(sample.candidates.map((candidate) => [candidate.key, candidate]));
  const reachable = input.projectWalkable(
    sample.candidates.map((candidate) => candidate.target)
  );
  const frontier = reachable.map((entry): TerrainFrontierChoice => {
    const candidate = byPoint.get(pointKey(entry.requested));
    if (!candidate) {
      throw new Error("Navigation returned a projection for an unknown frontier candidate");
    }
    return {
      choice_id: candidate.choiceId,
      target: entry.point,
      face_point: candidate.facePoint,
      travel_distance: Number(candidate.distance.toFixed(3)),
      unseen_neighbours: candidate.unseen,
      turn_degrees: Number(candidate.turnDegrees.toFixed(1)),
      motion_style: candidate.distance >= 8
        ? "stride"
        : candidate.turnDegrees >= 75 ? "turn" : "probe"
    };
  });

  return {
    cell_size: input.terrain.cell,
    block_height: input.terrain.block,
    grid: { columns: input.terrain.columns, rows: input.terrain.rows },
    robot_cell: here,
    origin_cell: { column: here.column - reach, row: here.row - reach },
    legend: "'.' walkable floor, '1'-'9' solid column of that many blocks, "
      + "'?' not yet seen, '#' outside the world",
    rows,
    exploration: {
      cells_seen: input.exploredCount,
      cells_total: input.exploredTotal
    },
    movement_sampling: {
      sample_id: sample.sampleId,
      strategy: "entropy_ordered_reachable_frontiers",
      decision_owner: "model",
      automatic_actuation: false,
      choice_count: frontier.length,
      fallback: "On a rejection, the model must re-observe or choose another returned frontier; "
        + "the harness never moves the base itself."
    },
    frontier
  };
}

export interface VisibleTerrainCellsInput {
  terrain: Terrain;
  sensorPosition: Vec3;
  sensorRotation: QuaternionLike;
  maximumRange: number;
  horizontalFieldOfView: number;
  verticalFieldOfView: number;
  isExplored: (index: number) => boolean;
  heightAt?: (column: number, row: number) => number;
  isOccluded: (direction: Vec3, maximumDistance: number) => boolean;
}

/**
 * Returns unseen columns that are inside the measured sensor cone and have a
 * clear ray. The caller supplies the real physics ray query and commits the
 * returned indices to exploration memory.
 */
export function visibleTerrainCells(input: VisibleTerrainCellsInput): number[] {
  const reach = Math.ceil(input.maximumRange / input.terrain.cell);
  const here = cellAt(input.terrain, input.sensorPosition);
  const visible: number[] = [];
  for (let row = here.row - reach; row <= here.row + reach; row += 1) {
    for (let column = here.column - reach; column <= here.column + reach; column += 1) {
      if (column < 0 || row < 0
        || column >= input.terrain.columns || row >= input.terrain.rows) continue;
      const index = row * input.terrain.columns + column;
      if (input.isExplored(index)) continue;
      const centre = cellCenter(input.terrain, { column, row });
      const top = (input.heightAt?.(column, row)
        ?? terrainHeight(input.terrain, { column, row })) * input.terrain.block;
      const surface = {
        x: centre.x,
        y: Math.min(top, input.sensorPosition.y),
        z: centre.z
      };
      const delta = subtract(surface, input.sensorPosition);
      const distance = vectorLength(delta);
      if (distance > input.maximumRange) continue;
      const direction = scale(delta, 1 / Math.max(distance, 1e-6));
      const local = rotateVector(inverseQuaternion(input.sensorRotation), direction);
      if (Math.abs(Math.atan2(local.x, local.z)) > input.horizontalFieldOfView / 2
        || Math.abs(Math.atan2(local.y, Math.hypot(local.x, local.z)))
          > input.verticalFieldOfView / 2) continue;
      if (!input.isOccluded(direction, Math.max(0, distance - input.terrain.cell * 0.5))) {
        visible.push(index);
      }
    }
  }
  return visible;
}
