import type { Terrain, Vec3 } from "../domain/schema.js";
import { createRandom, deriveSeed, shuffle } from "./random.js";
import { cellCenter } from "./terrain.js";

const CARDINAL_NEIGHBOURS: ReadonlyArray<readonly [number, number]> = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1]
];

export interface FrontierCandidate {
  key: string;
  choiceId: string;
  target: Vec3;
  facePoint: Vec3;
  distance: number;
  unseen: number;
  turnDegrees: number;
}

export interface FrontierSample {
  sampleId: string;
  candidates: FrontierCandidate[];
}

/**
 * Builds model-facing movement choices from current sensor memory.
 *
 * Entropy only orders valid candidates. It never selects or executes one: the
 * model still has to cite a target and face point in a planning call, then
 * explicitly execute the accepted plan. High-information cells stay ahead of
 * low-information ones, while equally useful directions are shuffled from an
 * independently random per-run motion seed. This prevents a model that often
 * reads the first option from tracing the same nearest-cell pattern every run.
 */
export function buildFrontierSample(input: {
  terrain: Terrain;
  here: { column: number; row: number };
  isExplored: (index: number) => boolean;
  motionSeed: number;
  worldRevision: number;
  exploredCount: number;
  robotYaw: number;
}): FrontierSample {
  const sampleSeed = deriveSeed(
    input.motionSeed,
    `frontier:${input.worldRevision}:${input.exploredCount}:${input.here.column}:${input.here.row}`
  );
  const random = createRandom(sampleSeed);
  const candidates: FrontierCandidate[] = [];

  for (let row = 0; row < input.terrain.rows; row += 1) {
    for (let column = 0; column < input.terrain.columns; column += 1) {
      const index = row * input.terrain.columns + column;
      if (!input.isExplored(index) || input.terrain.heights[index] !== 0) continue;

      const unseenCells: Vec3[] = [];
      for (const [dx, dz] of CARDINAL_NEIGHBOURS) {
        const nextColumn = column + dx;
        const nextRow = row + dz;
        if (nextColumn < 0 || nextRow < 0
          || nextColumn >= input.terrain.columns || nextRow >= input.terrain.rows) continue;
        if (!input.isExplored(nextRow * input.terrain.columns + nextColumn)) {
          unseenCells.push(cellCenter(input.terrain, { column: nextColumn, row: nextRow }));
        }
      }
      if (unseenCells.length === 0) continue;

      const target = cellCenter(input.terrain, { column, row });
      const facePoint = {
        x: unseenCells.reduce((sum, point) => sum + point.x, 0) / unseenCells.length,
        y: 0,
        z: unseenCells.reduce((sum, point) => sum + point.z, 0) / unseenCells.length
      };
      const bearing = Math.atan2(
        target.x - (input.here.column + 0.5) * input.terrain.cell,
        target.z - (input.here.row + 0.5) * input.terrain.cell
      );
      candidates.push({
        key: pointKey(target),
        choiceId: `frontier_${sampleSeed.toString(16)}_${column}_${row}`,
        target,
        facePoint,
        distance: Math.hypot(column - input.here.column, row - input.here.row)
          * input.terrain.cell,
        unseen: unseenCells.length,
        turnDegrees: Math.abs(normalizeAngle(bearing - input.robotYaw)) * 180 / Math.PI
      });
    }
  }

  // Modern ECMAScript sort is stable: shuffling first preserves entropy inside
  // an information-gain tier without allowing a one-neighbour cell to displace
  // a two- or three-neighbour frontier.
  const ordered = shuffle(candidates, random).sort((left, right) => right.unseen - left.unseen);
  return {
    sampleId: `${input.worldRevision}:${sampleSeed.toString(16).padStart(8, "0")}`,
    candidates: ordered
  };
}

export function pointKey(point: Vec3): string {
  return `${point.x},${point.z}`;
}

function normalizeAngle(value: number): number {
  let normalized = value;
  while (normalized > Math.PI) normalized -= Math.PI * 2;
  while (normalized < -Math.PI) normalized += Math.PI * 2;
  return normalized;
}
