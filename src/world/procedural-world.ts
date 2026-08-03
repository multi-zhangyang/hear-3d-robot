import type { Scenario, ScenarioTemplate } from "../domain/schema.js";
import { createRandom, deriveSeed, randomBetween, shuffle } from "./random.js";

type ProceduralTemplate = Extract<ScenarioTemplate, { kind: "procedural" }>;

interface GridCell {
  column: number;
  row: number;
}

const ENTITY_SEPARATION_CELLS = 2;
const ENTITY_OBSTACLE_CLEARANCE_CELLS = 1.45;

export function materializeProceduralWorld(
  template: ProceduralTemplate,
  seed: number
): Scenario {
  const shape = template.generate;
  const random = createRandom(deriveSeed(seed, "procedural-world"));
  const columns = Math.floor(shape.bounds.width / shape.cell);
  const rows = Math.floor(shape.bounds.depth / shape.cell);
  const cells = shuffle(interiorCells(columns, rows), random);
  const entityCount = 1 + shape.objects.length + shape.zones.length;
  const claimed: GridCell[] = [];
  for (let index = 0; index < entityCount; index += 1) {
    const cell = cells.find((candidate) => claimed.every((other) => (
      gridDistance(candidate, other) >= ENTITY_SEPARATION_CELLS
    )));
    if (!cell) {
      throw new Error(`Procedural world for seed ${seed} cannot place ${entityCount} entities`);
    }
    cells.splice(cells.indexOf(cell), 1);
    claimed.push(cell);
  }

  const obstacleCandidates = shuffle(interiorCells(columns, rows).filter((candidate) => (
    claimed.every((entity) => (
      gridDistance(candidate, entity) >= ENTITY_OBSTACLE_CLEARANCE_CELLS
    ))
  )), random);
  const blocked = new Set<string>();
  const targetObstacleCount = Math.floor(
    (columns - 2) * (rows - 2) * shape.obstacle_density
  );
  for (const candidate of obstacleCandidates) {
    if (blocked.size >= targetObstacleCount) break;
    blocked.add(cellKey(candidate));
    if (!claimsRemainConnected(columns, rows, blocked, claimed)) {
      blocked.delete(cellKey(candidate));
    }
  }

  const robotCell = claimed[0]!;
  const objectCells = claimed.slice(1, 1 + shape.objects.length);
  const zoneCells = claimed.slice(1 + shape.objects.length);
  const objects = shape.objects.map((object, index) => {
    const cell = objectCells[index]!;
    const centre = cellCentre(cell, shape.cell);
    const inset = Math.max(object.size.x, object.size.z) / 2 + 0.2;
    const range = Math.max(0, shape.cell / 2 - inset);
    return {
      ...structuredClone(object),
      position: {
        x: centre.x + randomBetween(random, -range, range),
        y: object.size.y / 2,
        z: centre.z + randomBetween(random, -range, range)
      }
    };
  });
  const zones = shape.zones.map((zone, index) => ({
    ...structuredClone(zone),
    center: { ...cellCentre(zoneCells[index]!, shape.cell), y: 0.01 }
  }));
  const interiorObstacles = [...blocked].sort().map((key, index) => {
    const cell = parseCellKey(key);
    const height = randomBetween(
      random,
      shape.minimum_obstacle_height,
      shape.maximum_obstacle_height
    );
    return {
      id: `world_block_${String(index + 1).padStart(3, "0")}`,
      center: { ...cellCentre(cell, shape.cell), y: height / 2 },
      size: {
        x: shape.cell * randomBetween(random, 0.58, 0.78),
        y: height,
        z: shape.cell * randomBetween(random, 0.58, 0.78)
      }
    };
  });
  const robot = cellCentre(robotCell, shape.cell);

  return {
    title: template.title,
    seed,
    bounds: structuredClone(shape.bounds),
    visibility_radius: shape.visibility_radius,
    robot: {
      x: robot.x,
      z: robot.z,
      yaw: randomBetween(random, -Math.PI, Math.PI)
    },
    obstacles: [...boundaryWalls(shape.bounds), ...interiorObstacles],
    objects,
    zones,
    default_goal: structuredClone(shape.default_goal)
  };
}

function interiorCells(columns: number, rows: number): GridCell[] {
  const cells: GridCell[] = [];
  for (let row = 1; row < rows - 1; row += 1) {
    for (let column = 1; column < columns - 1; column += 1) {
      cells.push({ column, row });
    }
  }
  return cells;
}

function claimsRemainConnected(
  columns: number,
  rows: number,
  blocked: ReadonlySet<string>,
  claims: readonly GridCell[]
): boolean {
  const first = claims[0];
  if (!first) return true;
  const visited = new Set<string>([cellKey(first)]);
  const queue = [first];
  for (let index = 0; index < queue.length; index += 1) {
    const current = queue[index]!;
    for (const [dx, dz] of NEIGHBOURS) {
      const next = { column: current.column + dx, row: current.row + dz };
      if (next.column < 0 || next.row < 0
        || next.column >= columns || next.row >= rows) continue;
      const key = cellKey(next);
      if (blocked.has(key) || visited.has(key)) continue;
      visited.add(key);
      queue.push(next);
    }
  }
  return claims.every((claim) => visited.has(cellKey(claim)));
}

function boundaryWalls(bounds: Scenario["bounds"]): Scenario["obstacles"] {
  const thickness = 0.3;
  const height = 2.4;
  return [
    {
      id: "world_boundary_north",
      center: { x: bounds.width / 2, y: height / 2, z: thickness / 2 },
      size: { x: bounds.width, y: height, z: thickness }
    },
    {
      id: "world_boundary_south",
      center: { x: bounds.width / 2, y: height / 2, z: bounds.depth - thickness / 2 },
      size: { x: bounds.width, y: height, z: thickness }
    },
    {
      id: "world_boundary_west",
      center: { x: thickness / 2, y: height / 2, z: bounds.depth / 2 },
      size: { x: thickness, y: height, z: bounds.depth }
    },
    {
      id: "world_boundary_east",
      center: { x: bounds.width - thickness / 2, y: height / 2, z: bounds.depth / 2 },
      size: { x: thickness, y: height, z: bounds.depth }
    }
  ];
}

function cellCentre(cell: GridCell, cellSize: number): { x: number; z: number } {
  return {
    x: (cell.column + 0.5) * cellSize,
    z: (cell.row + 0.5) * cellSize
  };
}

function gridDistance(left: GridCell, right: GridCell): number {
  return Math.hypot(left.column - right.column, left.row - right.row);
}

function cellKey(cell: GridCell): string {
  return `${cell.column}:${cell.row}`;
}

function parseCellKey(key: string): GridCell {
  const [column, row] = key.split(":").map(Number);
  if (!Number.isInteger(column) || !Number.isInteger(row)) {
    throw new Error(`Invalid procedural cell key: ${key}`);
  }
  return { column: column!, row: row! };
}

const NEIGHBOURS: ReadonlyArray<readonly [number, number]> = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1]
];
