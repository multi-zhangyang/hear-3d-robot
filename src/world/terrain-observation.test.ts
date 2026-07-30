import { describe, expect, it, vi } from "vitest";
import type { Terrain, Vec3 } from "../domain/schema.js";
import { buildTerrainSurvey, visibleTerrainCells } from "./terrain-observation.js";

function flatTerrain(size = 5): Terrain {
  return {
    cell: 1,
    block: 1,
    columns: size,
    rows: size,
    heights: new Array<number>(size * size).fill(0)
  };
}

describe("terrain survey projection", () => {
  it("renders only explored heights and exposes only Recast-projected frontiers", () => {
    const terrain = flatTerrain();
    terrain.heights[2 * terrain.columns + 3] = 2;
    const explored = new Set([2 * terrain.columns + 2, 2 * terrain.columns + 3]);
    const projectWalkable = vi.fn((candidates: readonly Vec3[]) => candidates
      .slice(0, 2)
      .map((requested) => ({
        requested,
        point: { ...requested, x: requested.x + 0.02 }
      })));

    const survey = buildTerrainSurvey({
      terrain,
      robotPosition: { x: 2.5, y: 0.3, z: 2.5 },
      radiusCells: 1,
      isExplored: (index) => explored.has(index),
      exploredCount: explored.size,
      exploredTotal: terrain.columns * terrain.rows,
      motionSeed: 41,
      worldRevision: 7,
      robotYaw: 0,
      projectWalkable
    });

    expect(survey).toMatchObject({
      robot_cell: { column: 2, row: 2 },
      origin_cell: { column: 1, row: 1 },
      rows: ["???", "?.2", "???"],
      exploration: { cells_seen: 2, cells_total: 25 },
      movement_sampling: {
        decision_owner: "model",
        automatic_actuation: false,
        choice_count: 1
      }
    });
    expect(projectWalkable).toHaveBeenCalledOnce();
    const projectedCandidates = projectWalkable.mock.calls[0]![0];
    expect(projectedCandidates.length).toBeGreaterThan(0);
    expect(survey.frontier).toHaveLength(1);
    expect(survey.frontier[0]!.target.x).toBeCloseTo(projectedCandidates[0]!.x + 0.02);
    expect(survey.frontier[0]).toMatchObject({
      choice_id: expect.stringMatching(/^frontier_/),
      unseen_neighbours: expect.any(Number),
      turn_degrees: expect.any(Number)
    });
  });

  it("propagates a navigation projection failure instead of advertising an empty frontier", () => {
    const terrain = flatTerrain();
    const explored = new Set([2 * terrain.columns + 2]);
    expect(() => buildTerrainSurvey({
      terrain,
      robotPosition: { x: 2.5, y: 0.3, z: 2.5 },
      radiusCells: 1,
      isExplored: (index) => explored.has(index),
      exploredCount: explored.size,
      exploredTotal: terrain.columns * terrain.rows,
      motionSeed: 41,
      worldRevision: 7,
      robotYaw: 0,
      projectWalkable: () => {
        throw new Error("navigation unavailable");
      }
    })).toThrow("navigation unavailable");
  });
});

describe("terrain sensor observation", () => {
  it("aims at a nearby tall column face and delegates occlusion to the physics ray", () => {
    const terrain = flatTerrain();
    const rayDistances: number[] = [];
    const visible = visibleTerrainCells({
      terrain,
      sensorPosition: { x: 2.5, y: 1.5, z: 2.5 },
      sensorRotation: { x: 0, y: 0, z: 0, w: 1 },
      maximumRange: 2,
      horizontalFieldOfView: Math.PI / 2,
      verticalFieldOfView: 0.2,
      isExplored: () => false,
      heightAt: (column, row) => column === 2 && (row === 1 || row === 3) ? 4 : 0,
      isOccluded: (direction, maximumDistance) => {
        rayDistances.push(maximumDistance);
        expect(direction.z).toBeGreaterThan(0);
        return false;
      }
    });

    expect(visible).toContain(3 * terrain.columns + 2);
    expect(visible).not.toContain(1 * terrain.columns + 2);
    expect(rayDistances).toEqual([0.5]);
  });

  it("excludes explored and physically occluded columns", () => {
    const terrain = flatTerrain();
    const target = 3 * terrain.columns + 2;
    const common = {
      terrain,
      sensorPosition: { x: 2.5, y: 1.5, z: 2.5 },
      sensorRotation: { x: 0, y: 0, z: 0, w: 1 },
      maximumRange: 2,
      horizontalFieldOfView: Math.PI / 2,
      verticalFieldOfView: 0.2,
      heightAt: (column: number, row: number) => column === 2 && row === 3 ? 4 : 0
    };

    expect(visibleTerrainCells({
      ...common,
      isExplored: (index) => index === target,
      isOccluded: () => false
    })).not.toContain(target);
    expect(visibleTerrainCells({
      ...common,
      isExplored: () => false,
      isOccluded: () => true
    })).not.toContain(target);
  });
});
