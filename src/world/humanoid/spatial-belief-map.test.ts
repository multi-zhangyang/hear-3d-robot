import { describe, expect, it } from "vitest";
import { HumanoidSpatialBeliefMap } from "./spatial-belief-map.js";

describe("HumanoidSpatialBeliefMap", () => {
  it("builds persistent information-gain frontiers from sensor coverage", () => {
    const map = new HumanoidSpatialBeliefMap({
      bounds: { width: 8, height: 3, depth: 8 }
    });
    map.observe({
      frame: 10,
      rootPosition: { x: 4, y: 0.8, z: 4 },
      sensor: {
        position: { x: 4, y: 1.5, z: 4 },
        rotation: { x: 0, y: 0, z: 0, w: 1 },
        maximumRange: 3,
        horizontalFieldOfView: Math.PI / 2,
        verticalFieldOfView: Math.PI
      },
      visibleSolids: [],
      pointVisibility: allPointsVisible
    });
    const observation = map.observation({ x: 4, y: 0.8, z: 4 });
    expect(observation.coverage_ratio).toBeGreaterThan(0);
    expect(observation.coverage_ratio).toBeLessThan(1);
    expect(observation.frontiers.length).toBeGreaterThan(0);
    expect(observation.frontiers[0]!.expected_information_gain).toBeGreaterThan(0);
    for (const [index, frontier] of observation.frontiers.entries()) {
      for (const other of observation.frontiers.slice(index + 1)) {
        expect(Math.hypot(
          frontier.target.x - other.target.x,
          frontier.target.z - other.target.z
        )).toBeGreaterThanOrEqual(1.5);
      }
    }

    const restored = new HumanoidSpatialBeliefMap(
      { bounds: { width: 8, height: 3, depth: 8 } },
      map.checkpoint()
    );
    expect(restored.observation({ x: 4, y: 0.8, z: 4 }))
      .toEqual(observation);
  });

  it("marks visible solids occupied without offering them as frontiers", () => {
    const map = new HumanoidSpatialBeliefMap({
      bounds: { width: 6, height: 3, depth: 6 }
    });
    map.observe({
      frame: 1,
      rootPosition: { x: 3, y: 0.8, z: 1 },
      sensor: {
        position: { x: 3, y: 1.5, z: 1 },
        rotation: { x: 0, y: 0, z: 0, w: 1 },
        maximumRange: 4,
        horizontalFieldOfView: Math.PI / 2,
        verticalFieldOfView: Math.PI
      },
      visibleSolids: [{
        center: { x: 3, y: 0.5, z: 3 },
        size: { x: 1, y: 1, z: 1 }
      }],
      pointVisibility: allPointsVisible
    });
    const observation = map.observation({ x: 3, y: 0.8, z: 1 });
    expect(observation.occupied_cell_count).toBeGreaterThan(0);
    expect(observation.frontiers.some(({ target }) => (
      Math.abs(target.x - 3) < 0.51 && Math.abs(target.z - 3) < 0.51
    ))).toBe(false);
  });

  it("keeps cells behind visible physical geometry unknown", () => {
    const map = new HumanoidSpatialBeliefMap({
      bounds: { width: 8, height: 3, depth: 8 }
    });
    map.observe({
      frame: 4,
      rootPosition: { x: 4, y: 0.8, z: 1 },
      sensor: {
        position: { x: 4, y: 1.5, z: 1 },
        rotation: { x: 0, y: 0, z: 0, w: 1 },
        maximumRange: 7,
        horizontalFieldOfView: Math.PI / 2,
        verticalFieldOfView: Math.PI
      },
      visibleSolids: [{
        center: { x: 4, y: 1, z: 3 },
        size: { x: 2, y: 2, z: 0.5 }
      }],
      pointVisibility: (points) => points.map(({ z }) => z <= 3.25)
    });

    const cells = map.checkpoint().cells;
    expect(cells.find(({ x, z }) => x === 8 && z === 4)?.occupied).toBe(false);
    expect(cells.some(({ x, z }) => x === 8 && z === 6 && z <= 7)).toBe(true);
    expect(cells.some(({ x, z }) => x === 8 && z === 10)).toBe(false);
    expect(map.observation({ x: 4, y: 0.8, z: 1 })).toMatchObject({
      protocol: "humanoid-spatial-belief-v2",
      visibility_model: "occlusion_aware_head_camera",
      frontier_model: "reachable_geodesic_diversity"
    });
  });

  it("clears stale occupancy only after the former cell is visible again", () => {
    const map = new HumanoidSpatialBeliefMap({
      bounds: { width: 6, height: 3, depth: 6 }
    });
    const sensor = {
      position: { x: 3, y: 1.5, z: 1 },
      rotation: { x: 0, y: 0, z: 0, w: 1 },
      maximumRange: 5,
      horizontalFieldOfView: Math.PI / 2,
      verticalFieldOfView: Math.PI
    };
    map.observe({
      frame: 1,
      rootPosition: { x: 3, y: 0.8, z: 1 },
      sensor,
      visibleSolids: [{
        center: { x: 3, y: 0.4, z: 3 },
        size: { x: 0.5, y: 0.8, z: 0.5 }
      }],
      pointVisibility: allPointsVisible
    });
    expect(map.checkpoint().cells.find(({ x, z }) => x === 6 && z === 6)?.occupied)
      .toBe(true);

    map.observe({
      frame: 2,
      rootPosition: { x: 3, y: 0.8, z: 1 },
      sensor,
      visibleSolids: [],
      pointVisibility: allPointsVisible
    });
    expect(map.checkpoint().cells.find(({ x, z }) => x === 6 && z === 6))
      .toMatchObject({ occupied: false, last_observed_frame: 2 });
  });

  it("uses a carried object as an occluder without turning it into a world obstacle", () => {
    const map = new HumanoidSpatialBeliefMap({
      bounds: { width: 6, height: 3, depth: 6 }
    });
    map.observe({
      frame: 8,
      rootPosition: { x: 3, y: 0.8, z: 1 },
      sensor: {
        position: { x: 3, y: 1.5, z: 1 },
        rotation: { x: 0, y: 0, z: 0, w: 1 },
        maximumRange: 5,
        horizontalFieldOfView: Math.PI / 2,
        verticalFieldOfView: Math.PI
      },
      visibleSolids: [{
        center: { x: 3, y: 1.2, z: 2 },
        size: { x: 1, y: 1, z: 0.5 },
        occupiesNavigationSpace: false
      }],
      pointVisibility: (points) => points.map(({ z }) => z <= 2.25)
    });

    const checkpoint = map.checkpoint();
    expect(checkpoint.cells.some(({ occupied }) => occupied)).toBe(false);
    expect(checkpoint.cells.some(({ x, z }) => x === 6 && z === 8)).toBe(false);
  });

  it("offers only frontiers connected through observed free space", () => {
    const freeCell = (x: number, z: number) => ({
      x,
      z,
      first_observed_frame: 1,
      last_observed_frame: 1,
      visit_count: x === 2 && z === 2 ? 1 : 0,
      occupied: false
    });
    const map = new HumanoidSpatialBeliefMap({
      bounds: { width: 6, height: 3, depth: 6 }
    }, {
      version: 1,
      resolution_m: 0.5,
      last_updated_frame: 1,
      cells: [
        freeCell(2, 2),
        freeCell(2, 3),
        freeCell(2, 4),
        freeCell(3, 4),
        freeCell(4, 4),
        freeCell(5, 4),
        freeCell(5, 3),
        freeCell(5, 2),
        freeCell(8, 2),
        freeCell(9, 2)
      ]
    });

    const observation = map.observation({ x: 1.25, y: 0.8, z: 1.25 });
    expect(observation.reachable_free_cell_count).toBe(8);
    expect(observation.frontiers.some(({ id }) => id === "frontier:2:2")).toBe(false);
    expect(observation.frontiers.some(({ id }) => id === "frontier:8:2")).toBe(false);
    expect(observation.frontiers.find(({ id }) => id === "frontier:5:4"))
      .toMatchObject({ travel_distance_m: 2.5 });
  });

  it("rejects duplicated, future and out-of-bounds checkpoint cells", () => {
    const cell = {
      x: 2,
      z: 2,
      first_observed_frame: 1,
      last_observed_frame: 1,
      visit_count: 0,
      occupied: false
    };
    const restore = (cells: Array<typeof cell>, lastUpdatedFrame = 1) => (
      new HumanoidSpatialBeliefMap({
        bounds: { width: 4, height: 3, depth: 4 }
      }, {
        version: 1,
        resolution_m: 0.5,
        last_updated_frame: lastUpdatedFrame,
        cells
      })
    );

    expect(() => restore([cell, { ...cell }])).toThrow("repeats cell 2:2");
    expect(() => restore([{ ...cell, last_observed_frame: 2 }]))
      .toThrow("newer than the checkpoint observation head");
    expect(() => restore([{ ...cell, x: 8 }]))
      .toThrow("outside the world bounds");
  });
});

function allPointsVisible(points: readonly unknown[]): boolean[] {
  return points.map(() => true);
}
