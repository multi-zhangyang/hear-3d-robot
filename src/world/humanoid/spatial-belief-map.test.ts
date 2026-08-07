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
        horizontalFieldOfView: Math.PI / 2
      },
      visibleSolids: []
    });
    const observation = map.observation({ x: 4, y: 0.8, z: 4 });
    expect(observation.coverage_ratio).toBeGreaterThan(0);
    expect(observation.coverage_ratio).toBeLessThan(1);
    expect(observation.frontiers.length).toBeGreaterThan(0);
    expect(observation.frontiers[0]!.expected_information_gain).toBeGreaterThan(0);

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
        horizontalFieldOfView: Math.PI / 2
      },
      visibleSolids: [{
        center: { x: 3, y: 0.5, z: 3 },
        size: { x: 1, y: 1, z: 1 }
      }]
    });
    const observation = map.observation({ x: 3, y: 0.8, z: 1 });
    expect(observation.occupied_cell_count).toBeGreaterThan(0);
    expect(observation.frontiers.some(({ target }) => (
      Math.abs(target.x - 3) < 0.51 && Math.abs(target.z - 3) < 0.51
    ))).toBe(false);
  });
});
