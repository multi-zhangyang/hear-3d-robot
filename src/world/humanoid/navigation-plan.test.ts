import { describe, expect, it } from "vitest";
import type { NavigationPlan } from "../navigation.js";
import { boundedNavigationChunk } from "./navigation-plan.js";

const route: NavigationPlan = {
  waypoints: [
    { x: 0, y: 0.8, z: 0 },
    { x: 0, y: 0.8, z: 2 },
    { x: 2, y: 0.8, z: 2 }
  ],
  distance: 4,
  resolvedTarget: { x: 2, y: 0.8, z: 2 },
  projectionDistance: 0
};

describe("bounded navigation chunks", () => {
  it("ends exactly on the distance budget without mutating the complete route", () => {
    const chunk = boundedNavigationChunk(route, 3);

    expect(chunk).toEqual({
      waypoints: [
        { x: 0, y: 0.8, z: 0 },
        { x: 0, y: 0.8, z: 2 },
        { x: 1, y: 0.8, z: 2 }
      ],
      distance: 3,
      resolvedTarget: { x: 1, y: 0.8, z: 2 },
      projectionDistance: 0
    });
    expect(route.resolvedTarget).toEqual({ x: 2, y: 0.8, z: 2 });
  });

  it("returns an isolated copy when the route already fits", () => {
    const complete = boundedNavigationChunk(route, 5);
    expect(complete).toEqual(route);
    expect(complete).not.toBe(route);
    expect(() => boundedNavigationChunk(route, 0)).toThrow("finite and positive");
  });
});
