import { describe, expect, it } from "vitest";
import { nextFirstPersonLook, pointerToNdc } from "./stage-interaction";

describe("stage interaction", () => {
  it("maps canvas points to normalized ray coordinates", () => {
    const bounds = { left: 100, top: 50, width: 800, height: 400 };

    expect(pointerToNdc({ x: 500, y: 250 }, bounds).toArray()).toEqual([0, -0]);
    expect(pointerToNdc({ x: 100, y: 50 }, bounds).toArray()).toEqual([-1, 1]);
    expect(pointerToNdc({ x: 900, y: 450 }, bounds).toArray()).toEqual([1, -1]);
  });

  it("turns pointer deltas into a bounded observation offset", () => {
    const turned = nextFirstPersonLook({ yaw: 0, pitch: 0 }, 100, -50, 0.01);
    expect(turned.yaw).toBeCloseTo(-1);
    expect(turned.pitch).toBeCloseTo(0.5);

    const clamped = nextFirstPersonLook(turned, 0, -10_000, 0.01);
    expect(clamped.pitch).toBeLessThan(Math.PI / 2);
    expect(clamped.pitch).toBeGreaterThan(1.4);
  });
});
