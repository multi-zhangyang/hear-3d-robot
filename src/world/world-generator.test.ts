import { describe, expect, it } from "vitest";
import { loadRuntimeCatalog } from "../config/load.js";
import { ScenarioSchema } from "../domain/schema.js";
import { ROBOT_SPEC } from "./robot-model.js";
import { cellAt, cellClearance } from "./terrain.js";
import { drawSeed, materializeScenario } from "./world-generator.js";

describe("world materialization", () => {
  it("rebuilds the same seed exactly and creates a genuinely different next seed", async () => {
    const catalog = await loadRuntimeCatalog();
    const template = catalog.templates.voxel_expanse!;
    const first = materializeScenario(template, 0x1234_abcd);
    const repeated = materializeScenario(template, 0x1234_abcd);
    const other = materializeScenario(template, 0x1234_abce);

    expect(first).toEqual(repeated);
    expect(first).not.toEqual(other);
    expect(first.terrain?.heights).not.toEqual(other.terrain?.heights);
    expect(first.robot).not.toEqual(other.robot);
    expect(first.objects.map((object) => object.position))
      .not.toEqual(other.objects.map((object) => object.position));
    expect(ScenarioSchema.safeParse(first).success).toBe(true);
    expect(first.bounds).toEqual({ width: 80, depth: 80 });
  });

  it("places every generated body on clear floor with separated locations", async () => {
    const catalog = await loadRuntimeCatalog();
    for (const seed of [0, 3, 10_001]) {
      const scenario = catalog.materialize("voxel_expanse", seed);
      const terrain = scenario.terrain!;
      expect(cellClearance(
        terrain,
        cellAt(terrain, { x: scenario.robot.x, y: 0, z: scenario.robot.z }),
        ROBOT_SPEC.base.footprintRadius
      )).toBe(true);

      const points = [
        { x: scenario.robot.x, z: scenario.robot.z },
        ...scenario.objects.map(({ position }) => position),
        ...scenario.zones.map(({ center }) => center)
      ];
      for (let left = 0; left < points.length; left += 1) {
        for (let right = left + 1; right < points.length; right += 1) {
          expect(Math.hypot(
            points[left]!.x - points[right]!.x,
            points[left]!.z - points[right]!.z
          )).toBeGreaterThanOrEqual(3.5);
        }
      }
      for (const [joint, value] of Object.entries(scenario.robot.joints!)) {
        const limit = ROBOT_SPEC.joints[joint as keyof typeof ROBOT_SPEC.joints];
        expect(value).toBeGreaterThanOrEqual(limit.minimum);
        expect(value).toBeLessThanOrEqual(limit.maximum);
      }
    }
  });

  it("clones authored scenarios and records the supplied run seed", async () => {
    const catalog = await loadRuntimeCatalog();
    const template = catalog.templates.open_navigation!;
    const first = materializeScenario(template, 17);
    const second = materializeScenario(template, 17);
    expect(first).toEqual(second);
    expect(first).not.toBe(second);
    expect(first.seed).toBe(17);
    expect(first.terrain).toBeUndefined();
  });

  it("keeps motion entropy independent from a fixed world", async () => {
    const catalog = await loadRuntimeCatalog();
    const template = catalog.templates.voxel_survey!;
    const first = materializeScenario(template, 91, 1001);
    const second = materializeScenario(template, 91, 2002);
    const { motion_seed: firstMotion, ...firstWorld } = first;
    const { motion_seed: secondMotion, ...secondWorld } = second;

    expect(firstWorld).toEqual(secondWorld);
    expect(firstMotion).toBe(1001);
    expect(secondMotion).toBe(2002);
  });

  it("draws platform-random unsigned 32-bit run seeds", () => {
    const seeds = Array.from({ length: 16 }, () => drawSeed());
    expect(seeds.every((seed) => Number.isInteger(seed) && seed >= 0 && seed <= 0xffff_ffff)).toBe(true);
    expect(new Set(seeds).size).toBeGreaterThan(1);
  });
});
