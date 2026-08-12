import { describe, expect, it } from "vitest";
import { loadRuntimeCatalog } from "../config/load.js";
import { ScenarioSchema } from "../domain/schema.js";
import { drawSeed, materializeScenario } from "./world-generator.js";

describe("world materialization", () => {
  it("builds deterministic but seed-varying connected humanoid block worlds", async () => {
    const catalog = await loadRuntimeCatalog();
    const template = catalog.templates.humanoid_frontier!;
    const first = materializeScenario(template, 0x3141_5926);
    const repeated = materializeScenario(template, 0x3141_5926);
    const other = materializeScenario(template, 0x3141_5927);

    expect(first).toEqual(repeated);
    expect(first).not.toEqual(other);
    expect(first.bounds).toEqual({ width: 36, depth: 36 });
    expect(first.robot).not.toEqual(other.robot);
    expect(first.obstacles).not.toEqual(other.obstacles);
    expect(first.objects.map((object) => object.position))
      .not.toEqual(other.objects.map((object) => object.position));
    expect(first.obstacles.filter((obstacle) => obstacle.id.startsWith("world_boundary_")))
      .toHaveLength(4);
    expect(first.obstacles.some((obstacle) => obstacle.id.startsWith("world_block_")))
      .toBe(true);
    expect(first.chunk_manifest).toMatchObject({
      version: 1,
      grid: { columns: 3, rows: 3 }
    });
    expect(first.chunk_manifest.chunks.flatMap(({ entity_ids }) => entity_ids.obstacles).sort())
      .toEqual(first.obstacles.map(({ id }) => id).sort());
    expect(ScenarioSchema.safeParse(first).success).toBe(true);
  });

  it("rebuilds the same large humanoid world seed and varies the next seed", async () => {
    const catalog = await loadRuntimeCatalog();
    const template = catalog.templates.humanoid_realm!;
    const first = materializeScenario(template, 0x1234_abcd);
    const repeated = materializeScenario(template, 0x1234_abcd);
    const other = materializeScenario(template, 0x1234_abce);

    expect(first).toEqual(repeated);
    expect(first).not.toEqual(other);
    expect(first.obstacles).not.toEqual(other.obstacles);
    expect(first.robot).not.toEqual(other.robot);
    expect(first.objects.map((object) => object.position))
      .not.toEqual(other.objects.map((object) => object.position));
    expect(ScenarioSchema.safeParse(first).success).toBe(true);
    expect(first.bounds).toEqual({ width: 54, depth: 54 });
  });

  it("places generated humanoid entities at separated locations", async () => {
    const catalog = await loadRuntimeCatalog();
    for (const seed of [0, 3, 10_001]) {
      const scenario = catalog.materialize("humanoid_frontier", seed);
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
          )).toBeGreaterThanOrEqual(2.5);
        }
      }
    }
  });

  it("clones authored scenarios and records the supplied run seed", async () => {
    const catalog = await loadRuntimeCatalog();
    const template = catalog.templates.humanoid_courtyard!;
    const first = materializeScenario(template, 17);
    const second = materializeScenario(template, 17);
    expect(first).toEqual(second);
    expect(first).not.toBe(second);
    expect(first.seed).toBe(17);
    expect(first.chunk_manifest).toEqual(second.chunk_manifest);
    expect(first.chunk_manifest.grid).toEqual({ columns: 2, rows: 2 });
  });

  it("materializes the authored full-body manipulation world", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("humanoid_workyard", 29);
    expect(scenario.bounds).toEqual({ width: 28, depth: 22 });
    const rod = scenario.objects.find(({ id }) => id === "assembly_rod")!;
    const stand = scenario.objects.find(({ id }) => id === "pickup_stand")!;
    expect(rod).toMatchObject({
      portable: true,
      position: { x: 4.2, y: 0.67, z: 4.8 },
      size: { x: 0.06, y: 0.22, z: 0.06 },
      capability: {
        shape: "cylinder",
        mass_kg: 0.35,
        friction: { sliding: 0.8, torsional: 0.012, rolling: 0.002 }
      }
    });
    expect(stand).toMatchObject({
      portable: false,
      position: { x: 4.2, y: 0.555, z: 4.8 },
      size: { x: 0.12, y: 0.01, z: 0.12 }
    });
    expect(rod.position.y - rod.size.y / 2).toBeCloseTo(
      stand.position.y + stand.size.y / 2
    );
    expect(Math.hypot(
      rod.position.x - scenario.robot.x,
      rod.position.z - scenario.robot.z
    )).toBeGreaterThan(0.8);
    expect(scenario.default_goal.predicates).toEqual([{
      type: "object_placed",
      object_id: "assembly_rod",
      zone_id: "assembly_bay",
      tolerance: 0.05
    }]);
    expect(ScenarioSchema.safeParse(scenario).success).toBe(true);
  });

  it("draws platform-random unsigned 32-bit run seeds", () => {
    const seeds = Array.from({ length: 16 }, () => drawSeed());
    expect(seeds.every((seed) => Number.isInteger(seed) && seed >= 0 && seed <= 0xffff_ffff)).toBe(true);
    expect(new Set(seeds).size).toBeGreaterThan(1);
  });
});
