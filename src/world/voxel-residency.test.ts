import { describe, expect, it } from "vitest";
import { ScenarioSchema, type BodyChannel, type Scenario } from "../domain/schema.js";
import { RapierWorld, type SourceCommand } from "./rapier-world.js";

describe("dynamic voxel residency", () => {
  it("keeps a portable object resting on a remote high platform", async () => {
    const scenario = platformScenario({
      chunkSize: 16,
      platformColumn: 80,
      platformRow: 80,
      robot: { x: 8.5, z: 8.5 }
    });
    const world = await RapierWorld.create(scenario);
    try {
      const before = world.snapshot();
      expect(before.voxels?.loaded_chunks).toContainEqual({ column: 5, row: 5 });
      expect(before.objects[0]?.position.y).toBeCloseTo(3.25, 2);

      const result = await world.driveBase(source("remote_settle"), 0, 0, 5);
      const after = world.snapshot();
      expect(result).toMatchObject({ accepted: true, code: "base_motion_completed" });
      expect(after.objects[0]?.position).toMatchObject({
        x: expect.closeTo(80.5, 2),
        y: expect.closeTo(3.25, 2),
        z: expect.closeTo(80.5, 2)
      });
      expect(world.planBasePath({ x: 10.5, y: 0, z: 8.5 })).toMatchObject({
        accepted: true,
        code: "base_path_planned"
      });
    } finally {
      world.dispose();
    }
  });

  it("preserves support while the robot leaves and re-enters the object's chunk window", async () => {
    const scenario = platformScenario({
      chunkSize: 8,
      platformColumn: 12,
      platformRow: 8,
      robot: { x: 8.5, z: 8.5 }
    });
    const world = await RapierWorld.create(scenario);
    try {
      const initialObject = world.snapshot().objects[0]!.position;
      for (let index = 0; index < 7; index += 1) {
        const result = await world.driveBase(
          source(`leave_${index}`),
          0.8,
          0,
          5
        );
        expect(result.accepted).toBe(true);
      }

      const distant = world.snapshot();
      expect(distant.robot.position.z).toBeGreaterThan(36);
      expect(distant.voxels?.loaded_chunks).toContainEqual({ column: 1, row: 1 });
      expect(distant.objects[0]?.position).toMatchObject({
        x: expect.closeTo(initialObject.x, 2),
        y: expect.closeTo(initialObject.y, 2),
        z: expect.closeTo(initialObject.z, 2)
      });

      for (let index = 0; index < 7; index += 1) {
        const result = await world.driveBase(
          source(`return_${index}`),
          -0.8,
          0,
          5
        );
        expect(result.accepted).toBe(true);
      }

      const returned = world.snapshot();
      expect(returned.robot.position.z).toBeCloseTo(8.5, 2);
      expect(returned.objects[0]?.position).toMatchObject({
        x: expect.closeTo(initialObject.x, 2),
        y: expect.closeTo(initialObject.y, 2),
        z: expect.closeTo(initialObject.z, 2)
      });
      expect(world.planBasePath({ x: 8.5, y: 0, z: 12.5 })).toMatchObject({
        accepted: true,
        code: "base_path_planned"
      });
    } finally {
      world.dispose();
    }
  }, 20_000);
});

function source(id: string): SourceCommand {
  return {
    id,
    agentId: "navigation_leaf",
    agentName: "Navigation leaf",
    skill: "drive_base",
    channels: ["base"] satisfies BodyChannel[]
  };
}

function platformScenario(input: {
  chunkSize: number;
  platformColumn: number;
  platformRow: number;
  robot: { x: number; z: number };
}): Scenario {
  const size = 96;
  const heights = new Array<number>(size * size).fill(0);
  for (let row = input.platformRow - 1; row <= input.platformRow + 1; row += 1) {
    for (let column = input.platformColumn - 1; column <= input.platformColumn + 1; column += 1) {
      heights[row * size + column] = 3;
    }
  }
  return ScenarioSchema.parse({
    title: "Dynamic residency",
    seed: 17,
    motion_seed: 29,
    bounds: { width: size, depth: size },
    terrain: {
      cell: 1,
      columns: size,
      rows: size,
      block: 1,
      chunk_size: input.chunkSize,
      maximum_height: 24,
      heights
    },
    visibility_radius: 8,
    robot: { ...input.robot, yaw: 0 },
    obstacles: [],
    objects: [{
      id: "payload",
      kind: "block",
      color: "#d84a4a",
      position: {
        x: input.platformColumn + 0.5,
        y: 3.25,
        z: input.platformRow + 0.5
      },
      size: { x: 0.5, y: 0.5, z: 0.5 },
      portable: true
    }],
    zones: [],
    affordances: [],
    default_goal: {
      summary: "Robot remains in the world.",
      predicates: [{
        type: "robot_at",
        target: { x: input.robot.x, y: 0, z: input.robot.z },
        tolerance: 0.2
      }]
    }
  });
}
