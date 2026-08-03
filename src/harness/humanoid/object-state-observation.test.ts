import { describe, expect, it } from "vitest";
import { ScenarioSchema } from "../../domain/schema.js";
import { HumanoidWorld } from "../../world/humanoid/world.js";
import { HumanoidActionRuntime } from "./runtime.js";

const scenario = ScenarioSchema.parse({
  title: "Role object observation",
  seed: 43,
  bounds: { width: 8, depth: 8 },
  visibility_radius: 5,
  robot: { x: 0, z: 0, yaw: 0 },
  obstacles: [],
  objects: [{
    id: "crate",
    kind: "crate",
    color: "#8b6b45",
    position: { x: 0, y: 0.15, z: 1.5 },
    size: { x: 0.3, y: 0.3, z: 0.3 },
    portable: true
  }],
  zones: [],
  default_goal: {
    summary: "Observe the crate",
    predicates: [{
      type: "robot_at",
      target: { x: 0, y: 0, z: 0 },
      tolerance: 0.2
    }]
  }
});

describe("humanoid role object observation", () => {
  it("publishes authority and observation identity to the model harness", async () => {
    const world = await HumanoidWorld.create(scenario);
    const runtime = new HumanoidActionRuntime(world);
    try {
      const receipt = await runtime.invoke(
        "observe_humanoid",
        {},
        "observe-role-object",
        "perception-agent"
      );
      expect(receipt.accepted).toBe(true);
      const detail = record(receipt.detail);
      const tokens = detail.object_tokens;
      if (!Array.isArray(tokens)) throw new Error("Expected object token array");
      expect(tokens).toHaveLength(1);
      expect(tokens[0]).toMatchObject({
        id: "crate",
        role: "manipulable",
        status: "visible",
        state: "active",
        authority: "mujoco_exact",
        exact: true,
        observable: true,
        observed_frame: detail.frame,
        observed_world_revision: receipt.worldAfterRevision,
        pose: {
          position: expect.any(Object),
          rotation: expect.any(Object)
        }
      });
    } finally {
      await world.dispose();
    }
  }, 30_000);
});

function record(value: unknown): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Expected object observation detail");
  }
  return value as Record<string, unknown>;
}
