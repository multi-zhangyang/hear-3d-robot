import { describe, expect, it } from "vitest";
import { ScenarioSchema } from "../../domain/schema.js";
import { HumanoidObjectMemory } from "./object-memory.js";
import { HumanoidSimulation } from "./simulation.js";

const scenario = ScenarioSchema.parse({
  title: "Humanoid object memory",
  seed: 31,
  bounds: { width: 8, depth: 8 },
  visibility_radius: 5,
  robot: { x: 0, z: 0, yaw: 0 },
  obstacles: [],
  objects: [
    {
      id: "near",
      kind: "crate",
      color: "#8b6b45",
      position: { x: 0, y: 1.05, z: 1.2 },
      size: { x: 0.3, y: 0.3, z: 0.3 },
      portable: true
    },
    {
      id: "hidden",
      kind: "crate",
      color: "#426a88",
      position: { x: 0, y: 1.05, z: -1.2 },
      size: { x: 0.3, y: 0.3, z: 0.3 },
      portable: true
    }
  ],
  zones: [],
  default_goal: {
    summary: "观察物体",
    predicates: [{
      type: "robot_at",
      target: { x: 0, y: 0, z: 0 },
      tolerance: 0.2
    }]
  }
});

describe("HumanoidObjectMemory", () => {
  it("keeps source revisions for seen objects without revealing unseen world state", async () => {
    const simulation = await HumanoidSimulation.create({
      objects: scenario.objects.map((object) => ({
        id: object.id,
        center: object.position,
        size: object.size,
        mass: 0.2
      }))
    });
    try {
      const memory = new HumanoidObjectMemory(scenario);
      const sensed = simulation.senseObjects(scenario.visibility_radius);
      expect(Object.keys(sensed.objects)).toEqual(["near"]);

      memory.observe(10, 5, sensed.objects);
      const visible = memory.tokens(simulation.snapshot(), 5, new Set(["near"]));
      expect(visible).toHaveLength(1);
      expect(visible[0]).toMatchObject({
        id: "near",
        kind: "crate",
        status: "visible",
        firstSeenRevision: 5,
        lastSeenRevision: 5,
        observationCount: 1,
        ageRevisions: 0
      });
      expect(visible.some((token) => token.id === "hidden")).toBe(false);

      const restored = new HumanoidObjectMemory(scenario, memory.checkpoint());
      const remembered = restored.tokens(simulation.snapshot(), 9, new Set());
      expect(remembered).toHaveLength(1);
      expect(remembered[0]).toMatchObject({
        id: "near",
        status: "remembered",
        lastSeenRevision: 5,
        ageRevisions: 4
      });
    } finally {
      await simulation.dispose();
    }
  }, 30_000);
});
