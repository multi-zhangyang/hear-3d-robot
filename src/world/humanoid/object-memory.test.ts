import { describe, expect, it } from "vitest";
import { ScenarioSchema } from "../../domain/schema.js";
import { HumanoidObjectMemory } from "./object-memory.js";
import { detectHumanoidMotionOption } from "./motion-option.js";
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

      const authoritative = simulation.snapshot();
      memory.refresh(
        10,
        5,
        authoritative.objects,
        new Set(Object.keys(sensed.objects))
      );
      const visible = memory.tokens(authoritative, 10, 5);
      expect(visible).toHaveLength(1);
      expect(visible[0]).toMatchObject({
        id: "near",
        kind: "crate",
        role: "manipulable",
        status: "visible",
        state: "active",
        authority: "mujoco_exact",
        exact: true,
        observable: true,
        observedFrame: 10,
        observedWorldRevision: 5,
        firstSeenRevision: 5,
        lastSeenRevision: 5,
        observationCount: 1,
        ageRevisions: 0
      });
      expect(visible.some((token) => token.id === "hidden")).toBe(false);
      expect(memory.activeObjectStates()).toHaveLength(2);
      const contract = {
        option_id: "near-at-observed-pose",
        predicates: [{
          type: "object_near_point" as const,
          object_id: "near",
          target: { ...visible[0]!.position },
          tolerance_m: 0.01
        }],
        stable_steps: 1
      };
      expect(detectHumanoidMotionOption(contract, {
        snapshot: authoritative,
        observableObjects: memory.observableObjectStates(10, 5).map((state) => ({
          id: state.id,
          position: state.pose.position,
          size: state.size
        })),
        zones: []
      })).toMatchObject({ allSatisfied: true, hasUncertain: false });

      memory.refresh(10, 5, authoritative.objects, new Set());
      const occluded = memory.tokens(authoritative, 10, 5);
      expect(occluded[0]).toMatchObject({
        id: "near",
        status: "remembered",
        state: "historical",
        authority: "sensor_history",
        exact: false,
        observable: false,
        observedFrame: 10,
        observedWorldRevision: 5
      });
      expect(memory.observedObjectIds(10, 5)).toEqual(new Set());
      expect(memory.observableObjectStates(10, 5)).toEqual([]);
      expect(detectHumanoidMotionOption(contract, {
        snapshot: authoritative,
        observableObjects: memory.observableObjectStates(10, 5).map((state) => ({
          id: state.id,
          position: state.pose.position,
          size: state.size
        })),
        zones: []
      })).toMatchObject({
        allSatisfied: false,
        hasUncertain: true,
        evidence: [expect.objectContaining({
          status: "uncertain",
          objectObservable: false,
          reason: "object_not_observable"
        })]
      });

      const restored = new HumanoidObjectMemory(scenario, memory.checkpoint());
      expect(restored.observedObjectIds(10, 5)).toEqual(new Set());
      expect(restored.tokens(authoritative, 10, 5)[0]).toMatchObject({
        status: "remembered",
        state: "historical",
        observable: false,
        exact: false
      });
      const remembered = restored.tokens(authoritative, 10, 9);
      expect(remembered).toHaveLength(1);
      expect(remembered[0]).toMatchObject({
        id: "near",
        status: "remembered",
        state: "historical",
        authority: "sensor_history",
        exact: false,
        observable: false,
        lastSeenRevision: 5,
        ageRevisions: 4
      });
    } finally {
      await simulation.dispose();
    }
  }, 30_000);
});
