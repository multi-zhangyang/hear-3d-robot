import { describe, expect, it } from "vitest";
import {
  blockedHumanoidContacts,
  HumanoidMotionPlanSchema,
  prepareHumanoidMotion,
  type HumanoidMotionPlan
} from "./motion-plan.js";
import { hydrateHumanoidReference } from "./motion-artifact.js";
import { neutralHumanoidReference } from "./reference.js";
import { inverseQuaternion, rotateVector, subtract } from "../geometry.js";
import {
  HumanoidSimulation,
  type HumanoidSimulationSnapshot
} from "./simulation.js";

describe("humanoid whole-body motion", () => {
  it("previews a continuous model-authored plan without mutating the physical world", async () => {
    const simulation = await HumanoidSimulation.create({
      spawn: {
        position: { x: 1.25, y: 0, z: 1.5 },
        yaw: Math.PI / 6
      },
      solids: [{
        id: "distant-wall",
        center: { x: 8, y: 1, z: 8 },
        size: { x: 1, y: 2, z: 4 }
      }]
    });
    try {
      const neutral = neutralHumanoidReference();
      for (let index = 0; index < 100; index += 1) await simulation.step(neutral);
      const before = simulation.snapshot();
      expect(before.rootPosition.x).toBeCloseTo(1.25, 1);
      expect(before.rootPosition.z).toBeCloseTo(1.5, 1);

      const leftWrist = rotateVector(
        inverseQuaternion(before.rootRotation),
        subtract(before.links.left_wrist_yaw_link.position, before.rootPosition)
      );
      const plan: HumanoidMotionPlan = {
        id: "candidate-1",
        intent: "向前探索时抬起左臂观察身体通道协调",
        duration_seconds: 2.6,
        keyframes: [
          { at_seconds: 0, root_velocity: { forward_mps: 0.3, lateral_mps: 0 } },
          {
            at_seconds: 1,
            root_velocity: { forward_mps: 0.3, lateral_mps: 0 },
            left_hand: {
              frame: "pelvis",
              position: { ...leftWrist, y: leftWrist.y + 0.08 },
              tolerance_m: 0.05
            }
          },
          { at_seconds: 1.6, root_velocity: { forward_mps: 0, lateral_mps: 0 } },
          { at_seconds: 2.6, root_velocity: { forward_mps: 0, lateral_mps: 0 } }
        ]
      };
      const prepared = await prepareHumanoidMotion(simulation, plan, neutral);
      const validation = prepared.validation;
      const afterPreview = simulation.snapshot();

      expect(validation.feasible).toBe(true);
      expect(validation.evidence.simulatedSteps).toBe(130);
      expect(validation.evidence.travelledDistance).toBeGreaterThan(0.05);
      expect(validation.finalSnapshot.fallen).toBe(false);
      expect(afterPreview.simulatedTime).toBe(before.simulatedTime);
      expect(afterPreview.rootPosition).toEqual(before.rootPosition);

      expect(prepared.artifact).not.toBeNull();
      for (const frame of prepared.artifact!.frames) {
        await simulation.step(hydrateHumanoidReference(frame.reference));
      }
      const executed = simulation.snapshot();
      expect(executed.simulatedTime).toBeGreaterThan(before.simulatedTime);
      expect(Math.hypot(
        executed.rootPosition.x - before.rootPosition.x,
        executed.rootPosition.z - before.rootPosition.z
      )).toBeGreaterThan(0.05);
      expect(executed.fallen).toBe(false);
    } finally {
      await simulation.dispose();
    }
  }, 30_000);

  it("rejects a candidate when physical rollout reaches a world obstacle", async () => {
    const simulation = await HumanoidSimulation.create({
      solids: [{
        id: "blocking-wall",
        center: { x: 0, y: 0.9, z: 0.9 },
        size: { x: 4, y: 1.8, z: 0.12 }
      }]
    });
    try {
      const neutral = neutralHumanoidReference();
      for (let index = 0; index < 80; index += 1) await simulation.step(neutral);
      const prepared = await prepareHumanoidMotion(simulation, {
        id: "blocked-candidate",
        intent: "穿过前方墙体",
        duration_seconds: 4,
        keyframes: [
          { at_seconds: 0, root_velocity: { forward_mps: 0.5, lateral_mps: 0 } },
          { at_seconds: 4, root_velocity: { forward_mps: 0.5, lateral_mps: 0 } }
        ]
      }, neutral);
      const validation = prepared.validation;

      expect(validation.feasible).toBe(false);
      expect(validation.failures.some((failure) => (
        failure.code === "environment_contact" || failure.code === "fallen"
      ))).toBe(true);
    } finally {
      await simulation.dispose();
    }
  }, 30_000);

  it("authorizes only the exact body-object contact pair", () => {
    const snapshot = {
      contacts: [
        contact({ firstBody: "left_ankle_roll_link", normalY: 1 }),
        contact({ firstBody: "left_wrist_yaw_link", secondObject: "crate" }),
        contact({ firstBody: "right_wrist_yaw_link", secondObject: "crate" }),
        contact({ firstBody: "torso_link" })
      ]
    } as unknown as HumanoidSimulationSnapshot;

    expect(blockedHumanoidContacts(snapshot, [{
      body: "left_wrist_yaw_link",
      object_id: "crate",
      required: true
    }])).toEqual([
      expect.objectContaining({ body: "right_wrist_yaw_link", objectId: "crate" }),
      expect.objectContaining({ body: "torso_link", objectId: null })
    ]);
  });

  it("rejects duplicate, unknown, and physically unsatisfied contact constraints", async () => {
    expect(HumanoidMotionPlanSchema.safeParse({
      id: "duplicate-contacts",
      intent: "触碰箱体",
      duration_seconds: 0.1,
      contact_constraints: [
        { body: "left_wrist_yaw_link", object_id: "crate", required: true },
        { body: "left_wrist_yaw_link", object_id: "crate", required: false }
      ],
      keyframes: [{ at_seconds: 0 }, { at_seconds: 0.1 }]
    }).success).toBe(false);

    const simulation = await HumanoidSimulation.create({
      objects: [{
        id: "crate",
        center: { x: 4, y: 0.2, z: 4 },
        size: { x: 0.3, y: 0.3, z: 0.3 },
        mass: 0.2
      }]
    });
    try {
      const neutral = neutralHumanoidReference();
      const unknown = (await prepareHumanoidMotion(simulation, {
        id: "unknown-contact",
        intent: "触碰不存在的物体",
        duration_seconds: 0.1,
        contact_constraints: [{
          body: "left_wrist_yaw_link",
          object_id: "missing",
          required: true
        }],
        keyframes: [{ at_seconds: 0 }, { at_seconds: 0.1 }]
      }, neutral)).validation;
      expect(unknown.feasible).toBe(false);
      expect(unknown.failures).toContainEqual(expect.objectContaining({
        code: "unknown_contact_object"
      }));
      expect(unknown.evidence.simulatedSteps).toBe(0);

      const missing = (await prepareHumanoidMotion(simulation, {
        id: "missing-contact",
        intent: "保持站立但要求左手必须触碰远处箱体",
        duration_seconds: 0.1,
        contact_constraints: [{
          body: "left_wrist_yaw_link",
          object_id: "crate",
          required: true
        }],
        keyframes: [{ at_seconds: 0 }, { at_seconds: 0.1 }]
      }, neutral)).validation;
      expect(missing.feasible).toBe(false);
      expect(missing.failures).toContainEqual(expect.objectContaining({
        code: "required_contact_missing"
      }));
      expect(missing.evidence.satisfiedRequiredContacts).toEqual([]);
    } finally {
      await simulation.dispose();
    }
  }, 30_000);
});

function contact(input: {
  firstBody: NonNullable<HumanoidSimulationSnapshot["contacts"][number]["firstBody"]>;
  secondObject?: string;
  normalY?: number;
}): HumanoidSimulationSnapshot["contacts"][number] {
  return {
    position: { x: 0, y: 0, z: 0 },
    normal: { x: 0, y: input.normalY ?? 0, z: 1 },
    normalForce: 12,
    firstBody: input.firstBody,
    secondBody: null,
    firstObject: null,
    secondObject: input.secondObject ?? null
  };
}
