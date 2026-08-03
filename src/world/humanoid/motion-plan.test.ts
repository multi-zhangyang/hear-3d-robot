import { describe, expect, it } from "vitest";
import { ScenarioSchema } from "../../domain/schema.js";
import {
  blockedHumanoidContacts,
  HumanoidMotionCandidateBatchSchema,
  HumanoidMotionPlanSchema,
  occupiedHumanoidChannels,
  prepareHumanoidMotion,
  type HumanoidMotionGenerator,
  type HumanoidMotionPlan
} from "./motion-plan.js";
import {
  hydrateHumanoidReference,
  serializeHumanoidReference
} from "./motion-artifact.js";
import { neutralHumanoidReference } from "./reference.js";
import { inverseQuaternion, rotateVector, subtract } from "../geometry.js";
import {
  HumanoidSimulation,
  type HumanoidSimulationSnapshot
} from "./simulation.js";

const optionScenario = ScenarioSchema.parse({
  title: "Option precondition field",
  seed: 11,
  bounds: { width: 10, depth: 10 },
  visibility_radius: 6,
  robot: { x: 0, z: 0, yaw: 0 },
  obstacles: [],
  objects: [],
  zones: [],
  default_goal: {
    summary: "保持站立",
    predicates: [{
      type: "robot_at",
      target: { x: 0, y: 0, z: 0 },
      tolerance: 0.25
    }]
  }
});

describe("humanoid whole-body motion", () => {
  it("rejects label-only duplicate candidates at the planning boundary", () => {
    const duplicateBatch = HumanoidMotionCandidateBatchSchema.safeParse({
      objective: "比较全身运动候选",
      termination: {
        option_id: "candidate-distinctness",
        predicates: [{
          type: "root_near_point",
          target: { x: 0, y: 0.76, z: 0.1 },
          tolerance_m: 0.05
        }],
        stable_steps: 2
      },
      candidates: [
        {
          id: "candidate-one",
          intent: "第一个显示标签",
          duration_seconds: 0.4,
          contact_constraints: null,
          keyframes: [{ at_seconds: 0 }, { at_seconds: 0.4 }]
        },
        {
          id: "candidate-two",
          intent: "第二个显示标签",
          duration_seconds: 0.4,
          contact_constraints: [],
          keyframes: [
            { at_seconds: 0, root_pitch: null },
            { at_seconds: 0.4, root_pitch: null }
          ]
        }
      ]
    });

    expect(duplicateBatch.success).toBe(false);
    if (duplicateBatch.success) throw new Error("Duplicate candidates were accepted");
    expect(duplicateBatch.error.issues).toContainEqual(expect.objectContaining({
      path: ["candidates", 1],
      message: expect.stringContaining("id and intent labels")
    }));
  });

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

      const plan: HumanoidMotionPlan = {
        id: "candidate-1",
        intent: "向前探索时轻微转动躯干观察身体通道协调",
        duration_seconds: 2.6,
        keyframes: [
          { at_seconds: 0, root_velocity: { forward_mps: 0.3, lateral_mps: 0 } },
          {
            at_seconds: 1,
            root_velocity: { forward_mps: 0.3, lateral_mps: 0 },
            torso_yaw: 0.04
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

  it("rejects an unsatisfied option precondition before the first preview step", async () => {
    const simulation = await HumanoidSimulation.create();
    try {
      const neutral = neutralHumanoidReference();
      for (let index = 0; index < 80; index += 1) await simulation.step(neutral);
      const before = simulation.snapshot();
      const prepared = await prepareHumanoidMotion(
        simulation,
        {
          id: "preview-precondition-gate",
          intent: "只有入口条件成立才允许开始移动",
          duration_seconds: 0.2,
          keyframes: [
            {
              at_seconds: 0,
              root_velocity: { forward_mps: 0.3, lateral_mps: 0 }
            },
            {
              at_seconds: 0.2,
              root_velocity: { forward_mps: 0.3, lateral_mps: 0 }
            }
          ]
        },
        neutral,
        {
          motionOption: {
            scenario: optionScenario,
            contract: {
              option_id: "preview-precondition-option",
              predicates: [
                {
                  type: "root_near_point",
                  target: { ...before.rootPosition, x: before.rootPosition.x + 1 },
                  tolerance_m: 0.02
                },
                {
                  type: "root_near_point",
                  target: { ...before.rootPosition, z: before.rootPosition.z + 0.1 },
                  tolerance_m: 0.04
                }
              ],
              stable_steps: 1,
              phases: {
                precondition: {
                  condition: { op: "predicate", predicate_index: 0 },
                  stable_steps: 1
                },
                during: null,
                terminal: {
                  condition: { op: "predicate", predicate_index: 1 }
                }
              }
            }
          }
        }
      );
      const after = simulation.snapshot();

      expect(prepared.validation).toMatchObject({
        feasible: false,
        failures: [expect.objectContaining({
          code: "motion_constraint_violated",
          atSeconds: 0,
          message: "Motion option precondition is not satisfied before execution"
        })],
        evidence: { simulatedSteps: 0 }
      });
      expect(after.simulatedTime).toBe(before.simulatedTime);
      expect(after.rootPosition).toEqual(before.rootPosition);
      expect(after.joints).toEqual(before.joints);
    } finally {
      await simulation.dispose();
    }
  }, 30_000);

  it("accepts only ankle targets that YAHMP and MuJoCo track within tolerance", async () => {
    const simulation = await HumanoidSimulation.create();
    try {
      const neutral = neutralHumanoidReference();
      for (let index = 0; index < 100; index += 1) await simulation.step(neutral);
      const before = simulation.snapshot();
      const leftFootInPelvis = rotateVector(
        inverseQuaternion(before.links.pelvis.rotation),
        subtract(
          before.links.left_ankle_roll_link.position,
          before.links.pelvis.position
        )
      );
      const heldFoot = {
        frame: "pelvis" as const,
        position: leftFootInPelvis,
        tolerance_m: 0.04
      };
      const heldPlan: HumanoidMotionPlan = {
        id: "held-left-ankle-target",
        intent: "保持当前左踝任务空间位置",
        duration_seconds: 0.4,
        keyframes: [
          { at_seconds: 0 },
          { at_seconds: 0.4, left_foot: heldFoot }
        ]
      };

      expect(occupiedHumanoidChannels(heldPlan)).toEqual(["left_leg"]);
      const held = await prepareHumanoidMotion(simulation, heldPlan, neutral);
      expect(held.validation.feasible, JSON.stringify(held.validation.failures)).toBe(true);
      const achievedHeldFoot = rotateVector(
        inverseQuaternion(held.validation.finalSnapshot.links.pelvis.rotation),
        subtract(
          held.validation.finalSnapshot.links.left_ankle_roll_link.position,
          held.validation.finalSnapshot.links.pelvis.position
        )
      );
      expect(Math.hypot(
        achievedHeldFoot.x - heldFoot.position.x,
        achievedHeldFoot.y - heldFoot.position.y,
        achievedHeldFoot.z - heldFoot.position.z
      )).toBeLessThanOrEqual(heldFoot.tolerance_m);

      const liftedFoot = {
        ...heldFoot,
        position: {
          ...leftFootInPelvis,
          y: leftFootInPelvis.y + 0.12,
          z: leftFootInPelvis.z + 0.03
        }
      };
      const missedPlan: HumanoidMotionPlan = {
        id: "missed-left-ankle-target",
        intent: "根据当前平衡状态改变左踝连续目标",
        duration_seconds: 3,
        keyframes: [
          { at_seconds: 0 },
          { at_seconds: 0.8, root_roll: 0.15 },
          { at_seconds: 1.8, root_roll: 0.15, left_foot: liftedFoot },
          { at_seconds: 3, root_roll: 0.15, left_foot: liftedFoot }
        ]
      };

      expect(occupiedHumanoidChannels(missedPlan)).toEqual(["locomotion", "left_leg"]);
      const missed = await prepareHumanoidMotion(simulation, missedPlan, neutral);
      expect(missed.artifact).not.toBeNull();
      expect(missed.validation.feasible).toBe(false);
      expect(missed.validation.finalSnapshot.controller.implementation).toBe("yahmp_onnx");
      const trackingFailure = missed.validation.failures.find((failure) => (
        failure.code === "task_space_target_unmet"
      ));
      expect(trackingFailure).toMatchObject({
        code: "task_space_target_unmet",
        atSeconds: 1.8,
        taskSpaceTarget: {
          body: "left_ankle_roll_link",
          frame: "pelvis",
          target: liftedFoot.position,
          toleranceMeters: liftedFoot.tolerance_m,
          requestedAtSeconds: 1.8,
          observedAtSeconds: 1.8
        }
      });
      expect(trackingFailure!.taskSpaceTarget!.errorMeters).toBeGreaterThan(
        trackingFailure!.taskSpaceTarget!.toleranceMeters
      );
      expect(simulation.snapshot().rootPosition).toEqual(before.rootPosition);

      const unreachable = await prepareHumanoidMotion(simulation, {
        id: "unreachable-left-ankle-target",
        intent: "提交不可达连续目标以验证硬拒绝",
        duration_seconds: 0.4,
        keyframes: [
          { at_seconds: 0 },
          {
            at_seconds: 0.4,
            left_foot: {
              ...liftedFoot,
              position: { ...liftedFoot.position, y: liftedFoot.position.y + 2 }
            }
          }
        ]
      }, neutral);
      expect(unreachable.artifact).toBeNull();
      expect(unreachable.validation.failures).toContainEqual(
        expect.objectContaining({ code: "invalid_reference" })
      );
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

  it("restores physics after a motion generator attempts to mutate simulation state", async () => {
    const simulation = await HumanoidSimulation.create({});
    try {
      const neutral = neutralHumanoidReference();
      for (let index = 0; index < 80; index += 1) await simulation.step(neutral);
      const before = simulation.snapshot();
      const generator: HumanoidMotionGenerator = {
        descriptor: {
          protocol: "humanoid-motion-generator-v1",
          implementation: "mutating_contract_probe",
          motionClass: "generative_model",
          sampling: "stochastic"
        },
        async generate(input) {
          await input.simulation.step(input.baseline);
          return {
            version: 1,
            protocol: "humanoid-motion-v1",
            generator: "mutating_contract_probe",
            controlStepSeconds: input.controlStepSeconds,
            durationSeconds: input.plan.duration_seconds,
            frames: Array.from({ length: 5 }, (_, index) => ({
              atSeconds: (index + 1) * input.controlStepSeconds,
              reference: serializeHumanoidReference(input.baseline)
            }))
          };
        },
        async dispose() {}
      };

      const prepared = await prepareHumanoidMotion(
        simulation,
        {
          id: "generator-mutation-isolation",
          intent: "验证生成器与权威物理状态隔离",
          duration_seconds: 0.1,
          keyframes: [{ at_seconds: 0 }, { at_seconds: 0.1 }]
        },
        neutral,
        {},
        generator
      );
      const after = simulation.snapshot();

      expect(
        prepared.validation.feasible,
        JSON.stringify(prepared.validation.failures)
      ).toBe(true);
      expect(prepared.validation.evidence.simulatedSteps).toBe(5);
      expect(after.simulatedTime).toBe(before.simulatedTime);
      expect(after.rootPosition).toEqual(before.rootPosition);
      expect(after.joints).toEqual(before.joints);
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
