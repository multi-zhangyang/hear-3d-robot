import { describe, expect, it } from "vitest";
import { ScenarioSchema } from "../../domain/schema.js";
import {
  blockedHumanoidContacts,
  humanoidContactKey,
  humanoidEnvironmentContacts,
  humanoidGraspContactAuthorizationFailures,
  humanoidHandContactKey,
  humanoidHandSolidContactKey,
  humanoidSolidContactKey,
  HumanoidEnvironmentContactSchema,
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
import { createG1HandArtifactCommand, type G1HandCoordination } from "./hand-coordination.js";
import { applyHumanoidMotionArtifactFrame } from "./motion-frame-application.js";
import { HumanoidMotionExecution } from "./motion-execution.js";
import type { StoredHumanoidMotionPlan } from "./world-plan-state.js";
import { HUMANOID_JOINT_INDEX, HUMANOID_JOINT_NAMES } from "./model.js";
import { neutralHumanoidReference } from "./reference.js";
import {
  inverseQuaternion,
  multiplyQuaternion,
  normalizeQuaternion,
  quaternionAngularDistance,
  rotateVector,
  subtract
} from "../geometry.js";
import {
  HumanoidSimulation,
  type HumanoidSimulationSnapshot
} from "./simulation.js";
import {
  humanoidEndEffectorJointIndexes,
  humanoidEndEffectorTrackingJointIndexes
} from "./task-space-targets.js";

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
  it("keeps legacy body contact identity while isolating exact hand surfaces", () => {
    const bodyConstraint = {
      body: "left_wrist_yaw_link" as const,
      object_id: "crate",
      required: true
    };
    const parsed = HumanoidMotionPlanSchema.parse({
      id: "legacy-contact-shape",
      intent: "验证兼容的身体接触合同",
      duration_seconds: 0.1,
      contact_constraints: [bodyConstraint],
      keyframes: [{ at_seconds: 0 }, { at_seconds: 0.1 }]
    });

    expect(parsed.contact_constraints).toEqual([bodyConstraint]);
    expect(humanoidContactKey("left_wrist_yaw_link", "crate"))
      .toBe("left_wrist_yaw_link\u0000crate");
    expect(humanoidHandContactKey("left_hand_palm_link", "crate"))
      .toBe("hand_surface\u0000left_hand_palm_link\u0000crate");
    expect(humanoidSolidContactKey("left_wrist_yaw_link", "block-a"))
      .toBe("left_wrist_yaw_link\u0000solid\u0000block-a");
    expect(humanoidHandSolidContactKey("left_hand_palm_link", "block-a"))
      .toBe("hand_surface\u0000left_hand_palm_link\u0000solid\u0000block-a");
    expect(HumanoidMotionPlanSchema.safeParse({
      ...parsed,
      contact_constraints: [{
        ...bodyConstraint,
        hand_surface: "left_hand_palm_link"
      }]
    }).success).toBe(false);
    expect(HumanoidMotionPlanSchema.safeParse({
      ...parsed,
      contact_constraints: [{ object_id: "crate", required: true }]
    }).success).toBe(false);
    expect(HumanoidEnvironmentContactSchema.safeParse({
      body: "left_wrist_yaw_link",
      handSurface: "left_hand_palm_link",
      objectId: "crate",
      normalForce: 4
    }).success).toBe(false);
    expect(HumanoidEnvironmentContactSchema.safeParse({
      objectId: "crate",
      normalForce: 4
    }).success).toBe(false);
    expect(HumanoidEnvironmentContactSchema.safeParse({
      body: "left_wrist_yaw_link",
      objectId: "crate",
      solidId: "block-a",
      normalForce: 4
    }).success).toBe(false);
  });

  it("requires explicit multi-surface authorization for grasp candidates", () => {
    const batch = {
      objective: "用左手完成经物理验证的抓取",
      termination: {
        option_id: "verified-grasp",
        predicates: [{
          type: "grasp_verified" as const,
          object_id: "crate",
          hand: "left" as const,
          grasp_contract_sha256: "a".repeat(64)
        }],
        stable_steps: 3
      },
      candidates: [
        {
          id: "palm-only",
          intent: "单掌面尝试",
          duration_seconds: 0.2,
          contact_constraints: [{
            hand_surface: "left_hand_palm_link" as const,
            object_id: "crate",
            required: false
          }],
          keyframes: [{ at_seconds: 0 }, { at_seconds: 0.2 }]
        },
        {
          id: "wrong-object",
          intent: "授权了错误物体",
          duration_seconds: 0.2,
          contact_constraints: [
            {
              hand_surface: "left_hand_palm_link" as const,
              object_id: "other",
              required: false
            },
            {
              hand_surface: "left_hand_index_1_link" as const,
              object_id: "other",
              required: false
            }
          ],
          keyframes: [
            { at_seconds: 0 },
            { at_seconds: 0.2, torso_yaw: 0.1 }
          ]
        }
      ]
    };
    const rejected = HumanoidMotionCandidateBatchSchema.safeParse(batch);
    expect(rejected.success).toBe(false);
    if (rejected.success) throw new Error("Under-authorized grasp was accepted");
    expect(rejected.error.issues.filter((issue) => (
      issue.message.includes("at least two distinct required hand_object surfaces")
    ))).toHaveLength(2);

    const authorized = HumanoidMotionCandidateBatchSchema.parse({
      ...batch,
      candidates: batch.candidates.map((candidate, index) => ({
        ...candidate,
        contact_constraints: [
          {
            hand_surface: "left_hand_palm_link" as const,
            object_id: "crate",
            required: false
          },
          {
            hand_surface: "left_hand_index_1_link" as const,
            object_id: "crate",
            required: false
          }
        ],
        keyframes: [
          { at_seconds: 0 },
          { at_seconds: 0.2, torso_yaw: index === 0 ? 0 : 0.1 }
        ]
      }))
    });
    expect(humanoidGraspContactAuthorizationFailures(
      authorized,
      () => 3
    )).toEqual([
      expect.objectContaining({
        candidateIndex: 0,
        minimumDistinctContactSurfaces: 3,
        authorizedContactSurfaces: [
          "left_hand_index_1_link",
          "left_hand_palm_link"
        ]
      }),
      expect.objectContaining({
        candidateIndex: 1,
        minimumDistinctContactSurfaces: 3
      })
    ]);
  });

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

  it("does not treat opposite quaternion signs as distinct motion candidates", () => {
    const pose = {
      position: { x: 0.2, y: 0.3, z: 0.4 },
      frame: "pelvis" as const,
      tolerance_m: 0.05,
      orientation_tolerance_rad: 0.1
    };
    const parsed = HumanoidMotionCandidateBatchSchema.safeParse({
      objective: "比较右腕位姿候选",
      termination: {
        option_id: "same-physical-pose",
        predicates: [{
          type: "root_near_point",
          target: { x: 0, y: 0.793, z: 0 },
          tolerance_m: 0.2
        }],
        stable_steps: 2
      },
      candidates: [
        {
          id: "positive-q",
          intent: "正号四元数",
          duration_seconds: 0.4,
          keyframes: [
            { at_seconds: 0 },
            {
              at_seconds: 0.4,
              right_hand: {
                ...pose,
                orientation: { x: 0, y: 0, z: 0.2, w: 0.98 }
              }
            }
          ]
        },
        {
          id: "negative-q",
          intent: "负号四元数",
          duration_seconds: 0.4,
          keyframes: [
            { at_seconds: 0 },
            {
              at_seconds: 0.4,
              right_hand: {
                ...pose,
                orientation: { x: 0, y: 0, z: -0.2, w: -0.98 }
              }
            }
          ]
        }
      ]
    });

    expect(parsed.success).toBe(false);
    if (parsed.success) throw new Error("Equivalent pose candidates were accepted");
    expect(parsed.error.issues).toContainEqual(expect.objectContaining({
      path: ["candidates", 1],
      message: expect.stringContaining("duplicates candidate 1")
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
      expect(trackingFailure, JSON.stringify(missed.validation.failures)).toMatchObject({
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
        expect.objectContaining({
          code: "invalid_reference",
          atSeconds: 0.4,
          message: expect.stringContaining("keyframe at 0.4s")
        })
      );
    } finally {
      await simulation.dispose();
    }
  }, 30_000);

  it("executes an interpolated right-hand target through the real controller and physics", async () => {
    const simulation = await HumanoidSimulation.create();
    try {
      const neutral = neutralHumanoidReference();
      for (let index = 0; index < 100; index += 1) await simulation.step(neutral);
      const before = simulation.snapshot();
      const initialWristWorld = before.links.right_wrist_yaw_link.position;
      const initialWristInPelvis = rotateVector(
        inverseQuaternion(before.links.pelvis.rotation),
        subtract(initialWristWorld, before.links.pelvis.position)
      );
      const target = {
        ...initialWristInPelvis,
        y: initialWristInPelvis.y + 0.08,
        z: initialWristInPelvis.z + 0.06
      };
      const tolerance = 0.06;
      const initialError = Math.hypot(
        initialWristInPelvis.x - target.x,
        initialWristInPelvis.y - target.y,
        initialWristInPelvis.z - target.z
      );
      expect(initialError).toBeGreaterThan(tolerance);

      const prepared = await prepareHumanoidMotion(simulation, {
        id: "physical-right-hand-target",
        intent: "抬起右手到新的任务空间目标",
        duration_seconds: 1.6,
        keyframes: [
          { at_seconds: 0 },
          {
            at_seconds: 1.6,
            right_hand: {
              frame: "pelvis",
              position: target,
              tolerance_m: tolerance
            }
          }
        ]
      }, neutral);

      expect(prepared.artifact).not.toBeNull();
      expect(
        prepared.validation.feasible,
        JSON.stringify(prepared.validation.failures)
      ).toBe(true);
      expect(prepared.validation.failures).toEqual([]);
      const frames = prepared.artifact!.frames;
      expect(frames.length).toBeGreaterThan(0);
      const rightArmIndexes = new Set(
        humanoidEndEffectorJointIndexes("right_wrist_yaw_link")
      );
      const rightHandTrackingIndexes = new Set(
        humanoidEndEffectorTrackingJointIndexes("right_wrist_yaw_link")
      );
      const wholeBodyReachIndexes = new Set(
        humanoidEndEffectorJointIndexes(
          "right_wrist_yaw_link",
          "whole_body_reach"
        )
      );
      expect(rightArmIndexes.size).toBe(6);
      expect(rightHandTrackingIndexes.size).toBe(6);
      expect(wholeBodyReachIndexes.size).toBe(9);
      expect([...rightArmIndexes].every((index) => rightHandTrackingIndexes.has(index))).toBe(true);
      expect([
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint"
      ].every((joint) => !rightHandTrackingIndexes.has(
        HUMANOID_JOINT_INDEX.get(joint)!
      ))).toBe(true);
      expect([
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint"
      ].every((joint) => wholeBodyReachIndexes.has(
        HUMANOID_JOINT_INDEX.get(joint)!
      ))).toBe(true);
      const middle = hydrateHumanoidReference(frames[Math.floor(frames.length / 2)]!.reference);
      const finalReference = hydrateHumanoidReference(frames.at(-1)!.reference);
      for (let index = 0; index < HUMANOID_JOINT_NAMES.length; index += 1) {
        if (rightHandTrackingIndexes.has(index)) {
          expect(middle.jointTrackingWeights[index]).toBeGreaterThan(0);
          expect(middle.jointTrackingWeights[index]).toBeLessThan(1);
          expect(finalReference.jointTrackingWeights[index]).toBe(1);
        } else {
          expect(middle.jointTrackingWeights[index]).toBe(0);
          expect(finalReference.jointTrackingWeights[index]).toBe(0);
        }
      }
      expect(finalReference.jointTrackingWeights[
        HUMANOID_JOINT_INDEX.get("right_wrist_yaw_joint")!
      ]).toBe(0);

      let executedFrames = 0;
      for (const frame of frames) {
        await applyHumanoidMotionArtifactFrame(simulation, frame);
        executedFrames += 1;
      }
      const after = simulation.snapshot();
      const finalWristWorld = after.links.right_wrist_yaw_link.position;
      const finalWristInPelvis = rotateVector(
        inverseQuaternion(after.links.pelvis.rotation),
        subtract(finalWristWorld, after.links.pelvis.position)
      );
      const finalError = Math.hypot(
        finalWristInPelvis.x - target.x,
        finalWristInPelvis.y - target.y,
        finalWristInPelvis.z - target.z
      );

      expect(executedFrames).toBe(frames.length);
      expect(executedFrames).toBeGreaterThan(0);
      expect(Math.hypot(
        finalWristWorld.x - initialWristWorld.x,
        finalWristWorld.y - initialWristWorld.y,
        finalWristWorld.z - initialWristWorld.z
      )).toBeGreaterThan(0.03);
      expect(finalError).toBeLessThan(initialError);
      expect(finalError).toBeLessThanOrEqual(tolerance);
      expect(after.fallen).toBe(false);
      expect(after.balance.upright).toBeGreaterThan(0.9);
      expect(after.feet.left.touching).toBe(true);
      expect(after.feet.right.touching).toBe(true);
      expect(after.balance.support).toBe("double");
    } finally {
      await simulation.dispose();
    }
  }, 30_000);

  it("keeps a world-anchored wrist target closed-loop while the pelvis moves", async () => {
    const simulation = await HumanoidSimulation.create();
    try {
      const neutral = neutralHumanoidReference();
      for (let index = 0; index < 100; index += 1) await simulation.step(neutral);
      const before = simulation.snapshot();
      const target = { ...before.links.right_wrist_yaw_link.position };
      const prepared = await prepareHumanoidMotion(simulation, {
        id: "world-anchored-right-wrist",
        intent: "移动骨盆时保持右腕世界位置",
        duration_seconds: 2,
        keyframes: [
          {
            at_seconds: 0,
            root_velocity: { forward_mps: 0.2, lateral_mps: 0 }
          },
          {
            at_seconds: 2,
            root_velocity: { forward_mps: 0.2, lateral_mps: 0 },
            right_hand: {
              frame: "world",
              position: target,
              tolerance_m: 0.03
            }
          }
        ]
      }, neutral);

      expect(
        prepared.validation.feasible,
        JSON.stringify(prepared.validation.failures)
      ).toBe(true);
      expect(prepared.artifact?.taskSpaceServo).toMatchObject({
        protocol: "humanoid-task-space-servo-v1"
      });
      expect(prepared.artifact?.frames.every((frame) => (
        frame.taskSpaceTargets?.[0]?.body === "right_wrist_yaw_link"
      ))).toBe(true);

      let final = before;
      for (const frame of prepared.artifact!.frames) {
        final = (await applyHumanoidMotionArtifactFrame(simulation, frame)).snapshot;
      }
      const pelvisTravel = Math.hypot(
        final.links.pelvis.position.x - before.links.pelvis.position.x,
        final.links.pelvis.position.z - before.links.pelvis.position.z
      );
      const wristError = Math.hypot(
        final.links.right_wrist_yaw_link.position.x - target.x,
        final.links.right_wrist_yaw_link.position.y - target.y,
        final.links.right_wrist_yaw_link.position.z - target.z
      );
      expect(pelvisTravel).toBeGreaterThan(0.035);
      expect(wristError).toBeLessThanOrEqual(0.03);
      expect(final.fallen).toBe(false);
    } finally {
      await simulation.dispose();
    }
  }, 30_000);

  it("physically tracks a wrist orientation without direct joint commands", async () => {
    const simulation = await HumanoidSimulation.create();
    try {
      const neutral = neutralHumanoidReference();
      for (let index = 0; index < 100; index += 1) await simulation.step(neutral);
      const before = simulation.snapshot();
      const wristPosition = rotateVector(
        inverseQuaternion(before.links.pelvis.rotation),
        subtract(
          before.links.right_wrist_yaw_link.position,
          before.links.pelvis.position
        )
      );
      const wristOrientation = normalizeQuaternion(multiplyQuaternion(
        inverseQuaternion(before.links.pelvis.rotation),
        before.links.right_wrist_yaw_link.rotation
      ));
      const targetOrientation = normalizeQuaternion(multiplyQuaternion(
        wristOrientation,
        { x: 0, y: 0, z: Math.sin(0.06), w: Math.cos(0.06) }
      ));
      const initialOrientationError = quaternionAngularDistance(
        wristOrientation,
        targetOrientation
      );
      const prepared = await prepareHumanoidMotion(simulation, {
        id: "physical-right-wrist-orientation",
        intent: "连续调整右腕朝向",
        duration_seconds: 0.8,
        keyframes: [
          { at_seconds: 0 },
          {
            at_seconds: 0.8,
            right_hand: {
              frame: "pelvis",
              position: wristPosition,
              tolerance_m: 0.04,
              orientation: targetOrientation,
              orientation_tolerance_rad: 0.07
            }
          }
        ]
      }, neutral);

      expect(initialOrientationError).toBeGreaterThan(0.07);
      expect(
        prepared.validation.feasible,
        JSON.stringify(prepared.validation.failures)
      ).toBe(true);
      const finalReference = hydrateHumanoidReference(
        prepared.artifact!.frames.at(-1)!.reference
      );
      expect(finalReference.jointTrackingWeights[
        HUMANOID_JOINT_INDEX.get("right_wrist_yaw_joint")!
      ]).toBe(1);

      for (const frame of prepared.artifact!.frames) {
        await applyHumanoidMotionArtifactFrame(simulation, frame);
      }
      const after = simulation.snapshot();
      const achievedOrientation = normalizeQuaternion(multiplyQuaternion(
        inverseQuaternion(after.links.pelvis.rotation),
        after.links.right_wrist_yaw_link.rotation
      ));
      const finalOrientationError = quaternionAngularDistance(
        achievedOrientation,
        targetOrientation
      );
      expect(finalOrientationError).toBeLessThan(initialOrientationError);
      expect(finalOrientationError).toBeLessThanOrEqual(0.07);
      expect(after.fallen).toBe(false);
    } finally {
      await simulation.dispose();
    }
  }, 30_000);

  it("rejects an end-effector orientation without its physical tolerance", () => {
    expect(HumanoidMotionPlanSchema.safeParse({
      id: "incomplete-pose",
      intent: "提交不完整末端姿态",
      duration_seconds: 0.2,
      keyframes: [
        { at_seconds: 0 },
        {
          at_seconds: 0.2,
          right_hand: {
            frame: "world",
            position: { x: 0, y: 1, z: 0 },
            tolerance_m: 0.05,
            orientation: { x: 0, y: 0, z: 0, w: 1 }
          }
        }
      ]
    }).success).toBe(false);
  });

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

  it("carries model-authored hand coordination through preview and real execution", async () => {
    const simulation = await HumanoidSimulation.create();
    try {
      const neutral = neutralHumanoidReference();
      for (let index = 0; index < 80; index += 1) await simulation.step(neutral);
      const plan: HumanoidMotionPlan = {
        id: "continuous-hand-coordination",
        intent: "连续改变双手协同关节目标",
        duration_seconds: 0.2,
        keyframes: [
          { at_seconds: 0, hand_coordination: OPEN_HAND_COORDINATION },
          { at_seconds: 0.2, hand_coordination: CURLED_HAND_COORDINATION }
        ]
      };

      expect(occupiedHumanoidChannels(plan)).toEqual(["left_arm", "right_arm"]);
      const prepared = await prepareHumanoidMotion(simulation, plan, neutral);
      expect(prepared.validation.feasible, JSON.stringify(prepared.validation.failures))
        .toBe(true);
      expect(prepared.artifact).toMatchObject({
        version: 2,
        protocol: "humanoid-motion-v2"
      });
      if (!prepared.artifact || prepared.artifact.version !== 2) {
        throw new Error("Expected a hand-coordinated motion artifact");
      }
      expect(prepared.artifact.frames).toHaveLength(10);
      expect(prepared.artifact.frames.at(-1)!.handCommand.coordination)
        .toEqual(CURLED_HAND_COORDINATION);
      expect(prepared.artifact.frames[4]!.handCommand.coordination.left.index_curl)
        .toBeCloseTo(0.3, 12);

      const stored: StoredHumanoidMotionPlan = {
        skillCallIdentity: null,
        plan,
        artifact: prepared.artifact,
        rollout: null,
        retainTerminalJointTracking: false,
        createdRevision: 0,
        validatedRevision: 0,
        validatedStateSha256: "a".repeat(64),
        expiresRevision: 100,
        intentSha256: "b".repeat(64),
        revalidationCount: 0,
        terminal: null,
        option: null,
        progress: {
          nextFrameIndex: 0,
          satisfiedContactKeys: [],
          driftStreak: 0,
          lastDrift: null,
          failure: null
        }
      };
      const execution = new HumanoidMotionExecution({
        stored,
        reference: neutral,
        detectorInput: () => {
          throw new Error("A motion without an Option must not invoke its detector");
        }
      });
      while (!execution.done) await execution.step(simulation);
      expect(execution.result().failures).toEqual([]);
      expect(stored.progress.nextFrameIndex).toBe(prepared.artifact.frames.length);
      const executed = simulation.snapshot();
      expect(executed.hands.joints).toEqual(
        prepared.validation.finalSnapshot.hands.joints
      );

      simulation.setHandJointTargets({ left_hand_index_0_joint: -0.37 });
      const heldTarget = simulation.snapshot().hands.joints.left_hand_index_0_joint.target;
      const bodyOnly = await prepareHumanoidMotion(simulation, {
        id: "body-only-preserves-hands",
        intent: "保持当前手指目标",
        duration_seconds: 0.1,
        keyframes: [{ at_seconds: 0 }, { at_seconds: 0.1 }]
      }, neutral);
      expect(bodyOnly.artifact).toMatchObject({
        version: 1,
        protocol: "humanoid-motion-v1"
      });
      expect(simulation.snapshot().hands.joints.left_hand_index_0_joint.target)
        .toBe(heldTarget);
      if (!bodyOnly.artifact) throw new Error("Expected a body-only artifact");
      for (const frame of bodyOnly.artifact.frames) {
        await applyHumanoidMotionArtifactFrame(simulation, frame);
      }
      expect(simulation.snapshot().hands.joints.left_hand_index_0_joint.target)
        .toBe(heldTarget);
    } finally {
      await simulation.dispose();
    }
  }, 30_000);

  it("requires time-zero hand state and rejects generator-injected hand behavior", async () => {
    expect(HumanoidMotionPlanSchema.safeParse({
      id: "missing-initial-hand-state",
      intent: "后续才声明手部状态",
      duration_seconds: 0.1,
      keyframes: [
        { at_seconds: 0 },
        { at_seconds: 0.1, hand_coordination: CURLED_HAND_COORDINATION }
      ]
    }).success).toBe(false);

    const simulation = await HumanoidSimulation.create();
    try {
      const neutral = neutralHumanoidReference();
      const handPlan: HumanoidMotionPlan = {
        id: "hand-contract",
        intent: "保持显式双手协同状态",
        duration_seconds: 0.1,
        keyframes: [
          { at_seconds: 0, hand_coordination: OPEN_HAND_COORDINATION },
          { at_seconds: 0.1, hand_coordination: OPEN_HAND_COORDINATION }
        ]
      };
      const bodyPlan: HumanoidMotionPlan = {
        id: "body-contract",
        intent: "不声明手部动作",
        duration_seconds: 0.1,
        keyframes: [{ at_seconds: 0 }, { at_seconds: 0.1 }]
      };

      const omitted = await prepareHumanoidMotion(
        simulation,
        handPlan,
        neutral,
        {},
        handContractProbe("omit")
      );
      expect(omitted.artifact).toBeNull();
      expect(omitted.validation.failures[0]?.message)
        .toMatch(/requires a version 2 motion artifact/);

      const injected = await prepareHumanoidMotion(
        simulation,
        bodyPlan,
        neutral,
        {},
        handContractProbe("inject")
      );
      expect(injected.artifact).toBeNull();
      expect(injected.validation.failures[0]?.message)
        .toMatch(/cannot add hand commands/);

      const tampered = await prepareHumanoidMotion(
        simulation,
        handPlan,
        neutral,
        {},
        handContractProbe("tamper")
      );
      expect(tampered.artifact).toBeNull();
      expect(tampered.validation.failures[0]?.message)
        .toMatch(/does not match its model plan/);
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

  it("classifies an exact hand surface before its wrist body", () => {
    const snapshot = {
      contacts: [{
        position: { x: 0, y: 0.8, z: 0.3 },
        normal: { x: 0, y: 0, z: 1 },
        normalForce: 9,
        firstBody: "left_wrist_yaw_link",
        secondBody: null,
        firstObject: null,
        secondObject: "crate",
        firstHandLink: "left_hand_palm_link",
        secondHandLink: null
      }]
    } as unknown as HumanoidSimulationSnapshot;

    expect(humanoidEnvironmentContacts(snapshot)).toEqual([{
      handSurface: "left_hand_palm_link",
      objectId: "crate",
      solidId: null,
      normalForce: 9
    }]);
    expect(blockedHumanoidContacts(snapshot, [{
      body: "left_wrist_yaw_link",
      object_id: "crate",
      required: false
    }])).toEqual([{
      handSurface: "left_hand_palm_link",
      objectId: "crate",
      solidId: null,
      normalForce: 9
    }]);
    expect(blockedHumanoidContacts(snapshot, [{
      hand_surface: "left_hand_palm_link",
      object_id: "crate",
      required: false
    }])).toEqual([]);
    expect(blockedHumanoidContacts(snapshot, [{
      hand_surface: "left_hand_index_1_link",
      object_id: "crate",
      required: false
    }])).toEqual([expect.objectContaining({
      handSurface: "left_hand_palm_link"
    })]);
  });

  it("preserves exact static solid identity for body and hand authorization", () => {
    const bodySnapshot = {
      contacts: [{
        normal: { x: 0, y: 0, z: 1 },
        normalForce: 8,
        firstBody: null,
        secondBody: "left_wrist_yaw_link",
        firstObject: null,
        secondObject: null,
        firstSolid: "block-a",
        secondSolid: null,
        firstHandLink: null,
        secondHandLink: null
      }]
    } as unknown as HumanoidSimulationSnapshot;
    expect(humanoidEnvironmentContacts(bodySnapshot)).toEqual([{
      body: "left_wrist_yaw_link",
      objectId: null,
      solidId: "block-a",
      normalForce: 8
    }]);
    expect(blockedHumanoidContacts(bodySnapshot, [{
      body: "left_wrist_yaw_link",
      solid_id: "block-a",
      required: true
    }])).toEqual([]);

    const handSnapshot = {
      contacts: [{
        normal: { x: 0, y: 0, z: 1 },
        normalForce: 9,
        firstBody: null,
        secondBody: "left_wrist_yaw_link",
        firstObject: null,
        secondObject: null,
        firstSolid: "block-a",
        secondSolid: null,
        firstHandLink: null,
        secondHandLink: "left_hand_palm_link"
      }]
    } as unknown as HumanoidSimulationSnapshot;
    expect(humanoidEnvironmentContacts(handSnapshot)).toEqual([{
      handSurface: "left_hand_palm_link",
      objectId: null,
      solidId: "block-a",
      normalForce: 9
    }]);
    expect(blockedHumanoidContacts(handSnapshot, [{
      hand_surface: "left_hand_palm_link",
      solid_id: "block-a",
      required: true
    }])).toEqual([]);
    expect(blockedHumanoidContacts(handSnapshot, [{
      body: "left_wrist_yaw_link",
      solid_id: "block-a",
      required: true
    }])).toHaveLength(1);
  });

  it("does not misclassify hand-to-body self contact as environment contact", () => {
    const snapshot = {
      contacts: [{
        position: { x: 0, y: 0.7, z: 0.3 },
        normal: { x: 0, y: 0, z: 1 },
        normalForce: 6,
        firstBody: null,
        secondBody: "right_hip_roll_link",
        firstObject: null,
        secondObject: null,
        firstHandLink: "right_hand_thumb_2_link",
        secondHandLink: null
      }]
    } as unknown as HumanoidSimulationSnapshot;

    expect(humanoidEnvironmentContacts(snapshot)).toEqual([]);
    expect(blockedHumanoidContacts(snapshot, [])).toEqual([]);
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

const OPEN_HAND_COORDINATION: G1HandCoordination = {
  left: {
    thumb_opposition: 0,
    thumb_curl: 0,
    index_curl: 0,
    middle_curl: 0
  },
  right: {
    thumb_opposition: 0,
    thumb_curl: 0,
    index_curl: 0,
    middle_curl: 0
  }
};

const CURLED_HAND_COORDINATION: G1HandCoordination = {
  left: {
    thumb_opposition: 0.4,
    thumb_curl: 0.5,
    index_curl: 0.6,
    middle_curl: 0.7
  },
  right: {
    thumb_opposition: 0.3,
    thumb_curl: 0.4,
    index_curl: 0.5,
    middle_curl: 0.6
  }
};

function handContractProbe(
  mode: "omit" | "inject" | "tamper"
): HumanoidMotionGenerator {
  return {
    descriptor: {
      protocol: "humanoid-motion-generator-v1",
      implementation: `hand_contract_${mode}`,
      motionClass: "generative_model",
      sampling: "stochastic"
    },
    async generate(input) {
      const frames = Math.ceil(
        input.plan.duration_seconds / input.controlStepSeconds
      );
      if (mode === "omit") {
        return {
          version: 1,
          protocol: "humanoid-motion-v1",
          generator: `hand_contract_${mode}`,
          controlStepSeconds: input.controlStepSeconds,
          durationSeconds: input.plan.duration_seconds,
          frames: Array.from({ length: frames }, (_, index) => ({
            atSeconds: Math.min(
              (index + 1) * input.controlStepSeconds,
              input.plan.duration_seconds
            ),
            reference: serializeHumanoidReference(input.baseline)
          }))
        };
      }
      const coordination = mode === "tamper"
        ? CURLED_HAND_COORDINATION
        : OPEN_HAND_COORDINATION;
      return {
        version: 2,
        protocol: "humanoid-motion-v2",
        generator: `hand_contract_${mode}`,
        controlStepSeconds: input.controlStepSeconds,
        durationSeconds: input.plan.duration_seconds,
        frames: Array.from({ length: frames }, (_, index) => ({
          atSeconds: Math.min(
            (index + 1) * input.controlStepSeconds,
            input.plan.duration_seconds
          ),
          reference: serializeHumanoidReference(input.baseline),
          handCommand: createG1HandArtifactCommand(coordination)
        }))
      };
    },
    async dispose() {}
  };
}
