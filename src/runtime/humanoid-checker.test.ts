import { describe, expect, it } from "vitest";
import { GoalSchema, ScenarioSchema } from "../domain/schema.js";
import { HumanoidWorld } from "../world/humanoid/world.js";
import {
  advanceHumanoidGoal,
  assertHumanoidGoalProgressIntegrity,
  assertHumanoidGoalSupported,
  checkHumanoidGoal,
  createHumanoidGoalProgress,
  inspectHumanoidGoal
} from "./humanoid-checker.js";
import { add, multiplyQuaternion, rotateVector } from "../world/geometry.js";
import type { HumanoidGraspAssessment } from "../world/humanoid/grasp-tracker.js";

const scenario = ScenarioSchema.parse({
  title: "人形目标检查场",
  seed: 41,
  bounds: { width: 12, depth: 12 },
  visibility_radius: 7,
  robot: { x: 2, z: 2, yaw: 0 },
  obstacles: [],
  objects: [{
    id: "crate",
    kind: "crate",
    color: "#b87942",
    position: { x: 8, y: 0.15, z: 8 },
    size: { x: 0.3, y: 0.3, z: 0.3 },
    portable: true
  }],
  zones: [{
    id: "arrival",
    color: "#55a88b",
    center: { x: 8, y: 0.01, z: 8 },
    size: { x: 2, y: 0.02, z: 2 }
  }],
  default_goal: {
    summary: "进入区域",
    predicates: [{ type: "robot_in_zone", zone_id: "arrival", tolerance: 0.1 }]
  }
});

describe("humanoid goal checker", () => {
  it("checks robot and physical object positions against one authoritative frame", async () => {
    const world = await HumanoidWorld.create(scenario);
    try {
      const snapshot = world.snapshot();
      snapshot.robot.rootPosition = { x: 8, y: snapshot.robot.rootPosition.y, z: 8 };
      snapshot.robot.objects.crate!.position = { x: 8, y: 0.15, z: 8 };
      const goal = GoalSchema.parse({
        summary: "机器人与箱体到达目标",
        predicates: [
          { type: "robot_at", target: { x: 8, y: 0, z: 8 }, tolerance: 0.1 },
          { type: "robot_in_zone", zone_id: "arrival", tolerance: 0.05 },
          {
            type: "object_at",
            object_id: "crate",
            target: { x: 8, y: 0.15, z: 8 },
            tolerance: 0.05
          },
          {
            type: "object_in_zone",
            object_id: "crate",
            zone_id: "arrival",
            expected: true,
            tolerance: 0.03
          }
        ]
      });

      assertHumanoidGoalSupported(goal, scenario);
      const result = checkHumanoidGoal(goal, scenario, snapshot);
      expect(result.success).toBe(true);
      expect(result.worldFrame).toBe(snapshot.frame);
      expect(result.worldRevision).toBe(snapshot.worldRevision);
      expect(result.checks).toHaveLength(4);
      expect(result.checks.every((check) => check.passed)).toBe(true);

      snapshot.robot.objects.crate!.position = { x: 5, y: 0.15, z: 5 };
      const moved = checkHumanoidGoal(goal, scenario, snapshot);
      expect(moved.success).toBe(false);
      expect(moved.checks.slice(2).every((check) => !check.passed)).toBe(true);
    } finally {
      await world.dispose();
    }
  }, 30_000);

  it("accepts placement only when the current frame proves zone, release and non-humanoid support", async () => {
    const world = await HumanoidWorld.create(scenario);
    try {
      const snapshot = world.snapshot();
      snapshot.frame += 1;
      snapshot.worldRevision += 1;
      const object = snapshot.robot.objects.crate!;
      object.position = { x: 8, y: 0.17, z: 8 };
      object.linearVelocity = { x: 0.005, y: 0, z: -0.004 };
      object.angularVelocity = { x: 0, y: 0.02, z: 0 };
      snapshot.robot.contacts = [{
        position: { x: 8, y: 0.02, z: 8 },
        normal: { x: 0, y: 1, z: 0 },
        normalForce: 3,
        firstBody: null,
        secondBody: null,
        firstObject: null,
        secondObject: "crate",
        firstHandLink: null,
        secondHandLink: null
      }];
      snapshot.robot.contactCount = snapshot.robot.contacts.length;
      snapshot.grasp = {
        contractSha256: "a".repeat(64),
        assessments: [
          graspAssessment(snapshot.frame, "left", false),
          graspAssessment(snapshot.frame, "right", false)
        ]
      };
      const goal = GoalSchema.parse({
        summary: "将箱体稳放到目标区域",
        predicates: [{
          type: "object_placed",
          object_id: "crate",
          zone_id: "arrival",
          tolerance: 0.03
        }]
      });

      assertHumanoidGoalSupported(goal, scenario);
      const beforeCheck = structuredClone(snapshot);
      expect(checkHumanoidGoal(goal, scenario, snapshot)).toMatchObject({
        success: true,
        checks: [{
          name: "1:object_placed",
          passed: true,
          actual: {
            object_id: "crate",
            inside: true,
            world_frame: snapshot.frame,
            grasp: {
              evidence_complete: true,
              assessed_hands: ["left", "right"],
              verified_hands: []
            },
            settled_support: {
              protocol: "humanoid-object-settled-support-assessment-v1",
              objectId: "crate",
              status: "satisfied",
              reason: "object_settled_on_support",
              evidence: {
                supportContactCount: 1,
                totalUpwardSupportForceN: 3,
                linearSpeedMps: expect.closeTo(Math.hypot(0.005, 0.004), 12),
                angularSpeedRadps: 0.02
              }
            }
          }
        }]
      });
      expect(snapshot).toEqual(beforeCheck);

      snapshot.grasp.assessments = [
        graspAssessment(snapshot.frame, "left", true),
        graspAssessment(snapshot.frame, "right", false)
      ];
      expect(checkHumanoidGoal(goal, scenario, snapshot)).toMatchObject({
        success: false,
        checks: [{
          actual: {
            inside: true,
            grasp: { evidence_complete: true, verified_hands: ["left"] },
            settled_support: { status: "satisfied" }
          }
        }]
      });

      snapshot.grasp.assessments = [graspAssessment(snapshot.frame, "left", false)];
      expect(checkHumanoidGoal(goal, scenario, snapshot)).toMatchObject({
        success: false,
        checks: [{ actual: { grasp: { evidence_complete: false, verified_hands: [] } } }]
      });

      snapshot.grasp.assessments = [
        graspAssessment(snapshot.frame, "left", false),
        graspAssessment(snapshot.frame, "right", false)
      ];
      object.linearVelocity = { x: 0.031, y: 0, z: 0 };
      expect(checkHumanoidGoal(goal, scenario, snapshot)).toMatchObject({
        success: false,
        checks: [{
          actual: {
            grasp: { evidence_complete: true, verified_hands: [] },
            settled_support: {
              status: "unsatisfied",
              reason: "linear_velocity_exceeded"
            }
          }
        }]
      });

      object.linearVelocity = { x: 0, y: 0, z: 0 };
      snapshot.robot.contacts[0]!.firstBody = "pelvis";
      expect(checkHumanoidGoal(goal, scenario, snapshot)).toMatchObject({
        success: false,
        checks: [{
          actual: {
            settled_support: {
              status: "unsatisfied",
              reason: "support_contact_missing"
            }
          }
        }]
      });
    } finally {
      await world.dispose();
    }
  }, 30_000);

  it("accepts a grasp only from complete current-frame authority assessments", async () => {
    const world = await HumanoidWorld.create(scenario);
    try {
      const snapshot = world.snapshot();
      snapshot.frame += 1;
      snapshot.worldRevision += 1;
      snapshot.grasp = {
        contractSha256: "a".repeat(64),
        assessments: [
          graspAssessment(snapshot.frame, "left", true),
          graspAssessment(snapshot.frame, "right", false)
        ]
      };
      const leftGoal = GoalSchema.parse({
        summary: "左手抓住箱体",
        predicates: [{ type: "object_grasped", object_id: "crate", hand: "left" }]
      });
      const rightGoal = GoalSchema.parse({
        summary: "右手抓住箱体",
        predicates: [{ type: "object_grasped", object_id: "crate", hand: "right" }]
      });
      const eitherGoal = GoalSchema.parse({
        summary: "任意手抓住箱体",
        predicates: [{ type: "object_grasped", object_id: "crate", hand: "either" }]
      });

      expect(checkHumanoidGoal(leftGoal, scenario, snapshot)).toMatchObject({
        success: true,
        checks: [{
          passed: true,
          actual: {
            object_id: "crate",
            requested_hand: "left",
            contract_sha256: "a".repeat(64),
            world_frame: snapshot.frame,
            evidence_complete: true,
            assessed_hands: ["left"],
            verified_hands: ["left"]
          }
        }]
      });
      expect(checkHumanoidGoal(rightGoal, scenario, snapshot).success).toBe(false);
      expect(checkHumanoidGoal(eitherGoal, scenario, snapshot).success).toBe(true);

      snapshot.grasp.assessments = [graspAssessment(snapshot.frame, "left", true)];
      expect(checkHumanoidGoal(eitherGoal, scenario, snapshot)).toMatchObject({
        success: false,
        checks: [{ actual: { evidence_complete: false, verified_hands: ["left"] } }]
      });

      snapshot.grasp.assessments = [
        graspAssessment(snapshot.frame - 1, "left", true)
      ];
      expect(checkHumanoidGoal(leftGoal, scenario, snapshot)).toMatchObject({
        success: false,
        checks: [{ actual: { evidence_complete: false, assessments: [] } }]
      });
    } finally {
      await world.dispose();
    }
  }, 30_000);

  it("rejects grasp and placement goals for unknown and non-portable objects", () => {
    expect(() => assertHumanoidGoalSupported(GoalSchema.parse({
      summary: "抓住未知物体",
      predicates: [{ type: "object_grasped", object_id: "missing", hand: "either" }]
    }), scenario)).toThrow("Unknown object: missing");

    const fixedScenario = {
      ...scenario,
      objects: scenario.objects.map((object) => ({ ...object, portable: false }))
    };
    expect(() => assertHumanoidGoalSupported(GoalSchema.parse({
      summary: "抓住固定物体",
      predicates: [{ type: "object_grasped", object_id: "crate", hand: "left" }]
    }), fixedScenario)).toThrow("Object is not movable: crate");

    expect(() => assertHumanoidGoalSupported(GoalSchema.parse({
      summary: "放置固定物体",
      predicates: [{
        type: "object_placed",
        object_id: "crate",
        zone_id: "arrival",
        tolerance: 0.03
      }]
    }), fixedScenario)).toThrow("Object is not movable: crate");
  });

  it("requires consecutive authoritative frames for world and pelvis end-effector goals", async () => {
    const world = await HumanoidWorld.create(scenario);
    try {
      const snapshot = world.snapshot();
      const pelvisPosition = { x: 6, y: 0.8, z: 6 };
      const halfAngle = Math.PI / 4;
      const pelvisRotation = {
        x: 0,
        y: Math.sin(halfAngle),
        z: 0,
        w: Math.cos(halfAngle)
      };
      const rightAnkleInPelvis = { x: -0.12, y: -0.74, z: 0.16 };
      const leftWristInWorld = { x: 6.35, y: 0.92, z: 6.18 };
      snapshot.robot.links.pelvis.position = pelvisPosition;
      snapshot.robot.links.pelvis.rotation = pelvisRotation;
      snapshot.robot.links.left_wrist_yaw_link.position = { ...leftWristInWorld };
      snapshot.robot.links.right_ankle_roll_link.position = add(
        pelvisPosition,
        rotateVector(pelvisRotation, rightAnkleInPelvis)
      );
      const goal = GoalSchema.parse({
        summary: "保持左腕和右踝末端位置",
        predicates: [
          {
            type: "end_effector_at",
            end_effector: "left_wrist",
            frame: "world",
            target: leftWristInWorld,
            tolerance: 0.01,
            stable_frames: 3
          },
          {
            type: "end_effector_at",
            end_effector: "right_ankle",
            frame: "pelvis",
            target: rightAnkleInPelvis,
            tolerance: 0.01,
            stable_frames: 3
          }
        ]
      });
      let progress = createHumanoidGoalProgress(goal, snapshot);
      expect(inspectHumanoidGoal(goal, scenario, snapshot, progress)).toMatchObject({
        success: false,
        checks: [
          { passed: false, actual: { current_stable_frames: 0 } },
          { passed: false, actual: { current_stable_frames: 0 } }
        ]
      });

      for (let expected = 1; expected <= 3; expected += 1) {
        snapshot.frame += 1;
        snapshot.worldRevision += 1;
        const advanced = advanceHumanoidGoal(goal, scenario, snapshot, progress);
        progress = advanced.progress;
        expect(advanced.checker.checks).toEqual(expect.arrayContaining([
          expect.objectContaining({
            passed: expected === 3,
            actual: expect.objectContaining({
              satisfied: true,
              current_stable_frames: expected,
              required_stable_frames: 3
            })
          })
        ]));
        expect(advanced.checker.success).toBe(expected === 3);
      }

      const duplicate = advanceHumanoidGoal(goal, scenario, snapshot, progress);
      expect(duplicate.progress).toEqual(progress);
      expect(duplicate.checker.success).toBe(true);

      snapshot.robot.links.left_wrist_yaw_link.position.x += 0.1;
      snapshot.frame += 1;
      snapshot.worldRevision += 1;
      const broken = advanceHumanoidGoal(goal, scenario, snapshot, progress);
      expect(broken.checker.success).toBe(false);
      expect(broken.progress.predicate_streaks).toEqual([0, 3]);

      snapshot.robot.links.left_wrist_yaw_link.position = { ...leftWristInWorld };
      snapshot.frame += 2;
      snapshot.worldRevision += 2;
      const afterGap = advanceHumanoidGoal(goal, scenario, snapshot, broken.progress);
      expect(afterGap.progress.predicate_streaks).toEqual([1, 1]);
      expect(afterGap.checker.success).toBe(false);
      assertHumanoidGoalProgressIntegrity(goal, snapshot, afterGap.progress);
    } finally {
      await world.dispose();
    }
  }, 30_000);

  it("checks pelvis-relative orientation from each authoritative link snapshot", async () => {
    const world = await HumanoidWorld.create(scenario);
    try {
      const snapshot = world.snapshot();
      const halfPelvisYaw = Math.PI / 4;
      const pelvisRotation = {
        x: 0,
        y: Math.sin(halfPelvisYaw),
        z: 0,
        w: Math.cos(halfPelvisYaw)
      };
      const wristPosition = { x: 0.22, y: 0.28, z: 0.12 };
      const wristRotation = {
        x: Math.sin(0.35),
        y: 0,
        z: 0,
        w: Math.cos(0.35)
      };
      snapshot.robot.links.pelvis.rotation = pelvisRotation;
      snapshot.robot.links.left_wrist_yaw_link.position = add(
        snapshot.robot.links.pelvis.position,
        rotateVector(pelvisRotation, wristPosition)
      );
      snapshot.robot.links.left_wrist_yaw_link.rotation = multiplyQuaternion(
        pelvisRotation,
        wristRotation
      );
      const goal = GoalSchema.parse({
        summary: "保持左腕位姿",
        predicates: [{
          type: "end_effector_at",
          end_effector: "left_wrist",
          frame: "pelvis",
          target: wristPosition,
          tolerance: 0.01,
          stable_frames: 1,
          orientation: {
            x: -wristRotation.x,
            y: -wristRotation.y,
            z: -wristRotation.z,
            w: -wristRotation.w
          },
          orientation_tolerance_rad: 0.03
        }]
      });
      let progress = createHumanoidGoalProgress(goal, snapshot);

      snapshot.frame += 1;
      snapshot.worldRevision += 1;
      const matched = advanceHumanoidGoal(goal, scenario, snapshot, progress);
      progress = matched.progress;
      expect(matched.checker).toMatchObject({
        success: true,
        checks: [{
          passed: true,
          actual: {
            satisfied: true,
            distance: expect.closeTo(0, 10),
            orientation_error_rad: expect.closeTo(0, 10),
            orientation_tolerance_rad: 0.03,
            current_stable_frames: 1
          }
        }]
      });

      snapshot.robot.links.left_wrist_yaw_link.rotation = pelvisRotation;
      snapshot.frame += 1;
      snapshot.worldRevision += 1;
      const rotatedAway = advanceHumanoidGoal(goal, scenario, snapshot, progress);
      progress = rotatedAway.progress;
      expect(rotatedAway.checker).toMatchObject({
        success: false,
        checks: [{
          passed: false,
          actual: {
            satisfied: false,
            distance: expect.closeTo(0, 10),
            orientation_error_rad: expect.closeTo(0.7, 10),
            current_stable_frames: 0
          }
        }]
      });

      snapshot.robot.links.left_wrist_yaw_link.rotation = { x: 0, y: 0, z: 0, w: 0 };
      snapshot.frame += 1;
      snapshot.worldRevision += 1;
      expect(advanceHumanoidGoal(goal, scenario, snapshot, progress).checker).toMatchObject({
        success: false,
        checks: [{ actual: { orientation: null, orientation_error_rad: null } }]
      });
    } finally {
      await world.dispose();
    }
  }, 30_000);

  it("rejects tampered, stale and partially advanced goal progress", async () => {
    const world = await HumanoidWorld.create(scenario);
    try {
      const snapshot = world.snapshot();
      const goal = GoalSchema.parse({
        summary: "保持左腕位置",
        predicates: [{
          type: "end_effector_at",
          end_effector: "left_wrist",
          frame: "world",
          target: snapshot.robot.links.left_wrist_yaw_link.position,
          tolerance: 0.05,
          stable_frames: 2
        }]
      });
      const progress = createHumanoidGoalProgress(goal, snapshot);
      expect(() => inspectHumanoidGoal(goal, scenario, snapshot, {
        ...progress,
        goal_sha256: "0".repeat(64)
      })).toThrow("another goal");
      expect(() => inspectHumanoidGoal(goal, scenario, snapshot, {
        ...progress,
        predicate_count: 2,
        predicate_streaks: [0, 0]
      })).toThrow("predicate count");
      expect(() => inspectHumanoidGoal(goal, scenario, snapshot, {
        ...progress,
        predicate_streaks: [3]
      })).toThrow("impossible stability streak");

      snapshot.frame += 1;
      expect(() => advanceHumanoidGoal(goal, scenario, snapshot, progress)).toThrow(
        "advance partially"
      );
    } finally {
      await world.dispose();
    }
  }, 30_000);
});

function graspAssessment(
  frame: number,
  hand: "left" | "right",
  verified: boolean
): HumanoidGraspAssessment {
  const prefix = `${hand}_hand_` as const;
  const firstLink = `${prefix}index_1_link` as HumanoidGraspAssessment[
    "evidence"
  ]["contact"]["distinct_force_qualified_links"][number];
  const secondLink = `${prefix}thumb_2_link` as typeof firstLink;
  return {
    protocol: "humanoid-grasp-assessment-v1",
    frame,
    object_id: "crate",
    hand,
    phase: verified ? "verified" : "idle",
    grasp_verified: verified,
    reason: verified ? "grasp_verified" : "contact_missing",
    reset_reason: verified ? null : "contact_lost",
    evidence: {
      contact: {
        status: verified ? "opposed" : "missing",
        observed_contact_count: verified ? 2 : 0,
        force_qualified_contact_count: verified ? 2 : 0,
        distinct_force_qualified_links: verified ? [firstLink, secondLink] : [],
        distinct_normal_qualified_links: verified ? [firstLink, secondLink] : [],
        opposing_pair: verified ? {
          first_link: firstLink,
          second_link: secondLink,
          first_position: { x: -0.03, y: 0.5, z: 0 },
          second_position: { x: 0.03, y: 0.5, z: 0 },
          first_normal_from_hand: { x: 1, y: 0, z: 0 },
          second_normal_from_hand: { x: -1, y: 0, z: 0 },
          first_normal_force_n: 10,
          second_normal_force_n: 11,
          separation_m: 0.06,
          normal_dot: -1,
          position_dot: -1
        } : null
      },
      support: {
        status: verified ? "unsupported" : "supported",
        candidate_contact_count: verified ? 0 : 1,
        force_qualified_contact_count: verified ? 0 : 1,
        upward_contact_count: verified ? 0 : 1,
        baseline_projection_m: 0.5,
        current_projection_m: verified ? 0.56 : 0.5,
        lift_m: verified ? 0.06 : 0
      },
      relative_pose: {
        stable_frames: verified ? 8 : 0,
        translation_drift_m: verified ? 0.001 : null,
        rotation_drift_rad: verified ? 0.002 : null
      },
      lifted_hold_frames: verified ? 8 : 0
    }
  };
}
