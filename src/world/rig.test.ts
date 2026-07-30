import { describe, expect, it } from "vitest";
import { rotateVector, subtract, vectorLength } from "./geometry.js";
import { rigTransforms } from "./rig.js";
import { ROBOT_SPEC, type RobotJointState } from "./robot-model.js";

const BASE = { x: 0, y: ROBOT_SPEC.base.centerY, z: 0 };

const POSES: RobotJointState[] = [
  { ...ROBOT_SPEC.defaultJoints },
  { head_yaw: 0, head_pitch: -0.4, shoulder: 0.15, elbow: -0.55, wrist: 0.3, gripper_aperture: 0.6 },
  { head_yaw: 0.9, head_pitch: 0.3, shoulder: -1.2, elbow: -1.9, wrist: 1.1, gripper_aperture: 0.12 },
  { head_yaw: -2.4, head_pitch: 0.8, shoulder: 1.55, elbow: -2.5, wrist: -1.7, gripper_aperture: 0.72 }
];

const YAWS = [0, 0.7, -1.9, Math.PI];

describe("rig forward kinematics", () => {
  /**
   * A link's orientation and the offset it contributes to the chain are derived
   * separately — one from armRotation, the other from armVector — so nothing
   * forces them to agree. When they disagreed, every arm link's frame was
   * mirrored about the horizontal plane: the drawn arm and the swept colliders
   * bent the opposite way from the joints that positioned them.
   */
  it("orients each arm link along the segment it spans", () => {
    for (const joints of POSES) {
      for (const yaw of YAWS) {
        const t = rigTransforms(joints, BASE, yaw);

        // Each link centre sits midway between the joints it connects, so
        // stepping half a length along the link's own +Z from one centre must
        // reach the next joint, which is where the next link's own step starts.
        const step = (
          link: { position: { x: number; y: number; z: number }; rotation: { x: number; y: number; z: number; w: number } },
          length: number,
          sign: number
        ): { x: number; y: number; z: number } => {
          const axis = rotateVector(link.rotation, { x: 0, y: 0, z: 1 });
          return {
            x: link.position.x + (axis.x * length * sign) / 2,
            y: link.position.y + (axis.y * length * sign) / 2,
            z: link.position.z + (axis.z * length * sign) / 2
          };
        };

        const upperTip = step(t.upperArm, ROBOT_SPEC.arm.upperLength, 1);
        const forearmRoot = step(t.forearm, ROBOT_SPEC.arm.forearmLength, -1);
        expect(vectorLength(subtract(upperTip, forearmRoot))).toBeLessThan(1e-9);

        const forearmTip = step(t.forearm, ROBOT_SPEC.arm.forearmLength, 1);
        const wristRoot = step(t.wrist, ROBOT_SPEC.arm.wristLength, -1);
        expect(vectorLength(subtract(forearmTip, wristRoot))).toBeLessThan(1e-9);

        const wristTip = step(t.wrist, ROBOT_SPEC.arm.wristLength, 1);
        expect(vectorLength(subtract(wristTip, t.gripper.position))).toBeLessThan(1e-9);
      }
    }
  });

  it("anchors the shoulder chain at the spec's shoulder mount", () => {
    for (const joints of POSES) {
      for (const yaw of YAWS) {
        const t = rigTransforms(joints, BASE, yaw);
        const axis = rotateVector(t.upperArm.rotation, { x: 0, y: 0, z: 1 });
        const shoulder = {
          x: t.upperArm.position.x - (axis.x * ROBOT_SPEC.arm.upperLength) / 2,
          y: t.upperArm.position.y - (axis.y * ROBOT_SPEC.arm.upperLength) / 2,
          z: t.upperArm.position.z - (axis.z * ROBOT_SPEC.arm.upperLength) / 2
        };
        expect(shoulder.y).toBeCloseTo(ROBOT_SPEC.arm.shoulderHeight, 9);
        const forward = { x: Math.sin(yaw), y: 0, z: Math.cos(yaw) };
        expect(shoulder.x).toBeCloseTo(BASE.x + forward.x * ROBOT_SPEC.arm.shoulderForwardOffset, 9);
        expect(shoulder.z).toBeCloseTo(BASE.z + forward.z * ROBOT_SPEC.arm.shoulderForwardOffset, 9);
      }
    }
  });

  it("separates the fingers along the base's lateral axis by the aperture", () => {
    for (const joints of POSES) {
      for (const yaw of YAWS) {
        const t = rigTransforms(joints, BASE, yaw);
        const span = subtract(t.leftFinger.position, t.rightFinger.position);
        expect(vectorLength(span)).toBeCloseTo(joints.gripper_aperture, 9);
        expect(span.y).toBeCloseTo(0, 9);
      }
    }
  });
});
