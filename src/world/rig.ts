import type { Vec3 } from "../domain/schema.js";
import {
  add,
  armRotation,
  armVector,
  multiplyQuaternion,
  pitchRotation,
  scale,
  yawRotation
} from "./geometry.js";
import type { Quaternion } from "./kinematics.js";
import { ROBOT_SPEC, type RobotJointState } from "./robot-model.js";

export interface RigTransform {
  position: Vec3;
  rotation: Quaternion;
}

export interface RigTransforms {
  torso: RigTransform;
  sensorHead: RigTransform;
  upperArm: RigTransform;
  forearm: RigTransform;
  wrist: RigTransform;
  gripper: RigTransform;
  lateral: Vec3;
  leftFinger: RigTransform;
  rightFinger: RigTransform;
}

/**
 * Forward kinematics for the whole rig: base pose and joint angles in, world
 * transforms for every link out. Pure, so it can be evaluated against a
 * candidate pose for collision preflight without touching the simulation.
 */
export function rigTransforms(
  joints: RobotJointState,
  base: Vec3,
  yaw: number
): RigTransforms {
  const forward = { x: Math.sin(yaw), y: 0, z: Math.cos(yaw) };
  const lateral = { x: Math.cos(yaw), y: 0, z: -Math.sin(yaw) };
  const shoulder = add(
    base,
    scale(forward, ROBOT_SPEC.arm.shoulderForwardOffset),
    { x: 0, y: ROBOT_SPEC.arm.shoulderHeight - base.y, z: 0 }
  );
  const angle1 = joints.shoulder;
  const angle2 = angle1 + joints.elbow;
  const angle3 = angle2 + joints.wrist;
  const elbow = add(shoulder, armVector(forward, angle1, ROBOT_SPEC.arm.upperLength));
  const wristStart = add(elbow, armVector(forward, angle2, ROBOT_SPEC.arm.forearmLength));
  const gripperPosition = add(
    wristStart,
    armVector(forward, angle3, ROBOT_SPEC.arm.wristLength)
  );
  const gripperRotation = armRotation(yaw, angle3);
  const halfAperture = joints.gripper_aperture / 2;
  return {
    torso: {
      position: { x: base.x, y: ROBOT_SPEC.torso.centerHeight, z: base.z },
      rotation: yawRotation(yaw)
    },
    sensorHead: {
      position: add(
        { x: base.x, y: ROBOT_SPEC.sensorHead.centerHeight, z: base.z },
        scale(forward, ROBOT_SPEC.sensorHead.forwardOffset)
      ),
      rotation: multiplyQuaternion(
        yawRotation(yaw + joints.head_yaw),
        pitchRotation(joints.head_pitch)
      )
    },
    upperArm: {
      position: add(shoulder, armVector(forward, angle1, ROBOT_SPEC.arm.upperLength / 2)),
      rotation: armRotation(yaw, angle1)
    },
    forearm: {
      position: add(elbow, armVector(forward, angle2, ROBOT_SPEC.arm.forearmLength / 2)),
      rotation: armRotation(yaw, angle2)
    },
    wrist: {
      position: add(
        wristStart,
        armVector(forward, angle3, ROBOT_SPEC.arm.wristLength / 2)
      ),
      rotation: gripperRotation
    },
    gripper: { position: gripperPosition, rotation: gripperRotation },
    lateral,
    leftFinger: {
      position: add(gripperPosition, scale(lateral, halfAperture)),
      rotation: gripperRotation
    },
    rightFinger: {
      position: add(gripperPosition, scale(lateral, -halfAperture)),
      rotation: gripperRotation
    }
  };
}
