import type { JsonValue, Vec3 } from "../domain/schema.js";
import { ROBOT_SPEC } from "./robot-model.js";

/** Below this radius the configured one-sided elbow and wrist limits make
 * many otherwise nearby points part of a tight folded workspace. It is a
 * planning comfort boundary, not a claim that every closer point is invalid. */
export const ARM_PREFERRED_MINIMUM_REACH = 0.75;
export const ARM_MOTION_PLANE_TOLERANCE = 0.025;
export const ARM_MAXIMUM_REACH = ROBOT_SPEC.arm.upperLength
  + ROBOT_SPEC.arm.forearmLength
  + ROBOT_SPEC.arm.wristLength;

export interface ArmReachMetrics {
  shoulderPosition: Vec3;
  targetDistanceFromShoulder: number;
  maximumArmReach: number;
  preferredMinimumArmReach: number;
  targetWithinReach: boolean;
  armMotionPlaneLateralError: number;
  armMotionPlaneTolerance: number;
}

/** Cheap analytic workspace evidence used to rank observations before the
 * full closed-chain solver is asked to produce a trajectory. It is only a
 * plausibility filter: collision-aware IK remains authoritative. */
export function armReachMetrics(input: {
  base: Vec3;
  yaw: number;
  target: Vec3;
}): ArmReachMetrics {
  const forward = { x: Math.sin(input.yaw), z: Math.cos(input.yaw) };
  const lateral = { x: Math.cos(input.yaw), z: -Math.sin(input.yaw) };
  const shoulderPosition = {
    x: input.base.x + forward.x * ROBOT_SPEC.arm.shoulderForwardOffset,
    y: ROBOT_SPEC.arm.shoulderHeight,
    z: input.base.z + forward.z * ROBOT_SPEC.arm.shoulderForwardOffset
  };
  const delta = {
    x: input.target.x - shoulderPosition.x,
    y: input.target.y - shoulderPosition.y,
    z: input.target.z - shoulderPosition.z
  };
  const targetDistanceFromShoulder = Math.hypot(delta.x, delta.y, delta.z);
  return {
    shoulderPosition,
    targetDistanceFromShoulder,
    maximumArmReach: ARM_MAXIMUM_REACH,
    preferredMinimumArmReach: ARM_PREFERRED_MINIMUM_REACH,
    targetWithinReach: targetDistanceFromShoulder <= ARM_MAXIMUM_REACH,
    armMotionPlaneLateralError: Math.abs(delta.x * lateral.x + delta.z * lateral.z),
    armMotionPlaneTolerance: ARM_MOTION_PLANE_TOLERANCE
  };
}

export function armReachDiagnosis(input: {
  base: Vec3;
  yaw: number;
  target: Vec3;
}): Record<string, JsonValue> {
  const metrics = armReachMetrics(input);
  const distance = metrics.targetDistanceFromShoulder;
  const maximumReach = metrics.maximumArmReach;
  const lateralError = metrics.armMotionPlaneLateralError;
  const detail: Record<string, JsonValue> = {
    shoulder_position: metrics.shoulderPosition,
    target_distance_from_shoulder: distance,
    maximum_arm_reach: maximumReach,
    preferred_minimum_arm_reach: metrics.preferredMinimumArmReach,
    target_within_reach: metrics.targetWithinReach,
    arm_motion_plane_lateral_error: lateralError,
    arm_motion_plane_tolerance: metrics.armMotionPlaneTolerance
  };

  if (lateralError > metrics.armMotionPlaneTolerance) {
    detail.recovery = `The target is ${lateralError.toFixed(2)}m sideways from the arm's `
      + "base-forward motion plane. Driving closer alone does not remove sideways error. If an "
      + "orientation constraint caused this denial, retry the measured point with "
      + "solve_end_effector_position. Otherwise inspect the target's reachable_standoff_poses, "
      + "plan_base_path with the interaction point as face_point, call execute_base_plan, then "
      + "solve_end_effector_position again from the aligned pose.";
    return detail;
  }
  if (distance > maximumReach) {
    detail.recovery = `The target is ${distance.toFixed(2)}m from the shoulder but the arm spans `
      + `at most ${maximumReach.toFixed(2)}m, so no joint angles can reach it from here. `
      + "Drive the base closer first: inspect_entity or inspect_voxel reports "
      + "reachable_standoff_poses, and plan_base_path plus execute_base_plan will put the robot "
      + "at one. Solve again only after the base has actually moved.";
    return detail;
  }
  if (distance < ARM_PREFERRED_MINIMUM_REACH) {
    detail.recovery = `The target is only ${distance.toFixed(2)}m from the shoulder. It is inside `
      + "the arm's tightly folded region for the configured shoulder, elbow and wrist limits; "
      + "driving closer makes this worse. Choose a farther reachable_standoff_pose aligned with "
      + "the interaction point, execute the base plan, then solve again.";
  }
  return detail;
}
