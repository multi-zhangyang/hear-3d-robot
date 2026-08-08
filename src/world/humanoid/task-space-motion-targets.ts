import type { HumanoidMotionKeyframe } from "./motion-plan-schema.js";
import {
  targetReference,
  type HumanoidReference,
  type HumanoidReferenceTarget
} from "./reference.js";
import type {
  HumanoidEndEffectorTarget,
  HumanoidPlanningRootPose,
  HumanoidSimulation
} from "./simulation.js";

export function taskSpaceReference(
  simulation: HumanoidSimulation,
  baseline: HumanoidReference,
  keyframe: HumanoidMotionKeyframe,
  planningRootPose?: HumanoidPlanningRootPose
): HumanoidReference {
  const rooted = taskSpaceRootReference(baseline, keyframe);
  return simulation.solveEndEffectorTargets(
    rooted,
    taskSpaceTargets(keyframe),
    planningRootPose ? { planningRootPose } : {}
  ).reference;
}

export function taskSpaceRootReference(
  baseline: HumanoidReference,
  keyframe: HumanoidMotionKeyframe
): HumanoidReference {
  const rootTarget: HumanoidReferenceTarget = {
    ...(keyframe.root_velocity
      ? {
          rootVelocity: [
            keyframe.root_velocity.forward_mps,
            keyframe.root_velocity.lateral_mps
          ]
        }
      : {}),
    ...(keyframe.root_yaw_velocity != null
      ? { rootYawVelocity: keyframe.root_yaw_velocity }
      : {}),
    ...(keyframe.root_height != null ? { rootHeight: keyframe.root_height } : {}),
    ...(keyframe.root_roll != null ? { rootRoll: keyframe.root_roll } : {}),
    ...(keyframe.root_pitch != null ? { rootPitch: keyframe.root_pitch } : {}),
    ...(keyframe.torso_yaw != null
      ? { joints: { waist_yaw_joint: keyframe.torso_yaw } }
      : {})
  };
  return targetReference(baseline, rootTarget);
}

export function taskSpaceTargets(
  keyframe: HumanoidMotionKeyframe
): HumanoidEndEffectorTarget[] {
  return [
    ...(keyframe.left_hand ? [{
      body: "left_wrist_yaw_link" as const,
      position: keyframe.left_hand.position,
      frame: keyframe.left_hand.frame,
      tolerance: keyframe.left_hand.tolerance_m,
      ...(keyframe.left_hand.kinematic_scope
        ? { kinematicScope: keyframe.left_hand.kinematic_scope }
        : {}),
      ...(keyframe.left_hand.servo_mode
        ? { servoMode: keyframe.left_hand.servo_mode }
        : {}),
      ...(keyframe.left_hand.orientation !== undefined
        && keyframe.left_hand.orientation_tolerance_rad !== undefined
        ? {
            orientation: keyframe.left_hand.orientation,
            orientationTolerance: keyframe.left_hand.orientation_tolerance_rad
          }
        : {})
    }] : []),
    ...(keyframe.right_hand ? [{
      body: "right_wrist_yaw_link" as const,
      position: keyframe.right_hand.position,
      frame: keyframe.right_hand.frame,
      tolerance: keyframe.right_hand.tolerance_m,
      ...(keyframe.right_hand.kinematic_scope
        ? { kinematicScope: keyframe.right_hand.kinematic_scope }
        : {}),
      ...(keyframe.right_hand.servo_mode
        ? { servoMode: keyframe.right_hand.servo_mode }
        : {}),
      ...(keyframe.right_hand.orientation !== undefined
        && keyframe.right_hand.orientation_tolerance_rad !== undefined
        ? {
            orientation: keyframe.right_hand.orientation,
            orientationTolerance: keyframe.right_hand.orientation_tolerance_rad
          }
        : {})
    }] : []),
    ...(keyframe.left_foot ? [{
      body: "left_ankle_roll_link" as const,
      position: keyframe.left_foot.position,
      frame: keyframe.left_foot.frame,
      tolerance: keyframe.left_foot.tolerance_m,
      ...(keyframe.left_foot.orientation !== undefined
        && keyframe.left_foot.orientation_tolerance_rad !== undefined
        ? {
            orientation: keyframe.left_foot.orientation,
            orientationTolerance: keyframe.left_foot.orientation_tolerance_rad
          }
        : {})
    }] : []),
    ...(keyframe.right_foot ? [{
      body: "right_ankle_roll_link" as const,
      position: keyframe.right_foot.position,
      frame: keyframe.right_foot.frame,
      tolerance: keyframe.right_foot.tolerance_m,
      ...(keyframe.right_foot.orientation !== undefined
        && keyframe.right_foot.orientation_tolerance_rad !== undefined
        ? {
            orientation: keyframe.right_foot.orientation,
            orientationTolerance: keyframe.right_foot.orientation_tolerance_rad
          }
        : {})
    }] : [])
  ];
}
