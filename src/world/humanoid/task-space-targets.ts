import {
  HUMANOID_JOINT_INDEX,
  type HumanoidJointName
} from "./model.js";

export const HUMANOID_END_EFFECTOR_BODIES = [
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link"
] as const;

export type HumanoidEndEffectorBody =
  typeof HUMANOID_END_EFFECTOR_BODIES[number];

export const HUMANOID_TASK_SPACE_KINEMATIC_SCOPES = [
  "arm_only",
  "whole_body_reach"
] as const;

export type HumanoidTaskSpaceKinematicScope =
  typeof HUMANOID_TASK_SPACE_KINEMATIC_SCOPES[number];

const HUMANOID_END_EFFECTOR_JOINT_CHAINS: Readonly<Record<
  HumanoidEndEffectorBody,
  readonly HumanoidJointName[]
>> = {
  left_ankle_roll_link: [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint"
  ],
  right_ankle_roll_link: [
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint"
  ],
  left_wrist_yaw_link: [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint"
  ],
  right_wrist_yaw_link: [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint"
  ]
};

const HUMANOID_WHOLE_BODY_REACH_JOINTS = [
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint"
] as const satisfies readonly HumanoidJointName[];

const HUMANOID_END_EFFECTOR_ORIENTATION_JOINTS: Readonly<Record<
  HumanoidEndEffectorBody,
  HumanoidJointName
>> = {
  left_ankle_roll_link: "left_ankle_roll_joint",
  right_ankle_roll_link: "right_ankle_roll_joint",
  left_wrist_yaw_link: "left_wrist_yaw_joint",
  right_wrist_yaw_link: "right_wrist_yaw_joint"
};

export function humanoidEndEffectorJointIndexes(
  body: HumanoidEndEffectorBody,
  scope: HumanoidTaskSpaceKinematicScope = "arm_only"
): number[] {
  return humanoidJointIndexes(jointChain(body, scope));
}

export function humanoidEndEffectorPoseJointIndexes(
  body: HumanoidEndEffectorBody,
  scope: HumanoidTaskSpaceKinematicScope = "arm_only"
): number[] {
  return humanoidJointIndexes([
    ...jointChain(body, scope),
    HUMANOID_END_EFFECTOR_ORIENTATION_JOINTS[body]
  ]);
}

export function humanoidEndEffectorTrackingJointIndexes(
  body: HumanoidEndEffectorBody,
  trackOrientation = false,
  scope: HumanoidTaskSpaceKinematicScope = "arm_only"
): number[] {
  return humanoidJointIndexes([
    ...jointChain(body, scope),
    ...(trackOrientation ? [HUMANOID_END_EFFECTOR_ORIENTATION_JOINTS[body]] : [])
  ]);
}

function jointChain(
  body: HumanoidEndEffectorBody,
  scope: HumanoidTaskSpaceKinematicScope
): readonly HumanoidJointName[] {
  const limb = HUMANOID_END_EFFECTOR_JOINT_CHAINS[body];
  return scope === "whole_body_reach" && body.endsWith("wrist_yaw_link")
    ? [...HUMANOID_WHOLE_BODY_REACH_JOINTS, ...limb]
    : limb;
}

function humanoidJointIndexes(joints: readonly HumanoidJointName[]): number[] {
  return joints.map((joint) => {
    const index = HUMANOID_JOINT_INDEX.get(joint);
    if (index === undefined) {
      throw new Error(`Missing humanoid task-space joint: ${joint}`);
    }
    return index;
  });
}
