import {
  HUMANOID_JOINT_INDEX,
  type HumanoidJointName
} from "./model.js";

export type HumanoidEndEffectorBody =
  | "left_ankle_roll_link"
  | "right_ankle_roll_link"
  | "left_wrist_yaw_link"
  | "right_wrist_yaw_link";

const HUMANOID_END_EFFECTOR_JOINT_CHAINS: Readonly<Record<
  HumanoidEndEffectorBody,
  readonly HumanoidJointName[]
>> = {
  left_ankle_roll_link: [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint"
  ],
  right_ankle_roll_link: [
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint"
  ],
  left_wrist_yaw_link: [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint"
  ],
  right_wrist_yaw_link: [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint"
  ]
};

export function humanoidEndEffectorJointIndexes(
  body: HumanoidEndEffectorBody
): number[] {
  return HUMANOID_END_EFFECTOR_JOINT_CHAINS[body].map((joint) => {
    const index = HUMANOID_JOINT_INDEX.get(joint);
    if (index === undefined) {
      throw new Error(`Missing humanoid task-space joint: ${joint}`);
    }
    return index;
  });
}
