import {
  HUMANOID_JOINT_NAMES,
  type HumanoidBodyName
} from "./model.js";

export const G1_HAND_JOINT_NAMES = [
  "left_hand_thumb_0_joint",
  "left_hand_thumb_1_joint",
  "left_hand_thumb_2_joint",
  "left_hand_middle_0_joint",
  "left_hand_middle_1_joint",
  "left_hand_index_0_joint",
  "left_hand_index_1_joint",
  "right_hand_thumb_0_joint",
  "right_hand_thumb_1_joint",
  "right_hand_thumb_2_joint",
  "right_hand_middle_0_joint",
  "right_hand_middle_1_joint",
  "right_hand_index_0_joint",
  "right_hand_index_1_joint"
] as const;

export type G1HandJointName = typeof G1_HAND_JOINT_NAMES[number];

export const G1_HAND_JOINT_LIMITS: Readonly<Record<
  G1HandJointName,
  readonly [minimum: number, maximum: number]
>> = {
  left_hand_thumb_0_joint: [-1.0472, 1.0472],
  left_hand_thumb_1_joint: [-0.724312, 1.0472],
  left_hand_thumb_2_joint: [0, 1.74533],
  left_hand_middle_0_joint: [-1.5708, 0],
  left_hand_middle_1_joint: [-1.74533, 0],
  left_hand_index_0_joint: [-1.5708, 0],
  left_hand_index_1_joint: [-1.74533, 0],
  right_hand_thumb_0_joint: [-1.0472, 1.0472],
  right_hand_thumb_1_joint: [-1.0472, 0.724312],
  right_hand_thumb_2_joint: [-1.74533, 0],
  right_hand_middle_0_joint: [0, 1.5708],
  right_hand_middle_1_joint: [0, 1.74533],
  right_hand_index_0_joint: [0, 1.5708],
  right_hand_index_1_joint: [0, 1.74533]
};

export const G1_HAND_LINK_NAMES = [
  "left_hand_thumb_0_link",
  "left_hand_thumb_1_link",
  "left_hand_thumb_2_link",
  "left_hand_middle_0_link",
  "left_hand_middle_1_link",
  "left_hand_index_0_link",
  "left_hand_index_1_link",
  "right_hand_thumb_0_link",
  "right_hand_thumb_1_link",
  "right_hand_thumb_2_link",
  "right_hand_middle_0_link",
  "right_hand_middle_1_link",
  "right_hand_index_0_link",
  "right_hand_index_1_link"
] as const;

export type G1HandLinkName = typeof G1_HAND_LINK_NAMES[number];

export const G1_HAND_CONTACT_SURFACE_NAMES = [
  "left_hand_palm_link",
  ...G1_HAND_LINK_NAMES.filter((name) => name.startsWith("left_")),
  "right_hand_palm_link",
  ...G1_HAND_LINK_NAMES.filter((name) => name.startsWith("right_"))
] as const;

export type G1HandContactSurfaceName =
  typeof G1_HAND_CONTACT_SURFACE_NAMES[number];

export function g1HandContactSurfaceHand(
  surface: G1HandContactSurfaceName
): "left" | "right" {
  return surface.startsWith("left_") ? "left" : "right";
}

const G1_ALL_JOINT_NAMES = [
  ...HUMANOID_JOINT_NAMES,
  ...G1_HAND_JOINT_NAMES
] as const;

export const G1_MORPHOLOGY = {
  id: "unitree_g1_43dof_with_hands",
  bodyJointCount: HUMANOID_JOINT_NAMES.length,
  handJointCount: G1_HAND_JOINT_NAMES.length,
  totalJointCount: G1_ALL_JOINT_NAMES.length,
  source: {
    repository: "google-deepmind/mujoco_menagerie",
    commit: "71f066ad0be9cd271f7ed58c030243ef157af9f4",
    model: "unitree_g1/g1_with_hands.xml"
  }
} as const;

const G1_MUJOCO_BODY_NAMES: Readonly<Record<HumanoidBodyName, string>> = {
  pelvis: "pelvis",
  left_hip_pitch_link: "left_hip_pitch_link",
  left_hip_roll_link: "left_hip_roll_link",
  left_hip_yaw_link: "left_hip_yaw_link",
  left_knee_link: "left_knee_link",
  left_ankle_pitch_link: "left_ankle_pitch_link",
  left_ankle_roll_link: "left_ankle_roll_link",
  right_hip_pitch_link: "right_hip_pitch_link",
  right_hip_roll_link: "right_hip_roll_link",
  right_hip_yaw_link: "right_hip_yaw_link",
  right_knee_link: "right_knee_link",
  right_ankle_pitch_link: "right_ankle_pitch_link",
  right_ankle_roll_link: "right_ankle_roll_link",
  waist_yaw_link: "waist_yaw_link",
  waist_roll_link: "waist_roll_link",
  torso_link: "torso_link",
  head_link: "torso_link",
  left_shoulder_pitch_link: "left_shoulder_pitch_link",
  left_shoulder_roll_link: "left_shoulder_roll_link",
  left_shoulder_yaw_link: "left_shoulder_yaw_link",
  left_elbow_link: "left_elbow_link",
  left_wrist_roll_link: "left_wrist_roll_link",
  left_wrist_pitch_link: "left_wrist_pitch_link",
  left_wrist_yaw_link: "left_wrist_yaw_link",
  right_shoulder_pitch_link: "right_shoulder_pitch_link",
  right_shoulder_roll_link: "right_shoulder_roll_link",
  right_shoulder_yaw_link: "right_shoulder_yaw_link",
  right_elbow_link: "right_elbow_link",
  right_wrist_roll_link: "right_wrist_roll_link",
  right_wrist_pitch_link: "right_wrist_pitch_link",
  right_wrist_yaw_link: "right_wrist_yaw_link"
};

export function g1MujocoBodyName(name: HumanoidBodyName): string {
  return G1_MUJOCO_BODY_NAMES[name];
}
