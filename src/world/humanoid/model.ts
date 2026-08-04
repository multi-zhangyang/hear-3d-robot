export const HUMANOID_JOINT_NAMES = [
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint"
] as const;

export type HumanoidJointName = typeof HUMANOID_JOINT_NAMES[number];

export const HUMANOID_BODY_NAMES = [
  "pelvis",
  "left_hip_pitch_link",
  "left_hip_roll_link",
  "left_hip_yaw_link",
  "left_knee_link",
  "left_ankle_pitch_link",
  "left_ankle_roll_link",
  "right_hip_pitch_link",
  "right_hip_roll_link",
  "right_hip_yaw_link",
  "right_knee_link",
  "right_ankle_pitch_link",
  "right_ankle_roll_link",
  "waist_yaw_link",
  "waist_roll_link",
  "torso_link",
  "head_link",
  "left_shoulder_pitch_link",
  "left_shoulder_roll_link",
  "left_shoulder_yaw_link",
  "left_elbow_link",
  "left_wrist_roll_link",
  "left_wrist_pitch_link",
  "left_wrist_yaw_link",
  "right_shoulder_pitch_link",
  "right_shoulder_roll_link",
  "right_shoulder_yaw_link",
  "right_elbow_link",
  "right_wrist_roll_link",
  "right_wrist_pitch_link",
  "right_wrist_yaw_link"
] as const;

export type HumanoidBodyName = typeof HUMANOID_BODY_NAMES[number];

export const HUMANOID_HEAD_SENSOR = {
  horizontalFieldOfView: 110 * Math.PI / 180,
  verticalFieldOfView: 80 * Math.PI / 180,
  localPosition: { x: 0, y: 0.43, z: 0.08 },
  localRotation: {
    x: Math.sin(25 * Math.PI / 360),
    y: 0,
    z: 0,
    w: Math.cos(25 * Math.PI / 360)
  }
} as const;

export const YAHMP_POLICY = {
  physicsDt: 0.005,
  controlDt: 0.02,
  historyLength: 10,
  observationSize: 1727,
  defaultJointPositions: [
    -0.312, 0, 0, 0.669, -0.363, 0,
    -0.312, 0, 0, 0.669, -0.363, 0,
    0, 0, 0,
    0.2, 0.2, 0, 0.6, 0, 0, 0,
    0.2, -0.2, 0, 0.6, 0, 0, 0
  ],
  actionScale: [
    0.5475464463, 0.3506614566, 0.5475464463, 0.3506614566, 0.4385773242,
    0.4385773242, 0.5475464463, 0.3506614566, 0.5475464463, 0.3506614566,
    0.4385773242, 0.4385773242, 0.5475464463, 0.4385773242, 0.4385773242,
    0.4385773242, 0.4385773242, 0.4385773242, 0.4385773242, 0.4385773242,
    0.0745008737, 0.0745008737, 0.4385773242, 0.4385773242, 0.4385773242,
    0.4385773242, 0.4385773242, 0.0745008737, 0.0745008737
  ],
  stiffness: [
    40.1792386345, 99.0984277767, 40.1792386345, 99.0984277767, 28.5012461957,
    28.5012461957, 40.1792386345, 99.0984277767, 40.1792386345, 99.0984277767,
    28.5012461957, 28.5012461957, 40.1792386345, 28.5012461957, 28.5012461957,
    14.2506230979, 14.2506230979, 14.2506230979, 14.2506230979, 14.2506230979,
    16.7783274809, 16.7783274809, 14.2506230979, 14.2506230979, 14.2506230979,
    14.2506230979, 14.2506230979, 16.7783274809, 16.7783274809
  ],
  damping: [
    2.5578897754, 6.3088018535, 2.5578897754, 6.3088018535, 1.8144456866,
    1.8144456866, 2.5578897754, 6.3088018535, 2.5578897754, 6.3088018535,
    1.8144456866, 1.8144456866, 2.5578897754, 1.8144456866, 1.8144456866,
    0.9072228433, 0.9072228433, 0.9072228433, 0.9072228433, 0.9072228433,
    1.0681415022, 1.0681415022, 0.9072228433, 0.9072228433, 0.9072228433,
    0.9072228433, 0.9072228433, 1.0681415022, 1.0681415022
  ]
} as const;

export const HUMANOID_JOINT_INDEX = new Map<HumanoidJointName, number>(
  HUMANOID_JOINT_NAMES.map((name, index) => [name, index])
);
