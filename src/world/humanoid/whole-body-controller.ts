import type { JsonValue, Quaternion, Vec3 } from "../../domain/schema.js";
import type {
  HumanoidLearnedPolicyCapability
} from "../../domain/humanoid-policy.js";
import type {
  HumanoidMotionOptionPredicate
} from "./motion-option-contract.js";
import type { HumanoidReference } from "./reference.js";

export const HUMANOID_POLICY_OBSERVATION_FEATURES = [
  "proprioception",
  "command_history",
  "root_kinematics",
  "hand_state",
  "end_effector_state",
  "contact_state",
  "object_state",
  "articulation_state",
  "task_space_command",
  "grasp_command"
] as const;

type HumanoidPolicyObservationFeature =
  typeof HUMANOID_POLICY_OBSERVATION_FEATURES[number];

interface HumanoidPolicyEnvironmentState {
  protocol: "humanoid-policy-environment-v1";
  authority: "mujoco_state";
  rootVelocityFrame: "pelvis_imu";
  rootLinearVelocity: readonly [x: number, y: number, z: number];
  rootAngularVelocity: readonly [x: number, y: number, z: number];
  endEffectors: Readonly<Record<string, {
    position: Vec3;
    rotation: Quaternion;
  }>>;
  hands: Readonly<Record<string, {
    position: number;
    velocity: number;
    target: number;
  }>>;
  contacts: ReadonlyArray<{
    position: Vec3;
    normal: Vec3;
    normalForce: number;
    firstBody: string | null;
    secondBody: string | null;
    firstObject: string | null;
    secondObject: string | null;
    firstSolid: string | null;
    secondSolid: string | null;
    firstHandLink: string | null;
    secondHandLink: string | null;
  }>;
  objects: ReadonlyArray<{
    id: string;
    position: Vec3;
    rotation: Quaternion;
    linearVelocity: Vec3;
    angularVelocity: Vec3;
    articulation?: {
      type: "hinge" | "slide";
      position: number;
      velocity: number;
      minimum: number;
      maximum: number;
    } | undefined;
  }>;
}

export interface HumanoidControllerTaskCommand {
  protocol: "humanoid-controller-task-v1";
  taskId: string;
  source: "motion_option" | "carry_navigation";
  requestedCapabilities: HumanoidLearnedPolicyCapability[];
  goal: HumanoidControllerTaskGoal | null;
  endEffectors: ReadonlyArray<{
    body: string;
    frame: "world" | "pelvis" | "torso";
    position: Vec3;
    tolerance: number;
    orientation?: Quaternion | undefined;
    orientationTolerance?: number | undefined;
  }>;
  grasps: ReadonlyArray<{
    objectId: string;
    hand: "left" | "right";
    minimumNormalForceN: number;
    minimumDistinctContactSurfaces: number;
  }>;
}

export type HumanoidControllerTaskGoal =
  | {
      protocol: "humanoid-controller-motion-goal-v1";
      predicates: HumanoidMotionOptionPredicate[];
      stableSteps: number;
    }
  | {
      protocol: "humanoid-controller-navigation-goal-v1";
      target: Vec3;
      positionTolerance: number;
      heading: null | {
        type: "face_point";
        target: Vec3;
        toleranceRadians: number;
      } | {
        type: "yaw";
        yawRadians: number;
        toleranceRadians: number;
      };
    };

export interface HumanoidPolicyState {
  jointPositions: ArrayLike<number>;
  jointVelocities: ArrayLike<number>;
  rootQuaternion: readonly [w: number, x: number, y: number, z: number];
  rootAngularVelocity: readonly [x: number, y: number, z: number];
  environment?: HumanoidPolicyEnvironmentState | undefined;
}

interface HumanoidLearnedPolicyDescriptor {
  protocol: "humanoid-learned-policy-v1";
  runtime: string;
  observationSpace: {
    protocol: string;
    size: number;
  };
  actionSpace: {
    protocol: string;
    size: number;
  };
  observationFeatures?: HumanoidPolicyObservationFeature[] | undefined;
  capabilities: HumanoidLearnedPolicyCapability[];
}

export interface HumanoidControllerDescriptor {
  protocol: "humanoid-controller-v1";
  implementation: string;
  actuation: "joint_position_pd";
  controlStepSeconds: number;
  physicsStepSeconds: number;
  commandResponseHorizonSeconds?: number | undefined;
  minimumEffectivePlanarSpeedMetersPerSecond?: number | undefined;
  learnedPolicy?: HumanoidLearnedPolicyDescriptor | undefined;
  capabilityRouting?: {
    protocol: "humanoid-controller-capability-routing-v1";
    strategy: "declared_capabilities";
    fallback: {
      mode: "reference_control";
      implementation: string;
    };
  } | undefined;
}

export interface HumanoidJointPositionCommand {
  kind: "joint_position_pd";
  positions: Float64Array;
  stiffness: Float64Array;
  damping: Float64Array;
}

export interface HumanoidControllerState {
  protocol: "humanoid-controller-state-v1";
  version: 1;
  implementation: string;
  payload: JsonValue;
}

export interface HumanoidControllerInferenceOptions {
  trackedJointPolicyCommand?: "measured" | "neutral";
  taskCommand?: HumanoidControllerTaskCommand | undefined;
}

export interface HumanoidWholeBodyController {
  readonly descriptor: HumanoidControllerDescriptor;
  reset(
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options?: HumanoidControllerInferenceOptions
  ): void;
  infer(
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options?: HumanoidControllerInferenceOptions
  ): Promise<HumanoidJointPositionCommand>;
  advanceHistory(
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options?: HumanoidControllerInferenceOptions
  ): void;
  captureState(): HumanoidControllerState;
  restoreState(state: HumanoidControllerState): void;
  dispose(): Promise<void>;
}

export type HumanoidWholeBodyControllerFactory =
  () => Promise<HumanoidWholeBodyController>;
