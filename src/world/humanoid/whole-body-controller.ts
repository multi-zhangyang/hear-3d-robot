import type { JsonValue, Quaternion, Vec3 } from "../../domain/schema.js";
import type {
  HumanoidLearnedPolicyCapability
} from "../../domain/humanoid-policy.js";
import type {
  HumanoidEmbodiedSkillCall,
  HumanoidEmbodiedSkillContract,
  HumanoidEmbodiedSkillIdentity
} from "./embodied-skill-call.js";
import type {
  HumanoidPolicyAdmissionAssessment,
  HumanoidPolicyCapabilityPosterior,
  HumanoidPolicySkillFamily
} from "./policy-capability-evidence.js";
import type {
  G1HandCoordination
} from "./hand-coordination.js";
import type {
  HumanoidHandPolicyAuthorityAssessment,
  HumanoidHandPolicyAuthorityState
} from "./hand-policy-authority.js";
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
  rootPosition?: Vec3 | undefined;
  endEffectors: Readonly<Record<string, {
    position: Vec3;
    rotation: Quaternion;
    linearVelocity?: Vec3 | undefined;
    angularVelocity?: Vec3 | undefined;
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
    shape?: "box" | "sphere" | "cylinder" | "capsule" | undefined;
    size?: Vec3 | undefined;
    massKg?: number | undefined;
    friction?: {
      sliding: number;
      torsional: number;
      rolling: number;
    } | undefined;
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
  zones?: ReadonlyArray<{
    id: string;
    center: Vec3;
    size: Vec3;
  }> | undefined;
  feet?: {
    left: {
      touching: boolean;
      normalForce: number;
    };
    right: {
      touching: boolean;
      normalForce: number;
    };
  } | undefined;
  centerOfMass?: Vec3 | undefined;
  centerOfMassVelocity?: Vec3 | undefined;
}

export type HumanoidControllerTaskCommand = HumanoidEmbodiedSkillCall;
export type HumanoidControllerTaskGoal = HumanoidEmbodiedSkillContract;

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
    strategy: "declared_capabilities" | "capability_evidence";
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
  handSynergy?: HumanoidHandSynergyCommand | undefined;
}

export interface HumanoidHandSynergyCommand {
  protocol: "humanoid-authorized-hand-synergy-command-v1";
  authority: HumanoidHandPolicyAuthorityState;
  action: Float64Array;
  coordination: G1HandCoordination;
  maximumClosingJointLeadRadians: 0.25;
}

export interface HumanoidControllerTensorTrace {
  protocol: "humanoid-controller-tensor-trace-v1";
  role: "direct" | "primary" | "fallback";
  implementation: string;
  observation: {
    protocol: string;
    values: number[];
  };
  action: {
    protocol: string;
    values: number[];
  };
}

export interface HumanoidControllerInferenceTrace {
  protocol: "humanoid-controller-inference-trace-v1";
  implementation: string;
  route: "direct" | "primary" | "fallback" | "upper_body_overlay";
  components: HumanoidControllerTensorTrace[];
}

export interface HumanoidControllerState {
  protocol: "humanoid-controller-state-v1";
  version: 1;
  implementation: string;
  payload: JsonValue;
}

export interface HumanoidControllerExecutionState {
  protocol: "humanoid-controller-execution-v1";
  mode: "learned_policy" | "reference_control" | "hybrid_control";
  activeImplementation: string;
  transition: {
    fromImplementation: string;
    toImplementation: string;
    progress: number;
    durationSeconds: number;
  } | null;
  routing?: {
    callId: string;
    route: "primary" | "fallback" | "upper_body_overlay";
    assessment: HumanoidPolicyAdmissionAssessment | null;
    attribution: {
      primarySteps: number;
      fallbackSteps: number;
      upperBodyOverlaySteps: number;
      memoryBridgeSteps: number;
    };
    memoryBridge: {
      protocol: "humanoid-policy-memory-bridge-v1";
      phase: "guiding" | "completed" | "timed_out" | "aborted";
      trigger: "entry_state_ood";
      completedSteps: number;
      maximumSteps: number;
      stableSteps: number;
      requiredStableSteps: number;
      progress: number;
      entryStateOodScore: number;
      jointPrototypeRmsError: number;
      maximumJointVelocity: number;
    } | null;
  } | undefined;
}

export interface HumanoidControllerSkillOutcome {
  protocol: "humanoid-controller-skill-outcome-v1";
  identity: HumanoidEmbodiedSkillIdentity;
  outcome: "succeeded" | "failed" | "interrupted";
  terminalReason: string;
}

export interface HumanoidControllerCapabilityEvidenceSummary {
  implementation: string;
  skillFamily: HumanoidPolicySkillFamily;
  posterior: HumanoidPolicyCapabilityPosterior;
}

export interface HumanoidControllerInferenceOptions {
  trackedJointPolicyCommand?: "measured" | "neutral";
  taskCommand?: HumanoidEmbodiedSkillCall | undefined;
  handPolicyAuthority?: HumanoidHandPolicyAuthorityAssessment | undefined;
}

export interface HumanoidWholeBodyController {
  readonly descriptor: HumanoidControllerDescriptor;
  executionState?(): HumanoidControllerExecutionState;
  inferenceTrace?(): HumanoidControllerInferenceTrace | null;
  capabilityEvidence?(): readonly HumanoidControllerCapabilityEvidenceSummary[];
  recordSkillOutcome?(outcome: HumanoidControllerSkillOutcome): void;
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
