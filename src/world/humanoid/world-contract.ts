import type { Vec3 } from "../../domain/schema.js";
import type { ScenarioChunkDeltaState } from "../../domain/scenario-chunk-delta-schema.js";
import type { HumanoidAuthorityIdentity } from "./authority-state.js";
import type {
  HumanoidMotionOptionExecutionState
} from "./checkpoint.js";
import type { HumanoidWorldCheckpoint } from "./checkpoint.js";
import type {
  HumanoidBodyChannel,
  HumanoidMotionGenerator,
  HumanoidMotionOptionCertificate,
  HumanoidMotionValidation
} from "./motion-plan.js";
import type { HumanoidMotionGeneratorDescriptor } from "./motion-generator-contract.js";
import type { humanoidMotionArtifactSummary } from "./motion-artifact.js";
import type { HumanoidMotionDriftEvidence } from "./motion-rollout.js";
import type { HumanoidMotionOptionContract } from "./motion-option.js";
import type { HumanoidObjectToken } from "./object-memory.js";
import type { HumanoidSolidToken } from "./solid-observation.js";
import type { HumanoidPhysicalSafetyEvidence } from "./physical-safety.js";
import type { HumanoidWorldGraspState } from "./grasp-world-state.js";
import type { HumanoidInteractionObservation } from "./interaction-observation.js";
import type {
  HumanoidCarriedObjectBindingSet,
  HumanoidCarriedObjectContinuationEvidence,
  HumanoidCarriedObjectUnauthorizedContact
} from "./carried-object-binding.js";
import type {
  HumanoidHandSurfaceObservation,
  HumanoidObjectSensorSnapshot,
  HumanoidSimulationSnapshot
} from "./simulation.js";
import type { G1HandContactSurfaceName } from "./morphology.js";
import type { G1HandCoordination } from "./hand-coordination.js";
import type { HumanoidNavigationArrivalHeading } from "./navigation-arrival.js";

export interface HumanoidWorldSnapshot {
  frame: number;
  worldRevision: number;
  motionGenerator: HumanoidMotionGeneratorDescriptor;
  physicalSafety?: {
    planId: string;
    evidence: HumanoidPhysicalSafetyEvidence;
  } | undefined;
  robot: HumanoidSimulationSnapshot;
  grasp: HumanoidWorldGraspState;
  navigation: {
    planId: string | null;
    status: "idle" | "planned" | "executing" | "completed" | "blocked";
    target: Vec3 | null;
    waypoints: Vec3[];
    waypointIndex: number | null;
  };
}

export interface HumanoidWorldObservation {
  frame: number;
  worldRevision: number;
  motionGenerator: HumanoidMotionGeneratorDescriptor;
  sensor: HumanoidObjectSensorSnapshot["sensor"];
  robot: Omit<HumanoidSimulationSnapshot, "objects">;
  handCoordination: G1HandCoordination;
  handSurfaces: HumanoidHandSurfaceObservation[];
  manipulationReachability: HumanoidManipulationReachabilityObservation[];
  manipulationBasePlacements: HumanoidManipulationBasePlacementObservation[];
  objectTokens: HumanoidObjectToken[];
  solidTokens: HumanoidSolidToken[];
  grasp: HumanoidWorldGraspState;
  interaction: HumanoidInteractionObservation;
  navigation: HumanoidWorldSnapshot["navigation"];
}

export interface HumanoidManipulationReachabilityObservation {
  objectId: string;
  handSurface: G1HandContactSurfaceName;
  wristWorldTarget: Vec3;
  ikReferenceReachable: boolean;
  ikResidualMeters: number | null;
}

export interface HumanoidManipulationBasePlacementObservation {
  objectId: string;
  handSurface: G1HandContactSurfaceName;
  rootWorldTarget: Vec3;
  rootTranslationWorld: Vec3;
  rootYawRadians: number;
  wristWorldTarget: Vec3;
  ikResidualMeters: number;
}

export interface WholeBodyPlanReceipt {
  accepted: boolean;
  planId: string;
  createdRevision: number;
  validatedStateSha256: string;
  expiresRevision: number;
  intentSha256: string;
  channels: HumanoidBodyChannel[];
  motion: ReturnType<typeof humanoidMotionArtifactSummary> | null;
  validation: HumanoidMotionValidation;
}

interface WholeBodyCandidateEvaluation {
  rank: number;
  planId: string;
  intent: string;
  intentSha256: string;
  channels: HumanoidBodyChannel[];
  motion: ReturnType<typeof humanoidMotionArtifactSummary> | null;
  optionCertificate: HumanoidMotionOptionCertificate | null;
  validation: HumanoidMotionValidation;
}

export interface WholeBodyCandidatePlanReceipt {
  accepted: boolean;
  planId: string;
  selectedCandidateId: string | null;
  selectedRank: number | null;
  createdRevision: number;
  validatedStateSha256: string;
  expiresRevision: number;
  intentSha256: string | null;
  channels: HumanoidBodyChannel[];
  motion: ReturnType<typeof humanoidMotionArtifactSummary> | null;
  option: {
    contract: HumanoidMotionOptionContract;
    certificate: HumanoidMotionOptionCertificate;
  } | null;
  selection: "model_rank_then_physics";
  candidates: WholeBodyCandidateEvaluation[];
}

export interface HumanoidExecutionReceipt {
  accepted: boolean;
  code: "motion_completed" | "navigation_completed" | "plan_stale"
    | "motion_failed" | "navigation_blocked" | "motion_option_succeeded"
    | "motion_goal_unmet" | "motion_goal_uncertain"
    | "motion_execution_drifted" | "motion_constraint_violated"
    | "plan_revalidation_failed";
  frames: number;
  finalSnapshot: HumanoidWorldSnapshot;
  terminalResultSha256?: string;
  detail: {
    failures?: HumanoidMotionValidation["failures"];
    reason?: string;
    travelledDistance?: number;
    motion?: ReturnType<typeof humanoidMotionArtifactSummary>;
    physical_safety?: HumanoidPhysicalSafetyEvidence;
    revalidation?: {
      performed: boolean;
      accepted: boolean;
      intent_sha256: string;
      planning_revision: number;
      previous_validation_revision: number;
      validation_revision: number;
      validation_state_sha256: string;
      admission_state_sha256: string;
      expires_revision: number;
      revalidation_count: number;
    };
    carry?: {
      binding_set: HumanoidCarriedObjectBindingSet;
      continuation: HumanoidCarriedObjectContinuationEvidence | null;
      unauthorized_contacts: HumanoidCarriedObjectUnauthorizedContact[];
    };
    option?: {
      option_id: string;
      status: HumanoidMotionOptionExecutionState["status"];
      termination_reason: HumanoidMotionOptionExecutionState["terminationReason"];
      full_frame_count: number;
      executed_prefix_frame_count: number;
      predicted_termination_frame: number;
      actual_termination_frame: number | null;
      artifact_sha256: string;
      rollout_sha256: string;
      drift_streak: number;
      drift_evidence: HumanoidMotionDriftEvidence | null;
      evidence: HumanoidMotionOptionExecutionState["lastEvidence"];
    };
  };
}

export interface NavigationPlanReceipt {
  accepted: boolean;
  planId: string;
  createdRevision: number;
  validatedStateSha256: string;
  expiresRevision: number;
  intentSha256: string;
  target: Vec3;
  chunkTarget: Vec3;
  requestedArrivalHeading: HumanoidNavigationArrivalHeading | null;
  arrivalHeading: HumanoidNavigationArrivalHeading | null;
  waypoints: Vec3[];
  distance: number;
  remainingDistance: number;
  partialEndpoint?: Vec3;
  previewFrames?: number;
  previewTravelledDistance?: number;
  carry: {
    binding_set_sha256: string;
    bindings: Array<{
      object_id: string;
      hand: "left" | "right";
    }>;
  };
  reason?: string;
}

export type HumanoidFrameSink = (
  snapshot: HumanoidWorldSnapshot
) => void | Promise<void>;

export interface HumanoidWorldPersistenceState {
  world: HumanoidWorldSnapshot;
  worldCheckpoint: HumanoidWorldCheckpoint;
  authority: HumanoidAuthorityIdentity;
}

export type HumanoidPersistenceSink = (
  state: HumanoidWorldPersistenceState
) => void | Promise<void>;

export interface HumanoidExecutionOptions {
  realtime?: boolean;
  retainTerminal?: boolean;
  persistenceSink?: HumanoidPersistenceSink;
  signal?: AbortSignal;
}

export interface HumanoidWorldOptions {
  motionGeneratorFactory?: () => Promise<HumanoidMotionGenerator>;
  planIntentLeaseSeconds?: number;
  scenarioChunks?: ScenarioChunkDeltaState;
}

export interface HumanoidWorldScenarioSynchronizationReceipt {
  changed: boolean;
  chunkRevision: number;
  resourceRebuilt: boolean;
  changedDomains: Array<"geometry" | "objects" | "zones">;
  invalidatedPlanIds: string[];
}
