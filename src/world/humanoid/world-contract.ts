import type { Quaternion, Vec3 } from "../../domain/schema.js";
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
  HumanoidPolicyFrameSink,
  HumanoidSimulationSnapshot
} from "./simulation.js";
import type { G1HandContactSurfaceName } from "./morphology.js";
import type { G1HandCoordination } from "./hand-coordination.js";
import type { HumanoidNavigationArrivalHeading } from "./navigation-arrival.js";
import type { HumanoidSpatialBeliefObservation } from "./spatial-belief-map.js";
import type {
  HumanoidControllerCapabilityEvidenceSummary,
  HumanoidControllerExecutionState,
  HumanoidWholeBodyControllerFactory
} from "./whole-body-controller.js";
import type {
  HumanoidNavigationCollisionEvidence
} from "./navigation-collision-evidence.js";
import type {
  HumanoidEmbodiedSkillStatus
} from "./embodied-skill-call.js";
import type {
  HumanoidSkillEventSink,
  HumanoidSkillEventStream
} from "./skill-event-stream.js";
import type {
  HumanoidRecoveryExecutionContract
} from "./recovery-execution-contract.js";
import type { HumanoidRecoveryFailure } from "./recovery-execution.js";

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
  spatialBelief: HumanoidSpatialBeliefObservation;
  grasp: HumanoidWorldGraspState;
  interaction: HumanoidInteractionObservation;
  navigation: HumanoidWorldSnapshot["navigation"];
}

export interface HumanoidManipulationReachabilityObservation {
  objectId: string;
  interactionPointId?: string | undefined;
  handSurface: G1HandContactSurfaceName;
  wristWorldTarget: Vec3;
  wristWorldOrientation?: Quaternion;
  ikReferenceReachable: boolean;
  ikResidualMeters: number | null;
}

export interface HumanoidManipulationBasePlacementObservation {
  objectId: string;
  interactionPointId?: string | undefined;
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

export interface HumanoidRecoveryPlanReceipt {
  accepted: boolean;
  planId: string;
  createdRevision: number;
  expiresRevision: number;
  intentSha256: string;
  contractSha256: string;
  channels: HumanoidBodyChannel[];
  contract: HumanoidRecoveryExecutionContract;
  reason?: string;
}

export interface HumanoidExecutionReceipt {
  accepted: boolean;
  code: "motion_completed" | "navigation_completed" | "plan_stale"
    | "motion_failed" | "navigation_blocked" | "motion_option_succeeded"
    | "motion_goal_unmet" | "motion_goal_uncertain"
    | "motion_execution_drifted" | "motion_constraint_violated"
    | "plan_revalidation_failed" | "recovery_completed" | "recovery_failed";
  frames: number;
  finalSnapshot: HumanoidWorldSnapshot;
  terminalResultSha256?: string;
  detail: {
    failures?: HumanoidMotionValidation["failures"];
    reason?: string;
    travelledDistance?: number;
    online_replans?: Array<{
      attempt: number;
      trigger: string;
      failure_class: "dynamic_obstruction";
      world_revision: number;
      accepted: boolean;
      reason: string | null;
      waypoint_count: number;
      budget: {
        tier: "local_controller_recovery";
        limit: number;
        used_before: number;
        used_after: number;
        remaining_after: number;
        model_calls_consumed: 0;
      };
      blocking_contacts?: HumanoidNavigationCollisionEvidence[];
    }>;
    online_replan_budget?: {
      tier: "local_controller_recovery";
      limit: number;
      used: number;
      remaining: number;
      terminal_failure_class: "dynamic_obstruction" | "unsafe_state"
        | "semantic_recovery" | "budget_exhausted" | null;
      model_calls_consumed: 0;
    };
    blocking_contacts?: HumanoidNavigationCollisionEvidence[];
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
    skill_status?: HumanoidEmbodiedSkillStatus;
    controller_routing?: {
      execution: NonNullable<HumanoidControllerExecutionState["routing"]> | null;
      capability_evidence: HumanoidControllerCapabilityEvidenceSummary[];
    };
    recovery?: {
      contract_sha256: string;
      safety_interrupt_id: string;
      recovery_implementation: string | null;
      stable_steps: number;
      required_stable_steps: number;
      handoff_steps: number;
      required_handoff_steps: number;
      handoff_completed: boolean;
      failure: HumanoidRecoveryFailure | null;
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
  acceptedPositionToleranceMeters: number | null;
  waypoints: Vec3[];
  distance: number;
  remainingDistance: number;
  partialEndpoint?: Vec3;
  previewFrames?: number;
  previewTravelledDistance?: number;
  blockingContacts?: HumanoidNavigationCollisionEvidence[];
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
  /**
   * Lightweight, ordered per-controller-frame projection. Long executions use
   * this between durable cuts so trajectory/Goal state advances without
   * serializing the complete MuJoCo checkpoint on every control step.
   */
  progressSink?: HumanoidFrameSink;
  persistenceSink?: HumanoidPersistenceSink;
  /** Persist every Nth world revision; terminal and replan cuts stay immediate. */
  persistenceFrameStride?: number;
  /** World revision at physical admission, used to align periodic cut cadence. */
  persistenceStartWorldRevision?: number;
  skillEventSink?: HumanoidSkillEventSink;
  /** Internal shared lifecycle for a deterministic multi-plan Skill horizon. */
  skillEventStream?: HumanoidSkillEventStream;
  /** The horizon owner, rather than an individual plan, emits progress. */
  deferSkillProgress?: boolean;
  /** The horizon owner, rather than an individual plan, emits the terminal. */
  deferSkillTerminal?: boolean;
  /** The horizon owner records one controller outcome for the complete Skill. */
  deferSkillControllerOutcome?: boolean;
  /** Continuous semantic window shared by deterministic multi-plan horizons. */
  skillWindow?: {
    maximumSteps: number;
    stepOffset: number;
  };
  policyFrameSink?: HumanoidPolicyFrameSink;
  signal?: AbortSignal;
}

export interface HumanoidWorldOptions {
  controllerFactory?: HumanoidWholeBodyControllerFactory;
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
