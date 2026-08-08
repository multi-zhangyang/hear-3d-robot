import type { Vec3 } from "../../domain/schema.js";
import type { NavigationPlan } from "../navigation.js";
import type {
  HumanoidMotionExecutionProgress,
  HumanoidMotionOptionExecutionState
} from "./checkpoint.js";
import {
  humanoidMotionArtifactSha256,
  type HumanoidMotionArtifact
} from "./motion-artifact.js";
import type { HumanoidMotionPlan } from "./motion-plan.js";
import {
  humanoidMotionRolloutSha256,
  type HumanoidMotionRollout
} from "./motion-rollout.js";
import { humanoidMotionOptionContractSha256 } from "./motion-option.js";
import {
  completeHumanoidPhysicalSafetyEvidence,
  type HumanoidPhysicalSafetyEvidence
} from "./physical-safety.js";
import type { HumanoidPlanTerminal } from "./execution-terminal.js";
import type { HumanoidNavigationExecutionProgress } from "./navigation-execution.js";
import type {
  HumanoidCarriedObjectBindingSet,
  HumanoidCarriedObjectContinuationEvidence,
  HumanoidCarriedObjectUnauthorizedContact
} from "./carried-object-binding.js";
import type { HumanoidCarryTaskSpaceTarget } from "./carry-task-space-servo.js";
import type { HumanoidNavigationArrivalHeading } from "./navigation-arrival.js";

export interface StoredHumanoidMotionPlan {
  plan: HumanoidMotionPlan;
  artifact: HumanoidMotionArtifact;
  rollout: HumanoidMotionRollout | null;
  retainTerminalJointTracking: boolean;
  createdRevision: number;
  validatedRevision: number;
  validatedStateSha256: string;
  expiresRevision: number;
  intentSha256: string;
  revalidationCount: number;
  terminal: HumanoidPlanTerminal | null;
  option: HumanoidMotionOptionExecutionState | null;
  carriedObjectBindings: HumanoidCarriedObjectBindingSet;
  carriedObjectTaskSpaceTargets: HumanoidCarryTaskSpaceTarget[];
  carriedObjectContinuation: HumanoidCarriedObjectContinuationEvidence | null;
  carriedObjectUnauthorizedContacts: HumanoidCarriedObjectUnauthorizedContact[];
  progress: HumanoidMotionExecutionProgress;
}

export interface StoredHumanoidNavigationPlan {
  id: string;
  plan: NavigationPlan;
  requestedTarget: Vec3;
  requestedArrivalHeading: HumanoidNavigationArrivalHeading | null;
  arrivalHeading: HumanoidNavigationArrivalHeading | null;
  acceptedPositionToleranceMeters: number | null;
  releaseJointTracking: boolean;
  createdRevision: number;
  validatedRevision: number;
  validatedStateSha256: string;
  expiresRevision: number;
  intentSha256: string;
  revalidationCount: number;
  carriedObjectBindings: HumanoidCarriedObjectBindingSet;
  carriedObjectTaskSpaceTargets: HumanoidCarryTaskSpaceTarget[];
  carriedObjectContinuation: HumanoidCarriedObjectContinuationEvidence | null;
  carriedObjectUnauthorizedContacts: HumanoidCarriedObjectUnauthorizedContact[];
  progress: HumanoidNavigationExecutionProgress;
  terminal: HumanoidPlanTerminal | null;
}

export function isTerminalMotionOption(
  option: HumanoidMotionOptionExecutionState
): boolean {
  return option.status === "succeeded"
    || option.status === "failed"
    || option.status === "goal_unmet";
}

export function storedMotionPhysicalSafety(
  stored: StoredHumanoidMotionPlan
): HumanoidPhysicalSafetyEvidence | undefined {
  return stored.progress.physicalSafety
    ? completeHumanoidPhysicalSafetyEvidence(stored.progress.physicalSafety)
    : undefined;
}

export function assertMotionOptionIntegrity(
  stored: StoredHumanoidMotionPlan
): void {
  const option = stored.option;
  if (!option) return;
  if (!stored.rollout) {
    throw new Error("Humanoid motion option is missing its validated rollout");
  }
  const certificate = option.certificate;
  const predictedFrame = stored.artifact.frames[
    certificate.predicted_termination_frame - 1
  ];
  const valid = certificate.artifact_sha256
      === humanoidMotionArtifactSha256(stored.artifact)
    && certificate.contract_sha256
      === humanoidMotionOptionContractSha256(option.contract)
    && option.monitor.contractSha256 === certificate.contract_sha256
    && option.successStreak === option.monitor.terminalStableSteps
    && certificate.rollout_sha256
      === humanoidMotionRolloutSha256(stored.rollout)
    && certificate.rollout_frame_count === stored.rollout.frames.length
    && certificate.rollout_frame_count === certificate.validated_frame_limit
    && certificate.drift_consecutive_steps
      === stored.rollout.limits.consecutive_steps
    && certificate.validated_frame_limit <= stored.artifact.frames.length
    && certificate.predicted_termination_frame === certificate.validated_frame_limit
    && certificate.predicted_termination_frame >= certificate.stable_steps
    && certificate.predicted_termination_frame <= certificate.validated_frame_limit
    && certificate.stable_steps === option.contract.stable_steps
    && (certificate.physical_safety === undefined
      || certificate.physical_safety.frame_count === certificate.validated_frame_limit
        && certificate.physical_safety.first_frame === 1
        && certificate.physical_safety.last_frame === certificate.validated_frame_limit)
    && predictedFrame !== undefined
    && Math.abs(predictedFrame.atSeconds - certificate.predicted_at_seconds) <= 1e-9;
  if (!valid) {
    throw new Error("Humanoid motion option certificate integrity check failed");
  }
}
