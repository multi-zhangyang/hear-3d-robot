import {
  g1HandObjectContactsForTarget,
  type G1ContactAwareGraspTarget
} from "./contact-aware-grasp-servo.js";
import type { HumanoidReference } from "./reference.js";
import type { HumanoidSimulationSnapshot } from "./simulation.js";
import type { HumanoidTaskSpaceServoTarget } from "./task-space-servo.js";
import { humanoidEndEffectorTrackingJointIndexes } from "./task-space-targets.js";

const MINIMUM_TASK_TRACKING_SCALE = 0.3;
const COMPLIANCE_ACTIVATION_FORCE_MULTIPLIER = 2;
const FULL_COMPLIANCE_FORCE_MULTIPLIER = 8;

export function contactAwareTaskSpaceCompliance(input: {
  reference: HumanoidReference;
  snapshot: HumanoidSimulationSnapshot;
  taskSpaceTargets: readonly HumanoidTaskSpaceServoTarget[];
  graspTargets: readonly G1ContactAwareGraspTarget[];
}): HumanoidReference {
  const reference = cloneReference(input.reference);
  for (const target of input.taskSpaceTargets) {
    const hand = target.body === "left_wrist_yaw_link"
      ? "left" as const
      : target.body === "right_wrist_yaw_link"
        ? "right" as const
        : null;
    if (!hand) continue;
    const grasp = input.graspTargets.find((candidate) => (
      candidate.hand === hand && candidate.referenceRelativePose === undefined
    ));
    if (!grasp) continue;
    const maximumForce = Math.max(0, ...g1HandObjectContactsForTarget(
      input.snapshot,
      grasp
    ).filter((contact) => contact.hand === hand)
      .map((contact) => contact.normalForce));
    const activation = grasp.minimumNormalForceN
      * COMPLIANCE_ACTIVATION_FORCE_MULTIPLIER;
    const full = grasp.minimumNormalForceN * FULL_COMPLIANCE_FORCE_MULTIPLIER;
    const progress = clamp01((maximumForce - activation) / (full - activation));
    const scale = 1 - progress * (1 - MINIMUM_TASK_TRACKING_SCALE);
    for (const index of humanoidEndEffectorTrackingJointIndexes(
      target.body,
      target.orientation !== undefined,
      target.kinematicScope ?? "arm_only"
    )) {
      reference.jointTrackingWeights[index] = Math.min(
        reference.jointTrackingWeights[index]!,
        scale
      );
    }
  }
  return reference;
}

function cloneReference(reference: HumanoidReference): HumanoidReference {
  return {
    ...reference,
    jointPositions: reference.jointPositions.slice(),
    jointVelocities: reference.jointVelocities.slice(),
    jointTrackingWeights: reference.jointTrackingWeights.slice(),
    rootVelocity: [...reference.rootVelocity]
  };
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}
