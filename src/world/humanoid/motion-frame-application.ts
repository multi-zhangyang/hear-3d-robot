import {
  hydrateHumanoidReference,
  type HumanoidMotionArtifactFrame
} from "./motion-artifact.js";
import {
  contactAwareG1GraspTargets,
  type G1ContactAwareGraspServoEvidence,
  type G1ContactAwareGraspTarget
} from "./contact-aware-grasp-servo.js";
import {
  applyHumanoidCarryTaskSpaceServo,
  type HumanoidCarryTaskSpaceTarget
} from "./carry-task-space-servo.js";
import type { HumanoidReference } from "./reference.js";
import type {
  HumanoidSimulation,
  HumanoidSimulationSnapshot
} from "./simulation.js";
import type { HumanoidEndEffectorBody } from "./task-space-targets.js";

export interface HumanoidMotionFrameApplication {
  reference: HumanoidReference;
  snapshot: HumanoidSimulationSnapshot;
  graspServoEvidence?: G1ContactAwareGraspServoEvidence;
}

export async function applyHumanoidMotionArtifactFrame(
  simulation: HumanoidSimulation,
  frame: HumanoidMotionArtifactFrame,
  options: {
    graspTargets?: readonly G1ContactAwareGraspTarget[];
    carryTaskSpaceTargets?: readonly HumanoidCarryTaskSpaceTarget[];
  } = {}
): Promise<HumanoidMotionFrameApplication> {
  const artifactReference = hydrateHumanoidReference(frame.reference);
  const modelReference = frame.taskSpaceTargets
    ? simulation.solveEndEffectorTargets(
        artifactReference,
        frame.taskSpaceTargets,
        {
          initialConfiguration: "current",
          preserveTrackingWeights: true
        }
      ).reference
    : artifactReference;
  const modelControlledBodies = new Set<HumanoidEndEffectorBody>(
    frame.taskSpaceTargets?.map((target) => target.body) ?? []
  );
  const reference = applyHumanoidCarryTaskSpaceServo({
    simulation,
    reference: modelReference,
    targets: options.carryTaskSpaceTargets ?? [],
    modelControlledBodies
  });
  let graspServoEvidence: G1ContactAwareGraspServoEvidence | undefined;
  if ("handCommand" in frame) {
    const controlled = options.graspTargets && options.graspTargets.length > 0
      ? contactAwareG1GraspTargets({
          command: frame.handCommand,
          snapshot: simulation.snapshot(),
          targets: options.graspTargets
        })
      : null;
    simulation.setHandJointTargets(frame.handCommand.jointTargets);
    if (controlled) simulation.applyHandServoJointTargets(controlled.jointTargets);
    graspServoEvidence = controlled?.evidence;
  }
  const snapshot = await simulation.step(reference, {
    trackedJointPolicyCommand: Math.hypot(...reference.rootVelocity) > 1e-9
        && ((options.carryTaskSpaceTargets?.length ?? 0) > 0
          || (options.graspTargets?.length ?? 0) > 0)
      ? "neutral"
      : "measured"
  });
  return {
    reference,
    snapshot,
    ...(graspServoEvidence ? { graspServoEvidence } : {})
  };
}
