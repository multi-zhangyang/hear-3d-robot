import {
  hydrateHumanoidReference,
  type HumanoidMotionArtifactFrame
} from "./motion-artifact.js";
import {
  contactAwareG1GraspTargets,
  contactAwareG1WristAdmittanceTargets,
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
import {
  stationKeepingHumanoidReference,
  type HumanoidStationKeepingAnchor
} from "./station-keeping.js";
import { contactAwareTaskSpaceCompliance } from "./contact-aware-task-space-compliance.js";

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
    stationKeepingAnchor?: HumanoidStationKeepingAnchor;
    stationKeepingCommand?: readonly [number, number];
  } = {}
): Promise<HumanoidMotionFrameApplication> {
  const artifactReference = hydrateHumanoidReference(frame.reference);
  let currentSnapshot: HumanoidSimulationSnapshot | undefined;
  const snapshotBeforeStep = (): HumanoidSimulationSnapshot => (
    currentSnapshot ??= simulation.snapshot()
  );
  const taskSpaceTargets = frame.taskSpaceTargets
    ? contactAwareG1WristAdmittanceTargets({
        snapshot: snapshotBeforeStep(),
        taskSpaceTargets: frame.taskSpaceTargets,
        graspTargets: options.graspTargets ?? []
      })
    : undefined;
  const modelReference = frame.taskSpaceTargets
    ? simulation.solveEndEffectorTargets(
        artifactReference,
        taskSpaceTargets!,
        {
          initialConfiguration: "current",
          preserveTrackingWeights: true,
          allowBestEffort: true
        }
      ).reference
    : artifactReference;
  const compliantModelReference = taskSpaceTargets
    ? contactAwareTaskSpaceCompliance({
        reference: modelReference,
        snapshot: snapshotBeforeStep(),
        taskSpaceTargets,
        graspTargets: options.graspTargets ?? []
      })
    : modelReference;
  const modelControlledBodies = new Set<HumanoidEndEffectorBody>(
    taskSpaceTargets?.map((target) => target.body) ?? []
  );
  const taskReference = applyHumanoidCarryTaskSpaceServo({
    simulation,
    reference: compliantModelReference,
    targets: options.carryTaskSpaceTargets ?? [],
    modelControlledBodies
  });
  const reference = options.stationKeepingAnchor
    ? stationKeepingHumanoidReference(
        taskReference,
        snapshotBeforeStep(),
        options.stationKeepingAnchor,
        {
          preserveTrackedLowerBody: taskSpaceTargets?.some((target) => (
            target.body === "left_ankle_roll_link"
              || target.body === "right_ankle_roll_link"
          )) ?? false,
          ...(options.stationKeepingCommand
            ? {
                previousPlanarCommand: options.stationKeepingCommand,
                controlStepSeconds:
                  simulation.controllerDescriptor().controlStepSeconds
              }
            : {})
        }
      )
    : taskReference;
  let graspServoEvidence: G1ContactAwareGraspServoEvidence | undefined;
  if ("handCommand" in frame) {
    const controlled = options.graspTargets && options.graspTargets.length > 0
      ? contactAwareG1GraspTargets({
          command: frame.handCommand,
          snapshot: snapshotBeforeStep(),
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
