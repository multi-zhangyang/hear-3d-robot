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
import type { HumanoidControllerTaskCommand } from "./whole-body-controller.js";
import type {
  HumanoidControllerTaskGoal
} from "./whole-body-controller.js";
import type {
  HumanoidLearnedPolicyCapability
} from "../../domain/humanoid-policy.js";
import type { HumanoidEndEffectorBody } from "./task-space-targets.js";
import type { HumanoidTaskSpaceServoTarget } from "./task-space-servo.js";
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
    taskId?: string;
    taskGoal?: HumanoidControllerTaskGoal | null;
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
      : "measured",
    ...((taskSpaceTargets?.length ?? 0) > 0
      || (options.carryTaskSpaceTargets?.length ?? 0) > 0
      || (options.graspTargets?.length ?? 0) > 0
      ? {
          taskCommand: controllerTaskCommand({
            taskId: options.taskId ?? "motion-option",
            taskGoal: options.taskGoal ?? null,
            taskSpaceTargets: taskSpaceTargets ?? [],
            carryTaskSpaceTargets: options.carryTaskSpaceTargets ?? [],
            graspTargets: options.graspTargets ?? []
          })
        }
      : {})
  });
  return {
    reference,
    snapshot,
    ...(graspServoEvidence ? { graspServoEvidence } : {})
  };
}

function controllerTaskCommand(input: {
  taskId: string;
  taskGoal: HumanoidControllerTaskGoal | null;
  taskSpaceTargets: readonly HumanoidTaskSpaceServoTarget[];
  carryTaskSpaceTargets: readonly HumanoidCarryTaskSpaceTarget[];
  graspTargets: readonly G1ContactAwareGraspTarget[];
}): HumanoidControllerTaskCommand {
  return {
    protocol: "humanoid-controller-task-v1",
    taskId: input.taskId,
    source: "motion_option",
    requestedCapabilities: requestedPolicyCapabilities(input),
    goal: input.taskGoal ? structuredClone(input.taskGoal) : null,
    endEffectors: [
      ...input.taskSpaceTargets.map((target) => ({
        body: target.body,
        frame: target.frame,
        position: { ...target.position },
        tolerance: target.tolerance,
        ...(target.orientation && target.orientationTolerance !== undefined
          ? {
              orientation: { ...target.orientation },
              orientationTolerance: target.orientationTolerance
            }
          : {})
      })),
      ...input.carryTaskSpaceTargets.map((target) => ({
        body: target.body,
        frame: target.frame,
        position: { ...target.position },
        tolerance: target.tolerance,
        orientation: { ...target.orientation },
        orientationTolerance: target.orientationTolerance
      }))
    ],
    grasps: input.graspTargets.map((target) => ({
      objectId: target.objectId,
      hand: target.hand,
      minimumNormalForceN: target.minimumNormalForceN,
      minimumDistinctContactSurfaces:
        target.minimumDistinctContactSurfaces ?? 1
    }))
  };
}

function requestedPolicyCapabilities(input: {
  taskGoal: HumanoidControllerTaskGoal | null;
  taskSpaceTargets: readonly HumanoidTaskSpaceServoTarget[];
  carryTaskSpaceTargets: readonly HumanoidCarryTaskSpaceTarget[];
  graspTargets: readonly G1ContactAwareGraspTarget[];
}): HumanoidLearnedPolicyCapability[] {
  const capabilities = new Set<HumanoidLearnedPolicyCapability>();
  if (input.taskSpaceTargets.length > 0
    || input.carryTaskSpaceTargets.length > 0) {
    capabilities.add("joint_reference_tracking");
  }
  const contactGoal = input.taskGoal?.protocol === "humanoid-controller-motion-goal-v1"
    && input.taskGoal.predicates.some(({ type }) => [
      "body_contact_object",
      "hand_contact_object",
      "hand_contact_object_any",
      "hand_contact_object_region",
      "body_contact_solid",
      "hand_contact_solid",
      "grasp_verified",
      "object_released",
      "object_settled_on_support",
      "articulation_state",
      "articulation_displaced"
    ].includes(type));
  if (input.graspTargets.length > 0 || contactGoal) {
    capabilities.add("contact_rich_manipulation");
  }
  const hands = new Set(input.graspTargets.map(({ hand }) => hand));
  for (const target of [
    ...input.taskSpaceTargets,
    ...input.carryTaskSpaceTargets
  ]) {
    if (target.body.startsWith("left_wrist")) hands.add("left");
    if (target.body.startsWith("right_wrist")) hands.add("right");
  }
  if (hands.size > 1 && capabilities.has("contact_rich_manipulation")) {
    capabilities.add("bimanual_manipulation");
  }
  return [...capabilities];
}
