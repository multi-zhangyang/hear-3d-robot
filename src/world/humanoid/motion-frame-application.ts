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
  HumanoidPolicyFrameSink,
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
import {
  inverseQuaternion,
  rotateVector
} from "../geometry.js";
import { humanoidControllerTaskCapabilities } from
  "./controller-task-capabilities.js";
import {
  HumanoidEmbodiedSkillCallSchema,
  legacyHumanoidEmbodiedSkillIdentity,
  type HumanoidEmbodiedSkillIdentity
} from "./embodied-skill-call.js";
import type { HumanoidContactConstraint } from "./motion-plan-schema.js";
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
    skillIdentity?: HumanoidEmbodiedSkillIdentity;
    authority?: { worldFrame: number; worldRevision: number };
    controlWindow?: { maximumSteps: number; stepIndex: number };
    authorizedContacts?: readonly HumanoidContactConstraint[];
    policyFrameSink?: HumanoidPolicyFrameSink;
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
    taskCommand: controllerTaskCommand({
      taskId: options.taskId ?? "motion-option",
      taskGoal: options.taskGoal ?? null,
      ...(options.skillIdentity
        ? { skillIdentity: options.skillIdentity }
        : {}),
      ...(options.authority ? { authority: options.authority } : {}),
      ...(options.controlWindow
        ? { controlWindow: options.controlWindow }
        : {}),
      authorizedContacts: options.authorizedContacts ?? [],
      ...(taskSpaceTargets || (options.carryTaskSpaceTargets?.length ?? 0) > 0
        ? { snapshot: snapshotBeforeStep() }
        : {}),
      controlStepSeconds: simulation.controllerDescriptor().controlStepSeconds,
      reference,
      taskSpaceTargets: taskSpaceTargets ?? [],
      carryTaskSpaceTargets: options.carryTaskSpaceTargets ?? [],
      graspTargets: options.graspTargets ?? []
    }),
    ...(options.policyFrameSink
      ? { policyFrameSink: options.policyFrameSink }
      : {})
  });
  return {
    reference,
    snapshot,
    ...(graspServoEvidence ? { graspServoEvidence } : {})
  };
}

export function controllerTaskCommand(input: {
  taskId: string;
  taskGoal: HumanoidControllerTaskGoal | null;
  skillIdentity?: HumanoidEmbodiedSkillIdentity;
  authority?: { worldFrame: number; worldRevision: number };
  controlWindow?: { maximumSteps: number; stepIndex: number };
  authorizedContacts: readonly HumanoidContactConstraint[];
  recoverySafety?: boolean;
  snapshot?: HumanoidSimulationSnapshot;
  controlStepSeconds: number;
  reference: HumanoidReference;
  taskSpaceTargets: readonly HumanoidTaskSpaceServoTarget[];
  carryTaskSpaceTargets: readonly HumanoidCarryTaskSpaceTarget[];
  graspTargets: readonly G1ContactAwareGraspTarget[];
}): HumanoidControllerTaskCommand {
  const authority = input.authority ?? {
    worldFrame: input.skillIdentity?.observedFrame ?? 0,
    worldRevision: input.skillIdentity?.observedWorldRevision ?? 0
  };
  const identity = input.skillIdentity ?? legacyHumanoidEmbodiedSkillIdentity({
    callId: input.taskId,
    runtimeKind: "legacy_motion",
    phase: "execute_reference",
    observedFrame: authority.worldFrame,
    observedWorldRevision: authority.worldRevision
  });
  const maximumSteps = input.controlWindow?.maximumSteps ?? 1;
  const stepIndex = input.controlWindow?.stepIndex ?? 0;
  const endEffectors = [
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
  ];
  const leftWristPositionPelvis = wristCommandInPelvis(
    endEffectors,
    "left_wrist_yaw_link",
    input.snapshot
  );
  const rightWristPositionPelvis = wristCommandInPelvis(
    endEffectors,
    "right_wrist_yaw_link",
    input.snapshot
  );
  return HumanoidEmbodiedSkillCallSchema.parse({
    protocol: "humanoid-embodied-skill-call-v2",
    identity,
    authority: {
      source: identity.runtimeKind === "semantic_skill"
        ? "agent_harness"
        : "deterministic_runtime",
      ...authority
    },
    window: {
      mode: "autonomous_closed_loop",
      replanPolicy: "event_driven",
      controlStepSeconds: input.controlStepSeconds,
      maximumSteps,
      stepIndex,
      remainingSteps: Math.max(0, maximumSteps - stepIndex)
    },
    requestedCapabilities: humanoidControllerTaskCapabilities(
      input.reference,
      requestedPolicyCapabilities(input)
    ),
    command: {
      baseTwist: {
        forwardMetersPerSecond: input.reference.rootVelocity[0],
        lateralMetersPerSecond: input.reference.rootVelocity[1],
        yawRadiansPerSecond: input.reference.rootYawVelocity
      },
      rootHeightMeters: input.reference.rootHeight,
      leftWristPositionPelvis,
      rightWristPositionPelvis,
      endEffectors,
      grasps: input.graspTargets.map((target) => ({
        objectId: target.objectId,
        hand: target.hand,
        minimumNormalForceN: target.minimumNormalForceN,
        minimumDistinctContactSurfaces:
          target.minimumDistinctContactSurfaces ?? 1
      }))
    },
    contract: input.taskGoal ? structuredClone(input.taskGoal) : null,
    safety: input.recoverySafety
      ? {
          authorizedContacts: input.authorizedContacts.map((contact) => ({ ...contact })),
          stopOnFall: false as const,
          stopOnUnauthorizedContact: true as const,
          stopOnContractViolation: true as const,
          recoveryTerrainContact: true as const
        }
      : {
          authorizedContacts: input.authorizedContacts.map((contact) => ({ ...contact })),
          stopOnFall: true as const,
          stopOnUnauthorizedContact: true as const,
          stopOnContractViolation: true as const
        },
    feedback: {
      mode: "event_driven",
      progressDelta: 0.1,
      events: [
        "accepted",
        "progress",
        "succeeded",
        "failed",
        "interrupted",
        "environment_changed"
      ]
    }
  });
}

function requestedPolicyCapabilities(input: {
  taskGoal: HumanoidControllerTaskGoal | null;
  taskSpaceTargets: readonly HumanoidTaskSpaceServoTarget[];
  carryTaskSpaceTargets: readonly HumanoidCarryTaskSpaceTarget[];
  graspTargets: readonly G1ContactAwareGraspTarget[];
}): HumanoidLearnedPolicyCapability[] {
  const capabilities = new Set<HumanoidLearnedPolicyCapability>();
  if (input.taskGoal?.protocol === "humanoid-embodied-recovery-contract-v1") {
    capabilities.add("whole_body_recovery");
  }
  if (input.taskSpaceTargets.length > 0
    || input.carryTaskSpaceTargets.length > 0) {
    capabilities.add("joint_reference_tracking");
  }
  const contactGoal = input.taskGoal?.protocol === "humanoid-embodied-motion-contract-v1"
    && input.taskGoal.option.predicates.some(({ type }) => [
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

function wristCommandInPelvis(
  targets: ReadonlyArray<{
    body: string;
    frame: "world" | "pelvis" | "torso";
    position: { x: number; y: number; z: number };
  }>,
  body: "left_wrist_yaw_link" | "right_wrist_yaw_link",
  snapshot: HumanoidSimulationSnapshot | undefined
): { x: number; y: number; z: number } | null {
  const target = targets.find((candidate) => candidate.body === body);
  if (!target || target.frame === "torso") return null;
  if (target.frame === "pelvis") return { ...target.position };
  if (!snapshot) {
    throw new Error("A world-frame wrist command requires a physical snapshot");
  }
  return rotateVector(
    inverseQuaternion(snapshot.rootRotation),
    {
      x: target.position.x - snapshot.rootPosition.x,
      y: target.position.y - snapshot.rootPosition.y,
      z: target.position.z - snapshot.rootPosition.z
    }
  );
}
