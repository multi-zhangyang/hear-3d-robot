import { z } from "zod";
import {
  JsonValueSchema,
  type JsonValue,
  type Quaternion,
  type Scenario,
  type Vec3
} from "../../domain/schema.js";
import {
  inverseQuaternion,
  multiplyQuaternion,
  normalizeQuaternion,
  quaternionAngularDistance,
  rotateVector,
  subtract
} from "../geometry.js";
import type { HumanoidBodyName } from "./model.js";
import type { G1HandContactSurfaceName } from "./morphology.js";
import {
  type HumanoidReference
} from "./reference.js";
import {
  HumanoidMotionArtifactSchema,
  humanoidMotionArtifactSha256,
  type HumanoidMotionArtifact
} from "./motion-artifact.js";
import { applyHumanoidMotionArtifactFrame } from "./motion-frame-application.js";
import {
  advanceHumanoidMotionOptionMonitor,
  createHumanoidMotionOptionMonitorState,
  humanoidMotionOptionContractSha256,
  type HumanoidMotionOptionContract,
  type HumanoidMotionOptionDetection,
  type HumanoidMotionOptionMonitorState
} from "./motion-option.js";
import {
  detectHumanoidMotionOptionFromSimulation,
  humanoidMotionOptionDetectorInputFromSimulation
} from "./motion-option-observation.js";
import {
  HumanoidMotionGeneratorDescriptorSchema,
  TASK_SPACE_MOTION_GENERATOR_DESCRIPTOR,
  type HumanoidMotionGeneratorDescriptor
} from "./motion-generator-contract.js";
import {
  captureHumanoidMotionRolloutFrame,
  createHumanoidMotionRollout,
  humanoidMotionRolloutSha256,
  type HumanoidMotionDriftEvidence,
  type HumanoidMotionRollout,
  type HumanoidMotionRolloutFrame
} from "./motion-rollout.js";
import {
  accumulateHumanoidPhysicalSafetyFrame,
  completeHumanoidPhysicalSafetyEvidence,
  createHumanoidPhysicalSafetyAccumulator,
  HumanoidPhysicalSafetyEvidenceSchema
} from "./physical-safety.js";
import {
  type HumanoidEndEffectorTarget,
  type HumanoidSimulation,
  type HumanoidSimulationSnapshot
} from "./simulation.js";
import { HumanoidGraspRegistry } from "./grasp-registry.js";
import {
  contactAwareG1GraspTargetsForBindings,
  contactAwareG1GraspTargetsForOption,
  mergeG1ContactAwareGraspTargets
} from "./contact-aware-grasp-servo.js";
import {
  HumanoidCarriedObjectBindingSetSchema,
  humanoidCarriedObjectContactConstraints,
  humanoidCarriedObjectContinuationEvidence,
  humanoidCarriedObjectUnauthorizedContacts,
  type HumanoidCarriedObjectBindingSet
} from "./carried-object-binding.js";
import {
  HumanoidCarryTaskSpaceTargetsSchema,
  humanoidCarryTaskSpaceTargetsMatchBindings,
  type HumanoidCarryTaskSpaceTarget
} from "./carry-task-space-servo.js";
import {
  authorizeHumanoidCarriedObjectRelease,
  type HumanoidCarriedObjectReleaseAuthority
} from "./carried-object-release.js";
import { assessHumanoidObjectReleased } from "./object-release.js";
import {
  distinctContactBodies,
  distinctContactHandSurfaces,
  humanoidEnvironmentContacts,
  humanoidContactConstraintKey as contactKey,
  humanoidEnvironmentContactKey,
  missingRequiredHumanoidContacts,
  type HumanoidEnvironmentContact
} from "./motion-contact-policy.js";

import {
  HumanoidMotionPlanSchema,
  type HumanoidContactConstraint,
  type HumanoidMotionPlan
} from "./motion-plan-schema.js";
import {
  HumanoidMotionGenerationError,
  plannedHandCommandAtTime,
  TaskSpaceHumanoidMotionGenerator,
  type HumanoidMotionGenerator
} from "./task-space-motion-generator.js";
import { taskSpaceTargets } from "./task-space-motion-targets.js";

export {
  HumanoidContactConstraintSchema,
  HumanoidMotionCandidateBatchSchema,
  HumanoidMotionPlanSchema,
  humanoidGraspContactAuthorizationFailures,
  type HumanoidContactConstraint,
  type HumanoidMotionCandidateBatch,
  type HumanoidMotionPlan
} from "./motion-plan-schema.js";

export {
  TaskSpaceHumanoidMotionGenerator,
  occupiedHumanoidChannels,
  type HumanoidBodyChannel,
  type HumanoidMotionGenerator,
  type HumanoidMotionGeneratorInput
} from "./task-space-motion-generator.js";

export {
  HumanoidEnvironmentContactSchema,
  blockedHumanoidContacts,
  humanoidContactConstraintKey,
  humanoidContactKey,
  humanoidEnvironmentContacts,
  humanoidHandContactKey,
  humanoidHandSolidContactKey,
  humanoidEnvironmentContactKey,
  humanoidSolidContactKey,
  missingRequiredHumanoidContacts
} from "./motion-contact-policy.js";

export const HumanoidMotionOptionCertificateSchema = z.object({
  artifact_sha256: z.string().regex(/^[a-f0-9]{64}$/),
  contract_sha256: z.string().regex(/^[a-f0-9]{64}$/),
  rollout_sha256: z.string().regex(/^[a-f0-9]{64}$/),
  rollout_frame_count: z.number().int().positive(),
  drift_consecutive_steps: z.number().int().positive(),
  validated_frame_limit: z.number().int().positive(),
  predicted_termination_frame: z.number().int().positive(),
  predicted_at_seconds: z.number().finite().positive(),
  stable_steps: z.number().int().positive(),
  evidence: JsonValueSchema,
  physical_safety: HumanoidPhysicalSafetyEvidenceSchema.optional()
}).strict();

export type HumanoidMotionOptionCertificate = z.infer<
  typeof HumanoidMotionOptionCertificateSchema
>;

export interface HumanoidMotionValidationOptions {
  requireFinalSupport?: boolean;
  contactObjectIds?: ReadonlySet<string>;
  contactSolidIds?: ReadonlySet<string>;
  graspRegistry?: HumanoidGraspRegistry;
  worldFrame?: number;
  worldRevision?: number;
  carriedObjectBindings?: HumanoidCarriedObjectBindingSet;
  carriedObjectTaskSpaceTargets?: readonly HumanoidCarryTaskSpaceTarget[];
  motionOption?: {
    contract: HumanoidMotionOptionContract;
    scenario: Scenario;
  };
}

export interface HumanoidMotionValidation {
  feasible: boolean;
  failures: Array<{
    code: "fallen" | "environment_contact" | "required_contact_missing"
      | "unknown_contact_object" | "contact_object_not_currently_visible"
      | "unknown_contact_solid" | "contact_solid_not_currently_visible"
      | "unsupported_finish" | "invalid_reference"
      | "task_space_target_unmet"
      | "motion_goal_already_satisfied" | "motion_goal_unmet"
      | "motion_goal_uncertain" | "motion_constraint_violated"
      | "execution_drift";
    atSeconds: number;
    bodies?: HumanoidBodyName[];
    handSurfaces?: G1HandContactSurfaceName[];
    contacts?: HumanoidEnvironmentContact[];
    constraints?: HumanoidContactConstraint[];
    drift?: HumanoidMotionDriftEvidence;
    taskSpaceTarget?: {
      body: HumanoidEndEffectorTarget["body"];
      frame: HumanoidEndEffectorTarget["frame"];
      target: Vec3;
      achieved: Vec3;
      errorMeters: number;
      toleranceMeters: number;
      orientationTarget?: Quaternion;
      orientationAchieved?: Quaternion;
      orientationErrorRadians?: number;
      orientationToleranceRadians?: number;
      requestedAtSeconds: number;
      observedAtSeconds: number;
    };
    message?: string;
  }>;
  evidence: {
    simulatedSteps: number;
    minimumRootHeight: number;
    minimumUpright: number;
    minimumSupportMargin: number | null;
    travelledDistance: number;
    environmentContactBodies: HumanoidBodyName[];
    environmentContactHandSurfaces: G1HandContactSurfaceName[];
    environmentContacts: HumanoidEnvironmentContact[];
    satisfiedRequiredContacts: HumanoidContactConstraint[];
  };
  finalSnapshot: HumanoidSimulationSnapshot;
}

export interface PreparedHumanoidMotion {
  artifact: HumanoidMotionArtifact | null;
  rollout: HumanoidMotionRollout | null;
  optionCertificate: HumanoidMotionOptionCertificate | null;
  validation: HumanoidMotionValidation;
}

export async function prepareHumanoidMotion(
  simulation: HumanoidSimulation,
  plan: HumanoidMotionPlan,
  baseline: HumanoidReference,
  options: HumanoidMotionValidationOptions = {},
  generator: HumanoidMotionGenerator = new TaskSpaceHumanoidMotionGenerator()
): Promise<PreparedHumanoidMotion> {
  const generationState = simulation.captureState();
  let artifact: HumanoidMotionArtifact | undefined;
  let generationError: unknown;
  try {
    const descriptor = HumanoidMotionGeneratorDescriptorSchema.parse(generator.descriptor);
    const controlStepSeconds = simulation.controllerDescriptor().controlStepSeconds;
    artifact = assertMotionArtifactContract(
      await generator.generate({
        simulation,
        plan,
        baseline,
        controlStepSeconds
      }),
      descriptor,
      plan,
      controlStepSeconds
    );
  } catch (error) {
    generationError = error;
  } finally {
    simulation.restoreState(generationState);
  }
  if (generationError !== undefined || artifact === undefined) {
    const snapshot = simulation.snapshot();
    return {
      artifact: null,
      rollout: null,
      optionCertificate: null,
      validation: validationResult(
        [{
          code: "invalid_reference",
          atSeconds: generationError instanceof HumanoidMotionGenerationError
            ? generationError.atSeconds
            : 0,
          message: generationError instanceof Error
            ? generationError.message
            : String(generationError ?? "Motion generator returned no artifact")
        }],
        snapshot,
        snapshot,
        0,
        snapshot.rootPosition.y,
        snapshot.balance.upright,
        snapshot.balance.supportMargin,
        new Set(),
        new Map(),
        [],
        new Set()
      )
    };
  }
  const validated = await validateHumanoidMotionArtifact(
    simulation,
    plan,
    artifact,
    options
  );
  return { artifact, ...validated };
}

function assertMotionArtifactContract(
  rawArtifact: HumanoidMotionArtifact,
  descriptor: HumanoidMotionGeneratorDescriptor,
  rawPlan: HumanoidMotionPlan,
  controlStepSeconds: number
): HumanoidMotionArtifact {
  const artifact = HumanoidMotionArtifactSchema.parse(rawArtifact);
  const plan = HumanoidMotionPlanSchema.parse(rawPlan);
  const expectedFrames = Math.ceil(plan.duration_seconds / controlStepSeconds);
  if (artifact.generator !== descriptor.implementation) {
    throw new Error("Humanoid motion artifact generator identity mismatch");
  }
  if (Math.abs(artifact.controlStepSeconds - controlStepSeconds) > 1e-9) {
    throw new Error("Humanoid motion artifact control step mismatch");
  }
  if (Math.abs(artifact.durationSeconds - plan.duration_seconds) > 1e-9
    || artifact.frames.length !== expectedFrames) {
    throw new Error("Humanoid motion artifact duration or frame count mismatch");
  }
  for (let index = 0; index < artifact.frames.length; index += 1) {
    const expectedTime = Math.min((index + 1) * controlStepSeconds, plan.duration_seconds);
    if (Math.abs(artifact.frames[index]!.atSeconds - expectedTime) > 1e-9) {
      throw new Error("Humanoid motion artifact frame cadence mismatch");
    }
    assertTaskSpaceServoArtifactFrame(
      artifact.frames[index]!,
      plan,
      descriptor.implementation === TASK_SPACE_MOTION_GENERATOR_DESCRIPTOR.implementation
    );
  }
  const coordinatedHands = plan.keyframes.some(
    (keyframe) => keyframe.hand_coordination != null
  );
  if (coordinatedHands !== (artifact.version === 2)) {
    throw new Error(
      coordinatedHands
        ? "Humanoid hand coordination requires a version 2 motion artifact"
        : "A motion generator cannot add hand commands without model coordination"
    );
  }
  if (artifact.version === 2) {
    for (const frame of artifact.frames) {
      const expected = plannedHandCommandAtTime(plan.keyframes, frame.atSeconds);
      if (!expected || JSON.stringify(expected.coordination)
        !== JSON.stringify(frame.handCommand.coordination)) {
        throw new Error("Humanoid motion artifact hand command does not match its model plan");
      }
    }
  }
  return artifact;
}

function assertTaskSpaceServoArtifactFrame(
  frame: HumanoidMotionArtifact["frames"][number],
  plan: HumanoidMotionPlan,
  required: boolean
): void {
  const endpointKeyframe = plan.keyframes.find(
    (keyframe) => frame.atSeconds <= keyframe.at_seconds + 1e-9
  );
  if (!endpointKeyframe) {
    throw new Error("Task-space servo frame exceeds its model plan");
  }
  const expected = taskSpaceTargets(endpointKeyframe);
  const actual = frame.taskSpaceTargets ?? [];
  if (required && (expected.length === 0) !== (actual.length === 0)) {
    throw new Error(
      expected.length === 0
        ? "Task-space generator added an unauthorized servo target"
        : "Task-space generator omitted an executable servo target"
    );
  }
  if (actual.length === 0) return;
  if (expected.length === 0 || actual.length !== expected.length) {
    throw new Error("Motion artifact task-space targets do not match its model plan");
  }
  const expectedByBody = new Map(expected.map((target) => [target.body, target]));
  for (const target of actual) {
    const endpoint = expectedByBody.get(target.body);
    if (!endpoint
      || target.frame !== endpoint.frame
      || target.tolerance !== endpoint.tolerance
      || target.orientationTolerance !== endpoint.orientationTolerance
      || (target.orientation === undefined) !== (endpoint.orientation === undefined)) {
      throw new Error("Motion artifact task-space authority exceeds its model plan");
    }
    if (Math.abs(frame.atSeconds - endpointKeyframe.at_seconds) > 1e-9) continue;
    const positionError = Math.hypot(
      target.position.x - endpoint.position.x,
      target.position.y - endpoint.position.y,
      target.position.z - endpoint.position.z
    );
    const orientationError = endpoint.orientation && target.orientation
      ? quaternionAngularDistance(target.orientation, endpoint.orientation)
      : 0;
    if (positionError > 1e-9 || orientationError > 1e-9) {
      throw new Error("Motion artifact task-space endpoint differs from its model plan");
    }
  }
}

async function validateHumanoidMotionArtifact(
  simulation: HumanoidSimulation,
  plan: HumanoidMotionPlan,
  artifact: HumanoidMotionArtifact,
  options: HumanoidMotionValidationOptions
): Promise<{
  validation: HumanoidMotionValidation;
  rollout: HumanoidMotionRollout | null;
  optionCertificate: HumanoidMotionOptionCertificate | null;
}> {
  const saved = simulation.captureState();
  const start = simulation.snapshot();
  const modelConstraints = plan.contact_constraints ?? [];
  const carriedObjectBindings = options.carriedObjectBindings
    ? HumanoidCarriedObjectBindingSetSchema.parse(options.carriedObjectBindings)
    : null;
  const carriedObjectTaskSpaceTargets = HumanoidCarryTaskSpaceTargetsSchema.parse(
    options.carriedObjectTaskSpaceTargets ?? []
  );
  const constraints = mergeHumanoidContactConstraints(
    modelConstraints,
    carriedObjectBindings
      ? humanoidCarriedObjectContactConstraints(carriedObjectBindings)
      : []
  );
  const carriedObjectIds = new Set(
    carriedObjectBindings?.bindings.map((binding) => binding.object_id) ?? []
  );
  const allowed = new Set(constraints.map(contactKey));
  const required = modelConstraints.filter((constraint) => constraint.required);
  const knownObjects = new Set(Object.keys(start.objects));
  const knownSolids = new Set(simulation.solidIds());
  const physicalTargets = scheduledTaskSpaceTargets(plan);
  let nextPhysicalTargetIndex = 0;
  const failures: HumanoidMotionValidation["failures"] = [];
  const contacted = new Set<HumanoidBodyName>();
  const environmentContacts = new Map<string, HumanoidEnvironmentContact>();
  const satisfiedRequired = new Set<string>();
  let minimumRootHeight = start.rootPosition.y;
  let minimumUpright = start.balance.upright;
  let minimumSupportMargin = start.balance.supportMargin;
  let finalSnapshot = start;
  let simulatedSteps = 0;
  let optionDetection: HumanoidMotionOptionDetection | null = null;
  let optionMonitor: HumanoidMotionOptionMonitorState | null = options.motionOption
    ? createHumanoidMotionOptionMonitorState(options.motionOption.contract)
    : null;
  let optionObservationStatus: "satisfied" | "unsatisfied" | "uncertain" | null = null;
  let predictedTerminationFrame: number | null = null;
  let predictedAtSeconds: number | null = null;
  let predictedEvidence: JsonValue | null = null;
  const rolloutFrames: HumanoidMotionRolloutFrame[] = [];
  let physicalSafety = createHumanoidPhysicalSafetyAccumulator();
  const graspRegistry = options.graspRegistry?.fork();
  const graspTargets = graspRegistry
    ? mergeG1ContactAwareGraspTargets(
        carriedObjectBindings
          ? contactAwareG1GraspTargetsForBindings({
              bindings: carriedObjectBindings.bindings,
              graspRegistry
            })
          : [],
        options.motionOption
          ? contactAwareG1GraspTargetsForOption({
              option: options.motionOption.contract,
              graspContract: graspRegistry.contract
            })
          : []
      )
    : [];
  const releaseAuthority = carriedObjectBindings && options.motionOption
    ? authorizeHumanoidCarriedObjectRelease({
        contract: options.motionOption.contract,
        bindingSet: carriedObjectBindings
      })
    : null;
  const releaseTrackedObjectIds = new Set(
    releaseAuthority?.bindings.map((binding) => binding.objectId) ?? []
  );
  let releaseSeparated = false;
  const rolloutStartFrame = options.worldFrame;
  if ((graspRegistry === undefined) !== (rolloutStartFrame === undefined)) {
    throw new Error(
      "Humanoid motion preview requires both grasp registry and world frame"
    );
  }
  if (graspRegistry && graspRegistry.lastFrame !== rolloutStartFrame) {
    throw new Error(
      "Humanoid motion preview grasp registry is not aligned with its world frame"
    );
  }
  if (graspRegistry && rolloutStartFrame !== undefined) {
    graspRegistry.observe(rolloutStartFrame, start);
  }
  if (carriedObjectBindings && (
    graspRegistry === undefined
      || rolloutStartFrame === undefined
      || options.worldRevision === undefined
  )) {
    throw new Error(
      "Humanoid carried-object preview requires grasp, frame and revision authority"
    );
  }
  if (carriedObjectBindings
    && !humanoidCarryTaskSpaceTargetsMatchBindings(
      carriedObjectTaskSpaceTargets,
      carriedObjectBindings
    )) {
    throw new Error(
      "Humanoid carried-object preview targets do not match their authority bindings"
    );
  }
  if (!carriedObjectBindings && carriedObjectTaskSpaceTargets.length > 0) {
    throw new Error(
      "Humanoid carried-object preview targets require authority bindings"
    );
  }
  try {
    const objectConstraints = constraints.filter((constraint): constraint is (
      HumanoidContactConstraint & { object_id: string }
    ) => "object_id" in constraint);
    const solidConstraints = constraints.filter((constraint): constraint is (
      HumanoidContactConstraint & { solid_id: string }
    ) => "solid_id" in constraint);
    const unknownObjects = objectConstraints.filter((constraint) => (
      !knownObjects.has(constraint.object_id)
    ));
    if (unknownObjects.length > 0) {
      failures.push({
        code: "unknown_contact_object",
        atSeconds: 0,
        constraints: unknownObjects
      });
      return {
        validation: validationResult(
          failures,
          start,
          start,
          0,
          minimumRootHeight,
          minimumUpright,
          minimumSupportMargin,
          contacted,
          environmentContacts,
          required,
          satisfiedRequired
        ),
        rollout: null,
        optionCertificate: null
      };
    }
    const unseenObjects = options.contactObjectIds
      ? objectConstraints.filter((constraint) => (
          !options.contactObjectIds?.has(constraint.object_id)
            && !carriedObjectIds.has(constraint.object_id)
        ))
      : [];
    if (unseenObjects.length > 0) {
      failures.push({
        code: "contact_object_not_currently_visible",
        atSeconds: 0,
        constraints: unseenObjects
      });
      return {
        validation: validationResult(
          failures,
          start,
          start,
          0,
          minimumRootHeight,
          minimumUpright,
          minimumSupportMargin,
          contacted,
          environmentContacts,
          required,
          satisfiedRequired
        ),
        rollout: null,
        optionCertificate: null
      };
    }
    const unknownSolids = solidConstraints.filter((constraint) => (
      !knownSolids.has(constraint.solid_id)
    ));
    if (unknownSolids.length > 0) {
      failures.push({
        code: "unknown_contact_solid",
        atSeconds: 0,
        constraints: unknownSolids
      });
      return {
        validation: validationResult(
          failures,
          start,
          start,
          0,
          minimumRootHeight,
          minimumUpright,
          minimumSupportMargin,
          contacted,
          environmentContacts,
          required,
          satisfiedRequired
        ),
        rollout: null,
        optionCertificate: null
      };
    }
    const unseenSolids = options.contactSolidIds
      ? solidConstraints.filter((constraint) => (
          !options.contactSolidIds?.has(constraint.solid_id)
        ))
      : [];
    if (unseenSolids.length > 0) {
      failures.push({
        code: "contact_solid_not_currently_visible",
        atSeconds: 0,
        constraints: unseenSolids
      });
      return {
        validation: validationResult(
          failures,
          start,
          start,
          0,
          minimumRootHeight,
          minimumUpright,
          minimumSupportMargin,
          contacted,
          environmentContacts,
          required,
          satisfiedRequired
        ),
        rollout: null,
        optionCertificate: null
      };
    }
    while (physicalTargets[nextPhysicalTargetIndex]?.atSeconds === 0) {
      const failure = physicalTaskSpaceTargetFailure(
        physicalTargets[nextPhysicalTargetIndex]!,
        start,
        0
      );
      if (failure) failures.push(failure);
      nextPhysicalTargetIndex += 1;
    }
    if (releaseAuthority
      && !artifactCommandsActiveRelease(start, artifact, releaseAuthority)) {
      failures.push({
        code: "motion_constraint_violated",
        atSeconds: 0,
        message: "Object release requires a model-authored opening hand command"
      });
    }
    if (failures.length > 0) {
      return {
        validation: validationResult(
          failures,
          start,
          start,
          0,
          minimumRootHeight,
          minimumUpright,
          minimumSupportMargin,
          contacted,
          environmentContacts,
          required,
          satisfiedRequired
        ),
        rollout: null,
        optionCertificate: null
      };
    }
    if (options.motionOption) {
      optionDetection = detectHumanoidMotionOptionFromSimulation({
        simulation,
        snapshot: start,
        option: options.motionOption,
        boundObjectIds: options.contactObjectIds,
        graspRegistry,
        worldFrame: rolloutStartFrame,
        trackedObjectIds: releaseTrackedObjectIds
      });
      const initialEvidenceUncertain = options.motionOption.contract.phases?.precondition
        ? optionDetection.phases.precondition?.status === "uncertain"
        : optionDetection.hasUncertain;
      if (initialEvidenceUncertain) {
        failures.push({
          code: "motion_goal_uncertain",
          atSeconds: 0,
          message: "Motion option references state that is not currently observable"
        });
      } else if (optionDetection.allSatisfied && required.length === 0) {
        failures.push({
          code: "motion_goal_already_satisfied",
          atSeconds: 0,
          message: "Motion option predicates are already satisfied before execution"
        });
      }
      if (failures.length > 0) {
        return {
          validation: validationResult(
            failures,
            start,
            start,
            0,
            minimumRootHeight,
            minimumUpright,
            minimumSupportMargin,
            contacted,
            environmentContacts,
            required,
            satisfiedRequired
          ),
          rollout: null,
          optionCertificate: null
        };
      }
    }
    for (const frame of artifact.frames) {
      if (options.motionOption && optionMonitor?.phase === "awaiting_precondition") {
        const update = advanceHumanoidMotionOptionMonitor(
          options.motionOption.contract,
          optionMonitor,
          humanoidMotionOptionDetectorInputFromSimulation({
            simulation,
            snapshot: finalSnapshot,
            option: options.motionOption,
            boundObjectIds: options.contactObjectIds,
            graspRegistry,
            worldFrame: rolloutStartFrame === undefined
              ? undefined
              : rolloutStartFrame + simulatedSteps,
            trackedObjectIds: releaseTrackedObjectIds
          })
        );
        optionMonitor = update.state;
        optionDetection = update.detection;
        optionObservationStatus = update.observationStatus;
        if (update.observationStatus !== "satisfied") {
          const atSeconds = simulatedSteps === 0
            ? 0
            : artifact.frames[simulatedSteps - 1]!.atSeconds;
          failures.push({
            code: update.observationStatus === "uncertain"
              ? "motion_goal_uncertain"
              : "motion_constraint_violated",
            atSeconds,
            message: update.observationStatus === "uncertain"
              ? "Motion option precondition is not observable before execution"
              : "Motion option precondition is not satisfied before execution"
          });
          break;
        }
      }
      try {
        finalSnapshot = (
          await applyHumanoidMotionArtifactFrame(simulation, frame, {
            graspTargets,
            carryTaskSpaceTargets: carriedObjectTaskSpaceTargets
          })
        ).snapshot;
      } catch (error) {
        failures.push({
          code: "invalid_reference",
          atSeconds: frame.atSeconds,
          message: error instanceof Error ? error.message : String(error)
        });
        break;
      }
      simulatedSteps += 1;
      if (graspRegistry && rolloutStartFrame !== undefined) {
        graspRegistry.observe(rolloutStartFrame + simulatedSteps, finalSnapshot);
      }
      if (carriedObjectBindings && graspRegistry
        && rolloutStartFrame !== undefined
        && options.worldRevision !== undefined) {
        const continuation = humanoidCarriedObjectContinuationEvidence({
          state: carriedObjectBindings,
          registry: graspRegistry,
          currentFrame: rolloutStartFrame + simulatedSteps,
          currentWorldRevision: options.worldRevision + simulatedSteps
        });
        const unauthorized = humanoidCarriedObjectUnauthorizedContacts(
          carriedObjectBindings,
          finalSnapshot.contacts
        );
        if (releaseAuthority) {
          releaseSeparated ||= releaseAuthority.bindings.every((binding) => (
            assessHumanoidObjectReleased({
              objectId: binding.objectId,
              hand: binding.hand,
              objectObservable: knownObjects.has(binding.objectId),
              contacts: finalSnapshot.contacts
            }).status === "satisfied"
          ));
        }
        if ((!releaseAuthority && !continuation.continued)
          || unauthorized.length > 0 && !releaseSeparated) {
          const failed = continuation.bindings.find((binding) => !binding.continued);
          const collision = unauthorized[0];
          failures.push({
            code: "motion_constraint_violated",
            atSeconds: frame.atSeconds,
            message: failed
              ? `Carried grasp continuation failed for ${failed.object_id}/${failed.hand}: ${
                  failed.failure ?? "unknown"
                }; ${failed.detail ?? "no detail"}`
              : `Carried object ${collision!.object_id} contacted an unauthorized ${
                  collision!.counterpart_kind
                } counterpart`
          });
          break;
        }
      }
      physicalSafety = accumulateHumanoidPhysicalSafetyFrame(
        physicalSafety,
        simulatedSteps,
        finalSnapshot
      );
      if (options.motionOption) {
        rolloutFrames.push(captureHumanoidMotionRolloutFrame(
          frame.atSeconds,
          finalSnapshot,
          artifact.version
        ));
      }
      minimumRootHeight = Math.min(minimumRootHeight, finalSnapshot.rootPosition.y);
      minimumUpright = Math.min(minimumUpright, finalSnapshot.balance.upright);
      if (finalSnapshot.balance.supportMargin !== null) {
        minimumSupportMargin = minimumSupportMargin === null
          ? finalSnapshot.balance.supportMargin
          : Math.min(minimumSupportMargin, finalSnapshot.balance.supportMargin);
      }
      const observedContacts = humanoidEnvironmentContacts(finalSnapshot);
      for (const contact of observedContacts) {
        if ("body" in contact) contacted.add(contact.body);
        const key = humanoidEnvironmentContactKey(contact);
        const previous = environmentContacts.get(key);
        if (!previous || contact.normalForce > previous.normalForce) {
          environmentContacts.set(key, contact);
        }
        if ((contact.objectId !== null || contact.solidId !== null)
          && allowed.has(key)) {
          satisfiedRequired.add(key);
        }
      }
      const blockedContacts = observedContacts.filter((contact) => (
        contact.objectId === null && contact.solidId === null
        || !allowed.has(humanoidEnvironmentContactKey(contact))
      ));
      if (blockedContacts.length > 0) {
        const bodies = distinctContactBodies(blockedContacts);
        const handSurfaces = distinctContactHandSurfaces(blockedContacts);
        failures.push({
          code: "environment_contact",
          atSeconds: frame.atSeconds,
          ...(bodies.length > 0 ? { bodies } : {}),
          ...(handSurfaces.length > 0 ? { handSurfaces } : {}),
          contacts: blockedContacts
        });
        break;
      }
      if (finalSnapshot.fallen) {
        failures.push({ code: "fallen", atSeconds: frame.atSeconds });
        break;
      }
      while ((physicalTargets[nextPhysicalTargetIndex]?.atSeconds ?? Infinity)
        <= frame.atSeconds + 1e-9) {
        const failure = physicalTaskSpaceTargetFailure(
          physicalTargets[nextPhysicalTargetIndex]!,
          finalSnapshot,
          frame.atSeconds
        );
        if (failure) failures.push(failure);
        nextPhysicalTargetIndex += 1;
      }
      if (failures.length > 0) break;
      if (options.motionOption && optionMonitor?.phase !== "awaiting_precondition") {
        if (!optionMonitor) {
          throw new Error("Humanoid motion option monitor is missing");
        }
        const update = advanceHumanoidMotionOptionMonitor(
          options.motionOption.contract,
          optionMonitor,
          humanoidMotionOptionDetectorInputFromSimulation({
            simulation,
            snapshot: finalSnapshot,
            option: options.motionOption,
            boundObjectIds: options.contactObjectIds,
            graspRegistry,
            worldFrame: rolloutStartFrame === undefined
              ? undefined
              : rolloutStartFrame + simulatedSteps,
            trackedObjectIds: releaseTrackedObjectIds
          })
        );
        optionMonitor = update.state;
        optionDetection = update.detection;
        optionObservationStatus = update.observationStatus;
        if (optionMonitor.phase === "violated") {
          failures.push({
            code: "motion_constraint_violated",
            atSeconds: frame.atSeconds,
            message: "Motion option violated its during constraint"
          });
          break;
        }
        if (optionMonitor.phase === "indeterminate") {
          failures.push({
            code: "motion_goal_uncertain",
            atSeconds: frame.atSeconds,
            message: "Motion option lost observable evidence for its during constraint"
          });
          break;
        }
        if (optionMonitor.phase === "succeeded"
          && missingRequiredHumanoidContacts(
            required,
            satisfiedRequired
          ).length === 0
          && finalSnapshot.balance.support !== "none"
          && predictedTerminationFrame === null) {
          predictedTerminationFrame = simulatedSteps;
          predictedAtSeconds = frame.atSeconds;
          predictedEvidence = asJson({
            predicates: optionDetection.evidence,
            phases: optionDetection.phases,
            monitor: optionMonitor
          });
          break;
        }
      }
    }
    const missingRequired = required.filter((constraint) => (
      !satisfiedRequired.has(contactKey(constraint))
    ));
    if (failures.length === 0 && missingRequired.length > 0) {
      failures.push({
        code: "required_contact_missing",
        atSeconds: plan.duration_seconds,
        constraints: missingRequired
      });
    }
    if (failures.length === 0
      && (options.requireFinalSupport ?? true)
      && finalSnapshot.balance.support === "none") {
      failures.push({
        code: "unsupported_finish",
        atSeconds: plan.duration_seconds
      });
    }
    if (failures.length === 0
      && options.motionOption
      && predictedTerminationFrame === null) {
      failures.push({
        code: optionObservationStatus === "uncertain" || optionDetection?.hasUncertain
          ? "motion_goal_uncertain"
          : "motion_goal_unmet",
        atSeconds: plan.duration_seconds,
        message: optionObservationStatus === "uncertain" || optionDetection?.hasUncertain
          ? "Motion option ended without observable success evidence"
          : "Motion option exhausted its verified horizon before physical success"
      });
    }
    const rollout = failures.length === 0
      && options.motionOption
      && predictedTerminationFrame !== null
      && rolloutFrames.length === predictedTerminationFrame
      ? createHumanoidMotionRollout(rolloutFrames, artifact.version)
      : null;
    const optionCertificate = failures.length === 0
      && options.motionOption
      && rollout !== null
      && predictedTerminationFrame !== null
      && predictedAtSeconds !== null
      && predictedEvidence !== null
      ? HumanoidMotionOptionCertificateSchema.parse({
          artifact_sha256: humanoidMotionArtifactSha256(artifact),
          contract_sha256: humanoidMotionOptionContractSha256(
            options.motionOption.contract
          ),
          rollout_sha256: humanoidMotionRolloutSha256(rollout),
          rollout_frame_count: rollout.frames.length,
          drift_consecutive_steps: rollout.limits.consecutive_steps,
          validated_frame_limit: predictedTerminationFrame,
          predicted_termination_frame: predictedTerminationFrame,
          predicted_at_seconds: predictedAtSeconds,
          stable_steps: options.motionOption.contract.stable_steps,
          evidence: predictedEvidence,
          physical_safety: completeHumanoidPhysicalSafetyEvidence(physicalSafety)
        })
      : null;
    return {
      rollout,
      validation: validationResult(
        failures,
        start,
        finalSnapshot,
        simulatedSteps,
        minimumRootHeight,
        minimumUpright,
        minimumSupportMargin,
        contacted,
        environmentContacts,
        required,
        satisfiedRequired
      ),
      optionCertificate
    };
  } finally {
    simulation.restoreState(saved);
  }
}

function artifactCommandsActiveRelease(
  start: HumanoidSimulationSnapshot,
  artifact: HumanoidMotionArtifact,
  authority: HumanoidCarriedObjectReleaseAuthority
): boolean {
  return authority.bindings.every((binding) => {
    const prefix = `${binding.hand}_hand_`;
    const baselineClosure = Object.entries(start.hands.joints)
      .filter(([joint]) => joint.startsWith(prefix))
      .reduce((total, [, state]) => total + Math.abs(state.target), 0);
    const commandedClosures = artifact.frames.flatMap((frame) => (
      "handCommand" in frame
        ? [Object.entries(frame.handCommand.jointTargets)
            .filter(([joint]) => joint.startsWith(prefix))
            .reduce((total, [, target]) => total + Math.abs(target), 0)]
        : []
    ));
    const minimumCommandedClosure = Math.min(...commandedClosures);
    return minimumCommandedClosure < baselineClosure - 1e-9;
  });
}

function asJson(value: unknown): JsonValue {
  return JSON.parse(JSON.stringify(value)) as JsonValue;
}

interface ScheduledTaskSpaceTarget {
  atSeconds: number;
  target: HumanoidEndEffectorTarget;
}

function scheduledTaskSpaceTargets(
  plan: HumanoidMotionPlan
): ScheduledTaskSpaceTarget[] {
  return plan.keyframes.flatMap((keyframe) => (
    taskSpaceTargets(keyframe).map((target) => ({
      atSeconds: keyframe.at_seconds,
      target
    }))
  ));
}

function physicalTaskSpaceTargetFailure(
  scheduled: ScheduledTaskSpaceTarget,
  snapshot: HumanoidSimulationSnapshot,
  observedAtSeconds: number
): HumanoidMotionValidation["failures"][number] | null {
  const target = scheduled.target;
  const achievedLink = snapshot.links[target.body];
  const achievedWorld = achievedLink.position;
  const achieved = target.frame === "world"
    ? { ...achievedWorld }
    : rotateVector(
        inverseQuaternion(snapshot.links.pelvis.rotation),
        subtract(achievedWorld, snapshot.links.pelvis.position)
      );
  const errorMeters = Math.hypot(
    achieved.x - target.position.x,
    achieved.y - target.position.y,
    achieved.z - target.position.z
  );
  const orientationTarget = target.orientation
    ? normalizeQuaternion(target.orientation)
    : undefined;
  const orientationAchieved = orientationTarget
    ? target.frame === "world"
      ? normalizeQuaternion(achievedLink.rotation)
      : normalizeQuaternion(multiplyQuaternion(
          inverseQuaternion(snapshot.links.pelvis.rotation),
          achievedLink.rotation
        ))
    : undefined;
  const orientationErrorRadians = orientationTarget && orientationAchieved
    ? quaternionAngularDistance(orientationTarget, orientationAchieved)
    : undefined;
  const positionSatisfied = errorMeters <= target.tolerance + 1e-9;
  const orientationSatisfied = orientationErrorRadians === undefined
    || target.orientationTolerance !== undefined
      && orientationErrorRadians <= target.orientationTolerance + 1e-9;
  if (positionSatisfied && orientationSatisfied) return null;
  const evidence = {
    body: target.body,
    frame: target.frame,
    target: { ...target.position },
    achieved,
    errorMeters,
    toleranceMeters: target.tolerance,
    ...(orientationTarget && orientationAchieved
      && orientationErrorRadians !== undefined
      && target.orientationTolerance !== undefined
      ? {
          orientationTarget,
          orientationAchieved,
          orientationErrorRadians,
          orientationToleranceRadians: target.orientationTolerance
        }
      : {}),
    requestedAtSeconds: scheduled.atSeconds,
    observedAtSeconds
  };
  return {
    code: "task_space_target_unmet",
    atSeconds: observedAtSeconds,
    taskSpaceTarget: evidence,
    message: `Physical task-space target missed: ${target.body} `
      + `error=${errorMeters.toFixed(3)}m tolerance=${target.tolerance.toFixed(3)}m`
      + (orientationErrorRadians === undefined || target.orientationTolerance === undefined
        ? ""
        : ` orientation=${orientationErrorRadians.toFixed(3)}rad`
          + ` tolerance=${target.orientationTolerance.toFixed(3)}rad`)
  };
}

function mergeHumanoidContactConstraints(
  primary: readonly HumanoidContactConstraint[],
  additional: readonly HumanoidContactConstraint[]
): HumanoidContactConstraint[] {
  const merged = new Map<string, HumanoidContactConstraint>();
  for (const constraint of [...additional, ...primary]) {
    merged.set(contactKey(constraint), { ...constraint });
  }
  return [...merged.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([, constraint]) => constraint);
}

function validationResult(
  failures: HumanoidMotionValidation["failures"],
  start: HumanoidSimulationSnapshot,
  finalSnapshot: HumanoidSimulationSnapshot,
  simulatedSteps: number,
  minimumRootHeight: number,
  minimumUpright: number,
  minimumSupportMargin: number | null,
  contacted: ReadonlySet<HumanoidBodyName>,
  environmentContacts: ReadonlyMap<string, HumanoidEnvironmentContact>,
  required: readonly HumanoidContactConstraint[],
  satisfiedRequired: ReadonlySet<string>
): HumanoidMotionValidation {
  return {
    feasible: failures.length === 0,
    failures,
    evidence: {
      simulatedSteps,
      minimumRootHeight,
      minimumUpright,
      minimumSupportMargin,
      travelledDistance: Math.hypot(
        finalSnapshot.rootPosition.x - start.rootPosition.x,
        finalSnapshot.rootPosition.z - start.rootPosition.z
      ),
      environmentContactBodies: [...contacted],
      environmentContactHandSurfaces: distinctContactHandSurfaces(
        [...environmentContacts.values()]
      ),
      environmentContacts: [...environmentContacts.values()],
      satisfiedRequiredContacts: required.filter((constraint) => (
        satisfiedRequired.has(contactKey(constraint))
      ))
    },
    finalSnapshot
  };
}
