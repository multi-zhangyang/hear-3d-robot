import { z } from "zod";
import { JsonValueSchema, Vec3Schema } from "../../domain/schema.js";
import {
  HumanoidMotionArtifactSchema,
  humanoidMotionArtifactSha256,
  HumanoidReferenceStateSchema
} from "./motion-artifact.js";
import {
  humanoidContactConstraintKey,
  HumanoidEnvironmentContactSchema,
  HumanoidMotionOptionCertificateSchema,
  HumanoidMotionPlanSchema
} from "./motion-plan.js";
import {
  HumanoidMotionRolloutSchema,
  humanoidMotionRolloutSha256
} from "./motion-rollout.js";
import {
  HumanoidMotionOptionContractSchema,
  HumanoidMotionOptionMonitorStateSchema,
  humanoidMotionOptionContractSha256
} from "./motion-option.js";
import {
  HumanoidMotionGeneratorDescriptorSchema,
  TASK_SPACE_MOTION_GENERATOR_DESCRIPTOR
} from "./motion-generator-contract.js";
import {
  humanoidMotionIntentSha256,
  humanoidNavigationIntentSha256
} from "./plan-lifecycle.js";
import { HUMANOID_BODY_NAMES } from "./model.js";
import { G1_HAND_CONTACT_SURFACE_NAMES } from "./morphology.js";
import { HumanoidObjectMemoryCheckpointSchema } from "./object-memory.js";
import {
  HumanoidPhysicalSafetyAccumulatorSchema,
  HumanoidPhysicalSafetyEvidenceSchema
} from "./physical-safety.js";
import { HumanoidPlanTerminalSchema } from "./execution-terminal.js";
import { HumanoidNavigationExecutionProgressSchema } from "./navigation-execution.js";
import {
  HumanoidCarryTaskSpaceTargetsSchema,
  humanoidCarryTaskSpaceTargetsMatchBindings
} from "./carry-task-space-servo.js";
import { HumanoidGraspRegistryCheckpointSchema } from "./grasp-registry.js";
import {
  HumanoidCarriedObjectBindingSetSchema,
  HumanoidCarriedObjectContinuationEvidenceSchema,
  HumanoidCarriedObjectUnauthorizedContactSchema,
  humanoidCarriedObjectBindingSha256,
  humanoidCarriedObjectBindingSetSha256,
  humanoidGraspRegistryCheckpointSha256
} from "./carried-object-binding.js";
import { humanoidMotionContactEvidenceSha256 } from "./motion-contact-evidence.js";
import { HumanoidCarriedObjectLifecycleCheckpointSchema } from "./carried-object-lifecycle.js";
import { HumanoidStationKeepingAnchorSchema } from "./station-keeping.js";
import { HumanoidNavigationArrivalHeadingSchema } from "./navigation-arrival.js";

const FiniteArraySchema = z.array(z.number().finite());

function sameSortedValues(
  actual: readonly string[] | undefined,
  expected: readonly string[]
): boolean {
  const normalized = [...new Set(actual ?? [])].sort();
  return normalized.length === expected.length
    && normalized.every((value, index) => value === expected[index]);
}

const NavigationPlanSchema = z.object({
  waypoints: z.array(Vec3Schema),
  distance: z.number().finite().nonnegative(),
  resolvedTarget: Vec3Schema,
  projectionDistance: z.number().finite().nonnegative()
}).strict();

const NavigationStateSchema = z.object({
  planId: z.string().min(1).nullable(),
  status: z.enum(["idle", "planned", "executing", "completed", "blocked"]),
  target: Vec3Schema.nullable(),
  waypoints: z.array(Vec3Schema),
  waypointIndex: z.number().int().nonnegative().nullable()
}).strict();

const HumanoidMotionDriftEvidenceSchema = z.object({
  drifted: z.boolean(),
  rootPositionErrorMeters: z.number().finite().nonnegative(),
  rootOrientationErrorRadians: z.number().finite().nonnegative(),
  jointRmsErrorRadians: z.number().finite().nonnegative(),
  handJointRmsErrorRadians: z.number().finite().nonnegative().optional(),
  maximumEndEffectorErrorMeters: z.number().finite().nonnegative(),
  maximumEndEffectorOrientationErrorRadians: z.number().finite().nonnegative().optional(),
  maximumHandTargetErrorRadians: z.number().finite().nonnegative().optional()
}).strict();

const HumanoidMotionExecutionFailureSchema = z.object({
  code: z.enum([
    "fallen",
    "environment_contact",
    "execution_drift",
    "motion_constraint_violated"
  ]),
  atSeconds: z.number().finite().nonnegative(),
  bodies: z.array(z.enum(HUMANOID_BODY_NAMES)).optional(),
  handSurfaces: z.array(z.enum(G1_HAND_CONTACT_SURFACE_NAMES)).optional(),
  contacts: z.array(HumanoidEnvironmentContactSchema).optional(),
  drift: HumanoidMotionDriftEvidenceSchema.optional()
}).strict().superRefine((failure, context) => {
  if (failure.code !== "environment_contact") return;
  if (!failure.contacts || failure.contacts.length === 0) {
    context.addIssue({
      code: "custom",
      path: ["contacts"],
      message: "An environment-contact failure requires exact physical contacts"
    });
    return;
  }
  const bodyContacts = [...new Set(failure.contacts.flatMap((contact) => (
    "body" in contact ? [contact.body] : []
  )))].sort();
  const handContacts = [...new Set(failure.contacts.flatMap((contact) => (
    "handSurface" in contact ? [contact.handSurface] : []
  )))].sort();
  if (!sameSortedValues(failure.bodies, bodyContacts)) {
    context.addIssue({
      code: "custom",
      path: ["bodies"],
      message: "Environment-contact body evidence must match its exact contacts"
    });
  }
  if (!sameSortedValues(failure.handSurfaces, handContacts)) {
    context.addIssue({
      code: "custom",
      path: ["handSurfaces"],
      message: "Environment-contact hand-surface evidence must match its exact contacts"
    });
  }
});

const HumanoidMotionExecutionProgressSchema = z.object({
  nextFrameIndex: z.number().int().nonnegative(),
  satisfiedContactKeys: z.array(z.string().min(1)),
  satisfiedContactEvidenceSha256: z.string().regex(/^[a-f0-9]{64}$/).optional(),
  driftStreak: z.number().int().nonnegative().default(0),
  lastDrift: HumanoidMotionDriftEvidenceSchema.nullable().default(null),
  failure: HumanoidMotionExecutionFailureSchema.nullable(),
  physicalSafety: HumanoidPhysicalSafetyAccumulatorSchema.optional()
}).strict();

export type HumanoidMotionExecutionProgress = z.infer<
  typeof HumanoidMotionExecutionProgressSchema
>;

const HumanoidMotionOptionExecutionStateSchema = z.object({
  contract: HumanoidMotionOptionContractSchema,
  certificate: HumanoidMotionOptionCertificateSchema,
  monitor: HumanoidMotionOptionMonitorStateSchema,
  status: z.enum(["planned", "executing", "succeeded", "failed", "goal_unmet"]),
  successStreak: z.number().int().nonnegative(),
  actualTerminationFrame: z.number().int().nonnegative().nullable(),
  terminationReason: z.enum([
    "physical_success",
    "fallen",
    "environment_contact",
    "execution_drift",
    "motion_constraint_violated",
    "motion_goal_unmet",
    "motion_goal_uncertain"
  ]).nullable(),
  lastEvidence: JsonValueSchema.nullable()
}).strict();

export type HumanoidMotionOptionExecutionState = z.infer<
  typeof HumanoidMotionOptionExecutionStateSchema
>;

const StoredMotionSchema = z.object({
  plan: HumanoidMotionPlanSchema,
  artifact: HumanoidMotionArtifactSchema,
  rollout: HumanoidMotionRolloutSchema.nullable().default(null),
  retainTerminalJointTracking: z.boolean().default(false),
  createdRevision: z.number().int().nonnegative(),
  validatedRevision: z.number().int().nonnegative().optional(),
  validatedStateSha256: z.string().regex(/^[a-f0-9]{64}$/).optional(),
  expiresRevision: z.number().int().nonnegative().optional(),
  intentSha256: z.string().regex(/^[a-f0-9]{64}$/).optional(),
  revalidationCount: z.number().int().nonnegative().default(0),
  terminal: HumanoidPlanTerminalSchema.nullable().default(null),
  option: HumanoidMotionOptionExecutionStateSchema.nullable().default(null),
  carriedObjectBindings: HumanoidCarriedObjectBindingSetSchema
    .nullable()
    .default(null),
  carriedObjectTaskSpaceTargets: HumanoidCarryTaskSpaceTargetsSchema.default([]),
  carriedObjectContinuation: HumanoidCarriedObjectContinuationEvidenceSchema
    .nullable()
    .default(null),
  carriedObjectUnauthorizedContacts: z.array(
    HumanoidCarriedObjectUnauthorizedContactSchema
  ).default([]),
  progress: HumanoidMotionExecutionProgressSchema.default({
    nextFrameIndex: 0,
    satisfiedContactKeys: [],
    driftStreak: 0,
    lastDrift: null,
    failure: null
  })
}).strict().superRefine((motion, context) => {
  if (motion.intentSha256 !== undefined
    && motion.intentSha256 !== humanoidMotionIntentSha256(motion.plan)) {
    context.addIssue({
      code: "custom",
      path: ["intentSha256"],
      message: "Humanoid motion intent identity does not match its model plan"
    });
  }
  if (motion.validatedRevision !== undefined
    && motion.validatedRevision < motion.createdRevision) {
    context.addIssue({
      code: "custom",
      path: ["validatedRevision"],
      message: "Humanoid motion validation cannot precede intent creation"
    });
  }
  if (motion.expiresRevision !== undefined
    && motion.expiresRevision < motion.createdRevision) {
    context.addIssue({
      code: "custom",
      path: ["expiresRevision"],
      message: "Humanoid motion intent lease cannot expire before creation"
    });
  }
  if (motion.progress.nextFrameIndex > motion.artifact.frames.length) {
    context.addIssue({
      code: "custom",
      path: ["progress", "nextFrameIndex"],
      message: "Humanoid motion progress exceeds its artifact frame count"
    });
  }
  if (motion.rollout && motion.rollout.version !== motion.artifact.version) {
    context.addIssue({
      code: "custom",
      path: ["rollout", "version"],
      message: "Humanoid motion rollout version must match its executable artifact"
    });
  }
  if (motion.terminal && (
    motion.terminal.plan_id !== motion.plan.id
      || motion.terminal.total_frames !== motion.progress.nextFrameIndex
      || !terminalRevisionMatchesCommittedProgress(motion.terminal, (
        (motion.validatedRevision ?? motion.createdRevision)
        + motion.progress.nextFrameIndex
      ))
  )) {
    context.addIssue({
      code: "custom",
      path: ["terminal"],
      message: "Humanoid motion terminal result does not match its committed plan progress"
    });
  }
  const allowedContactKeys = new Set(
    (motion.plan.contact_constraints ?? []).map((constraint) => (
      humanoidContactConstraintKey(constraint)
    ))
  );
  const uniqueContactKeys = new Set(motion.progress.satisfiedContactKeys);
  if (uniqueContactKeys.size !== motion.progress.satisfiedContactKeys.length) {
    context.addIssue({
      code: "custom",
      path: ["progress", "satisfiedContactKeys"],
      message: "Humanoid motion contact evidence cannot contain duplicate keys"
    });
  }
  if (motion.progress.satisfiedContactKeys.some((key) => !allowedContactKeys.has(key))) {
    context.addIssue({
      code: "custom",
      path: ["progress", "satisfiedContactKeys"],
      message: "Humanoid motion contact evidence is outside its current contact contract"
    });
  }
  if (motion.progress.nextFrameIndex === 0
    && motion.progress.satisfiedContactKeys.length > 0) {
    context.addIssue({
      code: "custom",
      path: ["progress", "satisfiedContactKeys"],
      message: "An unstarted humanoid motion cannot contain satisfied contact evidence"
    });
  }
  if (motion.progress.satisfiedContactEvidenceSha256 !== undefined) {
    const expectedContactEvidenceSha256 = humanoidMotionContactEvidenceSha256({
      planId: motion.plan.id,
      intentSha256: motion.intentSha256
        ?? humanoidMotionIntentSha256(motion.plan),
      artifactSha256: humanoidMotionArtifactSha256(motion.artifact),
      nextFrameIndex: motion.progress.nextFrameIndex,
      satisfiedContactKeys: motion.progress.satisfiedContactKeys
    });
    if (motion.progress.satisfiedContactEvidenceSha256
      !== expectedContactEvidenceSha256) {
      context.addIssue({
        code: "custom",
        path: ["progress", "satisfiedContactEvidenceSha256"],
        message: "Humanoid motion contact evidence identity does not match its executed prefix"
      });
    }
  }
  if (motion.progress.physicalSafety
    && motion.progress.physicalSafety.last_frame !== motion.progress.nextFrameIndex) {
    context.addIssue({
      code: "custom",
      path: ["progress", "physicalSafety"],
      message: "Humanoid physical safety evidence does not end at committed motion progress"
    });
  }
  if (motion.carriedObjectBindings
    && motion.carriedObjectTaskSpaceTargets.length > 0
    && !humanoidCarryTaskSpaceTargetsMatchBindings(
      motion.carriedObjectTaskSpaceTargets,
      motion.carriedObjectBindings
    )) {
    context.addIssue({
      code: "custom",
      path: ["carriedObjectTaskSpaceTargets"],
      message: "Motion carry targets do not match their authority bindings"
    });
  }
  if (!motion.carriedObjectBindings
    && motion.carriedObjectTaskSpaceTargets.length > 0) {
    context.addIssue({
      code: "custom",
      path: ["carriedObjectTaskSpaceTargets"],
      message: "Motion carry targets require an authority binding"
    });
  }
  if (motion.carriedObjectContinuation) {
    if (!motion.carriedObjectBindings
      || motion.carriedObjectContinuation.binding_set_sha256
        !== humanoidCarriedObjectBindingSetSha256(motion.carriedObjectBindings)) {
      context.addIssue({
        code: "custom",
        path: ["carriedObjectContinuation"],
        message: "Motion carry continuation does not match its authority binding"
      });
    }
    if (motion.carriedObjectContinuation.observed_world_revision !== (
      (motion.validatedRevision ?? motion.createdRevision)
        + motion.progress.nextFrameIndex
    )) {
      context.addIssue({
        code: "custom",
        path: ["carriedObjectContinuation", "observed_world_revision"],
        message: "Motion carry evidence is not aligned with committed progress"
      });
    }
  }
  if (motion.carriedObjectUnauthorizedContacts.length > 0
    && motion.carriedObjectBindings === null) {
    context.addIssue({
      code: "custom",
      path: ["carriedObjectUnauthorizedContacts"],
      message: "Motion carry collisions require an authority binding"
    });
  }
  if (!motion.option) return;
  const certificate = motion.option.certificate;
  if (!motion.rollout) {
    context.addIssue({
      code: "custom",
      path: ["rollout"],
      message: "A humanoid motion option requires its validated physical rollout"
    });
    return;
  }
  if (certificate.validated_frame_limit > motion.artifact.frames.length
    || certificate.rollout_frame_count !== motion.rollout.frames.length
    || certificate.rollout_frame_count !== certificate.validated_frame_limit
    || certificate.predicted_termination_frame !== certificate.validated_frame_limit
    || certificate.rollout_sha256 !== humanoidMotionRolloutSha256(motion.rollout)
    || certificate.drift_consecutive_steps !== motion.rollout.limits.consecutive_steps
    || certificate.predicted_termination_frame > certificate.validated_frame_limit
    || certificate.predicted_termination_frame < certificate.stable_steps
    || certificate.stable_steps !== motion.option.contract.stable_steps
    || certificate.physical_safety !== undefined
      && (certificate.physical_safety.frame_count !== certificate.validated_frame_limit
        || certificate.physical_safety.first_frame !== 1
        || certificate.physical_safety.last_frame !== certificate.validated_frame_limit)
    || certificate.artifact_sha256 !== humanoidMotionArtifactSha256(motion.artifact)
    || certificate.contract_sha256 !== humanoidMotionOptionContractSha256(
      motion.option.contract
    )
    || Math.abs(
      certificate.predicted_at_seconds
        - motion.artifact.frames[certificate.predicted_termination_frame - 1]!.atSeconds
    ) > 1e-9) {
    context.addIssue({
      code: "custom",
      path: ["option", "certificate"],
      message: "Humanoid motion option certificate does not match its artifact or contract"
    });
  }
  if (motion.option.monitor.contractSha256 !== certificate.contract_sha256
    || motion.option.successStreak !== motion.option.monitor.terminalStableSteps) {
    context.addIssue({
      code: "custom",
      path: ["option", "monitor"],
      message: "Humanoid motion option monitor does not match its contract or streak"
    });
  }
  const monitor = motion.option.monitor;
  const precondition = motion.option.contract.phases?.precondition ?? null;
  const requiredPreconditionSteps = precondition?.stable_steps ?? 1;
  const preconditionStateInvalid = monitor.phase === "awaiting_precondition"
    ? precondition === null
      || monitor.preconditionStableSteps >= requiredPreconditionSteps
      || monitor.terminalStableSteps !== 0
    : precondition === null
      ? monitor.preconditionStableSteps !== 0
      : monitor.preconditionStableSteps !== requiredPreconditionSteps;
  const terminalStateInvalid = monitor.phase === "succeeded"
    ? monitor.terminalStableSteps !== motion.option.contract.stable_steps
    : monitor.phase === "running"
      ? monitor.terminalStableSteps >= motion.option.contract.stable_steps
      : monitor.terminalStableSteps !== 0;
  const constraintStateInvalid = (monitor.phase === "violated"
      || monitor.phase === "indeterminate")
    && !motion.option.contract.phases?.during;
  if (preconditionStateInvalid || terminalStateInvalid || constraintStateInvalid) {
    context.addIssue({
      code: "custom",
      path: ["option", "monitor"],
      message: "Humanoid motion option monitor phase is inconsistent with its contract"
    });
  }
  if (motion.option.actualTerminationFrame !== null
    && (motion.option.actualTerminationFrame > certificate.validated_frame_limit
      || motion.option.actualTerminationFrame > motion.progress.nextFrameIndex)) {
    context.addIssue({
      code: "custom",
      path: ["option", "actualTerminationFrame"],
      message: "Humanoid motion option termination exceeds committed progress"
    });
  }
  if (motion.option.successStreak > motion.option.contract.stable_steps) {
    context.addIssue({
      code: "custom",
      path: ["option", "successStreak"],
      message: "Humanoid motion option success streak exceeds its stability window"
    });
  }
  if (motion.progress.driftStreak > certificate.drift_consecutive_steps) {
    context.addIssue({
      code: "custom",
      path: ["progress", "driftStreak"],
      message: "Humanoid motion drift streak exceeds its certified window"
    });
  }
  if (motion.option.status === "planned"
    && (motion.progress.nextFrameIndex !== 0
      || motion.option.successStreak !== 0
      || motion.option.monitor.preconditionStableSteps !== 0
      || motion.option.monitor.terminalStableSteps !== 0
      || (motion.option.monitor.phase !== "awaiting_precondition"
        && motion.option.monitor.phase !== "running")
      || motion.progress.driftStreak !== 0
      || motion.progress.lastDrift !== null
      || motion.progress.physicalSafety !== undefined
      || motion.option.actualTerminationFrame !== null
      || motion.option.terminationReason !== null
      || motion.option.lastEvidence !== null)) {
    context.addIssue({
      code: "custom",
      path: ["option", "status"],
      message: "A planned humanoid option cannot contain execution evidence"
    });
  }
  if (motion.option.status === "executing"
    && (motion.option.actualTerminationFrame !== null
      || motion.option.terminationReason !== null)) {
    context.addIssue({
      code: "custom",
      path: ["option", "status"],
      message: "An executing humanoid option cannot already have a terminal result"
    });
  }
  if (motion.option.status === "succeeded"
    && (motion.option.actualTerminationFrame === null
      || motion.option.terminationReason !== "physical_success"
      || motion.option.monitor.phase !== "succeeded"
      || motion.option.successStreak !== motion.option.contract.stable_steps
      || motion.option.actualTerminationFrame < motion.option.contract.stable_steps)) {
    context.addIssue({
      code: "custom",
      path: ["option", "status"],
      message: "A succeeded humanoid option requires its full physical stability window"
    });
  }
  if ((motion.option.status === "succeeded"
      || motion.option.status === "failed"
      || motion.option.status === "goal_unmet")
    && motion.option.actualTerminationFrame !== motion.progress.nextFrameIndex) {
    context.addIssue({
      code: "custom",
      path: ["option", "actualTerminationFrame"],
      message: "A terminal humanoid option must end at its committed progress frame"
    });
  }
  if (motion.option.status === "failed"
    && (motion.option.actualTerminationFrame === null
      || (motion.option.terminationReason !== "fallen"
        && motion.option.terminationReason !== "environment_contact"
        && motion.option.terminationReason !== "execution_drift"
        && motion.option.terminationReason !== "motion_constraint_violated"))) {
    context.addIssue({
      code: "custom",
      path: ["option", "status"],
      message: "A failed humanoid option requires a physical failure result"
    });
  }
  if (motion.option.terminationReason === "motion_constraint_violated"
    && motion.option.monitor.phase !== "violated") {
    context.addIssue({
      code: "custom",
      path: ["option", "monitor", "phase"],
      message: "A constraint violation requires a violated option monitor"
    });
  }
  if (motion.option.status === "goal_unmet"
    && (motion.option.actualTerminationFrame === null
      || (motion.option.terminationReason !== "motion_goal_unmet"
        && motion.option.terminationReason !== "motion_goal_uncertain"))) {
    context.addIssue({
      code: "custom",
      path: ["option", "status"],
      message: "An unmet humanoid option requires a bounded goal result"
    });
  }
});

export const HumanoidSimulationStateCheckpointSchema = z.object({
  time: z.number().finite().nonnegative(),
  positions: FiniteArraySchema,
  velocities: FiniteArraySchema,
  controls: FiniteArraySchema,
  activations: FiniteArraySchema,
  accelerationWarmstart: FiniteArraySchema,
  requestedActuatorTorques: FiniteArraySchema.optional(),
  handCommandTargets: FiniteArraySchema.optional(),
  controller: z.object({
    protocol: z.literal("humanoid-controller-state-v1"),
    version: z.literal(1),
    implementation: z.string().trim().min(1),
    payload: JsonValueSchema
  }).strict()
}).strict();

const HumanoidWorldCheckpointBaseSchema = z.object({
  version: z.literal(1),
  motionGenerator: HumanoidMotionGeneratorDescriptorSchema.default(
    TASK_SPACE_MOTION_GENERATOR_DESCRIPTOR
  ),
  frame: z.number().int().nonnegative(),
  worldRevision: z.number().int().nonnegative(),
  routeSequence: z.number().int().nonnegative(),
  planRegistryEpoch: z.number().int().nonnegative().default(0),
  simulation: HumanoidSimulationStateCheckpointSchema,
  reference: HumanoidReferenceStateSchema,
  stationKeeping: HumanoidStationKeepingAnchorSchema.nullable().default(null),
  motions: z.array(StoredMotionSchema),
  routes: z.array(z.object({
    id: z.string().min(1),
    plan: NavigationPlanSchema,
    requestedTarget: Vec3Schema,
    requestedArrivalHeading: HumanoidNavigationArrivalHeadingSchema
      .nullable()
      .default(null),
    arrivalHeading: HumanoidNavigationArrivalHeadingSchema
      .nullable()
      .default(null),
    releaseJointTracking: z.boolean().default(false),
    createdRevision: z.number().int().nonnegative(),
    validatedRevision: z.number().int().nonnegative().optional(),
    validatedStateSha256: z.string().regex(/^[a-f0-9]{64}$/).optional(),
    expiresRevision: z.number().int().nonnegative().optional(),
    intentSha256: z.string().regex(/^[a-f0-9]{64}$/).optional(),
    revalidationCount: z.number().int().nonnegative().default(0),
    carriedObjectBindings: HumanoidCarriedObjectBindingSetSchema
      .nullable()
      .default(null),
    carriedObjectTaskSpaceTargets: HumanoidCarryTaskSpaceTargetsSchema.default([]),
    carriedObjectContinuation: HumanoidCarriedObjectContinuationEvidenceSchema
      .nullable()
      .default(null),
    carriedObjectUnauthorizedContacts: z.array(
      HumanoidCarriedObjectUnauthorizedContactSchema
    ).default([]),
    progress: HumanoidNavigationExecutionProgressSchema.nullable().default(null),
    terminal: HumanoidPlanTerminalSchema.nullable().default(null)
  }).strict().superRefine((route, context) => {
    if (route.intentSha256 !== undefined
      && route.intentSha256 !== humanoidNavigationIntentSha256(
        route.requestedTarget,
        route.requestedArrivalHeading
      )) {
      context.addIssue({
        code: "custom",
        path: ["intentSha256"],
        message: "Humanoid navigation intent identity does not match its model target"
      });
    }
    if (route.validatedRevision !== undefined
      && route.validatedRevision < route.createdRevision) {
      context.addIssue({
        code: "custom",
        path: ["validatedRevision"],
        message: "Humanoid navigation validation cannot precede intent creation"
      });
    }
    if (route.expiresRevision !== undefined
      && route.expiresRevision < route.createdRevision) {
      context.addIssue({
        code: "custom",
        path: ["expiresRevision"],
        message: "Humanoid navigation intent lease cannot expire before creation"
      });
    }
    if (route.progress
      && route.progress.waypoint_index > route.plan.waypoints.length) {
      context.addIssue({
        code: "custom",
        path: ["progress", "waypoint_index"],
        message: "Humanoid navigation progress exceeds its route"
      });
    }
    if (route.carriedObjectBindings
      && route.carriedObjectTaskSpaceTargets.length > 0
      && !humanoidCarryTaskSpaceTargetsMatchBindings(
        route.carriedObjectTaskSpaceTargets,
        route.carriedObjectBindings
      )) {
      context.addIssue({
        code: "custom",
        path: ["carriedObjectTaskSpaceTargets"],
        message: "Navigation carry targets do not match their authority bindings"
      });
    }
    if (!route.carriedObjectBindings
      && route.carriedObjectTaskSpaceTargets.length > 0) {
      context.addIssue({
        code: "custom",
        path: ["carriedObjectTaskSpaceTargets"],
        message: "Navigation carry targets require an authority binding"
      });
    }
    if (route.carriedObjectContinuation) {
      if (!route.carriedObjectBindings
        || route.carriedObjectContinuation.binding_set_sha256
          !== humanoidCarriedObjectBindingSetSha256(route.carriedObjectBindings)) {
        context.addIssue({
          code: "custom",
          path: ["carriedObjectContinuation"],
          message: "Navigation carry continuation does not match its authority binding"
        });
      }
      if (!route.progress
        || route.carriedObjectContinuation.observed_world_revision !== (
          (route.validatedRevision ?? route.createdRevision)
          + route.progress.committed_frame_count
        )) {
        context.addIssue({
          code: "custom",
          path: ["carriedObjectContinuation", "observed_world_revision"],
          message: "Navigation carry evidence is not aligned with committed progress"
        });
      }
      if (route.carriedObjectBindings) {
        const expected = route.carriedObjectBindings.bindings.map((binding) => ({
          binding_sha256: humanoidCarriedObjectBindingSha256(binding),
          object_id: binding.object_id,
          hand: binding.hand
        }));
        const actual = route.carriedObjectContinuation.bindings.map((binding) => ({
          binding_sha256: binding.binding_sha256,
          object_id: binding.object_id,
          hand: binding.hand
        }));
        if (JSON.stringify(actual) !== JSON.stringify(expected)) {
          context.addIssue({
            code: "custom",
            path: ["carriedObjectContinuation", "bindings"],
            message: "Navigation carry continuation omits or changes an authority binding"
          });
        }
      }
    }
    if (route.carriedObjectUnauthorizedContacts.length > 0
      && route.carriedObjectBindings === null) {
      context.addIssue({
        code: "custom",
        path: ["carriedObjectUnauthorizedContacts"],
        message: "Navigation carry collisions require an authority binding"
      });
    }
    if (route.carriedObjectBindings) {
      const bindingObjects = new Map(route.carriedObjectBindings.bindings.map((binding) => (
        [humanoidCarriedObjectBindingSha256(binding), binding.object_id]
      )));
      route.carriedObjectUnauthorizedContacts.forEach((contact, index) => {
        if (bindingObjects.get(contact.binding_sha256) === contact.object_id) return;
        context.addIssue({
          code: "custom",
          path: ["carriedObjectUnauthorizedContacts", index],
          message: "Navigation carry collision does not match an authority binding"
        });
      });
    }
    if (route.terminal && (!route.progress
      || route.terminal.plan_id !== route.id
      || route.terminal.total_frames !== route.progress.committed_frame_count
      || !terminalRevisionMatchesCommittedProgress(route.terminal, (
        (route.validatedRevision ?? route.createdRevision)
        + route.progress.committed_frame_count
      )))) {
      context.addIssue({
        code: "custom",
        path: ["terminal"],
        message: "Humanoid navigation terminal result does not match route progress"
      });
    }
  })),
  navigation: NavigationStateSchema,
  graspRegistry: HumanoidGraspRegistryCheckpointSchema,
  carriedObjectLifecycle: HumanoidCarriedObjectLifecycleCheckpointSchema
    .nullable()
    .default(null),
  physicalSafety: z.object({
    planId: z.string().min(1),
    evidence: HumanoidPhysicalSafetyEvidenceSchema
  }).strict().optional(),
  objectMemory: HumanoidObjectMemoryCheckpointSchema.default({
    version: 1,
    records: []
  })
}).strict();

type HumanoidWorldCheckpointBase = z.infer<
  typeof HumanoidWorldCheckpointBaseSchema
>;

function validateHumanoidWorldCheckpointAlignment(
  checkpoint: HumanoidWorldCheckpointBase
    | Omit<HumanoidWorldCheckpointBase, "graspRegistry">,
  context: z.RefinementCtx
): void {
  if (checkpoint.stationKeeping
    && (checkpoint.stationKeeping.sourceFrame > checkpoint.frame
      || checkpoint.stationKeeping.sourceWorldRevision > checkpoint.worldRevision)) {
    context.addIssue({
      code: "custom",
      path: ["stationKeeping"],
      message: "Humanoid station-keeping anchor is newer than its world checkpoint"
    });
  }
  if ("graspRegistry" in checkpoint
    && checkpoint.graspRegistry.last_frame !== checkpoint.frame) {
    context.addIssue({
      code: "custom",
      path: ["graspRegistry", "last_frame"],
      message: "Humanoid grasp registry must end at the checkpoint frame"
    });
  }
  if ("graspRegistry" in checkpoint && checkpoint.carriedObjectLifecycle) {
    const lifecycle = checkpoint.carriedObjectLifecycle;
    if (lifecycle.transition_frame > checkpoint.frame
      || lifecycle.transition_world_revision > checkpoint.worldRevision) {
      context.addIssue({
        code: "custom",
        path: ["carriedObjectLifecycle"],
        message: "Carried-object lifecycle transition is newer than its world checkpoint"
      });
    }
    const active = lifecycle.active_binding_set;
    if (active && (active.source_frame > checkpoint.frame
      || active.source_world_revision > checkpoint.worldRevision
      || active.grasp_contract_sha256 !== checkpoint.graspRegistry.contract_sha256
      || active.bindings.some((binding) => (
        !checkpoint.graspRegistry.portable_object_ids.includes(binding.object_id)
      )))) {
      context.addIssue({
        code: "custom",
        path: ["carriedObjectLifecycle", "active_binding_set"],
        message: "Active carried-object lifecycle authority is outside the world checkpoint"
      });
    }
  }
  for (let index = 0; index < checkpoint.motions.length; index += 1) {
    const terminal = checkpoint.motions[index]!.terminal;
    if (!terminal) continue;
    const frameLag = checkpoint.frame - terminal.final_frame;
    const revisionLag = checkpoint.worldRevision - terminal.final_world_revision;
    if (frameLag < 0 || revisionLag < 0 || frameLag !== revisionLag) {
      context.addIssue({
        code: "custom",
        path: ["motions", index, "terminal"],
        message: "Humanoid terminal result is not aligned with checkpoint time"
      });
    }
  }
  for (let index = 0; index < checkpoint.routes.length; index += 1) {
    const route = checkpoint.routes[index]!;
    const terminal = route.terminal;
    if ("graspRegistry" in checkpoint && route.carriedObjectBindings) {
      const carried = route.carriedObjectBindings;
      if (carried.source_frame > checkpoint.frame
        || carried.source_world_revision > checkpoint.worldRevision
        || carried.grasp_contract_sha256 !== checkpoint.graspRegistry.contract_sha256
        || carried.bindings.some((binding) => (
          !checkpoint.graspRegistry.portable_object_ids.includes(binding.object_id)
        ))) {
        context.addIssue({
          code: "custom",
          path: ["routes", index, "carriedObjectBindings"],
          message: "Navigation carry authority is outside the world checkpoint"
        });
      }
      if (carried.source_frame === checkpoint.frame
        && carried.source_world_revision === checkpoint.worldRevision
        && carried.grasp_registry_checkpoint_sha256
          !== humanoidGraspRegistryCheckpointSha256(checkpoint.graspRegistry)) {
        context.addIssue({
          code: "custom",
          path: ["routes", index, "carriedObjectBindings"],
          message: "Current navigation carry authority does not match the grasp registry"
        });
      }
      if (terminal === null
        && (route.progress?.committed_frame_count ?? 0) > 0
        && (!route.carriedObjectContinuation
          || route.carriedObjectContinuation.observed_frame !== checkpoint.frame
          || !route.carriedObjectContinuation.continued
          || route.carriedObjectUnauthorizedContacts.length > 0)) {
        context.addIssue({
          code: "custom",
          path: ["routes", index, "carriedObjectContinuation"],
          message: "Recoverable navigation must retain current carried-object evidence"
        });
      }
    }
    if (!terminal) continue;
    const frameLag = checkpoint.frame - terminal.final_frame;
    const revisionLag = checkpoint.worldRevision - terminal.final_world_revision;
    if (frameLag < 0 || revisionLag < 0 || frameLag !== revisionLag) {
      context.addIssue({
        code: "custom",
        path: ["routes", index, "terminal"],
        message: "Humanoid navigation terminal is not aligned with checkpoint time"
      });
    }
  }
}

export const HumanoidWorldCheckpointSchema =
  HumanoidWorldCheckpointBaseSchema.superRefine(
    validateHumanoidWorldCheckpointAlignment
  );

export const LegacyHumanoidWorldCheckpointSchema =
  HumanoidWorldCheckpointBaseSchema
    .omit({ graspRegistry: true })
    .superRefine(validateHumanoidWorldCheckpointAlignment);

export type HumanoidWorldCheckpoint = z.infer<typeof HumanoidWorldCheckpointSchema>;

function terminalRevisionMatchesCommittedProgress(
  terminal: z.infer<typeof HumanoidPlanTerminalSchema>,
  committedRevision: number
): boolean {
  if (terminal.final_world_revision === committedRevision) return true;
  return !terminal.accepted
    && (terminal.code === "plan_stale"
      || terminal.code === "plan_revalidation_failed")
    && terminal.final_world_revision > committedRevision;
}
