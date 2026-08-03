import { z } from "zod";
import { JsonValueSchema, Vec3Schema } from "../../domain/schema.js";
import {
  HumanoidMotionArtifactSchema,
  humanoidMotionArtifactSha256,
  HumanoidReferenceStateSchema
} from "./motion-artifact.js";
import {
  humanoidContactKey,
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
import { HUMANOID_BODY_NAMES } from "./model.js";
import { HumanoidObjectMemoryCheckpointSchema } from "./object-memory.js";

const FiniteArraySchema = z.array(z.number().finite());

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
  maximumEndEffectorErrorMeters: z.number().finite().nonnegative()
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
  contacts: z.array(z.object({
    body: z.enum(HUMANOID_BODY_NAMES),
    objectId: z.string().min(1).nullable(),
    normalForce: z.number().finite()
  }).strict()).optional(),
  drift: HumanoidMotionDriftEvidenceSchema.optional()
}).strict();

const HumanoidMotionExecutionProgressSchema = z.object({
  nextFrameIndex: z.number().int().nonnegative(),
  satisfiedContactKeys: z.array(z.string().min(1)),
  driftStreak: z.number().int().nonnegative().default(0),
  lastDrift: HumanoidMotionDriftEvidenceSchema.nullable().default(null),
  failure: HumanoidMotionExecutionFailureSchema.nullable()
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
  actualTerminationFrame: z.number().int().positive().nullable(),
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
  createdRevision: z.number().int().nonnegative(),
  option: HumanoidMotionOptionExecutionStateSchema.nullable().default(null),
  progress: HumanoidMotionExecutionProgressSchema.default({
    nextFrameIndex: 0,
    satisfiedContactKeys: [],
    driftStreak: 0,
    lastDrift: null,
    failure: null
  })
}).strict().superRefine((motion, context) => {
  if (motion.progress.nextFrameIndex > motion.artifact.frames.length) {
    context.addIssue({
      code: "custom",
      path: ["progress", "nextFrameIndex"],
      message: "Humanoid motion progress exceeds its artifact frame count"
    });
  }
  const allowedContactKeys = new Set(
    (motion.plan.contact_constraints ?? []).map((constraint) => (
      humanoidContactKey(constraint.body, constraint.object_id)
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
  if (certificate.validated_frame_limit !== motion.artifact.frames.length
    || certificate.rollout_frame_count !== motion.rollout.frames.length
    || certificate.rollout_frame_count !== motion.artifact.frames.length
    || certificate.rollout_sha256 !== humanoidMotionRolloutSha256(motion.rollout)
    || certificate.drift_consecutive_steps !== motion.rollout.limits.consecutive_steps
    || certificate.predicted_termination_frame > certificate.validated_frame_limit
    || certificate.predicted_termination_frame < certificate.stable_steps
    || certificate.stable_steps !== motion.option.contract.stable_steps
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

export const HumanoidWorldCheckpointSchema = z.object({
  version: z.literal(1),
  motionGenerator: HumanoidMotionGeneratorDescriptorSchema.default(
    TASK_SPACE_MOTION_GENERATOR_DESCRIPTOR
  ),
  frame: z.number().int().nonnegative(),
  worldRevision: z.number().int().nonnegative(),
  routeSequence: z.number().int().nonnegative(),
  simulation: z.object({
    time: z.number().finite().nonnegative(),
    positions: FiniteArraySchema,
    velocities: FiniteArraySchema,
    controls: FiniteArraySchema,
    activations: FiniteArraySchema,
    accelerationWarmstart: FiniteArraySchema,
    controller: z.object({
      protocol: z.literal("humanoid-controller-state-v1"),
      version: z.literal(1),
      implementation: z.string().trim().min(1),
      payload: JsonValueSchema
    }).strict()
  }).strict(),
  reference: HumanoidReferenceStateSchema,
  motions: z.array(StoredMotionSchema),
  routes: z.array(z.object({
    id: z.string().min(1),
    plan: NavigationPlanSchema,
    requestedTarget: Vec3Schema,
    createdRevision: z.number().int().nonnegative()
  }).strict()),
  navigation: NavigationStateSchema,
  objectMemory: HumanoidObjectMemoryCheckpointSchema.default({
    version: 1,
    records: []
  })
}).strict();

export type HumanoidWorldCheckpoint = z.infer<typeof HumanoidWorldCheckpointSchema>;
