import { createHash } from "node:crypto";
import { z } from "zod";
import {
  JsonValueSchema,
  QuaternionSchema,
  Vec3Schema,
  type JsonValue
} from "./schema.js";
import { HUMANOID_JOINT_NAMES } from "../world/humanoid/model.js";
import {
  HumanoidEmbodiedSkillCallSchema
} from "../world/humanoid/embodied-skill-call.js";
import {
  G1HandCoordinationSchema
} from "../world/humanoid/hand-coordination.js";
import {
  HumanoidHandPolicyAuthorityStateSchema
} from "../world/humanoid/hand-policy-authority.js";
import type {
  HumanoidPolicyControlFrame
} from "../world/humanoid/simulation.js";
import type {
  HumanoidPolicyState
} from "../world/humanoid/whole-body-controller.js";

const SHA256_PATTERN = /^[a-f0-9]{64}$/;
const JointVectorSchema = z.array(z.number().finite()).length(
  HUMANOID_JOINT_NAMES.length
);
const Vector3TupleSchema = z.tuple([
  z.number().finite(),
  z.number().finite(),
  z.number().finite()
]);
const QuaternionTupleSchema = z.tuple([
  z.number().finite(),
  z.number().finite(),
  z.number().finite(),
  z.number().finite()
]);

const HumanoidPolicyEnvironmentStateSchema = z.object({
  protocol: z.literal("humanoid-policy-environment-v1"),
  authority: z.literal("mujoco_state"),
  root_velocity_frame: z.literal("pelvis_imu"),
  root_linear_velocity: Vector3TupleSchema,
  root_angular_velocity: Vector3TupleSchema,
  root_position: Vec3Schema.optional(),
  end_effectors: z.record(z.string().trim().min(1), z.object({
    position: Vec3Schema,
    rotation: QuaternionSchema,
    linear_velocity: Vec3Schema.optional(),
    angular_velocity: Vec3Schema.optional()
  }).strict()),
  hands: z.record(z.string().trim().min(1), z.object({
    position: z.number().finite(),
    velocity: z.number().finite(),
    target: z.number().finite()
  }).strict()),
  contacts: z.array(z.object({
    position: Vec3Schema,
    normal: Vec3Schema,
    normal_force_n: z.number().finite().nonnegative(),
    first_body: z.string().trim().min(1).nullable(),
    second_body: z.string().trim().min(1).nullable(),
    first_object: z.string().trim().min(1).nullable(),
    second_object: z.string().trim().min(1).nullable(),
    first_solid: z.string().trim().min(1).nullable(),
    second_solid: z.string().trim().min(1).nullable(),
    first_hand_link: z.string().trim().min(1).nullable(),
    second_hand_link: z.string().trim().min(1).nullable()
  }).strict()),
  objects: z.array(z.object({
    id: z.string().trim().min(1),
    shape: z.enum(["box", "sphere", "cylinder", "capsule"]).optional(),
    size: Vec3Schema.optional(),
    mass_kg: z.number().finite().positive().optional(),
    friction: z.object({
      sliding: z.number().finite().nonnegative(),
      torsional: z.number().finite().nonnegative(),
      rolling: z.number().finite().nonnegative()
    }).strict().optional(),
    position: Vec3Schema,
    rotation: QuaternionSchema,
    linear_velocity: Vec3Schema,
    angular_velocity: Vec3Schema,
    articulation: z.object({
      type: z.enum(["hinge", "slide"]),
      position: z.number().finite(),
      velocity: z.number().finite(),
      minimum: z.number().finite(),
      maximum: z.number().finite()
    }).strict().nullable()
  }).strict()),
  zones: z.array(z.object({
    id: z.string().trim().min(1),
    center: Vec3Schema,
    size: Vec3Schema
  }).strict()).optional(),
  feet: z.object({
    left: z.object({
      touching: z.boolean(),
      normal_force_n: z.number().finite().nonnegative()
    }).strict(),
    right: z.object({
      touching: z.boolean(),
      normal_force_n: z.number().finite().nonnegative()
    }).strict()
  }).strict().optional(),
  center_of_mass: Vec3Schema.optional(),
  center_of_mass_velocity: Vec3Schema.optional()
}).strict();

const HumanoidPolicyStateRecordSchema = z.object({
  joint_positions: JointVectorSchema,
  joint_velocities: JointVectorSchema,
  root_quaternion_wxyz: QuaternionTupleSchema,
  root_angular_velocity: Vector3TupleSchema,
  environment: HumanoidPolicyEnvironmentStateSchema.nullable()
}).strict();

const HumanoidReferenceRecordSchema = z.object({
  joint_positions: JointVectorSchema,
  joint_velocities: JointVectorSchema,
  joint_tracking_weights: z.array(z.number().finite().min(0).max(1)).length(
    HUMANOID_JOINT_NAMES.length
  ),
  root_velocity: z.tuple([z.number().finite(), z.number().finite()]),
  root_yaw_velocity: z.number().finite(),
  root_height: z.number().finite().positive(),
  root_roll: z.number().finite(),
  root_pitch: z.number().finite()
}).strict();

const HumanoidControllerTensorTraceSchema = z.object({
  protocol: z.literal("humanoid-controller-tensor-trace-v1"),
  role: z.enum(["direct", "primary", "fallback"]),
  implementation: z.string().trim().min(1),
  observation: z.object({
    protocol: z.string().trim().min(1),
    values: z.array(z.number().finite()).min(1)
  }).strict(),
  action: z.object({
    protocol: z.string().trim().min(1),
    values: z.array(z.number().finite()).min(1)
  }).strict()
}).strict();

const HumanoidControllerInferenceTraceSchema = z.object({
  protocol: z.literal("humanoid-controller-inference-trace-v1"),
  implementation: z.string().trim().min(1),
  route: z.enum(["direct", "primary", "fallback", "upper_body_overlay"]),
  components: z.array(HumanoidControllerTensorTraceSchema)
}).strict().superRefine((trace, context) => {
  const roles = trace.components.map(({ role }) => role);
  if (new Set(roles).size !== roles.length) {
    context.addIssue({
      code: "custom",
      path: ["components"],
      message: "Controller tensor component roles must be unique"
    });
  }
  if (trace.route === "upper_body_overlay"
    && (!roles.includes("primary") || !roles.includes("fallback"))) {
    context.addIssue({
      code: "custom",
      path: ["components"],
      message: "Upper-body overlay traces require primary and fallback tensors"
    });
  }
});

const HumanoidActuationRecordSchema = z.object({
  kind: z.literal("joint_position_pd"),
  positions: JointVectorSchema,
  stiffness: z.array(z.number().finite().nonnegative()).length(
    HUMANOID_JOINT_NAMES.length
  ),
  damping: z.array(z.number().finite().nonnegative()).length(
    HUMANOID_JOINT_NAMES.length
  ),
  hand_synergy: z.object({
    protocol: z.literal("humanoid-authorized-hand-synergy-command-v1"),
    authority: HumanoidHandPolicyAuthorityStateSchema,
    action: z.array(z.number().finite().min(-1).max(1)).length(8),
    coordination: G1HandCoordinationSchema,
    maximum_closing_joint_lead_radians: z.literal(0.25)
  }).strict().optional()
}).strict();

export const DenseHumanoidPolicyFrameSchema = z.object({
  protocol: z.literal("hear-dense-humanoid-policy-frame-v1"),
  run_id: z.string().regex(/^[a-zA-Z0-9_-]+$/),
  call_id: z.string().trim().min(1),
  local_frame_index: z.number().int().nonnegative(),
  call_step_index: z.number().int().nonnegative(),
  world_frame_before: z.number().int().nonnegative(),
  world_frame_after: z.number().int().positive(),
  world_revision_before: z.number().int().nonnegative(),
  world_revision_after: z.number().int().positive(),
  control_step_seconds: z.number().finite().positive(),
  task_command: HumanoidEmbodiedSkillCallSchema,
  pre_state: HumanoidPolicyStateRecordSchema,
  reference: HumanoidReferenceRecordSchema,
  controller: z.object({
    descriptor: JsonValueSchema,
    execution: JsonValueSchema.nullable(),
    inference: HumanoidControllerInferenceTraceSchema.nullable()
  }).strict(),
  actuation: HumanoidActuationRecordSchema,
  post_state: HumanoidPolicyStateRecordSchema,
  physical_outcome: z.object({
    root_position: Vec3Schema,
    root_rotation: QuaternionSchema,
    joint_positions: JointVectorSchema,
    joint_velocities: JointVectorSchema,
    support: z.enum(["none", "left", "right", "double"]),
    support_margin_m: z.number().finite().nullable(),
    contact_count: z.number().int().nonnegative(),
    non_foot_environment_contacts: z.array(z.string().trim().min(1)),
    fallen: z.boolean()
  }).strict(),
  supervision: z.object({
    kind: z.enum([
      "reference_teacher",
      "paired_teacher",
      "policy_only",
      "untyped_controller"
    ]),
    teacher_component_roles: z.array(z.enum(["direct", "primary", "fallback"]))
  }).strict(),
  previous_frame_sha256: z.string().regex(SHA256_PATTERN).nullable(),
  frame_sha256: z.string().regex(SHA256_PATTERN)
}).strict().superRefine((frame, context) => {
  if (frame.task_command.identity.callId !== frame.call_id
    || frame.task_command.identity.runtimeKind !== "semantic_skill") {
    context.addIssue({
      code: "custom",
      path: ["task_command", "identity"],
      message: "Dense policy frames require their semantic Skill Call identity"
    });
  }
  if (frame.task_command.window.stepIndex !== frame.call_step_index
    || frame.task_command.authority.worldFrame !== frame.world_frame_before
    || frame.task_command.authority.worldRevision !== frame.world_revision_before
    || frame.world_frame_after !== frame.world_frame_before + 1
    || frame.world_revision_after !== frame.world_revision_before + 1) {
    context.addIssue({
      code: "custom",
      path: ["world_frame_after"],
      message: "Dense policy frame authority and control-step progression are inconsistent"
    });
  }
  if (frame.frame_sha256 !== denseHumanoidPolicyFrameSha256(frame)) {
    context.addIssue({
      code: "custom",
      path: ["frame_sha256"],
      message: "Dense policy frame hash does not match its payload"
    });
  }
});

export type DenseHumanoidPolicyFrame = z.infer<
  typeof DenseHumanoidPolicyFrameSchema
>;

export const DensePolicyRolloutReferenceSchema = z.object({
  available: z.literal(true),
  protocol: z.literal("hear-dense-policy-rollout-reference-v1"),
  dataset_ref: z.string().regex(/^dense-policy-jsonl-v1:[^\\]+$/),
  file_sha256: z.string().regex(SHA256_PATTERN),
  frame_count: z.number().int().positive(),
  first_local_frame_index: z.literal(0),
  last_local_frame_index: z.number().int().nonnegative(),
  first_call_step_index: z.number().int().nonnegative(),
  last_call_step_index: z.number().int().nonnegative(),
  first_world_frame: z.number().int().nonnegative(),
  last_world_frame: z.number().int().positive(),
  first_world_revision: z.number().int().nonnegative(),
  last_world_revision: z.number().int().positive(),
  first_frame_sha256: z.string().regex(SHA256_PATTERN),
  last_frame_sha256: z.string().regex(SHA256_PATTERN),
  complete_from_window_start: z.boolean(),
  complete_through_execution_end: z.boolean(),
  missing_call_step_count: z.number().int().nonnegative(),
  missing_world_frame_count: z.number().int().nonnegative(),
  teacher_frame_count: z.number().int().nonnegative(),
  paired_teacher_frame_count: z.number().int().nonnegative(),
  observation_protocols: z.array(z.string().trim().min(1)),
  action_protocols: z.array(z.string().trim().min(1))
}).strict().superRefine((reference, context) => {
  if (reference.last_local_frame_index !== reference.frame_count - 1
    || reference.teacher_frame_count > reference.frame_count
    || reference.paired_teacher_frame_count > reference.teacher_frame_count) {
    context.addIssue({
      code: "custom",
      message: "Dense policy rollout counts are inconsistent"
    });
  }
});

export type DensePolicyRolloutReference = z.infer<
  typeof DensePolicyRolloutReferenceSchema
>;

export function createDenseHumanoidPolicyFrame(input: {
  runId: string;
  localFrameIndex: number;
  previousFrameSha256: string | null;
  frame: HumanoidPolicyControlFrame;
}): DenseHumanoidPolicyFrame {
  const task = HumanoidEmbodiedSkillCallSchema.parse(input.frame.taskCommand);
  if (task.identity.runtimeKind !== "semantic_skill") {
    throw new Error("Dense policy collection accepts only semantic Skill Calls");
  }
  const snapshot = input.frame.postSnapshot;
  const inference = input.frame.controllerInference
    ? HumanoidControllerInferenceTraceSchema.parse(
        input.frame.controllerInference
      )
    : null;
  const withoutHash = {
    protocol: "hear-dense-humanoid-policy-frame-v1" as const,
    run_id: input.runId,
    call_id: task.identity.callId,
    local_frame_index: input.localFrameIndex,
    call_step_index: task.window.stepIndex,
    world_frame_before: task.authority.worldFrame,
    world_frame_after: task.authority.worldFrame + 1,
    world_revision_before: task.authority.worldRevision,
    world_revision_after: task.authority.worldRevision + 1,
    control_step_seconds: task.window.controlStepSeconds,
    task_command: task,
    pre_state: policyStateRecord(input.frame.preState),
    reference: {
      joint_positions: [...input.frame.reference.jointPositions],
      joint_velocities: [...input.frame.reference.jointVelocities],
      joint_tracking_weights: [...input.frame.reference.jointTrackingWeights],
      root_velocity: [...input.frame.reference.rootVelocity],
      root_yaw_velocity: input.frame.reference.rootYawVelocity,
      root_height: input.frame.reference.rootHeight,
      root_roll: input.frame.reference.rootRoll,
      root_pitch: input.frame.reference.rootPitch
    },
    controller: {
      descriptor: json(input.frame.controller),
      execution: input.frame.controllerExecution
        ? json(input.frame.controllerExecution)
        : null,
      inference
    },
    actuation: {
      kind: "joint_position_pd" as const,
      positions: [...input.frame.actuation.positions],
      stiffness: [...input.frame.actuation.stiffness],
      damping: [...input.frame.actuation.damping],
      ...(input.frame.actuation.handSynergy
        ? {
            hand_synergy: {
              protocol: input.frame.actuation.handSynergy.protocol,
              authority: structuredClone(
                input.frame.actuation.handSynergy.authority
              ),
              action: [...input.frame.actuation.handSynergy.action],
              coordination: structuredClone(
                input.frame.actuation.handSynergy.coordination
              ),
              maximum_closing_joint_lead_radians:
                input.frame.actuation.handSynergy.maximumClosingJointLeadRadians
            }
          }
        : {})
    },
    post_state: policyStateRecord(input.frame.postState),
    physical_outcome: {
      root_position: { ...snapshot.rootPosition },
      root_rotation: { ...snapshot.rootRotation },
      joint_positions: HUMANOID_JOINT_NAMES.map(
        (name) => snapshot.joints[name].position
      ),
      joint_velocities: HUMANOID_JOINT_NAMES.map(
        (name) => snapshot.joints[name].velocity
      ),
      support: snapshot.balance.support,
      support_margin_m: snapshot.balance.supportMargin,
      contact_count: snapshot.contactCount,
      non_foot_environment_contacts: [...snapshot.nonFootEnvironmentContacts],
      fallen: snapshot.fallen
    },
    supervision: supervisionRecord(
      input.frame.controllerExecution?.mode ?? null,
      inference
    ),
    previous_frame_sha256: input.previousFrameSha256
  };
  return DenseHumanoidPolicyFrameSchema.parse({
    ...withoutHash,
    frame_sha256: denseHumanoidPolicyFrameSha256(withoutHash)
  });
}

function denseHumanoidPolicyFrameSha256(
  frame: unknown
): string {
  const { frame_sha256: _frameSha256, ...identity } = frame as
    DenseHumanoidPolicyFrame;
  return createHash("sha256")
    .update(canonicalJson(identity))
    .digest("hex");
}

function policyStateRecord(state: HumanoidPolicyState): z.infer<
  typeof HumanoidPolicyStateRecordSchema
> {
  const environment = state.environment;
  return HumanoidPolicyStateRecordSchema.parse({
    joint_positions: [...Array.from(state.jointPositions)],
    joint_velocities: [...Array.from(state.jointVelocities)],
    root_quaternion_wxyz: [...state.rootQuaternion],
    root_angular_velocity: [...state.rootAngularVelocity],
    environment: environment
      ? {
          protocol: environment.protocol,
          authority: environment.authority,
          root_velocity_frame: environment.rootVelocityFrame,
          root_linear_velocity: [...environment.rootLinearVelocity],
          root_angular_velocity: [...environment.rootAngularVelocity],
          ...(environment.rootPosition
            ? { root_position: { ...environment.rootPosition } }
            : {}),
          end_effectors: Object.fromEntries(Object.entries(
            environment.endEffectors
          ).map(([name, value]) => [name, {
            position: { ...value.position },
            rotation: { ...value.rotation },
            ...(value.linearVelocity
              ? { linear_velocity: { ...value.linearVelocity } }
              : {}),
            ...(value.angularVelocity
              ? { angular_velocity: { ...value.angularVelocity } }
              : {})
          }])),
          hands: Object.fromEntries(Object.entries(environment.hands).map(
            ([name, value]) => [name, { ...value }]
          )),
          contacts: environment.contacts.map((contact) => ({
            position: { ...contact.position },
            normal: { ...contact.normal },
            normal_force_n: contact.normalForce,
            first_body: contact.firstBody,
            second_body: contact.secondBody,
            first_object: contact.firstObject,
            second_object: contact.secondObject,
            first_solid: contact.firstSolid,
            second_solid: contact.secondSolid,
            first_hand_link: contact.firstHandLink,
            second_hand_link: contact.secondHandLink
          })),
          objects: environment.objects.map((object) => ({
            id: object.id,
            ...(object.shape ? { shape: object.shape } : {}),
            ...(object.size ? { size: { ...object.size } } : {}),
            ...(object.massKg !== undefined ? { mass_kg: object.massKg } : {}),
            ...(object.friction ? { friction: { ...object.friction } } : {}),
            position: { ...object.position },
            rotation: { ...object.rotation },
            linear_velocity: { ...object.linearVelocity },
            angular_velocity: { ...object.angularVelocity },
            articulation: object.articulation
              ? { ...object.articulation }
              : null
          })),
          ...(environment.zones
            ? {
                zones: environment.zones.map((zone) => ({
                  id: zone.id,
                  center: { ...zone.center },
                  size: { ...zone.size }
                }))
              }
            : {}),
          ...(environment.feet
            ? {
                feet: {
                  left: {
                    touching: environment.feet.left.touching,
                    normal_force_n: environment.feet.left.normalForce
                  },
                  right: {
                    touching: environment.feet.right.touching,
                    normal_force_n: environment.feet.right.normalForce
                  }
                }
              }
            : {}),
          ...(environment.centerOfMass
            ? { center_of_mass: { ...environment.centerOfMass } }
            : {}),
          ...(environment.centerOfMassVelocity
            ? {
                center_of_mass_velocity: {
                  ...environment.centerOfMassVelocity
                }
              }
            : {})
        }
      : null
  });
}

function supervisionRecord(
  mode: "learned_policy" | "reference_control" | "hybrid_control" | null,
  inference: z.infer<typeof HumanoidControllerInferenceTraceSchema> | null
): {
  kind: "reference_teacher" | "paired_teacher" | "policy_only" | "untyped_controller";
  teacher_component_roles: Array<"direct" | "primary" | "fallback">;
} {
  const roles = inference?.components.map(({ role }) => role) ?? [];
  if (mode === "reference_control"
    && (roles.includes("direct") || roles.includes("fallback"))) {
    return {
      kind: "reference_teacher",
      teacher_component_roles: roles.filter(
        (role): role is "direct" | "fallback" => (
          role === "direct" || role === "fallback"
        )
      )
    };
  }
  if (mode === "hybrid_control"
    && roles.includes("primary")
    && roles.includes("fallback")) {
    return { kind: "paired_teacher", teacher_component_roles: [
      "primary",
      "fallback"
    ] };
  }
  if (mode === "learned_policy" && inference) {
    return { kind: "policy_only", teacher_component_roles: [] };
  }
  return { kind: "untyped_controller", teacher_component_roles: [] };
}

function json(value: unknown): JsonValue {
  return JsonValueSchema.parse(JSON.parse(JSON.stringify(value)) as unknown);
}

function canonicalJson(value: unknown): string {
  return JSON.stringify(canonicalValue(value));
}

function canonicalValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalValue);
  if (value === null || typeof value !== "object") return value;
  return Object.fromEntries(Object.entries(value as Record<string, unknown>)
    .filter(([, entry]) => entry !== undefined)
    .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
    .map(([key, entry]) => [key, canonicalValue(entry)]));
}
