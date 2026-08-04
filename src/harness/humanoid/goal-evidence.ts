import { z } from "zod";
import {
  GoalPhysicalEvidenceSchema,
  type GoalPhysicalEvidence
} from "../../domain/goal-epoch.js";
import {
  GoalSchema,
  JsonValueSchema,
  Vec3Schema,
  type GoalPredicate,
  type JsonValue,
  type Scenario
} from "../../domain/schema.js";
import { goalSha256 } from "../../domain/goal-identity.js";
import { modelPayloadSha256 } from "../../domain/model-call-authority.js";
import { assertGoalSupported } from "../../runtime/goal-validation.js";
import type { HumanoidObjectToken } from "../../world/humanoid/object-memory.js";
import type { HumanoidWorldGraspState } from "../../world/humanoid/grasp-world-state.js";
import type { HumanoidSolidToken } from "../../world/humanoid/solid-observation.js";

const SHA256_PATTERN = /^[a-f0-9]{64}$/;

const WorldGraspAssessmentDescriptorSchema = z.object({
  frame: z.number().int().nonnegative(),
  object_id: z.string().trim().min(1),
  hand: z.enum(["left", "right"]),
  phase: z.enum(["idle", "stabilizing", "lifting", "holding", "verified"]),
  grasp_verified: z.boolean(),
  reason: z.string().trim().min(1),
  contact_status: z.enum([
    "missing",
    "insufficient_links",
    "insufficient_normal",
    "insufficient_geometry",
    "not_opposed",
    "opposed"
  ]),
  contact_links: z.array(z.string().trim().min(1)),
  support_status: z.enum(["supported", "unsupported", "insufficient_normal"]),
  lift_m: z.number().finite().nullable(),
  relative_pose_stable_frames: z.number().int().nonnegative(),
  lifted_hold_frames: z.number().int().nonnegative()
}).strict();

const WorldObservationDescriptorSchema = z.object({
  root_position: Vec3Schema,
  visible_object_ids: z.array(z.string().trim().min(1)),
  zone_ids: z.array(z.string().trim().min(1)),
  bounds: z.object({
    width: z.number().finite().positive(),
    depth: z.number().finite().positive()
  }).strict(),
  grasp: z.object({
    contract_sha256: z.string().regex(SHA256_PATTERN),
    assessments: z.array(WorldGraspAssessmentDescriptorSchema)
  }).strict()
}).strict();

const WorldObjectAffordanceDescriptorSchema = z.object({
  id: z.string().trim().min(1),
  role: z.enum(["manipulable", "fixture"]),
  kind: z.string().trim().min(1),
  color: z.string().trim().min(1),
  portable: z.boolean(),
  size: Vec3Schema,
  position: Vec3Schema,
  linear_velocity: Vec3Schema,
  angular_velocity: Vec3Schema,
  relation: z.object({
    distance_to_robot_m: z.number().finite().nonnegative(),
    bearing_rad: z.number().finite(),
    vertical_offset_m: z.number().finite(),
    distance_to_left_wrist_m: z.number().finite().nonnegative(),
    distance_to_right_wrist_m: z.number().finite().nonnegative()
  }).strict(),
  contacts: z.array(z.object({
    surface_kind: z.enum(["body", "hand_surface"]),
    surface: z.string().trim().min(1),
    normal_force_n: z.number().finite().nonnegative()
  }).strict())
}).strict();

const WorldZoneAffordanceDescriptorSchema = z.object({
  id: z.string().trim().min(1),
  color: z.string().trim().min(1),
  center: Vec3Schema,
  size: Vec3Schema
}).strict();

const WorldSolidAffordanceDescriptorSchema = z.object({
  id: z.string().trim().min(1),
  source_id: z.string().trim().min(1),
  kind: z.enum(["block", "fixed_object"]),
  center: Vec3Schema,
  size: Vec3Schema,
  relation: z.object({
    distance_to_robot_m: z.number().finite().nonnegative(),
    vertical_offset_m: z.number().finite()
  }).strict(),
  contacts: z.array(z.object({
    surface_kind: z.enum(["body", "hand_surface"]),
    surface: z.string().trim().min(1),
    normal_force_n: z.number().finite().nonnegative()
  }).strict())
}).strict();

const WorldObservationDescriptorV2Schema = WorldObservationDescriptorSchema.extend({
  objects: z.array(WorldObjectAffordanceDescriptorSchema),
  zones: z.array(WorldZoneAffordanceDescriptorSchema)
}).strict().superRefine((observation, context) => {
  const objectIds = observation.objects.map((object) => object.id);
  if (JSON.stringify(objectIds) !== JSON.stringify(observation.visible_object_ids)) {
    context.addIssue({
      code: "custom",
      path: ["objects"],
      message: "World affordance objects must match the sorted visible object identities"
    });
  }
  const zoneIds = observation.zones.map((zone) => zone.id);
  if (JSON.stringify(zoneIds) !== JSON.stringify(observation.zone_ids)) {
    context.addIssue({
      code: "custom",
      path: ["zones"],
      message: "World affordance zones must match the sorted zone identities"
    });
  }
});

const WorldObservationDescriptorV3Schema = WorldObservationDescriptorV2Schema.safeExtend({
  solid_ids: z.array(z.string().trim().min(1)),
  solids: z.array(WorldSolidAffordanceDescriptorSchema)
}).strict().superRefine((observation, context) => {
  const solidIds = observation.solids.map((solid) => solid.id);
  if (JSON.stringify(solidIds) !== JSON.stringify(observation.solid_ids)) {
    context.addIssue({
      code: "custom",
      path: ["solids"],
      message: "World affordance solids must match the sorted observable solid identities"
    });
  }
  if (new Set(solidIds).size !== solidIds.length) {
    context.addIssue({
      code: "custom",
      path: ["solid_ids"],
      message: "World affordance solid identities must be unique"
    });
  }
});

const GoalEvidenceArtifactV1Schema = z.object({
  version: z.literal(1),
  evidence: GoalPhysicalEvidenceSchema,
  payload: JsonValueSchema,
  observation: WorldObservationDescriptorSchema.nullable()
}).strict();

const GoalEvidenceArtifactV2Schema = z.object({
  version: z.literal(2),
  evidence: GoalPhysicalEvidenceSchema,
  payload: JsonValueSchema,
  observation: WorldObservationDescriptorV2Schema.nullable()
}).strict();

const GoalEvidenceArtifactV3Schema = z.object({
  version: z.literal(3),
  evidence: GoalPhysicalEvidenceSchema,
  payload: JsonValueSchema,
  observation: WorldObservationDescriptorV3Schema.nullable()
}).strict();

export const GoalEvidenceArtifactSchema = z.discriminatedUnion("version", [
  GoalEvidenceArtifactV1Schema,
  GoalEvidenceArtifactV2Schema,
  GoalEvidenceArtifactV3Schema
]).superRefine((artifact, context) => {
  if (artifact.evidence.content_sha256 !== modelPayloadSha256(artifact.payload)) {
    context.addIssue({
      code: "custom",
      path: ["evidence", "content_sha256"],
      message: "Goal evidence content hash does not match its durable payload"
    });
  }
  if ((artifact.evidence.kind === "world_checkpoint")
    !== (artifact.observation !== null)) {
    context.addIssue({
      code: "custom",
      path: ["observation"],
      message: "Only world checkpoint evidence can carry an observation descriptor"
    });
  }
  const payload = jsonRecord(artifact.payload);
  if (!payload) {
    context.addIssue({
      code: "custom",
      path: ["payload"],
      message: "Goal evidence payload must be an identity-bearing object"
    });
    return;
  }
  if (artifact.evidence.kind === "world_checkpoint") {
    if (payload.frame !== artifact.evidence.world_frame
      || payload.world_revision !== artifact.evidence.world_revision
      || payload.observation === undefined
      || modelPayloadSha256(payload.observation)
        !== modelPayloadSha256(artifact.observation)
      || artifact.evidence.ref !== `goal-world:${artifact.evidence.world_frame}:`
        + `${artifact.evidence.world_revision}:${artifact.evidence.content_sha256}`) {
      context.addIssue({
        code: "custom",
        path: ["payload"],
        message: "World Goal evidence metadata is not bound to its payload"
      });
    }
    artifact.observation?.grasp.assessments.forEach((assessment, index) => {
      if (assessment.frame !== artifact.evidence.world_frame) {
        context.addIssue({
          code: "custom",
          path: ["observation", "grasp", "assessments", index, "frame"],
          message: "Observed grasp assessment must belong to the evidence frame"
        });
      }
    });
    return;
  }
  if (artifact.evidence.kind === "action_receipt") {
    const receipt = jsonRecord(payload.receipt ?? null);
    if (payload.world_frame !== artifact.evidence.world_frame
      || payload.world_revision !== artifact.evidence.world_revision
      || typeof payload.transaction_id !== "string"
      || !receipt
      || receipt.transactionId !== payload.transaction_id
      || receipt.worldAfterRevision !== payload.world_revision
      || artifact.evidence.ref !== `action:${payload.transaction_id}`) {
      context.addIssue({
        code: "custom",
        path: ["payload"],
        message: "Action Goal evidence metadata is not bound to its receipt"
      });
    }
    return;
  }
  if (artifact.evidence.kind === "world_observation") {
    context.addIssue({
      code: "custom",
      path: ["evidence", "kind"],
      message: "World observation evidence requires a typed artifact contract"
    });
    return;
  }
  const evaluation = jsonRecord(payload.evaluation ?? null);
  const evaluatedGoal = GoalSchema.safeParse(evaluation?.goal);
  if (payload.world_frame !== artifact.evidence.world_frame
    || payload.world_revision !== artifact.evidence.world_revision
    || payload.goal_content_sha256 !== artifact.evidence.goal_content_sha256
    || !evaluation
    || evaluation.success !== true
    || evaluation.worldFrame !== payload.world_frame
    || evaluation.worldRevision !== payload.world_revision
    || !evaluatedGoal.success
    || goalSha256(evaluatedGoal.data) !== payload.goal_content_sha256
    || typeof payload.epoch_id !== "string"
    || artifact.evidence.ref !== `goal-evaluation:${payload.epoch_id}:`
      + `${artifact.evidence.world_revision}`) {
    context.addIssue({
      code: "custom",
      path: ["payload"],
      message: "Goal evaluation evidence metadata is not bound to its payload"
    });
  }
});

export type GoalEvidenceArtifact = z.infer<typeof GoalEvidenceArtifactSchema>;

export function createWorldGoalEvidence(input: {
  world: {
    frame: number;
    worldRevision: number;
    robot: {
      rootPosition: { x: number; y: number; z: number };
    };
    grasp: HumanoidWorldGraspState;
  };
  observation: {
    frame: number;
    worldRevision: number;
    objectTokens: ReadonlyArray<HumanoidObjectToken>;
    solidTokens: ReadonlyArray<HumanoidSolidToken>;
  };
  scenario: Scenario;
}): GoalEvidenceArtifact {
  if (input.observation.frame !== input.world.frame
    || input.observation.worldRevision !== input.world.worldRevision) {
    throw new Error("Goal evidence observation is not aligned with the authoritative world frame");
  }
  const visibleObjects = input.observation.objectTokens.filter((token) => (
    token.status === "visible"
      && token.observable
      && token.observedFrame === input.observation.frame
      && token.observedWorldRevision === input.observation.worldRevision
  )).sort((left, right) => compareCodePoints(left.id, right.id));
  const visibleObjectIds = visibleObjects.map((token) => token.id);
  if (new Set(visibleObjectIds).size !== visibleObjectIds.length) {
    throw new Error("Goal evidence contains duplicate visible object identities");
  }
  const observableSolids = [...input.observation.solidTokens]
    .sort((left, right) => compareCodePoints(left.id, right.id));
  const observableSolidIds = observableSolids.map((token) => token.id);
  if (new Set(observableSolidIds).size !== observableSolidIds.length) {
    throw new Error("Goal evidence contains duplicate observable solid identities");
  }
  for (const assessment of input.world.grasp.assessments) {
    if (assessment.frame !== input.world.frame) {
      throw new Error("Goal evidence grasp assessment is not from the authoritative world frame");
    }
  }
  const embodiedObjectIds = new Set([
    ...visibleObjectIds,
    ...input.world.grasp.assessments.flatMap((assessment) => (
      assessment.evidence.contact.status === "missing"
        ? []
        : [assessment.object_id]
    ))
  ]);
  const graspAssessments = input.world.grasp.assessments
    .filter((assessment) => embodiedObjectIds.has(assessment.object_id))
    .map((assessment) => ({
      frame: assessment.frame,
      object_id: assessment.object_id,
      hand: assessment.hand,
      phase: assessment.phase,
      grasp_verified: assessment.grasp_verified,
      reason: assessment.reason,
      contact_status: assessment.evidence.contact.status,
      contact_links: [...assessment.evidence.contact.distinct_force_qualified_links],
      support_status: assessment.evidence.support.status,
      lift_m: assessment.evidence.support.lift_m,
      relative_pose_stable_frames: assessment.evidence.relative_pose.stable_frames,
      lifted_hold_frames: assessment.evidence.lifted_hold_frames
    }));
  const observation = {
    root_position: structuredClone(input.world.robot.rootPosition),
    visible_object_ids: visibleObjectIds,
    zone_ids: input.scenario.zones.map((zone) => zone.id).sort(compareCodePoints),
    bounds: structuredClone(input.scenario.bounds),
    objects: visibleObjects.map((token) => ({
      id: token.id,
      role: token.role,
      kind: token.kind,
      color: token.color,
      portable: token.portable,
      size: structuredClone(token.size),
      position: structuredClone(token.position),
      linear_velocity: structuredClone(token.linearVelocity),
      angular_velocity: structuredClone(token.angularVelocity),
      relation: {
        distance_to_robot_m: token.relation.distanceToRobot,
        bearing_rad: token.relation.bearingRadians,
        vertical_offset_m: token.relation.verticalOffset,
        distance_to_left_wrist_m: token.relation.distanceToLeftWrist,
        distance_to_right_wrist_m: token.relation.distanceToRightWrist
      },
      contacts: token.currentContacts.map((contact) => "body" in contact
        ? {
            surface_kind: "body" as const,
            surface: contact.body,
            normal_force_n: contact.normalForce
          }
        : {
            surface_kind: "hand_surface" as const,
            surface: contact.handSurface,
            normal_force_n: contact.normalForce
          })
    })),
    zones: input.scenario.zones
      .map((zone) => ({
        id: zone.id,
        color: zone.color,
        center: structuredClone(zone.center),
        size: structuredClone(zone.size)
      }))
      .sort((left, right) => compareCodePoints(left.id, right.id)),
    solid_ids: observableSolidIds,
    solids: observableSolids.map((token) => ({
      id: token.id,
      source_id: token.sourceId,
      kind: token.kind,
      center: structuredClone(token.center),
      size: structuredClone(token.size),
      relation: {
        distance_to_robot_m: planarDistance(
          input.world.robot.rootPosition,
          token.center
        ),
        vertical_offset_m: token.center.y - input.world.robot.rootPosition.y
      },
      contacts: token.currentContacts.flatMap((contact) => {
        const descriptor = solidContactDescriptor(token.id, contact);
        return descriptor ? [descriptor] : [];
      })
    })),
    grasp: {
      contract_sha256: input.world.grasp.contractSha256,
      assessments: graspAssessments
    }
  };
  const payload = json({
    frame: input.world.frame,
    world_revision: input.world.worldRevision,
    observation
  });
  const contentSha256 = modelPayloadSha256(payload);
  return GoalEvidenceArtifactSchema.parse({
    version: 3,
    evidence: {
      ref: `goal-world:${input.world.frame}:${input.world.worldRevision}:${contentSha256}`,
      kind: "world_checkpoint",
      content_sha256: contentSha256,
      world_frame: input.world.frame,
      world_revision: input.world.worldRevision
    },
    payload,
    observation
  });
}

export function createActionGoalEvidence(input: {
  transactionId: string;
  worldFrame: number;
  worldRevision: number;
  receipt: JsonValue;
}): GoalEvidenceArtifact {
  return artifact({
    ref: `action:${input.transactionId}`,
    kind: "action_receipt",
    worldFrame: input.worldFrame,
    worldRevision: input.worldRevision,
    payload: json({
      transaction_id: input.transactionId,
      world_frame: input.worldFrame,
      world_revision: input.worldRevision,
      receipt: input.receipt
    })
  });
}

export function createGoalEvaluationEvidence(input: {
  epochId: string;
  goalContentSha256: string;
  worldFrame: number;
  worldRevision: number;
  evaluation: JsonValue;
}): GoalEvidenceArtifact {
  return artifact({
    ref: `goal-evaluation:${input.epochId}:${input.worldRevision}`,
    kind: "goal_evaluation",
    worldFrame: input.worldFrame,
    worldRevision: input.worldRevision,
    goalContentSha256: input.goalContentSha256,
    payload: json({
      epoch_id: input.epochId,
      goal_content_sha256: input.goalContentSha256,
      world_frame: input.worldFrame,
      world_revision: input.worldRevision,
      evaluation: input.evaluation
    })
  });
}

export function goalPredicateIsObservable(input: {
  predicate: GoalPredicate;
  worldRevision: number;
  evidenceRefs: readonly string[];
  artifacts: ReadonlyMap<string, GoalEvidenceArtifact>;
  scenario: Scenario;
}): boolean {
  try {
    assertGoalSupported({ summary: "observable predicate", predicates: [input.predicate] }, input.scenario);
  } catch {
    return false;
  }
  const observations = input.evidenceRefs.flatMap((ref) => {
    const artifact = input.artifacts.get(ref);
    return artifact?.evidence.world_revision === input.worldRevision && artifact.observation
      ? [artifact.observation]
      : [];
  });
  if (observations.length === 0) return false;
  if (input.predicate.type === "robot_at"
    || input.predicate.type === "robot_in_zone"
    || input.predicate.type === "end_effector_at") {
    return true;
  }
  if (input.predicate.type === "object_at"
    || input.predicate.type === "object_in_zone"
    || input.predicate.type === "object_placed") {
    const objectId = input.predicate.object_id;
    return observations.some((observation) => (
      observation.visible_object_ids.includes(objectId)
    ));
  }
  if (input.predicate.type === "object_grasped") {
    const objectId = input.predicate.object_id;
    const requiredHands = input.predicate.hand === "either"
      ? ["left", "right"] as const
      : [input.predicate.hand];
    return observations.some((observation) => {
      const assessments = observation.grasp.assessments.filter((assessment) => (
        assessment.object_id === objectId
          && requiredHands.includes(assessment.hand)
      ));
      const assessedHands = new Set(assessments.map((assessment) => assessment.hand));
      const evidenceComplete = requiredHands.every((hand) => assessedHands.has(hand));
      const groundedByBody = assessments.some((assessment) => (
        assessment.contact_status !== "missing"
      ));
      return evidenceComplete
        && (observation.visible_object_ids.includes(objectId)
          || groundedByBody);
    });
  }
  if (input.predicate.type === "block_removed") {
    const blockId = input.predicate.block_id;
    return observations.some((observation) => (
      "solid_ids" in observation
        && observation.solid_ids.includes(blockId)
        && observation.solids.some((solid) => (
          solid.id === blockId && solid.kind === "block"
        ))
    ));
  }
  return false;
}

function artifact(input: {
  ref: string;
  kind: "action_receipt" | "goal_evaluation";
  worldFrame: number;
  worldRevision: number;
  goalContentSha256?: string;
  payload: JsonValue;
}): GoalEvidenceArtifact {
  const contentSha256 = modelPayloadSha256(input.payload);
  const evidence: GoalPhysicalEvidence = input.kind === "goal_evaluation"
    ? {
        ref: input.ref,
        kind: input.kind,
        content_sha256: contentSha256,
        goal_content_sha256: input.goalContentSha256 ?? "0".repeat(64),
        world_frame: input.worldFrame,
        world_revision: input.worldRevision
      }
    : {
        ref: input.ref,
        kind: input.kind,
        content_sha256: contentSha256,
        world_frame: input.worldFrame,
        world_revision: input.worldRevision
      };
  return GoalEvidenceArtifactSchema.parse({
    version: 2,
    evidence,
    payload: input.payload,
    observation: null
  });
}

function compareCodePoints(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function planarDistance(
  left: Pick<z.infer<typeof Vec3Schema>, "x" | "z">,
  right: Pick<z.infer<typeof Vec3Schema>, "x" | "z">
): number {
  return Math.hypot(right.x - left.x, right.z - left.z);
}

function solidContactDescriptor(
  solidId: string,
  contact: HumanoidSolidToken["currentContacts"][number]
): z.infer<typeof WorldSolidAffordanceDescriptorSchema>["contacts"][number] | undefined {
  const solidIsFirst = contact.firstSolid === solidId;
  const handSurface = solidIsFirst
    ? contact.secondHandLink
    : contact.firstHandLink;
  if (handSurface) {
    return {
      surface_kind: "hand_surface",
      surface: handSurface,
      normal_force_n: contact.normalForce
    };
  }
  const body = solidIsFirst ? contact.secondBody : contact.firstBody;
  return body
    ? {
        surface_kind: "body",
        surface: body,
        normal_force_n: contact.normalForce
      }
    : undefined;
}

function jsonRecord(value: JsonValue): Record<string, JsonValue> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value
    : undefined;
}

function json(value: unknown): JsonValue {
  return JSON.parse(JSON.stringify(value)) as JsonValue;
}
