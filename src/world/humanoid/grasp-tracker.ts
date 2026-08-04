import { createHash } from "node:crypto";
import { z } from "zod";
import {
  QuaternionSchema,
  Vec3Schema,
  type Vec3
} from "../../domain/schema.js";
import {
  inverseQuaternion,
  multiplyQuaternion,
  normalizeQuaternion,
  quaternionAngularDistance,
  rotateVector,
  subtract,
  vectorLength
} from "../geometry.js";
import { g1HandObjectContacts } from "./hand-contact-evidence.js";
import {
  G1_HAND_CONTACT_SURFACE_NAMES,
  type G1HandContactSurfaceName
} from "./morphology.js";
import type {
  HumanoidContactSnapshot,
  HumanoidSimulationSnapshot
} from "./simulation.js";

const HandSchema = z.enum(["left", "right"]);
const HandLinkSchema = z.enum(G1_HAND_CONTACT_SURFACE_NAMES);
const UnitVec3Schema = Vec3Schema.refine(
  (value) => Math.abs(vectorLength(value) - 1) <= 1e-6,
  "Vector must be normalized"
);
const UnitQuaternionSchema = QuaternionSchema.refine(
  (value) => Math.abs(Math.hypot(value.x, value.y, value.z, value.w) - 1) <= 1e-6,
  "Quaternion must be normalized"
);
const PoseSchema = z.object({
  position: Vec3Schema,
  rotation: UnitQuaternionSchema
}).strict();

const MAX_LINKS_PER_HAND = Math.max(
  G1_HAND_CONTACT_SURFACE_NAMES.filter((name) => name.startsWith("left_")).length,
  G1_HAND_CONTACT_SURFACE_NAMES.filter((name) => name.startsWith("right_")).length
);

export const HumanoidGraspContractSchema = z.object({
  protocol: z.literal("humanoid-grasp-contract-v1"),
  world_up: UnitVec3Schema,
  minimum_distinct_contact_links: z.number().int().min(2).max(MAX_LINKS_PER_HAND),
  minimum_contact_normal_force_n: z.number().finite().positive(),
  maximum_opposing_normal_dot: z.number().finite().min(-1).negative(),
  maximum_opposing_position_dot: z.number().finite().min(-1).negative(),
  minimum_opposing_contact_separation_m: z.number().finite().positive(),
  minimum_contact_radial_distance_m: z.number().finite().positive(),
  maximum_relative_translation_drift_m: z.number().finite().nonnegative(),
  maximum_relative_rotation_drift_rad: z.number().finite().nonnegative().max(Math.PI),
  minimum_relative_pose_stable_frames: z.number().int().positive(),
  minimum_lift_m: z.number().finite().positive(),
  minimum_lifted_hold_frames: z.number().int().positive(),
  minimum_support_normal_force_n: z.number().finite().positive(),
  minimum_support_up_dot: z.number().finite().positive().max(1)
}).strict();

export type HumanoidGraspContract = z.infer<typeof HumanoidGraspContractSchema>;
export type HumanoidGraspHand = z.infer<typeof HandSchema>;

export function humanoidGraspContractSha256(
  contract: HumanoidGraspContract
): string {
  return createHash("sha256")
    .update(JSON.stringify(HumanoidGraspContractSchema.parse(contract)))
    .digest("hex");
}

const HandContactSchema = z.object({
  hand_link: HandLinkSchema,
  position: Vec3Schema,
  normal_from_hand: UnitVec3Schema.nullable(),
  normal_force_n: z.number().finite().nonnegative()
}).strict();

const SupportContactSchema = z.object({
  position: Vec3Schema,
  normal_toward_object: UnitVec3Schema.nullable(),
  normal_force_n: z.number().finite().nonnegative(),
  counterpart_kind: z.enum(["environment", "object", "humanoid"]),
  counterpart_id: z.string().min(1).nullable()
}).strict().superRefine((contact, context) => {
  if ((contact.counterpart_kind === "environment") !== (contact.counterpart_id === null)) {
    context.addIssue({
      code: "custom",
      path: ["counterpart_id"],
      message: "Environment support has no entity id; entity support requires one"
    });
  }
});

export const HumanoidGraspObservationSchema = z.object({
  protocol: z.literal("humanoid-grasp-observation-v1"),
  frame: z.number().int().nonnegative(),
  object_id: z.string().trim().min(1),
  hand: HandSchema,
  object_pose: PoseSchema,
  hand_anchor_pose: PoseSchema,
  hand_contacts: z.array(HandContactSchema),
  support_contacts: z.array(SupportContactSchema)
}).strict().superRefine((observation, context) => {
  const prefix = `${observation.hand}_hand_`;
  observation.hand_contacts.forEach((contact, index) => {
    if (!contact.hand_link.startsWith(prefix)) {
      context.addIssue({
        code: "custom",
        path: ["hand_contacts", index, "hand_link"],
        message: "Hand contact does not belong to the tracked hand"
      });
    }
  });
});

export type HumanoidGraspObservation = z.infer<typeof HumanoidGraspObservationSchema>;

const RelativePoseSchema = z.object({
  translation: Vec3Schema,
  rotation: UnitQuaternionSchema
}).strict();

const AttemptSchema = z.object({
  first_stable_frame: z.number().int().nonnegative(),
  stable_frames: z.number().int().positive(),
  lifted_hold_frames: z.number().int().nonnegative(),
  has_left_support: z.boolean(),
  reference_relative_pose: RelativePoseSchema
}).strict().superRefine((attempt, context) => {
  if (attempt.lifted_hold_frames > attempt.stable_frames) {
    context.addIssue({
      code: "custom",
      path: ["lifted_hold_frames"],
      message: "Lift hold cannot exceed stable contact frames"
    });
  }
  if (attempt.lifted_hold_frames > 0 && !attempt.has_left_support) {
    context.addIssue({
      code: "custom",
      path: ["has_left_support"],
      message: "Lift hold requires observed support departure"
    });
  }
});

const TrackSchema = z.object({
  object_id: z.string().trim().min(1),
  hand: HandSchema,
  last_frame: z.number().int().nonnegative(),
  support_baseline: z.object({
    frame: z.number().int().nonnegative(),
    projection_m: z.number().finite()
  }).strict().nullable(),
  attempt: AttemptSchema.nullable()
}).strict().superRefine((track, context) => {
  if (track.support_baseline && track.support_baseline.frame > track.last_frame) {
    context.addIssue({
      code: "custom",
      path: ["support_baseline", "frame"],
      message: "Support baseline cannot be newer than the track"
    });
  }
  if (!track.attempt) return;
  if (track.attempt.first_stable_frame + track.attempt.stable_frames - 1
    !== track.last_frame) {
    context.addIssue({
      code: "custom",
      path: ["attempt", "stable_frames"],
      message: "Stable grasp frames must be consecutive through the last frame"
    });
  }
  if (track.attempt.lifted_hold_frames > 0 && track.support_baseline === null) {
    context.addIssue({
      code: "custom",
      path: ["support_baseline"],
      message: "Lift hold requires a measured support baseline"
    });
  }
});

export const HumanoidGraspTrackerCheckpointSchema = z.object({
  protocol: z.literal("humanoid-grasp-tracker-checkpoint-v1"),
  contract: HumanoidGraspContractSchema,
  tracks: z.array(TrackSchema)
}).strict().superRefine((checkpoint, context) => {
  const keys = new Set<string>();
  checkpoint.tracks.forEach((track, index) => {
    const key = trackKey(track.object_id, track.hand);
    if (keys.has(key)) {
      context.addIssue({
        code: "custom",
        path: ["tracks", index],
        message: "Grasp tracker contains a duplicate object-hand track"
      });
    }
    keys.add(key);
  });
});

export type HumanoidGraspTrackerCheckpoint = z.infer<
  typeof HumanoidGraspTrackerCheckpointSchema
>;
type GraspTrack = HumanoidGraspTrackerCheckpoint["tracks"][number];
type GraspAttempt = NonNullable<GraspTrack["attempt"]>;

const OpposingPairSchema = z.object({
  first_link: HandLinkSchema,
  second_link: HandLinkSchema,
  first_position: Vec3Schema,
  second_position: Vec3Schema,
  first_normal_from_hand: UnitVec3Schema,
  second_normal_from_hand: UnitVec3Schema,
  first_normal_force_n: z.number().finite().nonnegative(),
  second_normal_force_n: z.number().finite().nonnegative(),
  separation_m: z.number().finite().nonnegative(),
  normal_dot: z.number().finite().min(-1).max(1),
  position_dot: z.number().finite().min(-1).max(1)
}).strict();

export const HumanoidGraspAssessmentSchema = z.object({
  protocol: z.literal("humanoid-grasp-assessment-v1"),
  frame: z.number().int().nonnegative(),
  object_id: z.string().trim().min(1),
  hand: HandSchema,
  phase: z.enum(["idle", "stabilizing", "lifting", "holding", "verified"]),
  grasp_verified: z.boolean(),
  reason: z.enum([
    "contact_missing",
    "contact_links_insufficient",
    "contact_normal_insufficient",
    "contact_geometry_insufficient",
    "contacts_not_opposed",
    "support_evidence_insufficient",
    "frame_discontinuity",
    "relative_pose_unstable",
    "support_returned",
    "relative_pose_stabilizing",
    "object_supported",
    "support_baseline_missing",
    "lift_height_insufficient",
    "lift_hold_incomplete",
    "grasp_verified"
  ]),
  reset_reason: z.enum([
    "contact_lost",
    "contact_links_insufficient",
    "contact_normal_insufficient",
    "contact_geometry_insufficient",
    "contacts_not_opposed",
    "support_evidence_insufficient",
    "frame_discontinuity",
    "relative_pose_unstable",
    "support_returned"
  ]).nullable(),
  evidence: z.object({
    contact: z.object({
      status: z.enum([
        "missing",
        "insufficient_links",
        "insufficient_normal",
        "insufficient_geometry",
        "not_opposed",
        "opposed"
      ]),
      observed_contact_count: z.number().int().nonnegative(),
      force_qualified_contact_count: z.number().int().nonnegative(),
      distinct_force_qualified_links: z.array(HandLinkSchema),
      distinct_normal_qualified_links: z.array(HandLinkSchema),
      opposing_pair: OpposingPairSchema.nullable()
    }).strict(),
    support: z.object({
      status: z.enum(["supported", "unsupported", "insufficient_normal"]),
      candidate_contact_count: z.number().int().nonnegative(),
      force_qualified_contact_count: z.number().int().nonnegative(),
      upward_contact_count: z.number().int().nonnegative(),
      baseline_projection_m: z.number().finite().nullable(),
      current_projection_m: z.number().finite(),
      lift_m: z.number().finite().nullable()
    }).strict(),
    relative_pose: z.object({
      stable_frames: z.number().int().nonnegative(),
      translation_drift_m: z.number().finite().nonnegative().nullable(),
      rotation_drift_rad: z.number().finite().nonnegative().nullable()
    }).strict(),
    lifted_hold_frames: z.number().int().nonnegative()
  }).strict()
}).strict().superRefine((assessment, context) => {
  if ((assessment.phase === "verified") !== assessment.grasp_verified
    || (assessment.reason === "grasp_verified") !== assessment.grasp_verified) {
    context.addIssue({
      code: "custom",
      message: "Verified grasp status is inconsistent"
    });
  }
  if ((assessment.evidence.contact.opposing_pair !== null)
    !== (assessment.evidence.contact.status === "opposed")) {
    context.addIssue({
      code: "custom",
      path: ["evidence", "contact", "opposing_pair"],
      message: "Opposing pair must exist exactly for opposed contact evidence"
    });
  }
});

export type HumanoidGraspAssessment = z.infer<typeof HumanoidGraspAssessmentSchema>;

interface ContactAssessment {
  status: HumanoidGraspAssessment["evidence"]["contact"]["status"];
  observed_contact_count: number;
  force_qualified_contact_count: number;
  distinct_force_qualified_links: G1HandContactSurfaceName[];
  distinct_normal_qualified_links: G1HandContactSurfaceName[];
  opposing_pair: z.infer<typeof OpposingPairSchema> | null;
}

interface SupportAssessment {
  status: HumanoidGraspAssessment["evidence"]["support"]["status"];
  candidate_contact_count: number;
  force_qualified_contact_count: number;
  upward_contact_count: number;
}

export class HumanoidGraspTracker {
  readonly #contract: HumanoidGraspContract;
  readonly #tracks = new Map<string, GraspTrack>();

  constructor(
    contract: HumanoidGraspContract,
    checkpoint?: HumanoidGraspTrackerCheckpoint
  ) {
    this.#contract = HumanoidGraspContractSchema.parse(contract);
    if (!checkpoint) return;
    const restored = HumanoidGraspTrackerCheckpointSchema.parse(checkpoint);
    if (JSON.stringify(restored.contract) !== JSON.stringify(this.#contract)) {
      throw new Error("Humanoid grasp checkpoint contract does not match the active contract");
    }
    for (const track of restored.tracks) {
      this.#tracks.set(trackKey(track.object_id, track.hand), structuredClone(track));
    }
  }

  observe(rawObservation: HumanoidGraspObservation): HumanoidGraspAssessment {
    const observation = HumanoidGraspObservationSchema.parse(rawObservation);
    const key = trackKey(observation.object_id, observation.hand);
    let track = this.#tracks.get(key);
    const previousAttempt = track?.attempt ?? null;
    const previousAttemptVerified = previousAttempt !== null
      && previousAttempt.stable_frames
        >= this.#contract.minimum_relative_pose_stable_frames
      && previousAttempt.lifted_hold_frames
        >= this.#contract.minimum_lifted_hold_frames;
    const discontinuous = track !== undefined && observation.frame !== track.last_frame + 1;
    if (!track) {
      track = {
        object_id: observation.object_id,
        hand: observation.hand,
        last_frame: observation.frame,
        support_baseline: null,
        attempt: null
      };
      this.#tracks.set(key, track);
    }

    let resetReason: HumanoidGraspAssessment["reset_reason"] = discontinuous
      ? "frame_discontinuity"
      : null;
    if (discontinuous) track.attempt = null;

    const contact = assessContacts(observation, this.#contract);
    const support = assessSupport(observation, this.#contract);
    const currentProjection = dot(observation.object_pose.position, this.#contract.world_up);
    if (support.status === "supported" && !previousAttemptVerified) {
      track.support_baseline = {
        frame: observation.frame,
        projection_m: currentProjection
      };
    }

    const invalidContactReset = contactResetReason(contact.status);
    if (contact.status !== "opposed") {
      if (track.attempt && resetReason === null) resetReason = invalidContactReset;
      track.attempt = null;
    } else if (support.status === "insufficient_normal") {
      if (track.attempt && resetReason === null) {
        resetReason = "support_evidence_insufficient";
      }
      track.attempt = null;
    } else if (!discontinuous && support.status === "supported"
      && previousAttempt?.has_left_support && !previousAttemptVerified) {
      resetReason = "support_returned";
      track.attempt = null;
    }

    const currentRelativePose = relativePose(
      observation.hand_anchor_pose,
      observation.object_pose
    );
    let translationDrift: number | null = null;
    let rotationDrift: number | null = null;
    if (contact.status === "opposed" && support.status !== "insufficient_normal") {
      if (!track.attempt) {
        track.attempt = newAttempt(observation.frame, currentRelativePose);
      } else {
        translationDrift = vectorLength(subtract(
          currentRelativePose.translation,
          track.attempt.reference_relative_pose.translation
        ));
        rotationDrift = quaternionAngularDistance(
          currentRelativePose.rotation,
          track.attempt.reference_relative_pose.rotation
        );
        if (translationDrift > this.#contract.maximum_relative_translation_drift_m
          || rotationDrift > this.#contract.maximum_relative_rotation_drift_rad) {
          resetReason ??= "relative_pose_unstable";
          track.attempt = newAttempt(observation.frame, currentRelativePose);
        } else {
          track.attempt.stable_frames += 1;
        }
      }
    }

    const attempt = track.attempt;
    const baselineProjection = track.support_baseline?.projection_m ?? null;
    const lift = baselineProjection === null
      ? null
      : currentProjection - baselineProjection;
    if (attempt) {
      if (support.status === "unsupported") attempt.has_left_support = true;
      const stable = attempt.stable_frames
        >= this.#contract.minimum_relative_pose_stable_frames;
      const lifted = attempt.has_left_support && lift !== null
        && lift >= this.#contract.minimum_lift_m;
      if (attempt.lifted_hold_frames
        < this.#contract.minimum_lifted_hold_frames) {
        if (stable && lifted) attempt.lifted_hold_frames += 1;
        else attempt.lifted_hold_frames = 0;
      }
    }

    track.last_frame = observation.frame;
    const verified = attempt !== null
      && attempt.stable_frames >= this.#contract.minimum_relative_pose_stable_frames
      && attempt.lifted_hold_frames >= this.#contract.minimum_lifted_hold_frames;
    if (verified && !previousAttemptVerified) {
      attempt.reference_relative_pose = structuredClone(currentRelativePose);
    }
    const phase = assessmentPhase(attempt, support, lift, verified, this.#contract);
    const reason = assessmentReason({
      contact,
      support,
      attempt,
      lift,
      verified,
      resetReason,
      contract: this.#contract
    });
    return HumanoidGraspAssessmentSchema.parse({
      protocol: "humanoid-grasp-assessment-v1",
      frame: observation.frame,
      object_id: observation.object_id,
      hand: observation.hand,
      phase,
      grasp_verified: verified,
      reason,
      reset_reason: resetReason,
      evidence: {
        contact,
        support: {
          ...support,
          baseline_projection_m: baselineProjection,
          current_projection_m: currentProjection,
          lift_m: lift
        },
        relative_pose: {
          stable_frames: attempt?.stable_frames ?? 0,
          translation_drift_m: translationDrift,
          rotation_drift_rad: rotationDrift
        },
        lifted_hold_frames: attempt?.lifted_hold_frames ?? 0
      }
    });
  }

  checkpoint(): HumanoidGraspTrackerCheckpoint {
    return HumanoidGraspTrackerCheckpointSchema.parse({
      protocol: "humanoid-grasp-tracker-checkpoint-v1",
      contract: this.#contract,
      tracks: [...this.#tracks.values()]
        .sort((left, right) => trackKey(left.object_id, left.hand)
          .localeCompare(trackKey(right.object_id, right.hand)))
        .map((track) => structuredClone(track))
    });
  }
}

export function humanoidGraspObservation(input: {
  frame: number;
  objectId: string;
  hand: HumanoidGraspHand;
  snapshot: HumanoidSimulationSnapshot;
}): HumanoidGraspObservation {
  const object = input.snapshot.objects[input.objectId];
  if (!object) {
    throw new Error(`Humanoid grasp observation is missing object: ${input.objectId}`);
  }
  const wrist = input.snapshot.links[
    input.hand === "left" ? "left_wrist_yaw_link" : "right_wrist_yaw_link"
  ];
  const handContacts = g1HandObjectContacts(input.snapshot.contacts, input.objectId)
    .filter((contact) => contact.hand === input.hand)
    .map((contact) => ({
      hand_link: contact.handLink,
      position: contact.position,
      normal_from_hand: contact.normalFromHand,
      normal_force_n: contact.normalForce
    }));
  return HumanoidGraspObservationSchema.parse({
    protocol: "humanoid-grasp-observation-v1",
    frame: input.frame,
    object_id: input.objectId,
    hand: input.hand,
    object_pose: {
      position: object.position,
      rotation: normalizeQuaternion(object.rotation)
    },
    hand_anchor_pose: {
      position: wrist.position,
      rotation: normalizeQuaternion(wrist.rotation)
    },
    hand_contacts: handContacts,
    support_contacts: objectSupportContacts(input.snapshot.contacts, input.objectId)
  });
}

function assessContacts(
  observation: HumanoidGraspObservation,
  contract: HumanoidGraspContract
): ContactAssessment {
  const manifolds = aggregateHandContactManifolds(observation.hand_contacts);
  const forceQualified = manifolds
    .filter((contact) => contact.normal_force_n >= contract.minimum_contact_normal_force_n)
    .sort(compareHandContacts);
  const forceLinks = sortedLinks(forceQualified.map((contact) => contact.hand_link));
  const normalQualified = forceQualified.filter((contact) => (
    contact.normal_from_hand !== null
      && contact.directed_normal_force_n >= contract.minimum_contact_normal_force_n
  ));
  const normalLinks = sortedLinks(normalQualified.map((contact) => contact.hand_link));
  const base = {
    observed_contact_count: observation.hand_contacts.length,
    force_qualified_contact_count: forceQualified.length,
    distinct_force_qualified_links: forceLinks,
    distinct_normal_qualified_links: normalLinks,
    opposing_pair: null
  };
  if (observation.hand_contacts.length === 0) return { ...base, status: "missing" };
  if (forceLinks.length < contract.minimum_distinct_contact_links) {
    return { ...base, status: "insufficient_links" };
  }
  if (normalLinks.length < contract.minimum_distinct_contact_links) {
    return { ...base, status: "insufficient_normal" };
  }

  let hasUsableGeometry = false;
  for (let firstIndex = 0; firstIndex < normalQualified.length; firstIndex += 1) {
    const first = normalQualified[firstIndex]!;
    for (let secondIndex = firstIndex + 1; secondIndex < normalQualified.length;
      secondIndex += 1) {
      const second = normalQualified[secondIndex]!;
      if (first.hand_link === second.hand_link) continue;
      const firstRadial = subtract(first.position, observation.object_pose.position);
      const secondRadial = subtract(second.position, observation.object_pose.position);
      const firstLength = vectorLength(firstRadial);
      const secondLength = vectorLength(secondRadial);
      if (firstLength < contract.minimum_contact_radial_distance_m
        || secondLength < contract.minimum_contact_radial_distance_m) continue;
      hasUsableGeometry = true;
      const separation = vectorLength(subtract(first.position, second.position));
      const normalDot = dot(first.normal_from_hand!, second.normal_from_hand!);
      const positionDot = dot(firstRadial, secondRadial) / (firstLength * secondLength);
      if (separation < contract.minimum_opposing_contact_separation_m
        || normalDot > contract.maximum_opposing_normal_dot
        || positionDot > contract.maximum_opposing_position_dot) continue;
      return {
        ...base,
        status: "opposed",
        opposing_pair: {
          first_link: first.hand_link,
          second_link: second.hand_link,
          first_position: first.position,
          second_position: second.position,
          first_normal_from_hand: first.normal_from_hand!,
          second_normal_from_hand: second.normal_from_hand!,
          first_normal_force_n: first.normal_force_n,
          second_normal_force_n: second.normal_force_n,
          separation_m: separation,
          normal_dot: clampUnit(normalDot),
          position_dot: clampUnit(positionDot)
        }
      };
    }
  }
  return { ...base, status: hasUsableGeometry ? "not_opposed" : "insufficient_geometry" };
}

type HandContactManifold = HumanoidGraspObservation["hand_contacts"][number] & {
  directed_normal_force_n: number;
};

function aggregateHandContactManifolds(
  contacts: HumanoidGraspObservation["hand_contacts"]
): HandContactManifold[] {
  const grouped = new Map<
    G1HandContactSurfaceName,
    HumanoidGraspObservation["hand_contacts"]
  >();
  for (const contact of contacts) {
    const group = grouped.get(contact.hand_link) ?? [];
    group.push(contact);
    grouped.set(contact.hand_link, group);
  }
  return [...grouped.entries()].map(([handLink, group]) => {
    const normalForce = group.reduce(
      (total, contact) => total + contact.normal_force_n,
      0
    );
    const positionWeight = group.reduce(
      (total, contact) => total + Math.sqrt(Math.max(0, contact.normal_force_n)),
      0
    );
    const position = group.reduce((sum, contact) => {
      const weight = positionWeight > 1e-12
        ? Math.sqrt(Math.max(0, contact.normal_force_n))
        : 1;
      return {
        x: sum.x + contact.position.x * weight,
        y: sum.y + contact.position.y * weight,
        z: sum.z + contact.position.z * weight
      };
    }, { x: 0, y: 0, z: 0 });
    const directed = group.filter((contact) => contact.normal_from_hand !== null);
    const directedNormalForce = directed.reduce(
      (total, contact) => total + contact.normal_force_n,
      0
    );
    const normalSum = directed.reduce((sum, contact) => ({
      x: sum.x + contact.normal_from_hand!.x
        * Math.sqrt(Math.max(0, contact.normal_force_n)),
      y: sum.y + contact.normal_from_hand!.y
        * Math.sqrt(Math.max(0, contact.normal_force_n)),
      z: sum.z + contact.normal_from_hand!.z
        * Math.sqrt(Math.max(0, contact.normal_force_n))
    }), { x: 0, y: 0, z: 0 });
    const normalMagnitude = vectorLength(normalSum);
    return {
      hand_link: handLink,
      position: {
        x: position.x / (positionWeight > 1e-12 ? positionWeight : group.length),
        y: position.y / (positionWeight > 1e-12 ? positionWeight : group.length),
        z: position.z / (positionWeight > 1e-12 ? positionWeight : group.length)
      },
      normal_from_hand: directedNormalForce > 0 && normalMagnitude > 1e-12
        ? {
            x: normalSum.x / normalMagnitude,
            y: normalSum.y / normalMagnitude,
            z: normalSum.z / normalMagnitude
          }
        : null,
      normal_force_n: normalForce,
      directed_normal_force_n: directedNormalForce
    };
  });
}

function assessSupport(
  observation: HumanoidGraspObservation,
  contract: HumanoidGraspContract
): SupportAssessment {
  const upward = observation.support_contacts.filter(
    (contact) => contact.normal_toward_object !== null
    && dot(contact.normal_toward_object, contract.world_up)
      >= contract.minimum_support_up_dot
  );
  const unknownNormal = observation.support_contacts.filter(
    (contact) => contact.normal_toward_object === null
  );
  const upwardNormalForce = upward.reduce(
    (total, contact) => total + contact.normal_force_n,
    0
  );
  const unknownNormalForce = unknownNormal.reduce(
    (total, contact) => total + contact.normal_force_n,
    0
  );
  const supported = upwardNormalForce >= contract.minimum_support_normal_force_n;
  const insufficientNormal = !supported
    && upwardNormalForce + unknownNormalForce
      >= contract.minimum_support_normal_force_n;
  return {
    status: supported
      ? "supported"
      : insufficientNormal ? "insufficient_normal" : "unsupported",
    candidate_contact_count: observation.support_contacts.length,
    force_qualified_contact_count: supported
      ? upward.length
      : insufficientNormal ? upward.length + unknownNormal.length : 0,
    upward_contact_count: upward.length
  };
}

function objectSupportContacts(
  contacts: readonly HumanoidContactSnapshot[],
  objectId: string
): z.infer<typeof SupportContactSchema>[] {
  const support: z.infer<typeof SupportContactSchema>[] = [];
  for (const contact of contacts) {
    const objectIsFirst = contact.firstObject === objectId;
    const objectIsSecond = contact.secondObject === objectId;
    if (objectIsFirst === objectIsSecond) continue;
    const counterpartHand = objectIsFirst ? contact.secondHandLink : contact.firstHandLink;
    if (counterpartHand !== null) continue;
    const counterpartObject = objectIsFirst ? contact.secondObject : contact.firstObject;
    const counterpartBody = objectIsFirst ? contact.secondBody : contact.firstBody;
    if (counterpartObject === objectId) continue;
    support.push({
      position: { ...contact.position },
      normal_toward_object: orientedContactNormal(contact, objectIsSecond),
      normal_force_n: contact.normalForce,
      counterpart_kind: counterpartObject !== null
        ? "object"
        : counterpartBody !== null ? "humanoid" : "environment",
      counterpart_id: counterpartObject ?? counterpartBody
    });
  }
  return support;
}

function orientedContactNormal(
  contact: HumanoidContactSnapshot,
  useRawDirection: boolean
): Vec3 | null {
  const normal = (contact as HumanoidContactSnapshot & { normal?: Vec3 }).normal;
  if (!normal || ![normal.x, normal.y, normal.z].every(Number.isFinite)) return null;
  const magnitude = vectorLength(normal);
  if (magnitude === 0) return null;
  const direction = useRawDirection ? 1 : -1;
  return {
    x: normalizedComponent(direction * normal.x / magnitude),
    y: normalizedComponent(direction * normal.y / magnitude),
    z: normalizedComponent(direction * normal.z / magnitude)
  };
}

function relativePose(
  hand: HumanoidGraspObservation["hand_anchor_pose"],
  object: HumanoidGraspObservation["object_pose"]
): z.infer<typeof RelativePoseSchema> {
  const inverseHand = inverseQuaternion(hand.rotation);
  return RelativePoseSchema.parse({
    translation: rotateVector(inverseHand, subtract(object.position, hand.position)),
    rotation: normalizeQuaternion(multiplyQuaternion(inverseHand, object.rotation))
  });
}

function newAttempt(
  frame: number,
  pose: z.infer<typeof RelativePoseSchema>
): GraspAttempt {
  return {
    first_stable_frame: frame,
    stable_frames: 1,
    lifted_hold_frames: 0,
    has_left_support: false,
    reference_relative_pose: structuredClone(pose)
  };
}

function contactResetReason(
  status: ContactAssessment["status"]
): HumanoidGraspAssessment["reset_reason"] {
  switch (status) {
    case "missing": return "contact_lost";
    case "insufficient_links": return "contact_links_insufficient";
    case "insufficient_normal": return "contact_normal_insufficient";
    case "insufficient_geometry": return "contact_geometry_insufficient";
    case "not_opposed": return "contacts_not_opposed";
    case "opposed": return null;
  }
}

function assessmentPhase(
  attempt: GraspAttempt | null,
  support: SupportAssessment,
  lift: number | null,
  verified: boolean,
  contract: HumanoidGraspContract
): HumanoidGraspAssessment["phase"] {
  if (verified) return "verified";
  if (!attempt) return "idle";
  if (attempt.stable_frames < contract.minimum_relative_pose_stable_frames) {
    return "stabilizing";
  }
  if (support.status === "supported" || lift === null || lift < contract.minimum_lift_m) {
    return "lifting";
  }
  return "holding";
}

function assessmentReason(input: {
  contact: ContactAssessment;
  support: SupportAssessment;
  attempt: GraspAttempt | null;
  lift: number | null;
  verified: boolean;
  resetReason: HumanoidGraspAssessment["reset_reason"];
  contract: HumanoidGraspContract;
}): HumanoidGraspAssessment["reason"] {
  if (input.verified) return "grasp_verified";
  if (input.resetReason === "frame_discontinuity") return "frame_discontinuity";
  if (input.resetReason === "relative_pose_unstable") return "relative_pose_unstable";
  if (input.resetReason === "support_returned") return "support_returned";
  switch (input.contact.status) {
    case "missing": return "contact_missing";
    case "insufficient_links": return "contact_links_insufficient";
    case "insufficient_normal": return "contact_normal_insufficient";
    case "insufficient_geometry": return "contact_geometry_insufficient";
    case "not_opposed": return "contacts_not_opposed";
    case "opposed": break;
  }
  if (input.support.status === "insufficient_normal") {
    return "support_evidence_insufficient";
  }
  if (!input.attempt
    || input.attempt.stable_frames < input.contract.minimum_relative_pose_stable_frames) {
    return "relative_pose_stabilizing";
  }
  if (input.support.status === "supported") return "object_supported";
  if (input.lift === null) return "support_baseline_missing";
  if (input.lift < input.contract.minimum_lift_m) return "lift_height_insufficient";
  return "lift_hold_incomplete";
}

function compareHandContacts(
  left: HumanoidGraspObservation["hand_contacts"][number],
  right: HumanoidGraspObservation["hand_contacts"][number]
): number {
  if (left.hand_link !== right.hand_link) return left.hand_link < right.hand_link ? -1 : 1;
  if (left.normal_force_n !== right.normal_force_n) {
    return right.normal_force_n - left.normal_force_n;
  }
  for (const axis of ["x", "y", "z"] as const) {
    if (left.position[axis] !== right.position[axis]) {
      return left.position[axis] - right.position[axis];
    }
  }
  return 0;
}

function sortedLinks(
  links: readonly G1HandContactSurfaceName[]
): G1HandContactSurfaceName[] {
  return [...new Set(links)].sort((left, right) => left < right ? -1 : left > right ? 1 : 0);
}

function trackKey(objectId: string, hand: HumanoidGraspHand): string {
  return JSON.stringify([objectId, hand]);
}

function dot(left: Vec3, right: Vec3): number {
  return left.x * right.x + left.y * right.y + left.z * right.z;
}

function clampUnit(value: number): number {
  return Math.max(-1, Math.min(1, value));
}

function normalizedComponent(value: number): number {
  return Object.is(value, -0) ? 0 : value;
}
