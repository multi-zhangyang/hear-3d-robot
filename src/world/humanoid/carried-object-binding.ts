import { createHash } from "node:crypto";
import { z } from "zod";
import { Vec3Schema } from "../../domain/schema.js";
import {
  HumanoidGraspRegistryCheckpointSchema,
  type HumanoidGraspRegistry,
  type HumanoidGraspRegistryCheckpoint
} from "./grasp-registry.js";
import {
  HumanoidGraspAssessmentSchema,
  type HumanoidGraspAssessment,
  type HumanoidGraspContract,
  type HumanoidGraspHand
} from "./grasp-tracker.js";
import {
  G1_HAND_CONTACT_SURFACE_NAMES,
  g1HandContactSurfaceHand,
  type G1HandContactSurfaceName
} from "./morphology.js";
import { HUMANOID_BODY_NAMES } from "./model.js";
import type { HumanoidContactSnapshot } from "./simulation.js";

const Sha256Schema = z.string().regex(/^[a-f0-9]{64}$/);
const FrameSchema = z.number().int().nonnegative();
const WorldRevisionSchema = z.number().int().nonnegative();
const HandSchema = z.enum(["left", "right"]);
const HandSurfaceSchema = z.enum(G1_HAND_CONTACT_SURFACE_NAMES);

const HumanoidCarriedObjectBindingRequestSchema = z.object({
  object_id: z.string().trim().min(1),
  hand: HandSchema
}).strict();

export type HumanoidCarriedObjectBindingRequest = z.infer<
  typeof HumanoidCarriedObjectBindingRequestSchema
>;

export const HumanoidCarriedObjectBindingSchema = z.object({
  protocol: z.literal("humanoid-carried-object-binding-v1"),
  object_id: z.string().trim().min(1),
  hand: HandSchema,
  grasp_contract_sha256: Sha256Schema,
  grasp_registry_checkpoint_sha256: Sha256Schema,
  grasp_assessment_sha256: Sha256Schema,
  source_frame: FrameSchema,
  source_world_revision: WorldRevisionSchema,
  verified_contact_surfaces: z.array(HandSurfaceSchema).min(2),
  allowed_hand_surfaces: z.array(HandSurfaceSchema).min(2)
}).strict().superRefine((binding, context) => {
  validateSortedUniqueSurfaces(
    binding.verified_contact_surfaces,
    ["verified_contact_surfaces"],
    context
  );
  validateSortedUniqueSurfaces(binding.allowed_hand_surfaces, ["allowed_hand_surfaces"], context);
  binding.verified_contact_surfaces.forEach((surface, index) => {
    if (g1HandContactSurfaceHand(surface) === binding.hand) return;
    context.addIssue({
      code: "custom",
      path: ["verified_contact_surfaces", index],
      message: "Verified carried-object contact surface belongs to the opposite hand"
    });
  });
  const expectedAllowed = allHandContactSurfaces(binding.hand);
  binding.allowed_hand_surfaces.forEach((surface, index) => {
    if (g1HandContactSurfaceHand(surface) === binding.hand) return;
    context.addIssue({
      code: "custom",
      path: ["allowed_hand_surfaces", index],
      message: "Allowed carried-object contact surface belongs to the opposite hand"
    });
  });
  if (JSON.stringify(binding.allowed_hand_surfaces) !== JSON.stringify(expectedAllowed)) {
    context.addIssue({
      code: "custom",
      path: ["allowed_hand_surfaces"],
      message: "Carried-object contact authority must cover exactly one hand"
    });
  }
  if (binding.verified_contact_surfaces.some((surface) => (
    !binding.allowed_hand_surfaces.includes(surface)
  ))) {
    context.addIssue({
      code: "custom",
      path: ["verified_contact_surfaces"],
      message: "Verified carried-object contacts must be within allowed hand surfaces"
    });
  }
});

export type HumanoidCarriedObjectBinding = z.infer<
  typeof HumanoidCarriedObjectBindingSchema
>;

export const HumanoidCarriedObjectBindingSetSchema = z.object({
  protocol: z.literal("humanoid-carried-object-binding-set-v1"),
  source_frame: FrameSchema,
  source_world_revision: WorldRevisionSchema,
  grasp_contract_sha256: Sha256Schema,
  grasp_registry_checkpoint_sha256: Sha256Schema,
  bindings: z.array(HumanoidCarriedObjectBindingSchema)
}).strict().superRefine((state, context) => {
  const objectHands = new Map<string, HumanoidGraspHand>();
  const handObjects = new Map<HumanoidGraspHand, string>();
  let previousKey: string | null = null;
  state.bindings.forEach((binding, index) => {
    const key = bindingKey(binding.object_id, binding.hand);
    if (previousKey !== null && compareStrings(previousKey, key) >= 0) {
      context.addIssue({
        code: "custom",
        path: ["bindings", index],
        message: "Carried-object bindings must be unique and deterministically sorted"
      });
    }
    previousKey = key;
    if (binding.source_frame !== state.source_frame
      || binding.source_world_revision !== state.source_world_revision) {
      context.addIssue({
        code: "custom",
        path: ["bindings", index],
        message: "Carried-object binding source does not match its binding set"
      });
    }
    if (binding.grasp_contract_sha256 !== state.grasp_contract_sha256
      || binding.grasp_registry_checkpoint_sha256
        !== state.grasp_registry_checkpoint_sha256) {
      context.addIssue({
        code: "custom",
        path: ["bindings", index],
        message: "Carried-object binding authority does not match its binding set"
      });
    }

    const existingHand = objectHands.get(binding.object_id);
    if (existingHand !== undefined) {
      context.addIssue({
        code: "custom",
        path: ["bindings", index, "object_id"],
        message: existingHand === binding.hand
          ? `Duplicate carried-object binding: ${binding.object_id}`
          : `Carried object is bound to multiple hands: ${binding.object_id}`
      });
    } else {
      objectHands.set(binding.object_id, binding.hand);
    }
    const existingObject = handObjects.get(binding.hand);
    if (existingObject !== undefined) {
      context.addIssue({
        code: "custom",
        path: ["bindings", index, "hand"],
        message: `Hand is bound to multiple carried objects: ${binding.hand}`
      });
    } else {
      handObjects.set(binding.hand, binding.object_id);
    }
  });
});

export type HumanoidCarriedObjectBindingSet = z.infer<
  typeof HumanoidCarriedObjectBindingSetSchema
>;

const HumanoidCarriedObjectContactConstraintSchema = z.object({
  hand_surface: HandSurfaceSchema,
  object_id: z.string().trim().min(1),
  required: z.literal(false)
}).strict();

export type HumanoidCarriedObjectContactConstraint = z.infer<
  typeof HumanoidCarriedObjectContactConstraintSchema
>;

export const HumanoidCarriedObjectUnauthorizedContactSchema = z.object({
  binding_sha256: Sha256Schema,
  object_id: z.string().trim().min(1),
  contact_index: z.number().int().nonnegative(),
  object_contact_side: z.enum(["first", "second"]),
  counterpart_kind: z.enum(["environment", "object", "humanoid"]),
  counterpart_object_id: z.string().trim().min(1).nullable(),
  counterpart_body: z.enum(HUMANOID_BODY_NAMES).nullable(),
  counterpart_hand_surface: HandSurfaceSchema.nullable(),
  position: Vec3Schema,
  normal_force_n: z.number().finite().nonnegative()
}).strict().superRefine((contact, context) => {
  const hasHumanoid = contact.counterpart_body !== null
    || contact.counterpart_hand_surface !== null;
  const valid = contact.counterpart_kind === "environment"
    ? contact.counterpart_object_id === null && !hasHumanoid
    : contact.counterpart_kind === "object"
      ? contact.counterpart_object_id !== null && !hasHumanoid
      : contact.counterpart_object_id === null && hasHumanoid;
  if (!valid) {
    context.addIssue({
      code: "custom",
      message: "Carried-object collision counterpart identity is inconsistent"
    });
  }
});

export type HumanoidCarriedObjectUnauthorizedContact = z.infer<
  typeof HumanoidCarriedObjectUnauthorizedContactSchema
>;

const HumanoidCarriedObjectContinuationBindingSchema = z.object({
  binding_sha256: Sha256Schema,
  object_id: z.string().trim().min(1),
  hand: HandSchema,
  continued: z.boolean(),
  failure: z.enum([
    "registry_not_current",
    "grasp_contract_changed",
    "current_assessment_missing",
    "grasp_not_verified",
    "grasp_evidence_invalid"
  ]).nullable(),
  source_assessment_sha256: Sha256Schema,
  current_assessment_sha256: Sha256Schema.nullable(),
  verified_contact_surfaces: z.array(HandSurfaceSchema),
  detail: z.string().min(1).nullable()
}).strict().superRefine((binding, context) => {
  if (binding.continued !== (binding.failure === null)) {
    context.addIssue({
      code: "custom",
      message: "Carried-object continuation status is inconsistent"
    });
  }
  if (binding.continued && binding.current_assessment_sha256 === null) {
    context.addIssue({
      code: "custom",
      path: ["current_assessment_sha256"],
      message: "Continued carrying requires current assessment evidence"
    });
  }
  if (binding.continued !== (binding.detail === null)) {
    context.addIssue({
      code: "custom",
      path: ["detail"],
      message: "Failed carried-object continuation requires a detail"
    });
  }
  validateSortedUniqueSurfaces(
    binding.verified_contact_surfaces,
    ["verified_contact_surfaces"],
    context
  );
  binding.verified_contact_surfaces.forEach((surface, index) => {
    if (g1HandContactSurfaceHand(surface) === binding.hand) return;
    context.addIssue({
      code: "custom",
      path: ["verified_contact_surfaces", index],
      message: "Continuation contact surface belongs to the opposite hand"
    });
  });
});

export const HumanoidCarriedObjectContinuationEvidenceSchema = z.object({
  protocol: z.literal("humanoid-carried-object-continuation-v1"),
  binding_set_sha256: Sha256Schema,
  source_frame: FrameSchema,
  source_world_revision: WorldRevisionSchema,
  observed_frame: FrameSchema,
  observed_world_revision: WorldRevisionSchema,
  grasp_contract_sha256: Sha256Schema,
  continued: z.boolean(),
  bindings: z.array(HumanoidCarriedObjectContinuationBindingSchema)
}).strict().superRefine((evidence, context) => {
  if (evidence.continued !== evidence.bindings.every((binding) => binding.continued)) {
    context.addIssue({
      code: "custom",
      path: ["continued"],
      message: "Carried-object continuation summary does not match its bindings"
    });
  }
});

export type HumanoidCarriedObjectContinuationEvidence = z.infer<
  typeof HumanoidCarriedObjectContinuationEvidenceSchema
>;

export function admitHumanoidCarriedObjectBindings(input: {
  registry: HumanoidGraspRegistry;
  currentFrame: number;
  currentWorldRevision: number;
  requests: readonly HumanoidCarriedObjectBindingRequest[];
}): HumanoidCarriedObjectBindingSet {
  const currentFrame = FrameSchema.parse(input.currentFrame);
  const currentWorldRevision = WorldRevisionSchema.parse(input.currentWorldRevision);
  const requests = input.requests.map((request) => (
    HumanoidCarriedObjectBindingRequestSchema.parse(request)
  ));
  validateBindingRequests(requests);
  const authority = currentGraspAuthority(input.registry, currentFrame);
  const bindings = requests.map((request) => {
    if (!authority.portableObjects.has(request.object_id)) {
      throw new Error(`Carried-object request references unknown object: ${request.object_id}`);
    }
    const assessment = authority.assessments.get(
      bindingKey(request.object_id, request.hand)
    );
    if (!assessment) {
      throw new Error(
        `Carried-object request has no current grasp assessment: ${request.object_id}/${request.hand}`
      );
    }
    const verifiedSurfaces = verifiedAcquisitionContactSurfaces(
      assessment,
      authority.contract
    );
    return HumanoidCarriedObjectBindingSchema.parse({
      protocol: "humanoid-carried-object-binding-v1",
      object_id: request.object_id,
      hand: request.hand,
      grasp_contract_sha256: authority.contractSha256,
      grasp_registry_checkpoint_sha256: authority.registryCheckpointSha256,
      grasp_assessment_sha256: humanoidGraspAssessmentSha256(assessment),
      source_frame: currentFrame,
      source_world_revision: currentWorldRevision,
      verified_contact_surfaces: verifiedSurfaces,
      allowed_hand_surfaces: allHandContactSurfaces(request.hand)
    });
  }).sort(compareBindings);

  const state = HumanoidCarriedObjectBindingSetSchema.parse({
    protocol: "humanoid-carried-object-binding-set-v1",
    source_frame: currentFrame,
    source_world_revision: currentWorldRevision,
    grasp_contract_sha256: authority.contractSha256,
    grasp_registry_checkpoint_sha256: authority.registryCheckpointSha256,
    bindings
  });
  return verifyHumanoidCarriedObjectBindingSet({
    registry: input.registry,
    currentFrame,
    currentWorldRevision,
    state
  });
}

export function verifyHumanoidCarriedObjectBindingSet(input: {
  registry: HumanoidGraspRegistry;
  currentFrame: number;
  currentWorldRevision: number;
  state: HumanoidCarriedObjectBindingSet;
}): HumanoidCarriedObjectBindingSet {
  const currentFrame = FrameSchema.parse(input.currentFrame);
  const currentWorldRevision = WorldRevisionSchema.parse(input.currentWorldRevision);
  const state = HumanoidCarriedObjectBindingSetSchema.parse(input.state);
  if (state.source_frame !== currentFrame
    || state.source_world_revision !== currentWorldRevision) {
    throw new Error("Carried-object binding set is not from the current world frame");
  }
  const authority = currentGraspAuthority(input.registry, currentFrame);
  if (state.grasp_contract_sha256 !== authority.contractSha256
    || state.grasp_registry_checkpoint_sha256
      !== authority.registryCheckpointSha256) {
    throw new Error("Carried-object binding set does not match current grasp authority");
  }
  for (const binding of state.bindings) {
    if (!authority.portableObjects.has(binding.object_id)) {
      throw new Error(`Carried-object binding references unknown object: ${binding.object_id}`);
    }
    const assessment = authority.assessments.get(
      bindingKey(binding.object_id, binding.hand)
    );
    if (!assessment) {
      throw new Error(
        `Carried-object binding has no current grasp assessment: ${binding.object_id}/${binding.hand}`
      );
    }
    const verifiedSurfaces = verifiedAcquisitionContactSurfaces(
      assessment,
      authority.contract
    );
    if (binding.grasp_assessment_sha256 !== humanoidGraspAssessmentSha256(assessment)
      || JSON.stringify(binding.verified_contact_surfaces)
        !== JSON.stringify(verifiedSurfaces)
      || JSON.stringify(binding.allowed_hand_surfaces)
        !== JSON.stringify(allHandContactSurfaces(binding.hand))) {
      throw new Error(
        `Carried-object binding does not match current grasp evidence: ${binding.object_id}/${binding.hand}`
      );
    }
  }
  return structuredClone(state);
}

export function humanoidCarriedObjectContactConstraints(
  rawState: HumanoidCarriedObjectBindingSet
): HumanoidCarriedObjectContactConstraint[] {
  const state = HumanoidCarriedObjectBindingSetSchema.parse(rawState);
  return state.bindings.flatMap((binding) => (
    binding.allowed_hand_surfaces.map((surface) => (
      HumanoidCarriedObjectContactConstraintSchema.parse({
        hand_surface: surface,
        object_id: binding.object_id,
        required: false
      })
    ))
  ));
}

export function humanoidCarriedObjectUnauthorizedContacts(
  rawState: HumanoidCarriedObjectBindingSet,
  contacts: readonly HumanoidContactSnapshot[]
): HumanoidCarriedObjectUnauthorizedContact[] {
  const state = HumanoidCarriedObjectBindingSetSchema.parse(rawState);
  const violations: HumanoidCarriedObjectUnauthorizedContact[] = [];
  for (const binding of state.bindings) {
    const allowed = new Set(binding.allowed_hand_surfaces);
    const bindingSha256 = humanoidCarriedObjectBindingSha256(binding);
    contacts.forEach((contact, contactIndex) => {
      const objectIsFirst = contact.firstObject === binding.object_id;
      const objectIsSecond = contact.secondObject === binding.object_id;
      if (!objectIsFirst && !objectIsSecond) return;
      if (objectIsFirst && objectIsSecond) {
        throw new Error(
          `Carried-object contact has the same object on both sides: ${binding.object_id}`
        );
      }
      const counterpartObject = objectIsFirst ? contact.secondObject : contact.firstObject;
      const counterpartBody = objectIsFirst ? contact.secondBody : contact.firstBody;
      const counterpartSurface = objectIsFirst
        ? contact.secondHandLink
        : contact.firstHandLink;
      if (counterpartObject !== null
        && (counterpartBody !== null || counterpartSurface !== null)) {
        throw new Error(
          `Carried-object contact has ambiguous counterpart identity: ${binding.object_id}`
        );
      }
      if (counterpartObject === null
        && counterpartSurface !== null
        && allowed.has(counterpartSurface)
        && g1HandContactSurfaceHand(counterpartSurface) === binding.hand
        && counterpartBody === expectedSurfaceBody(counterpartSurface)) return;

      const counterpartKind = counterpartObject !== null
        ? "object"
        : counterpartBody !== null || counterpartSurface !== null
          ? "humanoid"
          : "environment";
      violations.push(HumanoidCarriedObjectUnauthorizedContactSchema.parse({
        binding_sha256: bindingSha256,
        object_id: binding.object_id,
        contact_index: contactIndex,
        object_contact_side: objectIsFirst ? "first" : "second",
        counterpart_kind: counterpartKind,
        counterpart_object_id: counterpartObject,
        counterpart_body: counterpartObject === null ? counterpartBody : null,
        counterpart_hand_surface: counterpartObject === null ? counterpartSurface : null,
        position: contact.position,
        normal_force_n: contact.normalForce
      }));
    });
  }
  return violations;
}

export function humanoidCarriedObjectContinuationEvidence(input: {
  state: HumanoidCarriedObjectBindingSet;
  registry: HumanoidGraspRegistry;
  currentFrame: number;
  currentWorldRevision: number;
}): HumanoidCarriedObjectContinuationEvidence {
  const state = HumanoidCarriedObjectBindingSetSchema.parse(input.state);
  const currentFrame = FrameSchema.parse(input.currentFrame);
  const currentWorldRevision = WorldRevisionSchema.parse(input.currentWorldRevision);
  const registryCurrent = input.registry.lastFrame === currentFrame;
  const contractMatches = input.registry.contractSha256 === state.grasp_contract_sha256;
  const contract = input.registry.contract;
  const assessments = new Map(
    input.registry.assessmentsForFrame(currentFrame).map((assessment) => (
      [bindingKey(assessment.object_id, assessment.hand), assessment]
    ))
  );
  const bindings = state.bindings.map((binding) => {
    const base = {
      binding_sha256: humanoidCarriedObjectBindingSha256(binding),
      object_id: binding.object_id,
      hand: binding.hand,
      source_assessment_sha256: binding.grasp_assessment_sha256
    };
    if (!registryCurrent) {
      return continuationFailure(base, "registry_not_current", null,
        "Grasp registry has not observed the requested continuation frame");
    }
    if (!contractMatches) {
      return continuationFailure(base, "grasp_contract_changed", null,
        "Active grasp contract differs from the carried-object binding");
    }
    const assessment = assessments.get(bindingKey(binding.object_id, binding.hand));
    if (!assessment) {
      return continuationFailure(base, "current_assessment_missing", null,
        "Current grasp assessment is missing");
    }
    const assessmentSha256 = humanoidGraspAssessmentSha256(assessment);
    if (!assessment.grasp_verified) {
      return continuationFailure(base, "grasp_not_verified", assessmentSha256,
        `Current grasp assessment frame=${assessment.frame}; `
        + `phase=${assessment.phase}; `
        + `reason=${assessment.reason}; `
        + `contact=${assessment.evidence.contact.status}; `
        + `force_links=${assessment.evidence.contact.distinct_force_qualified_links.join(",")}; `
        + `stable_frames=${assessment.evidence.relative_pose.stable_frames}; `
        + `lifted_hold_frames=${assessment.evidence.lifted_hold_frames}; `
        + `translation_drift_m=${String(
          assessment.evidence.relative_pose.translation_drift_m
        )}; rotation_drift_rad=${String(
          assessment.evidence.relative_pose.rotation_drift_rad
        )}`);
    }
    try {
      const verifiedSurfaces = verifiedContinuationContactSurfaces(
        assessment,
        contract
      );
      return {
        ...base,
        continued: true as const,
        failure: null,
        current_assessment_sha256: assessmentSha256,
        verified_contact_surfaces: verifiedSurfaces,
        detail: null
      };
    } catch (error) {
      return continuationFailure(
        base,
        "grasp_evidence_invalid",
        assessmentSha256,
        error instanceof Error ? error.message : "Current grasp evidence is invalid"
      );
    }
  });
  return HumanoidCarriedObjectContinuationEvidenceSchema.parse({
    protocol: "humanoid-carried-object-continuation-v1",
    binding_set_sha256: humanoidCarriedObjectBindingSetSha256(state),
    source_frame: state.source_frame,
    source_world_revision: state.source_world_revision,
    observed_frame: currentFrame,
    observed_world_revision: currentWorldRevision,
    grasp_contract_sha256: input.registry.contractSha256,
    continued: bindings.every((binding) => binding.continued),
    bindings
  });
}

export function humanoidGraspAssessmentSha256(
  assessment: HumanoidGraspAssessment
): string {
  return sha256(JSON.stringify(HumanoidGraspAssessmentSchema.parse(assessment)));
}

export function humanoidGraspRegistryCheckpointSha256(
  checkpoint: HumanoidGraspRegistryCheckpoint
): string {
  return sha256(JSON.stringify(
    HumanoidGraspRegistryCheckpointSchema.parse(checkpoint)
  ));
}

export function humanoidCarriedObjectBindingSha256(
  binding: HumanoidCarriedObjectBinding
): string {
  return sha256(JSON.stringify(HumanoidCarriedObjectBindingSchema.parse(binding)));
}

export function humanoidCarriedObjectBindingSetSha256(
  state: HumanoidCarriedObjectBindingSet
): string {
  return sha256(JSON.stringify(HumanoidCarriedObjectBindingSetSchema.parse(state)));
}

interface CurrentGraspAuthority {
  contract: HumanoidGraspContract;
  contractSha256: string;
  registryCheckpointSha256: string;
  portableObjects: ReadonlySet<string>;
  assessments: ReadonlyMap<string, HumanoidGraspAssessment>;
}

type ContinuationFailure = NonNullable<
  z.infer<typeof HumanoidCarriedObjectContinuationBindingSchema>["failure"]
>;

function continuationFailure(
  base: {
    binding_sha256: string;
    object_id: string;
    hand: HumanoidGraspHand;
    source_assessment_sha256: string;
  },
  failure: ContinuationFailure,
  currentAssessmentSha256: string | null,
  detail: string
): z.input<typeof HumanoidCarriedObjectContinuationBindingSchema> {
  return {
    ...base,
    continued: false,
    failure,
    current_assessment_sha256: currentAssessmentSha256,
    verified_contact_surfaces: [],
    detail
  };
}

function currentGraspAuthority(
  registry: HumanoidGraspRegistry,
  currentFrame: number
): CurrentGraspAuthority {
  if (registry.lastFrame !== currentFrame) {
    throw new Error(
      `Carried-object binding requires current grasp frame ${currentFrame}; registry is at ${
        registry.lastFrame ?? "none"
      }`
    );
  }
  const checkpoint = registry.checkpoint();
  if (checkpoint.last_frame !== currentFrame) {
    throw new Error("Carried-object binding registry checkpoint is not current");
  }
  return {
    contract: registry.contract,
    contractSha256: registry.contractSha256,
    registryCheckpointSha256: humanoidGraspRegistryCheckpointSha256(checkpoint),
    portableObjects: new Set(registry.portableObjectIds),
    assessments: new Map(registry.assessmentsForFrame(currentFrame).map((assessment) => (
      [bindingKey(assessment.object_id, assessment.hand), assessment]
    )))
  };
}

function validateBindingRequests(
  requests: readonly HumanoidCarriedObjectBindingRequest[]
): void {
  const objectHands = new Map<string, HumanoidGraspHand>();
  const handObjects = new Map<HumanoidGraspHand, string>();
  for (const request of requests) {
    const existingHand = objectHands.get(request.object_id);
    if (existingHand !== undefined) {
      if (existingHand === request.hand) {
        throw new Error(`Duplicate carried-object request: ${request.object_id}`);
      }
      throw new Error(`Carried object requested for multiple hands: ${request.object_id}`);
    }
    const existingObject = handObjects.get(request.hand);
    if (existingObject !== undefined) {
      throw new Error(`Hand requested for multiple carried objects: ${request.hand}`);
    }
    objectHands.set(request.object_id, request.hand);
    handObjects.set(request.hand, request.object_id);
  }
}

function verifiedAcquisitionContactSurfaces(
  rawAssessment: HumanoidGraspAssessment,
  contract: HumanoidGraspContract
): G1HandContactSurfaceName[] {
  const { assessment, forceSurfaces } = verifiedContactEvidence(
    rawAssessment,
    contract
  );
  const support = assessment.evidence.support;
  if (support.status !== "unsupported"
    || support.baseline_projection_m === null
    || support.lift_m === null
    || support.lift_m < contract.minimum_lift_m) {
    throw new Error("Verified grasp acquisition has insufficient lift evidence");
  }
  validateLatchedStabilityEvidence(assessment, contract);
  return forceSurfaces;
}

function verifiedContinuationContactSurfaces(
  rawAssessment: HumanoidGraspAssessment,
  contract: HumanoidGraspContract
): G1HandContactSurfaceName[] {
  const { assessment, forceSurfaces } = verifiedContactEvidence(
    rawAssessment,
    contract
  );
  const support = assessment.evidence.support;
  if (support.status === "insufficient_normal"
    || support.baseline_projection_m === null) {
    throw new Error("Verified grasp continuation has invalid support evidence");
  }
  validateLatchedStabilityEvidence(assessment, contract);
  return forceSurfaces;
}

function verifiedContactEvidence(
  rawAssessment: HumanoidGraspAssessment,
  contract: HumanoidGraspContract
): {
  assessment: HumanoidGraspAssessment;
  forceSurfaces: G1HandContactSurfaceName[];
} {
  const assessment = HumanoidGraspAssessmentSchema.parse(rawAssessment);
  if (!assessment.grasp_verified
    || assessment.phase !== "verified"
    || assessment.reason !== "grasp_verified"
    || assessment.reset_reason !== null) {
    throw new Error(
      `Carried-object binding requires a verified grasp assessment: ${assessment.object_id}/${assessment.hand}`
    );
  }
  const contact = assessment.evidence.contact;
  const pair = contact.opposing_pair;
  if (contact.status !== "opposed" || pair === null) {
    throw new Error("Verified carried-object assessment is missing opposed contact evidence");
  }
  const forceSurfaces = canonicalSurfaces(contact.distinct_force_qualified_links);
  const normalSurfaces = canonicalSurfaces(contact.distinct_normal_qualified_links);
  if (contact.observed_contact_count < contact.force_qualified_contact_count
    || contact.force_qualified_contact_count < forceSurfaces.length
    || normalSurfaces.some((surface) => !forceSurfaces.includes(surface))) {
    throw new Error("Verified carried-object assessment has inconsistent contact counts");
  }
  validateAssessmentSurfaceSet(
    forceSurfaces,
    assessment.hand,
    contract.minimum_distinct_contact_links,
    "force-qualified"
  );
  validateAssessmentSurfaceSet(
    normalSurfaces,
    assessment.hand,
    contract.minimum_distinct_contact_links,
    "normal-qualified"
  );
  if (pair.first_link === pair.second_link
    || !normalSurfaces.includes(pair.first_link)
    || !normalSurfaces.includes(pair.second_link)
    || g1HandContactSurfaceHand(pair.first_link) !== assessment.hand
    || g1HandContactSurfaceHand(pair.second_link) !== assessment.hand
    || pair.first_normal_force_n < contract.minimum_contact_normal_force_n
    || pair.second_normal_force_n < contract.minimum_contact_normal_force_n
    || pair.separation_m < contract.minimum_opposing_contact_separation_m
    || pair.normal_dot > contract.maximum_opposing_normal_dot
    || pair.position_dot > contract.maximum_opposing_position_dot) {
    throw new Error("Verified carried-object assessment has invalid opposing contacts");
  }
  return { assessment, forceSurfaces };
}

function validateLatchedStabilityEvidence(
  assessment: HumanoidGraspAssessment,
  contract: HumanoidGraspContract
): void {
  const relativePose = assessment.evidence.relative_pose;
  if (relativePose.stable_frames < contract.minimum_relative_pose_stable_frames
    || (relativePose.translation_drift_m !== null
      && relativePose.translation_drift_m > contract.maximum_relative_translation_drift_m)
    || (relativePose.rotation_drift_rad !== null
      && relativePose.rotation_drift_rad > contract.maximum_relative_rotation_drift_rad)
    || assessment.evidence.lifted_hold_frames < contract.minimum_lifted_hold_frames) {
    throw new Error("Verified carried-object assessment has insufficient lift or stability evidence");
  }
}

function validateAssessmentSurfaceSet(
  surfaces: readonly G1HandContactSurfaceName[],
  hand: HumanoidGraspHand,
  minimumCount: number,
  label: string
): void {
  if (surfaces.length < minimumCount) {
    throw new Error(`Verified grasp has insufficient ${label} contact surfaces`);
  }
  if (surfaces.some((surface) => g1HandContactSurfaceHand(surface) !== hand)) {
    throw new Error(`Verified grasp ${label} contact surface belongs to the opposite hand`);
  }
}

function canonicalSurfaces(
  surfaces: readonly G1HandContactSurfaceName[]
): G1HandContactSurfaceName[] {
  const canonical = [...new Set(surfaces)].sort(compareStrings);
  if (canonical.length !== surfaces.length) {
    throw new Error("Verified grasp contact surfaces contain duplicates");
  }
  return canonical;
}

function validateSortedUniqueSurfaces(
  surfaces: readonly G1HandContactSurfaceName[],
  path: (string | number)[],
  context: z.RefinementCtx
): void {
  for (let index = 1; index < surfaces.length; index += 1) {
    if (compareStrings(surfaces[index - 1]!, surfaces[index]!) < 0) continue;
    context.addIssue({
      code: "custom",
      path: [...path, index],
      message: "Carried-object contact surfaces must be unique and deterministically sorted"
    });
  }
}

function compareBindings(
  left: HumanoidCarriedObjectBinding,
  right: HumanoidCarriedObjectBinding
): number {
  return compareStrings(
    bindingKey(left.object_id, left.hand),
    bindingKey(right.object_id, right.hand)
  );
}

function bindingKey(objectId: string, hand: HumanoidGraspHand): string {
  return `${objectId}\0${hand}`;
}

function expectedSurfaceBody(
  surface: G1HandContactSurfaceName
): "left_wrist_yaw_link" | "right_wrist_yaw_link" | null {
  if (surface === "left_hand_palm_link") return "left_wrist_yaw_link";
  if (surface === "right_hand_palm_link") return "right_wrist_yaw_link";
  return null;
}

function allHandContactSurfaces(
  hand: HumanoidGraspHand
): G1HandContactSurfaceName[] {
  return G1_HAND_CONTACT_SURFACE_NAMES
    .filter((surface) => g1HandContactSurfaceHand(surface) === hand)
    .sort(compareStrings);
}

function compareStrings(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}
