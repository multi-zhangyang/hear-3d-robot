import { z } from "zod";
import { Vec3Schema, type Vec3 } from "../../domain/schema.js";
import { inverseQuaternion, rotateVector } from "../geometry.js";
import {
  blockedHumanoidContacts,
  type HumanoidEnvironmentContact
} from "./motion-contact-policy.js";
import {
  HUMANOID_BODY_NAMES,
  type HumanoidBodyName
} from "./model.js";
import type { HumanoidContactConstraint } from "./motion-plan.js";
import {
  G1_HAND_CONTACT_SURFACE_NAMES,
  type G1HandContactSurfaceName
} from "./morphology.js";
import type {
  HumanoidContactSnapshot,
  HumanoidSimulationSnapshot
} from "./simulation.js";

const NavigationCollisionSurfaceSchema = z.discriminatedUnion("kind", [
  z.object({
    kind: z.literal("body"),
    name: z.enum(HUMANOID_BODY_NAMES)
  }).strict(),
  z.object({
    kind: z.literal("hand_surface"),
    name: z.enum(G1_HAND_CONTACT_SURFACE_NAMES)
  }).strict()
]);

const NavigationCollisionTargetSchema = z.discriminatedUnion("kind", [
  z.object({
    kind: z.literal("object"),
    id: z.string().trim().min(1)
  }).strict(),
  z.object({
    kind: z.literal("solid"),
    id: z.string().trim().min(1)
  }).strict(),
  z.object({
    kind: z.literal("environment"),
    id: z.null()
  }).strict()
]);

export const HumanoidNavigationCollisionEvidenceSchema = z.object({
  surface: NavigationCollisionSurfaceSchema,
  target: NavigationCollisionTargetSchema,
  contact_point_world: Vec3Schema,
  separation_normal_world: Vec3Schema,
  separation_normal_robot: Vec3Schema,
  normal_force_n: z.number().finite().nonnegative(),
  simulated_time: z.number().finite().nonnegative()
}).strict();

export type HumanoidNavigationCollisionEvidence = z.infer<
  typeof HumanoidNavigationCollisionEvidenceSchema
>;

interface RawNavigationCollision {
  surface:
    | { kind: "body"; name: HumanoidBodyName }
    | { kind: "hand_surface"; name: G1HandContactSurfaceName };
  objectId: string | null;
  solidId: string | null;
  contact: HumanoidContactSnapshot;
  humanoidIsFirst: boolean;
}

export function humanoidNavigationCollisionEvidence(
  snapshot: HumanoidSimulationSnapshot,
  constraints: readonly HumanoidContactConstraint[]
): HumanoidNavigationCollisionEvidence[] {
  const blocked = blockedHumanoidContacts(snapshot, constraints);
  if (blocked.length === 0) return [];
  const inverseRoot = inverseQuaternion(snapshot.rootRotation);
  return rawNavigationCollisions(snapshot.contacts)
    .filter((candidate) => blocked.some((contact) => (
      collisionMatchesBlockedContact(candidate, contact)
    )))
    .map((candidate) => {
      const normal = unitNormal(candidate.contact.normal);
      const direction = candidate.humanoidIsFirst ? -1 : 1;
      const separationNormalWorld = scaleNormal(normal, direction);
      return HumanoidNavigationCollisionEvidenceSchema.parse({
        surface: candidate.surface,
        target: collisionTarget(candidate.objectId, candidate.solidId),
        contact_point_world: { ...candidate.contact.position },
        separation_normal_world: separationNormalWorld,
        separation_normal_robot: rotateVector(
          inverseRoot,
          separationNormalWorld
        ),
        normal_force_n: candidate.contact.normalForce,
        simulated_time: snapshot.simulatedTime
      });
    })
    .sort((left, right) => right.normal_force_n - left.normal_force_n);
}

function rawNavigationCollisions(
  contacts: readonly HumanoidContactSnapshot[]
): RawNavigationCollision[] {
  return contacts.flatMap((contact) => {
    const handContacts: RawNavigationCollision[] = [
      ...(contact.firstHandLink && externalToHand(
        contact.secondBody,
        contact.secondHandLink
      )
        ? [{
            surface: {
              kind: "hand_surface" as const,
              name: contact.firstHandLink
            },
            objectId: contact.secondObject,
            solidId: contact.secondSolid ?? null,
            contact,
            humanoidIsFirst: true
          }]
        : []),
      ...(contact.secondHandLink && externalToHand(
        contact.firstBody,
        contact.firstHandLink
      )
        ? [{
            surface: {
              kind: "hand_surface" as const,
              name: contact.secondHandLink
            },
            objectId: contact.firstObject,
            solidId: contact.firstSolid ?? null,
            contact,
            humanoidIsFirst: false
          }]
        : [])
    ];
    if (handContacts.length > 0) return handContacts;
    if (contact.firstHandLink || contact.secondHandLink) return [];
    if ((contact.firstBody === null) === (contact.secondBody === null)) return [];
    const humanoidIsFirst = contact.firstBody !== null;
    const body = contact.firstBody ?? contact.secondBody;
    if (!body) return [];
    return [{
      surface: { kind: "body", name: body },
      objectId: humanoidIsFirst ? contact.secondObject : contact.firstObject,
      solidId: humanoidIsFirst
        ? contact.secondSolid ?? null
        : contact.firstSolid ?? null,
      contact,
      humanoidIsFirst
    }];
  });
}

function collisionMatchesBlockedContact(
  candidate: RawNavigationCollision,
  blocked: HumanoidEnvironmentContact
): boolean {
  const surfaceMatches = candidate.surface.kind === "body"
    ? "body" in blocked && blocked.body === candidate.surface.name
    : "handSurface" in blocked
      && blocked.handSurface === candidate.surface.name;
  return surfaceMatches
    && blocked.objectId === candidate.objectId
    && blocked.solidId === candidate.solidId;
}

function collisionTarget(
  objectId: string | null,
  solidId: string | null
): HumanoidNavigationCollisionEvidence["target"] {
  if (objectId !== null) return { kind: "object", id: objectId };
  if (solidId !== null) return { kind: "solid", id: solidId };
  return { kind: "environment", id: null };
}

function externalToHand(
  counterpartBody: HumanoidBodyName | null,
  counterpartHandSurface: G1HandContactSurfaceName | null
): boolean {
  return counterpartBody === null && counterpartHandSurface === null;
}

function unitNormal(normal: Vec3): Vec3 {
  const magnitude = Math.hypot(normal.x, normal.y, normal.z);
  if (!Number.isFinite(magnitude) || magnitude <= 1e-12) {
    throw new Error("Navigation collision contact has no finite normal");
  }
  return {
    x: normalizedComponent(normal.x / magnitude),
    y: normalizedComponent(normal.y / magnitude),
    z: normalizedComponent(normal.z / magnitude)
  };
}

function scaleNormal(normal: Vec3, scale: 1 | -1): Vec3 {
  return {
    x: normalizedComponent(normal.x * scale),
    y: normalizedComponent(normal.y * scale),
    z: normalizedComponent(normal.z * scale)
  };
}

function normalizedComponent(value: number): number {
  return Object.is(value, -0) ? 0 : value;
}
