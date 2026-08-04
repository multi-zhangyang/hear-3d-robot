import { z } from "zod";
import {
  HUMANOID_BODY_NAMES,
  type HumanoidBodyName
} from "./model.js";
import {
  G1_HAND_CONTACT_SURFACE_NAMES,
  type G1HandContactSurfaceName
} from "./morphology.js";
import type {
  HumanoidContactConstraint
} from "./motion-plan-schema.js";
import type { HumanoidSimulationSnapshot } from "./simulation.js";

const HumanoidBodyEnvironmentContactSchema = z.object({
  body: z.enum(HUMANOID_BODY_NAMES),
  objectId: z.string().min(1).nullable(),
  solidId: z.string().min(1).nullable().default(null),
  normalForce: z.number().finite().nonnegative()
}).strict().superRefine(validateSingleContactTarget);

const HumanoidHandSurfaceEnvironmentContactSchema = z.object({
  handSurface: z.enum(G1_HAND_CONTACT_SURFACE_NAMES),
  objectId: z.string().min(1).nullable(),
  solidId: z.string().min(1).nullable().default(null),
  normalForce: z.number().finite().nonnegative()
}).strict().superRefine(validateSingleContactTarget);

export const HumanoidEnvironmentContactSchema = z.union([
  HumanoidBodyEnvironmentContactSchema,
  HumanoidHandSurfaceEnvironmentContactSchema
]);

type HumanoidHandSurfaceEnvironmentContact = z.infer<
  typeof HumanoidHandSurfaceEnvironmentContactSchema
>;
export type HumanoidEnvironmentContact = z.infer<
  typeof HumanoidEnvironmentContactSchema
>;

export function humanoidEnvironmentContacts(
  snapshot: HumanoidSimulationSnapshot
): HumanoidEnvironmentContact[] {
  const contacts = new Map<string, HumanoidEnvironmentContact>();
  for (const contact of snapshot.contacts) {
    const handContacts: HumanoidHandSurfaceEnvironmentContact[] = [
      ...(contact.firstHandLink && externalToHand(
        contact.secondBody,
        contact.secondHandLink
      )
        ? [{
            handSurface: contact.firstHandLink,
            objectId: contact.secondObject,
            solidId: contact.secondSolid ?? null,
            normalForce: contact.normalForce
          }]
        : []),
      ...(contact.secondHandLink && externalToHand(
        contact.firstBody,
        contact.firstHandLink
      )
        ? [{
            handSurface: contact.secondHandLink,
            objectId: contact.firstObject,
            solidId: contact.firstSolid ?? null,
            normalForce: contact.normalForce
          }]
        : [])
    ];
    if (handContacts.length > 0) {
      for (const candidate of handContacts) {
        retainStrongestContact(contacts, candidate);
      }
      continue;
    }
    if (contact.firstHandLink || contact.secondHandLink) continue;
    if ((contact.firstBody === null) === (contact.secondBody === null)) continue;
    const body = contact.firstBody ?? contact.secondBody;
    if (!body) continue;
    const objectId = contact.firstObject ?? contact.secondObject;
    const solidId = contact.firstSolid ?? contact.secondSolid ?? null;
    const foot = body === "left_ankle_roll_link" || body === "right_ankle_roll_link";
    if (foot && objectId === null && solidId === null
      && Math.abs(contact.normal.y) >= 0.55) continue;
    retainStrongestContact(contacts, {
      body,
      objectId,
      solidId,
      normalForce: contact.normalForce
    });
  }
  return [...contacts.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([, contact]) => contact);
}

function externalToHand(
  counterpartBody: HumanoidBodyName | null,
  counterpartHandSurface: G1HandContactSurfaceName | null
): boolean {
  return counterpartBody === null && counterpartHandSurface === null;
}

export function blockedHumanoidContacts(
  snapshot: HumanoidSimulationSnapshot,
  constraints: readonly HumanoidContactConstraint[]
): HumanoidEnvironmentContact[] {
  const allowed = new Set(constraints.map(contactKey));
  return humanoidEnvironmentContacts(snapshot).filter((contact) => (
    contactTargetId(contact) === null
    || !allowed.has(humanoidEnvironmentContactKey(contact))
  ));
}

export function missingRequiredHumanoidContacts(
  constraints: readonly HumanoidContactConstraint[],
  satisfied: ReadonlySet<string>
): HumanoidContactConstraint[] {
  return constraints.filter((constraint) => (
    constraint.required && !satisfied.has(contactKey(constraint))
  ));
}

export function humanoidContactKey(
  body: HumanoidBodyName,
  objectId: string
): string {
  return contactPairKey(body, objectId);
}

export function humanoidHandContactKey(
  handSurface: G1HandContactSurfaceName,
  objectId: string
): string {
  return handSurfaceContactPairKey(handSurface, objectId);
}

export function humanoidSolidContactKey(
  body: HumanoidBodyName,
  solidId: string
): string {
  return solidContactPairKey(body, solidId);
}

export function humanoidHandSolidContactKey(
  handSurface: G1HandContactSurfaceName,
  solidId: string
): string {
  return handSurfaceSolidContactPairKey(handSurface, solidId);
}

export function humanoidContactConstraintKey(
  constraint: HumanoidContactConstraint
): string {
  return contactKey(constraint);
}

export function humanoidEnvironmentContactKey(
  contact: HumanoidEnvironmentContact
): string {
  if (contact.solidId !== null) {
    return "body" in contact
      ? solidContactPairKey(contact.body, contact.solidId)
      : handSurfaceSolidContactPairKey(contact.handSurface, contact.solidId);
  }
  return "body" in contact
    ? contactPairKey(contact.body, contact.objectId)
    : handSurfaceContactPairKey(contact.handSurface, contact.objectId);
}

function contactKey(constraint: HumanoidContactConstraint): string {
  if ("solid_id" in constraint) {
    return "body" in constraint
      ? solidContactPairKey(constraint.body, constraint.solid_id)
      : handSurfaceSolidContactPairKey(
          constraint.hand_surface,
          constraint.solid_id
        );
  }
  return "body" in constraint
    ? contactPairKey(constraint.body, constraint.object_id)
    : handSurfaceContactPairKey(constraint.hand_surface, constraint.object_id);
}

function contactPairKey(body: HumanoidBodyName, objectId: string | null): string {
  return `${body}\u0000${objectId ?? ""}`;
}

function handSurfaceContactPairKey(
  handSurface: G1HandContactSurfaceName,
  objectId: string | null
): string {
  return `hand_surface\u0000${handSurface}\u0000${objectId ?? ""}`;
}

function solidContactPairKey(body: HumanoidBodyName, solidId: string): string {
  return `${body}\u0000solid\u0000${solidId}`;
}

function handSurfaceSolidContactPairKey(
  handSurface: G1HandContactSurfaceName,
  solidId: string
): string {
  return `hand_surface\u0000${handSurface}\u0000solid\u0000${solidId}`;
}

function retainStrongestContact(
  contacts: Map<string, HumanoidEnvironmentContact>,
  candidate: HumanoidEnvironmentContact
): void {
  const key = humanoidEnvironmentContactKey(candidate);
  const previous = contacts.get(key);
  if (!previous || candidate.normalForce > previous.normalForce) {
    contacts.set(key, candidate);
  }
}

export function distinctContactBodies(
  contacts: readonly HumanoidEnvironmentContact[]
): HumanoidBodyName[] {
  return [...new Set(contacts.flatMap((contact) => (
    "body" in contact ? [contact.body] : []
  )))].sort();
}

export function distinctContactHandSurfaces(
  contacts: readonly HumanoidEnvironmentContact[]
): G1HandContactSurfaceName[] {
  return [...new Set(contacts.flatMap((contact) => (
    "handSurface" in contact ? [contact.handSurface] : []
  )))].sort();
}

function contactTargetId(contact: HumanoidEnvironmentContact): string | null {
  return contact.objectId ?? contact.solidId;
}

function validateSingleContactTarget(
  contact: { objectId: string | null; solidId: string | null },
  context: z.RefinementCtx
): void {
  if (contact.objectId !== null && contact.solidId !== null) {
    context.addIssue({
      code: "custom",
      path: ["solidId"],
      message: "A physical contact cannot target an object and a static solid together"
    });
  }
}
