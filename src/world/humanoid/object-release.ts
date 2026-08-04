import {
  G1_HAND_CONTACT_SURFACE_NAMES,
  g1HandContactSurfaceHand,
  type G1HandContactSurfaceName
} from "./morphology.js";

const handSurfaces = new Set<string>(G1_HAND_CONTACT_SURFACE_NAMES);

export interface HumanoidObjectReleaseContact {
  normalForce: number;
  firstObject: string | null;
  secondObject: string | null;
  firstHandLink?: string | null;
  secondHandLink?: string | null;
}

export interface HumanoidObjectReleaseAssessment {
  protocol: "humanoid-object-release-assessment-v1";
  objectId: string;
  hand: "left" | "right";
  status: "satisfied" | "unsatisfied" | "uncertain";
  reason: "object_released" | "hand_contact_present" | "object_not_observable";
  objectObservable: boolean;
  handContactCount: number | null;
  contactSurfaces: G1HandContactSurfaceName[];
  totalNormalForceN: number | null;
}

export function assessHumanoidObjectReleased(input: {
  objectId: string;
  hand: "left" | "right";
  objectObservable: boolean;
  contacts: readonly HumanoidObjectReleaseContact[];
}): HumanoidObjectReleaseAssessment {
  if (!input.objectObservable) {
    return {
      protocol: "humanoid-object-release-assessment-v1",
      objectId: input.objectId,
      hand: input.hand,
      status: "uncertain",
      reason: "object_not_observable",
      objectObservable: false,
      handContactCount: null,
      contactSurfaces: [],
      totalNormalForceN: null
    };
  }
  const contacts = humanoidObjectHandContacts(
    input.contacts,
    input.objectId,
    input.hand
  );
  return {
    protocol: "humanoid-object-release-assessment-v1",
    objectId: input.objectId,
    hand: input.hand,
    status: contacts.length === 0 ? "satisfied" : "unsatisfied",
    reason: contacts.length === 0 ? "object_released" : "hand_contact_present",
    objectObservable: true,
    handContactCount: contacts.length,
    contactSurfaces: [...new Set(contacts.map((contact) => contact.surface))].sort(),
    totalNormalForceN: contacts.reduce((total, contact) => (
      total + Math.max(0, contact.normalForce)
    ), 0)
  };
}

export function humanoidObjectHandContacts(
  contacts: readonly HumanoidObjectReleaseContact[],
  objectId: string,
  hand: "left" | "right"
): Array<{ surface: G1HandContactSurfaceName; normalForce: number }> {
  return contacts.flatMap((contact) => {
    const rawSurface = contact.firstObject === objectId
      && contact.secondObject !== objectId
      ? contact.secondHandLink
      : contact.secondObject === objectId
        && contact.firstObject !== objectId
        ? contact.firstHandLink
        : null;
    if (!isHandSurface(rawSurface)
      || g1HandContactSurfaceHand(rawSurface) !== hand) return [];
    return [{ surface: rawSurface, normalForce: contact.normalForce }];
  });
}

function isHandSurface(value: string | null | undefined):
  value is G1HandContactSurfaceName {
  return typeof value === "string" && handSurfaces.has(value);
}
