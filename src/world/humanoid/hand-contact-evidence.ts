import type { Vec3 } from "../../domain/schema.js";
import type { HumanoidContactSnapshot } from "./simulation.js";
import {
  g1HandContactSurfaceHand,
  type G1HandContactSurfaceName
} from "./morphology.js";

export interface G1HandContactEvidence {
  handLink: G1HandContactSurfaceName;
  hand: "left" | "right";
  kind: "object" | "environment" | "self";
  objectId: string | null;
  counterpartBody: string | null;
  counterpartHandLink: G1HandContactSurfaceName | null;
  position: Vec3;
  /** Unit contact normal directed from the hand link toward its counterpart. */
  normalFromHand: Vec3 | null;
  normalForce: number;
}

function g1HandContactEvidence(
  contacts: readonly HumanoidContactSnapshot[]
): G1HandContactEvidence[] {
  const evidence: G1HandContactEvidence[] = [];
  for (const contact of contacts) {
    if (contact.firstHandLink) {
      evidence.push(contactEvidence(
        contact.firstHandLink,
        contact.secondObject,
        contact.secondBody,
        contact.secondHandLink,
        true,
        contact
      ));
    }
    if (contact.secondHandLink) {
      evidence.push(contactEvidence(
        contact.secondHandLink,
        contact.firstObject,
        contact.firstBody,
        contact.firstHandLink,
        false,
        contact
      ));
    }
  }
  return evidence;
}

export function g1HandObjectContacts(
  contacts: readonly HumanoidContactSnapshot[],
  objectId?: string
): G1HandContactEvidence[] {
  return g1HandContactEvidence(contacts).filter((contact) => (
    contact.kind === "object" && (objectId === undefined || contact.objectId === objectId)
  ));
}

function contactEvidence(
  handLink: G1HandContactSurfaceName,
  objectId: string | null,
  counterpartBody: string | null,
  counterpartHandLink: G1HandContactSurfaceName | null,
  handIsFirst: boolean,
  contact: HumanoidContactSnapshot
): G1HandContactEvidence {
  return {
    handLink,
    hand: g1HandContactSurfaceHand(handLink),
    kind: objectId !== null
      ? "object"
      : counterpartBody !== null || counterpartHandLink !== null ? "self" : "environment",
    objectId,
    counterpartBody,
    counterpartHandLink,
    position: { ...contact.position },
    normalFromHand: contactNormalFromHand(contact, handIsFirst),
    normalForce: contact.normalForce
  };
}

function contactNormalFromHand(
  contact: HumanoidContactSnapshot,
  handIsFirst: boolean
): Vec3 | null {
  const normal = (contact as HumanoidContactSnapshot & { normal?: Vec3 }).normal;
  if (!normal || ![normal.x, normal.y, normal.z].every(Number.isFinite)) return null;
  const magnitude = Math.hypot(normal.x, normal.y, normal.z);
  if (magnitude === 0) return null;
  const direction = handIsFirst ? 1 : -1;
  return {
    x: normalizedComponent(direction * normal.x / magnitude),
    y: normalizedComponent(direction * normal.y / magnitude),
    z: normalizedComponent(direction * normal.z / magnitude)
  };
}

function normalizedComponent(value: number): number {
  return Object.is(value, -0) ? 0 : value;
}
