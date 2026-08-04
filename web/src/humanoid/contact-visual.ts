import type { HumanoidWorldSnapshot, Vec3 } from "../types";

export type HumanoidContactVisualKind = "hand" | "foot" | "body" | "solid";

export interface HumanoidContactVisual {
  kind: HumanoidContactVisualKind;
  position: Vec3;
  normalForce: number;
  scale: number;
}

const DEFAULT_CONTACT_LIMIT = 32;
const FOOT_BODY = /(?:^|_)(?:left|right)_(?:ankle|foot)(?:_|$)/u;

export function humanoidContactVisuals(
  snapshot: HumanoidWorldSnapshot,
  limit = DEFAULT_CONTACT_LIMIT
): HumanoidContactVisual[] {
  if (!Number.isSafeInteger(limit) || limit < 0) {
    throw new Error("Humanoid contact visual limit must be a non-negative integer");
  }
  if (limit === 0) return [];

  const contacts = snapshot.robot.contacts.map((contact, sourceIndex) => {
    const kind = contact.firstSolid || contact.secondSolid
      ? "solid"
      : contact.firstHandLink || contact.secondHandLink
        ? "hand"
        : isFootBody(contact.firstBody) || isFootBody(contact.secondBody)
          ? "foot"
          : "body";
    const normalForce = Math.max(0, contact.normalForce);
    return {
      sourceIndex,
      priority: kind === "solid" ? 0 : kind === "hand" ? 1 : kind === "body" ? 2 : 3,
      visual: {
        kind,
        position: { ...contact.position },
        normalForce,
        scale: contactScale(normalForce)
      } satisfies HumanoidContactVisual
    };
  });

  if (contacts.length > 0) {
    return contacts
      .sort((left, right) => left.priority - right.priority
        || right.visual.normalForce - left.visual.normalForce
        || left.sourceIndex - right.sourceIndex)
      .slice(0, limit)
      .map(({ visual }) => visual);
  }

  return [
    ...snapshot.robot.feet.left.points,
    ...snapshot.robot.feet.right.points
  ].slice(0, limit).map((position) => ({
    kind: "foot" as const,
    position: { ...position },
    normalForce: 0,
    scale: 0.72
  }));
}

function isFootBody(body: string | null): boolean {
  return body !== null && FOOT_BODY.test(body);
}

function contactScale(normalForce: number): number {
  return Math.min(1.65, 0.72 + Math.sqrt(normalForce) / 22);
}
