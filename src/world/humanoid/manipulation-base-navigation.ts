import type { Vec3 } from "../../domain/schema.js";
import { navigationObstaclePlanarExpansion } from "../navigation.js";
import { HUMANOID_NAVIGATION_PROFILE } from "./environment.js";
import type { HumanoidSolidToken } from "./solid-observation.js";

export const HUMANOID_MANIPULATION_SUPPORT_CLEARANCE_MARGIN_METERS = 0.01;

export function humanoidManipulationSupportSolids(
  solidTokens: readonly HumanoidSolidToken[],
  objectId: string
): HumanoidSolidToken[] {
  return solidTokens.filter((solid) => solid.currentContacts.some((contact) => (
    contact.firstObject === objectId && contact.secondSolid === solid.id
  ) || (
    contact.secondObject === objectId && contact.firstSolid === solid.id
  )));
}

export function humanoidManipulationBaseNavigationBlockerIds(input: {
  solidTokens: readonly HumanoidSolidToken[];
  objectId: string;
  rootWorldTarget: Vec3;
}): string[] {
  const expansion = navigationObstaclePlanarExpansion(
    HUMANOID_NAVIGATION_PROFILE.radius
  ) + HUMANOID_MANIPULATION_SUPPORT_CLEARANCE_MARGIN_METERS;
  return humanoidManipulationSupportSolids(
    input.solidTokens,
    input.objectId
  ).filter((solid) => (
    Math.abs(input.rootWorldTarget.x - solid.center.x)
      <= solid.size.x / 2 + expansion
      && Math.abs(input.rootWorldTarget.z - solid.center.z)
        <= solid.size.z / 2 + expansion
  )).map(({ id }) => id);
}
