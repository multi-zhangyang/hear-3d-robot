import type { Scenario, Vec3 } from "../../domain/schema.js";
import { orientedBoxWorldHalfExtents } from "../geometry.js";
import type { HumanoidSimulationSnapshot } from "./simulation.js";
import type { HumanoidSolidToken } from "./solid-observation.js";

export interface HumanoidSpatialBeliefSolid {
  center: Vec3;
  size: Vec3;
  occupiesNavigationSpace?: boolean;
}

export function humanoidSpatialBeliefSolids(input: {
  scenario: Scenario;
  robot: HumanoidSimulationSnapshot;
  visibleObjectIds: readonly string[];
  solidTokens: readonly HumanoidSolidToken[];
  carriedObjectIds: ReadonlySet<string>;
}): HumanoidSpatialBeliefSolid[] {
  const observedObjectIds = new Set([
    ...input.visibleObjectIds,
    ...input.robot.contacts.flatMap((contact) => [
      ...(contact.firstObject ? [contact.firstObject] : []),
      ...(contact.secondObject ? [contact.secondObject] : [])
    ])
  ]);
  const objects = [...observedObjectIds].sort(compareCodePoints).map((objectId) => {
    const object = input.robot.objects[objectId];
    const descriptor = input.scenario.objects.find(({ id }) => id === objectId);
    if (!object || !descriptor) {
      throw new Error(`Observed object has no spatial descriptor: ${objectId}`);
    }
    const halfExtents = orientedBoxWorldHalfExtents(
      descriptor.size,
      object.rotation
    );
    return {
      center: { ...object.position },
      size: {
        x: halfExtents.x * 2,
        y: halfExtents.y * 2,
        z: halfExtents.z * 2
      },
      occupiesNavigationSpace: !input.carriedObjectIds.has(objectId)
    };
  });
  return [
    ...input.solidTokens
      .filter((solid) => solid.kind === "block"
        || !observedObjectIds.has(solid.sourceId))
      .map(({ center, size }) => ({
        center: { ...center },
        size: { ...size }
      })),
    ...objects
  ];
}

function compareCodePoints(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}
