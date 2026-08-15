import type { Scenario } from "../../domain/schema.js";
import type { NavigationObstacle } from "../navigation.js";
import type { HumanoidObjectSnapshot } from "./simulation.js";

// Recast's base erosion covers the feet and pelvis.  A learned G1 gait sweeps
// wrists and palms roughly 0.5 m from the route center, so baked solids also
// need an articulated-body transit moat.  Dynamic manipulable objects keep the
// base profile because approach Skills deliberately bring a hand near them.
const HUMANOID_STATIC_TRANSIT_PADDING_METERS = 0.22;

export function humanoidDynamicNavigationObstacles(input: {
  scenario: Scenario;
  objectSnapshots: Readonly<Record<string, HumanoidObjectSnapshot>>;
  excludedPortableObjectIds?: ReadonlySet<string>;
}): NavigationObstacle[] {
  const excluded = input.excludedPortableObjectIds ?? new Set<string>();
  const staticTransitObstacles = input.scenario.obstacles.map((solid) => ({
    id: `static-transit-solid-${solid.id}`,
    center: { ...solid.center },
    halfExtents: {
      x: solid.size.x / 2 + HUMANOID_STATIC_TRANSIT_PADDING_METERS,
      y: solid.size.y / 2,
      z: solid.size.z / 2 + HUMANOID_STATIC_TRANSIT_PADDING_METERS
    },
    yaw: 0
  }));
  const objectObstacles = input.scenario.objects.flatMap((object) => {
    if (object.portable && excluded.has(object.id)) return [];
    const position = object.portable
      ? input.objectSnapshots[object.id]?.position ?? object.position
      : object.position;
    const top = Math.max(0.05, position.y + object.size.y / 2);
    return [{
      id: object.portable
        ? `portable-object-${object.id}`
        : `fixed-object-${object.id}`,
      center: {
        x: position.x,
        y: top / 2,
        z: position.z
      },
      halfExtents: {
        x: object.size.x / 2,
        y: top / 2,
        z: object.size.z / 2
      },
      yaw: 0
    }];
  });
  return [...staticTransitObstacles, ...objectObstacles];
}
