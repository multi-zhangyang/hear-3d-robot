import type { Scenario } from "../../domain/schema.js";
import type { NavigationObstacle } from "../navigation.js";
import type { HumanoidObjectSnapshot } from "./simulation.js";

export function humanoidDynamicNavigationObstacles(input: {
  scenario: Scenario;
  objectSnapshots: Readonly<Record<string, HumanoidObjectSnapshot>>;
  excludedPortableObjectIds?: ReadonlySet<string>;
}): NavigationObstacle[] {
  const excluded = input.excludedPortableObjectIds ?? new Set<string>();
  return input.scenario.objects.flatMap((object) => {
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
}
