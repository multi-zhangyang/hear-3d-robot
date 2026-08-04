import type { Vec3 } from "../../domain/schema.js";
import type { NavigationPlan } from "../navigation.js";

export function boundedNavigationChunk(
  plan: NavigationPlan,
  maximumDistance: number
): NavigationPlan {
  if (!Number.isFinite(maximumDistance) || maximumDistance <= 0) {
    throw new Error("Navigation chunk distance must be finite and positive");
  }
  if (plan.distance <= maximumDistance || plan.waypoints.length < 2) {
    return structuredClone(plan);
  }
  const waypoints: Vec3[] = [{ ...plan.waypoints[0]! }];
  let distance = 0;
  for (let index = 1; index < plan.waypoints.length; index += 1) {
    const from = plan.waypoints[index - 1]!;
    const to = plan.waypoints[index]!;
    const segment = Math.hypot(to.x - from.x, to.z - from.z);
    if (distance + segment <= maximumDistance) {
      waypoints.push({ ...to });
      distance += segment;
      continue;
    }
    const remaining = maximumDistance - distance;
    const ratio = segment <= 1e-9 ? 0 : remaining / segment;
    const endpoint = {
      x: from.x + (to.x - from.x) * ratio,
      y: from.y + (to.y - from.y) * ratio,
      z: from.z + (to.z - from.z) * ratio
    };
    waypoints.push(endpoint);
    return {
      waypoints,
      distance: maximumDistance,
      resolvedTarget: { ...endpoint },
      projectionDistance: 0
    };
  }
  return structuredClone(plan);
}
