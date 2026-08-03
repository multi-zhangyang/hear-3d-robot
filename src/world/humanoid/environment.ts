import type { Scenario } from "../../domain/schema.js";
import type { NavigationAgentProfile } from "../navigation.js";
import type { HumanoidSimulationOptions } from "./simulation.js";

export const HUMANOID_NAVIGATION_PROFILE: NavigationAgentProfile = {
  radius: 0.34,
  height: 1.45,
  maximumClimb: 0,
  maximumSlopeDegrees: 20,
  maximumTargetProjection: 0.15
};

export function humanoidEnvironment(scenario: Scenario): HumanoidSimulationOptions {
  const fixedObjects = scenario.objects
    .filter((object) => !object.portable)
    .map((object) => ({
      id: `object-${object.id}`,
      center: object.position,
      size: object.size
    }));
  const dynamicObjects = scenario.objects
    .filter((object) => object.portable)
    .map((object) => ({
      id: object.id,
      center: object.position,
      size: object.size,
      mass: Math.max(
        0.05,
        Math.min(2, object.size.x * object.size.y * object.size.z * 4)
      )
    }));
  return {
    spawn: {
      position: { x: scenario.robot.x, y: 0, z: scenario.robot.z },
      yaw: scenario.robot.yaw
    },
    solids: [...scenario.obstacles, ...fixedObjects],
    objects: dynamicObjects
  };
}
