import type { Scenario } from "../../domain/schema.js";
import type { NavigationAgentProfile } from "../navigation.js";
import type { HumanoidSimulationOptions } from "./simulation.js";
import {
  humanoidPhysicalRegionIncludesBox,
  type HumanoidPhysicalRegion
} from "./physical-region.js";

const PORTABLE_OBJECT_DENSITY_KG_PER_CUBIC_METER = 600;
const MINIMUM_PORTABLE_OBJECT_MASS_KG = 0.25;
const MAXIMUM_PORTABLE_OBJECT_MASS_KG = 2;

export const HUMANOID_NAVIGATION_PROFILE: NavigationAgentProfile = {
  radius: 0.34,
  height: 1.45,
  maximumClimb: 0,
  maximumSlopeDegrees: 20,
  maximumTargetProjection: 0.15
};

export function humanoidFixedObjectSolidId(objectId: string): string {
  if (!objectId.trim()) throw new Error("Fixed humanoid object ID is required");
  return `object-${objectId}`;
}

export function humanoidEnvironment(
  scenario: Scenario,
  physicalRegion?: HumanoidPhysicalRegion
): HumanoidSimulationOptions {
  const fixedObjects = scenario.objects
    .filter((object) => !object.portable && (!physicalRegion
      || humanoidPhysicalRegionIncludesBox(physicalRegion, {
        center: object.position,
        size: object.size
      })))
    .map((object) => ({
      id: humanoidFixedObjectSolidId(object.id),
      center: object.position,
      size: object.size
    }));
  const dynamicObjects = scenario.objects
    .filter((object) => object.portable)
    .map((object) => ({
      id: object.id,
      center: object.position,
      size: object.size,
      mass: Math.max(MINIMUM_PORTABLE_OBJECT_MASS_KG, Math.min(
        MAXIMUM_PORTABLE_OBJECT_MASS_KG,
        object.size.x
          * object.size.y
          * object.size.z
          * PORTABLE_OBJECT_DENSITY_KG_PER_CUBIC_METER
      ))
    }));
  return {
    spawn: {
      position: { x: scenario.robot.x, y: 0, z: scenario.robot.z },
      yaw: scenario.robot.yaw
    },
    solids: [
      ...scenario.obstacles.filter((solid) => !physicalRegion
        || humanoidPhysicalRegionIncludesBox(physicalRegion, solid)),
      ...fixedObjects
    ],
    objects: dynamicObjects
  };
}
