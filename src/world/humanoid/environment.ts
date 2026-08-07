import type { Scenario } from "../../domain/schema.js";
import type { NavigationAgentProfile } from "../navigation.js";
import type { HumanoidSimulationOptions } from "./simulation.js";
import { humanoidObjectCapability } from "./object-capability.js";
import {
  humanoidPhysicalRegionIncludesBox,
  type HumanoidPhysicalRegion
} from "./physical-region.js";

export const HUMANOID_NAVIGATION_PROFILE: NavigationAgentProfile = {
  radius: 0.18,
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
    .filter((object) => !object.portable && object.capability === undefined
      && (!physicalRegion
      || humanoidPhysicalRegionIncludesBox(physicalRegion, {
        center: object.position,
        size: object.size
      })))
    .map((object) => ({
      id: humanoidFixedObjectSolidId(object.id),
      center: object.position,
      size: object.size
    }));
  const physicalObjects = scenario.objects
    .filter((object) => object.portable || (
      object.capability !== undefined
        && (!physicalRegion || humanoidPhysicalRegionIncludesBox(physicalRegion, {
          center: object.position,
          size: object.size
        }))
    ))
    .map((object) => {
      const capability = humanoidObjectCapability(object);
      const articulation = capability.articulation;
      return {
        id: object.id,
        center: object.position,
        size: object.size,
        mass: capability.massKg,
        shape: capability.shape,
        friction: capability.friction,
        mobility: articulation
          ? {
              type: articulation.type,
              axis: { ...articulation.axis },
              anchor: { ...articulation.anchor_world },
              range: { ...articulation.range },
              initialPosition: articulation.initial_position,
              damping: articulation.damping,
              frictionLoss: articulation.friction_loss
            }
          : capability.mobility === "free"
            ? { type: "free" as const }
            : { type: "fixed" as const },
        ...(capability.container
          ? {
              container: {
                interiorCenter: { ...capability.container.interior_center },
                interiorSize: { ...capability.container.interior_size },
                openingDirection: { ...capability.container.opening_direction },
                wallThickness: capability.container.wall_thickness_m
              }
            }
          : {})
      };
    });
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
    objects: physicalObjects
  };
}
