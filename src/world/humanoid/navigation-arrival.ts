import { z } from "zod";
import { Vec3Schema, type Vec3 } from "../../domain/schema.js";

const HeadingToleranceSchema = z.number().finite().min(0.03).max(0.25);

export const HumanoidNavigationArrivalHeadingSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("face_point"),
    target: Vec3Schema,
    tolerance_radians: HeadingToleranceSchema
  }).strict(),
  z.object({
    type: z.literal("yaw"),
    yaw_radians: z.number().finite(),
    tolerance_radians: HeadingToleranceSchema
  }).strict()
]);

export type HumanoidNavigationArrivalHeading = z.infer<
  typeof HumanoidNavigationArrivalHeadingSchema
>;

function humanoidNavigationArrivalYaw(
  heading: HumanoidNavigationArrivalHeading,
  position: Vec3
): number {
  const parsed = HumanoidNavigationArrivalHeadingSchema.parse(heading);
  if (parsed.type === "yaw") return normalizeAngle(parsed.yaw_radians);
  return Math.atan2(
    parsed.target.x - position.x,
    parsed.target.z - position.z
  );
}

export function humanoidNavigationArrivalHeadingError(
  heading: HumanoidNavigationArrivalHeading,
  position: Vec3,
  currentYaw: number
): number {
  return normalizeAngle(humanoidNavigationArrivalYaw(heading, position) - currentYaw);
}

export function humanoidNavigationArrivalHeadingSatisfied(
  heading: HumanoidNavigationArrivalHeading | null,
  position: Vec3,
  currentYaw: number
): boolean {
  if (heading === null) return true;
  return Math.abs(
    humanoidNavigationArrivalHeadingError(heading, position, currentYaw)
  ) <= heading.tolerance_radians;
}

export function humanoidNavigationStoppingDistance(input: {
  planarSpeedMetersPerSecond: number;
  commandedPlanarSpeedMetersPerSecond?: number;
  maximumDecelerationMetersPerSecondSquared: number;
  commandResponseHorizonSeconds: number;
}): number {
  const speed = finiteNonnegative(
    input.planarSpeedMetersPerSecond,
    "Navigation planar speed"
  );
  const deceleration = finitePositive(
    input.maximumDecelerationMetersPerSecondSquared,
    "Navigation maximum deceleration"
  );
  const responseHorizon = finiteNonnegative(
    input.commandResponseHorizonSeconds,
    "Navigation command response horizon"
  );
  const commandedSpeed = input.commandedPlanarSpeedMetersPerSecond === undefined
    ? speed
    : finiteNonnegative(
        input.commandedPlanarSpeedMetersPerSecond,
        "Navigation commanded planar speed"
      );
  return commandedSpeed * responseHorizon + speed * speed / (2 * deceleration);
}

export function humanoidNavigationShouldBeginBraking(input: {
  distanceMeters: number;
  acceptedDistanceMeters: number;
  planarSpeedMetersPerSecond: number;
  commandedPlanarSpeedMetersPerSecond?: number;
  maximumDecelerationMetersPerSecondSquared: number;
  commandResponseHorizonSeconds: number;
}): boolean {
  const distance = finiteNonnegative(input.distanceMeters, "Navigation distance");
  const acceptedDistance = finiteNonnegative(
    input.acceptedDistanceMeters,
    "Navigation accepted distance"
  );
  if (distance <= acceptedDistance) return false;
  return humanoidNavigationStoppingDistance(input) >= distance - acceptedDistance;
}

function normalizeAngle(value: number): number {
  return Math.atan2(Math.sin(value), Math.cos(value));
}

function finiteNonnegative(value: number, label: string): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`${label} must be finite and nonnegative`);
  }
  return value;
}

function finitePositive(value: number, label: string): number {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${label} must be finite and positive`);
  }
  return value;
}
