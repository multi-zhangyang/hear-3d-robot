import { describe, expect, it } from "vitest";
import {
  HumanoidNavigationArrivalHeadingSchema,
  humanoidNavigationArrivalHeadingSatisfied,
  humanoidNavigationShouldBeginBraking,
  humanoidNavigationStoppingDistance
} from "./navigation-arrival.js";

describe("humanoid navigation arrival heading", () => {
  it("requires an explicit arrival heading to stay within an operational tolerance", () => {
    expect(HumanoidNavigationArrivalHeadingSchema.safeParse({
      type: "face_point",
      target: { x: 1, y: 0, z: 1 },
      tolerance_radians: 0.25
    }).success).toBe(true);
    expect(HumanoidNavigationArrivalHeadingSchema.safeParse({
      type: "face_point",
      target: { x: 1, y: 0, z: 1 },
      tolerance_radians: 0.251
    }).success).toBe(false);
  });

  it("does not impose a heading when the deciding model supplies null", () => {
    expect(humanoidNavigationArrivalHeadingSatisfied(
      null,
      { x: 0, y: 0, z: 0 },
      Math.PI
    )).toBe(true);
  });
});

describe("humanoid navigation arrival braking", () => {
  it("reserves controller response and physical deceleration distance", () => {
    expect(humanoidNavigationStoppingDistance({
      planarSpeedMetersPerSecond: 0.2,
      maximumDecelerationMetersPerSecondSquared: 1,
      commandResponseHorizonSeconds: 0.2
    })).toBeCloseTo(0.06, 12);
  });

  it("accounts for command latency without treating a command as measured momentum", () => {
    expect(humanoidNavigationStoppingDistance({
      planarSpeedMetersPerSecond: 0.03,
      commandedPlanarSpeedMetersPerSecond: 0.15,
      maximumDecelerationMetersPerSecondSquared: 0.3,
      commandResponseHorizonSeconds: 0.2
    })).toBeCloseTo(0.0315, 12);
  });

  it("begins braking before entering the accepted position tolerance", () => {
    expect(humanoidNavigationShouldBeginBraking({
      distanceMeters: 0.09,
      acceptedDistanceMeters: 0.04,
      planarSpeedMetersPerSecond: 0.2,
      maximumDecelerationMetersPerSecondSquared: 1,
      commandResponseHorizonSeconds: 0.2
    })).toBe(true);
    expect(humanoidNavigationShouldBeginBraking({
      distanceMeters: 0.11,
      acceptedDistanceMeters: 0.04,
      planarSpeedMetersPerSecond: 0.2,
      maximumDecelerationMetersPerSecondSquared: 1,
      commandResponseHorizonSeconds: 0.2
    })).toBe(false);
  });
});
