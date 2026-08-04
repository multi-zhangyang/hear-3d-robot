import {
  Euler,
  Quaternion as ThreeQuaternion
} from "three";
import type { Quaternion } from "../types";

export interface OrientationDegrees {
  roll: number;
  pitch: number;
  heading: number;
}

const EULER_ORDER = "YXZ";

export function orientationDegreesToQuaternion(
  orientation: OrientationDegrees
): Quaternion {
  assertFiniteOrientation(orientation);
  const value = new ThreeQuaternion().setFromEuler(new Euler(
    degreesToRadians(orientation.pitch),
    degreesToRadians(orientation.heading),
    degreesToRadians(orientation.roll),
    EULER_ORDER
  )).normalize();
  return { x: value.x, y: value.y, z: value.z, w: value.w };
}

export function quaternionToOrientationDegrees(
  orientation: Quaternion
): OrientationDegrees {
  const magnitude = Math.hypot(
    orientation.x,
    orientation.y,
    orientation.z,
    orientation.w
  );
  if (!Number.isFinite(magnitude) || magnitude <= 1e-9) {
    throw new Error("Orientation must be a finite non-zero quaternion");
  }
  const value = new ThreeQuaternion(
    orientation.x / magnitude,
    orientation.y / magnitude,
    orientation.z / magnitude,
    orientation.w / magnitude
  );
  const euler = new Euler().setFromQuaternion(value, EULER_ORDER);
  return {
    roll: radiansToDegrees(euler.z),
    pitch: radiansToDegrees(euler.x),
    heading: radiansToDegrees(euler.y)
  };
}

export function degreesToRadians(value: number): number {
  return value * Math.PI / 180;
}

export function radiansToDegrees(value: number): number {
  return value * 180 / Math.PI;
}

function assertFiniteOrientation(orientation: OrientationDegrees): void {
  if (!Number.isFinite(orientation.roll)
    || !Number.isFinite(orientation.pitch)
    || !Number.isFinite(orientation.heading)) {
    throw new Error("Orientation angles must be finite");
  }
}
