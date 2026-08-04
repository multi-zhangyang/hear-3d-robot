import type { Quaternion, Vec3 } from "../domain/schema.js";

export function add(...values: Vec3[]): Vec3 {
  return values.reduce(
    (result, value) => ({
      x: result.x + value.x,
      y: result.y + value.y,
      z: result.z + value.z
    }),
    { x: 0, y: 0, z: 0 }
  );
}

export function subtract(left: Vec3, right: Vec3): Vec3 {
  return { x: left.x - right.x, y: left.y - right.y, z: left.z - right.z };
}

export function scale(value: Vec3, amount: number): Vec3 {
  return { x: value.x * amount, y: value.y * amount, z: value.z * amount };
}

export function inverseQuaternion(value: Quaternion): Quaternion {
  const magnitudeSquared = value.x ** 2 + value.y ** 2 + value.z ** 2 + value.w ** 2;
  if (!Number.isFinite(magnitudeSquared) || magnitudeSquared <= 1e-18) {
    throw new Error("Quaternion must have a finite non-zero magnitude");
  }
  return {
    x: -value.x / magnitudeSquared,
    y: -value.y / magnitudeSquared,
    z: -value.z / magnitudeSquared,
    w: value.w / magnitudeSquared
  };
}

export function rotateVector(rotation: Quaternion, value: Vec3): Vec3 {
  const vectorRotation = { x: value.x, y: value.y, z: value.z, w: 0 };
  const rotated = multiplyQuaternion(
    multiplyQuaternion(rotation, vectorRotation),
    inverseQuaternion(rotation)
  );
  return { x: rotated.x, y: rotated.y, z: rotated.z };
}

export function orientedBoxWorldHalfExtents(
  size: Vec3,
  rotation: Quaternion
): Vec3 {
  if (![size.x, size.y, size.z].every((component) => (
    Number.isFinite(component) && component >= 0
  ))) {
    throw new Error("Box size must have finite non-negative components");
  }
  const normalized = normalizeQuaternion(rotation);
  const localAxes = [
    rotateVector(normalized, { x: size.x / 2, y: 0, z: 0 }),
    rotateVector(normalized, { x: 0, y: size.y / 2, z: 0 }),
    rotateVector(normalized, { x: 0, y: 0, z: size.z / 2 })
  ];
  return {
    x: localAxes.reduce((sum, axis) => sum + Math.abs(axis.x), 0),
    y: localAxes.reduce((sum, axis) => sum + Math.abs(axis.y), 0),
    z: localAxes.reduce((sum, axis) => sum + Math.abs(axis.z), 0)
  };
}

export function vectorLength(value: Vec3): number {
  return Math.hypot(value.x, value.y, value.z);
}

export function multiplyQuaternion(left: Quaternion, right: Quaternion): Quaternion {
  return {
    x: left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
    y: left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x,
    z: left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w,
    w: left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z
  };
}

export function normalizeQuaternion(value: Quaternion): Quaternion {
  const magnitude = Math.hypot(value.x, value.y, value.z, value.w);
  if (!Number.isFinite(magnitude) || magnitude <= 1e-9) {
    throw new Error("Quaternion must have a finite non-zero magnitude");
  }
  return {
    x: value.x / magnitude,
    y: value.y / magnitude,
    z: value.z / magnitude,
    w: value.w / magnitude
  };
}

export function quaternionRotationVector(
  target: Quaternion,
  current: Quaternion
): Vec3 {
  let error = normalizeQuaternion(multiplyQuaternion(
    normalizeQuaternion(target),
    inverseQuaternion(normalizeQuaternion(current))
  ));
  if (error.w < 0) {
    error = { x: -error.x, y: -error.y, z: -error.z, w: -error.w };
  }
  const vectorMagnitude = Math.hypot(error.x, error.y, error.z);
  if (vectorMagnitude <= 1e-9) {
    return { x: 2 * error.x, y: 2 * error.y, z: 2 * error.z };
  }
  const angle = 2 * Math.atan2(vectorMagnitude, Math.max(0, error.w));
  const factor = angle / vectorMagnitude;
  return { x: error.x * factor, y: error.y * factor, z: error.z * factor };
}

export function quaternionAngularDistance(
  left: Quaternion,
  right: Quaternion
): number {
  return vectorLength(quaternionRotationVector(left, right));
}
