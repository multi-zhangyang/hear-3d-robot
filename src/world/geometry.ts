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

export function vectorLength(value: Vec3): number {
  return Math.hypot(value.x, value.y, value.z);
}

function multiplyQuaternion(left: Quaternion, right: Quaternion): Quaternion {
  return {
    x: left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
    y: left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x,
    z: left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w,
    w: left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z
  };
}
