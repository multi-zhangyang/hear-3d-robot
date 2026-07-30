import type { Vec3 } from "../domain/schema.js";
import type { Quaternion } from "./kinematics.js";
import type { RobotJointState } from "./robot-model.js";

export function vector(value: { x: number; y: number; z: number }): Vec3 {
  return { x: value.x, y: value.y, z: value.z };
}

export function quaternion(value: { x: number; y: number; z: number; w: number }): Quaternion {
  return { x: value.x, y: value.y, z: value.z, w: value.w };
}

export function add(...values: Vec3[]): Vec3 {
  return values.reduce(
    (result, value) => ({ x: result.x + value.x, y: result.y + value.y, z: result.z + value.z }),
    { x: 0, y: 0, z: 0 }
  );
}

export function subtract(left: Vec3, right: Vec3): Vec3 {
  return { x: left.x - right.x, y: left.y - right.y, z: left.z - right.z };
}

export function scale(value: Vec3, amount: number): Vec3 {
  return { x: value.x * amount, y: value.y * amount, z: value.z * amount };
}

export function armVector(forward: Vec3, angle: number, length: number): Vec3 {
  return {
    x: forward.x * Math.cos(angle) * length,
    y: Math.sin(angle) * length,
    z: forward.z * Math.cos(angle) * length
  };
}

/**
 * Orientation of an arm link whose offset from its parent joint is
 * `armVector(forward, angle, length)`. The two must agree: armVector puts the
 * link along (0, sin angle, cos angle), and pitchRotation(angle) is what takes
 * local +Z there. Negating the angle here instead mirrors every link's frame
 * about the horizontal plane, so the colliders and the drawn arm both bend the
 * opposite way from the joint chain they belong to.
 */
export function armRotation(yaw: number, angle: number): Quaternion {
  return multiplyQuaternion(yawRotation(yaw), pitchRotation(angle));
}

export function yawRotation(yaw: number): Quaternion {
  return { x: 0, y: Math.sin(yaw / 2), z: 0, w: Math.cos(yaw / 2) };
}

export function quaternionYaw(rotation: Quaternion): number {
  return Math.atan2(
    2 * (rotation.w * rotation.y + rotation.x * rotation.z),
    1 - 2 * (rotation.y * rotation.y + rotation.z * rotation.z)
  );
}

export function pitchRotation(pitch: number): Quaternion {
  return { x: -Math.sin(pitch / 2), y: 0, z: 0, w: Math.cos(pitch / 2) };
}

export function multiplyQuaternion(left: Quaternion, right: Quaternion): Quaternion {
  return normalizeQuaternion({
    x: left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
    y: left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x,
    z: left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w,
    w: left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z
  });
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

function normalizeQuaternion(value: Quaternion): Quaternion {
  const length = Math.hypot(value.x, value.y, value.z, value.w);
  return { x: value.x / length, y: value.y / length, z: value.z / length, w: value.w / length };
}

export function rotateVector(rotation: Quaternion, value: Vec3): Vec3 {
  const vectorRotation = { x: value.x, y: value.y, z: value.z, w: 0 };
  const rotated = multiplyQuaternionRaw(
    multiplyQuaternionRaw(rotation, vectorRotation),
    inverseQuaternion(rotation)
  );
  return { x: rotated.x, y: rotated.y, z: rotated.z };
}

function multiplyQuaternionRaw(left: Quaternion, right: Quaternion): Quaternion {
  return {
    x: left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
    y: left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x,
    z: left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w,
    w: left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z
  };
}

export function normalizeVector(value: Vec3): Vec3 {
  const length = vectorLength(value);
  return length > 1e-9 ? scale(value, 1 / length) : { x: 0, y: 0, z: 0 };
}

export function vectorLength(value: Vec3): number {
  return Math.hypot(value.x, value.y, value.z);
}

export function dot(left: Vec3, right: Vec3): number {
  return left.x * right.x + left.y * right.y + left.z * right.z;
}

export function finiteVector(value: Vec3): boolean {
  return Number.isFinite(value.x) && Number.isFinite(value.y) && Number.isFinite(value.z);
}

export function sameVector(left: Vec3, right: Vec3): boolean {
  return left.x === right.x && left.y === right.y && left.z === right.z;
}

export function finiteQuaternion(value: Quaternion): boolean {
  return Number.isFinite(value.x)
    && Number.isFinite(value.y)
    && Number.isFinite(value.z)
    && Number.isFinite(value.w)
    && Math.hypot(value.x, value.y, value.z, value.w) > 1e-6;
}

export function normalizeAngle(value: number): number {
  return Math.atan2(Math.sin(value), Math.cos(value));
}

export function planarDistance(
  left: { x: number; z: number },
  right: { x: number; z: number }
): number {
  return Math.hypot(left.x - right.x, left.z - right.z);
}

export function moveTowards(current: number, target: number, maximumDelta: number): number {
  const delta = target - current;
  return Math.abs(delta) <= maximumDelta
    ? target
    : current + Math.sign(delta) * maximumDelta;
}

export function armTargetsReached(
  current: RobotJointState,
  target: RobotJointState,
  tolerance: number
): boolean {
  return Math.abs(current.shoulder - target.shoulder) <= tolerance
    && Math.abs(current.elbow - target.elbow) <= tolerance
    && Math.abs(current.wrist - target.wrist) <= tolerance;
}

export function boundedTolerance(
  value: number | undefined,
  defaultValue: number,
  minimum: number,
  maximum: number
): number {
  if (value === undefined) return defaultValue;
  return Number.isFinite(value) ? clamp(value, minimum, maximum) : defaultValue;
}

export function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}
