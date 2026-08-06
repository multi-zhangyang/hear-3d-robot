import { describe, expect, it } from "vitest";
import {
  multiplyQuaternion,
  normalizeQuaternion,
  orientedBoxWorldHalfExtents,
  quaternionAngularDistance,
  quaternionFromRotationMatrix,
  quaternionRotationVector,
  rotateVector,
  yawFromQuaternion
} from "./geometry.js";

describe("quaternion geometry", () => {
  it("measures the shortest world-frame rotation", () => {
    const quarterTurn = normalizeQuaternion({
      x: 0,
      y: Math.sin(Math.PI / 4),
      z: 0,
      w: Math.cos(Math.PI / 4)
    });
    const identity = { x: 0, y: 0, z: 0, w: 1 };

    const rotation = quaternionRotationVector(quarterTurn, identity);
    expect(rotation.x).toBeCloseTo(0, 12);
    expect(rotation.y).toBeCloseTo(Math.PI / 2, 12);
    expect(rotation.z).toBeCloseTo(0, 12);
    expect(quaternionAngularDistance(quarterTurn, identity)).toBeCloseTo(Math.PI / 2, 12);
    expect(quaternionAngularDistance(
      { ...quarterTurn, y: -quarterTurn.y, w: -quarterTurn.w },
      identity
    )).toBeCloseTo(Math.PI / 2, 12);
  });

  it("composes local orientations and rejects invalid quaternions", () => {
    const yaw = normalizeQuaternion({ x: 0, y: 0.2, z: 0, w: 0.98 });
    const pitch = normalizeQuaternion({ x: 0.15, y: 0, z: 0, w: 0.99 });
    const composed = multiplyQuaternion(yaw, pitch);
    const sequential = rotateVector(yaw, rotateVector(pitch, { x: 0, y: 0, z: 1 }));
    const direct = rotateVector(composed, { x: 0, y: 0, z: 1 });

    expect(direct.x).toBeCloseTo(sequential.x, 12);
    expect(direct.y).toBeCloseTo(sequential.y, 12);
    expect(direct.z).toBeCloseTo(sequential.z, 12);
    expect(() => normalizeQuaternion({ x: 0, y: 0, z: 0, w: 0 })).toThrow(
      "finite non-zero magnitude"
    );
  });

  it("projects normalized root yaw for locomotion", () => {
    expect(yawFromQuaternion({
      x: 0,
      y: Math.sin(Math.PI / 4),
      z: 0,
      w: Math.cos(Math.PI / 4)
    })).toBeCloseTo(Math.PI / 2, 12);
    expect(yawFromQuaternion({ x: 0, y: 2, z: 0, w: 2 })).toBeCloseTo(
      Math.PI / 2,
      12
    );
  });

  it("converts a row-major world rotation matrix without losing orientation", () => {
    const expected = normalizeQuaternion({
      x: Math.sin(Math.PI / 4),
      y: 0,
      z: 0,
      w: Math.cos(Math.PI / 4)
    });
    const converted = quaternionFromRotationMatrix([
      1, 0, 0,
      0, 0, -1,
      0, 1, 0
    ]);

    expect(quaternionAngularDistance(converted, expected)).toBeCloseTo(0, 12);
    expect(() => quaternionFromRotationMatrix([1, 0, 0])).toThrow(
      "nine finite values"
    );
  });

  it("projects a rotated box into world-axis half extents", () => {
    const quarterTurnAroundZ = normalizeQuaternion({
      x: 0,
      y: 0,
      z: Math.sin(Math.PI / 4),
      w: Math.cos(Math.PI / 4)
    });

    expect(orientedBoxWorldHalfExtents(
      { x: 0.03, y: 0.27, z: 0.03 },
      quarterTurnAroundZ
    )).toMatchObject({
      x: expect.closeTo(0.135, 12),
      y: expect.closeTo(0.015, 12),
      z: expect.closeTo(0.015, 12)
    });
    expect(() => orientedBoxWorldHalfExtents(
      { x: -1, y: 1, z: 1 },
      { x: 0, y: 0, z: 0, w: 1 }
    )).toThrow("finite non-negative");
  });
});
