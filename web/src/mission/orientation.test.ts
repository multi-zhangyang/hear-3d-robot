import { describe, expect, it } from "vitest";
import {
  orientationDegreesToQuaternion,
  quaternionToOrientationDegrees
} from "./orientation";

describe("姿态输入转换", () => {
  it("把侧倾、俯仰和朝向转换成单位四元数", () => {
    const quaternion = orientationDegreesToQuaternion({
      roll: 18,
      pitch: -24,
      heading: 72
    });
    expect(Math.hypot(
      quaternion.x,
      quaternion.y,
      quaternion.z,
      quaternion.w
    )).toBeCloseTo(1, 12);
    expect(quaternionToOrientationDegrees(quaternion)).toEqual({
      roll: expect.closeTo(18, 10),
      pitch: expect.closeTo(-24, 10),
      heading: expect.closeTo(72, 10)
    });
  });

  it("把相反符号的等价四元数显示为同一姿态", () => {
    const quaternion = orientationDegreesToQuaternion({
      roll: -12,
      pitch: 16,
      heading: -48
    });
    expect(quaternionToOrientationDegrees({
      x: -quaternion.x,
      y: -quaternion.y,
      z: -quaternion.z,
      w: -quaternion.w
    })).toEqual({
      roll: expect.closeTo(-12, 10),
      pitch: expect.closeTo(16, 10),
      heading: expect.closeTo(-48, 10)
    });
  });

  it("拒绝无法表达真实姿态的输入", () => {
    expect(() => orientationDegreesToQuaternion({
      roll: Number.NaN,
      pitch: 0,
      heading: 0
    })).toThrow("finite");
    expect(() => quaternionToOrientationDegrees({ x: 0, y: 0, z: 0, w: 0 }))
      .toThrow("non-zero quaternion");
  });
});
