import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { transformMujocoGeometry, transformMujocoLocalVector } from "./coordinates";

describe("MuJoCo to Web coordinates", () => {
  it("maps local x/y/z onto Web z/x/y", () => {
    const mapped = transformMujocoLocalVector([1, 2, 3]);
    expect(mapped.toArray()).toEqual([2, 3, 1]);
  });

  it("transforms STL vertices without changing their scale", () => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute([
      1, 2, 3,
      2, 2, 3,
      1, 3, 3
    ], 3));

    const transformed = transformMujocoGeometry(geometry);
    const positions = transformed.getAttribute("position");
    expect([positions.getX(0), positions.getY(0), positions.getZ(0)])
      .toEqual([2, 3, 1]);
    expect(transformed.boundingBox).not.toBeNull();
    expect(transformed.boundingSphere).not.toBeNull();
  });
});
