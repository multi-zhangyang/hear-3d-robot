import * as THREE from "three";
import { describe, expect, it, vi } from "vitest";
import { disposeObject } from "./three-kit";

describe("disposeObject", () => {
  it("disposes each instanced mesh and its shared resources exactly once", () => {
    const root = new THREE.Group();
    const geometry = new THREE.BoxGeometry(1, 1, 1);
    const material = new THREE.MeshStandardMaterial();
    const first = new THREE.InstancedMesh(geometry, material, 2);
    const second = new THREE.InstancedMesh(geometry, material, 3);
    root.add(first, second, new THREE.Mesh(geometry, material));
    const firstDispose = vi.spyOn(first, "dispose");
    const secondDispose = vi.spyOn(second, "dispose");
    const geometryDispose = vi.spyOn(geometry, "dispose");
    const materialDispose = vi.spyOn(material, "dispose");

    disposeObject(root);

    expect(firstDispose).toHaveBeenCalledOnce();
    expect(secondDispose).toHaveBeenCalledOnce();
    expect(geometryDispose).toHaveBeenCalledOnce();
    expect(materialDispose).toHaveBeenCalledOnce();
  });
});
