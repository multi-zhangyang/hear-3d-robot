import * as THREE from "three";

const MUJOCO_TO_WEB = new THREE.Matrix4().set(
  0, 1, 0, 0,
  0, 0, 1, 0,
  1, 0, 0, 0,
  0, 0, 0, 1
);

export function transformMujocoGeometry(geometry: THREE.BufferGeometry): THREE.BufferGeometry {
  geometry.applyMatrix4(MUJOCO_TO_WEB);
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

export function transformMujocoLocalVector(value: readonly [number, number, number]): THREE.Vector3 {
  return new THREE.Vector3(value[1], value[2], value[0]);
}
