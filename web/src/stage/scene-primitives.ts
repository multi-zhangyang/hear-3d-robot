import * as THREE from "three";
import type { Vec3 } from "../types";
import { SHADOW_MAP_SIZE } from "./render-quality";

export const STAGE_VOID = 0x080b11;
const STAGE_FLOOR = 0x121722;

export function addStudioLighting(scene: THREE.Scene, bounds: { width: number; depth: number }): void {
  const worldExtent = Math.max(bounds.width, bounds.depth);
  scene.add(new THREE.HemisphereLight(0x5f7391, 0x0a0e15, 1.05));

  const key = new THREE.DirectionalLight(0xfff2e0, 3.1);
  key.position.set(bounds.width * 0.45, 9.5, bounds.depth * 0.9);
  key.castShadow = true;
  key.shadow.mapSize.set(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
  key.shadow.camera.near = 0.5;
  key.shadow.camera.far = Math.max(60, worldExtent * 1.4);
  const extent = worldExtent * 0.75;
  key.shadow.camera.left = -extent;
  key.shadow.camera.right = extent;
  key.shadow.camera.top = extent;
  key.shadow.camera.bottom = -extent;
  key.shadow.bias = -0.0008;
  key.shadow.normalBias = 0.02;
  key.target.position.set(bounds.width / 2, 0, bounds.depth / 2);
  scene.add(key, key.target);

  const fill = new THREE.DirectionalLight(0x8fb4ff, 1.05);
  fill.position.set(-bounds.width * 0.6, 4.5, -bounds.depth * 0.2);
  scene.add(fill);

  const rim = new THREE.DirectionalLight(0x59f0d8, 1.35);
  rim.position.set(bounds.width * 0.1, 3.2, -bounds.depth * 1.1);
  scene.add(rim);
}

export function createFloor(
  bounds: { width: number; depth: number },
  voxelBlock: number | undefined
): THREE.Group {
  const group = new THREE.Group();
  const voxel = voxelBlock !== undefined;
  const plate = new THREE.Mesh(
    new THREE.BoxGeometry(bounds.width, 0.06, bounds.depth),
    new THREE.MeshStandardMaterial({ color: STAGE_FLOOR, roughness: 0.72, metalness: 0.22 })
  );
  plate.position.set(bounds.width / 2, voxel ? -voxelBlock - 0.04 : -0.04, bounds.depth / 2);
  plate.receiveShadow = true;
  group.add(plate);

  const size = Math.max(bounds.width, bounds.depth) * 2.4;
  const apron = new THREE.Mesh(
    new THREE.CircleGeometry(size / 2, 64),
    new THREE.MeshBasicMaterial({ color: STAGE_VOID })
  );
  apron.rotation.x = -Math.PI / 2;
  apron.position.set(bounds.width / 2, -0.075, bounds.depth / 2);
  group.add(apron);

  const grid = new THREE.GridHelper(
    Math.max(bounds.width, bounds.depth),
    Math.ceil(Math.max(bounds.width, bounds.depth) / (voxel ? 4 : 1)),
    voxel ? 0x596168 : 0x24c8ae,
    voxel ? 0x383e43 : 0x303940
  );
  const gridMaterial = grid.material as THREE.Material;
  gridMaterial.transparent = true;
  gridMaterial.opacity = voxel ? 0.13 : 0.26;
  grid.position.set(bounds.width / 2, 0.001, bounds.depth / 2);
  group.add(grid);

  const border = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.PlaneGeometry(bounds.width, bounds.depth)),
    new THREE.LineBasicMaterial({ color: 0x24c8ae, transparent: true, opacity: 0.45 })
  );
  border.rotation.x = -Math.PI / 2;
  border.position.set(bounds.width / 2, 0.004, bounds.depth / 2);
  group.add(border);
  return group;
}

export function zoneOutline(size: Vec3): THREE.LineSegments {
  const outline = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.PlaneGeometry(size.x, size.z)),
    new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.75 })
  );
  outline.rotation.x = -Math.PI / 2;
  outline.position.y = 0.012;
  return outline;
}
