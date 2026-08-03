import * as THREE from "three";
import type { Vec3 } from "../types";
import { SHADOW_MAP_SIZE } from "./render-quality";

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

export function zoneOutline(size: Vec3): THREE.LineSegments {
  const outline = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.PlaneGeometry(size.x, size.z)),
    new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.75 })
  );
  outline.rotation.x = -Math.PI / 2;
  outline.position.y = 0.012;
  return outline;
}
