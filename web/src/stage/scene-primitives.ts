import * as THREE from "three";
import type { Vec3 } from "../types";

export function zoneOutline(size: Vec3): THREE.LineSegments {
  const outline = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.PlaneGeometry(size.x, size.z)),
    new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.75 })
  );
  outline.rotation.x = -Math.PI / 2;
  outline.position.y = 0.012;
  return outline;
}
