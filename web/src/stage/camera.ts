import * as THREE from "three";
import type { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { WorldSnapshot } from "../types";
import { focusedCommand } from "../active-commands";
import type { FirstPersonLook } from "./stage-interaction";

export function fitRobot(
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  box: THREE.Box3,
  yaw: number
): void {
  camera.fov = 38;
  camera.up.set(0, 1, 0);
  camera.updateProjectionMatrix();
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const fov = THREE.MathUtils.degToRad(camera.fov);
  const portraitScale = camera.aspect < 0.72 ? 1.58 : 1;
  const distance = Math.max(
    2.9,
    size.y / (2 * Math.tan(fov / 2)) * 1.55,
    size.length() * 1.35
  ) * portraitScale;
  const forward = new THREE.Vector3(Math.sin(yaw), 0, Math.cos(yaw));
  const lateral = new THREE.Vector3(Math.cos(yaw), 0, -Math.sin(yaw));
  camera.position.copy(center)
    // A chase camera belongs behind the direction of travel. The former
    // front-facing placement put the target voxel between the camera and the
    // robot during manipulation, hiding the articulated body at the moment it
    // mattered most.
    .addScaledVector(forward, -distance * 0.78)
    .addScaledVector(lateral, distance * 0.48)
    .add(new THREE.Vector3(0, distance * 0.46, 0));
  controls.target.copy(center).add(new THREE.Vector3(0, 0.08, 0.08));
  controls.update();
}

/** First-person view anchored to the physical sensor-head transform. */
export function fitSensor(
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  snapshot: WorldSnapshot,
  look: FirstPersonLook = { yaw: 0, pitch: 0 }
): void {
  const head = snapshot.robot.links.sensor_head;
  const position = head
    ? new THREE.Vector3(head.position.x, head.position.y, head.position.z)
    : new THREE.Vector3(
        snapshot.robot.position.x,
        snapshot.robot.position.y + 0.94,
        snapshot.robot.position.z
      );
  const rotation = head
    ? new THREE.Quaternion(head.rotation.x, head.rotation.y, head.rotation.z, head.rotation.w)
    : new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), snapshot.robot.yaw);
  const physicalForward = new THREE.Vector3(0, 0, 1).applyQuaternion(rotation).normalize();
  const worldUp = new THREE.Vector3(0, 1, 0);
  const forward = physicalForward.clone().applyAxisAngle(worldUp, look.yaw).normalize();
  const right = new THREE.Vector3().crossVectors(worldUp, forward).normalize();
  if (right.lengthSq() > 1e-8) forward.applyAxisAngle(right, -look.pitch).normalize();
  camera.fov = 66;
  camera.up.copy(worldUp);
  camera.position.copy(position).addScaledVector(physicalForward, 0.24);
  controls.target.copy(position).addScaledVector(forward, 6);
  camera.updateProjectionMatrix();
  controls.update();
}

export function fitWorld(
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  box: THREE.Box3,
  bounds: { width: number; depth: number }
): void {
  camera.fov = 38;
  camera.up.set(0, 1, 0);
  camera.updateProjectionMatrix();
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const verticalFov = THREE.MathUtils.degToRad(camera.fov);
  const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * camera.aspect);
  const fitHeight = size.y / (2 * Math.tan(verticalFov / 2));
  const fitWidth = Math.max(size.x, size.z) / (2 * Math.tan(horizontalFov / 2));
  const distance = Math.max(4.5, fitHeight, fitWidth, Math.min(bounds.width, bounds.depth) * 0.42) * 1.3;
  camera.position.set(center.x + distance * 0.7, center.y + distance * 0.78, center.z + distance * 0.7);
  controls.target.copy(center);
  controls.update();
}

export function focusKey(snapshot: WorldSnapshot): string {
  const command = focusedCommand(snapshot);
  const focus = command?.focus;
  return focus
    ? `${command?.id}:${focus.position.x}:${focus.position.y}:${focus.position.z}`
    : command?.id ?? "world";
}
