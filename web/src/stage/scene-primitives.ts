import * as THREE from "three";
import type { Vec3 } from "../types";
import { SHADOW_MAP_SIZE } from "./render-quality";

export const STAGE_VOID = 0x080b11;
export const STAGE_FLOOR = 0x121722;

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

export function createFocusMarker(): THREE.Group {
  const marker = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({
    color: 0x8f7cf5,
    emissive: 0x8f7cf5,
    emissiveIntensity: 1.1,
    transparent: true,
    opacity: 0.9
  });
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.2, 0.012, 8, 32), material);
  ring.rotation.x = Math.PI / 2;
  marker.add(ring);
  for (const angle of [0, Math.PI / 2, Math.PI, Math.PI * 1.5]) {
    const tick = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.012, 0.08), material);
    tick.position.set(Math.cos(angle) * 0.26, 0, Math.sin(angle) * 0.26);
    tick.rotation.y = -angle;
    marker.add(tick);
  }
  const pin = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 0.4, 8), material);
  pin.position.y = 0.2;
  marker.add(pin);
  marker.visible = false;
  return marker;
}

const labelToCamera = new THREE.Vector3();
const LABEL_VIEWPORT_HEIGHT = 0.042;

export function addEntityLabel(
  parent: THREE.Object3D,
  y: number,
  text: string,
  groundRadius = 0,
  screenLift = 0
): void {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  if (!context) return;
  const font = '600 40px "Space Grotesk Variable", Inter, "Segoe UI", sans-serif';
  const scale = 2;
  context.font = font;
  const paddingX = 20;
  const width = Math.max(96, Math.ceil(context.measureText(text).width + paddingX * 2));
  const height = 62;
  canvas.width = width * scale;
  canvas.height = height * scale;
  context.scale(scale, scale);
  context.font = font;

  const radius = height / 2;
  context.beginPath();
  context.moveTo(radius, 0);
  context.lineTo(width - radius, 0);
  context.arc(width - radius, radius, radius, -Math.PI / 2, Math.PI / 2);
  context.lineTo(radius, height);
  context.arc(radius, radius, radius, Math.PI / 2, Math.PI * 1.5);
  context.closePath();
  context.fillStyle = "rgba(10, 14, 21, 0.82)";
  context.fill();
  context.lineWidth = 2;
  context.strokeStyle = "rgba(53, 224, 196, 0.55)";
  context.stroke();
  context.fillStyle = "#e8eef7";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text, width / 2, height / 2 + 1);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.minFilter = THREE.LinearFilter;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: true,
    depthWrite: false
  }));
  sprite.name = "entity-label";
  if (screenLift !== 0) sprite.center.set(0.5, -screenLift);
  sprite.onBeforeRender = (_renderer, _scene, camera): void => {
    const aspect = camera instanceof THREE.PerspectiveCamera ? camera.aspect : 1;
    sprite.scale.set(LABEL_VIEWPORT_HEIGHT * (width / height) / aspect, LABEL_VIEWPORT_HEIGHT, 1);
    if (groundRadius === 0) return;
    labelToCamera.setFromMatrixPosition(camera.matrixWorld);
    parent.worldToLocal(labelToCamera);
    labelToCamera.y = 0;
    if (labelToCamera.lengthSq() < 1e-6) return;
    labelToCamera.setLength(groundRadius);
    sprite.position.set(labelToCamera.x, y, labelToCamera.z);
  };
  sprite.position.set(0, y, 0);
  parent.add(sprite);
}
