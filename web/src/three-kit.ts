import * as THREE from "three";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

/**
 * A metal surface shows almost nothing but its reflection, so a MeshStandardMaterial
 * with high metalness renders near black when the scene has no environment — the
 * directional lights contribute a specular dot and nothing else. The rig's alloy
 * bearings, rails and wheel rims are exactly that material, so without this they
 * read as holes. Generating the map in-process keeps it working offline and in CI.
 */
export function addEnvironment(scene: THREE.Scene, renderer: THREE.WebGLRenderer): THREE.Texture {
  const generator = new THREE.PMREMGenerator(renderer);
  const environment = generator.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environment = environment;
  // Low enough that it only lifts the metals out of black: at full strength it
  // also floods the white shells and flattens the emissive accents, which are
  // the parts that carry state.
  scene.environmentIntensity = 0.22;
  generator.dispose();
  return environment;
}

/**
 * Three ships only a hard-edged BoxGeometry, and hard edges are what make a rig
 * read as programmer art: every silhouette is a rectangle and no edge catches a
 * highlight. Extruding a rounded profile with a bevel gives all twelve edges a
 * radius, so the shells pick up a specular line and the shape survives being
 * lit from any direction.
 */
export function roundedBoxGeometry(
  width: number,
  height: number,
  depth: number,
  radius: number
): THREE.BufferGeometry {
  const bevel = Math.min(radius, width / 2 - 1e-3, height / 2 - 1e-3, depth / 2 - 1e-3);
  const halfWidth = width / 2 - bevel;
  const halfHeight = height / 2 - bevel;
  const corner = Math.max(1e-3, Math.min(radius, halfWidth, halfHeight));

  const shape = new THREE.Shape();
  shape.moveTo(-halfWidth + corner, -halfHeight);
  shape.lineTo(halfWidth - corner, -halfHeight);
  shape.absarc(halfWidth - corner, -halfHeight + corner, corner, -Math.PI / 2, 0, false);
  shape.lineTo(halfWidth, halfHeight - corner);
  shape.absarc(halfWidth - corner, halfHeight - corner, corner, 0, Math.PI / 2, false);
  shape.lineTo(-halfWidth + corner, halfHeight);
  shape.absarc(-halfWidth + corner, halfHeight - corner, corner, Math.PI / 2, Math.PI, false);
  shape.lineTo(-halfWidth, -halfHeight + corner);
  shape.absarc(-halfWidth + corner, -halfHeight + corner, corner, Math.PI, Math.PI * 1.5, false);

  const extrusion = Math.max(1e-3, depth - bevel * 2);
  const geometry = new THREE.ExtrudeGeometry(shape, {
    depth: extrusion,
    bevelEnabled: true,
    bevelThickness: bevel,
    bevelSize: bevel,
    bevelSegments: 3,
    curveSegments: 8
  });
  // ExtrudeGeometry starts at z=0 and grows forward; the caller positions parts
  // by their centre, matching the physics link transforms.
  geometry.translate(0, 0, -extrusion / 2);
  geometry.computeVertexNormals();
  return geometry;
}

export function standardMaterial(options: {
  color: number;
  roughness: number;
  metalness: number;
  emissive?: number;
  emissiveIntensity?: number;
  transparent?: boolean;
  opacity?: number;
}): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({
    color: options.color,
    roughness: options.roughness,
    metalness: options.metalness,
    emissive: options.emissive ?? 0x000000,
    emissiveIntensity: options.emissiveIntensity ?? 0,
    transparent: options.transparent ?? false,
    opacity: options.opacity ?? 1
  });
}

export function addMesh(
  parent: THREE.Object3D,
  geometry: THREE.BufferGeometry,
  material: THREE.Material,
  position: [number, number, number] = [0, 0, 0]
): THREE.Mesh {
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(...position);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  parent.add(mesh);
  return mesh;
}

export function materialOf(mesh: THREE.Mesh): THREE.MeshStandardMaterial {
  if (!(mesh.material instanceof THREE.MeshStandardMaterial)) {
    throw new Error("Expected a standard material");
  }
  return mesh.material;
}

export function toVector3(value: { x: number; y: number; z: number }): THREE.Vector3 {
  return new THREE.Vector3(value.x, value.y, value.z);
}

export function disposeObject(root: THREE.Object3D): void {
  const instancedMeshes = new Set<THREE.InstancedMesh>();
  const geometries = new Set<THREE.BufferGeometry>();
  const materials = new Set<THREE.Material>();
  root.traverse((object) => {
    if (object instanceof THREE.InstancedMesh) instancedMeshes.add(object);
    if (object instanceof THREE.Mesh || object instanceof THREE.Line) {
      geometries.add(object.geometry);
    } else if (!(object instanceof THREE.Sprite)) {
      return;
    }
    const material = (object as THREE.Mesh | THREE.Line | THREE.Sprite).material;
    if (Array.isArray(material)) material.forEach((entry) => materials.add(entry));
    else materials.add(material);
  });
  instancedMeshes.forEach((mesh) => mesh.dispose());
  geometries.forEach((geometry) => geometry.dispose());
  materials.forEach(disposeMaterial);
}

export function disposeMaterial(material: THREE.Material | THREE.Material[]): void {
  if (Array.isArray(material)) {
    material.forEach((entry) => entry.dispose());
    return;
  }
  if (material instanceof THREE.SpriteMaterial) material.map?.dispose();
  material.dispose();
}
