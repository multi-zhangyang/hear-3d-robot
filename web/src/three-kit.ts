import * as THREE from "three";

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
