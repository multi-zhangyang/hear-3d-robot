import * as THREE from "three";
import { disposeObject } from "../three-kit";

const TILE_SIZE = 1;
const TILE_HEIGHT = 0.1;
const TILE_GAP = 0.012;
const TILE_COLORS = [0x151515, 0x181818, 0x121212, 0x1b1b1b, 0x161616] as const;

export interface HumanoidTerrainChunk {
  id: string;
  bounds: {
    minimum: { x: number; z: number };
    maximum: { x: number; z: number };
  };
}

export class HumanoidTerrain {
  readonly root = new THREE.Group();
  readonly #seed: number;
  readonly #chunks = new Map<string, THREE.InstancedMesh>();

  constructor(bounds: { width: number; depth: number }, seed: number) {
    if (!Number.isFinite(bounds.width) || bounds.width <= 0
      || !Number.isFinite(bounds.depth) || bounds.depth <= 0) {
      throw new Error("Humanoid terrain bounds must be positive and finite");
    }
    if (!Number.isSafeInteger(seed)) {
      throw new Error("Humanoid terrain seed must be a safe integer");
    }
    this.#seed = seed;
    this.root.name = "humanoid-voxel-terrain";
    this.root.add(
      terrainHorizon(bounds),
      terrainFoundation(bounds),
      terrainBorder(bounds),
      terrainMajorGrid(bounds),
      terrainPerimeterBeacons(bounds)
    );
    this.#updateTileCount();
  }

  update(residentChunks: readonly HumanoidTerrainChunk[]): void {
    residentChunks.forEach(assertTerrainChunk);
    const nextIds = new Set(residentChunks.map(({ id }) => id));
    if (nextIds.size !== residentChunks.length) {
      throw new Error("Humanoid terrain chunks must have unique identifiers");
    }
    for (const [id, mesh] of this.#chunks) {
      if (nextIds.has(id)) continue;
      mesh.removeFromParent();
      disposeObject(mesh);
      this.#chunks.delete(id);
    }
    for (const chunk of residentChunks) {
      if (this.#chunks.has(chunk.id)) continue;
      const mesh = terrainChunkMesh(chunk, this.#seed);
      this.#chunks.set(chunk.id, mesh);
      this.root.add(mesh);
    }
    this.#updateTileCount();
  }

  #updateTileCount(): void {
    this.root.userData.tileCount = [...this.#chunks.values()].reduce(
      (total, mesh) => total + mesh.count,
      0
    );
    this.root.userData.residentChunkCount = this.#chunks.size;
    this.root.userData.surfaceY = 0;
  }
}

function terrainChunkMesh(
  chunk: HumanoidTerrainChunk,
  seed: number
): THREE.InstancedMesh {
  const { minimum, maximum } = chunk.bounds;
  const firstColumn = Math.floor(minimum.x / TILE_SIZE);
  const lastColumn = Math.ceil(maximum.x / TILE_SIZE) - 1;
  const firstRow = Math.floor(minimum.z / TILE_SIZE);
  const lastRow = Math.ceil(maximum.z / TILE_SIZE) - 1;
  const capacity = (lastColumn - firstColumn + 1) * (lastRow - firstRow + 1);
  const mesh = new THREE.InstancedMesh(
    new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshStandardMaterial({
      color: 0xffffff,
      roughness: 0.9,
      metalness: 0.02,
      vertexColors: true
    }),
    capacity
  );
  mesh.name = `humanoid-terrain-${chunk.id}`;
  mesh.receiveShadow = true;
  const matrix = new THREE.Matrix4();
  let index = 0;
  for (let row = firstRow; row <= lastRow; row += 1) {
    for (let column = firstColumn; column <= lastColumn; column += 1) {
      const tileMinimumX = Math.max(minimum.x, column * TILE_SIZE);
      const tileMaximumX = Math.min(maximum.x, (column + 1) * TILE_SIZE);
      const tileMinimumZ = Math.max(minimum.z, row * TILE_SIZE);
      const tileMaximumZ = Math.min(maximum.z, (row + 1) * TILE_SIZE);
      const width = tileMaximumX - tileMinimumX;
      const depth = tileMaximumZ - tileMinimumZ;
      if (width <= 0 || depth <= 0) continue;
      matrix.compose(
        new THREE.Vector3(
          tileMinimumX + width / 2,
          -TILE_HEIGHT / 2,
          tileMinimumZ + depth / 2
        ),
        new THREE.Quaternion(),
        new THREE.Vector3(
          Math.max(0.02, width - TILE_GAP),
          TILE_HEIGHT,
          Math.max(0.02, depth - TILE_GAP)
        )
      );
      mesh.setMatrixAt(index, matrix);
      mesh.setColorAt(index, new THREE.Color(TILE_COLORS[
        tileVariant(seed, column, row)
      ]!));
      index += 1;
    }
  }
  mesh.count = index;
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  return mesh;
}

function terrainFoundation(bounds: { width: number; depth: number }): THREE.Mesh {
  const foundation = new THREE.Mesh(
    new THREE.BoxGeometry(bounds.width + 0.16, 0.28, bounds.depth + 0.16),
    new THREE.MeshStandardMaterial({ color: 0x080808, roughness: 0.98, metalness: 0 })
  );
  foundation.name = "humanoid-terrain-foundation";
  foundation.position.set(bounds.width / 2, -TILE_HEIGHT - 0.14, bounds.depth / 2);
  foundation.receiveShadow = true;
  return foundation;
}

function terrainHorizon(bounds: { width: number; depth: number }): THREE.Mesh {
  const horizon = new THREE.Mesh(
    new THREE.CircleGeometry(Math.max(bounds.width, bounds.depth) * 2.4, 64),
    new THREE.MeshBasicMaterial({ color: 0x020202 })
  );
  horizon.name = "humanoid-terrain-horizon";
  horizon.rotation.x = -Math.PI / 2;
  horizon.position.set(bounds.width / 2, -TILE_HEIGHT - 0.3, bounds.depth / 2);
  return horizon;
}

function terrainBorder(bounds: { width: number; depth: number }): THREE.LineSegments {
  const border = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(
      bounds.width + 0.08,
      TILE_HEIGHT,
      bounds.depth + 0.08
    )),
    new THREE.LineBasicMaterial({ color: 0xd4d4d4, transparent: true, opacity: 0.36 })
  );
  border.name = "humanoid-terrain-border";
  border.position.set(bounds.width / 2, -TILE_HEIGHT / 2, bounds.depth / 2);
  return border;
}

function terrainMajorGrid(bounds: { width: number; depth: number }): THREE.LineSegments {
  const points: THREE.Vector3[] = [];
  for (let x = 0; x <= bounds.width; x += 5) {
    points.push(new THREE.Vector3(x, 0.006, 0), new THREE.Vector3(x, 0.006, bounds.depth));
  }
  for (let z = 0; z <= bounds.depth; z += 5) {
    points.push(new THREE.Vector3(0, 0.006, z), new THREE.Vector3(bounds.width, 0.006, z));
  }
  const grid = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({
      color: 0x8a8a8a,
      transparent: true,
      opacity: 0.13,
      depthWrite: false
    })
  );
  grid.name = "humanoid-terrain-five-meter-grid";
  grid.renderOrder = 1;
  return grid;
}

function terrainPerimeterBeacons(bounds: { width: number; depth: number }): THREE.Group {
  const group = new THREE.Group();
  group.name = "humanoid-visual-perimeter-beacons";
  const poleGeometry = new THREE.CylinderGeometry(0.018, 0.025, 0.72, 8);
  const poleMaterial = new THREE.MeshStandardMaterial({
    color: 0x1c1c1c,
    roughness: 0.76,
    metalness: 0.34
  });
  const lightGeometry = new THREE.CylinderGeometry(0.045, 0.045, 0.055, 12);
  const lightMaterial = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    emissive: 0xb8b8b8,
    emissiveIntensity: 1.5,
    roughness: 0.3,
    metalness: 0.12
  });
  for (const [x, z] of [
    [-0.16, -0.16],
    [bounds.width + 0.16, -0.16],
    [-0.16, bounds.depth + 0.16],
    [bounds.width + 0.16, bounds.depth + 0.16]
  ] as const) {
    const marker = new THREE.Group();
    const pole = new THREE.Mesh(poleGeometry, poleMaterial);
    pole.position.y = 0.36;
    pole.castShadow = true;
    const light = new THREE.Mesh(lightGeometry, lightMaterial);
    light.position.y = 0.735;
    marker.position.set(x, 0, z);
    marker.add(pole, light);
    group.add(marker);
  }
  return group;
}

function finitePoint(point: { x: number; z: number }): boolean {
  return Number.isFinite(point.x) && Number.isFinite(point.z);
}

function assertTerrainChunk(chunk: HumanoidTerrainChunk): void {
  const { minimum, maximum } = chunk.bounds;
  if (chunk.id.trim().length === 0 || !finitePoint(minimum) || !finitePoint(maximum)
    || minimum.x >= maximum.x || minimum.z >= maximum.z) {
    throw new Error(`Invalid humanoid terrain chunk bounds: ${chunk.id}`);
  }
}

function tileVariant(seed: number, x: number, z: number): number {
  let value = (seed ^ Math.imul(x + 17, 73_856_093) ^ Math.imul(z + 29, 19_349_663)) >>> 0;
  value ^= value >>> 13;
  value = Math.imul(value, 1_274_126_177) >>> 0;
  return value % TILE_COLORS.length;
}
