import * as THREE from "three";

const TILE_SIZE = 1;
const TILE_HEIGHT = 0.18;
const TILE_GAP = 0.035;
const TILE_COLORS = [0x344b40, 0x3b5145, 0x30453d, 0x405548, 0x2f4140] as const;

export function createHumanoidTerrain(
  bounds: { width: number; depth: number },
  seed: number
): THREE.Group {
  const root = new THREE.Group();
  root.name = "humanoid-voxel-terrain";

  const columns = Math.ceil(bounds.width / TILE_SIZE);
  const rows = Math.ceil(bounds.depth / TILE_SIZE);
  const geometry = new THREE.BoxGeometry(1, 1, 1);
  const material = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    roughness: 0.9,
    metalness: 0.02,
    vertexColors: true
  });
  const tiles = new THREE.InstancedMesh(geometry, material, columns * rows);
  tiles.name = "humanoid-terrain-tiles";
  tiles.receiveShadow = true;
  const matrix = new THREE.Matrix4();
  let index = 0;
  for (let z = 0; z < rows; z += 1) {
    for (let x = 0; x < columns; x += 1) {
      const width = Math.min(TILE_SIZE, bounds.width - x * TILE_SIZE);
      const depth = Math.min(TILE_SIZE, bounds.depth - z * TILE_SIZE);
      matrix.compose(
        new THREE.Vector3(x * TILE_SIZE + width / 2, -TILE_HEIGHT / 2, z * TILE_SIZE + depth / 2),
        new THREE.Quaternion(),
        new THREE.Vector3(
          Math.max(0.02, width - TILE_GAP),
          TILE_HEIGHT,
          Math.max(0.02, depth - TILE_GAP)
        )
      );
      tiles.setMatrixAt(index, matrix);
      tiles.setColorAt(index, new THREE.Color(TILE_COLORS[tileVariant(seed, x, z)]!));
      index += 1;
    }
  }
  tiles.count = index;
  tiles.instanceMatrix.needsUpdate = true;
  if (tiles.instanceColor) tiles.instanceColor.needsUpdate = true;

  const foundation = new THREE.Mesh(
    new THREE.BoxGeometry(bounds.width + 0.16, 0.28, bounds.depth + 0.16),
    new THREE.MeshStandardMaterial({ color: 0x1c2a28, roughness: 0.96, metalness: 0 })
  );
  foundation.name = "humanoid-terrain-foundation";
  foundation.position.set(bounds.width / 2, -TILE_HEIGHT - 0.14, bounds.depth / 2);
  foundation.receiveShadow = true;

  const horizon = new THREE.Mesh(
    new THREE.CircleGeometry(Math.max(bounds.width, bounds.depth) * 2.4, 64),
    new THREE.MeshBasicMaterial({ color: 0x111a20 })
  );
  horizon.name = "humanoid-terrain-horizon";
  horizon.rotation.x = -Math.PI / 2;
  horizon.position.set(bounds.width / 2, -TILE_HEIGHT - 0.3, bounds.depth / 2);

  const border = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(bounds.width + 0.08, TILE_HEIGHT, bounds.depth + 0.08)),
    new THREE.LineBasicMaterial({ color: 0x68c7aa, transparent: true, opacity: 0.34 })
  );
  border.name = "humanoid-terrain-border";
  border.position.set(bounds.width / 2, -TILE_HEIGHT / 2, bounds.depth / 2);

  root.userData.tileCount = index;
  root.userData.surfaceY = 0;
  root.add(horizon, foundation, tiles, border);
  return root;
}

function tileVariant(seed: number, x: number, z: number): number {
  let value = (seed ^ Math.imul(x + 17, 73_856_093) ^ Math.imul(z + 29, 19_349_663)) >>> 0;
  value ^= value >>> 13;
  value = Math.imul(value, 1_274_126_177) >>> 0;
  return value % TILE_COLORS.length;
}
