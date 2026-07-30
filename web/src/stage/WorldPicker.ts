import * as THREE from "three";
import type { VoxelTerrain } from "../voxel-terrain";
import {
  sameWorldSelection,
  type WorldEntityType,
  type WorldSelection
} from "./world-selection";

interface SelectableEntity {
  type: WorldEntityType;
  id: string;
}

interface SelectionResolution {
  selection: WorldSelection;
  box: THREE.Box3;
}

type EntityResolver = (
  selection: Extract<WorldSelection, { kind: "entity" }>
) => THREE.Object3D | null;

const ENTITY_DATA_KEY = "hearWorldEntity";

export function markSelectableEntity(
  object: THREE.Object3D,
  type: WorldEntityType,
  id: string
): void {
  object.userData[ENTITY_DATA_KEY] = { type, id } satisfies SelectableEntity;
}

export class WorldPicker {
  readonly #root: THREE.Object3D;
  readonly #terrain: VoxelTerrain | null;
  readonly #resolveEntity: EntityResolver;
  readonly #onSelection: (selection: WorldSelection | null) => void;
  readonly #raycaster = new THREE.Raycaster();
  readonly #selectionBox = new THREE.Box3();
  readonly #outline = new THREE.Box3Helper(this.#selectionBox, 0xf5d477);
  #selection: WorldSelection | null = null;

  constructor(
    root: THREE.Object3D,
    terrain: VoxelTerrain | null,
    resolveEntity: EntityResolver,
    onSelection: (selection: WorldSelection | null) => void
  ) {
    this.#root = root;
    this.#terrain = terrain;
    this.#resolveEntity = resolveEntity;
    this.#onSelection = onSelection;
    this.#outline.name = "world-selection-outline";
    this.#outline.visible = false;
    this.#outline.renderOrder = 80;
    const material = this.#outline.material as THREE.LineBasicMaterial;
    material.depthTest = false;
    material.transparent = true;
    material.opacity = 0.95;
    root.add(this.#outline);
  }

  pickFrom(
    camera: THREE.Camera,
    normalizedPointer: THREE.Vector2,
    objects: THREE.Object3D[]
  ): WorldSelection | null {
    this.#root.updateWorldMatrix(true, true);
    camera.updateWorldMatrix(true, false);
    this.#raycaster.setFromCamera(normalizedPointer, camera);
    const intersections = this.#raycaster.intersectObjects(objects, false);
    for (const intersection of intersections) {
      const voxel = this.#terrain?.resolveIntersection(intersection);
      if (voxel) {
        this.#set({ selection: voxel.selection, box: voxel.box });
        return voxel.selection;
      }
      const entity = selectableEntityFrom(intersection.object);
      if (!entity) continue;
      const selection: Extract<WorldSelection, { kind: "entity" }> = {
        kind: "entity",
        entityType: entity.data.type,
        id: entity.data.id
      };
      this.#set({ selection, box: boundsFor(entity.object) });
      return selection;
    }
    this.clear();
    return null;
  }

  refresh(): void {
    if (!this.#selection) return;
    if (this.#selection.kind === "voxel") {
      const voxel = this.#terrain?.resolveSelection(this.#selection);
      this.#set(voxel ? { selection: voxel.selection, box: voxel.box } : null);
      return;
    }
    const object = this.#resolveEntity(this.#selection);
    this.#set(object ? { selection: this.#selection, box: boundsFor(object) } : null);
  }

  clear(): void {
    this.#set(null);
  }

  #set(resolution: SelectionResolution | null): void {
    const next = resolution?.selection ?? null;
    if (resolution && !resolution.box.isEmpty()) {
      this.#selectionBox.copy(resolution.box).expandByScalar(0.025);
      this.#outline.visible = true;
    } else {
      this.#outline.visible = false;
    }
    if (sameWorldSelection(this.#selection, next)) return;
    this.#selection = next;
    this.#onSelection(next);
  }
}

function selectableEntityFrom(object: THREE.Object3D): {
  object: THREE.Object3D;
  data: SelectableEntity;
} | null {
  let cursor: THREE.Object3D | null = object;
  while (cursor) {
    const data = cursor.userData[ENTITY_DATA_KEY] as SelectableEntity | undefined;
    if (data) return { object: cursor, data };
    cursor = cursor.parent;
  }
  return null;
}

function boundsFor(object: THREE.Object3D): THREE.Box3 {
  object.updateWorldMatrix(true, true);
  if (object instanceof THREE.Mesh) {
    object.geometry.computeBoundingBox();
    const local = object.geometry.boundingBox;
    if (local) return local.clone().applyMatrix4(object.matrixWorld);
  }
  return new THREE.Box3().setFromObject(object);
}
