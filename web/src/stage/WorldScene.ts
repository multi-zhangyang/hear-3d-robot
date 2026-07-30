import * as THREE from "three";
import { focusedCommand } from "../active-commands";
import { RobotRig } from "../robot-rig";
import { disposeMaterial, disposeObject, materialOf, toVector3 } from "../three-kit";
import type { ScenarioDefinition, Vec3, WorldSnapshot } from "../types";
import { VoxelTerrain } from "../voxel-terrain";
import { entityLabel } from "../ui-text";
import { PlanOverlay } from "./PlanOverlay";
import { addEntityLabel, createFloor, createFocusMarker, zoneOutline } from "./scene-primitives";
import { markSelectableEntity, WorldPicker } from "./WorldPicker";
import type { WorldSelection } from "./world-selection";

interface SizedMesh {
  mesh: THREE.Mesh;
  sizeKey: string;
}

const ZONE_LABEL_HEIGHT = 2.05;
const ZONE_LABEL_SCREEN_LIFT = 1.5;
const OBJECT_LABEL_SCREEN_LIFT = 0.35;

export class WorldScene {
  readonly #bounds: { width: number; depth: number };
  readonly #terrain: VoxelTerrain | null;
  readonly #terrainHeight: number;
  readonly #zones = new Map<string, SizedMesh>();
  readonly #obstacles = new Map<string, SizedMesh>();
  readonly #objects = new Map<string, SizedMesh>();
  readonly #root = new THREE.Group();
  readonly #robot = new RobotRig();
  readonly #robotPickProxy = createRobotPickProxy();
  readonly #focusMarker = createFocusMarker();
  readonly #intent = new PlanOverlay();
  readonly #picker: WorldPicker;
  #labelsVisible = true;
  #pathLine: THREE.Line | null = null;
  #pathKey = "";

  constructor(
    scene: THREE.Scene,
    scenario: ScenarioDefinition,
    onSelection: (selection: WorldSelection | null) => void
  ) {
    this.#bounds = scenario.bounds;
    this.#terrain = scenario.terrain ? new VoxelTerrain(scenario.terrain, scenario.seed) : null;
    this.#terrainHeight = scenario.terrain
      ? Math.max(0, ...scenario.terrain.heights) * scenario.terrain.block
      : 0;
    scene.add(createFloor(scenario.bounds, scenario.terrain?.block));
    scene.add(this.#root);
    this.#root.add(this.#robot.root, this.#robotPickProxy, this.#focusMarker, this.#intent.root);
    if (this.#terrain) this.#root.add(this.#terrain.root);
    markSelectableEntity(this.#robotPickProxy, "robot", "robot");
    this.#picker = new WorldPicker(
      this.#root,
      this.#terrain,
      (selection) => this.#resolveEntity(selection),
      onSelection
    );
  }

  update(snapshot: WorldSnapshot): void {
    this.#terrain?.update(snapshot.voxels, snapshot.explored.cells);
    this.#syncZones(snapshot);
    this.#syncObstacles(snapshot);
    this.updatePose(snapshot);
    this.#syncFocus(snapshot);
    this.#syncActualPath(snapshot.navigation.actual_path);
    this.#intent.update(snapshot);
    this.setLabelsVisible(this.#labelsVisible);
  }

  /** Updates only transforms that are safe to interpolate between real frames. */
  updatePose(snapshot: WorldSnapshot): void {
    this.#robot.update(snapshot);
    synchronizeRobotPickProxy(this.#robotPickProxy, snapshot);
    this.#syncObjects(snapshot);
    this.#picker.refresh();
  }

  pick(camera: THREE.Camera, pointer: THREE.Vector2): WorldSelection | null {
    return this.#picker.pickFrom(camera, pointer, [
      this.#robotPickProxy,
      ...Array.from(this.#objects.values(), (entry) => entry.mesh).filter((mesh) => mesh.visible),
      ...Array.from(this.#obstacles.values(), (entry) => entry.mesh).filter((mesh) => mesh.visible),
      ...Array.from(this.#zones.values(), (entry) => entry.mesh).filter((mesh) => mesh.visible),
      ...(this.#terrain?.pickables() ?? [])
    ]);
  }

  clearSelection(): void {
    this.#picker.clear();
  }

  setLabelsVisible(visible: boolean): void {
    this.#labelsVisible = visible;
    this.#root.traverse((object) => {
      if (object.name === "entity-label") object.visible = visible;
    });
  }

  robotBounds(): THREE.Box3 {
    return new THREE.Box3().setFromObject(this.#robot.root);
  }

  worldBounds(snapshot: WorldSnapshot): THREE.Box3 {
    const box = this.robotBounds();
    const maximumHeight = Math.max(
      1.4,
      this.#terrainHeight,
      ...snapshot.obstacles.map((obstacle) => obstacle.center.y + obstacle.size.y / 2),
      ...snapshot.objects
        .filter((object) => object.enabled)
        .map((object) => object.position.y + object.size.y / 2)
    );
    box.expandByPoint(new THREE.Vector3(0, 0, 0));
    box.expandByPoint(new THREE.Vector3(this.#bounds.width, maximumHeight, this.#bounds.depth));
    return box;
  }

  #syncZones(snapshot: WorldSnapshot): void {
    const present = new Set<string>();
    for (const zone of snapshot.zones) {
      present.add(zone.id);
      const entry = ensureSizedMesh(this.#zones, zone.id, zone.size, () => {
        const mesh = new THREE.Mesh(
          new THREE.BoxGeometry(zone.size.x, 0.012, zone.size.z),
          new THREE.MeshStandardMaterial({
            color: zone.color,
            emissive: zone.color,
            emissiveIntensity: 0.55,
            transparent: true,
            opacity: 0.32,
            roughness: 0.5,
            depthWrite: false
          })
        );
        mesh.receiveShadow = true;
        mesh.add(zoneOutline(zone.size));
        addEntityLabel(
          mesh,
          ZONE_LABEL_HEIGHT,
          entityLabel(zone.id),
          Math.max(zone.size.x, zone.size.z) / 2 + 0.3,
          ZONE_LABEL_SCREEN_LIFT
        );
        markSelectableEntity(mesh, "zone", zone.id);
        this.#root.add(mesh);
        return mesh;
      });
      entry.mesh.position.set(zone.center.x, 0.008, zone.center.z);
      const material = materialOf(entry.mesh);
      material.color.set(zone.color);
      material.emissive.set(zone.color);
    }
    removeMissing(this.#zones, present, this.#root);
  }

  #syncObstacles(snapshot: WorldSnapshot): void {
    const present = new Set<string>();
    for (const obstacle of snapshot.obstacles) {
      present.add(obstacle.id);
      const entry = ensureSizedMesh(this.#obstacles, obstacle.id, obstacle.size, () => {
        const mesh = new THREE.Mesh(
          new THREE.BoxGeometry(obstacle.size.x, obstacle.size.y, obstacle.size.z),
          new THREE.MeshStandardMaterial({ color: 0x2a3243, roughness: 0.86, metalness: 0.08 })
        );
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        const edges = new THREE.LineSegments(
          new THREE.EdgesGeometry(mesh.geometry),
          new THREE.LineBasicMaterial({ color: 0x4d5b76 })
        );
        mesh.add(edges);
        addEntityLabel(mesh, obstacle.size.y / 2 + 0.16, entityLabel(obstacle.id));
        markSelectableEntity(mesh, "obstacle", obstacle.id);
        this.#root.add(mesh);
        return mesh;
      });
      entry.mesh.position.set(obstacle.center.x, obstacle.center.y, obstacle.center.z);
    }
    removeMissing(this.#obstacles, present, this.#root);
  }

  #syncObjects(snapshot: WorldSnapshot): void {
    const present = new Set<string>();
    const attachedId = snapshot.robot.attachment?.object_id ?? null;
    for (const object of snapshot.objects) {
      present.add(object.id);
      const entry = ensureSizedMesh(this.#objects, object.id, object.size, () => {
        const mesh = new THREE.Mesh(
          new THREE.BoxGeometry(object.size.x, object.size.y, object.size.z),
          new THREE.MeshStandardMaterial({
            color: object.color,
            roughness: object.kind === "key" ? 0.22 : 0.44,
            metalness: object.kind === "key" ? 0.85 : 0.12
          })
        );
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        addEntityLabel(
          mesh,
          object.size.y / 2 + 0.14,
          entityLabel(object.id),
          0,
          OBJECT_LABEL_SCREEN_LIFT
        );
        markSelectableEntity(mesh, "object", object.id);
        this.#root.add(mesh);
        return mesh;
      });
      entry.mesh.visible = object.enabled;
      entry.mesh.position.set(object.position.x, object.position.y, object.position.z);
      entry.mesh.quaternion.set(object.rotation.x, object.rotation.y, object.rotation.z, object.rotation.w);
      const material = materialOf(entry.mesh);
      material.color.set(object.color);
      material.emissive.set(object.id === attachedId ? object.color : 0x000000);
      material.emissiveIntensity = object.id === attachedId ? 0.85 : 0;
    }
    removeMissing(this.#objects, present, this.#root);
  }

  #syncFocus(snapshot: WorldSnapshot): void {
    const focus = focusedCommand(snapshot)?.focus?.position;
    this.#focusMarker.visible = focus !== undefined;
    if (focus) this.#focusMarker.position.copy(toVector3(focus));
  }

  #syncActualPath(points: Vec3[] | undefined): void {
    const key = points?.map((point) => `${point.x},${point.y},${point.z}`).join(";") ?? "";
    if (key === this.#pathKey) return;
    this.#pathKey = key;
    if (this.#pathLine) {
      this.#root.remove(this.#pathLine);
      this.#pathLine.geometry.dispose();
      disposeMaterial(this.#pathLine.material);
      this.#pathLine = null;
    }
    if (!points || points.length < 2) return;
    const geometry = new THREE.BufferGeometry().setFromPoints(
      points.map((point) => new THREE.Vector3(point.x, 0.02, point.z))
    );
    const material = new THREE.LineBasicMaterial({ color: 0x35e0c4, transparent: true, opacity: 0.55 });
    this.#pathLine = new THREE.Line(geometry, material);
    this.#root.add(this.#pathLine);
  }

  #resolveEntity(
    selection: Extract<WorldSelection, { kind: "entity" }>
  ): THREE.Object3D | null {
    if (selection.entityType === "robot") return this.#robotPickProxy;
    if (selection.entityType === "object") return this.#objects.get(selection.id)?.mesh ?? null;
    if (selection.entityType === "obstacle") return this.#obstacles.get(selection.id)?.mesh ?? null;
    return this.#zones.get(selection.id)?.mesh ?? null;
  }
}

function createRobotPickProxy(): THREE.Mesh {
  const proxy = new THREE.Mesh(
    new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: false })
  );
  proxy.name = "robot-selection-volume";
  proxy.renderOrder = -1;
  return proxy;
}

function synchronizeRobotPickProxy(proxy: THREE.Mesh, snapshot: WorldSnapshot): void {
  const bounds = new THREE.Box3();
  for (const link of Object.values(snapshot.robot.links)) {
    bounds.expandByPoint(new THREE.Vector3(link.position.x, link.position.y, link.position.z));
  }
  if (bounds.isEmpty()) {
    bounds.expandByPoint(new THREE.Vector3(
      snapshot.robot.position.x,
      snapshot.robot.position.y,
      snapshot.robot.position.z
    ));
  }
  bounds.expandByScalar(0.28);
  bounds.getCenter(proxy.position);
  bounds.getSize(proxy.scale);
  proxy.updateMatrix();
}

function ensureSizedMesh(
  entries: Map<string, SizedMesh>,
  id: string,
  size: Vec3,
  create: () => THREE.Mesh
): SizedMesh {
  const sizeKey = `${size.x}:${size.y}:${size.z}`;
  const existing = entries.get(id);
  if (existing && existing.sizeKey === sizeKey) return existing;
  if (existing) {
    existing.mesh.removeFromParent();
    disposeObject(existing.mesh);
  }
  const entry = { mesh: create(), sizeKey };
  entries.set(id, entry);
  return entry;
}

function removeMissing(entries: Map<string, SizedMesh>, present: Set<string>, root: THREE.Group): void {
  for (const [id, entry] of entries) {
    if (present.has(id)) continue;
    root.remove(entry.mesh);
    disposeObject(entry.mesh);
    entries.delete(id);
  }
}
