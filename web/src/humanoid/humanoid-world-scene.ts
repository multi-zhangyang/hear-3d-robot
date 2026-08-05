import * as THREE from "three";
import { disposeMaterial, disposeObject } from "../three-kit";
import type {
  HumanoidWorldSnapshot,
  ScenarioChunkDeltaState,
  ScenarioDefinition,
  Vec3
} from "../types";
import { zoneOutline } from "../stage/scene-primitives";
import { scenarioChunkAt, visibleScenarioChunkIds } from "./chunk-visibility";
import {
  humanoidContactVisuals,
  type HumanoidContactVisualKind
} from "./contact-visual";
import { G1Rig } from "./g1-rig";
import { HumanoidTerrain } from "./humanoid-terrain";
import {
  changedScenarioVisualChunkIds,
  resolveScenarioWorldVisualState,
  type ScenarioVisualBlock,
  type ScenarioVisualObject,
  type ScenarioVisualZone,
  type ScenarioWorldVisualState
} from "./scenario-chunk-state";

interface ObjectVisual {
  mesh: THREE.Mesh;
  chunkId: string;
}

export class HumanoidWorldScene {
  readonly root = new THREE.Group();
  readonly rig: G1Rig;
  readonly #scenario: ScenarioDefinition;
  readonly #terrain: HumanoidTerrain;
  readonly #chunkGroups = new Map<string, THREE.Group>();
  readonly #objects = new Map<string, ObjectVisual>();
  readonly #residentChunkIds = new Set<string>();
  readonly #contactMarkers: THREE.Mesh[] = [];
  readonly #centerOfMass: THREE.Mesh;
  #path: THREE.Line | null = null;
  #pathKey = "";
  #visualState: ScenarioWorldVisualState;
  #scenarioChunksIdentity: string;

  static async create(
    scene: THREE.Scene,
    scenario: ScenarioDefinition,
    scenarioChunks: ScenarioChunkDeltaState,
    signal?: AbortSignal
  ): Promise<HumanoidWorldScene> {
    const rig = await G1Rig.create(signal);
    signal?.throwIfAborted();
    return new HumanoidWorldScene(scene, scenario, scenarioChunks, rig);
  }

  private constructor(
    scene: THREE.Scene,
    scenario: ScenarioDefinition,
    scenarioChunks: ScenarioChunkDeltaState,
    rig: G1Rig
  ) {
    this.#scenario = scenario;
    this.#visualState = resolveScenarioWorldVisualState(scenario, scenarioChunks);
    this.#scenarioChunksIdentity = scenarioChunksIdentity(scenarioChunks);
    this.rig = rig;
    this.root.name = "humanoid-world";
    scene.add(this.root);
    this.#terrain = new HumanoidTerrain(scenario.bounds, scenario.seed);
    this.root.add(this.#terrain.root, rig.root);
    this.#createChunkGroups();
    this.#centerOfMass = new THREE.Mesh(
      new THREE.SphereGeometry(0.035, 18, 12),
      new THREE.MeshStandardMaterial({
        color: 0xf5c86b,
        emissive: 0xd99a32,
        emissiveIntensity: 1.4,
        roughness: 0.3
      })
    );
    this.#centerOfMass.name = "balance-center";
    this.root.add(this.#centerOfMass);
    for (let index = 0; index < 32; index += 1) {
      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(0.018, 12, 8),
        new THREE.MeshBasicMaterial({ color: 0x5aebce, transparent: true, opacity: 0.9 })
      );
      marker.visible = false;
      marker.renderOrder = 3;
      this.#contactMarkers.push(marker);
      this.root.add(marker);
    }
  }

  update(snapshot: HumanoidWorldSnapshot): void {
    this.rig.update(snapshot);
    const visibleChunkIds = visibleScenarioChunkIds(
      this.#scenario,
      snapshot.robot.rootPosition
    );
    this.#updateChunkResidency(visibleChunkIds);
    this.#updateObjects(snapshot, visibleChunkIds);
    this.#updateContacts(snapshot);
    this.#centerOfMass.position.set(
      snapshot.robot.balance.centerOfMass.x,
      snapshot.robot.balance.centerOfMass.y,
      snapshot.robot.balance.centerOfMass.z
    );
    this.#centerOfMass.visible = !snapshot.robot.fallen;
    this.#updatePath(snapshot.navigation.waypoints);
  }

  updateScenarioChunks(scenarioChunks: ScenarioChunkDeltaState): void {
    if (scenarioChunks.revision < this.#visualState.revision) return;
    const identity = scenarioChunksIdentity(scenarioChunks);
    if (identity === this.#scenarioChunksIdentity) return;
    const next = resolveScenarioWorldVisualState(this.#scenario, scenarioChunks);
    const changed = changedScenarioVisualChunkIds(this.#visualState, next);
    const resident = changed.filter((chunkId) => this.#residentChunkIds.has(chunkId));
    for (const chunkId of resident) this.#evictChunk(chunkId);
    this.#visualState = next;
    this.#scenarioChunksIdentity = identity;
    for (const chunkId of resident) this.#realizeChunk(chunkId);
  }

  robotBounds(): THREE.Box3 {
    return this.rig.bounds();
  }

  worldBounds(): THREE.Box3 {
    const height = Math.max(
      2,
      ...[...this.#visualState.chunks.values()].flatMap((chunk) => (
        chunk.blocks.map((block) => block.center.y + block.size.y / 2)
      ))
    );
    return new THREE.Box3(
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(this.#scenario.bounds.width, height, this.#scenario.bounds.depth)
    ).union(this.robotBounds());
  }

  dispose(): void {
    disposeObject(this.root);
  }

  #createChunkGroups(): void {
    for (const chunk of this.#scenario.chunk_manifest.chunks) {
      const group = new THREE.Group();
      group.name = `scenario-${chunk.id}`;
      group.visible = false;
      this.#chunkGroups.set(chunk.id, group);
      this.root.add(group);
    }
  }

  #updateChunkResidency(visibleChunkIds: ReadonlySet<string>): void {
    for (const chunkId of [...this.#residentChunkIds]) {
      if (!visibleChunkIds.has(chunkId)) this.#evictChunk(chunkId);
    }
    for (const chunkId of visibleChunkIds) {
      if (!this.#residentChunkIds.has(chunkId)) {
        this.#realizeChunk(chunkId);
      }
    }
    for (const [chunkId, group] of this.#chunkGroups) {
      group.visible = visibleChunkIds.has(chunkId);
    }
    this.#terrain.update(this.#scenario.chunk_manifest.chunks.filter(({ id }) => (
      visibleChunkIds.has(id)
    )));
  }

  #realizeChunk(chunkId: string): void {
    const chunk = this.#scenario.chunk_manifest.chunks.find(({ id }) => id === chunkId);
    const contents = this.#visualState.chunks.get(chunkId);
    const group = this.#chunkGroups.get(chunkId);
    if (!chunk || !contents || !group) throw new Error(`Missing scenario chunk: ${chunkId}`);
    if (group.children.length > 0) {
      throw new Error(`Scenario chunk was not empty before realization: ${chunkId}`);
    }
    this.#residentChunkIds.add(chunkId);
    for (const block of contents.blocks) group.add(obstacleMesh(block));
    for (const zone of contents.zones) group.add(zoneMesh(zone));
    for (const descriptor of contents.objects) {
      if (!descriptor.portable) this.#createObjectVisual(descriptor, chunkId);
    }
    group.visible = true;
  }

  #evictChunk(chunkId: string): void {
    const group = this.#chunkGroups.get(chunkId);
    if (!group) throw new Error(`Missing scenario chunk group: ${chunkId}`);
    for (const [id, visual] of this.#objects) {
      if (visual.chunkId === chunkId) this.#objects.delete(id);
    }
    disposeObject(group);
    group.clear();
    group.visible = false;
    this.#residentChunkIds.delete(chunkId);
  }

  #updateObjects(
    snapshot: HumanoidWorldSnapshot,
    visibleChunkIds: ReadonlySet<string>
  ): void {
    for (const descriptor of this.#visualState.objects.values()) {
      if (!descriptor.portable) continue;
      const id = descriptor.id;
      const object = snapshot.robot.objects[id];
      const activeChunk = object
        ? scenarioChunkAt(this.#scenario, object.position)
        : undefined;
      if (!object || !activeChunk || !visibleChunkIds.has(activeChunk.id)) {
        this.#removeObjectVisual(id);
        continue;
      }
      const visual = this.#objects.get(id)
        ?? this.#createObjectVisual(descriptor, activeChunk.id);
      visual.mesh.position.set(object.position.x, object.position.y, object.position.z);
      visual.mesh.quaternion.set(
        object.rotation.x,
        object.rotation.y,
        object.rotation.z,
        object.rotation.w
      );
      updateObjectInteractionMaterial(
        visual.mesh,
        humanoidObjectInteractionState(snapshot, id)
      );
      if (activeChunk.id !== visual.chunkId) {
        const group = this.#chunkGroups.get(activeChunk.id);
        if (!group) throw new Error(`Missing visual group for ${activeChunk.id}`);
        group.add(visual.mesh);
        visual.chunkId = activeChunk.id;
      }
    }
  }

  #createObjectVisual(descriptor: ScenarioVisualObject, chunkId: string): ObjectVisual {
    const existing = this.#objects.get(descriptor.id);
    if (existing) return existing;
    const group = this.#chunkGroups.get(chunkId);
    if (!group || !this.#residentChunkIds.has(chunkId)) {
      throw new Error(`Cannot realize object outside a resident chunk: ${descriptor.id}`);
    }
    const mesh = objectMesh(descriptor);
    const visual = { mesh, chunkId };
    this.#objects.set(descriptor.id, visual);
    group.add(mesh);
    return visual;
  }

  #removeObjectVisual(id: string): void {
    const visual = this.#objects.get(id);
    if (!visual) return;
    visual.mesh.removeFromParent();
    disposeObject(visual.mesh);
    this.#objects.delete(id);
  }

  #updateContacts(snapshot: HumanoidWorldSnapshot): void {
    const contacts = humanoidContactVisuals(snapshot, this.#contactMarkers.length);
    this.#contactMarkers.forEach((marker, index) => {
      const contact = contacts[index];
      marker.visible = contact !== undefined;
      if (!contact) return;
      marker.position.set(
        contact.position.x,
        contact.position.y + 0.012,
        contact.position.z
      );
      marker.scale.setScalar(contact.scale);
      const material = marker.material as THREE.MeshBasicMaterial;
      material.color.setHex(contactColor(contact.kind));
      material.opacity = contact.kind === "body" ? 1 : 0.9;
    });
  }

  #updatePath(points: Vec3[]): void {
    const key = points.map((point) => `${point.x.toFixed(3)}:${point.z.toFixed(3)}`).join(";");
    if (key === this.#pathKey) return;
    this.#pathKey = key;
    if (this.#path) {
      this.root.remove(this.#path);
      this.#path.geometry.dispose();
      disposeMaterial(this.#path.material);
      this.#path = null;
    }
    if (points.length < 2) return;
    this.#path = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(
        points.map((point) => new THREE.Vector3(point.x, Math.max(0.035, point.y + 0.035), point.z))
      ),
      new THREE.LineBasicMaterial({ color: 0x5aebce, transparent: true, opacity: 0.78 })
    );
    this.#path.name = "authoritative-navigation-plan";
    this.root.add(this.#path);
  }
}

function contactColor(kind: HumanoidContactVisualKind): number {
  if (kind === "solid") return 0xffd166;
  if (kind === "hand") return 0xf5c86b;
  if (kind === "body") return 0xff746d;
  return 0x5aebce;
}

function obstacleMesh(obstacle: ScenarioVisualBlock): THREE.Mesh {
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(obstacle.size.x, obstacle.size.y, obstacle.size.z),
    new THREE.MeshStandardMaterial({
      color: obstacle.id.startsWith("world_boundary") ? 0x31463c : blockColor(obstacle.id),
      roughness: 0.9,
      metalness: 0.02
    })
  );
  mesh.position.set(obstacle.center.x, obstacle.center.y, obstacle.center.z);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.name = obstacle.id;
  mesh.add(new THREE.LineSegments(
    new THREE.EdgesGeometry(mesh.geometry),
    new THREE.LineBasicMaterial({ color: 0x9ab7a5, transparent: true, opacity: 0.38 })
  ));
  return mesh;
}

function zoneMesh(zone: ScenarioVisualZone): THREE.Mesh {
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(zone.size.x, 0.018, zone.size.z),
    new THREE.MeshStandardMaterial({
      color: zone.color,
      emissive: zone.color,
      emissiveIntensity: 0.85,
      transparent: true,
      opacity: 0.34,
      roughness: 0.5,
      depthWrite: false
    })
  );
  mesh.position.set(zone.center.x, 0.012, zone.center.z);
  mesh.add(zoneOutline(zone.size));
  mesh.name = zone.id;
  return mesh;
}

function objectMesh(object: ScenarioVisualObject): THREE.Mesh {
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(object.size.x, object.size.y, object.size.z, 2, 2, 2),
    new THREE.MeshStandardMaterial({
      color: object.color,
      roughness: 0.48,
      metalness: 0.12
    })
  );
  mesh.position.set(object.position.x, object.position.y, object.position.z);
  mesh.quaternion.set(
    object.rotation.x,
    object.rotation.y,
    object.rotation.z,
    object.rotation.w
  );
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.name = object.id;
  mesh.add(new THREE.LineSegments(
    new THREE.EdgesGeometry(mesh.geometry),
    new THREE.LineBasicMaterial({
      color: object.color,
      transparent: true,
      opacity: 0.72
    })
  ));
  return mesh;
}

export type HumanoidObjectInteractionState = "idle" | "contact" | "verified";

export function humanoidObjectInteractionState(
  snapshot: HumanoidWorldSnapshot,
  objectId: string
): HumanoidObjectInteractionState {
  const assessments = snapshot.grasp?.assessments?.filter((assessment) => (
    assessment.frame === snapshot.frame && assessment.object_id === objectId
  )) ?? [];
  if (assessments.some((assessment) => assessment.grasp_verified)) return "verified";
  return assessments.some((assessment) => assessment.evidence.contact.status !== "missing")
    ? "contact"
    : "idle";
}

function updateObjectInteractionMaterial(
  mesh: THREE.Mesh,
  state: HumanoidObjectInteractionState
): void {
  const material = mesh.material;
  if (!(material instanceof THREE.MeshStandardMaterial)) return;
  if (state === "verified") {
    material.emissive.setHex(0x42d9b8);
    material.emissiveIntensity = 0.8;
    return;
  }
  if (state === "contact") {
    material.emissive.setHex(0xd9ad50);
    material.emissiveIntensity = 0.55;
    return;
  }
  material.emissive.setHex(0x000000);
  material.emissiveIntensity = 0;
}

function blockColor(id: string): number {
  let hash = 2166136261;
  for (const character of id) {
    hash ^= character.codePointAt(0) ?? 0;
    hash = Math.imul(hash, 16777619);
  }
  const palette = [0x66786a, 0x806e55, 0x586f77, 0x71805f, 0x69627b];
  return palette[Math.abs(hash) % palette.length]!;
}

function scenarioChunksIdentity(state: ScenarioChunkDeltaState): string {
  return `${state.revision}:${state.scenario_sha256}:${JSON.stringify(state.chunks)}`;
}
