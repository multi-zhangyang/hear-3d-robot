import * as THREE from "three";
import { disposeMaterial, disposeObject } from "../three-kit";
import type { HumanoidWorldSnapshot, ScenarioDefinition, Vec3 } from "../types";
import { zoneOutline } from "../stage/scene-primitives";
import { G1Rig } from "./g1-rig";
import { createHumanoidTerrain } from "./humanoid-terrain";

interface ObjectVisual {
  mesh: THREE.Mesh;
  portable: boolean;
}

export class HumanoidWorldScene {
  readonly root = new THREE.Group();
  readonly rig: G1Rig;
  readonly #scenario: ScenarioDefinition;
  readonly #objects = new Map<string, ObjectVisual>();
  readonly #contactMarkers: THREE.Mesh[] = [];
  readonly #centerOfMass: THREE.Mesh;
  #path: THREE.Line | null = null;
  #pathKey = "";

  static async create(
    scene: THREE.Scene,
    scenario: ScenarioDefinition,
    signal?: AbortSignal
  ): Promise<HumanoidWorldScene> {
    const rig = await G1Rig.create(signal);
    signal?.throwIfAborted();
    return new HumanoidWorldScene(scene, scenario, rig);
  }

  private constructor(scene: THREE.Scene, scenario: ScenarioDefinition, rig: G1Rig) {
    this.#scenario = scenario;
    this.rig = rig;
    this.root.name = "humanoid-world";
    scene.add(this.root);
    this.root.add(createHumanoidTerrain(scenario.bounds, scenario.seed), rig.root);
    this.#addObstacles();
    this.#addZones();
    this.#addObjects();
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
    for (let index = 0; index < 20; index += 1) {
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
    this.#updateObjects(snapshot);
    this.#updateContacts(snapshot);
    this.#centerOfMass.position.set(
      snapshot.robot.balance.centerOfMass.x,
      snapshot.robot.balance.centerOfMass.y,
      snapshot.robot.balance.centerOfMass.z
    );
    this.#centerOfMass.visible = !snapshot.robot.fallen;
    this.#updatePath(snapshot.navigation.waypoints);
  }

  robotBounds(): THREE.Box3 {
    return this.rig.bounds();
  }

  worldBounds(): THREE.Box3 {
    const height = Math.max(
      2,
      ...this.#scenario.obstacles.map((obstacle) => obstacle.center.y + obstacle.size.y / 2)
    );
    return new THREE.Box3(
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(this.#scenario.bounds.width, height, this.#scenario.bounds.depth)
    ).union(this.robotBounds());
  }

  dispose(): void {
    disposeObject(this.root);
  }

  #addObstacles(): void {
    for (const obstacle of this.#scenario.obstacles) {
      const material = new THREE.MeshStandardMaterial({
        color: obstacle.id.startsWith("world_boundary") ? 0x31463c : blockColor(obstacle.id),
        roughness: 0.9,
        metalness: 0.02
      });
      const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(obstacle.size.x, obstacle.size.y, obstacle.size.z),
        material
      );
      mesh.position.set(obstacle.center.x, obstacle.center.y, obstacle.center.z);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.name = obstacle.id;
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(mesh.geometry),
        new THREE.LineBasicMaterial({ color: 0x9ab7a5, transparent: true, opacity: 0.38 })
      );
      mesh.add(edges);
      this.root.add(mesh);
    }
  }

  #addZones(): void {
    for (const zone of this.#scenario.zones) {
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
      this.root.add(mesh);
    }
  }

  #addObjects(): void {
    for (const object of this.#scenario.objects) {
      const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(object.size.x, object.size.y, object.size.z, 2, 2, 2),
        new THREE.MeshStandardMaterial({
          color: object.color,
          roughness: 0.48,
          metalness: 0.12
        })
      );
      mesh.position.set(object.position.x, object.position.y, object.position.z);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.name = object.id;
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(mesh.geometry),
        new THREE.LineBasicMaterial({ color: object.color, transparent: true, opacity: 0.72 })
      );
      mesh.add(edges);
      this.#objects.set(object.id, { mesh, portable: object.portable });
      this.root.add(mesh);
    }
  }

  #updateObjects(snapshot: HumanoidWorldSnapshot): void {
    for (const [id, visual] of this.#objects) {
      if (!visual.portable) continue;
      const object = snapshot.robot.objects[id];
      visual.mesh.visible = object !== undefined;
      if (!object) continue;
      visual.mesh.position.set(object.position.x, object.position.y, object.position.z);
      visual.mesh.quaternion.set(
        object.rotation.x,
        object.rotation.y,
        object.rotation.z,
        object.rotation.w
      );
    }
  }

  #updateContacts(snapshot: HumanoidWorldSnapshot): void {
    const points = [
      ...snapshot.robot.feet.left.points,
      ...snapshot.robot.feet.right.points
    ];
    this.#contactMarkers.forEach((marker, index) => {
      const point = points[index];
      marker.visible = point !== undefined;
      if (point) marker.position.set(point.x, point.y + 0.012, point.z);
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

function blockColor(id: string): number {
  let hash = 2166136261;
  for (const character of id) {
    hash ^= character.codePointAt(0) ?? 0;
    hash = Math.imul(hash, 16777619);
  }
  const palette = [0x66786a, 0x806e55, 0x586f77, 0x71805f, 0x69627b];
  return palette[Math.abs(hash) % palette.length]!;
}
