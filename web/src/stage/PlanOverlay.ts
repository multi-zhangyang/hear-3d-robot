import * as THREE from "three";
import type { Vec3, WorldSnapshot } from "../types";

/** Visualizes model-selected intent separately from the path already travelled. */
export class PlanOverlay {
  readonly root = new THREE.Group();
  readonly #routeMaterial = new THREE.LineDashedMaterial({
    color: 0xffb44d,
    dashSize: 0.18,
    gapSize: 0.12,
    transparent: true,
    opacity: 0.95
  });
  readonly #goal = new THREE.Group();
  readonly #facing = new THREE.ArrowHelper(
    new THREE.Vector3(0, 0, 1),
    new THREE.Vector3(),
    1,
    0xffb44d,
    0.2,
    0.12
  );
  #route: THREE.Line | null = null;
  #routeKey = "";

  constructor() {
    const material = new THREE.MeshStandardMaterial({
      color: 0xffb44d,
      emissive: 0xffb44d,
      emissiveIntensity: 1.2,
      transparent: true,
      opacity: 0.9
    });
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.26, 0.014, 8, 40), material);
    ring.rotation.x = Math.PI / 2;
    ring.position.y = 0.02;
    const inner = new THREE.Mesh(new THREE.TorusGeometry(0.14, 0.01, 8, 30), material);
    inner.rotation.x = Math.PI / 2;
    inner.position.y = 0.02;
    const pin = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.5, 8), material);
    pin.position.y = 0.25;
    this.#goal.add(ring, inner, pin);
    this.root.add(this.#goal, this.#facing);
  }

  update(snapshot: WorldSnapshot): void {
    const { navigation } = snapshot;
    const active = navigation.status === "planned" || navigation.status === "executing";
    const target = active ? navigation.target : null;
    this.#goal.visible = target !== null;
    if (target) this.#goal.position.set(target.x, 0, target.z);

    const face = active ? navigation.face : null;
    this.#facing.visible = target !== null && face !== null;
    if (target && face) {
      const direction = new THREE.Vector3(face.x - target.x, 0, face.z - target.z);
      const length = direction.length();
      if (length > 1e-4) {
        this.#facing.position.set(target.x, 0.05, target.z);
        this.#facing.setDirection(direction.divideScalar(length));
        this.#facing.setLength(Math.min(length, 1.1), 0.2, 0.12);
      } else {
        this.#facing.visible = false;
      }
    }
    this.#syncRoute(active ? navigation.waypoints : []);
  }

  #syncRoute(waypoints: Vec3[]): void {
    const key = waypoints.map((point) => `${point.x},${point.z}`).join(";");
    if (key === this.#routeKey) return;
    this.#routeKey = key;
    if (this.#route) {
      this.root.remove(this.#route);
      this.#route.geometry.dispose();
      this.#route = null;
    }
    if (waypoints.length < 2) return;
    const geometry = new THREE.BufferGeometry().setFromPoints(
      waypoints.map((point) => new THREE.Vector3(point.x, 0.07, point.z))
    );
    this.#route = new THREE.Line(geometry, this.#routeMaterial);
    this.#route.computeLineDistances();
    this.root.add(this.#route);
  }
}
