import * as THREE from "three";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import { disposeObject } from "../three-kit";
import type { HumanoidWorldSnapshot } from "../types";
import { transformMujocoGeometry, transformMujocoLocalVector } from "./coordinates";

interface G1Part {
  body: string;
  mesh: string;
  tone: "graphite" | "shell" | "joint" | "hand";
  offset?: readonly [number, number, number];
}

const BODY_PARTS: readonly G1Part[] = [
  { body: "pelvis", mesh: "pelvis", tone: "graphite" },
  { body: "pelvis", mesh: "pelvis_contour_link", tone: "shell" },
  { body: "left_hip_pitch_link", mesh: "left_hip_pitch_link", tone: "graphite" },
  { body: "left_hip_roll_link", mesh: "left_hip_roll_link", tone: "shell" },
  { body: "left_hip_yaw_link", mesh: "left_hip_yaw_link", tone: "shell" },
  { body: "left_knee_link", mesh: "left_knee_link", tone: "shell" },
  { body: "left_ankle_pitch_link", mesh: "left_ankle_pitch_link", tone: "joint" },
  { body: "left_ankle_roll_link", mesh: "left_ankle_roll_link", tone: "graphite" },
  { body: "right_hip_pitch_link", mesh: "right_hip_pitch_link", tone: "graphite" },
  { body: "right_hip_roll_link", mesh: "right_hip_roll_link", tone: "shell" },
  { body: "right_hip_yaw_link", mesh: "right_hip_yaw_link", tone: "shell" },
  { body: "right_knee_link", mesh: "right_knee_link", tone: "shell" },
  { body: "right_ankle_pitch_link", mesh: "right_ankle_pitch_link", tone: "joint" },
  { body: "right_ankle_roll_link", mesh: "right_ankle_roll_link", tone: "graphite" },
  { body: "waist_yaw_link", mesh: "waist_yaw_link", tone: "joint" },
  { body: "waist_roll_link", mesh: "waist_roll_link", tone: "joint" },
  { body: "torso_link", mesh: "torso_link", tone: "shell" },
  { body: "torso_link", mesh: "logo_link", tone: "graphite", offset: [0.0039635, 0, -0.054] },
  { body: "torso_link", mesh: "waist_support_link", tone: "shell", offset: [0.0039635, 0, -0.054] },
  { body: "head_link", mesh: "head_link", tone: "graphite", offset: [0.0039635, 0, -0.054] },
  { body: "left_shoulder_pitch_link", mesh: "left_shoulder_pitch_link", tone: "shell" },
  { body: "left_shoulder_roll_link", mesh: "left_shoulder_roll_link", tone: "shell" },
  { body: "left_shoulder_yaw_link", mesh: "left_shoulder_yaw_link", tone: "shell" },
  { body: "left_elbow_link", mesh: "left_elbow_link", tone: "shell" },
  { body: "left_wrist_roll_link", mesh: "left_wrist_roll_link", tone: "joint" },
  { body: "left_wrist_pitch_link", mesh: "left_wrist_pitch_link", tone: "joint" },
  { body: "left_wrist_yaw_link", mesh: "left_wrist_yaw_link", tone: "shell" },
  { body: "left_wrist_yaw_link", mesh: "left_rubber_hand", tone: "hand", offset: [0.0415, 0.003, 0] },
  { body: "right_shoulder_pitch_link", mesh: "right_shoulder_pitch_link", tone: "shell" },
  { body: "right_shoulder_roll_link", mesh: "right_shoulder_roll_link", tone: "shell" },
  { body: "right_shoulder_yaw_link", mesh: "right_shoulder_yaw_link", tone: "shell" },
  { body: "right_elbow_link", mesh: "right_elbow_link", tone: "shell" },
  { body: "right_wrist_roll_link", mesh: "right_wrist_roll_link", tone: "joint" },
  { body: "right_wrist_pitch_link", mesh: "right_wrist_pitch_link", tone: "joint" },
  { body: "right_wrist_yaw_link", mesh: "right_wrist_yaw_link", tone: "shell" },
  { body: "right_wrist_yaw_link", mesh: "right_rubber_hand", tone: "hand", offset: [0.0415, -0.003, 0] }
];

const COLORS = {
  graphite: 0x111821,
  shell: 0xc9d1d7,
  joint: 0x68727b,
  hand: 0x202830
} as const;

export class G1Rig {
  readonly root = new THREE.Group();
  readonly #links = new Map<string, THREE.Group>();
  readonly #footMaterials = new Map<"left" | "right", THREE.MeshStandardMaterial[]>();
  readonly #visionMaterial = new THREE.MeshStandardMaterial({
    color: 0x72e9d2,
    emissive: 0x35d7ba,
    emissiveIntensity: 2.4,
    roughness: 0.18,
    metalness: 0.42
  });

  static async create(signal?: AbortSignal): Promise<G1Rig> {
    const rig = new G1Rig();
    const loader = new STLLoader();
    const buffers = await Promise.all(BODY_PARTS.map(async (part) => {
      signal?.throwIfAborted();
      const response = await fetch(
        `/humanoid/g1/meshes/${part.mesh}.STL`,
        signal ? { signal } : undefined
      );
      if (!response.ok) throw new Error(`无法载入 G1 部件：${part.mesh}`);
      return response.arrayBuffer();
    }));
    const geometries: THREE.BufferGeometry[] = [];
    try {
      for (const buffer of buffers) {
        signal?.throwIfAborted();
        geometries.push(transformMujocoGeometry(loader.parse(buffer)));
        await yieldToBrowser();
      }
      signal?.throwIfAborted();
      BODY_PARTS.forEach((part, index) => rig.#addPart(part, geometries[index]!));
      rig.#addVisionBand();
      return rig;
    } catch (error) {
      geometries.forEach((geometry) => geometry.dispose());
      throw error;
    }
  }

  private constructor() {
    this.root.name = "unitree-g1-authoritative-rig";
  }

  update(snapshot: HumanoidWorldSnapshot): void {
    for (const [name, group] of this.#links) {
      const link = snapshot.robot.links[name];
      group.visible = link !== undefined;
      if (!link) continue;
      group.position.set(link.position.x, link.position.y, link.position.z);
      group.quaternion.set(link.rotation.x, link.rotation.y, link.rotation.z, link.rotation.w);
    }
    this.#setFootState("left", snapshot.robot.feet.left.touching, snapshot.robot.feet.left.normalForce);
    this.#setFootState("right", snapshot.robot.feet.right.touching, snapshot.robot.feet.right.normalForce);
    this.#visionMaterial.emissiveIntensity = snapshot.robot.fallen ? 0.25 : 2.4;
    this.#visionMaterial.color.set(snapshot.robot.fallen ? 0xd65a62 : 0x72e9d2);
    this.#visionMaterial.emissive.set(snapshot.robot.fallen ? 0xb62f3d : 0x35d7ba);
  }

  bounds(): THREE.Box3 {
    return new THREE.Box3().setFromObject(this.root);
  }

  dispose(): void {
    disposeObject(this.root);
  }

  #group(body: string): THREE.Group {
    const existing = this.#links.get(body);
    if (existing) return existing;
    const created = new THREE.Group();
    created.name = body;
    this.#links.set(body, created);
    this.root.add(created);
    return created;
  }

  #addPart(part: G1Part, geometry: THREE.BufferGeometry): void {
    const material = new THREE.MeshStandardMaterial({
      color: COLORS[part.tone],
      roughness: part.tone === "shell" ? 0.32 : 0.48,
      metalness: part.tone === "hand" ? 0.08 : 0.52,
      emissive: 0x000000,
      emissiveIntensity: 0
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = part.mesh;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    if (part.offset) mesh.position.copy(transformMujocoLocalVector(part.offset));
    this.#group(part.body).add(mesh);
    if (part.body === "left_ankle_roll_link") this.#rememberFootMaterial("left", material);
    if (part.body === "right_ankle_roll_link") this.#rememberFootMaterial("right", material);
  }

  #addVisionBand(): void {
    const head = this.#group("head_link");
    const headMesh = head.children.find((entry) => entry.name === "head_link") as THREE.Mesh | undefined;
    const box = headMesh?.geometry.boundingBox;
    const width = box ? Math.max(0.08, (box.max.x - box.min.x) * 0.54) : 0.12;
    const centerX = box ? (box.min.x + box.max.x) / 2 : 0;
    const centerY = box ? box.min.y + (box.max.y - box.min.y) * 0.72 : 0.18;
    const front = box ? box.max.z + 0.003 : 0.08;
    const band = new THREE.Mesh(new THREE.PlaneGeometry(width, 0.028), this.#visionMaterial);
    band.name = "head-vision-band";
    band.position.set(centerX, centerY, front);
    head.add(band);
  }

  #rememberFootMaterial(side: "left" | "right", material: THREE.MeshStandardMaterial): void {
    const current = this.#footMaterials.get(side) ?? [];
    current.push(material);
    this.#footMaterials.set(side, current);
  }

  #setFootState(side: "left" | "right", touching: boolean, force: number): void {
    for (const material of this.#footMaterials.get(side) ?? []) {
      material.emissive.set(touching ? 0x35d7ba : 0x000000);
      material.emissiveIntensity = touching ? Math.min(1.2, 0.25 + force / 900) : 0;
    }
  }
}

function yieldToBrowser(): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, 0));
}
