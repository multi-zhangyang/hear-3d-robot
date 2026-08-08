import loadMujoco from "@mujoco/mujoco";
import { readFile, readdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import {
  g1PhysicsModelXml
} from "./hand-collision-geometry.js";

export type MujocoModule = Awaited<ReturnType<typeof loadMujoco>>;
export type MujocoModel = InstanceType<MujocoModule["MjModel"]>;
export type MujocoData = InstanceType<MujocoModule["MjData"]>;

export interface HumanoidSceneSolid {
  id: string;
  center: { x: number; y: number; z: number };
  size: { x: number; y: number; z: number };
}

export interface HumanoidSceneObject extends HumanoidSceneSolid {
  mass: number;
  shape?: "box" | "sphere" | "cylinder" | "capsule" | undefined;
  friction?: {
    sliding: number;
    torsional: number;
    rolling: number;
  } | undefined;
  interactionPoints?: Array<{
    id: string;
    kind: "grasp" | "push" | "pull" | "press" | "turn" | "insert" | "support";
    localPosition: { x: number; y: number; z: number };
    clearanceMeters: number;
  }> | undefined;
  mobility?:
    | { type: "fixed" }
    | { type: "free" }
    | {
        type: "hinge" | "slide";
        axis: { x: number; y: number; z: number };
        anchor: { x: number; y: number; z: number };
        range: { minimum: number; maximum: number };
        initialPosition: number;
        damping: number;
        frictionLoss: number;
      } | undefined;
  container?: {
    interiorCenter: { x: number; y: number; z: number };
    interiorSize: { x: number; y: number; z: number };
    openingDirection: { x: number; y: number; z: number };
    wallThickness: number;
  } | undefined;
}

export interface ResolvedHumanoidSceneObject extends HumanoidSceneSolid {
  mass: number;
  shape: "box" | "sphere" | "cylinder" | "capsule";
  friction: {
    sliding: number;
    torsional: number;
    rolling: number;
  };
  interactionPoints: NonNullable<HumanoidSceneObject["interactionPoints"]>;
  mobility: NonNullable<HumanoidSceneObject["mobility"]>;
  container?: NonNullable<HumanoidSceneObject["container"]> | undefined;
}

export function resolveHumanoidSceneObject(
  object: HumanoidSceneObject
): ResolvedHumanoidSceneObject {
  return {
    id: object.id,
    center: { ...object.center },
    size: { ...object.size },
    mass: object.mass,
    shape: object.shape ?? "box",
    friction: object.friction
      ? { ...object.friction }
      : { sliding: 0.8, torsional: 0.01, rolling: 0.001 },
    interactionPoints: structuredClone(object.interactionPoints ?? []),
    mobility: object.mobility ? structuredClone(object.mobility) : { type: "free" },
    ...(object.container ? { container: structuredClone(object.container) } : {})
  };
}

const ASSET_ROOT = fileURLToPath(new URL("../../../assets/humanoid/g1/", import.meta.url));
const VIRTUAL_ROOT = "/hear/g1";
let runtimePromise: Promise<MujocoModule> | undefined;
let sceneSequence = 0;

export function humanoidModelPath(): string {
  return `${VIRTUAL_ROOT}/scene_with_hands_physics.xml`;
}

export function createHumanoidScenePath(
  runtime: MujocoModule,
  solids: readonly HumanoidSceneSolid[],
  objects: readonly HumanoidSceneObject[] = []
): string {
  const path = `${VIRTUAL_ROOT}/generated-scene-${sceneSequence++}.xml`;
  runtime.FS.writeFile(path, new TextEncoder().encode(sceneXml(solids, objects)));
  return path;
}

export function removeHumanoidScene(runtime: MujocoModule, path: string): void {
  runtime.FS.unlink(path);
}

export function humanoidSceneSolidGeomName(index: number, id: string): string {
  if (!Number.isSafeInteger(index) || index < 0) {
    throw new Error("Humanoid scene solid index must be a nonnegative integer");
  }
  return `world-solid-${index}-${xmlName(id)}`;
}

export function loadHumanoidMujoco(): Promise<MujocoModule> {
  runtimePromise ??= initializeMujoco();
  return runtimePromise;
}

async function initializeMujoco(): Promise<MujocoModule> {
  const runtime = await loadMujoco();
  let sourceModel: string | undefined;
  runtime.FS.mkdirTree(`${VIRTUAL_ROOT}/meshes`, 0o777);
  runtime.FS.mkdirTree(`${VIRTUAL_ROOT}/assets`, 0o777);
  for (const entry of await readdir(ASSET_ROOT, { recursive: true, withFileTypes: true })) {
    if (!entry.isFile()) continue;
    const source = path.join(entry.parentPath, entry.name);
    const relative = path.relative(ASSET_ROOT, source).split(path.sep).join("/");
    const target = `${VIRTUAL_ROOT}/${relative}`;
    runtime.FS.mkdirTree(target.slice(0, target.lastIndexOf("/")), 0o777);
    const contents = await readFile(source);
    if (relative === "g1_with_hands.xml") sourceModel = contents.toString("utf8");
    if (relative.startsWith("meshes/") && relative.endsWith(".STL")) {
      runtime.FS.writeFile(`${VIRTUAL_ROOT}/assets/${entry.name}`, contents);
    }
    runtime.FS.writeFile(target, contents);
  }
  if (!sourceModel) throw new Error("G1 43DoF source model is missing");
  runtime.FS.writeFile(
    `${VIRTUAL_ROOT}/g1_with_hands_physics.xml`,
    new TextEncoder().encode(g1PhysicsModelXml(sourceModel))
  );
  runtime.FS.writeFile(
    humanoidModelPath(),
    new TextEncoder().encode(sceneXml([], []))
  );
  return runtime;
}

function sceneXml(
  solids: readonly HumanoidSceneSolid[],
  objects: readonly HumanoidSceneObject[]
): string {
  const geoms = solids.map((solid, index) => {
    assertSolid(solid);
    const position = [solid.center.z, solid.center.x, solid.center.y];
    const halfSize = [solid.size.z / 2, solid.size.x / 2, solid.size.y / 2];
    return `    <geom name="${humanoidSceneSolidGeomName(index, solid.id)}" type="box" pos="${numbers(position)}" size="${numbers(halfSize)}" friction="0.9 0.01 0.001" rgba="0.34 0.38 0.31 1"/>`;
  }).join("\n");
  const objectBodies = objects.map(resolveHumanoidSceneObject).map((object, index) => {
    assertSolid(object);
    if (!Number.isFinite(object.mass) || object.mass <= 0) {
      throw new Error(`Invalid humanoid scene object mass: ${object.id}`);
    }
    assertSceneObject(object);
    const mobility = object.mobility;
    const articulated = mobility.type === "hinge" || mobility.type === "slide";
    const bodyPosition = articulated ? mobility.anchor : object.center;
    const geometryOffset = articulated
      ? subtractVector(object.center, mobility.anchor)
      : { x: 0, y: 0, z: 0 };
    const joint = sceneObjectJointXml(object, index);
    const geometry = object.container
      ? containerGeometriesXml(object, index)
      : sceneObjectGeometryXml(object, index, geometryOffset);
    const interactionGeometry = interactionPointGeometriesXml(
      object,
      index,
      geometryOffset
    );
    return `    <body name="world-object-${index}" pos="${numbers(worldVector(bodyPosition))}">
${joint}${geometry}${interactionGeometry ? `\n${interactionGeometry}` : ""}
    </body>`;
  }).join("\n");
  return `<mujoco model="g1_43dof world">
  <include file="g1_with_hands_physics.xml"/>
  <statistic center="0 0 0.7" extent="4"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>
  <asset>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.24" rgb2="0.12 0.18 0.15" markrgb="0.65 0.72 0.62" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="8 8" reflectance="0.08"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" friction="0.95 0.01 0.001"/>
${geoms}
${objectBodies}
  </worldbody>
</mujoco>`;
}

function interactionPointGeometriesXml(
  object: ResolvedHumanoidSceneObject,
  objectIndex: number,
  geometryOffset: HumanoidSceneSolid["center"]
): string {
  return object.interactionPoints
    .filter(({ kind }) => kind !== "insert" && kind !== "support")
    .map((point, pointIndex) => {
      const position = addVector(geometryOffset, point.localPosition);
      const radius = clamp(point.clearanceMeters * 0.32, 0.012, 0.032);
      const geometry = interactionPointGeometry(point.kind, radius);
      return `      <geom name="world-object-interaction-${objectIndex}-${pointIndex}-${xmlName(point.id)}" type="${geometry.type}" pos="${numbers(worldVector(position))}" size="${numbers(geometry.size)}" mass="0.015" friction="${friction(object)}" rgba="0.88 0.66 0.28 1"/>`;
    }).join("\n");
}

function interactionPointGeometry(
  kind: NonNullable<HumanoidSceneObject["interactionPoints"]>[number]["kind"],
  radius: number
): { type: "sphere" | "capsule"; size: number[] } {
  if (kind === "pull") {
    return {
      type: "capsule",
      size: [
        Math.min(radius, 0.022),
        clamp(radius * 2.4, 0.035, 0.065)
      ]
    };
  }
  return { type: "sphere", size: [radius] };
}

function sceneObjectJointXml(object: ResolvedHumanoidSceneObject, index: number): string {
  if (object.mobility.type === "fixed") return "";
  if (object.mobility.type === "free") {
    return `      <freejoint name="world-object-joint-${index}"/>\n`;
  }
  return `      <joint name="world-object-joint-${index}" type="${object.mobility.type}" axis="${numbers(worldVector(object.mobility.axis))}" range="${numbers([
    object.mobility.range.minimum,
    object.mobility.range.maximum
  ])}" limited="true" damping="${number(object.mobility.damping)}" frictionloss="${number(object.mobility.frictionLoss)}"/>\n`;
}

function sceneObjectGeometryXml(
  object: ResolvedHumanoidSceneObject,
  index: number,
  localPosition: HumanoidSceneSolid["center"]
): string {
  const geometry = shapeGeometry(object.shape, object.size);
  return `      <geom name="world-object-geom-${index}" type="${geometry.type}" pos="${numbers(worldVector(localPosition))}" size="${numbers(geometry.size)}" mass="${number(object.mass)}" friction="${friction(object)}" rgba="0.64 0.44 0.23 1"/>`;
}

function containerGeometriesXml(
  object: ResolvedHumanoidSceneObject,
  index: number
): string {
  const container = object.container;
  if (!container) throw new Error("Container geometry is missing");
  const opening = dominantSignedAxis(container.openingDirection);
  const half = {
    x: container.interiorSize.x / 2,
    y: container.interiorSize.y / 2,
    z: container.interiorSize.z / 2
  };
  const t = container.wallThickness;
  const faces = ([
    ["x+", { x: half.x + t / 2, y: 0, z: 0 }, { x: t, y: container.interiorSize.y + 2 * t, z: container.interiorSize.z + 2 * t }],
    ["x-", { x: -half.x - t / 2, y: 0, z: 0 }, { x: t, y: container.interiorSize.y + 2 * t, z: container.interiorSize.z + 2 * t }],
    ["y+", { x: 0, y: half.y + t / 2, z: 0 }, { x: container.interiorSize.x + 2 * t, y: t, z: container.interiorSize.z + 2 * t }],
    ["y-", { x: 0, y: -half.y - t / 2, z: 0 }, { x: container.interiorSize.x + 2 * t, y: t, z: container.interiorSize.z + 2 * t }],
    ["z+", { x: 0, y: 0, z: half.z + t / 2 }, { x: container.interiorSize.x + 2 * t, y: container.interiorSize.y + 2 * t, z: t }],
    ["z-", { x: 0, y: 0, z: -half.z - t / 2 }, { x: container.interiorSize.x + 2 * t, y: container.interiorSize.y + 2 * t, z: t }]
  ] as const).filter(([face]) => face !== opening);
  return faces.map(([face, offset, size], faceIndex) => {
    const localPosition = addVector(container.interiorCenter, offset);
    return `      <geom name="world-object-geom-${index}-${faceIndex}-${face}" type="box" pos="${numbers(worldVector(localPosition))}" size="${numbers(worldVector({
      x: size.x / 2,
      y: size.y / 2,
      z: size.z / 2
    }))}" mass="${number(object.mass / faces.length)}" friction="${friction(object)}" rgba="0.42 0.48 0.38 1"/>`;
  }).join("\n");
}

function shapeGeometry(
  shape: ResolvedHumanoidSceneObject["shape"],
  size: ResolvedHumanoidSceneObject["size"]
): { type: ResolvedHumanoidSceneObject["shape"]; size: number[] } {
  if (shape === "box") {
    return { type: shape, size: worldVector({ x: size.x / 2, y: size.y / 2, z: size.z / 2 }) };
  }
  if (shape === "sphere") {
    return { type: shape, size: [Math.min(size.x, size.y, size.z) / 2] };
  }
  const radius = Math.min(size.x, size.z) / 2;
  const halfHeight = shape === "capsule"
    ? Math.max(0.001, size.y / 2 - radius)
    : size.y / 2;
  return { type: shape, size: [radius, halfHeight] };
}

function dominantSignedAxis(direction: HumanoidSceneSolid["center"]): string {
  const axes = [
    ["x", direction.x],
    ["y", direction.y],
    ["z", direction.z]
  ] as const;
  const [axis, value] = [...axes].sort((left, right) => (
    Math.abs(right[1]) - Math.abs(left[1])
  ))[0]!;
  return `${axis}${value >= 0 ? "+" : "-"}`;
}

function friction(object: ResolvedHumanoidSceneObject): string {
  return numbers([
    object.friction.sliding,
    object.friction.torsional,
    object.friction.rolling
  ]);
}

function assertSceneObject(object: ResolvedHumanoidSceneObject): void {
  if (!object.id.trim()) throw new Error("Humanoid scene object ID is required");
  if (![object.friction.sliding, object.friction.torsional, object.friction.rolling]
    .every((value) => Number.isFinite(value) && value >= 0)) {
    throw new Error(`Invalid humanoid scene object friction: ${object.id}`);
  }
  if (object.mobility.type === "hinge" || object.mobility.type === "slide") {
    if (object.mobility.range.minimum >= object.mobility.range.maximum
      || object.mobility.initialPosition < object.mobility.range.minimum
      || object.mobility.initialPosition > object.mobility.range.maximum) {
      throw new Error(`Invalid humanoid scene articulation: ${object.id}`);
    }
  }
}

function worldVector(value: HumanoidSceneSolid["center"]): number[] {
  return [value.z, value.x, value.y];
}

function addVector(
  left: HumanoidSceneSolid["center"],
  right: HumanoidSceneSolid["center"]
): HumanoidSceneSolid["center"] {
  return { x: left.x + right.x, y: left.y + right.y, z: left.z + right.z };
}

function subtractVector(
  left: HumanoidSceneSolid["center"],
  right: HumanoidSceneSolid["center"]
): HumanoidSceneSolid["center"] {
  return { x: left.x - right.x, y: left.y - right.y, z: left.z - right.z };
}

function assertSolid(solid: HumanoidSceneSolid): void {
  const values = [
    solid.center.x,
    solid.center.y,
    solid.center.z,
    solid.size.x,
    solid.size.y,
    solid.size.z
  ];
  if (values.some((value) => !Number.isFinite(value))
    || solid.size.x <= 0
    || solid.size.y <= 0
    || solid.size.z <= 0) {
    throw new Error(`Invalid humanoid scene solid: ${solid.id}`);
  }
}

function numbers(values: readonly number[]): string {
  return values.map(number).join(" ");
}

function number(value: number): number {
  return Number(value.toFixed(6));
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function xmlName(value: string): string {
  const safe = value.replace(/[^A-Za-z0-9_.-]/g, "-").slice(0, 48);
  return safe || "solid";
}
