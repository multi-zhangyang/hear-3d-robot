import loadMujoco from "@mujoco/mujoco";
import { readFile, readdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

export type MujocoModule = Awaited<ReturnType<typeof loadMujoco>>;

export interface HumanoidSceneSolid {
  id: string;
  center: { x: number; y: number; z: number };
  size: { x: number; y: number; z: number };
}

export interface HumanoidSceneObject extends HumanoidSceneSolid {
  mass: number;
}

const ASSET_ROOT = fileURLToPath(new URL("../../../assets/humanoid/g1/", import.meta.url));
const VIRTUAL_ROOT = "/hear/g1";
let runtimePromise: Promise<MujocoModule> | undefined;
let sceneSequence = 0;

export function humanoidModelPath(): string {
  return `${VIRTUAL_ROOT}/scene_29dof.xml`;
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

export function loadHumanoidMujoco(): Promise<MujocoModule> {
  runtimePromise ??= initializeMujoco();
  return runtimePromise;
}

async function initializeMujoco(): Promise<MujocoModule> {
  const runtime = await loadMujoco();
  runtime.FS.mkdirTree(`${VIRTUAL_ROOT}/meshes`, 0o777);
  for (const entry of await readdir(ASSET_ROOT, { recursive: true, withFileTypes: true })) {
    if (!entry.isFile()) continue;
    const source = path.join(entry.parentPath, entry.name);
    const relative = path.relative(ASSET_ROOT, source).split(path.sep).join("/");
    const target = `${VIRTUAL_ROOT}/${relative}`;
    runtime.FS.mkdirTree(target.slice(0, target.lastIndexOf("/")), 0o777);
    runtime.FS.writeFile(target, await readFile(source));
  }
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
    return `    <geom name="world-solid-${index}-${xmlName(solid.id)}" type="box" pos="${numbers(position)}" size="${numbers(halfSize)}" friction="0.9 0.01 0.001" rgba="0.34 0.38 0.31 1"/>`;
  }).join("\n");
  const dynamicBodies = objects.map((object, index) => {
    assertSolid(object);
    if (!Number.isFinite(object.mass) || object.mass <= 0) {
      throw new Error(`Invalid humanoid scene object mass: ${object.id}`);
    }
    const position = [object.center.z, object.center.x, object.center.y];
    const halfSize = [object.size.z / 2, object.size.x / 2, object.size.y / 2];
    return `    <body name="world-object-${index}" pos="${numbers(position)}">
      <freejoint name="world-object-joint-${index}"/>
      <geom name="world-object-geom-${index}" type="box" size="${numbers(halfSize)}" mass="${Number(object.mass.toFixed(6))}" friction="0.8 0.01 0.001" rgba="0.64 0.44 0.23 1"/>
    </body>`;
  }).join("\n");
  return `<mujoco model="g1_29dof world">
  <include file="g1_29dof.xml"/>
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
${dynamicBodies}
  </worldbody>
</mujoco>`;
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
  return values.map((value) => Number(value.toFixed(6))).join(" ");
}

function xmlName(value: string): string {
  const safe = value.replace(/[^A-Za-z0-9_.-]/g, "-").slice(0, 48);
  return safe || "solid";
}
