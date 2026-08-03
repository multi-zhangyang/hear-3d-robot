import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { disposeObject } from "../three-kit";
import type { HumanoidFrameBuffer } from "../stage/humanoid-frame-buffer";
import { renderPixelRatio } from "../stage/render-quality";
import { createRenderBackend } from "../stage/render-backend";
import { addStudioLighting } from "../stage/scene-primitives";
import type { HumanoidWorldSnapshot, ScenarioDefinition } from "../types";
import { HumanoidWorldScene } from "./humanoid-world-scene";

export type HumanoidCameraMode = "follow" | "world" | "head";

export interface HumanoidStageController {
  setLive(live: boolean): void;
  setCameraMode(mode: HumanoidCameraMode): void;
  fit(): void;
  dispose(): void;
}

export async function createHumanoidStage(
  host: HTMLDivElement,
  scenario: ScenarioDefinition,
  initialFrame: HumanoidWorldSnapshot,
  frameBuffer: HumanoidFrameBuffer,
  initiallyLive: boolean,
  onError: (message: string | null) => void,
  signal?: AbortSignal
): Promise<HumanoidStageController> {
  signal?.throwIfAborted();
  if (!frameBuffer.latest) frameBuffer.reset(initialFrame);
  const extent = Math.max(scenario.bounds.width, scenario.bounds.depth);
  const scene = new THREE.Scene();
  const worldSky = 0x17242c;
  scene.background = new THREE.Color(worldSky);
  scene.fog = new THREE.Fog(worldSky, Math.max(18, extent * 0.9), Math.max(64, extent * 3.4));
  let contextLost = false;
  let reportedError: string | null = null;
  const report = (message: string | null): void => {
    if (reportedError === message) return;
    reportedError = message;
    onError(message);
  };
  const runtime = await createRenderBackend(scene, {
    onDeviceLost: (message) => {
      contextLost = true;
      report(message);
    },
    onError: report
  }, signal);
  signal?.throwIfAborted();
  const { renderer, environment } = runtime;
  renderer.setClearColor(worldSky, 1);
  renderer.toneMappingExposure = 1.32;
  scene.environmentIntensity = 0.5;
  renderer.domElement.className = "three-canvas humanoid-canvas";
  host.appendChild(renderer.domElement);
  addStudioLighting(scene, scenario.bounds);
  scene.add(new THREE.AmbientLight(0xb6d0c5, 0.42));
  const world = await HumanoidWorldScene.create(scene, scenario, signal);
  signal?.throwIfAborted();

  const camera = new THREE.PerspectiveCamera(42, 1, 0.03, Math.max(140, extent * 5));
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = false;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.minDistance = 1.2;
  controls.maxDistance = Math.max(40, extent * 3.5);
  let current = frameBuffer.latest ?? initialFrame;
  world.update(current);
  let live = initiallyLive;
  let mode: HumanoidCameraMode = "follow";
  let animationFrame = 0;
  let projectionRevision = 0;
  let disposed = false;
  let fitted = false;
  let previousRoot = new THREE.Vector3(
    current.robot.rootPosition.x,
    current.robot.rootPosition.y,
    current.robot.rootPosition.z
  );

  const fit = (): void => {
    const root = vector(current.robot.rootPosition);
    if (mode === "world") {
      const center = world.worldBounds().getCenter(new THREE.Vector3());
      controls.enabled = true;
      controls.target.copy(center);
      camera.position.set(
        center.x + scenario.bounds.width * 0.62,
        Math.max(12, extent * 0.7),
        center.z + scenario.bounds.depth * 0.62
      );
    } else if (mode === "head") {
      controls.enabled = false;
      placeHeadCamera(camera, current);
    } else {
      controls.enabled = true;
      const rotation = quaternion(current.robot.rootRotation);
      const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(rotation);
      const side = new THREE.Vector3(1, 0, 0).applyQuaternion(rotation);
      const robotBounds = world.robotBounds();
      const robotCenter = robotBounds.getCenter(new THREE.Vector3());
      const robotHeight = Math.max(1.25, robotBounds.getSize(new THREE.Vector3()).y);
      const portrait = camera.aspect < 0.72;
      const distance = Math.max(
        portrait ? 5.4 : 2.85,
        robotHeight * (portrait ? 4.55 : 2.2)
      );
      controls.target.copy(robotCenter)
        .addScaledVector(forward, portrait ? 0.16 : 0.38);
      controls.target.y = robotCenter.y - robotHeight * (portrait ? 0.01 : 0.08);
      camera.position.copy(controls.target)
        .addScaledVector(forward, -distance)
        .addScaledVector(side, distance * (portrait ? 0.08 : 0.16));
      camera.position.y = controls.target.y + robotHeight * (portrait ? 0.82 : 0.76);
    }
    controls.update();
    previousRoot = root;
    fitted = true;
  };

  const follow = (): void => {
    if (!fitted) {
      fit();
      return;
    }
    if (mode === "head") {
      placeHeadCamera(camera, current);
      return;
    }
    if (mode !== "follow") return;
    const root = vector(current.robot.rootPosition);
    const delta = root.sub(previousRoot);
    camera.position.add(delta);
    controls.target.add(delta);
    controls.update();
    previousRoot.add(delta);
  };

  function schedule(): void {
    if (disposed || animationFrame !== 0) return;
    animationFrame = window.requestAnimationFrame(render);
  }

  function render(wallTimeMs: number): void {
    animationFrame = 0;
    if (disposed) return;
    try {
      const visual = frameBuffer.sample(wallTimeMs, live) ?? frameBuffer.latest ?? current;
      if (visual !== current) world.update(visual);
      current = visual;
      follow();
      if (!contextLost) {
        renderer.render(scene, camera);
        publishRobotScreenBounds(
          renderer.domElement,
          world.robotBounds(),
          camera,
          host,
          mode,
          ++projectionRevision
        );
        report(null);
      }
      if (live && frameBuffer.pending) schedule();
    } catch (error) {
      report(error instanceof Error ? error.message : String(error));
    }
  }

  const onContextLost = (event: Event): void => {
    event.preventDefault();
    contextLost = true;
    report("浏览器图形上下文已丢失");
  };
  const onContextRestored = (): void => {
    contextLost = false;
    report(null);
    schedule();
  };
  renderer.domElement.addEventListener("webglcontextlost", onContextLost);
  renderer.domElement.addEventListener("webglcontextrestored", onContextRestored);
  controls.addEventListener("change", schedule);

  const resize = (): void => {
    const width = Math.max(1, host.clientWidth);
    const height = Math.max(1, host.clientHeight);
    renderer.setPixelRatio(renderPixelRatio(width, height, window.devicePixelRatio));
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    fit();
    if (!contextLost) {
      renderer.render(scene, camera);
      publishRobotScreenBounds(
        renderer.domElement,
        world.robotBounds(),
        camera,
        host,
        mode,
        ++projectionRevision
      );
    }
  };
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(host);
  const unsubscribe = frameBuffer.subscribe(schedule);
  resize();
  schedule();

  return {
    setLive(value) {
      live = value;
      schedule();
    },
    setCameraMode(value) {
      mode = value;
      fitted = false;
      fit();
      schedule();
    },
    fit() {
      fit();
      schedule();
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      if (animationFrame !== 0) window.cancelAnimationFrame(animationFrame);
      unsubscribe();
      resizeObserver.disconnect();
      controls.removeEventListener("change", schedule);
      controls.dispose();
      renderer.domElement.removeEventListener("webglcontextlost", onContextLost);
      renderer.domElement.removeEventListener("webglcontextrestored", onContextRestored);
      disposeObject(scene);
      environment.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    }
  };
}

function placeHeadCamera(camera: THREE.PerspectiveCamera, frame: HumanoidWorldSnapshot): void {
  const head = frame.robot.links.head_link;
  if (!head) return;
  const rotation = quaternion(head.rotation);
  const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(rotation);
  const up = new THREE.Vector3(0, 1, 0).applyQuaternion(rotation);
  camera.position.copy(vector(head.position))
    .addScaledVector(up, 0.43)
    .addScaledVector(forward, 0.08);
  camera.up.copy(up);
  camera.lookAt(camera.position.clone().addScaledVector(forward, 6));
}

function vector(value: { x: number; y: number; z: number }): THREE.Vector3 {
  return new THREE.Vector3(value.x, value.y, value.z);
}

function quaternion(value: { x: number; y: number; z: number; w: number }): THREE.Quaternion {
  return new THREE.Quaternion(value.x, value.y, value.z, value.w);
}

function publishRobotScreenBounds(
  canvas: HTMLCanvasElement,
  bounds: THREE.Box3,
  camera: THREE.PerspectiveCamera,
  host: HTMLDivElement,
  mode: HumanoidCameraMode,
  revision: number
): void {
  const points: THREE.Vector3[] = [];
  for (const x of [bounds.min.x, bounds.max.x]) {
    for (const y of [bounds.min.y, bounds.max.y]) {
      for (const z of [bounds.min.z, bounds.max.z]) points.push(new THREE.Vector3(x, y, z));
    }
  }
  camera.updateMatrixWorld(true);
  const projected = points.map((point) => point.project(camera));
  const left = Math.min(...projected.map((point) => (point.x + 1) * host.clientWidth / 2));
  const right = Math.max(...projected.map((point) => (point.x + 1) * host.clientWidth / 2));
  const top = Math.min(...projected.map((point) => (1 - point.y) * host.clientHeight / 2));
  const bottom = Math.max(...projected.map((point) => (1 - point.y) * host.clientHeight / 2));
  canvas.dataset.robotScreenBounds = JSON.stringify({ left, right, top, bottom, mode, revision });
}
