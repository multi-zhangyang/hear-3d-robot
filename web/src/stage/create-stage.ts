import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { addEnvironment, disposeObject } from "../three-kit";
import type { ScenarioDefinition, Vec3, WorldSnapshot } from "../types";
import type { AuthoritativeFrameBuffer } from "./authoritative-frame-buffer";
import { fitRobot, fitSensor, fitWorld, focusKey } from "./camera";
import { renderPixelRatio } from "./render-quality";
import { addStudioLighting, STAGE_VOID } from "./scene-primitives";
import { StageInteraction } from "./stage-interaction";
import type { StageCallbacks, StageController } from "./stage-types";
import type { CameraMode } from "./WorldHud";
import { WorldScene } from "./WorldScene";
import {
  clearChaseCameraSightline,
  easeChaseCameraHeight,
  VoxelSurfaceHeightField
} from "./chase-camera-clearance";

export function createStage(
  host: HTMLDivElement,
  scenario: ScenarioDefinition,
  initialFrame: WorldSnapshot,
  frameBuffer: AuthoritativeFrameBuffer,
  initiallyLive: boolean,
  callbacks: StageCallbacks
): StageController {
  const bounds = scenario.bounds;
  const worldExtent = Math.max(bounds.width, bounds.depth);
  if (!window.WebGL2RenderingContext && !window.WebGLRenderingContext) {
    throw new Error("This browser does not expose a WebGL context");
  }
  if (!frameBuffer.latest) frameBuffer.reset(initialFrame);

  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: false,
    preserveDrawingBuffer: false,
    powerPreference: "high-performance"
  });
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.setClearColor(STAGE_VOID, 1);
  renderer.domElement.className = "three-canvas";

  let contextLost = false;
  let reportedError: string | null = null;
  const reportError = (message: string | null): void => {
    if (reportedError === message) return;
    reportedError = message;
    callbacks.onError(message);
  };
  const handleContextLost = (event: Event): void => {
    event.preventDefault();
    contextLost = true;
    reportError("The rendering context was lost");
  };
  const handleContextRestored = (): void => {
    contextLost = false;
    reportError(null);
    scheduleRender();
  };
  renderer.domElement.addEventListener("webglcontextlost", handleContextLost);
  renderer.domElement.addEventListener("webglcontextrestored", handleContextRestored);
  host.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(STAGE_VOID);
  const robotFogNear = Math.max(12, worldExtent * 0.38);
  const robotFog = new THREE.Fog(STAGE_VOID, robotFogNear, Math.max(robotFogNear + 22, worldExtent * 1.65));
  const worldFog = new THREE.Fog(
    0x111821,
    Math.max(20, worldExtent * 1.2),
    Math.max(84, worldExtent * 6)
  );
  scene.fog = robotFog;
  const environment = addEnvironment(scene, renderer);

  const camera = new THREE.PerspectiveCamera(38, 1, 0.05, Math.max(120, worldExtent * 5));
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = false;
  controls.maxPolarAngle = Math.PI * 0.495;
  controls.minDistance = 1.6;
  controls.maxDistance = Math.max(48, worldExtent * 4.5);

  addStudioLighting(scene, bounds);
  const world = new WorldScene(scene, scenario, callbacks.onSelection);
  let currentSnapshot = frameBuffer.latest ?? initialFrame;
  world.update(currentSnapshot);
  const terrainSurface = scenario.terrain
    ? new VoxelSurfaceHeightField(scenario.terrain)
    : null;
  terrainSurface?.update(currentSnapshot.voxels);
  let lastSynchronizedVersion = frameBuffer.version;
  let cameraMode: CameraMode = "robot";
  let canvasWidth = 1;
  let live = initiallyLive;
  let fitted = false;
  let lastRobotPosition: Vec3 | null = null;
  let lastFocusKey = "";
  let animationFrame = 0;
  let disposed = false;
  let internalCameraUpdate = false;
  let cameraSettling = false;
  let preferredRobotCamera: THREE.Vector3 | null = null;
  let preferredRobotTarget: THREE.Vector3 | null = null;
  let robotSightlineTarget: THREE.Vector3 | null = null;
  const interaction = new StageInteraction(renderer.domElement, {
    getCameraMode: () => cameraMode,
    onPick: (pointer) => {
      world.pick(camera, pointer);
      scheduleRender();
    },
    onLook: scheduleRender,
    onFirstPersonStatus: callbacks.onFirstPersonStatus
  });

  const withInternalCameraUpdate = (update: () => void): void => {
    internalCameraUpdate = true;
    try {
      update();
    } finally {
      internalCameraUpdate = false;
    }
  };

  const applyRobotClearance = (instant: boolean): void => {
    if (!preferredRobotCamera || !preferredRobotTarget || !robotSightlineTarget) {
      cameraSettling = false;
      return;
    }
    const resolved = terrainSurface
      ? clearChaseCameraSightline({
          preferred: preferredRobotCamera,
          target: robotSightlineTarget,
          terrain: terrainSurface,
          aspect: camera.aspect,
          targetRadius: 0.78
        })
      : preferredRobotCamera;
    const currentY = camera.position.y;
    camera.position.copy(preferredRobotCamera);
    camera.position.y = instant
      ? resolved.y
      : easeChaseCameraHeight(currentY, resolved.y);
    cameraSettling = !instant && Math.abs(camera.position.y - resolved.y) > 0.005;
    controls.target.copy(preferredRobotTarget);
    controls.update();
  };

  const fitSnapshot = (snapshot: WorldSnapshot): void => {
    const fittedBounds = cameraMode === "robot" ? world.robotBounds() : world.worldBounds(snapshot);
    withInternalCameraUpdate(() => {
      if (cameraMode === "robot") {
        fitRobot(camera, controls, fittedBounds, snapshot.robot.yaw);
        preferredRobotCamera = camera.position.clone();
        preferredRobotTarget = controls.target.clone();
        robotSightlineTarget = fittedBounds.getCenter(new THREE.Vector3());
        robotSightlineTarget.y = fittedBounds.min.y + Math.min(0.12, fittedBounds.getSize(new THREE.Vector3()).y * 0.1);
        applyRobotClearance(true);
      }
      else if (cameraMode === "sensor") fitSensor(camera, controls, snapshot, interaction.look);
      else fitWorld(camera, controls, fittedBounds, bounds);
      if (cameraMode !== "robot") cameraSettling = false;
    });
    fitted = true;
    lastRobotPosition = { ...snapshot.robot.position };
    lastFocusKey = focusKey(snapshot);
  };

  const followSnapshot = (snapshot: WorldSnapshot): void => {
    if (!fitted) {
      fitSnapshot(snapshot);
      return;
    }
    withInternalCameraUpdate(() => {
      if (cameraMode === "sensor") {
        fitSensor(camera, controls, snapshot, interaction.look);
      } else if (cameraMode === "robot" && lastRobotPosition) {
        const delta = new THREE.Vector3(
          snapshot.robot.position.x - lastRobotPosition.x,
          snapshot.robot.position.y - lastRobotPosition.y,
          snapshot.robot.position.z - lastRobotPosition.z
        );
        camera.position.add(delta);
        preferredRobotCamera?.add(delta);
        preferredRobotTarget?.add(delta);
        robotSightlineTarget?.add(delta);
        applyRobotClearance(false);
      } else if (cameraMode === "world" && focusKey(snapshot) !== lastFocusKey) {
        fitSnapshot(snapshot);
      }
    });
    lastRobotPosition = { ...snapshot.robot.position };
    lastFocusKey = focusKey(snapshot);
  };

  function scheduleRender(): void {
    if (disposed || animationFrame !== 0) return;
    animationFrame = window.requestAnimationFrame(renderFrame);
  }

  function renderFrame(wallTimeMs: number): void {
    animationFrame = 0;
    if (disposed) return;
    try {
      const authoritative = frameBuffer.latest ?? currentSnapshot;
      if (lastSynchronizedVersion !== frameBuffer.version) {
        world.update(authoritative);
        terrainSurface?.update(authoritative.voxels);
        lastSynchronizedVersion = frameBuffer.version;
      }
      const visual = frameBuffer.sample(wallTimeMs, live) ?? authoritative;
      world.updatePose(visual);
      currentSnapshot = visual;
      followSnapshot(visual);
      if (!contextLost) {
        renderer.render(scene, camera);
        reportError(null);
      }
      if ((live && frameBuffer.pending) || cameraSettling) scheduleRender();
    } catch (error) {
      reportError(error instanceof Error ? error.message : String(error));
    }
  }

  const handleControlsChange = (): void => {
    if (!internalCameraUpdate) {
      if (cameraMode === "robot") {
        preferredRobotCamera = camera.position.clone();
        preferredRobotTarget = controls.target.clone();
        if (robotSightlineTarget) {
          robotSightlineTarget.x = controls.target.x;
          robotSightlineTarget.z = controls.target.z;
        }
      }
      scheduleRender();
    }
  };
  controls.addEventListener("change", handleControlsChange);

  const resize = (): void => {
    const width = Math.max(1, host.clientWidth);
    const height = Math.max(1, host.clientHeight);
    canvasWidth = width;
    renderer.domElement.style.touchAction = cameraMode === "sensor" || width > 560 ? "none" : "pan-y";
    renderer.setPixelRatio(renderPixelRatio(width, height, window.devicePixelRatio));
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    fitSnapshot(currentSnapshot);
    if (!contextLost) renderer.render(scene, camera);
  };

  const observer = new ResizeObserver(resize);
  observer.observe(host);
  const unsubscribeFrames = frameBuffer.subscribe(scheduleRender);

  resize();
  if (live && frameBuffer.pending) scheduleRender();
  return {
    setLive: (value) => {
      live = value;
      scheduleRender();
    },
    setCameraMode: (mode) => {
      cameraMode = mode;
      renderer.domElement.style.touchAction = mode === "sensor" || canvasWidth > 560 ? "none" : "pan-y";
      interaction.setCameraMode(mode);
      controls.enabled = mode !== "sensor";
      scene.fog = mode === "world" ? worldFog : robotFog;
      renderer.toneMappingExposure = mode === "world" ? 1.62 : mode === "sensor" ? 1.28 : 1.15;
      world.setLabelsVisible(mode === "robot");
      preferredRobotCamera = null;
      preferredRobotTarget = null;
      robotSightlineTarget = null;
      cameraSettling = false;
      fitted = false;
      scheduleRender();
    },
    fit: () => {
      if (cameraMode === "sensor") interaction.resetLook();
      fitSnapshot(currentSnapshot);
      if (!contextLost) renderer.render(scene, camera);
    },
    clearSelection: () => {
      world.clearSelection();
      scheduleRender();
    },
    dispose: () => {
      if (disposed) return;
      disposed = true;
      if (animationFrame !== 0) window.cancelAnimationFrame(animationFrame);
      unsubscribeFrames();
      observer.disconnect();
      interaction.dispose();
      controls.removeEventListener("change", handleControlsChange);
      controls.dispose();
      renderer.domElement.removeEventListener("webglcontextlost", handleContextLost);
      renderer.domElement.removeEventListener("webglcontextrestored", handleContextRestored);
      disposeObject(scene);
      environment.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    }
  };
}
