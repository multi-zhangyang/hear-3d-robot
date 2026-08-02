import * as THREE from "three";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import {
  PMREMGenerator,
  WebGPURenderer
} from "three/webgpu";
import { disposeObject } from "../three-kit";

type StageRenderBackend = "webgpu" | "webgl2";

interface StageRenderRuntime {
  renderer: WebGPURenderer;
  backend: StageRenderBackend;
  environment: THREE.Texture;
}

interface RenderBackendCallbacks {
  onDeviceLost: (message: string) => void;
  onError: (message: string) => void;
}

/**
 * Initializes Three.js' current renderer abstraction. It selects WebGPU on a
 * capable client and lets the upstream renderer fall back to its WebGL2
 * backend when WebGPU is unavailable.
 */
export async function createRenderBackend(
  scene: THREE.Scene,
  callbacks: RenderBackendCallbacks,
  signal?: AbortSignal
): Promise<StageRenderRuntime> {
  signal?.throwIfAborted();
  const renderer = new WebGPURenderer({
    antialias: true,
    alpha: false,
    powerPreference: "high-performance"
  });
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  const markDeviceLost = renderer.onDeviceLost;
  renderer.onDeviceLost = (info) => {
    markDeviceLost.call(renderer, info);
    callbacks.onDeviceLost(deviceLossMessage(info));
  };
  renderer.onError = ((info: unknown) => {
    callbacks.onError(rendererErrorMessage(info));
  }) as typeof renderer.onError;

  try {
    await renderer.init();
    signal?.throwIfAborted();
    const backend = renderBackendKind(renderer.backend);
    renderer.domElement.dataset.renderBackend = backend;
    const environment = createEnvironment(scene, renderer);
    return { renderer, backend, environment };
  } catch (error) {
    renderer.dispose();
    throw error;
  }
}

export function renderBackendKind(backend: object): StageRenderBackend {
  if ("isWebGPUBackend" in backend && backend.isWebGPUBackend === true) return "webgpu";
  if ("isWebGLBackend" in backend && backend.isWebGLBackend === true) return "webgl2";
  throw new Error("Three.js initialized an unknown rendering backend");
}

function createEnvironment(scene: THREE.Scene, renderer: WebGPURenderer): THREE.Texture {
  const generator = new PMREMGenerator(renderer);
  const room = new RoomEnvironment();
  try {
    const environment = generator.fromScene(room, 0.04).texture;
    scene.environment = environment;
    scene.environmentIntensity = 0.22;
    return environment;
  } finally {
    disposeObject(room);
    generator.dispose();
  }
}

function deviceLossMessage(info: unknown): string {
  if (typeof info === "object" && info !== null && "message" in info
    && typeof info.message === "string" && info.message.trim() !== "") {
    return info.message;
  }
  return "The rendering device was lost";
}

function rendererErrorMessage(info: unknown): string {
  if (typeof info === "string" && info.trim() !== "") return info;
  if (typeof info === "object" && info !== null) {
    const message = "message" in info && typeof info.message === "string"
      ? info.message.trim()
      : "";
    const type = "type" in info && typeof info.type === "string"
      ? info.type.trim()
      : "";
    if (message) return type ? `${type}: ${message}` : message;
    if (type) return type;
  }
  return "The rendering backend reported an error";
}
