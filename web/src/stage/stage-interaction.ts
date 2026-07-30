import * as THREE from "three";
import type { CameraMode } from "./WorldHud";

export interface FirstPersonLook {
  yaw: number;
  pitch: number;
}

export interface FirstPersonStatus {
  available: boolean;
  locked: boolean;
}

interface StageInteractionOptions {
  getCameraMode: () => CameraMode;
  onPick: (pointer: THREE.Vector2) => void;
  onLook: () => void;
  onFirstPersonStatus: (status: FirstPersonStatus) => void;
}

interface PointerStart {
  id: number;
  x: number;
  y: number;
}

const LOOK_SENSITIVITY = 0.0022;
const MAX_PITCH = THREE.MathUtils.degToRad(82);
const MAX_CLICK_TRAVEL_SQUARED = 36;

export class StageInteraction {
  readonly #canvas: HTMLCanvasElement;
  readonly #options: StageInteractionOptions;
  readonly #pointerLockAvailable: boolean;
  #pointerStart: PointerStart | null = null;
  #look: FirstPersonLook = { yaw: 0, pitch: 0 };

  constructor(canvas: HTMLCanvasElement, options: StageInteractionOptions) {
    this.#canvas = canvas;
    this.#options = options;
    this.#pointerLockAvailable = typeof canvas.requestPointerLock === "function"
      && window.matchMedia?.("(hover: hover) and (pointer: fine)").matches === true
      && window.innerWidth > 560
      && navigator.maxTouchPoints === 0;
    canvas.addEventListener("pointerdown", this.#handlePointerDown);
    canvas.addEventListener("pointerup", this.#handlePointerUp);
    canvas.addEventListener("pointercancel", this.#handlePointerCancel);
    document.addEventListener("mousemove", this.#handleMouseMove);
    document.addEventListener("pointerlockchange", this.#handlePointerLockChange);
    this.#syncCanvasState();
    this.#publishStatus();
  }

  get look(): FirstPersonLook {
    return this.#look;
  }

  setCameraMode(mode: CameraMode): void {
    if (mode !== "sensor" && document.pointerLockElement === this.#canvas) {
      document.exitPointerLock();
    }
    this.#syncCanvasState(mode);
    this.#publishStatus();
  }

  resetLook(): void {
    this.#look = { yaw: 0, pitch: 0 };
    this.#options.onLook();
  }

  dispose(): void {
    this.#canvas.removeEventListener("pointerdown", this.#handlePointerDown);
    this.#canvas.removeEventListener("pointerup", this.#handlePointerUp);
    this.#canvas.removeEventListener("pointercancel", this.#handlePointerCancel);
    document.removeEventListener("mousemove", this.#handleMouseMove);
    document.removeEventListener("pointerlockchange", this.#handlePointerLockChange);
    if (document.pointerLockElement === this.#canvas) document.exitPointerLock();
    this.#canvas.classList.remove("first-person-ready", "first-person-locked");
  }

  readonly #handlePointerDown = (event: PointerEvent): void => {
    if (!event.isPrimary || event.button !== 0) return;
    this.#pointerStart = { id: event.pointerId, x: event.clientX, y: event.clientY };
  };

  readonly #handlePointerUp = (event: PointerEvent): void => {
    const start = this.#pointerStart;
    this.#pointerStart = null;
    if (!start || start.id !== event.pointerId || !event.isPrimary || event.button !== 0) return;
    const travelled = (event.clientX - start.x) ** 2 + (event.clientY - start.y) ** 2;
    if (travelled > MAX_CLICK_TRAVEL_SQUARED) return;

    const firstPerson = this.#options.getCameraMode() === "sensor";
    const locked = document.pointerLockElement === this.#canvas;
    this.#options.onPick(pointerToNdc(
      locked && firstPerson
        ? canvasCenter(this.#canvas.getBoundingClientRect())
        : { x: event.clientX, y: event.clientY },
      this.#canvas.getBoundingClientRect()
    ));
    if (firstPerson && this.#pointerLockAvailable && !locked && event.pointerType !== "touch") {
      this.#requestPointerLock();
    }
  };

  readonly #handlePointerCancel = (): void => {
    this.#pointerStart = null;
  };

  readonly #handleMouseMove = (event: MouseEvent): void => {
    if (document.pointerLockElement !== this.#canvas
      || this.#options.getCameraMode() !== "sensor") return;
    this.#look = nextFirstPersonLook(this.#look, event.movementX, event.movementY);
    this.#options.onLook();
  };

  readonly #handlePointerLockChange = (): void => {
    this.#syncCanvasState();
    this.#publishStatus();
  };

  #requestPointerLock(): void {
    try {
      const requested = this.#canvas.requestPointerLock();
      if (requested) void requested.catch(() => undefined);
    } catch {
      this.#publishStatus();
    }
  }

  #syncCanvasState(mode = this.#options.getCameraMode()): void {
    const firstPerson = mode === "sensor";
    const locked = document.pointerLockElement === this.#canvas;
    this.#canvas.classList.toggle("first-person-ready", firstPerson && this.#pointerLockAvailable && !locked);
    this.#canvas.classList.toggle("first-person-locked", firstPerson && locked);
  }

  #publishStatus(): void {
    this.#options.onFirstPersonStatus({
      available: this.#pointerLockAvailable,
      locked: document.pointerLockElement === this.#canvas
    });
  }
}

export function nextFirstPersonLook(
  current: FirstPersonLook,
  movementX: number,
  movementY: number,
  sensitivity = LOOK_SENSITIVITY
): FirstPersonLook {
  const yaw = normalizeAngle(current.yaw - movementX * sensitivity);
  const pitch = THREE.MathUtils.clamp(
    current.pitch - movementY * sensitivity,
    -MAX_PITCH,
    MAX_PITCH
  );
  return { yaw, pitch };
}

export function pointerToNdc(
  point: { x: number; y: number },
  bounds: Pick<DOMRect, "left" | "top" | "width" | "height">
): THREE.Vector2 {
  const width = Math.max(1, bounds.width);
  const height = Math.max(1, bounds.height);
  return new THREE.Vector2(
    (point.x - bounds.left) / width * 2 - 1,
    -((point.y - bounds.top) / height * 2 - 1)
  );
}

function canvasCenter(bounds: Pick<DOMRect, "left" | "top" | "width" | "height">): {
  x: number;
  y: number;
} {
  return { x: bounds.left + bounds.width / 2, y: bounds.top + bounds.height / 2 };
}

function normalizeAngle(value: number): number {
  return THREE.MathUtils.euclideanModulo(value + Math.PI, Math.PI * 2) - Math.PI;
}
