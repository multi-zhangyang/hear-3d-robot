import { afterEach, describe, expect, it, vi } from "vitest";
import { StageInteraction, nextFirstPersonLook, pointerToNdc } from "./stage-interaction";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("stage interaction", () => {
  it("maps canvas points to normalized ray coordinates", () => {
    const bounds = { left: 100, top: 50, width: 800, height: 400 };

    expect(pointerToNdc({ x: 500, y: 250 }, bounds).toArray()).toEqual([0, -0]);
    expect(pointerToNdc({ x: 100, y: 50 }, bounds).toArray()).toEqual([-1, 1]);
    expect(pointerToNdc({ x: 900, y: 450 }, bounds).toArray()).toEqual([1, -1]);
  });

  it("turns pointer deltas into a bounded observation offset", () => {
    const turned = nextFirstPersonLook({ yaw: 0, pitch: 0 }, 100, -50, 0.01);
    expect(turned.yaw).toBeCloseTo(-1);
    expect(turned.pitch).toBeCloseTo(0.5);

    const clamped = nextFirstPersonLook(turned, 0, -10_000, 0.01);
    expect(clamped.pitch).toBeLessThan(Math.PI / 2);
    expect(clamped.pitch).toBeGreaterThan(1.4);
  });

  it("uses a primary touch drag to look without turning it into a pick", () => {
    const canvas = new FakeCanvas();
    const fakeDocument = Object.assign(new EventTarget(), {
      pointerLockElement: null,
      exitPointerLock: vi.fn()
    });
    vi.stubGlobal("document", fakeDocument);
    vi.stubGlobal("window", {
      innerWidth: 390,
      matchMedia: () => ({ matches: true }),
      clearTimeout,
      setTimeout
    });
    vi.stubGlobal("navigator", { maxTouchPoints: 5 });
    const onLook = vi.fn();
    const onPick = vi.fn();
    const statuses: unknown[] = [];
    const interaction = new StageInteraction(canvas as unknown as HTMLCanvasElement, {
      getCameraMode: () => "sensor",
      onLook,
      onPick,
      onFirstPersonStatus: (status) => statuses.push(status)
    });

    canvas.dispatchEvent(pointerEvent("pointerdown", 11, 100, 100));
    canvas.dispatchEvent(pointerEvent("pointermove", 11, 140, 80));
    canvas.dispatchEvent(pointerEvent("pointermove", 11, 101, 101));
    canvas.dispatchEvent(pointerEvent("pointerup", 11, 101, 101));

    expect(interaction.look.yaw).toBeLessThan(0);
    expect(Math.abs(interaction.look.pitch)).toBeGreaterThan(0);
    expect(onLook).toHaveBeenCalledTimes(2);
    expect(onPick).not.toHaveBeenCalled();
    expect(canvas.captured).toEqual([11]);
    expect(statuses.at(-1)).toEqual({ available: true, locked: false, touch: true });
    interaction.dispose();
  });

  it("keeps a short touch gesture as a world pick", () => {
    const canvas = new FakeCanvas();
    vi.stubGlobal("document", Object.assign(new EventTarget(), {
      pointerLockElement: null,
      exitPointerLock: vi.fn()
    }));
    vi.stubGlobal("window", {
      innerWidth: 390,
      matchMedia: () => ({ matches: true }),
      clearTimeout,
      setTimeout
    });
    vi.stubGlobal("navigator", { maxTouchPoints: 5 });
    const onPick = vi.fn();
    const interaction = new StageInteraction(canvas as unknown as HTMLCanvasElement, {
      getCameraMode: () => "sensor",
      onLook: vi.fn(),
      onPick,
      onFirstPersonStatus: vi.fn()
    });

    canvas.dispatchEvent(pointerEvent("pointerdown", 3, 200, 120));
    canvas.dispatchEvent(pointerEvent("pointerup", 3, 202, 122));

    expect(onPick).toHaveBeenCalledOnce();
    interaction.dispose();
  });

  it("keeps mouse lock and touch look available together on a hybrid device", () => {
    const canvas = new FakeCanvas();
    const fakeDocument = Object.assign(new EventTarget(), {
      pointerLockElement: null as HTMLCanvasElement | null,
      exitPointerLock: vi.fn()
    });
    vi.stubGlobal("document", fakeDocument);
    vi.stubGlobal("window", {
      innerWidth: 1280,
      matchMedia: (query: string) => ({
        matches: query === "(hover: hover) and (pointer: fine)"
      }),
      clearTimeout,
      setTimeout
    });
    vi.stubGlobal("navigator", { maxTouchPoints: 5 });
    const statuses: unknown[] = [];
    const onLook = vi.fn();
    const interaction = new StageInteraction(canvas as unknown as HTMLCanvasElement, {
      getCameraMode: () => "sensor",
      onLook,
      onPick: vi.fn(),
      onFirstPersonStatus: (status) => statuses.push(status)
    });

    expect(statuses.at(-1)).toEqual({ available: true, locked: false, touch: false });
    canvas.dispatchEvent(pointerEvent("pointerdown", 7, 100, 100, "mouse"));
    canvas.dispatchEvent(pointerEvent("pointerup", 7, 100, 100, "mouse"));
    expect(canvas.requestPointerLock).toHaveBeenCalledOnce();

    canvas.dispatchEvent(pointerEvent("pointerdown", 8, 100, 100));
    canvas.dispatchEvent(pointerEvent("pointermove", 8, 130, 90));
    canvas.dispatchEvent(pointerEvent("pointerup", 8, 130, 90));
    expect(onLook).toHaveBeenCalledOnce();
    expect(statuses.at(-1)).toEqual({ available: true, locked: false, touch: true });
    interaction.dispose();
  });
});

class FakeCanvas extends EventTarget {
  readonly classList = {
    toggle: vi.fn(),
    remove: vi.fn()
  };
  readonly captured: number[] = [];
  readonly requestPointerLock = vi.fn(() => Promise.resolve());

  setPointerCapture(pointerId: number): void {
    this.captured.push(pointerId);
  }

  getBoundingClientRect(): DOMRect {
    return { left: 0, top: 0, width: 400, height: 240 } as DOMRect;
  }
}

function pointerEvent(
  type: string,
  pointerId: number,
  clientX: number,
  clientY: number,
  pointerType = "touch"
): Event {
  const event = new Event(type, { cancelable: true });
  Object.defineProperties(event, {
    pointerId: { value: pointerId },
    clientX: { value: clientX },
    clientY: { value: clientY },
    isPrimary: { value: true },
    button: { value: 0 },
    pointerType: { value: pointerType }
  });
  return event;
}
