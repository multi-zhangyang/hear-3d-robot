import { afterEach, describe, expect, it, vi } from "vitest";
import { HumanoidPhysicsClock, HumanoidStationarySafetyError } from "./physics-clock.js";
import type { HumanoidWorldSnapshot } from "./world-contract.js";

afterEach(() => {
  vi.useRealTimers();
});

describe("HumanoidPhysicsClock", () => {
  it("advances authoritative stationary frames without overlapping ticks", async () => {
    vi.useFakeTimers();
    let frame = 0;
    let active = 0;
    let maximumActive = 0;
    const published: number[] = [];
    const world = fakeWorld(async (sink) => {
      active += 1;
      maximumActive = Math.max(maximumActive, active);
      await Promise.resolve();
      frame += 1;
      const snapshot = fakeSnapshot(frame);
      await sink?.(snapshot);
      active -= 1;
      return snapshot;
    });
    const clock = new HumanoidPhysicsClock({
      world,
      frameSink: (snapshot) => {
        published.push(snapshot.frame);
      },
      onError: vi.fn()
    });

    clock.start();
    await vi.advanceTimersByTimeAsync(65);
    await clock.stop();

    expect(published.length).toBeGreaterThanOrEqual(3);
    expect(published).toEqual(Array.from({ length: published.length }, (_, index) => index + 1));
    expect(maximumActive).toBe(1);
    expect(clock.running).toBe(false);
  });

  it("keeps scheduling when authority temporarily yields no frame", async () => {
    vi.useFakeTimers();
    let calls = 0;
    const published: number[] = [];
    const world = fakeWorld(async (sink) => {
      calls += 1;
      if (calls === 2) return null;
      const snapshot = fakeSnapshot(calls === 1 ? 1 : 2);
      await sink?.(snapshot);
      return snapshot;
    });
    const clock = new HumanoidPhysicsClock({
      world,
      frameSink: (snapshot) => {
        published.push(snapshot.frame);
      },
      onError: vi.fn()
    });

    clock.start();
    await vi.advanceTimersByTimeAsync(70);
    await clock.stop();

    expect(calls).toBeGreaterThanOrEqual(3);
    expect(published).toEqual([1, 2]);
  });

  it("stops and exposes a real stationary safety failure", async () => {
    vi.useFakeTimers();
    const fallen = fakeSnapshot(1, true);
    const onError = vi.fn();
    const clock = new HumanoidPhysicsClock({
      world: fakeWorld(async (sink) => {
        await sink?.(fallen);
        return fallen;
      }),
      frameSink: vi.fn(),
      onError
    });

    clock.start();
    await vi.advanceTimersByTimeAsync(20);

    expect(clock.running).toBe(false);
    expect(clock.failure).toBeInstanceOf(HumanoidStationarySafetyError);
    expect(() => clock.throwIfFailed()).toThrow(HumanoidStationarySafetyError);
    expect(() => clock.start()).toThrow(HumanoidStationarySafetyError);
    expect(onError).toHaveBeenCalledTimes(1);
    await clock.stop();
  });
});

function fakeWorld(
  advance: (
    sink?: (snapshot: HumanoidWorldSnapshot) => void | Promise<void>
  ) => Promise<HumanoidWorldSnapshot | null>
): Pick<
  import("./world.js").HumanoidWorld,
  "advanceStationary" | "flushFramePublications" | "snapshot"
> {
  const initial = fakeSnapshot(0);
  return {
    advanceStationary: advance,
    flushFramePublications: async () => undefined,
    snapshot: () => initial
  };
}

function fakeSnapshot(frame: number, fallen = false): HumanoidWorldSnapshot {
  return {
    frame,
    worldRevision: frame,
    motionGenerator: {
      protocol: "humanoid-motion-generator-v1",
      implementation: "test",
      motionClass: "constraint_solver",
      sampling: "deterministic"
    },
    robot: {
      simulatedTime: frame * 0.02,
      controller: {
        protocol: "humanoid-controller-v1",
        implementation: "test",
        controlStepSeconds: 0.02,
        physicsStepSeconds: 0.005,
        actuation: "joint_position_pd"
      },
      rootPosition: { x: 0, y: 0.8, z: 0 },
      rootRotation: { x: 0, y: 0, z: 0, w: 1 },
      joints: {},
      links: {} as HumanoidWorldSnapshot["robot"]["links"],
      objects: {},
      contactCount: 0,
      contacts: [],
      feet: {
        left: { touching: true, contactCount: 1, normalForce: 200, points: [] },
        right: { touching: true, contactCount: 1, normalForce: 200, points: [] }
      },
      balance: {
        centerOfMass: { x: 0, y: 0.7, z: 0 },
        support: "double",
        supportMargin: 0.1,
        upright: fallen ? 0.2 : 1
      },
      nonFootEnvironmentContacts: [],
      fallen
    },
    navigation: {
      planId: null,
      status: "idle",
      target: null,
      waypoints: [],
      waypointIndex: null
    }
  };
}
