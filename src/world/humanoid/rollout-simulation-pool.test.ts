import { describe, expect, it } from "vitest";
import type { JsonValue } from "../../domain/schema.js";
import { HUMANOID_JOINT_NAMES } from "./model.js";
import { neutralHumanoidReference, type HumanoidReference } from "./reference.js";
import { HumanoidRolloutSimulationPool } from "./rollout-simulation-pool.js";
import { HumanoidSimulation } from "./simulation.js";
import type {
  HumanoidControllerDescriptor,
  HumanoidControllerState,
  HumanoidJointPositionCommand,
  HumanoidPolicyState,
  HumanoidWholeBodyController
} from "./whole-body-controller.js";

describe("HumanoidRolloutSimulationPool", () => {
  it("clones authoritative state into independent simulation and controller ownership", async () => {
    const authorityController = new RolloutTestController();
    const rolloutControllers: RolloutTestController[] = [];
    const zones = [{
      id: "assembly_bay",
      center: { x: 4.2, y: -0.025, z: 6.4 },
      size: { x: 1.4, y: 0.05, z: 1.4 }
    }];
    const authority = await HumanoidSimulation.create({
      zones,
      controllerFactory: async () => authorityController
    });
    const pool = await HumanoidRolloutSimulationPool.create({
      zones,
      controllerFactory: async () => {
        const controller = new RolloutTestController();
        rolloutControllers.push(controller);
        return controller;
      }
    });
    try {
      const reference = neutralHumanoidReference();
      await authority.step(reference);
      await authority.step(reference);
      const authoritativeState = authority.captureState();
      const authoritativeSnapshot = authority.snapshot();

      const rolledOut = await pool.lease(authoritativeState, async (simulation) => {
        expect(simulation).not.toBe(authority);
        expect(simulation.captureState()).toEqual(authoritativeState);
        await simulation.step(reference);
        return simulation.captureState();
      });

      expect(rolledOut.time - authoritativeState.time).toBeCloseTo(0.02, 9);
      expect(rolledOut.controller.payload).toEqual(expect.objectContaining({
        primary: expect.objectContaining({
          payload: { inference_count: 3 }
        })
      }));
      expect(authority.captureState()).toEqual(authoritativeState);
      expect(authority.snapshot()).toEqual(authoritativeSnapshot);
      expect(authorityController.inferenceCount).toBe(2);
      expect(authorityController.seenZoneIds).toEqual(["assembly_bay"]);
      expect(rolloutControllers).toHaveLength(1);
      expect(rolloutControllers[0]).not.toBe(authorityController);
      expect(rolloutControllers[0]!.seenZoneIds).toEqual(["assembly_bay"]);
    } finally {
      await pool.dispose();
      await authority.dispose();
    }
    expect(authorityController.disposed).toBe(true);
    expect(rolloutControllers[0]!.disposed).toBe(true);
  }, 30_000);

  it("recovers its reusable simulation after operation and state restore failures", async () => {
    const authority = await HumanoidSimulation.create({
      controllerFactory: async () => new RolloutTestController()
    });
    const pool = await HumanoidRolloutSimulationPool.create({
      controllerFactory: async () => new RolloutTestController()
    });
    try {
      const reference = neutralHumanoidReference();
      await authority.step(reference);
      const authoritativeState = authority.captureState();
      const failure = new Error("rollout failed");

      await expect(pool.lease(authoritativeState, async (simulation) => {
        await simulation.step(reference);
        throw failure;
      })).rejects.toBe(failure);
      expect(authority.captureState()).toEqual(authoritativeState);
      await expect(pool.lease(
        { ...authoritativeState, velocities: new Float64Array(1) },
        () => undefined
      )).rejects.toThrow("invalid velocities length");

      const recovered = await pool.lease(
        authoritativeState,
        (simulation) => simulation.captureState()
      );
      expect(recovered).toEqual(authoritativeState);
      expect(authority.captureState()).toEqual(authoritativeState);
    } finally {
      await pool.dispose();
      await authority.dispose();
    }
  }, 30_000);

  it("serializes leases and waits for an active lease before disposal", async () => {
    const controllers: RolloutTestController[] = [];
    const authority = await HumanoidSimulation.create({
      controllerFactory: async () => new RolloutTestController()
    });
    const pool = await HumanoidRolloutSimulationPool.create({
      controllerFactory: async () => {
        const controller = new RolloutTestController();
        controllers.push(controller);
        return controller;
      }
    });
    const firstEntered = deferred();
    const releaseFirst = deferred();
    const disposalEntered = deferred();
    const releaseDisposal = deferred();
    let active = 0;
    let maximumActive = 0;
    const order: string[] = [];
    try {
      const state = authority.captureState();
      const first = pool.lease(state, async () => {
        active += 1;
        maximumActive = Math.max(maximumActive, active);
        order.push("first-entered");
        firstEntered.resolve();
        await releaseFirst.promise;
        order.push("first-finished");
        active -= 1;
      });
      await firstEntered.promise;
      const second = pool.lease(state, async () => {
        active += 1;
        maximumActive = Math.max(maximumActive, active);
        order.push("second-entered");
        active -= 1;
      });
      await Promise.resolve();
      expect(order).toEqual(["first-entered"]);
      releaseFirst.resolve();
      await Promise.all([first, second]);
      expect(order).toEqual(["first-entered", "first-finished", "second-entered"]);
      expect(maximumActive).toBe(1);

      const activeLease = pool.lease(state, async () => {
        disposalEntered.resolve();
        await releaseDisposal.promise;
      });
      await disposalEntered.promise;
      const disposal = pool.dispose();
      await expect(pool.lease(state, () => undefined)).rejects.toThrow(
        "closing or disposed"
      );
      expect(controllers[0]!.disposed).toBe(false);
      releaseDisposal.resolve();
      await activeLease;
      await disposal;
      await pool.dispose();
      expect(controllers[0]!.disposed).toBe(true);
    } finally {
      releaseFirst.resolve();
      releaseDisposal.resolve();
      await pool.dispose();
      await authority.dispose();
    }
  }, 30_000);
});

class RolloutTestController implements HumanoidWholeBodyController {
  readonly descriptor: HumanoidControllerDescriptor = {
    protocol: "humanoid-controller-v1",
    implementation: "rollout_pool_test_pd",
    actuation: "joint_position_pd",
    controlStepSeconds: 0.02,
    physicsStepSeconds: 0.005,
    learnedPolicy: {
      protocol: "humanoid-learned-policy-v1",
      runtime: "test",
      observationSpace: {
        protocol: "rollout-pool-environment-test-v1",
        size: 1
      },
      actionSpace: {
        protocol: "rollout-pool-joint-position-test-v1",
        size: HUMANOID_JOINT_NAMES.length
      },
      observationFeatures: ["object_state"],
      capabilities: ["joint_reference_tracking"]
    }
  };
  inferenceCount = 0;
  disposed = false;
  seenZoneIds: string[] = [];

  reset(_state: HumanoidPolicyState, _reference: HumanoidReference): void {
    this.inferenceCount = 0;
  }

  async infer(
    state: HumanoidPolicyState,
    reference: HumanoidReference
  ): Promise<HumanoidJointPositionCommand> {
    this.inferenceCount += 1;
    this.seenZoneIds = state.environment?.zones?.map(({ id }) => id) ?? [];
    return {
      kind: "joint_position_pd",
      positions: Float64Array.from(reference.jointPositions),
      stiffness: new Float64Array(HUMANOID_JOINT_NAMES.length),
      damping: new Float64Array(HUMANOID_JOINT_NAMES.length)
    };
  }

  advanceHistory(_state: HumanoidPolicyState, _reference: HumanoidReference): void {}

  captureState(): HumanoidControllerState {
    return {
      protocol: "humanoid-controller-state-v1",
      version: 1,
      implementation: this.descriptor.implementation,
      payload: { inference_count: this.inferenceCount }
    };
  }

  restoreState(state: HumanoidControllerState): void {
    const inferenceCount = record(state.payload).inference_count;
    if (state.protocol !== "humanoid-controller-state-v1"
      || state.version !== 1
      || state.implementation !== this.descriptor.implementation
      || !Number.isSafeInteger(inferenceCount)
      || (inferenceCount as number) < 0) {
      throw new Error("Invalid rollout test controller state");
    }
    this.inferenceCount = inferenceCount as number;
  }

  async dispose(): Promise<void> {
    this.disposed = true;
  }
}

function deferred(): {
  promise: Promise<void>;
  resolve: () => void;
} {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function record(value: JsonValue): Record<string, JsonValue> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Invalid rollout test controller payload");
  }
  return value;
}
