import { Mutex } from "async-mutex";
import {
  HumanoidSimulation,
  type HumanoidSimulationOptions,
  type HumanoidSimulationState
} from "./simulation.js";
import { resolveHumanoidSceneObject } from "./mujoco-runtime.js";

type LeaseOperation<T> = (
  simulation: HumanoidSimulation
) => T | Promise<T>;

type LeaseOutcome<T> = {
  ok: true;
  value: T;
} | {
  ok: false;
  error: unknown;
};

export class HumanoidRolloutSimulationPool {
  readonly #options: HumanoidSimulationOptions;
  readonly #mutex = new Mutex();
  #simulation: HumanoidSimulation | undefined;
  #parkingState: HumanoidSimulationState | undefined;
  #closing = false;
  #disposed = false;
  #failure: unknown;
  #disposePromise: Promise<void> | undefined;

  static async create(
    options: HumanoidSimulationOptions = {}
  ): Promise<HumanoidRolloutSimulationPool> {
    const ownedOptions = cloneSimulationOptions(options);
    const simulation = await HumanoidSimulation.create(ownedOptions);
    return new HumanoidRolloutSimulationPool(ownedOptions, simulation);
  }

  private constructor(
    options: HumanoidSimulationOptions,
    simulation: HumanoidSimulation
  ) {
    this.#options = options;
    this.#simulation = simulation;
    this.#parkingState = cloneSimulationState(simulation.captureState());
  }

  async lease<T>(
    authoritativeState: HumanoidSimulationState,
    operation: LeaseOperation<T>
  ): Promise<T> {
    this.#assertAcceptingLeases();
    const seed = cloneSimulationState(authoritativeState);
    return this.#mutex.runExclusive(async () => {
      const simulation = this.#requiredSimulation();
      let outcome: LeaseOutcome<T>;
      try {
        simulation.restoreState(seed);
        outcome = { ok: true, value: await operation(simulation) };
      } catch (error) {
        outcome = { ok: false, error };
      }

      let recoveryError: unknown;
      try {
        await this.#recoverSimulation();
      } catch (error) {
        recoveryError = error;
      }

      if (!outcome.ok) {
        if (recoveryError !== undefined) {
          throw new AggregateError(
            [outcome.error, recoveryError],
            "Humanoid rollout operation and simulation recovery both failed"
          );
        }
        throw outcome.error;
      }
      if (recoveryError !== undefined) throw recoveryError;
      return outcome.value;
    });
  }

  dispose(): Promise<void> {
    if (this.#disposePromise) return this.#disposePromise;
    this.#closing = true;
    this.#disposePromise = this.#mutex.runExclusive(async () => {
      const simulation = this.#simulation;
      this.#simulation = undefined;
      this.#parkingState = undefined;
      this.#disposed = true;
      if (simulation) await simulation.dispose();
    });
    return this.#disposePromise;
  }

  #assertAcceptingLeases(): void {
    if (this.#closing || this.#disposed) {
      throw new Error("Humanoid rollout simulation pool is closing or disposed");
    }
    if (this.#failure !== undefined) {
      throw new AggregateError(
        [this.#failure],
        "Humanoid rollout simulation pool is unavailable"
      );
    }
  }

  #requiredSimulation(): HumanoidSimulation {
    if (this.#failure !== undefined) {
      throw new AggregateError(
        [this.#failure],
        "Humanoid rollout simulation pool is unavailable"
      );
    }
    if (!this.#simulation || !this.#parkingState || this.#disposed) {
      throw new Error("Humanoid rollout simulation pool is disposed");
    }
    return this.#simulation;
  }

  async #recoverSimulation(): Promise<void> {
    const simulation = this.#requiredSimulation();
    const parkingState = this.#parkingState!;
    try {
      simulation.restoreState(parkingState);
      return;
    } catch (restoreError) {
      this.#simulation = undefined;
      this.#parkingState = undefined;
      let disposalError: unknown;
      try {
        await simulation.dispose();
      } catch (error) {
        disposalError = error;
      }
      try {
        const replacement = await HumanoidSimulation.create(this.#options);
        this.#simulation = replacement;
        this.#parkingState = cloneSimulationState(replacement.captureState());
      } catch (replacementError) {
        const errors = disposalError === undefined
          ? [restoreError, replacementError]
          : [restoreError, disposalError, replacementError];
        this.#failure = new AggregateError(
          errors,
          "Humanoid rollout simulation pool could not replace a damaged simulation"
        );
        throw this.#failure;
      }
      const errors = disposalError === undefined
        ? [restoreError]
        : [restoreError, disposalError];
      throw new AggregateError(
        errors,
        "Humanoid rollout simulation was replaced after recovery failed"
      );
    }
  }
}

function cloneSimulationOptions(
  options: HumanoidSimulationOptions
): HumanoidSimulationOptions {
  return {
    ...(options.spawn
      ? {
          spawn: {
            position: { ...options.spawn.position },
            yaw: options.spawn.yaw
          }
        }
      : {}),
    ...(options.solids
      ? {
          solids: options.solids.map((solid) => ({
            id: solid.id,
            center: { ...solid.center },
            size: { ...solid.size }
          }))
        }
      : {}),
    ...(options.objects
      ? {
          objects: options.objects.map(resolveHumanoidSceneObject)
        }
      : {}),
    ...(options.controllerFactory
      ? { controllerFactory: options.controllerFactory }
      : {})
  };
}

function cloneSimulationState(
  state: HumanoidSimulationState
): HumanoidSimulationState {
  return {
    time: state.time,
    positions: state.positions.slice(),
    velocities: state.velocities.slice(),
    controls: state.controls.slice(),
    activations: state.activations.slice(),
    accelerationWarmstart: state.accelerationWarmstart.slice(),
    ...(state.requestedActuatorTorques
      ? { requestedActuatorTorques: state.requestedActuatorTorques.slice() }
      : {}),
    ...(state.handCommandTargets
      ? { handCommandTargets: state.handCommandTargets.slice() }
      : {}),
    controller: structuredClone(state.controller)
  };
}
