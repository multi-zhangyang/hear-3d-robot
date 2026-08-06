import type { HumanoidWorld } from "./world.js";
import type {
  HumanoidFrameSink,
  HumanoidWorldSnapshot
} from "./world-contract.js";

type ContinuousWorld = Pick<
  HumanoidWorld,
  "advanceStationary" | "flushFramePublications" | "snapshot"
>;

export class HumanoidStationarySafetyError extends Error {
  readonly snapshot: HumanoidWorldSnapshot;

  constructor(snapshot: HumanoidWorldSnapshot) {
    super(`Humanoid lost its stationary safety envelope at frame ${snapshot.frame}`);
    this.name = "HumanoidStationarySafetyError";
    this.snapshot = structuredClone(snapshot);
  }
}

export class HumanoidPhysicsClock {
  readonly #world: ContinuousWorld;
  readonly #frameSink: HumanoidFrameSink;
  readonly #onError: (error: unknown) => void | Promise<void>;
  readonly #intervalMilliseconds: number;
  #timer: ReturnType<typeof setTimeout> | undefined;
  #inFlight: Promise<void> | undefined;
  #running = false;
  #failure: unknown;

  constructor(input: {
    world: ContinuousWorld;
    frameSink: HumanoidFrameSink;
    onError: (error: unknown) => void | Promise<void>;
  }) {
    this.#world = input.world;
    this.#frameSink = input.frameSink;
    this.#onError = input.onError;
    const controlStepSeconds = input.world.snapshot().robot.controller.controlStepSeconds;
    if (!Number.isFinite(controlStepSeconds) || controlStepSeconds <= 0) {
      throw new Error("Humanoid continuous physics requires a positive control step");
    }
    this.#intervalMilliseconds = controlStepSeconds * 1_000;
  }

  get running(): boolean {
    return this.#running;
  }

  get failure(): unknown {
    return this.#failure;
  }

  start(): void {
    if (this.#running) return;
    this.throwIfFailed();
    this.#running = true;
    this.#schedule(this.#intervalMilliseconds);
  }

  async stop(): Promise<void> {
    this.#running = false;
    if (this.#timer !== undefined) {
      clearTimeout(this.#timer);
      this.#timer = undefined;
    }
    await this.#inFlight;
    await this.#world.flushFramePublications();
  }

  throwIfFailed(): void {
    if (this.#failure !== undefined) throw this.#failure;
  }

  #schedule(delayMilliseconds: number): void {
    if (!this.#running || this.#timer !== undefined) return;
    this.#timer = setTimeout(() => {
      this.#timer = undefined;
      const operation = this.#tick();
      this.#inFlight = operation;
      void operation
        .catch(() => undefined)
        .finally(() => {
          if (this.#inFlight === operation) this.#inFlight = undefined;
        });
    }, delayMilliseconds);
  }

  async #tick(): Promise<void> {
    const startedAt = performance.now();
    try {
      const snapshot = await this.#world.advanceStationary(this.#frameSink);
      if (snapshot?.robot.fallen) throw new HumanoidStationarySafetyError(snapshot);
    } catch (error) {
      this.#failure = error;
      this.#running = false;
      try {
        await this.#onError(error);
      } catch (reportingError) {
        this.#failure = new AggregateError(
          [error, reportingError],
          "Humanoid continuous physics failed and its failure could not be recorded"
        );
      }
      return;
    }
    if (!this.#running) return;
    const elapsed = performance.now() - startedAt;
    this.#schedule(Math.max(0, this.#intervalMilliseconds - elapsed));
  }
}
