import { serializeHumanoidReference } from "./motion-artifact.js";
import type {
  HumanoidMotionGenerator,
  HumanoidMotionGeneratorInput
} from "./motion-plan.js";

export class HoldingMotionGenerator implements HumanoidMotionGenerator {
  readonly descriptor = {
    protocol: "humanoid-motion-generator-v1" as const,
    implementation: "contract_test_generator",
    motionClass: "constraint_solver" as const,
    sampling: "deterministic" as const
  };
  calls = 0;
  disposed = false;

  constructor(private readonly invalidCadence = false) {}

  async generate(input: HumanoidMotionGeneratorInput) {
    this.calls += 1;
    const count = Math.ceil(input.plan.duration_seconds / input.controlStepSeconds);
    return {
      version: 1 as const,
      protocol: "humanoid-motion-v1" as const,
      generator: this.descriptor.implementation,
      controlStepSeconds: input.controlStepSeconds,
      durationSeconds: input.plan.duration_seconds,
      frames: Array.from({ length: count }, (_, index) => ({
        atSeconds: this.invalidCadence && index === 0
          ? input.controlStepSeconds / 2
          : Math.min(
              (index + 1) * input.controlStepSeconds,
              input.plan.duration_seconds
            ),
        reference: serializeHumanoidReference(input.baseline)
      }))
    };
  }

  async dispose(): Promise<void> {
    this.disposed = true;
  }
}

export class BlockingFirstMotionGenerator extends HoldingMotionGenerator {
  readonly #entered = deferredSignal();
  readonly #release = deferredSignal();
  #blocked = false;

  get firstCallEntered(): Promise<void> {
    return this.#entered.promise;
  }

  releaseFirstCall(): void {
    this.#release.resolve();
  }

  override async generate(input: HumanoidMotionGeneratorInput) {
    if (!this.#blocked) {
      this.#blocked = true;
      this.#entered.resolve();
      await this.#release.promise;
    }
    return super.generate(input);
  }
}

function deferredSignal(): { promise: Promise<void>; resolve(): void } {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
