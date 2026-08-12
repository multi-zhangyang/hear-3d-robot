import {
  HumanoidEmbodiedSkillEventSchema,
  type HumanoidEmbodiedSkillEvent,
  type HumanoidEmbodiedSkillIdentity,
  type HumanoidEmbodiedSkillStatus
} from "./embodied-skill-call.js";

export type HumanoidSkillEventSink = (
  event: HumanoidEmbodiedSkillEvent
) => void | Promise<void>;

export interface HumanoidSkillProgressEvidence {
  elapsedRatio: number;
  physicalCompletionRatio: number | null;
  satisfiedPredicateRatio: number | null;
  stableSteps: number;
  requiredStableSteps: number | null;
  confidence: number;
}

export interface HumanoidSkillEventStreamState {
  nextSequence: number;
  accepted: boolean;
  lastProgress: number;
  lastStableSteps: number;
  terminal: boolean;
}

export function restoreHumanoidSkillEventStreamStates(
  events: readonly HumanoidEmbodiedSkillEvent[]
): Map<string, HumanoidSkillEventStreamState> {
  const states = new Map<string, HumanoidSkillEventStreamState>();
  for (const source of events) {
    const event = HumanoidEmbodiedSkillEventSchema.parse(source);
    const callId = event.status.callId;
    const state = states.get(callId) ?? initialSkillEventStreamState();
    if (event.sequence !== state.nextSequence) {
      throw new Error(`Skill Call ${callId} has a non-contiguous event sequence`);
    }
    if (state.terminal) {
      throw new Error(`Skill Call ${callId} emitted an event after its terminal status`);
    }
    if (event.type === "accepted") {
      if (state.accepted || event.sequence !== 0) {
        throw new Error(`Skill Call ${callId} has duplicate accepted authority`);
      }
      state.accepted = true;
    } else if (!state.accepted) {
      throw new Error(`Skill Call ${callId} emitted ${event.type} before accepted`);
    }
    if (event.type === "progress") {
      const progress = progressValue(event.status);
      if (progress < state.lastProgress) {
        throw new Error(`Skill Call ${callId} has regressing physical progress`);
      }
      state.lastProgress = progress;
      state.lastStableSteps = event.status.progress.stableSteps;
    }
    if (event.type === "succeeded" || event.type === "failed"
      || event.type === "interrupted") {
      state.terminal = true;
    }
    state.nextSequence += 1;
    states.set(callId, state);
  }
  return new Map([...states].map(([callId, state]) => [
    callId,
    structuredClone(state)
  ]));
}

export class HumanoidSkillEventStream {
  readonly #identity: HumanoidEmbodiedSkillIdentity;
  readonly #sink: HumanoidSkillEventSink | undefined;
  #sequence = 0;
  #lastProgress = -1;
  #lastStableSteps = -1;
  #accepted = false;
  #terminal = false;
  #writeTail: Promise<void> = Promise.resolve();

  constructor(
    identity: HumanoidEmbodiedSkillIdentity,
    sink?: HumanoidSkillEventSink,
    restoredState?: HumanoidSkillEventStreamState
  ) {
    this.#identity = structuredClone(identity);
    this.#sink = sink;
    if (restoredState) {
      assertSkillEventStreamState(restoredState);
      this.#sequence = restoredState.nextSequence;
      this.#lastProgress = restoredState.lastProgress;
      this.#lastStableSteps = restoredState.lastStableSteps;
      this.#accepted = restoredState.accepted;
      this.#terminal = restoredState.terminal;
    }
  }

  async accepted(status: HumanoidEmbodiedSkillStatus): Promise<void> {
    await this.#enqueue(async () => {
      if (this.#accepted || this.#terminal) return;
      await this.#emit("accepted", status);
      this.#accepted = true;
    });
  }

  async progress(status: HumanoidEmbodiedSkillStatus): Promise<void> {
    await this.#enqueue(async () => {
      if (this.#terminal) return;
      const progress = progressValue(status);
      if (progress < this.#lastProgress
        || (progress < 1
          && this.#lastProgress >= 0
          && progress - this.#lastProgress < 0.1
          && status.progress.stableSteps === this.#lastStableSteps)) {
        return;
      }
      await this.#emit("progress", status);
      this.#lastProgress = Math.max(this.#lastProgress, progress);
      this.#lastStableSteps = status.progress.stableSteps;
    });
  }

  async terminal(
    type: "succeeded" | "failed" | "interrupted",
    status: HumanoidEmbodiedSkillStatus
  ): Promise<void> {
    await this.#enqueue(async () => {
      if (this.#terminal) return;
      await this.#emit(type, status);
      this.#terminal = true;
    });
  }

  async environmentChanged(status: HumanoidEmbodiedSkillStatus): Promise<void> {
    await this.#enqueue(async () => {
      if (this.#terminal) return;
      await this.#emit("environment_changed", status);
    });
  }

  async #enqueue(operation: () => Promise<void>): Promise<void> {
    const write = this.#writeTail.then(operation);
    this.#writeTail = write.catch(() => undefined);
    await write;
  }

  async #emit(
    type: HumanoidEmbodiedSkillEvent["type"],
    status: HumanoidEmbodiedSkillStatus
  ): Promise<void> {
    if (status.callId !== this.#identity.callId) {
      throw new Error("Skill status call identity does not match its event stream");
    }
    const event = HumanoidEmbodiedSkillEventSchema.parse({
      protocol: "humanoid-embodied-skill-event-v1",
      sequence: this.#sequence,
      type,
      status
    });
    await this.#sink?.(structuredClone(event));
    this.#sequence += 1;
  }
}

function initialSkillEventStreamState(): HumanoidSkillEventStreamState {
  return {
    nextSequence: 0,
    accepted: false,
    lastProgress: -1,
    lastStableSteps: -1,
    terminal: false
  };
}

function assertSkillEventStreamState(
  state: HumanoidSkillEventStreamState
): void {
  if (!Number.isSafeInteger(state.nextSequence) || state.nextSequence < 0
    || !Number.isFinite(state.lastProgress)
    || state.lastProgress < -1 || state.lastProgress > 1
    || !Number.isSafeInteger(state.lastStableSteps)
    || state.lastStableSteps < -1
    || (state.nextSequence === 0 && (state.accepted || state.terminal))
    || (state.terminal && !state.accepted)) {
    throw new Error("Restored Skill event stream state is invalid");
  }
}

function progressValue(status: HumanoidEmbodiedSkillStatus): number {
  return status.progress.physicalCompletionRatio
    ?? status.progress.satisfiedPredicateRatio
    ?? status.progress.elapsedRatio;
}
