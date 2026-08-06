import type { HumanoidAuthorityFrameSource } from "./authority-loop.js";

interface Publication<Snapshot> {
  source: HumanoidAuthorityFrameSource;
  commandId: string | null;
  sink: (snapshot: Snapshot) => void | Promise<void>;
  snapshot: Snapshot;
}

interface CommandPublicationState {
  pending: number;
  closed: boolean;
  settled: boolean;
  error: unknown;
  barrier: Promise<void>;
  resolve: () => void;
  reject: (error: unknown) => void;
}

export class HumanoidAuthorityFramePublisher<Snapshot> {
  readonly #maximumQueuedFrames: number;
  readonly #onError: (error: unknown) => void;
  readonly #queue: Array<Publication<Snapshot>> = [];
  readonly #commands = new Map<string, CommandPublicationState>();
  #draining: Promise<void> | undefined;
  #disposed = false;

  constructor(input: {
    maximumQueuedFrames?: number;
    onError?: (error: unknown) => void;
  } = {}) {
    const maximum = input.maximumQueuedFrames ?? 256;
    if (!Number.isSafeInteger(maximum) || maximum <= 0) {
      throw new Error("Humanoid authority publication capacity must be positive");
    }
    this.#maximumQueuedFrames = maximum;
    this.#onError = input.onError ?? (() => undefined);
  }

  openCommand(commandId: string): Promise<void> {
    if (this.#disposed) throw new Error("Humanoid authority frame publisher is disposed");
    if (this.#commands.has(commandId)) {
      throw new Error(`Duplicate humanoid publication command: ${commandId}`);
    }
    const state = commandPublicationState();
    this.#commands.set(commandId, state);
    return state.barrier;
  }

  enqueue(input: {
    source: HumanoidAuthorityFrameSource;
    commandId: string | null;
    sink: (snapshot: Snapshot) => void | Promise<void>;
    snapshot: Snapshot;
  }): void {
    if (this.#disposed) return;
    if (input.commandId) {
      const state = this.#commands.get(input.commandId);
      if (!state) throw new Error(`Unknown humanoid publication command: ${input.commandId}`);
      if (state.error !== undefined) return;
      if (this.#queue.length >= this.#maximumQueuedFrames) {
        state.error = new Error(
          `Humanoid authority execution publication queue exceeded ${this.#maximumQueuedFrames} frames`
        );
        this.#onError(state.error);
        this.#settleCommand(input.commandId, state);
        return;
      }
      state.pending += 1;
    } else {
      const existing = this.#queue.findLastIndex((entry) => (
        entry.commandId === null && entry.sink === input.sink
      ));
      if (existing >= 0) {
        this.#queue[existing] = {
          source: input.source,
          commandId: null,
          sink: input.sink,
          snapshot: structuredClone(input.snapshot)
        };
        return;
      }
      if (this.#queue.length >= this.#maximumQueuedFrames) return;
    }
    this.#queue.push({
      source: input.source,
      commandId: input.commandId,
      sink: input.sink,
      snapshot: structuredClone(input.snapshot)
    });
    this.#startDrain();
  }

  closeCommand(commandId: string): void {
    const state = this.#commands.get(commandId);
    if (!state) return;
    state.closed = true;
    this.#settleCommand(commandId, state);
  }

  abortCommand(commandId: string): void {
    const state = this.#commands.get(commandId);
    if (!state) return;
    state.closed = true;
    this.#settleCommand(commandId, state);
  }

  async flush(): Promise<void> {
    while (this.#draining) await this.#draining;
  }

  async dispose(): Promise<void> {
    this.#disposed = true;
    await this.flush();
    for (const [commandId, state] of this.#commands) {
      state.closed = true;
      this.#settleCommand(commandId, state);
    }
  }

  #startDrain(): void {
    if (this.#draining) return;
    this.#draining = Promise.resolve()
      .then(() => this.#drain())
      .finally(() => {
        this.#draining = undefined;
        if (this.#queue.length > 0) this.#startDrain();
      });
  }

  async #drain(): Promise<void> {
    while (this.#queue.length > 0) {
      const publication = this.#queue.shift()!;
      const command = publication.commandId
        ? this.#commands.get(publication.commandId)
        : undefined;
      try {
        if (!command || command.error === undefined) {
          await publication.sink(publication.snapshot);
        }
      } catch (error) {
        if (command) command.error = error;
        this.#onError(error);
      } finally {
        if (command && publication.commandId) {
          command.pending -= 1;
          this.#settleCommand(publication.commandId, command);
        }
      }
    }
  }

  #settleCommand(commandId: string, state: CommandPublicationState): void {
    if (state.settled || !state.closed || state.pending > 0) return;
    state.settled = true;
    this.#commands.delete(commandId);
    if (state.error !== undefined) state.reject(state.error);
    else state.resolve();
  }
}

function commandPublicationState(): CommandPublicationState {
  let resolve!: () => void;
  let reject!: (error: unknown) => void;
  const barrier = new Promise<void>((onResolve, onReject) => {
    resolve = onResolve;
    reject = onReject;
  });
  void barrier.catch(() => undefined);
  return {
    pending: 0,
    closed: false,
    settled: false,
    error: undefined,
    barrier,
    resolve,
    reject
  };
}
