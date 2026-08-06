import { Mutex } from "async-mutex";
import type { HumanoidAuthorityIdentity } from "./authority-state.js";
import { HumanoidAuthorityFramePublisher } from "./authority-frame-publisher.js";

export type HumanoidAuthorityFrameSource = "stationary" | "motion" | "navigation";

interface HumanoidAuthorityCommandStep<Snapshot, Result> {
  snapshot?: Snapshot;
  done: boolean;
  result?: Result;
}

export interface HumanoidAuthorityCommand<Snapshot, Result> {
  id: string;
  source: Exclude<HumanoidAuthorityFrameSource, "stationary">;
  admission: HumanoidAuthorityIdentity;
  frameSink?: (snapshot: Snapshot) => void | Promise<void>;
  admit(): void;
  step(): Promise<HumanoidAuthorityCommandStep<Snapshot, Result>>;
}

export interface HumanoidAuthorityCommandHandle<Result> {
  readonly id: string;
  readonly result: Promise<Result>;
  readonly publication: Promise<void>;
  readonly settled: boolean;
}

export interface HumanoidAuthorityTick<Snapshot> {
  source: HumanoidAuthorityFrameSource;
  snapshot: Snapshot | null;
  commandId: string | null;
  commandCompleted: boolean;
}

interface PendingCommand<Snapshot, Result> {
  command: HumanoidAuthorityCommand<Snapshot, Result>;
  result: Promise<Result>;
  publication: Promise<void>;
  resolve: (value: Result) => void;
  reject: (error: unknown) => void;
  settled: boolean;
}

export class HumanoidAuthorityAdmissionError extends Error {
  readonly expected: HumanoidAuthorityIdentity;
  readonly actual: HumanoidAuthorityIdentity;

  constructor(
    expected: HumanoidAuthorityIdentity,
    actual: HumanoidAuthorityIdentity
  ) {
    super(
      "Humanoid authority admission state changed: "
      + `expected_revision=${expected.revision}, actual_revision=${actual.revision}, `
      + `expected_state=${expected.stateSha256}, actual_state=${actual.stateSha256}`
    );
    this.name = "HumanoidAuthorityAdmissionError";
    this.expected = { ...expected };
    this.actual = { ...actual };
  }
}

export class HumanoidAuthorityLoop<Snapshot> {
  readonly #mutex = new Mutex();
  readonly #identity: () => HumanoidAuthorityIdentity;
  readonly #stationaryStep: () => Promise<Snapshot>;
  readonly #queued: Array<PendingCommand<Snapshot, unknown>> = [];
  readonly #commandIds = new Set<string>();
  readonly #publisher: HumanoidAuthorityFramePublisher<Snapshot>;
  #active: PendingCommand<Snapshot, unknown> | undefined;
  #publicationFailure: unknown;
  #disposed = false;

  constructor(input: {
    identity: () => HumanoidAuthorityIdentity;
    stationaryStep: () => Promise<Snapshot>;
    maximumQueuedPublications?: number;
    onPublicationError?: (error: unknown) => void;
  }) {
    this.#identity = input.identity;
    this.#stationaryStep = input.stationaryStep;
    this.#publisher = new HumanoidAuthorityFramePublisher({
      ...(input.maximumQueuedPublications !== undefined
        ? { maximumQueuedFrames: input.maximumQueuedPublications }
        : {}),
      onError: (error) => {
        if (this.#publicationFailure === undefined) this.#publicationFailure = error;
        input.onPublicationError?.(error);
      }
    });
  }

  get busy(): boolean {
    return this.#active !== undefined || this.#queued.length > 0;
  }

  capture<T>(read: () => T): Promise<T> {
    return this.#mutex.runExclusive(() => {
      this.#assertOpen();
      return read();
    });
  }

  submit<Result>(
    command: HumanoidAuthorityCommand<Snapshot, Result>
  ): Promise<HumanoidAuthorityCommandHandle<Result>> {
    return this.#mutex.runExclusive(() => {
      this.#assertOpen();
      if (!command.id.trim()) throw new Error("Humanoid authority command id is required");
      if (this.#commandIds.has(command.id)) {
        throw new Error(`Duplicate humanoid authority command: ${command.id}`);
      }
      const pending = pendingCommand(command);
      pending.publication = this.#publisher.openCommand(command.id);
      this.#commandIds.add(command.id);
      if (!this.#active && this.#queued.length === 0) {
        try {
          this.#assertAdmission(command.admission);
          command.admit();
          this.#active = pending as PendingCommand<Snapshot, unknown>;
        } catch (error) {
          this.#commandIds.delete(command.id);
          pending.settled = true;
          this.#publisher.abortCommand(command.id);
          pending.reject(error);
          throw error;
        }
      } else {
        this.#queued.push(pending as PendingCommand<Snapshot, unknown>);
      }
      return {
        id: command.id,
        result: pending.result,
        publication: pending.publication,
        get settled() {
          return pending.settled;
        }
      };
    });
  }

  cancel(commandId: string, reason: unknown): Promise<boolean> {
    return this.#mutex.runExclusive(() => {
      if (this.#active?.command.id === commandId) {
        const active = this.#active;
        this.#active = undefined;
        this.#settleRejected(active, reason);
        return true;
      }
      const queuedIndex = this.#queued.findIndex((entry) => (
        entry.command.id === commandId
      ));
      if (queuedIndex < 0) return false;
      const [queued] = this.#queued.splice(queuedIndex, 1);
      if (!queued) return false;
      this.#settleRejected(queued, reason);
      return true;
    });
  }

  async tick(
    stationarySink?: (snapshot: Snapshot) => void | Promise<void>
  ): Promise<HumanoidAuthorityTick<Snapshot>> {
    const committed = await this.#mutex.runExclusive(async () => {
      this.#assertOpen();
      if (this.#publicationFailure !== undefined) {
        const failure = this.#publicationFailure;
        if (this.#active) {
          const active = this.#active;
          this.#active = undefined;
          this.#settleRejected(active, failure);
        }
        throw failure;
      }
      this.#promoteQueuedCommand();
      const active = this.#active;
      if (!active) {
        const snapshot = await this.#stationaryStep();
        if (stationarySink) {
          this.#publisher.enqueue({
            source: "stationary",
            commandId: null,
            sink: stationarySink,
            snapshot
          });
        }
        return {
          source: "stationary" as const,
          snapshot,
          commandId: null,
          done: false
        };
      }
      try {
        const step = await active.command.step();
        if (step.done && step.result === undefined) {
          throw new Error("Completed humanoid authority command has no result");
        }
        if (step.done) this.#active = undefined;
        const snapshot = step.snapshot ?? null;
        if (snapshot !== null && active.command.frameSink) {
          this.#publisher.enqueue({
            source: active.command.source,
            commandId: active.command.id,
            sink: active.command.frameSink,
            snapshot
          });
        }
        if (step.done) {
          this.#settleResolved(active, step.result);
          this.#publisher.closeCommand(active.command.id);
        }
        return {
          source: active.command.source,
          snapshot,
          commandId: active.command.id,
          done: step.done
        };
      } catch (error) {
        this.#active = undefined;
        this.#settleRejected(active, error);
        throw error;
      }
    });
    return {
      source: committed.source,
      snapshot: committed.snapshot,
      commandId: committed.commandId,
      commandCompleted: committed.done
    };
  }

  async dispose(reason: unknown = new Error("Humanoid authority loop disposed")): Promise<void> {
    await this.#mutex.runExclusive(() => {
      if (this.#disposed) return;
      this.#disposed = true;
      if (this.#active) this.#settleRejected(this.#active, reason);
      this.#active = undefined;
      for (const pending of this.#queued.splice(0)) {
        this.#settleRejected(pending, reason);
      }
    });
    await this.#publisher.dispose();
  }

  async flushPublications(): Promise<void> {
    await this.#publisher.flush();
    if (this.#publicationFailure !== undefined) throw this.#publicationFailure;
  }

  #promoteQueuedCommand(): void {
    while (!this.#active && this.#queued.length > 0) {
      const pending = this.#queued.shift()!;
      try {
        this.#assertAdmission(pending.command.admission);
        pending.command.admit();
        this.#active = pending;
      } catch (error) {
        this.#settleRejected(pending, error);
      }
    }
  }

  #assertAdmission(expected: HumanoidAuthorityIdentity): void {
    const actual = this.#identity();
    if (actual.revision !== expected.revision
      || actual.stateSha256 !== expected.stateSha256) {
      throw new HumanoidAuthorityAdmissionError(expected, actual);
    }
  }

  #settleResolved(
    pending: PendingCommand<Snapshot, unknown>,
    result: unknown
  ): void {
    if (pending.settled) return;
    pending.settled = true;
    this.#commandIds.delete(pending.command.id);
    pending.resolve(result);
  }

  #settleRejected(
    pending: PendingCommand<Snapshot, unknown>,
    error: unknown
  ): void {
    if (pending.settled) return;
    pending.settled = true;
    this.#commandIds.delete(pending.command.id);
    this.#publisher.abortCommand(pending.command.id);
    pending.reject(error);
  }

  #assertOpen(): void {
    if (this.#disposed) throw new Error("Humanoid authority loop is disposed");
  }
}

function pendingCommand<Snapshot, Result>(
  command: HumanoidAuthorityCommand<Snapshot, Result>
): PendingCommand<Snapshot, Result> {
  let resolve!: (value: Result) => void;
  let reject!: (error: unknown) => void;
  const result = new Promise<Result>((onResolve, onReject) => {
    resolve = onResolve;
    reject = onReject;
  });
  void result.catch(() => undefined);
  return {
    command,
    result,
    publication: Promise.resolve(),
    resolve,
    reject,
    settled: false
  };
}
