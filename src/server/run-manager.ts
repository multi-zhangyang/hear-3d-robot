import { basename } from "node:path";
import type { ProviderConfig, RuntimeCatalog } from "../config/load.js";
import type { Goal, JsonValue, RunCheckpoint } from "../domain/schema.js";
import type { RuntimeEvent, RuntimeEventSink } from "../harness/runtime-context.js";
import type { MutationFence } from "../persistence/mutation-fence.js";
import {
  createLifecycleEvent,
  reconcileLifecycleOutbox
} from "../persistence/lifecycle-outbox.js";
import {
  type JournalPage,
  listRunDirectories,
  resolveRunDirectory,
  RunStore
} from "../persistence/run-store.js";
import { resumeMission, startMission } from "../runtime/mission-runner.js";

export interface RunListItem {
  run_id: string;
  scenario_id: string | null;
  mission: string | null;
  status: RunCheckpoint["status"] | "local_artifact";
  created_at: string | null;
  updated_at: string | null;
  error: string | null;
}

type Subscriber = (event: RuntimeEvent) => void | Promise<void>;

const MAX_BUFFERED_LIVE_EVENTS_DURING_BACKFILL = 256;
const MAX_BUFFERED_LIVE_BYTES_DURING_BACKFILL = 1024 * 1024;

export class RunManager {
  readonly #runsDir: string;
  readonly #catalog: RuntimeCatalog;
  readonly #provider: ProviderConfig | undefined;
  readonly #providerError: string | undefined;
  readonly #mutationFence: MutationFence | undefined;
  readonly #subscribers = new Map<string, Set<Subscriber>>();
  readonly #controllers = new Map<string, AbortController>();
  readonly #operations = new Map<string, Promise<void>>();
  #launchController: AbortController | undefined;
  #launchOperation: Promise<void> | undefined;
  #launching = false;
  #accepting = true;

  constructor(input: {
    runsDir: string;
    catalog: RuntimeCatalog;
    provider?: ProviderConfig;
    providerError?: string;
    mutationFence?: MutationFence;
  }) {
    this.#runsDir = input.runsDir;
    this.#catalog = input.catalog;
    this.#provider = input.provider;
    this.#providerError = input.providerError;
    this.#mutationFence = input.mutationFence;
  }

  /** Converts process-owned nonterminal checkpoints left by a prior operator into resumable state. */
  async recoverOrphanedRuns(): Promise<number> {
    const directories = await listRunDirectories(this.#runsDir, this.#storeOptions());
    let recovered = 0;
    for (const directory of directories) {
      let store: RunStore;
      let checkpoint: RunCheckpoint;
      try {
        store = await RunStore.open(directory, this.#storeOptions());
        checkpoint = await store.readCheckpoint();
      } catch {
        // A malformed local artifact remains visible through list() but cannot
        // be rewritten safely without a valid definition and checkpoint.
        continue;
      }
      await reconcileLifecycleOutbox({
        store,
        checkpoint,
        persistCheckpoint: () => store.writeCheckpoint(checkpoint)
      });
      if (checkpoint.status !== "starting" && checkpoint.status !== "running") continue;
      const at = new Date().toISOString();
      const reason = "The previous operator process ended before this mission reached a terminal state.";
      checkpoint.status = "interrupted";
      checkpoint.error = reason;
      checkpoint.updated_at = at;
      checkpoint.pending_lifecycle_events.push(createLifecycleEvent({
        runId: store.definition.run_id,
        type: "run_interrupted",
        at,
        data: { reason, recovered_on_operator_start: true }
      }));
      await store.writeCheckpoint(checkpoint);
      await store.append("provider", {
        status: "operator_process_recovered",
        automatic_actuation: false,
        at
      });
      await reconcileLifecycleOutbox({
        store,
        checkpoint,
        persistCheckpoint: () => store.writeCheckpoint(checkpoint)
      });
      recovered += 1;
    }
    return recovered;
  }

  async start(input: {
    mission: string;
    scenarioId: string;
    goal: Goal;
    seed?: number;
  }): Promise<string> {
    const provider = this.#requireProvider();
    this.#assertCapacity();
    this.#launching = true;
    const controller = new AbortController();
    const created = deferred<string>();
    const sink = this.#sink(created.resolve);
    const operation = startMission({
      runsDir: this.#runsDir,
      mission: input.mission,
      scenarioId: input.scenarioId,
      goal: input.goal,
      catalog: this.#catalog,
      provider,
      ...(input.seed === undefined ? {} : { seed: input.seed }),
      eventSink: sink,
      signal: controller.signal,
      ...(this.#mutationFence ? { mutationFence: this.#mutationFence } : {})
    });
    return this.#trackLaunch(operation, created, controller);
  }

  async resume(runId: string): Promise<string> {
    const provider = this.#requireProvider();
    this.#assertCapacity();
    this.#launching = true;
    const controller = new AbortController();
    const created = deferred<string>();
    const sink = this.#sink(created.resolve);
    const operation = resumeMission({
      runDir: resolveRunDirectory(this.#runsDir, runId),
      catalog: this.#catalog,
      provider,
      eventSink: sink,
      signal: controller.signal,
      ...(this.#mutationFence ? { mutationFence: this.#mutationFence } : {})
    });
    return this.#trackLaunch(operation, created, controller);
  }

  isActive(runId: string): boolean {
    return this.#controllers.has(runId);
  }

  async subscribe(
    runId: string,
    after: string | undefined,
    subscriber: Subscriber,
    signal?: AbortSignal,
    onReady: () => void | Promise<void> = () => undefined
  ): Promise<() => void> {
    const buffered: RuntimeEvent[] = [];
    const bufferedIds = new Set<string>();
    const replayedBufferedIds = new Set<string>();
    let bufferedBytes = 0;
    let backfilling = true;
    let backfillFailure: Error | undefined;
    const throwIfUnavailable = (): void => {
      signal?.throwIfAborted();
      if (backfillFailure) throw backfillFailure;
    };
    const deliver = async (
      event: RuntimeEvent,
      source: "journal" | "buffered" | "live"
    ): Promise<void> => {
      throwIfUnavailable();
      if (source === "journal" && bufferedIds.has(event.event_id)) {
        replayedBufferedIds.add(event.event_id);
      } else if (
        source === "buffered"
        && (event.event_id === after || replayedBufferedIds.has(event.event_id))
      ) {
        return;
      }
      await subscriber(event);
      throwIfUnavailable();
    };
    const unsubscribe = this.#addSubscriber(runId, (event) => {
      if (!backfilling) return deliver(event, "live");
      if (bufferedIds.has(event.event_id)) return;
      const bytes = Buffer.byteLength(JSON.stringify(event));
      if (
        buffered.length >= MAX_BUFFERED_LIVE_EVENTS_DURING_BACKFILL
        || bytes > MAX_BUFFERED_LIVE_BYTES_DURING_BACKFILL - bufferedBytes
      ) {
        backfillFailure ??= new Error(
          `Event stream for run ${runId} fell behind during journal backfill`
        );
        throw backfillFailure;
      }
      buffered.push(event);
      bufferedIds.add(event.event_id);
      bufferedBytes += bytes;
    });

    try {
      throwIfUnavailable();
      if (after) {
        await this.#backfillRuntimeEvents(
          runId,
          after,
          (event) => deliver(event, "journal"),
          onReady,
          signal
        );
      } else {
        await RunStore.open(resolveRunDirectory(this.#runsDir, runId), this.#storeOptions());
        throwIfUnavailable();
        await onReady();
      }
      let bufferedIndex = 0;
      while (bufferedIndex < buffered.length) {
        await deliver(buffered[bufferedIndex]!, "buffered");
        bufferedIndex += 1;
      }
      throwIfUnavailable();
      backfilling = false;
      buffered.length = 0;
      bufferedBytes = 0;
      bufferedIds.clear();
      replayedBufferedIds.clear();
      return unsubscribe;
    } catch (error) {
      unsubscribe();
      throw error;
    }
  }

  #addSubscriber(runId: string, subscriber: Subscriber): () => void {
    const subscribers = this.#subscribers.get(runId) ?? new Set<Subscriber>();
    subscribers.add(subscriber);
    this.#subscribers.set(runId, subscribers);
    let subscribed = true;
    return () => {
      if (!subscribed) return;
      subscribed = false;
      subscribers.delete(subscriber);
      if (subscribers.size === 0 && this.#subscribers.get(runId) === subscribers) {
        this.#subscribers.delete(runId);
      }
    };
  }

  stopAll(reason = "Operator server stopped"): void {
    this.#launchController?.abort(new Error(reason));
    for (const controller of this.#controllers.values()) controller.abort(reason);
  }

  async drain(reason = "Operator server stopped"): Promise<void> {
    this.#accepting = false;
    this.stopAll(reason);
    const operations = new Set(this.#operations.values());
    if (this.#launchOperation) operations.add(this.#launchOperation);
    await Promise.allSettled([...operations]);
  }

  stop(runId: string, reason = "Mission stopped by operator"): void {
    const controller = this.#controllers.get(runId);
    if (!controller) throw new RunNotActiveError(`Run ${runId} is not active`);
    controller.abort(new Error(reason));
  }

  async list(): Promise<RunListItem[]> {
    const directories = await listRunDirectories(this.#runsDir, this.#storeOptions());
    const items = await Promise.all(directories.map((directory) => this.#summarize(directory)));
    return items.sort((left, right) => (right.created_at ?? "").localeCompare(left.created_at ?? ""));
  }

  async details(runId: string, limits: {
    actions: number;
    provider: number;
    framework: number;
  }): Promise<{
    definition: RunStore["definition"];
    checkpoint: RunCheckpoint;
    actions: JsonValue[];
    provider: JsonValue[];
    framework: JsonValue[];
    event_cursor: string | null;
  }> {
    const store = await RunStore.open(
      resolveRunDirectory(this.#runsDir, runId),
      this.#storeOptions()
    );
    // Read the journal high-water mark first, then the checkpoint. If a live
    // event lands between those reads, the checkpoint is at least as new and
    // the SSE subscription safely re-delivers the event after this cursor.
    // This avoids replaying an arbitrarily large events journal just to render
    // the current operator view.
    const [actionsPage, providerPage, frameworkPage, eventPage] = await Promise.all([
      store.readJournalTail("actions", limits.actions),
      store.readJournalTail("provider", limits.provider),
      store.readJournalTail("framework", limits.framework),
      store.readJournalTail("events", 1)
    ]);
    const checkpoint = await store.readCheckpoint();
    const cursorEntry = eventPage.entries.at(-1);
    const eventCursor = cursorEntry === undefined
      ? null
      : runtimeEvent(cursorEntry, runId).event_id;
    return {
      definition: store.definition,
      checkpoint,
      actions: actionsPage.entries,
      provider: providerPage.entries,
      framework: frameworkPage.entries,
      event_cursor: eventCursor
    };
  }

  async journal(
    runId: string,
    name: Parameters<RunStore["readJournal"]>[0],
    from: number,
    limit: number
  ): Promise<JournalPage> {
    const store = await RunStore.open(
      resolveRunDirectory(this.#runsDir, runId),
      this.#storeOptions()
    );
    return store.readJournalPage(name, from, limit);
  }

  /**
   * Replays only the missing suffix. The normal details -> SSE path passes the
   * current last event, which is answered from the journal offset index without
   * scanning history. Older cursors use a constant-memory stream.
   */
  async #backfillRuntimeEvents(
    runId: string,
    after: string,
    deliver: (event: RuntimeEvent) => void | Promise<void>,
    onReady: () => void | Promise<void>,
    signal?: AbortSignal
  ): Promise<void> {
    signal?.throwIfAborted();
    const store = await RunStore.open(
      resolveRunDirectory(this.#runsDir, runId),
      this.#storeOptions()
    );
    const tail = await store.readJournalTail("events", 1);
    const latest = tail.entries.at(-1);
    if (latest && runtimeEvent(latest, runId).event_id === after) {
      signal?.throwIfAborted();
      await onReady();
      return;
    }

    let found = false;
    await store.scanJournal("events", async (entry) => {
      signal?.throwIfAborted();
      const event = runtimeEvent(entry, runId);
      if (!found) {
        found = event.event_id === after;
        if (found) await onReady();
        return;
      }
      await deliver(event);
    });
    if (!found) {
      throw new EventCursorError(`Unknown event cursor for run ${runId}`);
    }
  }

  #assertCapacity(): void {
    if (!this.#accepting) throw new RunConflictError("Operator is shutting down");
    if (this.#launching || this.#controllers.size > 0) {
      throw new RunConflictError("Another mission is already active");
    }
  }

  #storeOptions(): { mutationFence?: MutationFence } {
    return this.#mutationFence ? { mutationFence: this.#mutationFence } : {};
  }

  #requireProvider(): ProviderConfig {
    if (!this.#provider) {
      throw new ProviderUnavailableError(this.#providerError ?? "Provider is not configured");
    }
    return this.#provider;
  }

  #sink(onCreated: (runId: string) => void): RuntimeEventSink {
    return (event) => {
      onCreated(event.run_id);
      const subscribers = this.#subscribers.get(event.run_id);
      if (!subscribers) return;
      for (const subscriber of subscribers) {
        try {
          const delivery = subscriber(event);
          if (delivery) {
            void delivery.catch(() => {
              subscribers.delete(subscriber);
              if (
                subscribers.size === 0
                && this.#subscribers.get(event.run_id) === subscribers
              ) {
                this.#subscribers.delete(event.run_id);
              }
            });
          }
        } catch {
          subscribers.delete(subscriber);
        }
      }
      if (
        subscribers.size === 0
        && this.#subscribers.get(event.run_id) === subscribers
      ) {
        this.#subscribers.delete(event.run_id);
      }
    };
  }

  async #trackLaunch(
    operation: ReturnType<typeof startMission>,
    created: Deferred<string>,
    controller: AbortController
  ): Promise<string> {
    operation.catch(created.reject);
    const settled = operation.then(() => undefined, () => undefined);
    this.#launchController = controller;
    this.#launchOperation = settled;
    void settled.finally(() => {
      if (this.#launchOperation === settled) this.#launchOperation = undefined;
      if (this.#launchController === controller) this.#launchController = undefined;
    });
    const runId = await created.promise.finally(() => {
      this.#launching = false;
    });
    this.#controllers.set(runId, controller);
    this.#operations.set(runId, settled);
    void settled.finally(() => {
        this.#controllers.delete(runId);
        this.#operations.delete(runId);
      });
    return runId;
  }

  async #summarize(directory: string): Promise<RunListItem> {
    try {
      const store = await RunStore.open(directory, this.#storeOptions());
      const checkpoint = await store.readCheckpoint();
      return {
        run_id: store.definition.run_id,
        scenario_id: store.definition.scenario_id,
        mission: store.definition.mission,
        status: checkpoint.status,
        created_at: checkpoint.created_at,
        updated_at: checkpoint.updated_at,
        error: checkpoint.error
      };
    } catch {
      return {
        run_id: basename(directory),
        scenario_id: null,
        mission: null,
        status: "local_artifact",
        created_at: null,
        updated_at: null,
        error: null
      };
    }
  }
}

export class RunConflictError extends Error {}

class RunNotActiveError extends Error {
  readonly statusCode = 409;
}

class ProviderUnavailableError extends Error {
  readonly statusCode = 503;
}

class EventCursorError extends Error {
  readonly statusCode = 409;
}

interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason: unknown) => void;
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function runtimeEvent(value: JsonValue, runId: string): RuntimeEvent {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`Run ${runId} contains an invalid runtime event`);
  }
  const eventId = value.event_id;
  const eventRunId = value.run_id;
  const type = value.type;
  const at = value.at;
  if (
    typeof eventId !== "string" || eventId.length === 0 || /[\r\n\0]/.test(eventId)
    || eventRunId !== runId
    || typeof type !== "string" || type.length === 0 || /[\r\n\0]/.test(type)
    || typeof at !== "string"
    || !("data" in value)
  ) {
    throw new Error(`Run ${runId} contains an invalid runtime event`);
  }
  return value as unknown as RuntimeEvent;
}
