import { basename } from "node:path";
import type { ProviderConfig, RuntimeCatalog } from "../config/load.js";
import type { Goal, JsonValue } from "../domain/schema.js";
import type { HumanoidRunMode } from "../domain/run-mode.js";
import type { AnyRunCheckpoint } from "../domain/run-checkpoint.js";
import type { RuntimeEvent, RuntimeEventSink } from "../runtime/events.js";
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
import {
  parseRuntimeEventCursor,
  runtimeEventCursor,
  runtimeEventCursorMatches
} from "../persistence/event-cursor.js";
import {
  resumeHumanoidMission,
  startHumanoidMission
} from "../runtime/humanoid-mission-runner.js";
import {
  RunPauseRequestedError,
  RunRestartRequestedError
} from "../runtime/run-pause.js";
import type {
  HumanoidControllerSource
} from "../world/humanoid/controller-module.js";
import {
  buildFoxgloveMcap,
  type FoxgloveMcapArtifact
} from "./foxglove-mcap.js";

export interface RunListItem {
  run_id: string;
  scenario_id: string | null;
  mission: string | null;
  status: AnyRunCheckpoint["status"] | "local_artifact";
  created_at: string | null;
  updated_at: string | null;
  error: string | null;
}

type Subscriber = (event: RuntimeEvent) => void | Promise<void>;

const MAX_BUFFERED_LIVE_EVENTS_DURING_BACKFILL = 256;
const MAX_BUFFERED_LIVE_BYTES_DURING_BACKFILL = 1024 * 1024;
const EVENT_REPLAY_PAGE_SIZE = 64;
const EVENT_REPLAY_PAGE_MAX_BYTES = 512 * 1024;
const CONTINUOUS_RECOVERY_INITIAL_DELAY_MS = 1_000;
const CONTINUOUS_RECOVERY_MAX_DELAY_MS = 60_000;
const CONTINUOUS_RECOVERY_STABILITY_MS = 5 * 60_000;

export class RunManager {
  readonly #runsDir: string;
  readonly #catalog: RuntimeCatalog;
  readonly #provider: ProviderConfig | undefined;
  readonly #providerError: string | undefined;
  readonly #mutationFence: MutationFence | undefined;
  readonly #controllerSource: HumanoidControllerSource | undefined;
  readonly #densePolicyRolloutDir: string | undefined;
  readonly #subscribers = new Map<string, Set<Subscriber>>();
  readonly #controllers = new Map<string, AbortController>();
  readonly #operations = new Map<string, Promise<void>>();
  readonly #continuousRecoveryAttempts = new Map<string, number>();
  readonly #continuousRecoveryTimers = new Map<
    string,
    ReturnType<typeof setTimeout>
  >();
  readonly #continuousRecoveryScheduling = new Map<string, symbol>();
  readonly #pendingContinuousRecoveries = new Map<string, number>();
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
    controllerSource?: HumanoidControllerSource;
    densePolicyRolloutDir?: string;
  }) {
    this.#runsDir = input.runsDir;
    this.#catalog = input.catalog;
    this.#provider = input.provider;
    this.#providerError = input.providerError;
    this.#mutationFence = input.mutationFence;
    this.#controllerSource = input.controllerSource;
    this.#densePolicyRolloutDir = input.densePolicyRolloutDir;
  }

  /**
   * Reclaims process-owned checkpoints after an unclean operator exit. The
   * newest continuous run is resumed immediately under the new process lease;
   * finite missions remain explicitly resumable by an operator.
   */
  async recoverOrphanedRuns(): Promise<number> {
    const directories = await listRunDirectories(this.#runsDir, this.#storeOptions());
    const continuousCandidates: Array<{
      runId: string;
      updatedAt: string;
      store: RunStore;
    }> = [];
    let recovered = 0;
    for (const directory of directories) {
      let store: RunStore;
      let checkpoint: AnyRunCheckpoint;
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
      if (checkpoint.status === "interrupted") {
        if (store.definition.run_mode === "continuous") {
          continuousCandidates.push({
            runId: store.definition.run_id,
            updatedAt: checkpoint.updated_at ?? checkpoint.created_at,
            store
          });
        }
        continue;
      }
      if (checkpoint.status !== "starting" && checkpoint.status !== "running") continue;
      // Preserve the last activity ordering before recovery rewrites every
      // orphan with a new interruption timestamp.  Sorting on the rewritten
      // value can resume whichever directory happened to be enumerated first,
      // rather than the continuous robot that was actually active most
      // recently.
      const lastActivityAt = checkpoint.updated_at ?? checkpoint.created_at;
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
      if (store.definition.run_mode === "continuous") {
        continuousCandidates.push({
          runId: store.definition.run_id,
          updatedAt: lastActivityAt,
          store
        });
      }
    }
    const autonomousResume = continuousCandidates.sort(
      (left, right) => right.updatedAt.localeCompare(left.updatedAt)
    )[0];
    if (autonomousResume && this.#provider) {
      const at = new Date().toISOString();
      await autonomousResume.store.append("provider", {
        status: "operator_process_autonomous_resume_requested",
        source: "continuous_run_lease_recovery",
        automatic_actuation: true,
        at
      });
      try {
        await this.resume(autonomousResume.runId);
      } catch (error) {
        await autonomousResume.store.append("provider", {
          status: "operator_process_autonomous_resume_failed",
          source: "continuous_run_lease_recovery",
          error: error instanceof Error ? error.message : String(error),
          automatic_actuation: false,
          at: new Date().toISOString()
        });
      }
    }
    return recovered;
  }

  async start(input: {
    mission: string;
    scenarioId: string;
    goal: Goal;
    runMode?: HumanoidRunMode;
    seed?: number;
  }): Promise<string> {
    const provider = this.#requireProvider();
    this.#assertCapacity();
    this.#launching = true;
    const controller = new AbortController();
    const created = deferred<string>();
    const sink = this.#sink(created.resolve);
    const operation = startHumanoidMission({
      runsDir: this.#runsDir,
      mission: input.mission,
      scenarioId: input.scenarioId,
      goal: input.goal,
      runMode: input.runMode ?? "continuous",
      catalog: this.#catalog,
      provider,
      ...(this.#densePolicyRolloutDir
        ? { densePolicyRolloutDir: this.#densePolicyRolloutDir }
        : {}),
      ...(input.seed === undefined ? {} : { seed: input.seed }),
      eventSink: sink,
      signal: controller.signal,
      ...(this.#controllerSource
        ? { controllerSource: this.#controllerSource }
        : {}),
      ...(this.#mutationFence ? { mutationFence: this.#mutationFence } : {})
    });
    return this.#trackLaunch(operation, created, controller);
  }

  async resume(
    runId: string,
    options: { freshAgentEpoch?: boolean } = {}
  ): Promise<string> {
    const provider = this.#requireProvider();
    this.#assertCapacity();
    const runDir = resolveRunDirectory(this.#runsDir, runId);
    await RunStore.open(runDir, this.#storeOptions());
    this.#launching = true;
    const controller = new AbortController();
    const created = deferred<string>();
    const sink = this.#sink(created.resolve);
    const operation = resumeHumanoidMission({
      runDir,
      catalog: this.#catalog,
      provider,
      ...(options.freshAgentEpoch ? { freshAgentEpoch: true } : {}),
      ...(this.#densePolicyRolloutDir
        ? { densePolicyRolloutDir: this.#densePolicyRolloutDir }
        : {}),
      eventSink: sink,
      signal: controller.signal,
      ...(this.#controllerSource
        ? { controllerSource: this.#controllerSource }
        : {}),
      ...(this.#mutationFence ? { mutationFence: this.#mutationFence } : {})
    });
    try {
      return await this.#trackLaunch(operation, created, controller);
    } catch (error) {
      void this.#scheduleContinuousRecovery(runId, 0);
      throw error;
    }
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
    const bufferedKeys = new Set<string>();
    const replayedBufferedKeys = new Set<string>();
    let bufferedBytes = 0;
    let backfilling = true;
    let backfillFailure: Error | undefined;
    const throwIfUnavailable = (): void => {
      signal?.throwIfAborted();
      if (backfillFailure) throw backfillFailure;
    };
    const declareReady = async (): Promise<void> => {
      throwIfUnavailable();
      await onReady();
      throwIfUnavailable();
    };
    const deliver = async (
      event: RuntimeEvent,
      source: "journal" | "buffered" | "live"
    ): Promise<void> => {
      throwIfUnavailable();
      const key = runtimeEventDedupeKey(event);
      if (source === "journal") {
        if (bufferedKeys.has(key)) replayedBufferedKeys.add(key);
        // Older in-process emitters did not carry the durable row cursor. Keep
        // their exact replay deduplication without collapsing current events
        // that intentionally reuse an event_id but have distinct cursors.
        for (const bufferedEvent of buffered) {
          if (bufferedEvent.cursor === undefined && bufferedEvent.event_id === event.event_id) {
            replayedBufferedKeys.add(runtimeEventDedupeKey(bufferedEvent));
          }
        }
      } else if (
        source === "buffered"
        && (event.cursor === after
          || (event.cursor === undefined && event.event_id === after)
          || replayedBufferedKeys.has(key))
      ) {
        return;
      }
      await subscriber(event);
      throwIfUnavailable();
    };
    const unsubscribe = this.#addSubscriber(runId, (event) => {
      if (!backfilling) return deliver(event, "live");
      const key = runtimeEventDedupeKey(event);
      if (bufferedKeys.has(key)) return;
      const bytes = Buffer.byteLength(JSON.stringify(event));
      if (
        buffered.length >= MAX_BUFFERED_LIVE_EVENTS_DURING_BACKFILL
        || (buffered.length > 0
          && bytes > MAX_BUFFERED_LIVE_BYTES_DURING_BACKFILL - bufferedBytes)
      ) {
        backfillFailure ??= new Error(
          `Event stream for run ${runId} fell behind during journal backfill`
        );
        throw backfillFailure;
      }
      buffered.push(event);
      bufferedKeys.add(key);
      bufferedBytes += bytes;
    });

    try {
      throwIfUnavailable();
      if (after) {
        await this.#backfillRuntimeEvents(
          runId,
          after,
          (event) => deliver(event, "journal"),
          declareReady,
          signal
        );
      } else {
        await RunStore.open(resolveRunDirectory(this.#runsDir, runId), this.#storeOptions());
        await declareReady();
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
      bufferedKeys.clear();
      replayedBufferedKeys.clear();
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
    this.#cancelContinuousRecoveries();
    this.#abortAll(() => new RunPauseRequestedError(reason));
  }

  async drain(reason = "Operator server stopped"): Promise<void> {
    await this.#drain(() => new RunPauseRequestedError(reason));
  }

  async drainForRestart(reason = "Operator server stopped"): Promise<void> {
    await this.#drain(() => new RunRestartRequestedError(reason));
  }

  async #drain(reason: () => Error): Promise<void> {
    this.#accepting = false;
    this.#cancelContinuousRecoveries();
    this.#abortAll(reason);
    const operations = new Set(this.#operations.values());
    if (this.#launchOperation) operations.add(this.#launchOperation);
    await Promise.allSettled([...operations]);
  }

  #abortAll(reason: () => Error): void {
    this.#launchController?.abort(reason());
    for (const controller of this.#controllers.values()) {
      controller.abort(reason());
    }
  }

  stop(runId: string, reason = "Mission stopped by operator"): void {
    const controller = this.#controllers.get(runId);
    if (!controller) throw new RunNotActiveError(`Run ${runId} is not active`);
    controller.abort(new RunPauseRequestedError(reason));
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
    checkpoint: AnyRunCheckpoint;
    scenario_chunks: Awaited<ReturnType<RunStore["readScenarioChunkDeltaState"]>>;
    actions: JsonValue[];
    provider: JsonValue[];
    framework: JsonValue[];
    event_cursor: string | null;
  }> {
    const store = await RunStore.open(
      resolveRunDirectory(this.#runsDir, runId),
      this.#storeOptions()
    );
    // One fenced cut is essential here. Independent parallel tail reads can see
    // an event after its matching action/provider/framework tail was sampled;
    // publishing that event as the cursor would then make SSE skip the missing
    // domain record forever. RunStore samples every durable surface while all
    // writers are excluded, and current telemetry carries a stable identity for
    // the inverse (domain-ahead-of-event) cut so the browser can deduplicate it.
    const snapshot = await store.readDetailsSnapshot(limits);
    const cursorEntry = snapshot.events.entries.at(-1);
    const eventCursor = cursorEntry === undefined
      ? null
      : runtimeEventAt(cursorEntry, runId, snapshot.events.total - 1).cursor!;
    return {
      definition: store.definition,
      checkpoint: snapshot.checkpoint,
      scenario_chunks: snapshot.scenarioChunks,
      actions: snapshot.actions.entries,
      provider: snapshot.provider.entries,
      framework: snapshot.framework.entries,
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

  async foxgloveMcap(runId: string): Promise<FoxgloveMcapArtifact> {
    const store = await RunStore.open(
      resolveRunDirectory(this.#runsDir, runId),
      this.#storeOptions()
    );
    const [checkpoint, events] = await Promise.all([
      store.readCheckpoint(),
      store.readJournal("events")
    ]);
    return buildFoxgloveMcap({
      definition: store.definition,
      checkpoint,
      events
    });
  }

  /** Replays the suffix after an exact durable journal row. */
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
    const parsed = parseRuntimeEventCursor(after);
    if (parsed.kind === "invalid") {
      throw new EventCursorError(`Malformed event cursor for run ${runId}`);
    }

    let cursorIndex: number;
    let replayEnd: number;
    if (parsed.kind === "versioned") {
      const page = await store.readJournalPage(
        "events",
        parsed.index,
        1,
        EVENT_REPLAY_PAGE_MAX_BYTES
      );
      const entry = page.entries[0];
      if (entry === undefined) {
        throw new EventCursorError(`Unknown event cursor for run ${runId}`);
      }
      const event = runtimeEventAt(entry, runId, parsed.index);
      if (!runtimeEventCursorMatches(parsed, runId, event.event_id)) {
        throw new EventCursorError(`Unknown event cursor for run ${runId}`);
      }
      cursorIndex = parsed.index;
      replayEnd = page.total;
    } else {
      let match: number | undefined;
      let matches = 0;
      await store.scanJournal("events", (entry, index) => {
        signal?.throwIfAborted();
        if (runtimeEvent(entry, runId).event_id !== after) return;
        matches += 1;
        match ??= index;
      });
      if (matches === 0 || match === undefined) {
        throw new EventCursorError(`Unknown event cursor for run ${runId}`);
      }
      if (matches > 1) {
        throw new EventCursorError(`Ambiguous legacy event cursor for run ${runId}`);
      }
      const page = await store.readJournalPage(
        "events",
        match,
        1,
        EVENT_REPLAY_PAGE_MAX_BYTES
      );
      const entry = page.entries[0];
      if (entry === undefined || runtimeEventAt(entry, runId, match).event_id !== after) {
        throw new EventCursorError(`Unknown event cursor for run ${runId}`);
      }
      cursorIndex = match;
      replayEnd = page.total;
    }

    signal?.throwIfAborted();
    await onReady();
    for (let from = cursorIndex + 1; from < replayEnd;) {
      signal?.throwIfAborted();
      const limit = Math.min(EVENT_REPLAY_PAGE_SIZE, replayEnd - from);
      const page = await store.readJournalPage(
        "events",
        from,
        limit,
        EVENT_REPLAY_PAGE_MAX_BYTES
      );
      if (page.entries.length === 0 || page.total < replayEnd) {
        throw new Error(`Event journal for run ${runId} changed during replay`);
      }
      for (const [offset, entry] of page.entries.entries()) {
        signal?.throwIfAborted();
        await deliver(runtimeEventAt(entry, runId, from + offset));
      }
      from += page.entries.length;
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
    operation: ReturnType<typeof startHumanoidMission>,
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
    let runId: string;
    try {
      runId = await created.promise;
    } catch (error) {
      this.#launching = false;
      this.#flushPendingContinuousRecovery();
      throw error;
    }
    this.#controllers.set(runId, controller);
    this.#operations.set(runId, settled);
    this.#launching = false;
    const activeSince = Date.now();
    void settled.finally(() => {
      if (this.#controllers.get(runId) === controller) {
        this.#controllers.delete(runId);
      }
      if (this.#operations.get(runId) === settled) {
        this.#operations.delete(runId);
      }
      void this.#scheduleContinuousRecovery(
        runId,
        Math.max(0, Date.now() - activeSince)
      );
      this.#flushPendingContinuousRecovery();
    });
    return runId;
  }

  async #scheduleContinuousRecovery(
    runId: string,
    activeDurationMs: number
  ): Promise<void> {
    if (!this.#accepting
      || this.#continuousRecoveryTimers.has(runId)
      || this.#continuousRecoveryScheduling.has(runId)) return;
    const schedulingToken = Symbol(runId);
    this.#continuousRecoveryScheduling.set(runId, schedulingToken);
    let store: RunStore;
    let checkpoint: AnyRunCheckpoint;
    try {
      store = await RunStore.open(
        resolveRunDirectory(this.#runsDir, runId),
        this.#storeOptions()
      );
      checkpoint = await store.readCheckpoint();
    } catch {
      if (this.#continuousRecoveryScheduling.get(runId) !== schedulingToken
        || !this.#accepting) return;
      this.#continuousRecoveryScheduling.delete(runId);
      const timer = setTimeout(() => {
        this.#continuousRecoveryTimers.delete(runId);
        void this.#scheduleContinuousRecovery(runId, activeDurationMs);
      }, CONTINUOUS_RECOVERY_INITIAL_DELAY_MS);
      this.#continuousRecoveryTimers.set(runId, timer);
      return;
    }
    if (this.#continuousRecoveryScheduling.get(runId) !== schedulingToken
      || !this.#accepting) return;
    this.#continuousRecoveryScheduling.delete(runId);
    if (store.definition.run_mode !== "continuous"
      || checkpoint.status !== "interrupted") {
      this.#continuousRecoveryAttempts.delete(runId);
      this.#pendingContinuousRecoveries.delete(runId);
      this.#flushPendingContinuousRecovery();
      return;
    }
    if (this.#controllers.size > 0 || this.#launching) {
      const pendingDuration = this.#pendingContinuousRecoveries.get(runId) ?? 0;
      this.#pendingContinuousRecoveries.set(
        runId,
        Math.max(pendingDuration, activeDurationMs)
      );
      return;
    }
    this.#pendingContinuousRecoveries.delete(runId);

    const previousAttempts = activeDurationMs >= CONTINUOUS_RECOVERY_STABILITY_MS
      ? 0
      : this.#continuousRecoveryAttempts.get(runId) ?? 0;
    const delayMs = Math.min(
      CONTINUOUS_RECOVERY_INITIAL_DELAY_MS * (2 ** Math.min(previousAttempts, 6)),
      CONTINUOUS_RECOVERY_MAX_DELAY_MS
    );
    this.#continuousRecoveryAttempts.set(runId, previousAttempts + 1);
    const timer = setTimeout(() => {
      this.#continuousRecoveryTimers.delete(runId);
      if (!this.#accepting) return;
      if (this.#controllers.size > 0 || this.#launching) {
        this.#pendingContinuousRecoveries.set(runId, activeDurationMs);
        return;
      }
      void this.resume(runId).catch(() => undefined);
    }, delayMs);
    this.#continuousRecoveryTimers.set(runId, timer);
  }

  #flushPendingContinuousRecovery(): void {
    if (!this.#accepting
      || this.#controllers.size > 0
      || this.#launching
      || this.#continuousRecoveryTimers.size > 0
      || this.#continuousRecoveryScheduling.size > 0) return;
    const next = this.#pendingContinuousRecoveries.entries().next().value as
      | [string, number]
      | undefined;
    if (!next) return;
    const [runId, activeDurationMs] = next;
    this.#pendingContinuousRecoveries.delete(runId);
    void this.#scheduleContinuousRecovery(runId, activeDurationMs);
  }

  #cancelContinuousRecoveries(): void {
    for (const timer of this.#continuousRecoveryTimers.values()) {
      clearTimeout(timer);
    }
    this.#continuousRecoveryTimers.clear();
    this.#continuousRecoveryAttempts.clear();
    this.#continuousRecoveryScheduling.clear();
    this.#pendingContinuousRecoveries.clear();
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
    || ("cursor" in value && (typeof value.cursor !== "string" || value.cursor.length === 0))
    || !("data" in value)
  ) {
    throw new Error(`Run ${runId} contains an invalid runtime event`);
  }
  return value as unknown as RuntimeEvent;
}

function runtimeEventAt(
  value: JsonValue,
  runId: string,
  index: number
): RuntimeEvent & { cursor: string } {
  const event = runtimeEvent(value, runId);
  if (event.durable === false) {
    throw new Error(`Run ${runId} contains a non-durable event in its journal`);
  }
  const cursor = runtimeEventCursor(runId, event.event_id, index);
  if (event.cursor !== undefined && event.cursor !== cursor) {
    throw new Error(`Run ${runId} contains an invalid runtime event cursor`);
  }
  return event.cursor === cursor ? event as RuntimeEvent & { cursor: string } : { ...event, cursor };
}

function runtimeEventDedupeKey(event: RuntimeEvent): string {
  return event.cursor ?? `event:${event.event_id}`;
}
