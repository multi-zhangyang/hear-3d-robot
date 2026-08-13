import { randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import {
  access,
  cp,
  mkdir,
  readFile,
  readdir,
  rename,
  rm,
  unlink
} from "node:fs/promises";
import { basename, dirname, resolve } from "node:path";
import { createInterface } from "node:readline";
import {
  GoalSchema,
  ScenarioSchema,
  type Goal,
  type JsonValue,
  type Scenario
} from "../domain/schema.js";
import {
  applyScenarioChunkDeltaMutations,
  type ScenarioChunkDeltaMutation
} from "../domain/scenario-chunk-delta.js";
import {
  createScenarioChunkDeltaState,
  restoreScenarioChunkDeltaState,
  type ScenarioChunkDeltaState
} from "../domain/scenario-chunk-delta-schema.js";
import {
  AnyRunCheckpointSchema,
  type AnyRunCheckpoint
} from "../domain/run-checkpoint.js";
import {
  HumanoidRunCheckpointSchema,
  type HumanoidRunCheckpoint
} from "../domain/humanoid-run.js";
import {
  HumanoidRunModeSchema,
  type HumanoidRunMode
} from "../domain/run-mode.js";
import {
  AgentManifestSchema,
  HistoricalAgentAuthorityManifestSchema,
  type HistoricalAgentAuthorityManifest,
  type AgentManifest
} from "../domain/agent-manifest.js";
import { NEURAL_HIERARCHY_CONTRACT_VERSION } from "../domain/neural-hierarchy.js";
import { z } from "zod";
import {
  appendIndexedRecordsBuilt,
  ensureJournalIndex,
  readIndexedTail,
  readIndexedWindow
} from "./journal-index.js";
import { runtimeEventCursor } from "./event-cursor.js";
import { writeTextAtomically } from "./atomic-file.js";
import {
  runFencedMutation,
  type MutationFence
} from "./mutation-fence.js";
import {
  assertFreshNeuralHierarchyEpochSafe,
  humanoidRunCheckpointNeedsPhysicalMigration,
  normalizeHumanoidRunCheckpoint
} from "./humanoid-checkpoint-migration.js";

const RunDefinitionSchema = z.object({
  version: z.literal(1),
  run_id: z.string().min(1),
  mission: z.string().min(1),
  scenario_id: z.string().min(1),
  scenario: ScenarioSchema,
  goal: GoalSchema,
  runtime: z.literal("humanoid_g1"),
  run_mode: HumanoidRunModeSchema.default("mission"),
  controller_source_sha256: z.string().regex(/^[a-f0-9]{64}$/).optional(),
  created_at: z.string().datetime()
}).strict();
const AGENT_ID_PATTERN = /^[a-zA-Z0-9_-]+$/;
const AgentSessionStateIdentitySchema = z.object({
  item_count: z.number().int().nonnegative(),
  items_sha256: z.string().regex(/^[a-f0-9]{64}$/)
}).strict();
const AgentSessionStateBaselineSchema = z.record(
  z.string().regex(AGENT_ID_PATTERN),
  AgentSessionStateIdentitySchema
);
const AgentStateEnvelopeV1Schema = z.object({
  version: z.literal(1),
  checkpoint_fingerprint: z.string().regex(/^[a-f0-9]{64}$/),
  state: z.string().min(1)
}).strict();
const AgentStateEnvelopeV2Schema = z.object({
  version: z.literal(2),
  checkpoint_fingerprint: z.string().regex(/^[a-f0-9]{64}$/),
  session_baseline: AgentSessionStateBaselineSchema,
  state: z.string().min(1)
}).strict();
const AgentStateEnvelopeSchema = z.union([
  AgentStateEnvelopeV2Schema,
  AgentStateEnvelopeV1Schema
]);

export type RunDefinition = z.infer<typeof RunDefinitionSchema>;

export type JournalName =
  | "events"
  | "provider"
  | "framework"
  | "actions"
  | "hierarchy"
  | "checker"
  | "episodes"
  | "experiences"
  | "context"
  | "goal_evidence"
  | "goal_history"
  | "model_calls"
  | "compaction_model_calls"
  | "action_identities";

export interface JournalPage {
  entries: JsonValue[];
  next: number | null;
  total: number;
}

export interface RunDetailsSnapshot {
  checkpoint: AnyRunCheckpoint;
  scenarioChunks: ScenarioChunkDeltaState;
  actions: JournalPage;
  provider: JournalPage;
  framework: JournalPage;
  events: JournalPage;
}

export interface RunStoreOptions {
  mutationFence?: MutationFence;
  freshNeuralHierarchyEpoch?: boolean;
}

export interface AgentStateRecord {
  state: string;
  checkpointFingerprint?: string;
  sessionBaseline?: AgentSessionStateBaseline;
}

export type AgentSessionStateBaseline = z.infer<
  typeof AgentSessionStateBaselineSchema
>;

export interface DurableRuntimeEventRecord {
  event_id: string;
  run_id: string;
  type: string;
  at: string;
  data: JsonValue;
  durable?: boolean;
  cursor?: string;
}

export class RunStore {
  readonly runDir: string;
  readonly definition: RunDefinition;
  readonly #mutationFence: MutationFence | undefined;
  readonly #journalWrites = new Map<JournalName, Promise<void>>();
  #chunkDeltaWrites: Promise<void> = Promise.resolve();

  static async create(
    runsDir: string,
    input: {
      mission: string;
      scenarioId: string;
      scenario: Scenario;
      goal: Goal;
      runtime?: RunDefinition["runtime"];
      runMode?: HumanoidRunMode;
      controllerSourceSha256?: string;
    },
    options: RunStoreOptions = {}
  ): Promise<RunStore> {
    const runId = createRunId(input.scenarioId);
    const runDir = resolve(runsDir, runId);
    const scenario = ScenarioSchema.parse(input.scenario);
    const definition = RunDefinitionSchema.parse({
      version: 1,
      run_id: runId,
      mission: input.mission,
      scenario_id: input.scenarioId,
      scenario: structuredClone(scenario),
      goal: structuredClone(input.goal),
      runtime: input.runtime ?? "humanoid_g1",
      run_mode: input.runMode ?? "mission",
      ...(input.controllerSourceSha256
        ? { controller_source_sha256: input.controllerSourceSha256 }
        : {}),
      created_at: new Date().toISOString()
    });
    const chunkDeltas = createScenarioChunkDeltaState(scenario);
    await runFencedMutation(options.mutationFence, async () => {
      await mkdir(runDir, { recursive: false });
      await atomicJson(resolve(runDir, "run.json"), definition);
      await atomicJson(resolve(runDir, "chunk-deltas.json"), chunkDeltas);
    });
    return new RunStore(runDir, definition, options.mutationFence);
  }

  static async open(runDir: string, options: RunStoreOptions = {}): Promise<RunStore> {
    const resolved = resolve(runDir);
    const definition = RunDefinitionSchema.parse(
      JSON.parse(await readFile(resolve(resolved, "run.json"), "utf8"))
    );
    if (basename(resolved) !== definition.run_id) throw new Error("Run directory identity mismatch");
    const store = new RunStore(resolved, definition, options.mutationFence);
    if (options.freshNeuralHierarchyEpoch === true) {
      // Archive the old model epoch before changing the checkpoint. If archival
      // fails, the old hierarchy, Sessions, and checkpoint remain a coherent
      // resumable unit rather than a half-switched epoch.
      await store.#assertFreshNeuralHierarchyEpochSafeIfPresent();
      await store.archiveAgentEpochForFreshHierarchy();
    }
    await store.#normalizeCheckpointIfPresent(
      options.freshNeuralHierarchyEpoch === true
    );
    return store;
  }

  private constructor(
    runDir: string,
    definition: RunDefinition,
    mutationFence: MutationFence | undefined
  ) {
    this.runDir = runDir;
    this.definition = definition;
    this.#mutationFence = mutationFence;
  }

  get mutationFence(): MutationFence | undefined {
    return this.#mutationFence;
  }

  async append(name: JournalName, value: JsonValue): Promise<void> {
    await this.#append(name, [JSON.stringify(value)]);
  }

  async appendMany(name: JournalName, values: JsonValue[]): Promise<void> {
    if (values.length === 0) return;
    await this.#append(name, values.map((value) => JSON.stringify(value)));
  }

  /**
   * Persists durable events with cursors derived from their exact JSONL row.
   * The ordinal is assigned while the append lock is held, so concurrent
   * writers and index rebuilds cannot make the cursor point at another row.
   */
  async appendRuntimeEvents<T extends DurableRuntimeEventRecord>(
    events: readonly T[]
  ): Promise<Array<T & { cursor: string }>> {
    if (events.length === 0) return [];
    let persisted: Array<T & { cursor: string }> = [];
    await this.#appendBuilt("events", (startIndex) => {
      persisted = events.map((event, offset) => ({
        ...structuredClone(event),
        cursor: runtimeEventCursor(event.run_id, event.event_id, startIndex + offset)
      }));
      return persisted.map((event) => JSON.stringify(event));
    });
    return persisted;
  }

  async writeCheckpoint(checkpoint: AnyRunCheckpoint): Promise<void> {
    const parsed = AnyRunCheckpointSchema.parse(checkpoint);
    this.#assertCheckpointRuntime(parsed);
    await this.#runMutation(
      () => atomicJson(resolve(this.runDir, "checkpoint.json"), parsed)
    );
  }

  async readCheckpoint(): Promise<AnyRunCheckpoint> {
    return this.#runMutation(() => this.#readCheckpoint());
  }

  async readHumanoidCheckpoint(): Promise<HumanoidRunCheckpoint> {
    return HumanoidRunCheckpointSchema.parse(await this.readCheckpoint());
  }

  async readScenarioChunkDeltaState(): Promise<ScenarioChunkDeltaState> {
    await this.#chunkDeltaWrites;
    return this.#readScenarioChunkDeltaState();
  }

  /** Serializes one pure delta transition and publishes it with atomic replace. */
  async applyScenarioChunkDeltaMutation(
    mutation: ScenarioChunkDeltaMutation
  ): Promise<ScenarioChunkDeltaState> {
    return this.applyScenarioChunkDeltaMutations([mutation]);
  }

  /** Publishes one validated multi-entity world transition with one atomic replace. */
  async applyScenarioChunkDeltaMutations(
    mutations: readonly ScenarioChunkDeltaMutation[]
  ): Promise<ScenarioChunkDeltaState> {
    let result: ScenarioChunkDeltaState | undefined;
    const write = this.#chunkDeltaWrites.then(async () => {
      await this.#runMutation(async () => {
        const current = await this.#readScenarioChunkDeltaState();
        result = applyScenarioChunkDeltaMutations(
          this.definition.scenario,
          current,
          mutations
        );
        if (result.revision !== current.revision) {
          await atomicJson(resolve(this.runDir, "chunk-deltas.json"), result);
        }
      });
    });
    this.#chunkDeltaWrites = write.catch(() => undefined);
    await write;
    if (!result) throw new Error("Scenario chunk delta mutation did not produce a state");
    return result;
  }

  async readJournal(name: JournalName): Promise<JsonValue[]> {
    const entries: JsonValue[] = [];
    await this.#scanJournal(name, (entry) => {
      entries.push(entry);
    });
    return entries;
  }

  async readJournalPage(
    name: JournalName,
    from: number,
    limit: number,
    maximumBytes?: number
  ): Promise<JournalPage> {
    if (!Number.isSafeInteger(from) || from < 0) throw new Error("Journal offset must be nonnegative");
    if (!Number.isSafeInteger(limit) || limit < 1) throw new Error("Journal limit must be positive");
    if (maximumBytes !== undefined
      && (!Number.isSafeInteger(maximumBytes) || maximumBytes < 1)) {
      throw new Error("Journal byte limit must be positive");
    }

    await this.#journalWrites.get(name);
    const window = await this.#runMutation(() => readIndexedWindow(
      this.#journalPath(name),
      this.#journalIndexPath(name),
      from,
      limit,
      maximumBytes
    ));
    return {
      entries: window.lines.map(parseJournalLine),
      next: window.next,
      total: window.total
    };
  }

  async readJournalTail(name: JournalName, limit: number): Promise<JournalPage> {
    if (!Number.isSafeInteger(limit) || limit < 1) throw new Error("Journal limit must be positive");

    await this.#journalWrites.get(name);
    return this.#runMutation(() => this.#readJournalTail(name, limit));
  }

  /**
   * Captures the operator's checkpoint and durable journal tails under one
   * runs-directory mutation fence. Runtime writers always persist a domain
   * record before its matching durable event, so an event visible in this cut
   * can never point past an omitted action/provider/framework record.
   *
   * A domain record may be visible just before its event is appended. Current
   * records carry the same stable runtime_event_id as that later event, allowing
   * the browser to merge the details response and SSE suffix exactly once.
   */
  async readDetailsSnapshot(limits: {
    actions: number;
    provider: number;
    framework: number;
  }): Promise<RunDetailsSnapshot> {
    for (const limit of [limits.actions, limits.provider, limits.framework]) {
      if (!Number.isSafeInteger(limit) || limit < 1) {
        throw new Error("Journal limit must be positive");
      }
    }
    const journals = ["actions", "provider", "framework", "events"] as const;
    await Promise.all([
      ...journals.map((name) => this.#journalWrites.get(name)),
      this.#chunkDeltaWrites
    ]);
    return this.#runMutation(async () => {
      const [
        actions,
        provider,
        framework,
        events,
        checkpoint,
        scenarioChunks
      ] = await Promise.all([
        this.#readJournalTail("actions", limits.actions),
        this.#readJournalTail("provider", limits.provider),
        this.#readJournalTail("framework", limits.framework),
        this.#readJournalTail("events", 1),
        this.#readCheckpoint(),
        this.#readScenarioChunkDeltaState()
      ]);
      return { checkpoint, scenarioChunks, actions, provider, framework, events };
    });
  }

  async scanJournal(
    name: JournalName,
    visit: (entry: JsonValue, index: number) => void | Promise<void>
  ): Promise<void> {
    await this.#scanJournal(name, visit);
  }

  async writeAgentState(
    state: string,
    checkpointFingerprint?: string,
    sessionBaseline?: AgentSessionStateBaseline
  ): Promise<void> {
    if (sessionBaseline !== undefined && checkpointFingerprint === undefined) {
      throw new Error("Agent Session baseline requires a checkpoint fingerprint");
    }
    const persisted = checkpointFingerprint === undefined
      ? state
      : JSON.stringify(sessionBaseline === undefined
          ? AgentStateEnvelopeV1Schema.parse({
              version: 1,
              checkpoint_fingerprint: checkpointFingerprint,
              state
            })
          : AgentStateEnvelopeV2Schema.parse({
              version: 2,
              checkpoint_fingerprint: checkpointFingerprint,
              session_baseline: sessionBaseline,
              state
            }));
    await this.#runMutation(
      () => atomicText(resolve(this.runDir, "agent-state.json"), persisted)
    );
  }

  async readAgentState(): Promise<string | undefined> {
    return (await this.readAgentStateRecord())?.state;
  }

  async readAgentStateRecord(): Promise<AgentStateRecord | undefined> {
    try {
      const persisted = await readFile(resolve(this.runDir, "agent-state.json"), "utf8");
      try {
        const envelope = AgentStateEnvelopeSchema.safeParse(JSON.parse(persisted));
        if (envelope.success) {
          return {
            state: envelope.data.state,
            checkpointFingerprint: envelope.data.checkpoint_fingerprint,
            ...(envelope.data.version === 2
              ? { sessionBaseline: envelope.data.session_baseline }
              : {})
          };
        }
      } catch {
        // Raw SDK RunState files written before the envelope remain readable.
      }
      return { state: persisted };
    } catch (error) {
      if (isMissing(error)) return undefined;
      throw error;
    }
  }

  /**
   * Drop only the replaceable SDK conversation branch. The append-only context,
   * action and hierarchy journals plus the authoritative checkpoint remain.
   */
  async clearAgentState(): Promise<void> {
    await this.#runMutation(async () => {
      try {
        await unlink(resolve(this.runDir, "agent-state.json"));
      } catch (error) {
        if (!isMissing(error)) throw error;
      }
    });
  }

  async writeAgentManifest(manifest: AgentManifest): Promise<void> {
    const parsed = AgentManifestSchema.parse(manifest);
    await this.#runMutation(
      () => atomicJson(resolve(this.runDir, "agent-manifest.json"), parsed)
    );
  }

  async readAgentManifest(): Promise<AgentManifest> {
    try {
      return parsePersistedAgentManifest(
        await readFile(resolve(this.runDir, "agent-manifest.json"), "utf8"),
        "current"
      );
    } catch (error) {
      if (isMissing(error)) {
        throw new Error(
          "Agent manifest is missing; refusing to reuse unverified model or Session state"
        );
      }
      throw error;
    }
  }

  async readArchivedAgentManifests(): Promise<HistoricalAgentAuthorityManifest[]> {
    return this.#runMutation(async () => {
      const directory = resolve(this.runDir, "agent-epochs");
      let entries;
      try {
        entries = await readdir(directory, { withFileTypes: true });
      } catch (error) {
        if (isMissing(error)) return [];
        throw error;
      }
      const manifests: HistoricalAgentAuthorityManifest[] = [];
      for (const entry of entries
        .filter((candidate) => candidate.isDirectory())
        .sort((left, right) => left.name.localeCompare(right.name))) {
        const manifest = historicalAgentAuthorityManifest(JSON.parse(await readFile(
          resolve(directory, entry.name, "agent-manifest.json"),
          "utf8"
        )) as unknown, `archived epoch ${entry.name}`);
        if (manifest.epoch_id !== entry.name) {
          throw new Error(`Archived Agent manifest epoch mismatch: ${entry.name}`);
        }
        manifests.push(manifest);
      }
      return manifests;
    });
  }

  async archiveCurrentAgentEpoch(): Promise<string> {
    return this.#runMutation(async () => {
      const manifest = await this.readAgentManifest();
      const archiveRoot = resolve(this.runDir, "agent-epochs");
      const destination = resolve(archiveRoot, manifest.epoch_id);
      const staging = resolve(
        archiveRoot,
        `.${manifest.epoch_id}.${randomUUID()}.tmp`
      );
      await mkdir(archiveRoot, { recursive: true });
      let destinationExists = false;
      try {
        await access(destination);
        destinationExists = true;
      } catch {
        destinationExists = false;
      }
      if (destinationExists) {
        const archived = AgentManifestSchema.parse(JSON.parse(await readFile(
          resolve(destination, "agent-manifest.json"),
          "utf8"
        )));
        if (archived.identity_sha256 !== manifest.identity_sha256) {
          throw new Error(`Archived Agent epoch identity mismatch: ${manifest.epoch_id}`);
        }
      } else {
        await mkdir(staging, { recursive: false });
        try {
          for (const name of [
            "agent-manifest.json",
            "agent-state.json",
            "session.json",
            "sessions"
          ] as const) {
            const source = resolve(this.runDir, name);
            try {
              await access(source);
            } catch {
              continue;
            }
            await cp(source, resolve(staging, name), {
              recursive: name === "sessions",
              errorOnExist: true
            });
          }
          await rename(staging, destination);
        } catch (error) {
          await rm(staging, { recursive: true, force: true });
          throw error;
        }
      }
      for (const name of [
        "agent-manifest.json",
        "agent-state.json",
        "session.json",
        "sessions"
      ] as const) {
        await rm(resolve(this.runDir, name), { recursive: true, force: true });
      }
      return manifest.epoch_id;
    });
  }

  /**
   * Archive an incompatible pre-contract manifest without parsing it as the
   * current manifest schema. This path is reachable only through the explicit
   * fresh hierarchy epoch operation; normal resume remains strict.
   */
  async archiveIncompatibleAgentEpoch(): Promise<string> {
    return this.#runMutation(async () => {
      const manifestPath = resolve(this.runDir, "agent-manifest.json");
      const raw = JSON.parse(await readFile(manifestPath, "utf8")) as unknown;
      if (!isRecord(raw) || typeof raw.epoch_id !== "string"
        || !z.string().uuid().safeParse(raw.epoch_id).success
        || raw.runtime !== "humanoid_g1") {
        throw new Error("Incompatible Agent manifest has no safe archive identity");
      }
      const epochId = raw.epoch_id;
      const archiveRoot = resolve(this.runDir, "agent-epochs");
      const destination = resolve(archiveRoot, epochId);
      const staging = resolve(archiveRoot, `.${epochId}.${randomUUID()}.tmp`);
      await mkdir(archiveRoot, { recursive: true });
      try {
        await access(destination);
      } catch {
        await mkdir(staging, { recursive: false });
        try {
          for (const name of [
            "agent-manifest.json",
            "agent-state.json",
            "session.json",
            "sessions"
          ] as const) {
            const source = resolve(this.runDir, name);
            try {
              await access(source);
            } catch {
              continue;
            }
            await cp(source, resolve(staging, name), {
              recursive: name === "sessions",
              errorOnExist: true
            });
          }
          await rename(staging, destination);
        } catch (error) {
          await rm(staging, { recursive: true, force: true });
          throw error;
        }
      }
      for (const name of [
        "agent-manifest.json",
        "agent-state.json",
        "session.json",
        "sessions"
      ] as const) {
        await rm(resolve(this.runDir, name), { recursive: true, force: true });
      }
      return epochId;
    });
  }

  async archiveAgentEpochForFreshHierarchy(): Promise<string | undefined> {
    try {
      await access(resolve(this.runDir, "agent-manifest.json"));
    } catch (error) {
      if (isMissing(error)) return undefined;
      throw error;
    }
    try {
      return await this.archiveCurrentAgentEpoch();
    } catch (error) {
      if (!(error instanceof Error)
        || (!error.message.includes("neural contract")
          && !error.message.includes("pre-invocation neural manifest"))) {
        throw error;
      }
      return this.archiveIncompatibleAgentEpoch();
    }
  }

  sessionPath(): string {
    return resolve(this.runDir, "session.json");
  }

  workerSessionPath(agentId: string): string {
    if (!AGENT_ID_PATTERN.test(agentId)) throw new Error("Invalid hierarchy node identifier");
    return resolve(this.runDir, "sessions", `${agentId}.json`);
  }

  async clearWorkerSessions(): Promise<void> {
    await this.#runMutation(async () => {
      const directory = resolve(this.runDir, "sessions");
      let entries;
      try {
        entries = await readdir(directory, { withFileTypes: true });
      } catch (error) {
        if (isMissing(error)) return;
        throw error;
      }
      await Promise.all(entries
        .filter((entry) => entry.isFile() && entry.name.endsWith(".json"))
        .map((entry) => unlink(resolve(directory, entry.name))));
    });
  }

  async #append(name: JournalName, records: string[]): Promise<void> {
    await this.#appendBuilt(name, () => records);
  }

  async #appendBuilt(
    name: JournalName,
    build: (startIndex: number) => readonly string[]
  ): Promise<void> {
    const previous = this.#journalWrites.get(name) ?? Promise.resolve();
    const write = previous.then(async () => {
      await this.#runMutation(() => appendIndexedRecordsBuilt(
        this.#journalPath(name),
        this.#journalIndexPath(name),
        build
      ));
    });
    this.#journalWrites.set(name, write.catch(() => undefined));
    await write;
  }

  async #readCheckpoint(): Promise<AnyRunCheckpoint> {
    const normalized = await normalizeHumanoidRunCheckpoint(
      JSON.parse(await readFile(resolve(this.runDir, "checkpoint.json"), "utf8")),
      this.definition.scenario
    );
    const checkpoint = AnyRunCheckpointSchema.parse(normalized.checkpoint);
    this.#assertCheckpointRuntime(checkpoint);
    if (normalized.migrated) {
      await atomicJson(resolve(this.runDir, "checkpoint.json"), checkpoint);
    }
    return checkpoint;
  }

  async #normalizeCheckpointIfPresent(freshNeuralHierarchyEpoch = false): Promise<void> {
    let raw: unknown;
    try {
      raw = JSON.parse(
        await readFile(resolve(this.runDir, "checkpoint.json"), "utf8")
      );
    } catch (error) {
      if (isMissing(error)) return;
      throw error;
    }
    if (!freshNeuralHierarchyEpoch
      && !humanoidRunCheckpointNeedsPhysicalMigration(raw)) {
      // Opening a run is the compatibility boundary. Validate the neural
      // contract now so callers never receive a store that fails later with a
      // raw schema error on its first checkpoint read.
      await normalizeHumanoidRunCheckpoint(raw, this.definition.scenario);
      return;
    }
    await this.#runMutation(async () => {
      const normalized = await normalizeHumanoidRunCheckpoint(
        raw,
        this.definition.scenario,
        { freshNeuralHierarchyEpoch }
      );
      const checkpoint = AnyRunCheckpointSchema.parse(normalized.checkpoint);
      this.#assertCheckpointRuntime(checkpoint);
      if (normalized.migrated) {
        await atomicJson(resolve(this.runDir, "checkpoint.json"), checkpoint);
      }
    });
  }

  async #assertFreshNeuralHierarchyEpochSafeIfPresent(): Promise<void> {
    try {
      const raw: unknown = JSON.parse(
        await readFile(resolve(this.runDir, "checkpoint.json"), "utf8")
      );
      assertFreshNeuralHierarchyEpochSafe(raw);
    } catch (error) {
      if (isMissing(error)) return;
      throw error;
    }
  }

  async #readScenarioChunkDeltaState(): Promise<ScenarioChunkDeltaState> {
    try {
      return restoreScenarioChunkDeltaState(
        this.definition.scenario,
        JSON.parse(await readFile(resolve(this.runDir, "chunk-deltas.json"), "utf8"))
      );
    } catch (error) {
      if (isMissing(error)) return createScenarioChunkDeltaState(this.definition.scenario);
      throw error;
    }
  }

  #assertCheckpointRuntime(checkpoint: AnyRunCheckpoint): void {
    if (checkpoint.runtime !== this.definition.runtime) {
      throw new Error(
        `Run checkpoint runtime mismatch: expected ${this.definition.runtime}, `
        + `received ${checkpoint.runtime}`
      );
    }
  }

  async #readJournalTail(name: JournalName, limit: number): Promise<JournalPage> {
    const window = await readIndexedTail(
      this.#journalPath(name),
      this.#journalIndexPath(name),
      limit
    );
    return {
      entries: window.lines.map(parseJournalLine),
      next: null,
      total: window.total
    };
  }

  #journalPath(name: JournalName): string {
    return resolve(this.runDir, `${name}.jsonl`);
  }

  #journalIndexPath(name: JournalName): string {
    return resolve(this.runDir, `${name}.offsets`);
  }

  async #scanJournal(
    name: JournalName,
    visit: (entry: JsonValue, index: number) => void | Promise<void>
  ): Promise<void> {
    await this.#journalWrites.get(name);
    const path = resolve(this.runDir, `${name}.jsonl`);
    const state = await this.#runMutation(
      () => ensureJournalIndex(path, this.#journalIndexPath(name))
    );
    if (state.completeByteLength === 0) return;
    try {
      await access(path);
    } catch (error) {
      if (isMissing(error)) return;
      throw error;
    }
    const stream = createReadStream(path, {
      encoding: "utf8",
      end: state.completeByteLength - 1
    });
    const lines = createInterface({ input: stream, crlfDelay: Infinity });
    let index = 0;
    try {
      for await (const line of lines) {
        if (line.length === 0) continue;
        await visit(JSON.parse(line) as JsonValue, index);
        index += 1;
      }
    } catch (error) {
      if (isMissing(error)) return;
      throw error;
    } finally {
      lines.close();
      stream.destroy();
    }
  }

  async #runMutation<T>(operation: () => Promise<T>): Promise<T> {
    return runFencedMutation(this.#mutationFence, operation);
  }
}

function historicalAgentAuthorityManifest(
  raw: unknown,
  label: string
): HistoricalAgentAuthorityManifest {
  if (!isRecord(raw) || raw.runtime !== "humanoid_g1"
    || typeof raw.epoch_id !== "string"
    || typeof raw.identity_sha256 !== "string"
    || !isRecord(raw.agents)) {
    throw new Error(`The ${label} manifest has no historical authority identity`);
  }
  const agents = Object.values(raw.agents).filter(isRecord);
  const agentIds = agents.flatMap((agent) => (
    typeof agent.agent_id === "string" && agent.agent_id.trim().length > 0
      ? [agent.agent_id]
      : []
  ));
  if (agentIds.length !== agents.length || agentIds.length === 0) {
    throw new Error(`The ${label} manifest has incomplete Agent identities`);
  }
  const legacyAgents = raw.agents;
  const neural = raw.version !== 1;
  const goalManagerAgentId = neural
    ? historicalStructuralAgentId(raw, "executive", "goal_valuation", label)
    : historicalLegacyAgentId(legacyAgents, "goal_manager", label);
  const groundingManagerAgentId = neural
    ? historicalStructuralAgentId(
        raw,
        "perceptual_association",
        "perceptual_association",
        label
      )
    : historicalLegacyAgentId(legacyAgents, "coordinator", label);
  const executionManagerAgentId = neural
    ? historicalStructuralAgentId(raw, "sensorimotor", "sensorimotor_selection", label)
    : historicalLegacyAgentId(legacyAgents, "coordinator", label);
  return HistoricalAgentAuthorityManifestSchema.parse({
    epoch_id: raw.epoch_id,
    identity_sha256: raw.identity_sha256,
    agent_ids: agentIds,
    goal_manager_agent_id: goalManagerAgentId,
    grounding_manager_agent_id: groundingManagerAgentId,
    execution_manager_agent_id: executionManagerAgentId
  });
}

function historicalStructuralAgentId(
  manifest: Record<string, unknown>,
  layer: string,
  pathway: string,
  label: string
): string {
  const agents = isRecord(manifest.agents) ? Object.values(manifest.agents) : [];
  const candidates = agents.filter(isRecord).filter((agent) => (
    agent.execution_kind === "model_agent"
      && agent.layer === layer
      && agent.pathway === pathway
  ));
  const candidate = candidates[0];
  if (candidates.length !== 1 || !candidate
    || typeof candidate.agent_id !== "string") {
    throw new Error(`The ${label} manifest has ambiguous ${pathway} authority`);
  }
  return candidate.agent_id;
}

function historicalLegacyAgentId(
  agents: Record<string, unknown>,
  key: "goal_manager" | "coordinator",
  label: string
): string {
  const agent = agents[key];
  if (!isRecord(agent) || typeof agent.agent_id !== "string"
    || agent.agent_id.trim().length === 0) {
    throw new Error(`The ${label} manifest has no ${key} authority`);
  }
  return agent.agent_id;
}

function parsePersistedAgentManifest(
  source: string,
  label: string
): AgentManifest {
  const raw: unknown = JSON.parse(source);
  const parsed = AgentManifestSchema.safeParse(raw);
  if (parsed.success) return parsed.data;
  if (isRecord(raw)
    && raw.runtime === "humanoid_g1"
    && (raw.version === 2
      || (typeof raw.neural_contract_version === "number"
        && raw.neural_contract_version !== NEURAL_HIERARCHY_CONTRACT_VERSION))) {
    throw new Error(
      `The ${label} Agent manifest uses ${raw.version === 2
        ? "the pre-invocation neural manifest v2"
        : `neural contract v${raw.neural_contract_version}`}; `
      + `the current invocation-scoped hierarchy contract is v${NEURAL_HIERARCHY_CONTRACT_VERSION}. `
      + "Existing Sessions and "
      + "physical state remain unchanged; create an explicit fresh Agent/hierarchy epoch "
      + "instead of mixing contexts across contracts."
    );
  }
  if (isRecord(raw)
    && raw.runtime === "humanoid_g1"
    && typeof raw.harness_contract_version === "number"
    && raw.harness_contract_version < 22
    && isRecord(raw.agents)
    && raw.agents.motion_planner === undefined) {
    throw new Error(
      `The ${label} Agent manifest uses the legacy single-Motion topology `
      + `(Harness contract v${raw.harness_contract_version}); it cannot be attached `
      + "to the independent Motion Planner/Actor topology. The old run and its "
      + "Sessions remain unchanged; start a new run instead of mixing Agent contexts."
    );
  }
  throw parsed.error;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function parseJournalLine(line: string): JsonValue {
  return JSON.parse(line) as JsonValue;
}

export async function listRunDirectories(
  runsDir: string,
  options: RunStoreOptions = {}
): Promise<string[]> {
  await runFencedMutation(options.mutationFence, async () => {
    await mkdir(runsDir, { recursive: true });
  });
  const entries = await readdir(runsDir, { withFileTypes: true });
  return entries
    .filter((entry) => entry.isDirectory() && !entry.name.startsWith("."))
    .map((entry) => resolve(runsDir, entry.name));
}

export function resolveRunDirectory(runsDir: string, runId: string): string {
  if (!/^[a-zA-Z0-9_-]+$/.test(runId)) throw new Error("Invalid run identifier");
  const root = resolve(runsDir);
  const candidate = resolve(root, runId);
  if (dirname(candidate) !== root) throw new Error("Invalid run directory");
  return candidate;
}

async function atomicJson(path: string, value: unknown): Promise<void> {
  await atomicText(path, `${JSON.stringify(value, null, 2)}\n`);
}

async function atomicText(path: string, value: string): Promise<void> {
  await writeTextAtomically(path, value);
}

function createRunId(scenarioId: string): string {
  const timestamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
  const safeScenario = scenarioId.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 48);
  return `${timestamp}_${safeScenario}_${randomUUID().slice(0, 8)}`;
}

function isMissing(error: unknown): boolean {
  return error instanceof Error && "code" in error && error.code === "ENOENT";
}
