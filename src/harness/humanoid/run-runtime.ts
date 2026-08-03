import { randomUUID } from "node:crypto";
import type {
  ContextCompactionSummary,
  ContextMemoryState,
  Goal,
  JsonValue,
  TaskNode
} from "../../domain/schema.js";
import {
  HumanoidEmbodiedEpisodeSchema,
  PersistedHumanoidActionReceiptSchema,
  type HumanoidEmbodiedEpisode,
  type HumanoidRunCheckpoint
} from "../../domain/humanoid-run.js";
import type { RunStore } from "../../persistence/run-store.js";
import {
  createLifecycleEvent,
  reconcileLifecycleOutbox
} from "../../persistence/lifecycle-outbox.js";
import type { RuntimeEvent, RuntimeEventSink } from "../../runtime/events.js";
import {
  advanceHumanoidGoal,
  assertHumanoidGoalProgressIntegrity,
  inspectHumanoidGoal
} from "../../runtime/humanoid-checker.js";
import type { HumanoidWorld, HumanoidWorldSnapshot } from "../../world/humanoid/world.js";
import type { LongRunContextRuntime } from "../context-runtime.js";
import {
  HumanoidActionRuntime,
  type HumanoidActionReceipt
} from "./runtime.js";
import {
  appendEmbodiedEpisode,
  recentEmbodiedEpisodes,
  retainRecentActionReceipts
} from "./embodied-memory.js";
import { reconcileHumanoidHierarchyCapabilities } from "./run-checkpoint.js";

const FRAME_CHECKPOINT_INTERVAL = 10;
const EMBODIED_RECALL_PAGE_SIZE = 64;

export interface HumanoidEmbodiedRecallRequest {
  source_refs?: string[];
  before_sequence?: number;
  limit: number;
}

type HistoricalHumanoidAction = HumanoidActionReceipt & {
  source_ref: string;
  historical_only: true;
};

export class HumanoidRunRuntime implements LongRunContextRuntime {
  readonly #store: RunStore;
  readonly #goal: Goal;
  readonly #world: HumanoidWorld;
  readonly #eventSink: RuntimeEventSink;
  readonly #signal: AbortSignal | undefined;
  readonly #actions: HumanoidActionRuntime;
  #checkpoint: HumanoidRunCheckpoint;

  constructor(input: {
    store: RunStore;
    goal: Goal;
    world: HumanoidWorld;
    checkpoint: HumanoidRunCheckpoint;
    eventSink?: RuntimeEventSink;
    signal?: AbortSignal;
  }) {
    this.#store = input.store;
    this.#goal = structuredClone(input.goal);
    this.#world = input.world;
    this.#checkpoint = reconcileHumanoidHierarchyCapabilities(input.checkpoint);
    assertHumanoidGoalProgressIntegrity(
      this.#goal,
      this.#world.snapshot(),
      this.#checkpoint.goal_progress
    );
    this.#eventSink = input.eventSink ?? (() => undefined);
    this.#signal = input.signal;
    this.#actions = new HumanoidActionRuntime(this.#world, {
      receipts: this.#checkpoint.committed_actions,
      frameSink: (frame) => this.#recordFrame(frame),
      receiptSink: (receipt) => this.#commitReceipt(receipt)
    });
  }

  get runId(): string {
    return this.#checkpoint.run_id;
  }

  get rootAgentId(): string {
    return this.#checkpoint.root_id;
  }

  get store(): RunStore {
    return this.#store;
  }

  get signal(): AbortSignal | undefined {
    return this.#signal;
  }

  get checkpoint(): HumanoidRunCheckpoint {
    return structuredClone(this.#checkpoint);
  }

  invoke(...args: Parameters<HumanoidActionRuntime["invoke"]>): ReturnType<HumanoidActionRuntime["invoke"]> {
    return this.#actions.invoke(...args);
  }

  receipt(transactionId: string): HumanoidActionReceipt | undefined {
    return this.#actions.receipt(transactionId);
  }

  async recallEmbodiedHistory(
    request: HumanoidEmbodiedRecallRequest
  ): Promise<JsonValue> {
    const requestedRefs = new Set(request.source_refs ?? []);
    const requestedEpisodeRefs = new Set(
      [...requestedRefs].filter((sourceRef) => sourceRef.startsWith("episode:"))
    );
    const requestedActionRefs = new Set(
      [...requestedRefs].filter((sourceRef) => sourceRef.startsWith("action:"))
    );
    const episodes = new Map<string, HumanoidEmbodiedEpisode>();
    let actionBeforeTime: string | undefined;
    const consider = (rawEpisode: HumanoidEmbodiedEpisode): void => {
      const episode = HumanoidEmbodiedEpisodeSchema.parse(rawEpisode);
      const sourceRef = episode.source_ref ?? `episode:${episode.sequence}`;
      if (requestedRefs.size === 0
        && request.before_sequence === episode.sequence) {
        actionBeforeTime = episode.recorded_at;
      }
      if (request.before_sequence !== undefined
        && episode.sequence >= request.before_sequence) return;
      if (requestedRefs.size > 0 && !requestedEpisodeRefs.has(sourceRef)) return;
      episodes.set(sourceRef, { ...episode, source_ref: sourceRef });
    };
    for (const episode of [...this.#checkpoint.embodied_memory.recent_episodes].reverse()) {
      consider(episode);
    }

    const enoughEpisodes = (): boolean => requestedRefs.size > 0
      ? [...requestedEpisodeRefs].every((sourceRef) => episodes.has(sourceRef))
      : episodes.size >= request.limit;
    if (!enoughEpisodes()) {
      const tail = await this.#store.readJournalTail("episodes", 1);
      for (let end = tail.total; end > 0 && !enoughEpisodes();) {
        const from = Math.max(0, end - EMBODIED_RECALL_PAGE_SIZE);
        const page = await this.#store.readJournalPage("episodes", from, end - from);
        for (let index = page.entries.length - 1; index >= 0; index -= 1) {
          consider(HumanoidEmbodiedEpisodeSchema.parse(page.entries[index]));
          if (enoughEpisodes()) break;
        }
        end = from;
      }
    }

    const actions = new Map<string, HistoricalHumanoidAction>();
    const enoughActions = (): boolean => requestedRefs.size > 0
      ? [...requestedActionRefs].every((sourceRef) => actions.has(sourceRef))
      : actions.size >= request.limit;
    if (!enoughActions()) {
      const tail = await this.#store.readJournalTail("actions", 1);
      for (let end = tail.total; end > 0 && !enoughActions();) {
        const from = Math.max(0, end - EMBODIED_RECALL_PAGE_SIZE);
        const page = await this.#store.readJournalPage("actions", from, end - from);
        for (let index = page.entries.length - 1; index >= 0; index -= 1) {
          const receipt = executeActionJournalReceipt(page.entries[index]!);
          if (!receipt) continue;
          if (requestedRefs.size === 0
            && actionBeforeTime !== undefined
            && receipt.committedAt >= actionBeforeTime) continue;
          const sourceRef = `action:${receipt.transactionId}`;
          if (requestedRefs.size > 0 && !requestedActionRefs.has(sourceRef)) continue;
          actions.set(sourceRef, {
            ...receipt,
            source_ref: sourceRef,
            historical_only: true
          });
          if (enoughActions()) break;
        }
        end = from;
      }
    }

    const selectedRecords = [
      ...[...episodes.values()].map((episode) => ({
        kind: "episode" as const,
        sourceRef: episode.source_ref!,
        recordedAt: episode.recorded_at,
        value: episode
      })),
      ...[...actions.values()].map((action) => ({
        kind: "action" as const,
        sourceRef: action.source_ref,
        recordedAt: action.committedAt,
        value: action
      }))
    ].sort((left, right) => (
      right.recordedAt.localeCompare(left.recordedAt)
        || right.kind.localeCompare(left.kind)
        || right.sourceRef.localeCompare(left.sourceRef)
    )).slice(0, request.limit);
    const selectedEpisodes = selectedRecords.flatMap((record) => (
      record.kind === "episode" ? [record.value as HumanoidEmbodiedEpisode] : []
    ));
    const selectedActions = selectedRecords.flatMap((record) => (
      record.kind === "action" ? [record.value as HistoricalHumanoidAction] : []
    ));
    const returnedRefs = new Set(selectedRecords.map((record) => record.sourceRef));
    return json({
      historical_only: true,
      current_world_revision: this.#world.snapshot().worldRevision,
      ordered_source_refs: selectedRecords.map((record) => record.sourceRef),
      episodes: selectedEpisodes,
      actions: selectedActions,
      missing_source_refs: [...requestedRefs].filter((sourceRef) => (
        !returnedRefs.has(sourceRef)
      )),
      next_before_sequence: requestedRefs.size === 0
        && selectedEpisodes.length > 0
        && Math.min(...selectedEpisodes.map((episode) => episode.sequence)) > 1
        ? Math.min(...selectedEpisodes.map((episode) => episode.sequence))
        : null
    });
  }

  validateCycleEvidence(
    evidenceTransactionIds: readonly string[]
  ): HumanoidActionReceipt {
    const evidence = evidenceTransactionIds.map((transactionId) => {
      const receipt = this.#checkpoint.committed_actions[transactionId];
      if (!receipt) throw new Error(`Unknown humanoid cycle evidence: ${transactionId}`);
      return receipt;
    });
    const currentRevision = this.#world.snapshot().worldRevision;
    const execution = evidence.findLast((receipt) => (
      completedPhysicalExecution(receipt)
      && receipt.worldAfterRevision === currentRevision
    ));
    if (!execution) {
      throw new Error(
        `Autonomous cycle requires accepted execution evidence at world revision ${currentRevision}`
      );
    }
    if (previousCycleEvidence(this.#checkpoint.last_cycle).has(execution.transactionId)) {
      throw new Error(`Humanoid execution evidence was already consumed: ${execution.transactionId}`);
    }
    const receipts = Object.values(this.#checkpoint.committed_actions);
    const executionIndex = receipts.findIndex((receipt) => (
      receipt.transactionId === execution.transactionId
    ));
    const pendingPlan = receipts.slice(executionIndex + 1).find((receipt) => (
      receipt.accepted
      && (receipt.action === "plan_whole_body_motion"
        || receipt.action === "plan_whole_body_motion_candidates"
        || receipt.action === "plan_humanoid_navigation")
    ));
    if (pendingPlan) {
      throw new Error(
        `Autonomous cycle has an unconsumed accepted plan: ${pendingPlan.transactionId}`
      );
    }
    return structuredClone(execution);
  }

  snapshot(): HumanoidWorldSnapshot {
    return this.#world.snapshot();
  }

  activeNode(agentId?: string): TaskNode {
    const id = agentId ?? this.#checkpoint.active_agent_id ?? this.rootAgentId;
    const node = this.#checkpoint.nodes[id];
    if (!node) throw new Error(`Unknown humanoid hierarchy node: ${id}`);
    return structuredClone(node);
  }

  contextMemory(): ContextMemoryState {
    return structuredClone(this.#checkpoint.context_memory);
  }

  contextWorldIdentity(): { worldRevision: number } {
    return { worldRevision: this.#checkpoint.world.worldRevision };
  }

  contextReceipts(): Record<string, { accepted: boolean; worldRevision: number }> {
    return Object.fromEntries(Object.entries(this.#checkpoint.committed_actions).map(
      ([transactionId, receipt]) => [transactionId, {
        accepted: receipt.accepted,
        worldRevision: receipt.worldAfterRevision
      }]
    ));
  }

  contextAnchor(agentId: string): JsonValue {
    const node = this.activeNode(agentId);
    const world = this.#world.snapshot();
    const checker = inspectHumanoidGoal(
      this.#goal,
      this.#store.definition.scenario,
      world,
      this.#checkpoint.goal_progress
    );
    const recentReceipts = Object.values(this.#checkpoint.committed_actions)
      .slice(-16)
      .map((receipt) => ({
        transaction_id: receipt.transactionId,
        agent_id: receipt.agentId,
        action: receipt.action,
        accepted: receipt.accepted,
        code: receipt.code,
        world_before_revision: receipt.worldBeforeRevision,
        world_after_revision: receipt.worldAfterRevision,
        frame_count: receipt.frameCount
      }));
    return json({
      mission: this.#store.definition.mission,
      scenario_id: this.#store.definition.scenario_id,
      goal: this.#goal,
      cycle_index: this.#checkpoint.cycle_index,
      world_frame: world.frame,
      world_revision: world.worldRevision,
      robot: {
        root_position: world.robot.rootPosition,
        root_rotation: world.robot.rootRotation,
        fallen: world.robot.fallen,
        balance: world.robot.balance,
        feet: world.robot.feet,
        navigation: world.navigation
      },
      goal_state: checker,
      recent_physical_episodes: recentEmbodiedEpisodes(
        this.#checkpoint.embodied_memory
      ).map((episode) => ({
        ...episode,
        historical_only: true
      })),
      active_agent: {
        id: node.id,
        name: node.name,
        objective: node.objective,
        capabilities: node.capabilities,
        status: node.status
      },
      recent_receipts: recentReceipts
    });
  }

  assertContextSummaryEvidence(summary: ContextCompactionSummary): void {
    for (const item of summary.completed) {
      for (const transactionId of item.transaction_ids) {
        const receipt = this.#checkpoint.committed_actions[transactionId];
        if (!receipt) {
          throw new Error(`Context compaction referenced an unknown transaction: ${transactionId}`);
        }
        if (!receipt.accepted) {
          throw new Error(`Completed context memory referenced a rejected transaction: ${transactionId}`);
        }
      }
    }
    for (const item of summary.blockers) {
      for (const transactionId of item.transaction_ids) {
        if (!this.#checkpoint.committed_actions[transactionId]) {
          throw new Error(`Context compaction referenced an unknown transaction: ${transactionId}`);
        }
      }
    }
  }

  async updateContextMemory(
    state: ContextMemoryState,
    journalRecord?: JsonValue
  ): Promise<void> {
    if (journalRecord !== undefined) await this.#store.append("context", journalRecord);
    this.#checkpoint.context_memory = structuredClone(state);
    await this.#persist();
    await this.emit("context_memory_updated", { context_memory: json(state) });
  }

  async recordModelCallStarted(agentId: string): Promise<void> {
    this.#signal?.throwIfAborted();
    const node = this.#node(agentId);
    node.model_calls_used += 1;
    node.updated_at = new Date().toISOString();
    this.#checkpoint.total_model_calls += 1;
    await this.#persist();
    await this.emit("model_request_started", {
      agent_id: agentId,
      agent_name: node.name,
      purpose: "agent_decision",
      node_model_calls: node.model_calls_used,
      total_model_calls: this.#checkpoint.total_model_calls
    });
  }

  async recordCompactionModelCall(agentId: string): Promise<void> {
    await this.#recordCompactionCalls(agentId, 1, "model_request_started");
  }

  async reconcileCompactionModelCalls(agentId: string, additionalCalls: number): Promise<void> {
    if (!Number.isSafeInteger(additionalCalls) || additionalCalls <= 0) return;
    await this.#recordCompactionCalls(agentId, additionalCalls, "model_requests_reconciled");
  }

  async setActiveAgent(agentId: string): Promise<void> {
    if (this.#checkpoint.active_agent_id === agentId) return;
    const at = new Date().toISOString();
    for (const node of Object.values(this.#checkpoint.nodes)) {
      if (node.id === this.rootAgentId) {
        node.status = agentId === node.id ? "active" : "waiting";
      } else {
        node.status = agentId === node.id ? "active" : "ready";
      }
      node.updated_at = at;
    }
    this.#checkpoint.active_agent_id = agentId;
    this.#checkpoint.active_agent_ids = [agentId];
    await this.#persist();
    await this.emit("hierarchy_focus_changed", {
      active_agent_id: agentId,
      nodes: json(this.#checkpoint.nodes)
    });
  }

  async start(resumed: boolean): Promise<void> {
    const at = new Date().toISOString();
    this.#checkpoint.status = "running";
    this.#checkpoint.error = null;
    this.#checkpoint.active_agent_id = this.rootAgentId;
    this.#checkpoint.active_agent_ids = [this.rootAgentId];
    this.#node(this.rootAgentId).status = "active";
    const type = resumed ? "run_resumed" : "run_started";
    this.#checkpoint.pending_lifecycle_events.push(createLifecycleEvent({
      runId: this.runId,
      type,
      at,
      data: {
        runtime: "humanoid_g1",
        world_frame: this.#checkpoint.world.frame,
        world_revision: this.#checkpoint.world.worldRevision
      }
    }));
    await this.#persist();
    await reconcileLifecycleOutbox({
      store: this.#store,
      checkpoint: this.#checkpoint,
      persistCheckpoint: () => this.#persist(),
      eventSink: this.#eventSink
    });
  }

  async completeCycle(output: string): Promise<boolean> {
    let cycle: JsonValue;
    try {
      cycle = json(JSON.parse(output));
    } catch {
      cycle = output;
    }
    const checker = inspectHumanoidGoal(
      this.#goal,
      this.#store.definition.scenario,
      this.#world.snapshot(),
      this.#checkpoint.goal_progress
    );
    const world = this.#world.snapshot();
    const evidence = previousCycleEvidence(cycle);
    const execution = Object.values(this.#checkpoint.committed_actions).findLast((receipt) => (
      evidence.has(receipt.transactionId)
      && completedPhysicalExecution(receipt)
      && receipt.worldAfterRevision === world.worldRevision
    ));
    if (!execution) {
      throw new Error("Autonomous cycle completion lacks current physical execution evidence");
    }
    this.#checkpoint.cycle_index += 1;
    this.#checkpoint.last_cycle = cycle;
    this.#checkpoint.checker = checker;
    const memory = appendEmbodiedEpisode({
      state: this.#checkpoint.embodied_memory,
      sequence: this.#checkpoint.cycle_index,
      execution,
      modelSummary: cycleSummary(cycle),
      world,
      goalSuccess: checker.success
    });
    await this.#persistEmbodiedEpisode(memory.episode);
    this.#checkpoint.embodied_memory = memory.state;
    const actionWindow = retainRecentActionReceipts(
      this.#checkpoint.committed_actions
    );
    this.#checkpoint.committed_actions = actionWindow.receipts;
    await this.#persist(true);
    await this.#store.append("checker", json(checker));
    await this.emit("embodied_episode_recorded", {
      episode: json(memory.episode),
      embodied_memory: json(memory.state),
      retained_episodes: memory.state.recent_episodes.length,
      total_episodes: memory.state.total_episodes,
      pruned_checkpoint_receipts: actionWindow.removed,
      historical_only: false
    });
    await this.emit("autonomous_cycle_completed", {
      cycle_index: this.#checkpoint.cycle_index,
      output: cycle,
      checker: json(checker)
    });
    return checker.success;
  }

  async succeed(output: string): Promise<void> {
    await this.#finish("succeeded", output, null, "run_succeeded");
  }

  async fail(error: string): Promise<void> {
    await this.#finish("failed", null, error, "run_failed");
  }

  async interrupt(error: string): Promise<void> {
    await this.#finish("interrupted", null, error, "run_interrupted");
  }

  async recordFramework(scope: string, event: JsonValue, agentId?: string): Promise<void> {
    const runtimeEventId = randomUUID();
    const record = {
      scope,
      ...(agentId ? { agent_id: agentId } : {}),
      event,
      at: new Date().toISOString(),
      runtime_event_id: runtimeEventId
    };
    await this.#store.append("framework", json(record));
    await this.emit("framework_event", json(record), runtimeEventId);
  }

  async recordProvider(event: JsonValue, agentId?: string): Promise<void> {
    const runtimeEventId = randomUUID();
    const record = {
      ...(agentId ? { agent_id: agentId } : {}),
      ...object(event),
      at: new Date().toISOString(),
      runtime_event_id: runtimeEventId
    };
    await this.#store.append("provider", json(record));
    await this.emit("provider_event", json(record), runtimeEventId);
  }

  async emit(
    type: string,
    data: JsonValue,
    eventId = randomUUID(),
    durable = true
  ): Promise<void> {
    const event: RuntimeEvent = {
      event_id: eventId,
      run_id: this.runId,
      type,
      at: new Date().toISOString(),
      data,
      ...(durable ? {} : { durable: false })
    };
    if (!durable) {
      await this.#eventSink(event);
      return;
    }
    const [persisted] = await this.#store.appendRuntimeEvents([event]);
    await this.#eventSink(persisted!);
  }

  async #recordFrame(frame: HumanoidWorldSnapshot): Promise<void> {
    const advanced = advanceHumanoidGoal(
      this.#goal,
      this.#store.definition.scenario,
      frame,
      this.#checkpoint.goal_progress
    );
    this.#checkpoint.world = structuredClone(frame);
    this.#checkpoint.goal_progress = advanced.progress;
    this.#checkpoint.checker = advanced.checker;
    if (frame.frame % FRAME_CHECKPOINT_INTERVAL === 0) {
      this.#checkpoint.world_checkpoint = this.#world.checkpoint();
      await this.#persist();
    }
    this.#signal?.throwIfAborted();
    await this.emit("humanoid_world_frame", json({
      world: frame,
      checker: advanced.checker,
      goal_progress: advanced.progress
    }), randomUUID(), false);
  }

  async #commitReceipt(receipt: HumanoidActionReceipt): Promise<void> {
    const node = this.#node(receipt.agentId);
    node.steps_used += 1;
    node.status = "ready";
    node.updated_at = new Date().toISOString();
    this.#checkpoint.committed_actions[receipt.transactionId] = structuredClone(receipt);
    this.#checkpoint.world = this.#world.snapshot();
    this.#checkpoint.world_checkpoint = this.#world.checkpoint();
    this.#checkpoint.checker = inspectHumanoidGoal(
      this.#goal,
      this.#store.definition.scenario,
      this.#checkpoint.world,
      this.#checkpoint.goal_progress
    );
    await this.#persist();
    const runtimeEventId = randomUUID();
    const record = {
      ...receipt,
      runtime_event_id: runtimeEventId
    };
    await this.#store.append("actions", json(record));
    await this.emit("humanoid_action_committed", json({
      receipt: record,
      world: this.#checkpoint.world,
      checker: this.#checkpoint.checker
    }), runtimeEventId);
  }

  async #recordCompactionCalls(
    agentId: string,
    count: number,
    eventType: "model_request_started" | "model_requests_reconciled"
  ): Promise<void> {
    this.#signal?.throwIfAborted();
    const node = this.#node(agentId);
    node.model_calls_used += count;
    node.updated_at = new Date().toISOString();
    this.#checkpoint.total_model_calls += count;
    await this.#persist();
    await this.emit(eventType, {
      agent_id: agentId,
      agent_name: node.name,
      purpose: "context_compaction",
      model_calls: count,
      node_model_calls: node.model_calls_used,
      total_model_calls: this.#checkpoint.total_model_calls
    });
  }

  async #persistEmbodiedEpisode(episode: HumanoidEmbodiedEpisode): Promise<void> {
    const parsed = HumanoidEmbodiedEpisodeSchema.parse(episode);
    const tail = await this.#store.readJournalTail("episodes", 1);
    const last = tail.entries[0];
    if (last !== undefined) {
      const previous = HumanoidEmbodiedEpisodeSchema.parse(last);
      if (previous.source_ref === parsed.source_ref) return;
    }
    await this.#store.append("episodes", json(parsed));
  }

  async #finish(
    status: "succeeded" | "failed" | "interrupted",
    output: string | null,
    error: string | null,
    eventType: "run_succeeded" | "run_failed" | "run_interrupted"
  ): Promise<void> {
    const at = new Date().toISOString();
    this.#checkpoint.status = status;
    this.#checkpoint.final_output = output;
    this.#checkpoint.error = error;
    this.#checkpoint.active_agent_id = null;
    this.#checkpoint.active_agent_ids = [];
    this.#node(this.rootAgentId).status = status === "succeeded" ? "completed" : "failed";
    this.#checkpoint.pending_lifecycle_events.push(createLifecycleEvent({
      runId: this.runId,
      type: eventType,
      at,
      data: {
        runtime: "humanoid_g1",
        ...(output ? { output } : {}),
        ...(error ? { error } : {})
      }
    }));
    await this.#persist(true);
    await reconcileLifecycleOutbox({
      store: this.#store,
      checkpoint: this.#checkpoint,
      persistCheckpoint: () => this.#persist(),
      eventSink: this.#eventSink
    });
  }

  #node(agentId: string): TaskNode {
    const node = this.#checkpoint.nodes[agentId];
    if (!node) throw new Error(`Unknown humanoid hierarchy node: ${agentId}`);
    return node;
  }

  async #persist(refreshWorld = false): Promise<void> {
    if (refreshWorld) {
      this.#checkpoint.world = this.#world.snapshot();
      this.#checkpoint.world_checkpoint = this.#world.checkpoint();
    }
    assertHumanoidGoalProgressIntegrity(
      this.#goal,
      this.#checkpoint.world,
      this.#checkpoint.goal_progress
    );
    this.#checkpoint.updated_at = new Date().toISOString();
    await this.#store.writeCheckpoint(this.#checkpoint);
  }
}

function object(value: JsonValue): Record<string, JsonValue> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value
    : { value };
}

function json(value: unknown): JsonValue {
  const serialized = JSON.stringify(value);
  if (serialized === undefined) return null;
  return JSON.parse(serialized) as JsonValue;
}

function previousCycleEvidence(value: JsonValue): Set<string> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return new Set();
  const ids = value.evidence_transaction_ids;
  if (!Array.isArray(ids)) return new Set();
  return new Set(ids.filter((entry): entry is string => typeof entry === "string"));
}

function cycleSummary(value: JsonValue): string {
  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    const summary = value.summary;
    if (typeof summary === "string" && summary.trim()) return summary.trim();
  }
  return "完成一次有物理回执的人形自主循环";
}

function completedPhysicalExecution(receipt: HumanoidActionReceipt): boolean {
  return receipt.accepted && (
    receipt.action === "execute_whole_body_motion"
      ? receipt.code === "motion_option_succeeded"
      : receipt.action === "execute_humanoid_navigation"
        && receipt.code === "navigation_completed"
  );
}

function executeActionJournalReceipt(value: JsonValue): HumanoidActionReceipt | undefined {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Humanoid action journal contains a non-object record");
  }
  if (value.action !== "execute_whole_body_motion"
    && value.action !== "execute_humanoid_navigation") return undefined;
  const { runtime_event_id: _runtimeEventId, ...receipt } = value;
  return PersistedHumanoidActionReceiptSchema.parse(receipt);
}
