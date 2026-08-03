import { randomUUID } from "node:crypto";
import type {
  ContextCompactionSummary,
  ContextMemoryState,
  Goal,
  JsonValue,
  TaskNode
} from "../../domain/schema.js";
import type { HumanoidRunCheckpoint } from "../../domain/humanoid-run.js";
import type { RunStore } from "../../persistence/run-store.js";
import {
  createLifecycleEvent,
  reconcileLifecycleOutbox
} from "../../persistence/lifecycle-outbox.js";
import type { RuntimeEvent, RuntimeEventSink } from "../../runtime/events.js";
import { checkHumanoidGoal } from "../../runtime/humanoid-checker.js";
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

const FRAME_CHECKPOINT_INTERVAL = 10;

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
    this.#checkpoint = structuredClone(input.checkpoint);
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
      receipt.accepted
      && receipt.worldAfterRevision === currentRevision
      && (receipt.action === "execute_whole_body_motion"
        || receipt.action === "execute_humanoid_navigation")
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
    const checker = checkHumanoidGoal(this.#goal, this.#store.definition.scenario, world);
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
        historical_only: episode.world_after_revision !== world.worldRevision
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
    const checker = checkHumanoidGoal(
      this.#goal,
      this.#store.definition.scenario,
      this.#world.snapshot()
    );
    const world = this.#world.snapshot();
    const evidence = previousCycleEvidence(cycle);
    const execution = Object.values(this.#checkpoint.committed_actions).findLast((receipt) => (
      evidence.has(receipt.transactionId)
      && receipt.accepted
      && receipt.worldAfterRevision === world.worldRevision
      && (receipt.action === "execute_whole_body_motion"
        || receipt.action === "execute_humanoid_navigation")
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
    this.#signal?.throwIfAborted();
    this.#checkpoint.world = structuredClone(frame);
    if (frame.frame % FRAME_CHECKPOINT_INTERVAL === 0) {
      this.#checkpoint.world_checkpoint = this.#world.checkpoint();
      await this.#persist();
    }
    await this.emit("humanoid_world_frame", json({ world: frame }), randomUUID(), false);
  }

  async #commitReceipt(receipt: HumanoidActionReceipt): Promise<void> {
    const node = this.#node(receipt.agentId);
    node.steps_used += 1;
    node.status = "ready";
    node.updated_at = new Date().toISOString();
    this.#checkpoint.committed_actions[receipt.transactionId] = structuredClone(receipt);
    this.#checkpoint.world = this.#world.snapshot();
    this.#checkpoint.world_checkpoint = this.#world.checkpoint();
    this.#checkpoint.checker = checkHumanoidGoal(
      this.#goal,
      this.#store.definition.scenario,
      this.#checkpoint.world
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
