import { z } from "zod";
import {
  NeuralPathwaySchema,
  type NeuralPathway
} from "../../domain/neural-hierarchy.js";
import {
  HUMANOID_NEURAL_AGENT_IDS,
  HUMANOID_NEURAL_NODE_BY_ID,
  type HumanoidNeuralAgentId
} from "./neural-hierarchy-contract.js";

const NeuralSchedulerEventBaseSchema = z.object({
  event_id: z.string().uuid(),
  at: z.string().datetime(),
  world_revision: z.number().int().nonnegative(),
  causal_signal_ids: z.array(z.string().uuid()).max(64).default([]),
  causal_interrupt_ids: z.array(z.string().uuid()).max(64).optional()
});

export const NeuralSchedulerEventSchema = z.discriminatedUnion("kind", [
  NeuralSchedulerEventBaseSchema.extend({
    kind: z.literal("run_started")
  }).strict(),
  NeuralSchedulerEventBaseSchema.extend({
    kind: z.literal("no_active_goal")
  }).strict(),
  NeuralSchedulerEventBaseSchema.extend({
    kind: z.literal("goal_selected")
  }).strict(),
  NeuralSchedulerEventBaseSchema.extend({
    kind: z.literal("world_revision_changed")
  }).strict(),
  NeuralSchedulerEventBaseSchema.extend({
    kind: z.literal("perception_stale")
  }).strict(),
  NeuralSchedulerEventBaseSchema.extend({
    kind: z.literal("commitment_absent")
  }).strict(),
  NeuralSchedulerEventBaseSchema.extend({
    kind: z.literal("commitment_released")
  }).strict(),
  NeuralSchedulerEventBaseSchema.extend({
    kind: z.literal("rollout_completed")
  }).strict(),
  NeuralSchedulerEventBaseSchema.extend({
    kind: z.literal("execution_completed")
  }).strict(),
  NeuralSchedulerEventBaseSchema.extend({
    kind: z.literal("execution_failed")
  }).strict(),
  NeuralSchedulerEventBaseSchema.extend({
    kind: z.literal("prediction_error"),
    correction_scope: z.enum(["local", "pathway", "supervisory"])
  }).strict(),
  NeuralSchedulerEventBaseSchema.extend({
    kind: z.literal("skill_completed")
  }).strict(),
  NeuralSchedulerEventBaseSchema.extend({
    kind: z.literal("skill_failed")
  }).strict(),
  NeuralSchedulerEventBaseSchema.extend({
    kind: z.literal("cycle_ready")
  }).strict(),
  NeuralSchedulerEventBaseSchema.extend({
    kind: z.literal("escalation"),
    source_node_id: z.string().trim().min(1)
  }).strict()
]);

export type NeuralSchedulerEvent = z.infer<typeof NeuralSchedulerEventSchema>;

export interface NeuralWakeAuthorityHop {
  parentNodeId: HumanoidNeuralAgentId;
  childNodeId: HumanoidNeuralAgentId;
  authorityLeaseId: string;
}

export interface NeuralWakeAuthority {
  targetNodeId: HumanoidNeuralAgentId;
  parentNodeId: HumanoidNeuralAgentId | null;
  authorityLeaseId: string | null;
  /** Ordered Executive -> target control path. Empty only for Executive. */
  authorityPath: readonly NeuralWakeAuthorityHop[];
}

export interface NeuralSchedulerDispatch {
  executiveNodeId: typeof HUMANOID_NEURAL_AGENT_IDS.executive;
  requestedTargetNodeId: HumanoidNeuralAgentId;
  authorizedTargetNodeId: HumanoidNeuralAgentId;
  parentNodeId: HumanoidNeuralAgentId | null;
  authorityLeaseId: string | null;
  authorityPath: readonly NeuralWakeAuthorityHop[];
  pathway: NeuralPathway;
  event: NeuralSchedulerEvent;
  signal: AbortSignal;
}

export interface NeuralHierarchySchedulerOptions {
  dispatch(input: NeuralSchedulerDispatch): Promise<void>;
  resolveAuthority(input: {
    requestedTargetNodeId: HumanoidNeuralAgentId;
    event: NeuralSchedulerEvent;
  }): Promise<NeuralWakeAuthority> | NeuralWakeAuthority;
  maximumQueuedEvents?: number;
  onFailure?: (error: unknown, event: NeuralSchedulerEvent) => void | Promise<void>;
}

/**
 * Event-driven wake-up scheduler for the neural hierarchy.
 *
 * It owns no robot authority. A wake-up only lets a structural node consume
 * already-valid directed signals. The node's parent, route contract, world
 * revision and TTL remain the sources of control authority.
 */
export class NeuralHierarchyScheduler {
  readonly #dispatch: NeuralHierarchySchedulerOptions["dispatch"];
  readonly #resolveAuthority: NeuralHierarchySchedulerOptions["resolveAuthority"];
  readonly #onFailure: NeuralHierarchySchedulerOptions["onFailure"];
  readonly #maximumQueuedEvents: number;
  readonly #lifetime = new AbortController();
  readonly #queued = new Map<string, NeuralSchedulerEvent>();
  readonly #activeByNode = new Map<string, Promise<void>>();
  readonly #deadlineTimers = new Map<NeuralPathway, ReturnType<typeof setTimeout>>();
  readonly #idleWaiters = new Set<{
    resolve: () => void;
    reject: (error: unknown) => void;
  }>();
  #pumping = false;
  #stopped = false;
  #failure: unknown;

  constructor(options: NeuralHierarchySchedulerOptions) {
    this.#dispatch = options.dispatch;
    this.#resolveAuthority = options.resolveAuthority;
    this.#onFailure = options.onFailure;
    this.#maximumQueuedEvents = positiveInteger(
      options.maximumQueuedEvents ?? 256,
      "Neural scheduler queue limit"
    );
  }

  publish(rawEvent: NeuralSchedulerEvent): void {
    if (this.#stopped) throw new Error("Neural hierarchy scheduler is stopped");
    if (this.#failure) throw schedulerFailure(this.#failure);
    const event = NeuralSchedulerEventSchema.parse(rawEvent);
    const target = neuralWakeTarget(event);
    if (target === null) return;
    const key = schedulerEventKey(event, target);
    const existing = this.#queued.get(key);
    if (existing && existing.world_revision > event.world_revision) return;
    if (!existing && this.#queued.size >= this.#maximumQueuedEvents) {
      throw new Error(
        `Neural hierarchy scheduler queue exceeded ${this.#maximumQueuedEvents} events`
      );
    }
    this.#queued.set(key, event);
    this.#requestPump();
  }

  scheduleStalenessDeadline(input: {
    pathway: NeuralPathway;
    delayMs: number;
    event: NeuralSchedulerEvent;
  }): void {
    if (this.#stopped) throw new Error("Neural hierarchy scheduler is stopped");
    const pathway = NeuralPathwaySchema.parse(input.pathway);
    if (!Number.isFinite(input.delayMs) || input.delayMs < 0) {
      throw new Error("Neural staleness deadline must be non-negative");
    }
    const event = NeuralSchedulerEventSchema.parse(input.event);
    const previous = this.#deadlineTimers.get(pathway);
    if (previous) clearTimeout(previous);
    const timer = setTimeout(() => {
      this.#deadlineTimers.delete(pathway);
      if (this.#stopped || this.#lifetime.signal.aborted) return;
      try {
        this.publish(event);
      } catch (error) {
        void this.#fail(error, event);
      }
    }, input.delayMs);
    this.#deadlineTimers.set(pathway, timer);
  }

  async waitForIdle(): Promise<void> {
    if (this.#failure) throw schedulerFailure(this.#failure);
    if (this.#queued.size === 0 && this.#activeByNode.size === 0 && !this.#pumping) {
      return;
    }
    await new Promise<void>((resolve, reject) => {
      this.#idleWaiters.add({ resolve, reject });
    });
  }

  async shutdown(reason: unknown = new Error("Neural scheduler shutdown")): Promise<void> {
    if (!this.#stopped) {
      this.#stopped = true;
      this.#queued.clear();
      for (const timer of this.#deadlineTimers.values()) clearTimeout(timer);
      this.#deadlineTimers.clear();
      this.#lifetime.abort(reason);
    }
    await Promise.allSettled(this.#activeByNode.values());
    this.#settleIdleWaiters();
  }

  snapshot(): {
    queuedEvents: number;
    activeNodeIds: string[];
    deadlinePathways: NeuralPathway[];
    stopped: boolean;
    failed: boolean;
  } {
    return {
      queuedEvents: this.#queued.size,
      activeNodeIds: [...this.#activeByNode.keys()],
      deadlinePathways: [...this.#deadlineTimers.keys()],
      stopped: this.#stopped,
      failed: this.#failure !== undefined
    };
  }

  #requestPump(): void {
    if (this.#pumping || this.#stopped) return;
    this.#pumping = true;
    queueMicrotask(() => this.#pump());
  }

  #pump(): void {
    if (this.#stopped || this.#failure) {
      this.#pumping = false;
      this.#settleIdleWaiters();
      return;
    }
    // The scheduler is an interrupt mux for one Executive root, not an Agent
    // worker pool. Legal parallelism is opened only inside a running Manager
    // through its explicit read-only child fan-out.
    while (this.#activeByNode.size === 0) {
      const next = this.#nextRunnableEvent();
      if (!next) break;
      const { key, event, requestedTargetNodeId } = next;
      this.#queued.delete(key);
      const reservationKey = requestedTargetNodeId;
      const task = Promise.resolve(this.#resolveAuthority({
        requestedTargetNodeId,
        event
      })).then(async (authority) => {
        assertResolvedWakeAuthority(requestedTargetNodeId, authority);
        const descriptor = HUMANOID_NEURAL_NODE_BY_ID.get(authority.targetNodeId);
        if (!descriptor || descriptor.executionKind !== "model_agent") {
          throw new Error(
            `Scheduler cannot wake a non-model node: ${authority.targetNodeId}`
          );
        }
        await this.#dispatch({
          ...authority,
          executiveNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
          requestedTargetNodeId,
          authorizedTargetNodeId: authority.targetNodeId,
          pathway: descriptor.pathway,
          event,
          signal: this.#lifetime.signal
        });
      }).catch((error) => this.#fail(error, event)).finally(() => {
        this.#activeByNode.delete(reservationKey);
        this.#requestPump();
        this.#settleIdleWaiters();
      });
      this.#activeByNode.set(reservationKey, task);
    }
    this.#pumping = false;
    if (this.#queued.size > 0 && this.#activeByNode.size === 0) {
      this.#requestPump();
    }
    this.#settleIdleWaiters();
  }

  #nextRunnableEvent(): {
    key: string;
    event: NeuralSchedulerEvent;
    requestedTargetNodeId: HumanoidNeuralAgentId;
  } | undefined {
    for (const [key, event] of this.#queued) {
      const targetNodeId = neuralWakeTarget(event);
      if (targetNodeId === null || this.#activeByNode.has(targetNodeId)) continue;
      return { key, event, requestedTargetNodeId: targetNodeId };
    }
    return undefined;
  }

  async #fail(error: unknown, event: NeuralSchedulerEvent): Promise<void> {
    if (this.#failure === undefined) {
      this.#failure = error;
      this.#queued.clear();
      this.#lifetime.abort(error);
      try {
        await this.#onFailure?.(error, event);
      } catch (reportingError) {
        this.#failure = new AggregateError(
          [error, reportingError],
          "Neural scheduler dispatch and failure reporting both failed"
        );
      }
    }
    this.#settleIdleWaiters();
  }

  #settleIdleWaiters(): void {
    if (this.#failure !== undefined) {
      for (const waiter of this.#idleWaiters) {
        waiter.reject(schedulerFailure(this.#failure));
      }
      this.#idleWaiters.clear();
      return;
    }
    if (this.#queued.size > 0 || this.#activeByNode.size > 0 || this.#pumping) return;
    for (const waiter of this.#idleWaiters) waiter.resolve();
    this.#idleWaiters.clear();
  }
}

export function neuralWakeTarget(
  event: NeuralSchedulerEvent
): HumanoidNeuralAgentId | null {
  switch (event.kind) {
    case "run_started":
    case "no_active_goal":
    case "cycle_ready":
      return HUMANOID_NEURAL_AGENT_IDS.executive;
    case "goal_selected":
    case "commitment_absent":
    case "commitment_released":
    case "skill_completed":
    case "skill_failed":
      return HUMANOID_NEURAL_AGENT_IDS.actionSelection;
    case "world_revision_changed":
    case "perception_stale":
      return HUMANOID_NEURAL_AGENT_IDS.perceptionManager;
    case "rollout_completed":
      return HUMANOID_NEURAL_AGENT_IDS.predictive;
    case "execution_completed":
    case "execution_failed":
      return HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager;
    case "prediction_error":
      if (event.correction_scope === "local") return null;
      return event.correction_scope === "pathway"
        ? HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager
        : HUMANOID_NEURAL_AGENT_IDS.actionSelection;
    case "escalation": {
      const source = HUMANOID_NEURAL_NODE_BY_ID.get(event.source_node_id);
      if (!source || source.parentKey === null) {
        return HUMANOID_NEURAL_AGENT_IDS.executive;
      }
      const parentId = HUMANOID_NEURAL_AGENT_IDS[source.parentKey];
      const parent = HUMANOID_NEURAL_NODE_BY_ID.get(parentId);
      if (!parent || parent.executionKind !== "model_agent") {
        return HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager;
      }
      return parent.id;
    }
  }
}

function assertResolvedWakeAuthority(
  requestedTargetNodeId: HumanoidNeuralAgentId,
  resolved: NeuralWakeAuthority
): void {
  const target = HUMANOID_NEURAL_NODE_BY_ID.get(resolved.targetNodeId);
  if (!target || target.executionKind !== "model_agent") {
    throw new Error(`Authority resolver returned unknown model node: ${resolved.targetNodeId}`);
  }
  if (!isStructuralAncestor(resolved.targetNodeId, requestedTargetNodeId)) {
    throw new Error(
      `Scheduler authority resolver crossed branches: ${requestedTargetNodeId} -> ${resolved.targetNodeId}`
    );
  }
  if (target.parentKey === null) {
    if (resolved.parentNodeId !== null || resolved.authorityLeaseId !== null
      || resolved.authorityPath.length !== 0) {
      throw new Error("The Executive root wake cannot claim a parent lease");
    }
    return;
  }
  const structuralParentId = HUMANOID_NEURAL_AGENT_IDS[target.parentKey];
  if (resolved.parentNodeId !== structuralParentId || resolved.authorityLeaseId === null) {
    throw new Error(
      `Non-root scheduler wake lacks its structural parent lease: ${resolved.targetNodeId}`
    );
  }
  assertCompleteAuthorityPath(resolved);
}

function assertCompleteAuthorityPath(resolved: NeuralWakeAuthority): void {
  let expectedParentId: HumanoidNeuralAgentId = HUMANOID_NEURAL_AGENT_IDS.executive;
  const leaseIds = new Set<string>();
  for (const hop of resolved.authorityPath) {
    if (hop.parentNodeId !== expectedParentId) {
      throw new Error(
        `Scheduler authority path is discontinuous at ${hop.parentNodeId} -> ${hop.childNodeId}`
      );
    }
    const child = HUMANOID_NEURAL_NODE_BY_ID.get(hop.childNodeId);
    if (!child || child.parentKey === null
      || HUMANOID_NEURAL_AGENT_IDS[child.parentKey] !== hop.parentNodeId) {
      throw new Error(
        `Scheduler authority path contains a non-structural edge: ${hop.parentNodeId} -> ${hop.childNodeId}`
      );
    }
    if (leaseIds.has(hop.authorityLeaseId)) {
      throw new Error("Scheduler authority path cannot reuse one lease across hierarchy edges");
    }
    leaseIds.add(hop.authorityLeaseId);
    expectedParentId = hop.childNodeId;
  }
  const tail = resolved.authorityPath.at(-1);
  if (!tail || expectedParentId !== resolved.targetNodeId
    || tail.parentNodeId !== resolved.parentNodeId
    || tail.authorityLeaseId !== resolved.authorityLeaseId) {
    throw new Error(
      `Scheduler wake lacks a complete Executive-to-target authority path: ${resolved.targetNodeId}`
    );
  }
}

function isStructuralAncestor(
  ancestorId: HumanoidNeuralAgentId,
  descendantId: HumanoidNeuralAgentId
): boolean {
  let current = HUMANOID_NEURAL_NODE_BY_ID.get(descendantId);
  while (current) {
    if (current.id === ancestorId) return true;
    if (current.parentKey === null) return false;
    current = HUMANOID_NEURAL_NODE_BY_ID.get(
      HUMANOID_NEURAL_AGENT_IDS[current.parentKey]
    );
  }
  return false;
}

function schedulerEventKey(
  event: NeuralSchedulerEvent,
  targetNodeId: HumanoidNeuralAgentId
): string {
  if ([
    "world_revision_changed",
    "perception_stale",
    "commitment_absent",
    "commitment_released"
  ].includes(event.kind)) {
    return `${targetNodeId}:${event.kind}`;
  }
  return `${targetNodeId}:${event.event_id}`;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new Error(`${label} must be a positive integer`);
  }
  return value;
}

function schedulerFailure(cause: unknown): Error {
  return cause instanceof Error
    ? cause
    : new Error("Neural hierarchy scheduler failed", { cause });
}
