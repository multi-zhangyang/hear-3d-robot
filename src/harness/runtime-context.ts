import { randomUUID } from "node:crypto";
import { isDeepStrictEqual } from "node:util";
import {
  AgentSpecSchema,
  EmptyContextMemoryState,
  type ActionReceipt,
  type AgentReference,
  type AgentSpec,
  type BodyChannel,
  type ContextCompactionSummary,
  type ContextMemoryState,
  type Goal,
  type JsonValue,
  type RunCheckpoint,
  type RunLifecycleEventType,
  type TaskNode,
  type WorldSnapshot
} from "../domain/schema.js";
import {
  createLifecycleEvent,
  reconcileLifecycleOutbox
} from "../persistence/lifecycle-outbox.js";
import type { RunStore } from "../persistence/run-store.js";
import {
  AgentSkillInputs,
  executeSkill,
  executeTool,
  isSkillName,
  requiredChannels
} from "../runtime/actions.js";
import { checkGoal, unmetGoalRecovery } from "../runtime/checker.js";
import { errorMessage } from "../runtime/error-message.js";
import type { CommandResult, RapierWorld } from "../world/rapier-world.js";
import {
  voxelAffordanceContractStale,
  withoutVoxelDynamicAffordances
} from "../world/voxel-affordance.js";
import { DenialLedger } from "./denial-ledger.js";
import type { DelegationRecoveryState } from "./delegation-drain.js";
import {
  assertEvidenceRequirementsJointlySatisfiable,
  assertReceiptRequirementDefinition,
  verifyBlockerEvidence,
  verifyGoalPredicateBlockerEvidence,
  verifyReceiptEvidence
} from "./evidence-contract.js";
import {
  goalMemoryContextRecords,
  goalRelevantSpatialMemory
} from "./goal-memory.js";
import { assertGoalCapabilityContract } from "./goal-capability.js";
import { pendingPlanReceipts, requiredPlanHandoff } from "./pending-plans.js";
import { SpatialMemory } from "./spatial-memory.js";
import {
  HierarchyProjection,
  type DelegationEntry
} from "./hierarchy-projection.js";

export interface RuntimeEvent {
  event_id: string;
  run_id: string;
  type: string;
  at: string;
  data: JsonValue;
  durable?: boolean;
  cursor?: string;
}

export type RuntimeEventSink = (event: RuntimeEvent) => void | Promise<void>;

interface ActionCommitSignature {
  agent_id: string;
  kind: ActionReceipt["kind"];
  name: string;
  input: JsonValue;
}

interface PendingActionCommit {
  signature: ActionCommitSignature;
  result: Promise<string>;
}

const FAILED_DELEGATION_DENIAL_LIMIT = 3;
export const PARTIAL_WORLD_CHECKPOINT_INTERVAL_FRAMES = 15;

/** The capability that consumes each planning capability's output. */
const PLAN_EXECUTORS: Record<string, string | undefined> = {
  plan_base_path: "execute_base_plan",
  plan_joint_targets: "execute_joint_plan",
  solve_end_effector_position: "execute_joint_plan",
  solve_end_effector_pose: "execute_joint_plan"
};

export class HarnessRuntimeContext {
  readonly #store: RunStore;
  readonly #goal: Goal;
  readonly #world: RapierWorld;
  readonly #hierarchy: HierarchyProjection;
  readonly #eventSink: RuntimeEventSink;
  readonly #signal: AbortSignal | undefined;
  readonly #bodyLeases = new Map<BodyChannel, string>();
  readonly #pendingActionCommits = new Map<string, PendingActionCommit>();
  readonly #capabilities: Set<string>;
  readonly #denials = new DenialLedger();
  readonly #spatialMemory: SpatialMemory;
  #checkpoint: RunCheckpoint;
  #lastBroadcastFrame: number;
  #lastPersistedWorldFrame: number;
  #checkpointWrite: Promise<void> = Promise.resolve();

  constructor(input: {
    store: RunStore;
    goal: Goal;
    world: RapierWorld;
    hierarchy: HierarchyProjection;
    checkpoint: RunCheckpoint;
    eventSink?: RuntimeEventSink;
    signal?: AbortSignal;
  }) {
    this.#store = input.store;
    this.#goal = structuredClone(input.goal);
    this.#world = input.world;
    this.#hierarchy = input.hierarchy;
    this.#checkpoint = structuredClone(input.checkpoint);
    this.#lastBroadcastFrame = input.checkpoint.world.frame;
    this.#lastPersistedWorldFrame = input.checkpoint.world.frame;
    this.#eventSink = input.eventSink ?? (() => undefined);
    this.#signal = input.signal;
    this.#capabilities = new Set(this.#checkpoint.capability_catalog);
    this.#spatialMemory = new SpatialMemory(
      `${this.#checkpoint.scenario_id}:${this.#store.definition.scenario.seed}`,
      this.#checkpoint.spatial_memory
    );
    if (this.#capabilities.size !== this.#checkpoint.capability_catalog.length) {
      throw new Error("Capability catalog contains duplicate names");
    }
    this.#world.setFrameSink((frames) => this.#recordWorldFrames(frames));
  }

  get runId(): string {
    return this.#checkpoint.run_id;
  }

  get rootAgentId(): string {
    return this.#hierarchy.rootId;
  }

  get store(): RunStore {
    return this.#store;
  }

  get checkpoint(): RunCheckpoint {
    return structuredClone(this.#checkpoint);
  }

  get signal(): AbortSignal | undefined {
    return this.#signal;
  }

  goal(): Goal {
    return structuredClone(this.#goal);
  }

  worldObservation(): JsonValue {
    return this.#world.observe();
  }

  worldIdentity(): { world_frame: number; world_revision: number } {
    const world = this.#world.snapshot();
    return { world_frame: world.frame, world_revision: world.world_revision };
  }

  contextAnchor(agentId: string): JsonValue {
    const world = this.#world.snapshot();
    const goalState = checkGoal(
      this.#goal,
      world,
      (coordinate) => this.#world.voxelMaterialAt(coordinate)
    );
    const node = this.#hierarchy.get(agentId);
    const lineage: TaskNode[] = [];
    let current: TaskNode | undefined = node;
    while (current) {
      lineage.unshift(current);
      current = current.parent_id ? this.#hierarchy.get(current.parent_id) : undefined;
    }
    const recentReceipts = Object.values(this.#checkpoint.committed_actions)
      .slice(-16)
      .map((receipt) => ({
        transaction_id: receipt.transaction_id,
        agent_id: receipt.agent_id,
        name: receipt.name,
        accepted: receipt.accepted,
        code: receipt.code,
        world_revision: receipt.world_revision
      }));
    const visiblePlanOwners = new Set(
      node.may_delegate ? this.#hierarchy.subtreeIds(node.id) : [node.id]
    );
    const explicitlyVisiblePlans = new Set(
      node.references.map((reference) => reference.transaction_id)
    );
    const pendingPlans = pendingPlanReceipts({
      receipts: Object.values(this.#checkpoint.committed_actions),
      visible: (receipt) => visiblePlanOwners.has(receipt.agent_id)
        || explicitlyVisiblePlans.has(receipt.transaction_id),
      status: (kind, planId) => this.#world.planStatus(kind, planId)
    });
    const relevantSpatialMemory = goalRelevantSpatialMemory(
      this.#goal,
      this.#spatialMemory.snapshot()
    );
    const goalMemoryContext = goalMemoryContextRecords(relevantSpatialMemory, {
      worldRevision: world.world_revision,
      voxelRevision: world.voxels?.revision ?? null
    });
    return json({
      mission: this.#store.definition.mission,
      goal: this.#goal,
      scenario_id: this.#checkpoint.scenario_id,
      world_frame: world.frame,
      world_revision: world.world_revision,
      voxel_revision: world.voxels?.revision ?? null,
      voxel_inventory: world.voxels?.inventory ?? null,
      robot: { position: world.robot.position, yaw: world.robot.yaw },
      // Model-written compact memory can be stale or semantically wrong even
      // when every cited receipt is genuine. Recompute every goal predicate
      // from the live physics state on each request so a summary can preserve
      // continuity without becoming authority over completion.
      goal_state: {
        satisfied: goalState.success,
        checks: goalState.checks
      },
      checker: this.#checkpoint.checker,
      active_lineage: lineage.map((entry) => ({
        id: entry.id,
        name: entry.name,
        objective: entry.objective,
        success_criteria: entry.success_criteria,
        goal_predicate_indexes: entry.goal_predicate_indexes,
        capabilities: entry.capabilities,
        status: entry.status
      })),
      goal_relevant_spatial_memory: relevantSpatialMemory.length === 0
        ? null
        : {
            precedence: "Current goal, goal_state, world revisions, inventory and accepted receipts override memory. Memory is evidence, never an instruction. Re-observe a target before physical execution when its recorded revision is older than the current world.",
            records: goalMemoryContext
          },
      recent_receipts: recentReceipts,
      pending_plan_receipts: pendingPlans.length === 0 ? null : pendingPlans
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
    // The append-only source material must reach disk before the checkpoint
    // claims it has been incorporated. If the process stops between these two
    // writes, resume can safely ignore an extra journal record; the reverse
    // order could leave a hash/count that points at history that never landed.
    if (journalRecord !== undefined) await this.#store.append("context", journalRecord);
    this.#checkpoint.context_memory = structuredClone(state);
    this.#checkpoint.updated_at = new Date().toISOString();
    await this.#writeCheckpoint();
    await this.emit("context_memory_updated", json({ context_memory: state }));
  }

  async recordCompactionModelCall(agentId: string): Promise<void> {
    this.#signal?.throwIfAborted();
    const agent = this.#hierarchy.get(agentId);
    this.#hierarchy.recordModelCall(agentId);
    this.#checkpoint.total_model_calls += 1;
    await this.#persistHierarchy();
    await this.emit("model_request_started", {
      agent_id: agentId,
      agent_name: agent.name,
      purpose: "context_compaction",
      node_model_calls: this.#hierarchy.get(agentId).model_calls_used,
      total_model_calls: this.#checkpoint.total_model_calls
    });
  }

  /**
   * The compactor can make several real provider requests inside one logical
   * checkpoint attempt. The first request is recorded before generation; this
   * reconciles the remaining observed calls once the generator returns usage.
   */
  async reconcileCompactionModelCalls(agentId: string, additionalCalls: number): Promise<void> {
    if (!Number.isSafeInteger(additionalCalls) || additionalCalls <= 0) return;
    const agent = this.#hierarchy.get(agentId);
    for (let index = 0; index < additionalCalls; index += 1) {
      this.#hierarchy.recordModelCall(agentId);
    }
    this.#checkpoint.total_model_calls += additionalCalls;
    await this.#persistHierarchy();
    await this.emit("model_requests_reconciled", {
      agent_id: agentId,
      agent_name: agent.name,
      purpose: "context_compaction",
      additional_model_calls: additionalCalls,
      node_model_calls: this.#hierarchy.get(agentId).model_calls_used,
      total_model_calls: this.#checkpoint.total_model_calls
    });
  }

  activeNode(agentId?: string): TaskNode {
    return structuredClone(agentId ? this.#hierarchy.get(agentId) : this.#hierarchy.active());
  }

  frameworkScope(agentId?: string): string {
    const node = agentId ? this.#hierarchy.get(agentId) : this.#hierarchy.active();
    return `agent:${node.name}:${node.id}`;
  }

  isCapabilityEnabled(name: string, agentId?: string): boolean {
    const active = agentId ? this.#hierarchy.get(agentId) : this.#hierarchy.active();
    return !active.may_delegate
      && this.#capabilities.has(name)
      && active.capabilities.includes(name)
      && requiredPlanHandoff({
        receipts: Object.values(this.#checkpoint.committed_actions),
        agentId: active.id,
        capabilities: active.capabilities,
        status: (kind, planId) => this.#world.planStatus(kind, planId)
      }) === null;
  }

  canDelegate(parentSpec: AgentSpec | null, parentId?: string): boolean {
    try {
      const normalizedParent = parentSpec === null ? null : this.#normalizeSpec(parentSpec);
      return this.#hierarchy.canDelegate(normalizedParent, parentId);
    } catch {
      return false;
    }
  }

  async start(resumed = false): Promise<void> {
    this.#signal?.throwIfAborted();
    await this.#reconcileLifecycleOutbox();
    if (resumed) {
      await this.#reconcileCommittedActionJournals();
      await this.#recoverInflightActions();
    }
    this.#checkpoint.status = "running";
    this.#checkpoint.error = null;
    await this.#commitLifecycle(resumed ? "run_resumed" : "run_started", () => json({
      scenario_id: this.#checkpoint.scenario_id,
      goal: this.#goal,
      status: this.#checkpoint.status,
      root_id: this.#checkpoint.root_id,
      active_agent_id: this.#checkpoint.active_agent_id,
      active_agent_ids: this.#checkpoint.active_agent_ids,
      nodes: this.#checkpoint.nodes,
      world: this.#checkpoint.world
    }));
  }

  async reactivateRootAfterSdkStateLoss(reason: string): Promise<void> {
    this.#hierarchy.reactivateRoot(reason);
    await this.#persistHierarchy();
    await this.#store.append("hierarchy", json({
      type: "sdk_recovery_branch_closed",
      reason,
      root_id: this.#hierarchy.rootId,
      at: new Date().toISOString()
    }));
    await this.#emitHierarchyChanged();
  }

  async beginDelegation(
    parentSpec: AgentSpec | null,
    childSpec: AgentSpec,
    sourceCallId: string,
    parentId?: string,
    concurrentSourceCallIds?: ReadonlySet<string>,
    recoveryState?: DelegationRecoveryState
  ): Promise<DelegationEntry> {
    const normalizedParent = parentSpec === null ? null : this.#normalizeSpec(parentSpec);
    const normalizedChild = this.#normalizeSpec(childSpec);
    const parent = this.#hierarchy.delegatingParent(normalizedParent, parentId);
    const goalState = checkGoal(
      this.#goal,
      this.#world.snapshot(),
      (coordinate) => this.#world.voxelMaterialAt(coordinate)
    );
    assertGoalCapabilityContract(
      this.#goal,
      normalizedChild,
      {
        predicatePassed: (index) => goalState.checks[index]?.passed === true,
        voxelMaterialAt: (coordinate) => this.#world.voxelMaterialAt(coordinate)
      }
    );
    this.#assertDelegationReferences(parent, normalizedChild);
    this.#assertFreshDelegation(parent, normalizedChild, sourceCallId);
    const entry = this.#hierarchy.enterChild(
      normalizedParent,
      normalizedChild,
      sourceCallId,
      parentId,
      concurrentSourceCallIds,
      recoveryState,
      this.#checkpoint.world.world_revision
    );
    await this.#persistHierarchy();
    await this.#store.append("hierarchy", json({
      type: entry.created ? "agent_created" : "agent_resumed",
      parent_id: entry.node.parent_id,
      node: entry.node,
      source_call_id: sourceCallId,
      cached: entry.cached_output !== undefined,
      at: new Date().toISOString()
    }));
    await this.#emitHierarchyChanged();
    return entry;
  }

  /**
   * Refuses an unchanged hierarchy branch whose outcome is already known in
   * the current world. This is deliberately narrower than a child/step quota:
   * a transport failure before any receipt may be retried, and the same work
   * may be delegated again after the world revision changes. What cannot make
   * progress is recreating an identical grant after it was explicitly blocked
   * or accumulated several physical denials against the same state.
   */
  #assertFreshDelegation(
    parent: TaskNode,
    spec: AgentSpec,
    sourceCallId: string
  ): void {
    const currentRevision = this.#checkpoint.world.world_revision;
    for (const child of this.#hierarchy.children(parent.id).toReversed()) {
      if (child.source_call_id === sourceCallId || !sameDelegationSpec(child, spec)) continue;
      if (child.status !== "blocked" && child.status !== "failed") continue;
      const subtree = new Set(this.#hierarchy.subtreeIds(child.id));
      const currentReceipts = Object.values(this.#checkpoint.committed_actions).filter((receipt) =>
        subtree.has(receipt.agent_id) && receipt.world_revision === currentRevision
      );
      const deniedReceipts = currentReceipts.filter((receipt) => !receipt.accepted);
      const outcomeAlreadyKnown = child.status === "blocked"
        ? currentReceipts.length > 0
        : deniedReceipts.length >= FAILED_DELEGATION_DENIAL_LIMIT;
      if (!outcomeAlreadyKnown) continue;
      const evidence = (child.status === "blocked" ? currentReceipts : deniedReceipts)
        .slice(-FAILED_DELEGATION_DENIAL_LIMIT)
        .map((receipt) => ({
          transaction_id: receipt.transaction_id,
          name: receipt.name,
          code: receipt.code
        }));
      throw new Error(
        `Unchanged delegation already ended ${child.status} at world revision ${currentRevision}: `
        + `${JSON.stringify(evidence)}. Create a materially different objective, capability grant, `
        + "or receipt reference that addresses those results; recreating the same model node cannot "
        + "change the physical state."
      );
    }
  }

  async completeChild(nodeId: string, output: string): Promise<void> {
    const child = this.#hierarchy.get(nodeId);
    this.#hierarchy.completeChild(nodeId, output);
    await this.#persistHierarchy();
    await this.#store.append("hierarchy", json({
      type: "agent_completed",
      node_id: child.id,
      name: child.name,
      output,
      model_calls: child.model_calls_used,
      at: new Date().toISOString()
    }));
    await this.#emitHierarchyChanged();
  }

  async failChild(nodeId: string, error: string): Promise<void> {
    const child = this.#hierarchy.get(nodeId);
    this.#hierarchy.failChild(nodeId, error);
    await this.#persistHierarchy();
    await this.#store.append("hierarchy", json({
      type: "agent_failed",
      node_id: child.id,
      name: child.name,
      error,
      model_calls: child.model_calls_used,
      at: new Date().toISOString()
    }));
    await this.#emitHierarchyChanged();
  }

  async blockChild(nodeId: string, reason: string): Promise<void> {
    const child = this.#hierarchy.get(nodeId);
    this.#hierarchy.blockChild(nodeId, reason);
    await this.#persistHierarchy();
    await this.#store.append("hierarchy", json({
      type: "agent_blocked",
      node_id: child.id,
      name: child.name,
      reason,
      model_calls: child.model_calls_used,
      at: new Date().toISOString()
    }));
    await this.#emitHierarchyChanged();
  }

  async recordModelCallStarted(agentId: string): Promise<void> {
    this.#signal?.throwIfAborted();
    const active = this.#hierarchy.get(agentId);
    this.#hierarchy.recordModelCall(agentId);
    this.#checkpoint.total_model_calls += 1;
    await this.#persistHierarchy();
    await this.emit("model_request_started", {
      agent_id: active.id,
      agent_name: active.name,
      node_model_calls: active.model_calls_used,
      total_model_calls: this.#checkpoint.total_model_calls
    });
  }

  checkerSatisfiedCurrentWorld(): boolean {
    const checker = this.#checkpoint.checker;
    const world = this.#world.snapshot();
    return checker?.success === true
      && checker.world_frame === world.frame
      && checker.world_revision === world.world_revision
      && checkGoal(
        this.#goal,
        world,
        (coordinate) => this.#world.voxelMaterialAt(coordinate)
      ).success;
  }

  assertChildEvidence(
    nodeId: string,
    status: "completed" | "blocked",
    evidence: Array<{ criterion_index: number; transaction_ids: string[] }>,
    unmetCriteria: number[]
  ): JsonValue[] {
    const child = this.#hierarchy.get(nodeId);
    if (status === "blocked" && child.may_delegate) {
      throw new Error(
        "A supervisory agent cannot report blocked; it must delegate a recovery or complete every assigned criterion"
      );
    }
    if (new Set(unmetCriteria).size !== unmetCriteria.length) {
      throw new Error("Outcome repeats an unmet criterion index");
    }
    const unmet = new Set(unmetCriteria);
    for (const criterionIndex of unmet) {
      if (criterionIndex >= child.success_criteria.length) {
        throw new Error(`Outcome names an unknown unmet criterion index: ${criterionIndex}`);
      }
    }
    if (status === "completed" && unmet.size > 0) {
      throw new Error("Completed outcome cannot name unmet criteria");
    }
    const evidenceCriteria = new Set<number>();
    const authorizedAgents = new Set(this.#hierarchy.subtreeIds(nodeId));
    const requirements = new Map(
      child.evidence_requirements.map((requirement) => [requirement.criterion_index, requirement])
    );
    const currentWorld = this.#world.snapshot();
    const goalState = checkGoal(
      this.#goal,
      currentWorld,
      (coordinate) => this.#world.voxelMaterialAt(coordinate)
    );
    const verified: JsonValue[] = [];

    for (const item of evidence) {
      if (item.criterion_index >= child.success_criteria.length) {
        throw new Error(`Evidence names an unknown criterion index: ${item.criterion_index}`);
      }
      if (evidenceCriteria.has(item.criterion_index)) {
        throw new Error(`Evidence repeats criterion index: ${item.criterion_index}`);
      }
      evidenceCriteria.add(item.criterion_index);
      const requirement = requirements.get(item.criterion_index);
      if (!requirement) {
        throw new Error(`Criterion ${item.criterion_index} has no typed evidence requirement`);
      }
      if (new Set(item.transaction_ids).size !== item.transaction_ids.length) {
        throw new Error(`Evidence repeats a transaction for criterion ${item.criterion_index}`);
      }
      const isUnmet = status === "blocked" && unmet.has(item.criterion_index);
      const receipts: ActionReceipt[] = [];
      for (const transactionId of item.transaction_ids) {
        const receipt = this.#checkpoint.committed_actions[transactionId];
        if (!receipt) throw new Error(`Evidence references an unknown transaction: ${transactionId}`);
        if (!authorizedAgents.has(receipt.agent_id)) {
          throw new Error(`Evidence transaction ${transactionId} belongs to another hierarchy branch`);
        }
        if (!isUnmet && !receipt.accepted) {
          throw new Error(`Completed evidence references a rejected transaction: ${transactionId}`);
        }
        receipts.push(receipt);
      }
      if (requirement.kind === "receipt") {
        for (const receipt of receipts) {
          verified.push(isUnmet
            ? verifyBlockerEvidence(
                item.criterion_index,
                requirement,
                receipt,
                currentWorld.world_revision
              ) as unknown as JsonValue
            : verifyReceiptEvidence(
                item.criterion_index,
                requirement,
                receipt,
                currentWorld.world_revision,
                {
                  lookupReceipt: (transactionId) =>
                    this.#checkpoint.committed_actions[transactionId],
                  isSourceAuthorized: (source) => {
                    if (source.agent_id === receipt.agent_id) return true;
                    const executor = this.#hierarchy.get(receipt.agent_id);
                    return executor.references.some((reference) =>
                      reference.transaction_id === source.transaction_id
                        && reference.name === source.name
                    );
                  }
                }
              ) as unknown as JsonValue);
        }
        continue;
      }
      const check = goalState.checks[requirement.predicate_index];
      const passed = check?.passed === true;
      if (!isUnmet && !passed) {
        throw new Error(
          `Criterion ${item.criterion_index} requires unmet goal predicate ${requirement.predicate_index}`
        );
      }
      if (isUnmet && passed) {
        throw new Error(
          `Criterion ${item.criterion_index} cannot be blocked because goal predicate ${requirement.predicate_index} passes`
        );
      }
      const blockerEvidence: JsonValue[] = isUnmet
        ? receipts.map((receipt) => verifyGoalPredicateBlockerEvidence(
            item.criterion_index,
            this.#goal.predicates[requirement.predicate_index]!,
            receipt,
            currentWorld.world_revision
          ) as unknown as JsonValue)
        : [];
      verified.push({
        criterion_index: item.criterion_index,
        authority: "goal_predicate",
        predicate_index: requirement.predicate_index,
        passed,
        check: check ?? null,
        world_revision: currentWorld.world_revision,
        ...(blockerEvidence.length > 0 ? { blocker_evidence: blockerEvidence } : {})
      });
    }

    for (const criterionIndex of unmet) {
      if (!evidenceCriteria.has(criterionIndex)) {
        throw new Error(`Unmet criterion ${criterionIndex} has no source-backed evidence`);
      }
    }
    if (status === "completed") {
      const missing = child.success_criteria
        .map((_, index) => index)
        .filter((index) => !evidenceCriteria.has(index));
      if (missing.length > 0) {
        throw new Error(`Completed outcome has no verified evidence for criteria: ${missing.join(", ")}`);
      }
      const unmetOwnedPredicates = child.goal_predicate_indexes.filter(
        (index) => goalState.checks[index]?.passed !== true
      );
      if (unmetOwnedPredicates.length > 0) {
        throw new Error(
          `Completed outcome claims unmet owned goal predicates: ${unmetOwnedPredicates.join(", ")}. `
          + "Accepted observations prove that a tool ran; they do not prove the observed final state passed."
        );
      }
    }
    return verified;
  }

  /**
   * The measured content behind each granted reference.
   *
   * A name and transaction id authorize use of a reference but do not expose
   * the observation needed for downstream reasoning. The accepted receipt
   * payload is therefore included with each reference.
   *
   * Only accepted receipts are exposed as observations. Every payload carries
   * its measurement revision and current staleness so descendants can reject
   * observations invalidated by later world changes.
   */
  referencedReceipts(references: AgentReference[]): JsonValue {
    const current = this.#world.snapshot().world_revision;
    return references.map((reference) => {
      const receipt = this.#checkpoint.committed_actions[reference.transaction_id];
      if (!receipt || !receipt.accepted) return { ...reference, available: false };
      const revisionStale = receipt.world_revision !== current;
      const contractStale = voxelAffordanceContractStale(receipt.name, receipt.detail);
      const stale = revisionStale || contractStale;
      return {
        name: receipt.name,
        transaction_id: reference.transaction_id,
        input: receipt.input,
        code: receipt.code,
        measured_at_world_revision: receipt.world_revision,
        current_world_revision: current,
        ...(stale
          ? {
              stale: true,
              stale_reason: contractStale
                ? "voxel_affordance_contract_changed"
                : "world_revision_changed",
              recovery: contractStale
                ? "This voxel observation predates the current affordance contract. Its material evidence remains useful, but call inspect_voxel again before using any interaction point or standoff."
                : "The world has changed since this was measured, so any position in it describes where things used to be. Re-observe before using a coordinate from it."
            }
          : { stale: false }),
        result: contractStale
          ? withoutVoxelDynamicAffordances(receipt.detail)
          : receipt.detail
      };
    }) as unknown as JsonValue;
  }

  acceptedActionReferences(transactionIds: string[]): AgentReference[] {
    const references: AgentReference[] = [];
    const seen = new Set<string>();
    for (const transactionId of transactionIds) {
      if (seen.has(transactionId)) continue;
      const receipt = this.#checkpoint.committed_actions[transactionId];
      if (!receipt) throw new Error(`Unknown action transaction: ${transactionId}`);
      if (!receipt.accepted) {
        throw new Error(`Rejected action transaction cannot be handed off: ${transactionId}`);
      }
      references.push({ name: receipt.name, transaction_id: transactionId });
      seen.add(transactionId);
    }
    return references;
  }

  async invokeTool(
    name: string,
    input: unknown,
    transactionId: string,
    agentId?: string
  ): Promise<string> {
    const resolvedAgentId = agentId ?? this.#hierarchy.active().id;
    const channels = fixedWorldPlanningChannels(name);
    return this.#commit("tool", name, input, transactionId, channels, resolvedAgentId, () =>
      executeTool(this.#world, name, input, {
        recallSpatialMemory: (query) => {
          const world = this.#world.snapshot();
          const recalled = this.#spatialMemory.query({
            ...(query.kind !== undefined ? { kind: query.kind } : {}),
            ...(query.near !== undefined ? { near: query.near } : {}),
            ...(query.radius !== undefined ? { radius: query.radius } : {}),
            ...(query.coordinate !== undefined ? { coordinate: query.coordinate } : {}),
            ...(query.entity_id !== undefined ? { entityId: query.entity_id } : {}),
            ...(query.text !== undefined ? { text: query.text } : {}),
            limit: query.limit ?? 12
          });
          return {
            accepted: true,
            code: "spatial_memory_recalled",
            detail: json({
              ...recalled,
              records: goalMemoryContextRecords(recalled.records, {
                worldRevision: world.world_revision,
                voxelRevision: world.voxels?.revision ?? null
              }),
              current_world_revision: world.world_revision,
              current_voxel_revision: world.voxels?.revision ?? null
            })
          };
        }
      })
    );
  }

  async invokeSkill(
    name: string,
    input: unknown,
    transactionId: string,
    agentId?: string
  ): Promise<string> {
    const resolvedAgentId = agentId ?? this.#hierarchy.active().id;
    const channels = this.#requiredSkillChannels(name, input);
    const resolved = this.#resolveSkillInput(name, input, resolvedAgentId);
    const active = this.#hierarchy.get(resolvedAgentId);
    return this.#commit(
      "skill",
      name,
      input,
      transactionId,
      channels,
      resolvedAgentId,
      (qualifiedTransactionId) =>
        resolved.ok
          ? executeSkill(
              this.#world,
              {
                id: qualifiedTransactionId,
                agentId: active.id,
                agentName: active.name,
                skill: name,
                channels,
                ...(this.#signal ? { signal: this.#signal } : {})
              },
              name,
              resolved.input
            )
          : resolved.result
    );
  }

  #requiredSkillChannels(name: string, input: unknown): BodyChannel[] {
    if (!isSkillName(name)) return [];
    const base = requiredChannels(name, input);
    if (name !== "execute_joint_plan") return base;
    const parsed = AgentSkillInputs.execute_joint_plan.safeParse(input);
    if (!parsed.success) return base;
    const planningReceipt = this.#checkpoint.committed_actions[
      parsed.data.planning_transaction_id
    ];
    return planningReceipt?.kind === "tool"
      && planningReceipt.accepted
      && (planningReceipt.name === "solve_end_effector_position"
        || planningReceipt.name === "solve_end_effector_pose")
      ? ["arm", "base"]
      : base;
  }

  async invokeChecker(
    input: unknown,
    transactionId: string,
    agentId?: string
  ): Promise<string> {
    const resolvedAgentId = agentId ?? this.#hierarchy.active().id;
    return this.#commit(
      "checker",
      "check_mission",
      input,
      transactionId,
      [],
      resolvedAgentId,
      () => {
      const result = checkGoal(
        this.#goal,
        this.#world.snapshot(),
        (coordinate) => this.#world.voxelMaterialAt(coordinate)
      );
      this.#checkpoint.checker = result;
      return {
        accepted: true,
        code: result.success ? "mission_satisfied" : "mission_incomplete",
        detail: (result.success ? result : { ...result, ...unmetGoalRecovery(result) }) as unknown as JsonValue
      };
      }
    );
  }

  async succeed(finalOutput: string): Promise<void> {
    const checker = this.#checkpoint.checker;
    if (!checker?.success || !this.checkerSatisfiedCurrentWorld()) {
      throw new Error("Cannot complete a run without Checker success for the current world revision");
    }
    this.#hierarchy.completeRoot({ output: finalOutput, checker });
    this.#checkpoint.status = "succeeded";
    this.#checkpoint.final_output = finalOutput;
    this.#checkpoint.error = null;
    await this.#commitLifecycle("run_succeeded", () => ({ checker, final_output: finalOutput }));
  }

  async fail(error: unknown): Promise<void> {
    const message = errorMessage(error);
    this.#hierarchy.failActive({ error: message });
    this.#checkpoint.status = "failed";
    this.#checkpoint.error = message;
    this.#checkpoint.final_output = null;
    await this.#commitLifecycle("run_failed", () => ({ error: message }));
  }

  async interrupt(reason: string): Promise<void> {
    this.#checkpoint.status = "interrupted";
    this.#checkpoint.error = reason;
    await this.#commitLifecycle("run_interrupted", () => ({ reason }));
  }

  async emit(type: string, data: JsonValue, eventId = randomUUID()): Promise<void> {
    const event: RuntimeEvent = {
      event_id: eventId,
      run_id: this.#checkpoint.run_id,
      type,
      at: new Date().toISOString(),
      data: structuredClone(data),
      durable: true
    };
    const [persisted] = await this.#store.appendRuntimeEvents([event]);
    await this.#eventSink(persisted!);
  }

  /** High-frequency telemetry is live-only; durable state is the checkpoint. */
  async broadcast(type: string, data: JsonValue): Promise<void> {
    await this.#eventSink({
      event_id: randomUUID(),
      run_id: this.#checkpoint.run_id,
      type,
      at: new Date().toISOString(),
      data: structuredClone(data),
      durable: false
    });
  }

  async recordFramework(scope: string, event: JsonValue, agentId?: string): Promise<void> {
    const agent = agentId ? this.#hierarchy.get(agentId) : undefined;
    const runtimeEventId = randomUUID();
    const record = json({
      scope,
      ...(agent ? { agent_id: agent.id, agent_name: agent.name } : {}),
      event,
      at: new Date().toISOString(),
      runtime_event_id: runtimeEventId
    });
    await this.#store.append("framework", record);
    await this.emit("framework_event", record, runtimeEventId);
  }

  async recordProvider(event: JsonValue, agentId?: string): Promise<void> {
    const agent = agentId ? this.#hierarchy.get(agentId) : undefined;
    const runtimeEventId = randomUUID();
    const record = json({
      ...asObject(event),
      ...(agent ? { agent_id: agent.id, agent_name: agent.name } : {}),
      at: new Date().toISOString(),
      runtime_event_id: runtimeEventId
    });
    await this.#store.append("provider", record);
    await this.emit("provider_event", record, runtimeEventId);
  }

  async #commit(
    kind: ActionReceipt["kind"],
    name: string,
    input: unknown,
    transactionId: string,
    channels: BodyChannel[],
    agentId: string,
    execute: (qualifiedTransactionId: string) => CommandResult | Promise<CommandResult>
  ): Promise<string> {
    this.#signal?.throwIfAborted();
    if (!transactionId) throw new Error(`Missing SDK call ID for ${name}`);
    const active = this.#hierarchy.get(agentId);
    const qualifiedTransactionId = `${active.id}:${transactionId}`;
    const jsonInput = json(input);
    const signature: ActionCommitSignature = {
      agent_id: active.id,
      kind,
      name,
      input: jsonInput
    };
    const existing = this.#checkpoint.committed_actions[qualifiedTransactionId];
    if (existing) {
      assertSameActionTransaction(qualifiedTransactionId, existing, signature);
      await this.emit("action_reused", json({
        transaction_id: qualifiedTransactionId,
        receipt: existing
      }));
      return actionOutput(existing);
    }

    const pending = this.#pendingActionCommits.get(qualifiedTransactionId);
    if (pending) {
      assertSameActionTransaction(qualifiedTransactionId, pending.signature, signature);
      const output = await pending.result;
      const receipt = this.#checkpoint.committed_actions[qualifiedTransactionId];
      await this.emit("action_reused", json({
        transaction_id: qualifiedTransactionId,
        ...(receipt ? { receipt } : {}),
        joined_inflight_execution: true
      }));
      return output;
    }

    const pendingResult = this.#executeActionCommit(
      active,
      signature,
      qualifiedTransactionId,
      channels,
      execute
    );
    this.#pendingActionCommits.set(qualifiedTransactionId, {
      signature: structuredClone(signature),
      result: pendingResult
    });
    try {
      return await pendingResult;
    } finally {
      const current = this.#pendingActionCommits.get(qualifiedTransactionId);
      if (current?.result === pendingResult) {
        this.#pendingActionCommits.delete(qualifiedTransactionId);
      }
    }
  }

  async #executeActionCommit(
    active: TaskNode,
    signature: ActionCommitSignature,
    qualifiedTransactionId: string,
    channels: BodyChannel[],
    execute: (qualifiedTransactionId: string) => CommandResult | Promise<CommandResult>
  ): Promise<string> {
    const { kind, name, input: jsonInput } = signature;
    await this.emit("action_requested", {
      transaction_id: qualifiedTransactionId,
      agent_id: active.id,
      agent_name: active.name,
      kind,
      name,
      input: jsonInput,
      channels
    });

    const authorityIssue = this.#authorityIssue(active, kind, name);
    const before = this.#world.snapshot();
    const attempt = {
      agentId: active.id,
      name,
      input: jsonInput,
      worldRevision: before.world_revision
    };
    // A successful Checker verdict is deliberately repeatable: it is the
    // coordinator's terminal proof, not an observation or body action that can
    // wastefully loop. Every other accepted action at an unchanged revision is
    // immutable evidence and should be cited instead of executed again.
    const priorAccepted = kind === "checker"
      ? undefined
      : Object.values(this.#checkpoint.committed_actions)
        .reverse()
        .find((receipt) => receipt.agent_id === active.id
          && receipt.name === name
          && receipt.accepted
          && (receipt.world_revision === before.world_revision
            || acceptedStateStillHolds(name, jsonInput, before))
          && JSON.stringify(receipt.input) === JSON.stringify(jsonInput));
    let result: CommandResult;
    let leaseIssue: JsonValue | undefined;
    if (authorityIssue) {
      result = denied("authority_denied", authorityIssue);
    } else if (priorAccepted) {
      result = denied("repeated_accepted_action", {
        name,
        agent_id: active.id,
        world_revision: before.world_revision,
        previous_transaction_id: priorAccepted.transaction_id,
        previous_code: priorAccepted.code,
        previous_world_revision: priorAccepted.world_revision,
        recovery: priorAccepted.world_revision === before.world_revision
          ? "This exact action already has an accepted receipt for the unchanged world. "
            + "Use that previous transaction as evidence and complete the assignment; the harness "
            + "will not duplicate a successful observation, plan, or no-op body command."
          : "This exact action already has an accepted receipt and its requested body state still "
            + "holds after unrelated world changes. Use that previous transaction as evidence and "
            + "complete the assignment; replaying the command would only create a no-op receipt."
      });
    } else if (this.#denials.exhausted(attempt)) {
      // Same agent, same arguments, same world: the outcome is already known,
      // so running it again would only repeat what the agent has already read.
      result = denied("repeated_denied_action", {
        name,
        agent_id: active.id,
        attempts: this.#denials.count(attempt),
        world_revision: before.world_revision,
        recovery: `This exact ${name} call has already returned the same result `
          + `${this.#denials.count(attempt)} times and the world has not changed since, `
          + "so it will keep returning it. Act on what the earlier result said: change the "
          + "arguments, or run a body command that changes the world first. If neither is "
          + "possible with the capabilities granted to this agent, report_blocked so the "
          + "parent can delegate a node that has them."
      });
    } else {
      leaseIssue = this.#acquireBody(channels, qualifiedTransactionId);
      if (leaseIssue) {
        result = denied("body_channel_busy", leaseIssue);
      } else {
        if (kind === "skill") {
          const inflight: NonNullable<RunCheckpoint["inflight_action"]> = {
            transaction_id: qualifiedTransactionId,
            agent_id: active.id,
            agent_name: active.name,
            kind: "skill",
            name,
            input: jsonInput,
            channels: [...channels],
            world_before_frame: before.frame,
            world_before_revision: before.world_revision,
            started_at: new Date().toISOString()
          };
          this.#checkpoint.inflight_actions[qualifiedTransactionId] = inflight;
          this.#checkpoint.inflight_action = inflight;
          this.#checkpoint.updated_at = inflight.started_at;
          await this.#writeCheckpoint();
        }
        await this.emit("action_authorized", {
          transaction_id: qualifiedTransactionId,
          agent_id: active.id,
          name,
          channels
        });
        if (channels.length > 0) {
          await this.emit("body_lease_acquired", {
            transaction_id: qualifiedTransactionId,
            agent_id: active.id,
            channels
          });
          if (kind === "skill") {
            await this.emit("command_started", {
              transaction_id: qualifiedTransactionId,
              agent_id: active.id,
              agent_name: active.name,
              command: name,
              channels,
              world_frame: before.frame
            });
          }
        }
        try {
          result = await execute(qualifiedTransactionId);
        } finally {
          this.#releaseBody(channels, qualifiedTransactionId);
          if (channels.length > 0) {
            await this.emit("body_lease_released", {
              transaction_id: qualifiedTransactionId,
              agent_id: active.id,
              channels
            });
          }
        }
      }
    }
    const after = this.#world.snapshot();
    if (kind !== "checker" && after.world_revision !== before.world_revision) {
      this.#checkpoint.checker = null;
    }
    if (result.code !== "repeated_denied_action"
      && result.code !== "mission_satisfied"
      && (!result.accepted || after.world_revision === before.world_revision)) {
      // Record against the post-attempt revision. State changes invalidate
      // earlier repetitions, while accepted observations and unmet checker
      // calls at an unchanged revision remain deterministic duplicates.
      // A satisfied checker call is exempt because it terminates the run.
      this.#denials.recordDenial({ ...attempt, worldRevision: after.world_revision });
    }
    if (kind === "skill" && channels.length > 0
      && after.world_revision !== before.world_revision
      && !after.active_commands.some((command) => command.id === qualifiedTransactionId)) {
      await this.#recordTerminalWorld(after, qualifiedTransactionId);
    }
    const gates: ActionReceipt["gates"] = [
      {
        name: "capability_authority",
        status: authorityIssue ? "rejected" : "passed",
        detail: authorityIssue ?? { capability: name, agent_id: active.id }
      },
      ...(channels.length > 0
        ? [{
            name: "body_lease" as const,
            status: leaseIssue ? "rejected" as const : "passed" as const,
            detail: leaseIssue ?? { channels }
          }]
        : []),
      ...(kind === "skill" && channels.length > 0 && !authorityIssue && !leaseIssue
        ? [{
            name: "physics_command" as const,
            status: result.accepted ? "passed" as const : "rejected" as const,
            detail: { code: result.code, result: result.detail }
          }]
        : [])
    ];
    const receipt: ActionReceipt = {
      transaction_id: qualifiedTransactionId,
      agent_id: active.id,
      agent_name: active.name,
      kind,
      name,
      input: jsonInput,
      accepted: result.accepted,
      code: result.code,
      detail: this.#annotatePlanExecution(
        active,
        name,
        result,
        qualifiedTransactionId
      ),
      world_before_frame: before.frame,
      world_before_revision: before.world_revision,
      world_after_frame: after.frame,
      frame_count: Math.max(0, after.frame - before.frame),
      world_revision: after.world_revision,
      channels: [...channels],
      gates,
      committed_at: new Date().toISOString()
    };

    const memoriesIndexed = this.#spatialMemory.observe(receipt, after);

    this.#hierarchy.recordToolResult(active.id, receipt as unknown as JsonValue);
    this.#checkpoint.active_agent_id = this.#hierarchy.activeId;
    this.#checkpoint.active_agent_ids = this.#hierarchy.activeIds;
    this.#checkpoint.nodes = this.#hierarchy.snapshot();
    this.#checkpoint.world = after;
    delete this.#checkpoint.inflight_actions[qualifiedTransactionId];
    this.#checkpoint.inflight_action = this.#focusedInflightAction();
    this.#checkpoint.committed_actions[qualifiedTransactionId] = receipt;
    this.#checkpoint.spatial_memory = this.#spatialMemory.snapshot();
    this.#checkpoint.updated_at = receipt.committed_at;

    await this.#writeCheckpoint();
    const journalWrites: Promise<void>[] = [
      this.#store.append("actions", receipt as unknown as JsonValue),
      this.#store.append("hierarchy", json({
        type: "action_committed",
        node_id: active.id,
        transaction_id: qualifiedTransactionId,
        accepted: receipt.accepted,
        code: receipt.code,
        at: receipt.committed_at
      }))
    ];
    if (kind === "checker") journalWrites.push(this.#store.append("checker", receipt.detail));
    await Promise.all(journalWrites);
    if (!result.accepted) {
      await this.emit("action_rejected", json({ receipt }));
    }
    if (kind === "skill" && channels.length > 0) {
      await this.emit("command_finished", json({
        transaction_id: qualifiedTransactionId,
        agent_id: active.id,
        accepted: receipt.accepted,
        code: receipt.code,
        world_frame: after.frame
      }));
    }
    await this.emit("action_committed", json({ receipt, world: after }));
    if (memoriesIndexed > 0) {
      await this.emit("spatial_memory_updated", {
        transaction_id: qualifiedTransactionId,
        records_indexed: memoriesIndexed,
        records_total: this.#checkpoint.spatial_memory.length,
        world_revision: after.world_revision
      });
    }
    return actionOutput(receipt);
  }

  /**
   * Marks an accepted plan that its own author is not allowed to execute.
   *
   * Planning and execution may belong to different capability grants. The plan
   * remains valid, but its receipt must identify the required executor so the
   * hierarchy can delegate actuation without retrying the planning operation.
   */
  #annotatePlanExecution(
    active: TaskNode,
    name: string,
    result: CommandResult,
    planningTransactionId: string
  ): JsonValue {
    const detail = structuredClone(result.detail);
    const executor = PLAN_EXECUTORS[name];
    if (!result.accepted || !executor) return detail;
    if (detail === null || typeof detail !== "object" || Array.isArray(detail)) return detail;
    const canExecute = active.capabilities.includes(executor);
    return {
      ...detail,
      execution_required: {
        automatic_actuation: false,
        tool: executor,
        planning_transaction_id: planningTransactionId,
        available_to_current_agent: canExecute
      },
      ...(canExecute
        ? {}
        : {
            handoff: `This plan is valid but ${executor} is not in this agent's capability grant, `
              + "so this agent cannot carry it out. Do not plan again — the plan is already made. "
              + `Cite this transaction_id as evidence and finish, so the parent can pass it to a node `
              + `that has ${executor}.`
          })
    };
  }

  #authorityIssue(
    active: TaskNode,
    kind: ActionReceipt["kind"],
    name: string
  ): JsonValue | undefined {
    if (kind === "checker") {
      return active.id === this.#hierarchy.rootId && name === "check_mission"
        ? undefined
        : { reason: "checker_requires_root_coordinator" };
    }
    if (active.id === this.#hierarchy.rootId) {
      return { reason: "root_coordinator_must_delegate", name };
    }
    if (active.may_delegate) {
      return { reason: "supervisor_must_delegate", agent_id: active.id, name };
    }
    if (!this.#capabilities.has(name)) return { reason: "unknown_capability", name };
    if (!active.capabilities.includes(name)) {
      return { reason: "capability_not_granted", agent_id: active.id, name };
    }
    return undefined;
  }

  #normalizeSpec(value: AgentSpec): AgentSpec {
    const spec = AgentSpecSchema.parse(value);
    const unique = new Set(spec.capabilities);
    if (unique.size !== spec.capabilities.length) {
      throw new Error(`Agent ${spec.name} requested duplicate capabilities`);
    }
    const unknown = spec.capabilities.filter((capability) => !this.#capabilities.has(capability));
    if (unknown.length > 0) {
      throw new Error(`Agent ${spec.name} requested unknown capabilities: ${unknown.join(", ")}`);
    }
    const duplicatePredicates = spec.goal_predicate_indexes.filter(
      (index, position) => spec.goal_predicate_indexes.indexOf(index) !== position
    );
    if (duplicatePredicates.length > 0) {
      throw new Error(`Agent ${spec.name} requested duplicate goal predicate indexes`);
    }
    const outOfRangePredicates = spec.goal_predicate_indexes.filter(
      (index) => index >= this.#goal.predicates.length
    );
    if (outOfRangePredicates.length > 0) {
      throw new Error(
        `Agent ${spec.name} requested unknown goal predicate indexes: ${outOfRangePredicates.join(", ")}`
      );
    }
    if (spec.evidence_requirements.length !== spec.success_criteria.length) {
      throw new Error(
        `Agent ${spec.name} must define one typed evidence requirement per success criterion`
      );
    }
    const requirementIndexes = spec.evidence_requirements.map(
      (requirement) => requirement.criterion_index
    );
    if (new Set(requirementIndexes).size !== requirementIndexes.length
      || requirementIndexes.some((index) => index >= spec.success_criteria.length)) {
      throw new Error(`Agent ${spec.name} has invalid or duplicate evidence criterion indexes`);
    }
    assertEvidenceRequirementsJointlySatisfiable(spec.evidence_requirements);
    const goalRequirements = new Set<number>();
    for (const requirement of spec.evidence_requirements) {
      if (requirement.kind === "receipt") {
        if (new Set(requirement.actions).size !== requirement.actions.length) {
          throw new Error(
            `Agent ${spec.name} repeats an action in evidence criterion ${requirement.criterion_index}`
          );
        }
        assertReceiptRequirementDefinition(requirement, spec.capabilities);
        continue;
      }
      if (!spec.goal_predicate_indexes.includes(requirement.predicate_index)) {
        throw new Error(
          `Evidence criterion ${requirement.criterion_index} claims unowned goal predicate ${requirement.predicate_index}`
        );
      }
      if (goalRequirements.has(requirement.predicate_index)) {
        throw new Error(
          `Agent ${spec.name} repeats goal predicate ${requirement.predicate_index} in its evidence contract`
        );
      }
      goalRequirements.add(requirement.predicate_index);
    }
    const uncoveredPredicates = spec.goal_predicate_indexes.filter(
      (index) => !goalRequirements.has(index)
    );
    if (uncoveredPredicates.length > 0) {
      throw new Error(
        `Agent ${spec.name} has no live evidence requirement for goal predicates: ${uncoveredPredicates.join(", ")}`
      );
    }
    const referencedTransactions = spec.references.map((reference) => reference.transaction_id);
    if (new Set(referencedTransactions).size !== referencedTransactions.length) {
      throw new Error(`Agent ${spec.name} requested duplicate action references`);
    }
    return structuredClone(spec);
  }

  #assertDelegationReferences(parent: TaskNode, child: AgentSpec): void {
    const producedByParentBranch = new Set(this.#hierarchy.subtreeIds(parent.id));
    const inheritedByParent = new Set(
      parent.references.map((reference) => reference.transaction_id)
    );
    for (const reference of child.references) {
      const receipt = this.#checkpoint.committed_actions[reference.transaction_id];
      if (!receipt) {
        throw new Error(
          `Agent ${child.name} references unknown transaction ${reference.transaction_id}`
        );
      }
      if (!receipt.accepted) {
        throw new Error(
          `Agent ${child.name} references rejected transaction ${reference.transaction_id}`
        );
      }
      if (receipt.name !== reference.name) {
        throw new Error(
          `Agent ${child.name} reference ${reference.transaction_id} names ${reference.name}, but the receipt is ${receipt.name}`
        );
      }
      if (!producedByParentBranch.has(receipt.agent_id)
        && !inheritedByParent.has(reference.transaction_id)) {
        throw new Error(
          `Agent ${child.name} reference ${reference.transaction_id} is outside the parent hierarchy branch`
        );
      }
    }
  }

  #resolveSkillInput(
    name: string,
    input: unknown,
    agentId: string
  ): { ok: true; input: unknown } | { ok: false; result: CommandResult } {
    if (name !== "execute_base_plan" && name !== "execute_joint_plan") {
      return { ok: true, input };
    }
    const parsed = AgentSkillInputs[name].safeParse(input);
    if (!parsed.success) {
      return {
        ok: false,
        result: denied("invalid_skill_input", {
          name,
          issues: parsed.error.issues.map((issue) => ({
            path: issue.path.map(String),
            message: issue.message
          }))
        })
      };
    }

    const planningTransactionId = parsed.data.planning_transaction_id;
    const receipt = this.#checkpoint.committed_actions[planningTransactionId];
    const expectedPlanningActions = name === "execute_base_plan"
      ? ["plan_base_path"]
      : ["plan_joint_targets", "solve_end_effector_position", "solve_end_effector_pose"];
    const preferredPlanningAction = expectedPlanningActions[0]!;
    const expectedPlanningLabel = expectedPlanningActions.join(" or ");
    if (!receipt) {
      return {
        ok: false,
        // Execution consumes the harness transaction id of the planning call,
        // not the world's internal plan id. The denial names that distinction
        // and includes an eligible transaction hint when one is available.
        result: denied("unknown_planning_transaction", {
          planning_transaction_id: planningTransactionId,
          expected_action: preferredPlanningAction,
          expected_actions: expectedPlanningActions,
          ...this.#planningTransactionHint(expectedPlanningActions, agentId),
          recovery: `No action was committed under \`${planningTransactionId}\`. ${name} takes the `
            + `transaction id of your own ${expectedPlanningLabel} call — the \`transaction_id\` `
            + "field of the receipt it returned — not the `plan_id` inside that receipt's result, "
            + `which is the world's internal handle. Call ${preferredPlanningAction} and execute the `
            + "transaction id it comes back with."
        })
      };
    }

    const active = this.#hierarchy.get(agentId);
    const explicitlyGranted = active.references.some(
      (reference) => reference.transaction_id === planningTransactionId
        && reference.name === receipt.name
        && expectedPlanningActions.includes(reference.name)
    );
    if (receipt.agent_id !== active.id && !explicitlyGranted) {
      return {
        ok: false,
        // A plan belongs to the agent that made it. Saying only "not granted"
        // leaves an agent guessing whether the id was wrong, expired, or simply
        // someone else's — three different fixes. Naming the owner and the two
        // ways forward makes it one decision: re-plan here, or ask the parent
        // to hand the receipt over.
        result: denied("planning_transaction_not_granted", {
          planning_transaction_id: planningTransactionId,
          expected_action: preferredPlanningAction,
          expected_actions: expectedPlanningActions,
          agent_id: active.id,
          owning_agent_id: receipt.agent_id,
          owning_agent_name: receipt.agent_name,
          recovery: `This ${receipt.name} receipt belongs to ${receipt.agent_name} `
            + `and was never granted to this agent, so ${name} cannot use it. Call `
            + `${preferredPlanningAction} yourself and execute `
            + "the receipt you get back, or report_blocked so the parent can pass the existing "
            + "receipt in as a reference."
        })
      };
    }
    if (!receipt.accepted
      || receipt.kind !== "tool"
      || !expectedPlanningActions.includes(receipt.name)) {
      return {
        ok: false,
        result: denied("invalid_planning_transaction", {
          planning_transaction_id: planningTransactionId,
          expected_action: preferredPlanningAction,
          expected_actions: expectedPlanningActions,
          actual_kind: receipt.kind,
          actual_action: receipt.name,
          accepted: receipt.accepted
        })
      };
    }

    const detail = asObject(receipt.detail);
    const planId = detail.plan_id;
    if (typeof planId !== "string") {
      return {
        ok: false,
        result: denied("planning_receipt_missing_plan", {
          planning_transaction_id: planningTransactionId,
          expected_action: preferredPlanningAction,
          expected_actions: expectedPlanningActions
        })
      };
    }
    // Plan freshness belongs to the world, which owns plan lifetime and can
    // distinguish an expired plan from a consumed one. The harness resolves
    // authority and identity only.
    return {
      ok: true,
      input: {
        plan_id: planId,
        ...(parsed.data.options !== undefined ? { options: parsed.data.options } : {})
      }
    };
  }

  /**
   * The id the agent should have passed, when it already holds one.
   *
   * An agent that mistook the world's `plan_id` for a transaction id has, by
   * definition, just made the planning call that produced it — so the right
   * answer is usually already in its own history. Naming it turns a denial the
   * agent has to decode into one it can act on directly. Nothing is granted
   * here that the agent could not already execute: only its own accepted
   * receipts of the expected kind are considered.
   */
  #planningTransactionHint(
    expectedPlanningActions: string[],
    agentId: string
  ): Record<string, JsonValue> {
    const activeId = this.#hierarchy.get(agentId).id;
    const candidates = Object.entries(this.#checkpoint.committed_actions)
      .filter(([, receipt]) => receipt.agent_id === activeId
        && receipt.accepted
        && receipt.kind === "tool"
        && expectedPlanningActions.includes(receipt.name))
      .map(([transactionId]) => transactionId);
    const latest = candidates.at(-1);
    return latest === undefined ? {} : { your_latest_matching_transaction_id: latest };
  }

  #acquireBody(channels: BodyChannel[], transactionId: string): JsonValue | undefined {
    const busy = channels.flatMap((channel) => {
      const owner = this.#bodyLeases.get(channel);
      return owner && owner !== transactionId ? [{ channel, owner }] : [];
    });
    if (busy.length > 0) return { busy };
    for (const channel of channels) this.#bodyLeases.set(channel, transactionId);
    return undefined;
  }

  #releaseBody(channels: BodyChannel[], transactionId: string): void {
    for (const channel of channels) {
      if (this.#bodyLeases.get(channel) === transactionId) this.#bodyLeases.delete(channel);
    }
  }

  async #persistHierarchy(): Promise<void> {
    this.#synchronizeHierarchy(new Date().toISOString());
    await this.#writeCheckpoint();
  }

  #synchronizeHierarchy(at: string): void {
    this.#checkpoint.active_agent_id = this.#hierarchy.activeId;
    this.#checkpoint.active_agent_ids = this.#hierarchy.activeIds;
    this.#checkpoint.nodes = this.#hierarchy.snapshot();
    this.#checkpoint.world = this.#world.snapshot();
    this.#checkpoint.updated_at = at;
  }

  async #commitLifecycle(
    type: RunLifecycleEventType,
    data: () => JsonValue
  ): Promise<void> {
    const at = new Date().toISOString();
    this.#synchronizeHierarchy(at);
    this.#checkpoint.pending_lifecycle_events.push(createLifecycleEvent({
      runId: this.#checkpoint.run_id,
      type,
      at,
      data: data()
    }));
    await this.#writeCheckpoint();
    await this.#reconcileLifecycleOutbox();
  }

  async #reconcileLifecycleOutbox(): Promise<void> {
    await reconcileLifecycleOutbox({
      store: this.#store,
      checkpoint: this.#checkpoint,
      persistCheckpoint: () => this.#writeCheckpoint(),
      eventSink: this.#eventSink
    });
  }

  async #emitHierarchyChanged(): Promise<void> {
    await this.emit("hierarchy_changed", json({
      nodes: this.#hierarchy.snapshot(),
      active_agent_id: this.#hierarchy.activeId,
      active_agent_ids: this.#hierarchy.activeIds
    }));
  }

  async #recordWorldFrames(frames: WorldSnapshot[]): Promise<void> {
    const fresh = frames.filter((frame) => frame.frame > this.#lastBroadcastFrame);
    if (fresh.length === 0) return;
    const latest = fresh.at(-1)!;
    this.#lastBroadcastFrame = latest.frame;
    this.#checkpoint.world = structuredClone(latest);
    this.#checkpoint.updated_at = new Date().toISOString();
    const firstProgressForActiveCommand = Object.values(
      this.#checkpoint.inflight_actions
    ).some((action) =>
      latest.frame > action.world_before_frame
        && action.world_before_frame >= this.#lastPersistedWorldFrame
    );
    if (
      firstProgressForActiveCommand
      || latest.frame - this.#lastPersistedWorldFrame
        >= PARTIAL_WORLD_CHECKPOINT_INTERVAL_FRAMES
    ) {
      await this.#writeCheckpoint();
    }
    await this.broadcast("world_frames", json({
      transaction_id: this.#checkpoint.inflight_action?.transaction_id ?? null,
      transaction_ids: Object.keys(this.#checkpoint.inflight_actions),
      frames: fresh
    }));
  }

  async #recordTerminalWorld(frame: WorldSnapshot, transactionId: string): Promise<void> {
    this.#lastBroadcastFrame = Math.max(this.#lastBroadcastFrame, frame.frame);
    this.#checkpoint.world = structuredClone(frame);
    this.#checkpoint.updated_at = new Date().toISOString();
    await this.#writeCheckpoint();
    await this.broadcast("world_frames", json({
      transaction_id: transactionId,
      transaction_ids: Object.keys(this.#checkpoint.inflight_actions),
      frames: [frame]
    }));
  }

  /**
   * A committed receipt becomes authoritative with the checkpoint, before its
   * audit records are appended. Rebuild only those missing records after a
   * restart; physical execution is never entered from this path.
   */
  async #reconcileCommittedActionJournals(): Promise<void> {
    const receipts = Object.values(this.#checkpoint.committed_actions);
    if (receipts.length === 0) return;

    const expectedReceipts = new Map(
      receipts.map((receipt) => [receipt.transaction_id, receipt])
    );
    const journaledActions = new Set<string>();
    await this.#store.scanJournal("actions", (entry) => {
      const transactionId = journalTransactionId(entry);
      if (!transactionId) return;
      const expected = expectedReceipts.get(transactionId);
      if (!expected) return;
      if (!isDeepStrictEqual(entry, expected)) {
        throw new Error(
          `Action journal conflicts with committed checkpoint for transaction ${transactionId}`
        );
      }
      journaledActions.add(transactionId);
    });
    await this.#store.appendMany(
      "actions",
      receipts
        .filter((receipt) => !journaledActions.has(receipt.transaction_id))
        .map((receipt) => receipt as unknown as JsonValue)
    );

    const hierarchyKeys = new Set<string>();
    await this.#store.scanJournal("hierarchy", (entry) => {
      const key = typedTransactionKey(entry);
      if (key) hierarchyKeys.add(key);
    });
    const missingHierarchy = receipts.flatMap((receipt) => {
      const type = isInterruptedRecoveryReceipt(receipt)
        ? "action_interrupted"
        : "action_committed";
      const key = transactionEventKey(type, receipt.transaction_id);
      if (hierarchyKeys.has(key)) return [];
      return [json({
        type,
        node_id: receipt.agent_id,
        transaction_id: receipt.transaction_id,
        ...(!isInterruptedRecoveryReceipt(receipt)
          ? { accepted: receipt.accepted, code: receipt.code }
          : {}),
        at: receipt.committed_at
      })];
    });
    await this.#store.appendMany("hierarchy", missingHierarchy);

    const checkerCounts = new Map<string, number>();
    await this.#store.scanJournal("checker", (entry) => {
      const serialized = JSON.stringify(entry);
      checkerCounts.set(serialized, (checkerCounts.get(serialized) ?? 0) + 1);
    });
    const missingChecker: JsonValue[] = [];
    for (const receipt of receipts) {
      if (receipt.kind !== "checker") continue;
      const serialized = JSON.stringify(receipt.detail);
      const available = checkerCounts.get(serialized) ?? 0;
      if (available > 0) {
        checkerCounts.set(serialized, available - 1);
      } else {
        missingChecker.push(receipt.detail);
      }
    }
    await this.#store.appendMany("checker", missingChecker);

    const eventKeys = new Set<string>();
    await this.#store.scanJournal("events", (entry) => {
      const key = runtimeEventTransactionKey(entry);
      if (key) eventKeys.add(key);
    });
    const missingEvents: RuntimeEvent[] = [];
    for (const receipt of receipts) {
      for (const type of expectedPostCommitEventTypes(receipt)) {
        const key = transactionEventKey(type, receipt.transaction_id);
        if (eventKeys.has(key)) continue;
        eventKeys.add(key);
        missingEvents.push(reconciledRuntimeEvent(this.#checkpoint.run_id, type, receipt));
      }
    }
    const persistedEvents = await this.#store.appendRuntimeEvents(missingEvents);
    for (const event of persistedEvents) await this.#eventSink(event);
  }

  async #recoverInflightActions(): Promise<void> {
    const inflightActions = Object.values(this.#checkpoint.inflight_actions);
    if (inflightActions.length === 0 && this.#checkpoint.inflight_action) {
      inflightActions.push(this.#checkpoint.inflight_action);
    }
    if (inflightActions.length === 0) return;
    const recoveredWorld = this.#world.recoverInterruptedCommands() ?? this.#world.snapshot();
    const committedAt = new Date().toISOString();
    const receipts = inflightActions.map((inflight): ActionReceipt => ({
        transaction_id: inflight.transaction_id,
        agent_id: inflight.agent_id,
        agent_name: inflight.agent_name,
        kind: "skill",
        name: inflight.name,
        input: structuredClone(inflight.input),
        accepted: false,
        code: "command_interrupted",
        detail: {
          reason: "The process stopped after actuation began; persisted partial state was retained and the command will not be repeated."
        },
        world_before_frame: inflight.world_before_frame,
        ...(inflight.world_before_revision === undefined
          ? {}
          : { world_before_revision: inflight.world_before_revision }),
        world_after_frame: recoveredWorld.frame,
        frame_count: Math.max(0, recoveredWorld.frame - inflight.world_before_frame),
        world_revision: recoveredWorld.world_revision,
        channels: [...inflight.channels],
        gates: [
          { name: "capability_authority", status: "passed", detail: { agent_id: inflight.agent_id } },
          { name: "body_lease", status: "passed", detail: { channels: inflight.channels } },
          { name: "exactly_once_recovery", status: "rejected", detail: { interrupted: true } }
        ],
        committed_at: committedAt
    }));
    for (const receipt of receipts) {
      this.#hierarchy.recordToolResult(receipt.agent_id, receipt as unknown as JsonValue);
      this.#checkpoint.committed_actions[receipt.transaction_id] = receipt;
      this.#spatialMemory.observe(receipt, recoveredWorld);
    }
    this.#checkpoint.inflight_action = null;
    this.#checkpoint.inflight_actions = {};
    this.#checkpoint.world = recoveredWorld;
    this.#checkpoint.spatial_memory = this.#spatialMemory.snapshot();
    this.#checkpoint.nodes = this.#hierarchy.snapshot();
    this.#checkpoint.updated_at = committedAt;
    await this.#writeCheckpoint();
    await Promise.all([
      this.#store.appendMany("actions", receipts as unknown as JsonValue[]),
      this.#store.appendMany("hierarchy", inflightActions.map((inflight) => json({
        type: "action_interrupted",
        node_id: inflight.agent_id,
        transaction_id: inflight.transaction_id,
        at: committedAt
      })))
    ]);
    for (const receipt of receipts) {
      await this.emit("action_rejected", json({ receipt, recovered: true }));
    }
  }

  #focusedInflightAction(): RunCheckpoint["inflight_action"] {
    const focused = this.#checkpoint.active_agent_id;
    const actions = Object.values(this.#checkpoint.inflight_actions);
    return actions.find((action) => action.agent_id === focused) ?? actions.at(-1) ?? null;
  }

  async #writeCheckpoint(): Promise<void> {
    const checkpoint = structuredClone(this.#checkpoint);
    const write = this.#checkpointWrite.then(() => this.#store.writeCheckpoint(checkpoint));
    this.#checkpointWrite = write.catch(() => undefined);
    await write;
    this.#lastPersistedWorldFrame = Math.max(
      this.#lastPersistedWorldFrame,
      checkpoint.world.frame
    );
  }

}

export function createCheckpoint(input: {
  store: RunStore;
  hierarchy: HierarchyProjection;
  capabilityCatalog: string[];
  world: RapierWorld;
}): RunCheckpoint {
  const now = new Date().toISOString();
  return {
    version: 3,
    run_id: input.store.definition.run_id,
    scenario_id: input.store.definition.scenario_id,
    goal: structuredClone(input.store.definition.goal),
    capability_catalog: [...input.capabilityCatalog],
    status: "starting",
    root_id: input.hierarchy.rootId,
    active_agent_id: input.hierarchy.activeId,
    active_agent_ids: input.hierarchy.activeIds,
    nodes: input.hierarchy.snapshot(),
    world: input.world.snapshot(),
    inflight_action: null,
    inflight_actions: {},
    committed_actions: {},
    spatial_memory: [],
    context_memory: structuredClone(EmptyContextMemoryState),
    pending_lifecycle_events: [],
    total_model_calls: 0,
    checker: null,
    final_output: null,
    error: null,
    created_at: now,
    updated_at: now
  };
}

function denied(code: string, detail: JsonValue): CommandResult {
  return { accepted: false, code, detail };
}

function journalObject(value: JsonValue | undefined): Record<string, JsonValue> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value
    : undefined;
}

function journalTransactionId(entry: JsonValue): string | undefined {
  const transactionId = journalObject(entry)?.transaction_id;
  return typeof transactionId === "string" ? transactionId : undefined;
}

function transactionEventKey(type: string, transactionId: string): string {
  return `${type}\u0000${transactionId}`;
}

function typedTransactionKey(entry: JsonValue): string | undefined {
  const record = journalObject(entry);
  const type = record?.type;
  const transactionId = record?.transaction_id;
  return typeof type === "string" && typeof transactionId === "string"
    ? transactionEventKey(type, transactionId)
    : undefined;
}

function runtimeEventTransactionKey(entry: JsonValue): string | undefined {
  const event = journalObject(entry);
  const type = event?.type;
  const data = journalObject(event?.data);
  const receipt = journalObject(data?.receipt);
  const transactionId = data?.transaction_id ?? receipt?.transaction_id;
  return typeof type === "string" && typeof transactionId === "string"
    ? transactionEventKey(type, transactionId)
    : undefined;
}

function isInterruptedRecoveryReceipt(receipt: ActionReceipt): boolean {
  return receipt.code === "command_interrupted"
    && receipt.gates.some((gate) => gate.name === "exactly_once_recovery");
}

function expectedPostCommitEventTypes(receipt: ActionReceipt): string[] {
  if (isInterruptedRecoveryReceipt(receipt)) return ["action_rejected"];
  return [
    ...(!receipt.accepted ? ["action_rejected"] : []),
    ...(receipt.kind === "skill" && receipt.channels.length > 0
      ? ["command_finished"]
      : []),
    "action_committed"
  ];
}

function reconciledRuntimeEvent(
  runId: string,
  type: string,
  receipt: ActionReceipt
): RuntimeEvent {
  const recovered = isInterruptedRecoveryReceipt(receipt);
  const data = type === "command_finished"
    ? json({
        transaction_id: receipt.transaction_id,
        agent_id: receipt.agent_id,
        accepted: receipt.accepted,
        code: receipt.code,
        world_frame: receipt.world_after_frame,
        receipt,
        reconciled: true
      })
    : json({
        receipt,
        ...(recovered ? { recovered: true } : {}),
        reconciled: true
      });
  return {
    event_id: randomUUID(),
    run_id: runId,
    type,
    at: receipt.committed_at,
    data,
    durable: true
  };
}

function fixedWorldPlanningChannels(name: string): BodyChannel[] {
  return name === "solve_end_effector_position" || name === "solve_end_effector_pose"
    ? ["base", "arm"]
    : [];
}

function assertSameActionTransaction(
  transactionId: string,
  existing: ActionCommitSignature,
  attempted: ActionCommitSignature
): void {
  if (existing.agent_id !== attempted.agent_id
    || existing.kind !== attempted.kind
    || existing.name !== attempted.name
    || !isDeepStrictEqual(existing.input, attempted.input)) {
    throw new Error(`Transaction ${transactionId} was reused with different action data`);
  }
}

function actionOutput(receipt: ActionReceipt): string {
  return JSON.stringify({
    accepted: receipt.accepted,
    code: receipt.code,
    detail: receipt.detail,
    transaction_id: receipt.transaction_id,
    world_frame: receipt.world_after_frame
  });
}

function json(value: unknown): JsonValue {
  const text = JSON.stringify(value);
  if (text === undefined) return null;
  return JSON.parse(text) as JsonValue;
}

function sameDelegationSpec(node: TaskNode, spec: AgentSpec): boolean {
  return node.name === spec.name
    && node.objective === spec.objective
    && node.may_delegate === spec.may_delegate
    && sameOrderedStrings(node.success_criteria, spec.success_criteria)
    && isDeepStrictEqual(node.evidence_requirements, spec.evidence_requirements ?? [])
    && sameOrderedNumbers(node.goal_predicate_indexes, spec.goal_predicate_indexes)
    && sameOrderedStrings(node.capabilities, spec.capabilities)
    && node.references.length === spec.references.length
    && node.references.every((reference, index) =>
      reference.name === spec.references[index]?.name
        && reference.transaction_id === spec.references[index]?.transaction_id
    );
}

function sameOrderedStrings(left: string[], right: string[]): boolean {
  return left.length === right.length
    && left.every((value, index) => value === right[index]);
}

function sameOrderedNumbers(left: number[], right: number[]): boolean {
  return left.length === right.length
    && left.every((value, index) => value === right[index]);
}

function asObject(value: JsonValue): Record<string, JsonValue> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value
    : { value };
}

/**
 * Some body targets remain authoritative across unrelated channel revisions.
 * This does not infer or choose a target: it only observes that the exact
 * model-selected head command already succeeded and the live joints still
 * satisfy it. Plans and sensor reads remain revision-sensitive.
 */
function acceptedStateStillHolds(
  name: string,
  input: JsonValue,
  world: WorldSnapshot
): boolean {
  if (name !== "set_head_target"
    || typeof input !== "object"
    || input === null
    || Array.isArray(input)) return false;
  const yaw = input.yaw;
  const pitch = input.pitch;
  if (typeof yaw !== "number" || typeof pitch !== "number") return false;
  const options = input.options;
  const requestedTolerance = typeof options === "object"
    && options !== null
    && !Array.isArray(options)
    && typeof options.tolerance === "number"
    ? options.tolerance
    : 0.004;
  const tolerance = Math.min(0.03, Math.max(0.001, requestedTolerance));
  return Math.abs(world.robot.joints.head_yaw - yaw) <= tolerance
    && Math.abs(world.robot.joints.head_pitch - pitch) <= tolerance;
}
