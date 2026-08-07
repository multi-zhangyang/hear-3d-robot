import { z } from "zod";
import {
  HUMANOID_END_EFFECTORS,
  type JsonValue,
  type Vec3
} from "../../domain/schema.js";
import {
  modelPayloadSha256,
  type ModelDecisionRef
} from "../../domain/model-call-authority.js";
import type { AutonomousCycleRef } from "../../domain/autonomous-cycle.js";
import type { ScenarioBlockRemovalTransaction } from "../../domain/scenario-block-removal.js";
import { yawFromQuaternion } from "../../world/geometry.js";
import { humanoidEndEffectorPosition } from "../../world/humanoid/end-effectors.js";
import type { HumanoidBodyChannel } from "../../world/humanoid/motion-plan.js";
import {
  type HumanoidFrameSink,
  type HumanoidPersistenceSink,
  type HumanoidWorld,
  type HumanoidWorldObservation,
  type HumanoidWorldSnapshot
} from "../../world/humanoid/world.js";
import {
  HumanoidActionInputs,
  type HumanoidActionName
} from "./actions.js";
import { MAX_CHECKPOINT_ACTION_RECEIPTS } from "./embodied-memory.js";
import { normalizeHumanoidMotionCandidateBatchInput } from "./motion-candidate-input.js";
import { BlockRemovalAuthorityError } from "./block-removal-authority.js";
import {
  navigationTransitClearanceContext,
  navigationTransitClearanceFromRejection,
  navigationTransitClearanceMotionRejection,
  type NavigationTransitClearanceRequirement
} from "./navigation-transit-clearance.js";
import {
  bindHumanoidSkill,
  ActiveHumanoidSkillBindingSchema,
  validateSkillPlanningReference,
  type ActiveHumanoidSkillBinding
} from "./skill-binding.js";
import {
  createHumanoidRecoveryPolicy,
  humanoidRecoverySelectionAccepted,
  HumanoidRecoveryPolicyStateSchema,
  type HumanoidRecoveryPolicyState
} from "./recovery-policy.js";
import {
  advanceHumanoidSkillPlan,
  authorizeHumanoidSkillPlanNode,
  registerHumanoidSkillPlan,
  RegisteredHumanoidSkillPlanSchema,
  type RegisteredHumanoidSkillPlan
} from "./skill-plan.js";
import { planAutonomousHumanoidSkill } from "./autonomous-skill-planner.js";

type HumanoidPlanningActionName = "plan_humanoid_skill"
  | "plan_whole_body_motion"
  | "plan_whole_body_motion_candidates"
  | "plan_humanoid_navigation";

type HumanoidPhysicalActionName = "execute_humanoid_skill"
  | "execute_whole_body_motion"
  | "execute_humanoid_navigation";

interface ManipulationBasePlacementRequirement {
  observedWorldRevision: number;
  objects: Array<{
    objectId: string;
    objectCenterWorld: Vec3;
    placements: HumanoidWorldObservation["manipulationBasePlacements"];
  }>;
}

interface RepeatedPlanningFailure {
  action: HumanoidPlanningActionName;
  fingerprint: string;
  physicalExecutionRevision: number;
  count: number;
  lastCode: string;
}

const HumanoidActionRuntimeStateSchema = z.object({
  version: z.literal(1),
  latest_physical_execution_revision: z.number().int().nonnegative(),
  skill_plans: z.array(RegisteredHumanoidSkillPlanSchema),
  active_skill_plan_transactions: z.record(
    z.string().trim().min(1),
    z.string().trim().min(1)
  ),
  active_skills: z.array(ActiveHumanoidSkillBindingSchema),
  planning_skill_bindings: z.array(z.object({
    planning_transaction_id: z.string().trim().min(1),
    binding: ActiveHumanoidSkillBindingSchema
  }).strict()),
  recovery_policies: z.array(z.object({
    agent_id: z.string().trim().min(1),
    policy: HumanoidRecoveryPolicyStateSchema
  }).strict())
}).strict().superRefine((state, context) => {
  const planIds = new Set(state.skill_plans.map(({ transaction_id }) => transaction_id));
  if (planIds.size !== state.skill_plans.length) {
    context.addIssue({
      code: "custom",
      path: ["skill_plans"],
      message: "Persisted Skill plan transaction identities must be unique"
    });
  }
  for (const [agentId, transactionId] of Object.entries(
    state.active_skill_plan_transactions
  )) {
    const plan = state.skill_plans.find((entry) => entry.transaction_id === transactionId);
    if (!plan || plan.agent_id !== agentId) {
      context.addIssue({
        code: "custom",
        path: ["active_skill_plan_transactions", agentId],
        message: "Active Skill plan index does not reference its Agent plan"
      });
    }
  }
  const activeAgents = state.active_skills.map(({ agent_id }) => agent_id);
  if (new Set(activeAgents).size !== activeAgents.length) {
    context.addIssue({
      code: "custom",
      path: ["active_skills"],
      message: "Only one active Skill binding is allowed per Agent"
    });
  }
  const planningIds = state.planning_skill_bindings.map(
    ({ planning_transaction_id }) => planning_transaction_id
  );
  if (new Set(planningIds).size !== planningIds.length) {
    context.addIssue({
      code: "custom",
      path: ["planning_skill_bindings"],
      message: "Planning transaction Skill bindings must be unique"
    });
  }
  const recoveryAgents = state.recovery_policies.map(({ agent_id }) => agent_id);
  if (new Set(recoveryAgents).size !== recoveryAgents.length) {
    context.addIssue({
      code: "custom",
      path: ["recovery_policies"],
      message: "Only one recovery policy is allowed per Agent"
    });
  }
});

const MANIPULATION_APPROACH_RADIUS_METERS = 1;
const MANIPULATION_BASE_TARGET_TOLERANCE_METERS = 0.04;
const SKILL_AUTHORITY_OBJECT_DRIFT_METERS = 0.015;
const SKILL_AUTHORITY_ORIENTATION_DRIFT_RADIANS = 0.05;

export interface HumanoidPhysicalExecutionIntent {
  transactionId: string;
  agentId: string;
  action: HumanoidPhysicalActionName;
  fingerprint: string;
  planningTransactionId: string;
  planId: string;
}

export interface HumanoidActionToolCallAuthority {
  tool_call_id: string;
  tool_name: string;
  arguments_sha256: string;
  normalized_arguments_sha256?: string;
}

export interface HumanoidActionInvoker {
  isActionAvailable?(
    name: HumanoidActionName,
    agentId: string
  ): boolean;
  invoke(
    name: HumanoidActionName,
    rawInput: unknown,
    transactionId: string,
    agentId: string,
    authority: HumanoidActionToolCallAuthority
  ): Promise<HumanoidActionReceipt>;
}

export interface HumanoidActionReceipt {
  transactionId: string;
  agentId: string;
  decision?: ModelDecisionRef | undefined;
  cycle?: AutonomousCycleRef | undefined;
  action: HumanoidActionName;
  input: JsonValue;
  fingerprint: string;
  accepted: boolean;
  code: string;
  worldBeforeRevision: number;
  worldAfterRevision: number;
  frameCount: number;
  channels: HumanoidBodyChannel[];
  detail: JsonValue;
  committedAt: string;
}

export interface HumanoidActionRuntimeOptions {
  frameSink?: HumanoidFrameSink;
  physicalFrameSink?: HumanoidPersistenceSink;
  receiptSink?: (receipt: HumanoidActionReceipt) => void | Promise<void>;
  beforePhysicalExecution?: (
    intent: HumanoidPhysicalExecutionIntent
  ) => void | Promise<void>;
  receiptNormalizer?: (
    receipt: HumanoidActionReceipt
  ) => HumanoidActionReceipt | Promise<HumanoidActionReceipt>;
  prepareBlockRemoval?: (input: {
    transactionId: string;
    agentId: string;
    solidId: string;
    executionTransactionId: string;
  }) => ScenarioBlockRemovalTransaction | Promise<ScenarioBlockRemovalTransaction>;
  receipts?: Readonly<Record<string, HumanoidActionReceipt>>;
  state?: JsonValue | null;
  realtimeExecution?: boolean;
  retainPhysicalTerminals?: boolean;
  requireSkillBinding?: boolean;
  signal?: AbortSignal;
}

export class HumanoidActionRuntime {
  readonly #world: HumanoidWorld;
  readonly #frameSink: HumanoidFrameSink | undefined;
  readonly #physicalFrameSink: HumanoidPersistenceSink | undefined;
  readonly #receiptSink: HumanoidActionRuntimeOptions["receiptSink"];
  readonly #beforePhysicalExecution: HumanoidActionRuntimeOptions[
    "beforePhysicalExecution"
  ];
  readonly #receiptNormalizer: HumanoidActionRuntimeOptions["receiptNormalizer"];
  readonly #prepareBlockRemoval: HumanoidActionRuntimeOptions["prepareBlockRemoval"];
  readonly #realtimeExecution: boolean;
  readonly #retainPhysicalTerminals: boolean;
  readonly #requireSkillBinding: boolean;
  readonly #signal: AbortSignal | undefined;
  readonly #receipts = new Map<string, HumanoidActionReceipt>();
  readonly #transactions = new Map<string, {
    fingerprint: string;
    decisionSha256: string | undefined;
    promise: Promise<HumanoidActionReceipt>;
  }>();
  readonly #receiptCommits = new Map<string, Promise<void>>();
  readonly #uncommittedReceiptIds = new Set<string>();
  readonly #planChannels = new Map<string, HumanoidBodyChannel[]>();
  readonly #inFlightTransactions = new Set<string>();
  readonly #observationRevisionByAgent = new Map<string, number>();
  readonly #observationByAgent = new Map<string, HumanoidWorldObservation>();
  readonly #activeSkillByAgent = new Map<string, ActiveHumanoidSkillBinding>();
  readonly #skillByPlanningTransactionId = new Map<
    string,
    ActiveHumanoidSkillBinding
  >();
  readonly #recoveryPolicyByAgent = new Map<string, HumanoidRecoveryPolicyState>();
  readonly #skillPlansByTransactionId = new Map<
    string,
    RegisteredHumanoidSkillPlan
  >();
  readonly #activeSkillPlanTransactionByAgent = new Map<string, string>();
  readonly #planningAttemptRevisionByAgent = new Map<string, number>();
  readonly #manipulationBasePlacementRequirementByAgent = new Map<
    string,
    ManipulationBasePlacementRequirement
  >();
  readonly #navigationTransitClearanceRequirementByAgent = new Map<
    string,
    NavigationTransitClearanceRequirement
  >();
  readonly #planningFailureByFingerprint = new Map<
    string,
    RepeatedPlanningFailure
  >();
  readonly #latestPlanningFailureKeyByAgent = new Map<string, string>();
  #latestPhysicalExecutionRevision = 0;

  constructor(world: HumanoidWorld, options: HumanoidActionRuntimeOptions = {}) {
    this.#world = world;
    this.#frameSink = options.frameSink;
    this.#physicalFrameSink = options.physicalFrameSink;
    this.#receiptSink = options.receiptSink;
    this.#beforePhysicalExecution = options.beforePhysicalExecution;
    this.#receiptNormalizer = options.receiptNormalizer;
    this.#prepareBlockRemoval = options.prepareBlockRemoval;
    this.#realtimeExecution = options.realtimeExecution ?? false;
    this.#retainPhysicalTerminals = options.retainPhysicalTerminals ?? false;
    this.#requireSkillBinding = options.requireSkillBinding ?? false;
    this.#signal = options.signal;
    if (options.state !== undefined && options.state !== null) {
      this.#restoreState(options.state);
    }
    for (const [transactionId, source] of Object.entries(options.receipts ?? {})) {
      const receipt = structuredClone(source);
      if (receipt.transactionId !== transactionId) {
        throw new Error(`Humanoid receipt identity mismatch: ${transactionId}`);
      }
      const fingerprint = humanoidActionFingerprint(
        receipt.action,
        receipt.agentId,
        receipt.input
      );
      if (receipt.fingerprint !== fingerprint) {
        throw new Error(`Humanoid receipt fingerprint mismatch: ${transactionId}`);
      }
      this.#receipts.set(transactionId, receipt);
      if (receipt.accepted && receipt.action === "observe_humanoid") {
        if (receipt.agentId !== "humanoid-motion-reference") {
          this.#observationRevisionByAgent.set(
            receipt.agentId,
            Math.max(
              this.#observationRevisionByAgent.get(receipt.agentId) ?? 0,
              receipt.worldAfterRevision
            )
          );
        }
      }
      if (receipt.accepted
        && (receipt.action === "execute_humanoid_skill"
          || receipt.action === "execute_whole_body_motion"
          || receipt.action === "execute_humanoid_navigation")) {
        this.#latestPhysicalExecutionRevision = Math.max(
          this.#latestPhysicalExecutionRevision,
          receipt.worldAfterRevision
        );
        this.#planningFailureByFingerprint.clear();
        this.#latestPlanningFailureKeyByAgent.clear();
      }
      this.#recordPlanningOutcome(receipt);
      this.#recordNavigationTransitClearance(receipt);
      this.#transactions.set(transactionId, {
        fingerprint,
        decisionSha256: receipt.decision
          ? modelPayloadSha256(receipt.decision)
          : undefined,
        promise: Promise.resolve(receipt)
      });
      if (receipt.accepted
        && (receipt.action === "plan_humanoid_skill"
          || receipt.action === "plan_whole_body_motion"
          || receipt.action === "plan_whole_body_motion_candidates"
          || receipt.action === "plan_humanoid_navigation")) {
        const planId = jsonObject(receipt.detail)?.plan_id;
        if (typeof planId === "string" && planId) {
          this.#planChannels.set(planId, [...receipt.channels]);
        }
      }
    }
    this.#pruneTransactionHistory();
  }

  persistenceState(): JsonValue {
    return jsonValue(HumanoidActionRuntimeStateSchema.parse({
      version: 1,
      latest_physical_execution_revision: this.#latestPhysicalExecutionRevision,
      skill_plans: [...this.#skillPlansByTransactionId.values()]
        .map((plan) => structuredClone(plan)),
      active_skill_plan_transactions: Object.fromEntries(
        this.#activeSkillPlanTransactionByAgent
      ),
      active_skills: [...this.#activeSkillByAgent.values()]
        .map((binding) => structuredClone(binding)),
      planning_skill_bindings: [...this.#skillByPlanningTransactionId]
        .map(([planningTransactionId, binding]) => ({
          planning_transaction_id: planningTransactionId,
          binding: structuredClone(binding)
        })),
      recovery_policies: [...this.#recoveryPolicyByAgent]
        .map(([agentId, policy]) => ({
          agent_id: agentId,
          policy: structuredClone(policy)
        }))
    }));
  }

  #restoreState(rawState: JsonValue): void {
    const state = HumanoidActionRuntimeStateSchema.parse(rawState);
    const currentWorldRevision = this.#world.snapshot().worldRevision;
    if (state.latest_physical_execution_revision > currentWorldRevision) {
      throw new Error(
        "Humanoid action runtime state is ahead of the authoritative world"
      );
    }
    this.#latestPhysicalExecutionRevision = state.latest_physical_execution_revision;
    for (const plan of state.skill_plans) {
      this.#skillPlansByTransactionId.set(
        plan.transaction_id,
        structuredClone(plan)
      );
    }
    for (const [agentId, transactionId] of Object.entries(
      state.active_skill_plan_transactions
    )) {
      this.#activeSkillPlanTransactionByAgent.set(agentId, transactionId);
    }
    for (const binding of state.active_skills) {
      this.#activeSkillByAgent.set(binding.agent_id, structuredClone(binding));
    }
    for (const entry of state.planning_skill_bindings) {
      this.#skillByPlanningTransactionId.set(
        entry.planning_transaction_id,
        structuredClone(entry.binding)
      );
    }
    for (const { agent_id: agentId, policy } of state.recovery_policies) {
      this.#recoveryPolicyByAgent.set(agentId, structuredClone(policy));
    }
  }

  snapshot(): HumanoidWorldSnapshot {
    return this.#world.snapshot();
  }

  receipt(transactionId: string): HumanoidActionReceipt | undefined {
    const receipt = this.#receipts.get(transactionId);
    return receipt ? structuredClone(receipt) : undefined;
  }

  isActionAvailable(name: HumanoidActionName, agentId: string): boolean {
    const normalizedAgentId = agentId.trim();
    if (normalizedAgentId !== "humanoid-motion-reference") return true;
    if (name === "plan_humanoid_navigation"
      && this.#navigationTransitClearanceRequirementByAgent.has(normalizedAgentId)) {
      return false;
    }
    const observedRevision = this.#observationRevisionByAgent.get(
      normalizedAgentId
    );
    const latestFailureKey = this.#latestPlanningFailureKeyByAgent.get(
      normalizedAgentId
    );
    const latestFailure = latestFailureKey === undefined
      ? undefined
      : this.#planningFailureByFingerprint.get(latestFailureKey);
    const hasCurrentPlanningFailure = latestFailure !== undefined
      && latestFailure.physicalExecutionRevision
        === this.#latestPhysicalExecutionRevision;
    if (name === "observe_humanoid") {
      const lastPlanningAttempt = this.#planningAttemptRevisionByAgent.get(
        normalizedAgentId
      );
      return observedRevision === undefined
        || observedRevision < this.#latestPhysicalExecutionRevision
        || (lastPlanningAttempt ?? -1) >= observedRevision;
    }
    if (name === "begin_humanoid_skill") {
      const activePlan = this.#activeSkillPlan(normalizedAgentId);
      return observedRevision !== undefined
        && observedRevision >= this.#latestPhysicalExecutionRevision
        && this.#observationByAgent.has(normalizedAgentId)
        && (!this.#requireSkillBinding
          || (activePlan !== null
            && !this.#activeSkillByAgent.has(normalizedAgentId)));
    }
    if (name === "submit_humanoid_skill_plan") {
      const observation = this.#observationByAgent.get(normalizedAgentId);
      const activePlan = this.#activeSkillPlan(normalizedAgentId);
      return observedRevision !== undefined
        && observedRevision >= this.#latestPhysicalExecutionRevision
        && observation !== undefined
        && (!this.#requireSkillBinding
          || (!this.#activeSkillByAgent.has(normalizedAgentId)
            && (activePlan === null
              || activePlan.world_revision !== observation.worldRevision))
          || hasCurrentPlanningFailure);
    }
    if (!isPlanningAction(name)) return true;
    if (observedRevision === undefined
      || observedRevision < this.#latestPhysicalExecutionRevision) {
      return false;
    }
    if (this.#requireSkillBinding) {
      const skill = this.#activeSkillByAgent.get(normalizedAgentId);
      if (!skill
        || skill.observed_world_revision < this.#latestPhysicalExecutionRevision
        || skill.planning_action !== name) return false;
    }
    if (latestFailure?.action === name
      && latestFailure.physicalExecutionRevision
        === this.#latestPhysicalExecutionRevision
      && latestFailure.count >= 3) {
      return false;
    }
    return true;
  }

  planningToolState(agentId: string): JsonValue {
    const normalizedAgentId = agentId.trim();
    const activeSkill = this.#activeSkillByAgent.get(normalizedAgentId);
    const latestFailureKey = this.#latestPlanningFailureKeyByAgent.get(
      normalizedAgentId
    );
    const latestFailure = latestFailureKey === undefined
      ? undefined
      : this.#planningFailureByFingerprint.get(latestFailureKey);
    const cooldown = latestFailure
      && latestFailure.physicalExecutionRevision
        === this.#latestPhysicalExecutionRevision
      && latestFailure.count >= 3
      ? {
          action: latestFailure.action,
          code: "repeated_planning_failure",
          repeated_failure_count: latestFailure.count,
          previous_code: latestFailure.lastCode,
          physical_execution_revision: latestFailure.physicalExecutionRevision,
          clears_after: "a materially different planning action or physical execution",
          recovery: latestFailure.action === "plan_humanoid_navigation"
            ? "Navigation is temporarily unavailable. Use plan_whole_body_motion_candidates for a physically different task-space posture strategy before retrying navigation; do not encode pure root navigation as a whole-body workaround."
            : "Whole-body planning is temporarily unavailable. Use a materially different navigation strategy before retrying whole-body planning."
        }
      : null;
    const transitClearance = this.#navigationTransitClearanceRequirementByAgent.get(
      normalizedAgentId
    );
    return jsonValue({
      planning_actions: [
        "plan_humanoid_skill",
        "plan_whole_body_motion_candidates",
        "plan_humanoid_navigation"
      ].map((action) => ({
        action,
        available: this.isActionAvailable(
          action as HumanoidActionName,
          normalizedAgentId
        )
      })),
      cooldown,
      transit_clearance: transitClearance
        ? navigationTransitClearanceContext(transitClearance)
        : null,
      recovery_policy: this.#recoveryPolicyByAgent.get(normalizedAgentId) ?? null,
      skill_plan: this.#activeSkillPlan(normalizedAgentId),
      active_skill: activeSkill
        ? {
            transaction_id: activeSkill.transaction_id,
            skill_plan_transaction_id: activeSkill.skill_plan_transaction_id,
            skill_node_id: activeSkill.skill_node_id,
            invocation: activeSkill.invocation,
            phase: activeSkill.phase,
            phase_authority: activeSkill.phase_authority,
            planning_action: activeSkill.planning_action,
            observed_world_revision: activeSkill.observed_world_revision
          }
        : null
    });
  }

  async invoke(
    name: HumanoidActionName,
    rawInput: unknown,
    transactionId: string,
    agentId: string,
    decision?: ModelDecisionRef
  ): Promise<HumanoidActionReceipt> {
    const normalizedTransactionId = transactionId.trim();
    const normalizedAgentId = agentId.trim();
    if (!normalizedTransactionId) throw new Error("Humanoid action transaction id is required");
    if (!normalizedAgentId) throw new Error("Humanoid action agent id is required");
    const fingerprint = humanoidActionFingerprint(name, normalizedAgentId, rawInput);
    const decisionSha256 = decision ? modelPayloadSha256(decision) : undefined;
    const existing = this.#transactions.get(normalizedTransactionId);
    if (existing) {
      if (existing.fingerprint !== fingerprint
        || existing.decisionSha256 !== decisionSha256) {
        throw new Error(
          `Humanoid action transaction conflict: ${normalizedTransactionId}`
        );
      }
      const receipt = await existing.promise;
      await this.#ensureReceiptCommitted(receipt);
      return structuredClone(receipt);
    }
    const unresolved = this.#uncommittedReceiptIds.values().next().value;
    if (unresolved !== undefined) {
      throw new Error(
        `Humanoid action commit is unresolved; retry transaction ${unresolved} before executing another action`
      );
    }
    this.#inFlightTransactions.add(normalizedTransactionId);
    const promise = this.#invokeOnce(
      name,
      rawInput,
      normalizedTransactionId,
      normalizedAgentId,
      fingerprint,
      decision
    );
    this.#transactions.set(normalizedTransactionId, {
      fingerprint,
      decisionSha256,
      promise
    });
    try {
      const receipt = await promise;
      await this.#ensureReceiptCommitted(receipt);
      return structuredClone(receipt);
    } catch (error) {
      if (!this.#receipts.has(normalizedTransactionId)) {
        this.#transactions.delete(normalizedTransactionId);
      }
      throw error;
    } finally {
      this.#inFlightTransactions.delete(normalizedTransactionId);
      this.#pruneTransactionHistory();
    }
  }

  #pruneTransactionHistory(): void {
    const activePlanIds = new Set(this.#world.consumablePlanIds());
    const protectedTransactions = new Set(this.#inFlightTransactions);
    for (const [transactionId, receipt] of this.#receipts) {
      const planId = planIdFromReceipt(receipt);
      if (receipt.accepted
        && isPlanningAction(receipt.action)
        && planId !== undefined
        && activePlanIds.has(planId)) {
        protectedTransactions.add(transactionId);
      }
    }
    for (const transactionId of this.#uncommittedReceiptIds) {
      protectedTransactions.add(transactionId);
    }
    const recentTransactions = new Set(
      [...this.#receipts.keys()].slice(-MAX_CHECKPOINT_ACTION_RECEIPTS)
    );
    for (const transactionId of this.#receipts.keys()) {
      if (!recentTransactions.has(transactionId)
        && !protectedTransactions.has(transactionId)) {
        this.#receipts.delete(transactionId);
      }
    }
    for (const transactionId of this.#transactions.keys()) {
      if (!this.#receipts.has(transactionId)
        && !protectedTransactions.has(transactionId)) {
        this.#transactions.delete(transactionId);
      }
    }
    for (const planId of this.#planChannels.keys()) {
      if (!activePlanIds.has(planId)) this.#planChannels.delete(planId);
    }
  }

  async #invokeOnce(
    name: HumanoidActionName,
    rawInput: unknown,
    transactionId: string,
    agentId: string,
    fingerprint: string,
    decision: ModelDecisionRef | undefined
  ): Promise<HumanoidActionReceipt> {
    const before = this.#world.snapshot();
    const result = await this.#execute(name, rawInput, {
      transactionId,
      agentId,
      fingerprint
    });
    const after = this.#world.snapshot();
    const baseReceipt: HumanoidActionReceipt = {
      transactionId,
      agentId,
      ...(decision ? { decision: structuredClone(decision) } : {}),
      action: name,
      input: jsonValue(rawInput),
      fingerprint,
      accepted: result.accepted,
      code: result.code,
      worldBeforeRevision: before.worldRevision,
      worldAfterRevision: result.causalWorldAfterRevision
        ?? after.worldRevision,
      frameCount: result.causalFrameCount ?? 0,
      channels: result.channels,
      detail: jsonValue(result.detail),
      committedAt: new Date().toISOString()
    };
    const receipt = this.#receiptNormalizer
      ? await this.#receiptNormalizer(structuredClone(baseReceipt))
      : baseReceipt;
    assertNormalizedReceiptIdentity(baseReceipt, receipt);
    this.#receipts.set(transactionId, receipt);
    if (receipt.accepted && isPlanningAction(receipt.action)) {
      const skill = this.#activeSkillByAgent.get(receipt.agentId);
      if (skill) {
        this.#skillByPlanningTransactionId.set(
          receipt.transactionId,
          structuredClone(skill)
        );
      }
    }
    if (receipt.action === "execute_humanoid_skill"
      || receipt.action === "execute_whole_body_motion"
      || receipt.action === "execute_humanoid_navigation") {
      const planningTransactionId = jsonObject(receipt.input)
        ?.planning_transaction_id;
      const skill = typeof planningTransactionId === "string"
        ? this.#skillByPlanningTransactionId.get(planningTransactionId)
        : undefined;
      const physicallyAttempted = receipt.accepted
        || receipt.frameCount > 0
        || receipt.worldAfterRevision > receipt.worldBeforeRevision;
      if (physicallyAttempted) {
        this.#latestPhysicalExecutionRevision = Math.max(
          this.#latestPhysicalExecutionRevision,
          receipt.worldAfterRevision
        );
        this.#planningFailureByFingerprint.clear();
        this.#latestPlanningFailureKeyByAgent.clear();
        this.#activeSkillByAgent.clear();
        this.#observationByAgent.clear();
      }
      if (skill && receipt.accepted) {
        this.#recoveryPolicyByAgent.delete(skill.agent_id);
      } else if (skill && receipt.frameCount > 0
        && typeof planningTransactionId === "string") {
        this.#recoveryPolicyByAgent.set(
          skill.agent_id,
          createHumanoidRecoveryPolicy({
            executionTransactionId: receipt.transactionId,
            planningTransactionId,
            physicalFailureCode: receipt.code,
            worldRevision: receipt.worldAfterRevision,
            binding: skill
          })
        );
      }
      const skillPlanTransactionId = skill?.skill_plan_transaction_id;
      const skillPlan = skillPlanTransactionId
        ? this.#skillPlansByTransactionId.get(skillPlanTransactionId)
        : undefined;
      if (skill && skillPlan && receipt.accepted) {
        const advanced = advanceHumanoidSkillPlan({
          plan: skillPlan,
          binding: skill,
          worldRevision: receipt.worldAfterRevision,
          executionSucceeded: true
        });
        if (advanced) {
          this.#skillPlansByTransactionId.set(skillPlan.transaction_id, advanced);
        }
      } else if (skillPlanTransactionId && receipt.frameCount > 0) {
        this.#skillPlansByTransactionId.delete(skillPlanTransactionId);
        this.#activeSkillPlanTransactionByAgent.delete(skill!.agent_id);
      }
      if (typeof planningTransactionId === "string") {
        this.#skillByPlanningTransactionId.delete(planningTransactionId);
      }
    }
    this.#recordPlanningOutcome(receipt);
    this.#recordNavigationTransitClearance(receipt);
    if (this.#receiptSink) this.#uncommittedReceiptIds.add(transactionId);
    return structuredClone(receipt);
  }

  #recordPlanningOutcome(receipt: HumanoidActionReceipt): void {
    if (receipt.agentId !== "humanoid-motion-reference"
      || !isPlanningAction(receipt.action)) {
      return;
    }
    if (receipt.accepted) {
      this.#planningFailureByFingerprint.clear();
      this.#latestPlanningFailureKeyByAgent.clear();
      return;
    }
    const physicalFingerprint = planningFailureFingerprint(
      receipt.action,
      receipt.agentId,
      receipt.input,
      receipt.fingerprint
    );
    const key = planningFailureKey(receipt.agentId, physicalFingerprint);
    const previous = this.#planningFailureByFingerprint.get(key);
    const repeated = previous?.action === receipt.action
      && previous.physicalExecutionRevision === this.#latestPhysicalExecutionRevision;
    this.#planningFailureByFingerprint.set(key, {
      action: receipt.action,
      fingerprint: physicalFingerprint,
      physicalExecutionRevision: this.#latestPhysicalExecutionRevision,
      count: repeated ? previous.count + 1 : 1,
      lastCode: receipt.code === "repeated_planning_failure" && previous
        ? previous.lastCode
        : receipt.code
    });
    this.#latestPlanningFailureKeyByAgent.set(receipt.agentId, key);
  }

  #recordNavigationTransitClearance(receipt: HumanoidActionReceipt): void {
    const motionAgentId = "humanoid-motion-reference";
    if (receipt.accepted
      && (receipt.action === "execute_humanoid_skill"
        || receipt.action === "execute_whole_body_motion"
        || receipt.action === "execute_humanoid_navigation")) {
      this.#navigationTransitClearanceRequirementByAgent.delete(motionAgentId);
      return;
    }
    if (receipt.agentId !== motionAgentId
      || receipt.action !== "plan_humanoid_navigation"
      || receipt.accepted) return;
    const detail = jsonObject(receipt.detail);
    const requirement = navigationTransitClearanceFromRejection({
      reason: detail?.reason,
      transactionId: receipt.transactionId,
      worldRevision: receipt.worldAfterRevision,
      snapshot: this.#world.snapshot()
    });
    if (requirement) {
      this.#navigationTransitClearanceRequirementByAgent.set(
        motionAgentId,
        requirement
      );
    }
  }

  #renewObservationAuthority(agentId: string): HumanoidWorldObservation | undefined {
    const observed = this.#observationByAgent.get(agentId);
    if (!observed) return undefined;
    const currentRevision = this.#world.snapshot().worldRevision;
    if (observed.worldRevision === currentRevision) return observed;
    if (observed.worldRevision < this.#latestPhysicalExecutionRevision) {
      return undefined;
    }
    const current = this.#world.observe();
    if (!humanoidSkillObservationCompatible(observed, current)) return undefined;
    this.#observationRevisionByAgent.set(agentId, current.worldRevision);
    this.#observationByAgent.set(agentId, current);
    return current;
  }

  #renewPlanningSkillAuthority(agentId: string):
    | { accepted: true; binding: ActiveHumanoidSkillBinding }
    | { accepted: false; code: string; detail: JsonValue } {
    const previous = this.#activeSkillByAgent.get(agentId);
    if (!previous) {
      return {
        accepted: false,
        code: "active_skill_required",
        detail: jsonValue({
          automatic_actuation: false,
          recovery: "Observe, submit a local Skill DAG, and bind one model-selected Skill phase"
        })
      };
    }
    const observation = this.#renewObservationAuthority(agentId);
    if (!observation) {
      return {
        accepted: false,
        code: "skill_observation_changed",
        detail: jsonValue({
          automatic_actuation: false,
          skill: previous.invocation.skill,
          observed_world_revision: previous.observed_world_revision,
          current_world_revision: this.#world.snapshot().worldRevision,
          recovery: "The physical state changed materially; observe and submit a new local Skill DAG"
        })
      };
    }
    if (previous.observed_world_revision === observation.worldRevision) {
      return { accepted: true, binding: previous };
    }
    const rebound = bindHumanoidSkill({
      transactionId: previous.transaction_id,
      agentId,
      request: {
        skill_plan_transaction_id: previous.skill_plan_transaction_id,
        skill_node_id: previous.skill_node_id,
        invocation: previous.invocation,
        phase: previous.phase
      },
      observation
    });
    if (!rebound.accepted) return rebound;
    this.#activeSkillByAgent.set(agentId, rebound.binding);
    const planTransactionId = rebound.binding.skill_plan_transaction_id;
    const plan = planTransactionId
      ? this.#skillPlansByTransactionId.get(planTransactionId)
      : undefined;
    if (plan) {
      this.#skillPlansByTransactionId.set(plan.transaction_id, {
        ...plan,
        world_revision: observation.worldRevision
      });
    }
    return { accepted: true, binding: rebound.binding };
  }

  #planningSkillDetail(agentId: string): Record<string, JsonValue> {
    const binding = this.#activeSkillByAgent.get(agentId);
    return binding ? { skill_binding: jsonValue(binding) } : {};
  }

  #activeSkillPlan(agentId: string): RegisteredHumanoidSkillPlan | null {
    const transactionId = this.#activeSkillPlanTransactionByAgent.get(agentId);
    const plan = transactionId
      ? this.#skillPlansByTransactionId.get(transactionId)
      : undefined;
    return plan ? structuredClone(plan) : null;
  }

  async #ensureReceiptCommitted(receipt: HumanoidActionReceipt): Promise<void> {
    const receiptSink = this.#receiptSink;
    if (!receiptSink || !this.#uncommittedReceiptIds.has(receipt.transactionId)) return;
    let commit = this.#receiptCommits.get(receipt.transactionId);
    if (!commit) {
      commit = Promise.resolve()
        .then(() => receiptSink(structuredClone(receipt)))
        .then(() => {
          this.#uncommittedReceiptIds.delete(receipt.transactionId);
        })
        .finally(() => {
          this.#receiptCommits.delete(receipt.transactionId);
        });
      this.#receiptCommits.set(receipt.transactionId, commit);
    }
    await commit;
  }

  async #execute(
    name: HumanoidActionName,
    rawInput: unknown,
    invocation: {
      transactionId: string;
      agentId: string;
      fingerprint: string;
    }
  ): Promise<{
    accepted: boolean;
    code: string;
    channels: HumanoidBodyChannel[];
    detail: unknown;
    causalFrameCount?: number;
    causalWorldAfterRevision?: number;
  }> {
    if (isPlanningAction(name)
      && invocation.agentId === "humanoid-motion-reference") {
      const observedRevision = this.#observationRevisionByAgent.get(
        invocation.agentId
      );
      if (observedRevision === undefined
        || observedRevision < this.#latestPhysicalExecutionRevision) {
        return {
          accepted: false,
          code: "fresh_motion_observation_required",
          channels: [],
          detail: {
            automatic_actuation: false,
            motion_agent_id: invocation.agentId,
            observed_world_revision: observedRevision ?? null,
            latest_physical_execution_revision:
              this.#latestPhysicalExecutionRevision,
            recovery: "Call observe_humanoid from the Motion Agent before planning. The Harness will not share another Agent's observation or generate motion parameters."
          }
        };
      }
      if (this.#requireSkillBinding && name !== "plan_whole_body_motion") {
        const renewed = this.#renewPlanningSkillAuthority(invocation.agentId);
        if (!renewed.accepted) {
          return {
            accepted: false,
            code: renewed.code,
            channels: [],
            detail: renewed.detail
          };
        }
        const skillReference = validateSkillPlanningReference({
          binding: renewed.binding,
          action: name,
          rawInput,
          currentWorldRevision: this.#world.snapshot().worldRevision
        });
        if (!skillReference.accepted) {
          return {
            accepted: false,
            code: skillReference.code,
            channels: [],
            detail: skillReference.detail
          };
        }
      }
      this.#planningAttemptRevisionByAgent.set(
        invocation.agentId,
        this.#world.snapshot().worldRevision
      );
      const repeatedFailure = this.#planningFailureByFingerprint.get(
        planningFailureKey(
          invocation.agentId,
          planningFailureFingerprint(
            name,
            invocation.agentId,
            rawInput,
            invocation.fingerprint
          )
        )
      );
      if (repeatedFailure
        && repeatedFailure.physicalExecutionRevision
          === this.#latestPhysicalExecutionRevision
        && repeatedFailure.count >= 2) {
        return {
          accepted: false,
          code: "repeated_planning_failure",
          channels: [],
          detail: {
            automatic_actuation: false,
            repeated_action: name,
            repeated_failure_count: repeatedFailure.count,
            previous_code: repeatedFailure.lastCode,
            physical_execution_revision: this.#latestPhysicalExecutionRevision,
            recovery: "This exact action and complete physical input already failed repeatedly without an intervening execution. Choose materially different physical parameters or another available planning tool; changing only labels or identifiers is not a new strategy."
          }
        };
      }
    }
    if (name === "observe_humanoid") {
      HumanoidActionInputs.observe_humanoid.parse(rawInput);
      const observation = invocation.agentId === "humanoid-motion-reference"
        ? await this.#world.observeManipulationReachability()
        : this.#world.observe();
      this.#observationRevisionByAgent.set(
        invocation.agentId,
        observation.worldRevision
      );
      this.#observationByAgent.set(invocation.agentId, observation);
      this.#activeSkillByAgent.delete(invocation.agentId);
      if (invocation.agentId === "humanoid-motion-reference") {
        const requirement = manipulationBasePlacementRequirement(observation);
        if (requirement) {
          this.#manipulationBasePlacementRequirementByAgent.set(
            invocation.agentId,
            requirement
          );
        } else {
          this.#manipulationBasePlacementRequirementByAgent.delete(
            invocation.agentId
          );
        }
      }
      return {
        accepted: true,
        code: "humanoid_observed",
        channels: [],
        detail: modelObservation(observation)
      };
    }
    if (name === "submit_humanoid_skill_plan") {
      const proposal = HumanoidActionInputs.submit_humanoid_skill_plan.parse(rawInput);
      const observation = this.#renewObservationAuthority(invocation.agentId);
      const currentWorldRevision = this.#world.snapshot().worldRevision;
      if (!observation || observation.worldRevision !== currentWorldRevision) {
        return {
          accepted: false,
          code: "fresh_skill_observation_required",
          channels: [],
          detail: {
            automatic_actuation: false,
            observed_world_revision: observation?.worldRevision ?? null,
            current_world_revision: currentWorldRevision,
            recovery: "Call observe_humanoid before submitting a local Skill DAG"
          }
        };
      }
      const previous = this.#activeSkillPlanTransactionByAgent.get(invocation.agentId);
      if (previous) this.#skillPlansByTransactionId.delete(previous);
      const plan = registerHumanoidSkillPlan({
        transactionId: invocation.transactionId,
        agentId: invocation.agentId,
        proposal,
        observedFrame: observation.frame,
        worldRevision: observation.worldRevision
      });
      this.#skillPlansByTransactionId.set(invocation.transactionId, plan);
      this.#activeSkillPlanTransactionByAgent.set(
        invocation.agentId,
        invocation.transactionId
      );
      this.#activeSkillByAgent.delete(invocation.agentId);
      this.#latestPlanningFailureKeyByAgent.delete(invocation.agentId);
      return {
        accepted: true,
        code: "humanoid_skill_plan_registered",
        channels: [],
        detail: {
          automatic_actuation: false,
          skill_plan: plan
        }
      };
    }
    if (name === "begin_humanoid_skill") {
      const request = HumanoidActionInputs.begin_humanoid_skill.parse(rawInput);
      const observation = this.#renewObservationAuthority(invocation.agentId);
      const currentWorldRevision = this.#world.snapshot().worldRevision;
      if (!observation || observation.worldRevision !== currentWorldRevision) {
        return {
          accepted: false,
          code: "fresh_skill_observation_required",
          channels: [],
          detail: {
            automatic_actuation: false,
            observed_world_revision: observation?.worldRevision ?? null,
            current_world_revision: currentWorldRevision,
            recovery: "Call observe_humanoid before binding a skill phase"
          }
        };
      }
      const planTransactionId = request.skill_plan_transaction_id;
      let plan = planTransactionId
        ? this.#skillPlansByTransactionId.get(planTransactionId)
        : undefined;
      if (plan && plan.agent_id === invocation.agentId
        && plan.world_revision !== currentWorldRevision) {
        plan = { ...plan, world_revision: currentWorldRevision };
        this.#skillPlansByTransactionId.set(plan.transaction_id, plan);
      }
      if (this.#requireSkillBinding || planTransactionId || request.skill_node_id) {
        const nodeAuthority = authorizeHumanoidSkillPlanNode({
          plan,
          planTransactionId,
          nodeId: request.skill_node_id,
          invocation: request.invocation,
          agentId: invocation.agentId,
          currentWorldRevision
        });
        if (!nodeAuthority.accepted) {
          return {
            accepted: false,
            code: nodeAuthority.code,
            channels: [],
            detail: nodeAuthority.detail
          };
        }
      }
      const result = bindHumanoidSkill({
        transactionId: invocation.transactionId,
        agentId: invocation.agentId,
        request,
        observation
      });
      if (!result.accepted) {
        return {
          accepted: false,
          code: result.code,
          channels: [],
          detail: result.detail
        };
      }
      const recovery = this.#recoveryPolicyByAgent.get(invocation.agentId);
      if (recovery
        && !humanoidRecoverySelectionAccepted(recovery, result.binding.invocation)) {
        return {
          accepted: false,
          code: "recovery_skill_selection_required",
          channels: [],
          detail: {
            automatic_actuation: false,
            recovery_policy: recovery,
            selected_skill: result.binding.invocation.skill,
            reason: "The selected skill is not an authorized recovery entry for the last physical failure"
          }
        };
      }
      this.#activeSkillByAgent.set(invocation.agentId, result.binding);
      return {
        accepted: true,
        code: "humanoid_skill_bound",
        channels: [],
        detail: {
          automatic_actuation: false,
          binding: result.binding,
          recovery_policy: recovery ?? null
        }
      };
    }
    if (name === "plan_humanoid_skill") {
      HumanoidActionInputs.plan_humanoid_skill.parse(rawInput);
      const binding = this.#activeSkillByAgent.get(invocation.agentId);
      const observation = this.#observationByAgent.get(invocation.agentId);
      if (!binding || !observation) {
        return {
          accepted: false,
          code: "autonomous_skill_authority_missing",
          channels: [],
          detail: {
            automatic_actuation: false,
            reason: "A current Skill binding and physical observation are required"
          }
        };
      }
      let plan: ReturnType<typeof planAutonomousHumanoidSkill>;
      try {
        plan = planAutonomousHumanoidSkill({ binding, observation });
      } catch (error) {
        return {
          accepted: false,
          code: "autonomous_skill_solver_failed",
          channels: [],
          detail: {
            ...this.#planningSkillDetail(invocation.agentId),
            automatic_actuation: false,
            failure_class: "semantic_or_geometric_infeasibility",
            reason: error instanceof Error ? error.message : String(error)
          }
        };
      }
      if (plan.kind === "motion") {
        const result = await this.#world.planWholeBodyMotionCandidates(plan.batch);
        if (result.accepted) this.#planChannels.set(result.planId, result.channels);
        return {
          accepted: result.accepted,
          code: result.accepted
            ? "autonomous_skill_motion_validated"
            : "autonomous_skill_motion_rejected",
          channels: result.channels,
          detail: {
            ...this.#planningSkillDetail(invocation.agentId),
            automatic_actuation: false,
            autonomous_plan_kind: "motion",
            plan_id: result.planId,
            created_revision: result.createdRevision,
            expires_revision: result.expiresRevision,
            intent_sha256: result.intentSha256,
            selection: result.selection,
            selected_candidate_id: result.selectedCandidateId,
            selected_rank: result.selectedRank,
            candidate_count: result.candidates.length,
            option: result.option,
            motion: result.motion,
            candidates: result.candidates.map((candidate) => ({
              rank: candidate.rank,
              selected: candidate.planId === result.selectedCandidateId,
              intent: candidate.intent,
              validation: {
                feasible: candidate.validation.feasible,
                failures: candidate.validation.failures,
                evidence: candidate.validation.evidence
              }
            }))
          }
        };
      }
      const attempts: Array<{
        target: Vec3;
        score: number;
        accepted: boolean;
        reason: string | null;
      }> = [];
      let selected: Awaited<ReturnType<HumanoidWorld["planNavigation"]>> | null = null;
      for (const candidate of plan.targets.slice(0, 8)) {
        const result = await this.#world.planNavigation(
          candidate.target,
          candidate.arrivalHeading
        );
        attempts.push({
          target: { ...candidate.target },
          score: candidate.score,
          accepted: result.accepted,
          reason: result.reason ?? null
        });
        if (result.accepted) {
          selected = result;
          break;
        }
      }
      if (!selected) {
        return {
          accepted: false,
          code: "autonomous_skill_route_rejected",
          channels: ["locomotion"],
          detail: {
            ...this.#planningSkillDetail(invocation.agentId),
            automatic_actuation: false,
            autonomous_plan_kind: "navigation",
            failure_class: "path_or_physical_preview_infeasible",
            attempts
          }
        };
      }
      this.#planChannels.set(selected.planId, ["locomotion"]);
      return {
        accepted: true,
        code: "autonomous_skill_route_validated",
        channels: ["locomotion"],
        detail: {
          ...this.#planningSkillDetail(invocation.agentId),
          automatic_actuation: false,
          autonomous_plan_kind: "navigation",
          plan_id: selected.planId,
          created_revision: selected.createdRevision,
          expires_revision: selected.expiresRevision,
          intent_sha256: selected.intentSha256,
          target: selected.target,
          chunk_target: selected.chunkTarget,
          arrival_heading: selected.arrivalHeading,
          waypoints: selected.waypoints,
          distance: selected.distance,
          remaining_distance: selected.remainingDistance,
          attempts
        }
      };
    }
    if (name === "execute_humanoid_skill") {
      const input = HumanoidActionInputs.execute_humanoid_skill.parse(rawInput);
      const reference = this.#planningReference(
        input.planning_transaction_id,
        ["plan_humanoid_skill"]
      );
      if (!reference.accepted) return reference.result;
      const planningReceipt = this.#receipts.get(input.planning_transaction_id);
      const planKind = planningReceipt
        ? jsonObject(planningReceipt.detail)?.autonomous_plan_kind
        : undefined;
      if (planKind !== "motion" && planKind !== "navigation") {
        return {
          accepted: false,
          code: "autonomous_skill_plan_kind_invalid",
          channels: [],
          detail: {
            planning_transaction_id: input.planning_transaction_id,
            automatic_actuation: false
          }
        };
      }
      const channels = this.#planChannels.get(reference.planId)
        ?? (planKind === "navigation" ? ["locomotion"] : []);
      if (!this.#world.consumablePlanIds().includes(reference.planId)) {
        this.#planChannels.delete(reference.planId);
        return {
          accepted: false,
          code: "plan_stale",
          channels,
          detail: {
            planning_transaction_id: input.planning_transaction_id,
            plan_id: reference.planId,
            autonomous_plan_kind: planKind,
            automatic_actuation: false
          }
        };
      }
      await this.#beforePhysicalExecution?.({
        transactionId: invocation.transactionId,
        agentId: invocation.agentId,
        action: name,
        fingerprint: invocation.fingerprint,
        planningTransactionId: input.planning_transaction_id,
        planId: reference.planId
      });
      const options = {
        realtime: this.#realtimeExecution,
        retainTerminal: this.#retainPhysicalTerminals,
        ...(this.#physicalFrameSink
          ? { persistenceSink: this.#physicalFrameSink }
          : {}),
        ...(this.#signal ? { signal: this.#signal } : {})
      };
      const result = planKind === "motion"
        ? await this.#world.executeWholeBodyMotion(
            reference.planId,
            this.#frameSink,
            options
          )
        : await this.#world.executeNavigation(
            reference.planId,
            this.#frameSink,
            options
          );
      if (!this.#retainPhysicalTerminals) this.#planChannels.delete(reference.planId);
      return {
        accepted: result.accepted,
        code: result.code,
        channels,
        causalFrameCount: result.frames,
        causalWorldAfterRevision: result.finalSnapshot.worldRevision,
        detail: {
          planning_transaction_id: input.planning_transaction_id,
          planning_action: "plan_humanoid_skill",
          ...reference.candidateSelection,
          plan_id: reference.planId,
          autonomous_plan_kind: planKind,
          frames: result.frames,
          ...(result.terminalResultSha256
            ? { terminal_result_sha256: result.terminalResultSha256 }
            : {}),
          result: result.detail,
          final: conciseRobot(result.finalSnapshot.robot)
        }
      };
    }
    if (name === "plan_whole_body_motion") {
      const plan = HumanoidActionInputs.plan_whole_body_motion.parse(rawInput);
      const transitClearance = this.#navigationTransitClearanceRequirementByAgent.get(
        invocation.agentId
      );
      const clearanceRejection = transitClearance
        ? navigationTransitClearanceMotionRejection([plan], transitClearance)
        : null;
      if (clearanceRejection) return clearanceRejection;
      const placementRequirement = this.#manipulationBasePlacementRequirementByAgent.get(
        invocation.agentId
      );
      const placementRejection = placementRequirement
        ? manipulationBasePlacementMotionRejection([plan], placementRequirement)
        : null;
      if (placementRejection) return placementRejection;
      const result = await this.#world.planWholeBodyMotion(plan, {
        retainTerminalJointTracking: transitClearance !== undefined
      });
      if (result.accepted) this.#planChannels.set(result.planId, result.channels);
      return {
        accepted: result.accepted,
        code: result.accepted ? "whole_body_plan_validated" : "whole_body_plan_rejected",
        channels: result.channels,
        detail: {
          ...this.#planningSkillDetail(invocation.agentId),
          plan_id: result.planId,
          created_revision: result.createdRevision,
          expires_revision: result.expiresRevision,
          intent_sha256: result.intentSha256,
          motion: result.motion,
          validation: {
            feasible: result.validation.feasible,
            failures: result.validation.failures,
            evidence: result.validation.evidence,
            predicted_final: conciseRobot(result.validation.finalSnapshot)
          }
        }
      };
    }
    if (name === "plan_whole_body_motion_candidates") {
      const batch = normalizeHumanoidMotionCandidateBatchInput(
        HumanoidActionInputs.plan_whole_body_motion_candidates.parse(rawInput)
      );
      const transitClearance = this.#navigationTransitClearanceRequirementByAgent.get(
        invocation.agentId
      );
      const clearanceRejection = transitClearance
        ? navigationTransitClearanceMotionRejection(
            batch.candidates,
            transitClearance
          )
        : null;
      if (clearanceRejection) return clearanceRejection;
      const placementRequirement = this.#manipulationBasePlacementRequirementByAgent.get(
        invocation.agentId
      );
      const placementRejection = placementRequirement
        ? manipulationBasePlacementMotionRejection(
            batch.candidates,
            placementRequirement
          )
        : null;
      if (placementRejection) return placementRejection;
      const result = await this.#world.planWholeBodyMotionCandidates(batch, {
        retainTerminalJointTracking: transitClearance !== undefined
      });
      if (result.accepted) this.#planChannels.set(result.planId, result.channels);
      return {
        accepted: result.accepted,
        code: result.accepted
          ? "whole_body_candidates_validated"
          : "whole_body_candidates_rejected",
        channels: result.channels,
        detail: {
          ...this.#planningSkillDetail(invocation.agentId),
          plan_id: result.planId,
          objective: batch.objective,
          created_revision: result.createdRevision,
          expires_revision: result.expiresRevision,
          intent_sha256: result.intentSha256,
          selection: result.selection,
          selected_candidate_id: result.selectedCandidateId,
          selected_rank: result.selectedRank,
          candidate_count: result.candidates.length,
          termination: batch.termination,
          option: result.option,
          motion: result.motion,
          candidates: result.candidates.map((candidate) => ({
            rank: candidate.rank,
            plan_id: candidate.planId,
            intent: candidate.intent,
            intent_sha256: candidate.intentSha256,
            selected: candidate.planId === result.selectedCandidateId,
            channels: candidate.channels,
            motion: candidate.motion,
            option_certificate: candidate.optionCertificate,
            validation: {
              feasible: candidate.validation.feasible,
              failures: candidate.validation.failures,
              evidence: candidate.validation.evidence,
              predicted_final: conciseRobot(candidate.validation.finalSnapshot)
            }
          }))
        }
      };
    }
    if (name === "execute_whole_body_motion") {
      const input = HumanoidActionInputs.execute_whole_body_motion.parse(rawInput);
      const reference = this.#planningReference(
        input.planning_transaction_id,
        ["plan_whole_body_motion", "plan_whole_body_motion_candidates"]
      );
      if (!reference.accepted) return reference.result;
      const channels = this.#planChannels.get(reference.planId) ?? [];
      if (!this.#world.consumablePlanIds().includes(reference.planId)) {
        this.#planChannels.delete(reference.planId);
        return {
          accepted: false,
          code: "plan_stale",
          channels,
          detail: {
            planning_transaction_id: input.planning_transaction_id,
            planning_action: reference.planningAction,
            ...reference.candidateSelection,
            plan_id: reference.planId,
            automatic_actuation: false,
            reason: "validated plan is no longer consumable at the current world revision"
          }
        };
      }
      await this.#beforePhysicalExecution?.({
        transactionId: invocation.transactionId,
        agentId: invocation.agentId,
        action: name,
        fingerprint: invocation.fingerprint,
        planningTransactionId: input.planning_transaction_id,
        planId: reference.planId
      });
      const result = await this.#world.executeWholeBodyMotion(
        reference.planId,
        this.#frameSink,
        {
          realtime: this.#realtimeExecution,
          retainTerminal: this.#retainPhysicalTerminals,
          ...(this.#physicalFrameSink
            ? { persistenceSink: this.#physicalFrameSink }
            : {}),
          ...(this.#signal ? { signal: this.#signal } : {})
        }
      );
      if (!this.#retainPhysicalTerminals) this.#planChannels.delete(reference.planId);
      return {
        accepted: result.accepted,
        code: result.code,
        channels,
        causalFrameCount: result.frames,
        causalWorldAfterRevision: result.finalSnapshot.worldRevision,
        detail: {
          planning_transaction_id: input.planning_transaction_id,
          planning_action: reference.planningAction,
          ...reference.candidateSelection,
          plan_id: reference.planId,
          frames: result.frames,
          ...(result.terminalResultSha256
            ? { terminal_result_sha256: result.terminalResultSha256 }
            : {}),
          result: result.detail,
          final: conciseRobot(result.finalSnapshot.robot)
        }
      };
    }
    if (name === "plan_humanoid_navigation") {
      const input = HumanoidActionInputs.plan_humanoid_navigation.parse(rawInput);
      const transitClearance = this.#navigationTransitClearanceRequirementByAgent.get(
        invocation.agentId
      );
      if (transitClearance) {
        return {
          accepted: false,
          code: "navigation_transit_clearance_required",
          channels: [],
          detail: {
            ...navigationTransitClearanceContext(transitClearance) as Record<string, JsonValue>,
            recovery: "Plan, execute, and observe a model-selected collision-side arm-clearance posture before requesting another navigation preview."
          }
        };
      }
      const placementRequirement = this.#manipulationBasePlacementRequirementByAgent.get(
        invocation.agentId
      );
      const placementRejection = placementRequirement
        ? manipulationBasePlacementNavigationRejection(input, placementRequirement)
        : null;
      if (placementRejection) return placementRejection;
      const result = await this.#world.planNavigation(
        input.target,
        input.arrival_heading
      );
      if (result.accepted) this.#planChannels.set(result.planId, ["locomotion"]);
      return {
        accepted: result.accepted,
        code: result.accepted ? "humanoid_route_validated" : "humanoid_route_rejected",
        channels: ["locomotion"],
        detail: {
          ...this.#planningSkillDetail(invocation.agentId),
          plan_id: result.planId,
          created_revision: result.createdRevision,
          expires_revision: result.expiresRevision,
          intent_sha256: result.intentSha256,
          target: result.target,
          chunk_target: result.chunkTarget,
          requested_arrival_heading: result.requestedArrivalHeading,
          arrival_heading: result.arrivalHeading,
          waypoints: result.waypoints,
          distance: result.distance,
          remaining_distance: result.remainingDistance,
          partial_endpoint: result.partialEndpoint ?? null,
          preview_frames: result.previewFrames ?? null,
          preview_travelled_m: result.previewTravelledDistance ?? null,
          carry: result.carry,
          ...(!result.accepted && placementRequirement
            ? {
                reachable_base_placements: reachableBasePlacementContext(
                  placementRequirement.objects
                )
              }
            : {}),
          ...(result.reason ? { reason: result.reason } : {})
        }
      };
    }
    if (name === "remove_world_block") {
      const input = HumanoidActionInputs.remove_world_block.parse(rawInput);
      if (!this.#prepareBlockRemoval) {
        throw new Error("Block-removal authority is unavailable");
      }
      try {
        const transaction = await this.#prepareBlockRemoval({
          transactionId: invocation.transactionId,
          agentId: invocation.agentId,
          solidId: input.solid_id,
          executionTransactionId: input.execution_transaction_id
        });
        return {
          accepted: true,
          code: "world_block_removal_authorized",
          channels: [],
          detail: {
            solid_id: input.solid_id,
            execution_transaction_id: input.execution_transaction_id,
            automatic_actuation: false,
            removal_transaction: transaction
          }
        };
      } catch (error) {
        if (!(error instanceof BlockRemovalAuthorityError)) throw error;
        return {
          accepted: false,
          code: error.code,
          channels: [],
          detail: {
            solid_id: input.solid_id,
            execution_transaction_id: input.execution_transaction_id,
            automatic_actuation: false,
            reason: error.message
          }
        };
      }
    }
    const input = HumanoidActionInputs.execute_humanoid_navigation.parse(rawInput);
    const reference = this.#planningReference(
      input.planning_transaction_id,
      ["plan_humanoid_navigation"]
    );
    if (!reference.accepted) return reference.result;
    if (!this.#world.consumablePlanIds().includes(reference.planId)) {
      this.#planChannels.delete(reference.planId);
      return {
        accepted: false,
        code: "plan_stale",
        channels: ["locomotion"],
        detail: {
          planning_transaction_id: input.planning_transaction_id,
          planning_action: reference.planningAction,
          plan_id: reference.planId,
          automatic_actuation: false,
          reason: "validated route is no longer consumable at the current world revision"
        }
      };
    }
    await this.#beforePhysicalExecution?.({
      transactionId: invocation.transactionId,
      agentId: invocation.agentId,
      action: name,
      fingerprint: invocation.fingerprint,
      planningTransactionId: input.planning_transaction_id,
      planId: reference.planId
    });
    const result = await this.#world.executeNavigation(
      reference.planId,
      this.#frameSink,
      {
        realtime: this.#realtimeExecution,
        retainTerminal: this.#retainPhysicalTerminals,
        ...(this.#physicalFrameSink
          ? { persistenceSink: this.#physicalFrameSink }
          : {}),
        ...(this.#signal ? { signal: this.#signal } : {})
      }
    );
    if (!this.#retainPhysicalTerminals) this.#planChannels.delete(reference.planId);
    return {
      accepted: result.accepted,
      code: result.code,
      channels: ["locomotion"],
      causalFrameCount: result.frames,
      causalWorldAfterRevision: result.finalSnapshot.worldRevision,
      detail: {
        planning_transaction_id: input.planning_transaction_id,
        planning_action: reference.planningAction,
        plan_id: reference.planId,
        frames: result.frames,
        ...(result.terminalResultSha256
          ? { terminal_result_sha256: result.terminalResultSha256 }
          : {}),
        result: result.detail,
        final: conciseRobot(result.finalSnapshot.robot)
      }
    };
  }

  #planningReference(
    transactionId: string,
    expectedActions: readonly HumanoidPlanningActionName[]
  ): {
    accepted: true;
    planId: string;
    planningAction: HumanoidPlanningActionName;
    candidateSelection: {
      candidate_count: number;
      selected_rank: number;
      selected_candidate_id: string;
    } | undefined;
  } | {
    accepted: false;
    result: {
      accepted: false;
      code: string;
      channels: HumanoidBodyChannel[];
      detail: JsonValue;
    };
  } {
    const receipt = this.#receipts.get(transactionId);
    if (!receipt) {
      return rejectedPlanningReference(
        "planning_receipt_missing",
        transactionId,
        expectedActions
      );
    }
    if (!expectedActions.includes(receipt.action as HumanoidPlanningActionName)) {
      return rejectedPlanningReference(
        "planning_receipt_action_mismatch",
        transactionId,
        expectedActions,
        receipt.action
      );
    }
    if (!receipt.accepted) {
      return rejectedPlanningReference(
        "planning_receipt_rejected",
        transactionId,
        expectedActions,
        receipt.action
      );
    }
    if (this.#requireSkillBinding
      && receipt.action !== "plan_whole_body_motion"
      && !this.#skillByPlanningTransactionId.has(transactionId)) {
      return rejectedPlanningReference(
        "planning_skill_authority_missing",
        transactionId,
        expectedActions,
        receipt.action
      );
    }
    const planId = jsonObject(receipt.detail)?.plan_id;
    if (typeof planId !== "string" || !planId) {
      return rejectedPlanningReference(
        "planning_receipt_missing_plan",
        transactionId,
        expectedActions,
        receipt.action
      );
    }
    const detail = jsonObject(receipt.detail);
    const candidateCount = detail?.candidate_count;
    const selectedRank = detail?.selected_rank;
    const selectedCandidateId = detail?.selected_candidate_id;
    const candidateSelection = (receipt.action === "plan_whole_body_motion_candidates"
        || receipt.action === "plan_humanoid_skill")
      && typeof candidateCount === "number"
      && Number.isSafeInteger(candidateCount)
      && typeof selectedRank === "number"
      && Number.isSafeInteger(selectedRank)
      && typeof selectedCandidateId === "string"
      && selectedCandidateId
      ? {
          candidate_count: candidateCount,
          selected_rank: selectedRank,
          selected_candidate_id: selectedCandidateId
        }
      : undefined;
    return {
      accepted: true,
      planId,
      planningAction: receipt.action as HumanoidPlanningActionName,
      candidateSelection
    };
  }
}

function isPlanningAction(action: HumanoidActionName): action is HumanoidPlanningActionName {
  return action === "plan_humanoid_skill"
    || action === "plan_whole_body_motion"
    || action === "plan_whole_body_motion_candidates"
    || action === "plan_humanoid_navigation";
}

function humanoidSkillObservationCompatible(
  observed: HumanoidWorldObservation,
  current: HumanoidWorldObservation
): boolean {
  if (observed.robot.fallen !== current.robot.fallen
    || pointDistance(observed.robot.rootPosition, current.robot.rootPosition)
      > MANIPULATION_BASE_TARGET_TOLERANCE_METERS
    || quaternionDistance(
      observed.robot.rootRotation,
      current.robot.rootRotation
    ) > SKILL_AUTHORITY_ORIENTATION_DRIFT_RADIANS) {
    return false;
  }
  const observedObjects = new Map(observed.interaction.object_world_model.objects
    .filter(({ status }) => status === "visible")
    .map((object) => [object.id, object]));
  const currentObjects = new Map(current.interaction.object_world_model.objects
    .filter(({ status }) => status === "visible")
    .map((object) => [object.id, object]));
  if (observedObjects.size !== currentObjects.size) return false;
  for (const [objectId, before] of observedObjects) {
    const after = currentObjects.get(objectId);
    if (!after
      || pointDistance(before.pose.position, after.pose.position)
        > SKILL_AUTHORITY_OBJECT_DRIFT_METERS
      || quaternionDistance(before.pose.rotation, after.pose.rotation)
        > SKILL_AUTHORITY_ORIENTATION_DRIFT_RADIANS
      || before.articulation?.joint_id !== after.articulation?.joint_id
      || nullableDistance(
        before.articulation?.position,
        after.articulation?.position
      ) > SKILL_AUTHORITY_OBJECT_DRIFT_METERS) {
      return false;
    }
  }
  if (!sameBindings(
    observed.interaction.carrying.bindings,
    current.interaction.carrying.bindings
  )) return false;
  const observedSolids = new Map(observed.solidTokens.map((solid) => [solid.id, solid]));
  const currentSolids = new Map(current.solidTokens.map((solid) => [solid.id, solid]));
  if (observedSolids.size !== currentSolids.size) return false;
  for (const [solidId, before] of observedSolids) {
    const after = currentSolids.get(solidId);
    if (!after
      || pointDistance(before.center, after.center) > 1e-9
      || pointDistance(before.size, after.size) > 1e-9) return false;
  }
  return true;
}

function pointDistance(left: Vec3, right: Vec3): number {
  return Math.hypot(left.x - right.x, left.y - right.y, left.z - right.z);
}

function quaternionDistance(
  left: { x: number; y: number; z: number; w: number },
  right: { x: number; y: number; z: number; w: number }
): number {
  const dot = Math.abs(
    left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w
  );
  return 2 * Math.acos(Math.max(-1, Math.min(1, dot)));
}

function nullableDistance(
  left: number | null | undefined,
  right: number | null | undefined
): number {
  if (left === undefined || left === null || right === undefined || right === null) {
    return left === right ? 0 : Number.POSITIVE_INFINITY;
  }
  return Math.abs(left - right);
}

function sameBindings(
  left: HumanoidWorldObservation["interaction"]["carrying"]["bindings"],
  right: HumanoidWorldObservation["interaction"]["carrying"]["bindings"]
): boolean {
  const identity = (bindings: typeof left) => bindings
    .map(({ object_id: objectId, hand }) => `${objectId}\0${hand}`)
    .sort();
  return JSON.stringify(identity(left)) === JSON.stringify(identity(right));
}

function planIdFromReceipt(receipt: HumanoidActionReceipt): string | undefined {
  const planId = jsonObject(receipt.detail)?.plan_id;
  return typeof planId === "string" && planId ? planId : undefined;
}

function modelObservation(snapshot: HumanoidWorldObservation): unknown {
  const robot = snapshot.robot;
  const rootYaw = yawFromQuaternion(robot.rootRotation);
  const exposeHandGeometry = requiresHandSurfaceGeometry(snapshot);
  const embodiedObjectIds = new Set([
    ...snapshot.objectTokens.flatMap((token) => (
      token.status === "visible" && token.observable
        ? [token.id]
        : []
    )),
    ...snapshot.grasp.assessments.flatMap((assessment) => (
      assessment.evidence.contact.status === "missing"
        ? []
        : [assessment.object_id]
    ))
  ]);
  const contactReferenceObjects = snapshot.objectTokens.filter((token) => (
    token.portable
      && token.observable
      && embodiedObjectIds.has(token.id)
      && token.position !== null
  ));
  const manipulationGeometry = exposeHandGeometry
    ? {
        objects: contactReferenceObjects.map((object) => ({
          object_id: object.id,
          object_center_world: object.position,
          object_size: object.size,
          reachable_base_placements: snapshot.manipulationBasePlacements
            .filter((entry) => entry.objectId === object.id)
            .map((entry) => ({
              hand_surface: entry.handSurface,
              root_world_target: entry.rootWorldTarget,
              root_translation_world: entry.rootTranslationWorld,
              root_yaw_radians: entry.rootYawRadians,
              wrist_world_if_surface_at_object_center: entry.wristWorldTarget,
              ik_residual_m: entry.ikResidualMeters,
              navigation_validation_required: true
            })),
          hands: Object.fromEntries((["left", "right"] as const).map((hand) => {
            const wrist = robot.links[
              hand === "left" ? "left_wrist_yaw_link" : "right_wrist_yaw_link"
            ];
            return [hand, {
              current_wrist_world: wrist.position,
              surface_center_alignments: snapshot.handSurfaces
                .filter((surface) => surface.hand === hand)
                .map((surface) => {
                  const reachability = snapshot.manipulationReachability.find((entry) => (
                    entry.objectId === object.id
                      && entry.handSurface === surface.handSurface
                  ));
                  const wristTarget = {
                    x: object.position!.x - surface.surfaceFromWristWorld.x,
                    y: object.position!.y - surface.surfaceFromWristWorld.y,
                    z: object.position!.z - surface.surfaceFromWristWorld.z
                  };
                  const delta = {
                    x: wristTarget.x - wrist.position.x,
                    y: wristTarget.y - wrist.position.y,
                    z: wristTarget.z - wrist.position.z
                  };
                  return {
                    hand_surface: surface.handSurface,
                    wrist_world_if_surface_at_object_center: wristTarget,
                    delta_from_current_wrist_world: delta,
                    distance_from_current_wrist_m: Math.hypot(
                      delta.x,
                      delta.y,
                      delta.z
                    ),
                    ik_reference_reachable:
                      reachability?.ikReferenceReachable ?? false,
                    ik_residual_m: reachability?.ikResidualMeters ?? null
                  };
                })
            }];
          }))
        }))
      }
    : null;
  return {
    frame: snapshot.frame,
    world_revision: snapshot.worldRevision,
    sensor: {
      id: "head_sensor",
      position: snapshot.sensor.position,
      rotation: snapshot.sensor.rotation,
      maximum_range: snapshot.sensor.maximumRange,
      horizontal_field_of_view: snapshot.sensor.horizontalFieldOfView,
      vertical_field_of_view: snapshot.sensor.verticalFieldOfView
    },
    root: {
      position: robot.rootPosition,
      rotation: robot.rootRotation,
      heading: {
        yaw_radians: rootYaw,
        forward_world: {
          x: Math.sin(rootYaw),
          y: 0,
          z: Math.cos(rootYaw)
        },
        left_world: {
          x: Math.cos(rootYaw),
          y: 0,
          z: -Math.sin(rootYaw)
        }
      }
    },
    fallen: robot.fallen,
    balance: robot.balance,
    feet: robot.feet,
    key_links: Object.fromEntries([
      "pelvis",
      "head_link",
      "torso_link",
      "left_ankle_roll_link",
      "right_ankle_roll_link",
      "left_wrist_yaw_link",
      "right_wrist_yaw_link"
    ].flatMap((name) => robot.links[name as keyof typeof robot.links]
      ? [[name, robot.links[name as keyof typeof robot.links]]]
      : [])),
    end_effectors: Object.fromEntries(HUMANOID_END_EFFECTORS.map((endEffector) => [
      endEffector,
      {
        world_position: humanoidEndEffectorPosition(robot, endEffector, "world"),
        pelvis_relative_position: humanoidEndEffectorPosition(robot, endEffector, "pelvis")
      }
    ])),
    manipulation_geometry: manipulationGeometry,
    spatial_belief: snapshot.spatialBelief,
    hand_coordination: snapshot.handCoordination,
    hand_surfaces: exposeHandGeometry
      ? Object.fromEntries((["left", "right"] as const).map((hand) => {
          const wrist = robot.links[
            hand === "left" ? "left_wrist_yaw_link" : "right_wrist_yaw_link"
          ];
          return [hand, {
            wrist_world_position: wrist.position,
            wrist_world_rotation: wrist.rotation,
            contact_surfaces: snapshot.handSurfaces
              .filter((surface) => surface.hand === hand)
              .map((surface) => ({
                hand_surface: surface.handSurface,
                world_position: surface.worldPosition,
                world_rotation: surface.worldRotation,
                surface_from_wrist_world: surface.surfaceFromWristWorld
              }))
          }];
        }))
      : null,
    object_tokens: snapshot.objectTokens.map((token) => ({
      id: token.id,
      role: token.role,
      kind: token.kind,
      color: token.color,
      size: token.size,
      portable: token.portable,
      status: token.status,
      state: token.state,
      authority: token.authority,
      exact: token.exact,
      observable: token.observable,
      pose: token.pose,
      observed_frame: token.observedFrame,
      observed_world_revision: token.observedWorldRevision,
      position: token.position,
      rotation: token.rotation,
      linear_velocity: token.linearVelocity,
      angular_velocity: token.angularVelocity,
      first_seen_revision: token.firstSeenRevision,
      last_seen_revision: token.lastSeenRevision,
      last_seen_frame: token.lastSeenFrame,
      observation_count: token.observationCount,
      age_revisions: token.ageRevisions,
      relation: {
        distance_to_robot: token.relation.distanceToRobot,
        bearing_radians: token.relation.bearingRadians,
        vertical_offset: token.relation.verticalOffset,
        distance_to_left_wrist: token.relation.distanceToLeftWrist,
        distance_to_right_wrist: token.relation.distanceToRightWrist
      },
      current_contacts: token.currentContacts.map((contact) => (
        "body" in contact
          ? {
              body: contact.body,
              normal_force: contact.normalForce
            }
          : {
              hand_surface: contact.handSurface,
              normal_force: contact.normalForce
            }
      ))
    })),
    solid_tokens: snapshot.solidTokens.map((token) => ({
      id: token.id,
      source_id: token.sourceId,
      kind: token.kind,
      center: token.center,
      size: token.size,
      current_contacts: token.currentContacts
    })),
    grasp: {
      contractSha256: snapshot.grasp.contractSha256,
      assessments: snapshot.grasp.assessments.filter((assessment) => (
        embodiedObjectIds.has(assessment.object_id)
      ))
    },
    interaction: snapshot.interaction,
    contacts: modelRelevantContacts(robot),
    non_foot_environment_contacts: robot.nonFootEnvironmentContacts,
    navigation: snapshot.navigation
  };
}

function modelRelevantContacts(
  robot: HumanoidWorldObservation["robot"]
): HumanoidWorldObservation["robot"]["contacts"] {
  const unsafeBodies = new Set(robot.nonFootEnvironmentContacts);
  return robot.contacts.filter((contact) => (
    contact.firstHandLink !== null
      || contact.secondHandLink !== null
      || contact.firstBody !== null && (
        unsafeBodies.has(contact.firstBody)
          || contact.firstObject !== null
          || contact.secondObject !== null
          || contact.firstSolid != null
          || contact.secondSolid != null
      )
      || contact.secondBody !== null && (
        unsafeBodies.has(contact.secondBody)
          || contact.firstObject !== null
          || contact.secondObject !== null
          || contact.firstSolid != null
          || contact.secondSolid != null
      )
  ));
}

function requiresHandSurfaceGeometry(
  snapshot: HumanoidWorldObservation
): boolean {
  if ((snapshot.interaction.manipulable_objects?.length ?? 0) > 0
    || (snapshot.interaction.carrying?.bindings.length ?? 0) > 0
    || snapshot.grasp.assessments.some((assessment) => (
      assessment.evidence.contact.status !== "missing"
    ))) {
    return true;
  }
  return snapshot.solidTokens.some((solid) => (
    solid.kind === "block"
      && Math.hypot(
        solid.center.x - snapshot.robot.rootPosition.x,
        solid.center.z - snapshot.robot.rootPosition.z
      ) <= 1.25
  ));
}

function manipulationBasePlacementRequirement(
  observation: HumanoidWorldObservation
): ManipulationBasePlacementRequirement | null {
  const objects = observation.objectTokens.flatMap((object) => {
    if (!object.portable
      || object.status !== "visible"
      || !object.observable
      || object.position === null) return [];
    const reachability = observation.manipulationReachability.filter((entry) => (
      entry.objectId === object.id
    ));
    if (reachability.length === 0
      || reachability.some((entry) => entry.ikReferenceReachable)) return [];
    const placements = observation.manipulationBasePlacements.filter((entry) => (
      entry.objectId === object.id
    ));
    return placements.length === 0
      ? []
      : [{
          objectId: object.id,
          objectCenterWorld: { ...object.position },
          placements: structuredClone(placements)
        }];
  });
  return objects.length === 0
    ? null
    : {
        observedWorldRevision: observation.worldRevision,
        objects
      };
}

function manipulationBasePlacementNavigationRejection(
  input: ReturnType<typeof HumanoidActionInputs.plan_humanoid_navigation.parse>,
  requirement: ManipulationBasePlacementRequirement
): {
  accepted: false;
  code: string;
  channels: HumanoidBodyChannel[];
  detail: JsonValue;
} | null {
  const approached = requirement.objects.filter((object) => planarDistance(
    input.target,
    object.objectCenterWorld
  ) <= MANIPULATION_APPROACH_RADIUS_METERS);
  if (approached.length === 0) return null;
  const matchingPlacement = approached.flatMap((object) => object.placements)
    .find((placement) => (
      planarDistance(input.target, placement.rootWorldTarget)
        <= MANIPULATION_BASE_TARGET_TOLERANCE_METERS
        && input.arrival_heading?.type === "yaw"
        && Math.abs(normalizeAngle(
          input.arrival_heading.yaw_radians - placement.rootYawRadians
        )) <= input.arrival_heading.tolerance_radians
    ));
  if (matchingPlacement) return null;
  return {
    accepted: false,
    code: "manipulation_base_placement_required",
    channels: ["locomotion"],
    detail: jsonValue({
      automatic_actuation: false,
      observed_world_revision: requirement.observedWorldRevision,
      requested_target: input.target,
      requested_arrival_heading: input.arrival_heading,
      reachable_base_placements: reachableBasePlacementContext(approached),
      reason: "The current manipulation observation has no directly reachable hand surface. A near-object route must preserve one model-visible IK base-placement sample and its yaw before endpoint planning."
    })
  };
}

function manipulationBasePlacementMotionRejection(
  plans: Array<ReturnType<typeof HumanoidActionInputs.plan_whole_body_motion.parse>>,
  requirement: ManipulationBasePlacementRequirement
): {
  accepted: false;
  code: string;
  channels: HumanoidBodyChannel[];
  detail: JsonValue;
} | null {
  const blockedPlans = plans.flatMap((plan) => {
    const contactedObjects = requirement.objects.filter((object) => (
      (plan.contact_constraints ?? []).some((constraint) => (
        "object_id" in constraint
          && constraint.object_id === object.objectId
      ))
    ));
    const blockedObjects = contactedObjects.filter((object) => (
      !isContactGuidedMobileManipulation(plan, object)
    ));
    return blockedObjects.length === 0
      ? []
      : [{ plan, objects: blockedObjects }];
  });
  const approached = requirement.objects.filter((object) => (
    blockedPlans.some((blocked) => blocked.objects.includes(object))
  ));
  if (approached.length === 0) return null;
  return {
    accepted: false,
    code: "manipulation_base_placement_required",
    channels: [],
    detail: manipulationBasePlacementDetail(
      requirement.observedWorldRevision,
      approached,
      {
        reason: "The requested endpoint contact has no directly reachable hand surface at the observed root pose. Use a validated base placement, or submit a contact-guided mobile manipulation candidate with non-zero root translation and an exact required hand-surface contact for the same object.",
        requested_plan_ids: blockedPlans.map(({ plan }) => plan.id)
      }
    )
  };
}

function isContactGuidedMobileManipulation(
  plan: ReturnType<typeof HumanoidActionInputs.plan_whole_body_motion.parse>,
  object: ManipulationBasePlacementRequirement["objects"][number]
): boolean {
  const hasRootTranslation = plan.keyframes.some((keyframe) => {
    const velocity = keyframe.root_velocity;
    return velocity != null
      && Math.hypot(velocity.forward_mps, velocity.lateral_mps) > 1e-6;
  });
  if (!hasRootTranslation) return false;
  const reachableSurfaces = new Set(object.placements.map((placement) => (
    placement.handSurface
  )));
  return (plan.contact_constraints ?? []).some((constraint) => (
    "hand_surface" in constraint
      && "object_id" in constraint
      && constraint.object_id === object.objectId
      && constraint.required
      && reachableSurfaces.has(constraint.hand_surface)
  ));
}

function manipulationBasePlacementDetail(
  observedWorldRevision: number,
  objects: ManipulationBasePlacementRequirement["objects"],
  extra: Record<string, unknown>
): JsonValue {
  return jsonValue({
    automatic_actuation: false,
    observed_world_revision: observedWorldRevision,
    reachable_base_placements: reachableBasePlacementContext(objects),
    ...extra
  });
}

function reachableBasePlacementContext(
  objects: ManipulationBasePlacementRequirement["objects"]
): Array<Record<string, unknown>> {
  return objects.flatMap((object) => object.placements.map((placement) => ({
    object_id: object.objectId,
    hand_surface: placement.handSurface,
    root_world_target: placement.rootWorldTarget,
    root_yaw_radians: placement.rootYawRadians,
    wrist_world_target: placement.wristWorldTarget,
    ik_residual_m: placement.ikResidualMeters,
    navigation_validation_required: true
  })));
}

function planarDistance(left: Vec3, right: Vec3): number {
  return Math.hypot(left.x - right.x, left.z - right.z);
}

function normalizeAngle(value: number): number {
  return Math.atan2(Math.sin(value), Math.cos(value));
}

function planningFailureKey(agentId: string, fingerprint: string): string {
  return `${agentId}\0${fingerprint}`;
}

function planningFailureFingerprint(
  action: HumanoidPlanningActionName,
  agentId: string,
  rawInput: unknown,
  fallback: string
): string {
  try {
    if (action === "plan_humanoid_skill") {
      return humanoidActionFingerprint(
        action,
        agentId,
        HumanoidActionInputs.plan_humanoid_skill.parse(rawInput)
      );
    }
    if (action === "plan_humanoid_navigation") {
      const navigation = HumanoidActionInputs.plan_humanoid_navigation.parse(rawInput);
      return humanoidActionFingerprint(action, agentId, {
        target: {
          x: planningFailureScalar(navigation.target.x, 0.01),
          z: planningFailureScalar(navigation.target.z, 0.01)
        },
        arrival_heading: navigation.arrival_heading === null
          ? null
          : navigation.arrival_heading.type === "yaw"
            ? {
                type: "yaw",
                yaw_radians: planningFailureScalar(
                  normalizeAngle(navigation.arrival_heading.yaw_radians),
                  0.01
                ),
                tolerance_radians: planningFailureScalar(
                  navigation.arrival_heading.tolerance_radians,
                  0.01
                )
              }
            : {
                type: "face_point",
                target: {
                  x: planningFailureScalar(
                    navigation.arrival_heading.target.x,
                    0.01
                  ),
                  z: planningFailureScalar(
                    navigation.arrival_heading.target.z,
                    0.01
                  )
                },
                tolerance_radians: planningFailureScalar(
                  navigation.arrival_heading.tolerance_radians,
                  0.01
                )
              }
      });
    }
    if (action === "plan_whole_body_motion") {
      const { id: _id, intent: _intent, ...physical } =
        HumanoidActionInputs.plan_whole_body_motion.parse(rawInput);
      return humanoidActionFingerprint(action, agentId, physical);
    }
    const batch = normalizeHumanoidMotionCandidateBatchInput(
      HumanoidActionInputs.plan_whole_body_motion_candidates.parse(rawInput)
    );
    const {
      objective: _objective,
      termination,
      candidates
    } = batch;
    const { option_id: _optionId, ...physicalTermination } = termination;
    return humanoidActionFingerprint(action, agentId, {
      termination: physicalTermination,
      candidates: candidates.map(({ id: _id, intent: _intent, ...physical }) => (
        physical
      ))
    });
  } catch {
    return fallback;
  }
}

function planningFailureScalar(value: number, resolution: number): number {
  const rounded = Math.round(value / resolution) * resolution;
  return Object.is(rounded, -0) ? 0 : rounded;
}

function conciseRobot(robot: HumanoidWorldSnapshot["robot"]): unknown {
  return {
    simulated_time: robot.simulatedTime,
    root_position: robot.rootPosition,
    fallen: robot.fallen,
    balance: robot.balance,
    feet: robot.feet,
    non_foot_environment_contacts: robot.nonFootEnvironmentContacts
  };
}

function jsonValue(value: unknown): JsonValue {
  const serialized = JSON.stringify(value);
  if (serialized === undefined) throw new Error("Humanoid action detail is not serializable");
  return JSON.parse(serialized) as JsonValue;
}

function jsonObject(value: JsonValue): Record<string, JsonValue> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value
    : undefined;
}

function rejectedPlanningReference(
  code: string,
  transactionId: string,
  expectedActions: readonly HumanoidPlanningActionName[],
  actualAction?: HumanoidActionName
): {
  accepted: false;
  result: {
    accepted: false;
    code: string;
    channels: HumanoidBodyChannel[];
    detail: JsonValue;
  };
} {
  return {
    accepted: false,
    result: {
      accepted: false,
      code,
      channels: [],
      detail: {
        planning_transaction_id: transactionId,
        expected_action: expectedActions.length === 1
          ? expectedActions[0]!
          : [...expectedActions],
        actual_action: actualAction ?? null,
        automatic_actuation: false
      }
    }
  };
}

export function humanoidActionFingerprint(
  action: HumanoidActionName,
  agentId: string,
  input: unknown
): string {
  return `${action}\n${agentId}\n${stableJson(input)}`;
}

function assertNormalizedReceiptIdentity(
  source: HumanoidActionReceipt,
  normalized: HumanoidActionReceipt
): void {
  if (normalized.transactionId !== source.transactionId
    || normalized.agentId !== source.agentId
    || normalized.action !== source.action
    || normalized.fingerprint !== source.fingerprint
    || stableJson(normalized.input) !== stableJson(source.input)) {
    throw new Error(
      `Humanoid receipt normalizer changed action identity: ${source.transactionId}`
    );
  }
}

function stableJson(value: unknown): string {
  if (value === null || typeof value === "string" || typeof value === "boolean") {
    return JSON.stringify(value);
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error("Humanoid action input must be finite JSON");
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => stableJson(item)).join(",")}]`;
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, item]) => item !== undefined)
      .sort(([left], [right]) => left.localeCompare(right));
    return `{${entries.map(([key, item]) => (
      `${JSON.stringify(key)}:${stableJson(item)}`
    )).join(",")}}`;
  }
  throw new Error("Humanoid action input must be JSON serializable");
}
