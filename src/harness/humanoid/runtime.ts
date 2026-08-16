import { z } from "zod";
import {
  HUMANOID_END_EFFECTORS,
  type Goal,
  JsonValueSchema,
  type JsonValue,
  type Vec3
} from "../../domain/schema.js";
import {
  modelPayloadSha256,
  type ModelDecisionRef
} from "../../domain/model-call-authority.js";
import type { HumanoidGroundingReceipt } from
  "../../domain/humanoid-grounding.js";
import type { AutonomousCycleRef } from "../../domain/autonomous-cycle.js";
import type { NeuralRolloutExecutionAdmission } from
  "../../domain/action-execution-ledger.js";
import type { NeuralSafetyInterrupt } from "../../domain/neural-hierarchy.js";
import type { ScenarioBlockRemovalTransaction } from "../../domain/scenario-block-removal.js";
import {
  HUMANOID_SKILL_CONTRACTS,
  type HumanoidSkillInvocation
} from "../../domain/humanoid-skill.js";
import { yawFromQuaternion } from "../../world/geometry.js";
import { humanoidEndEffectorPosition } from "../../world/humanoid/end-effectors.js";
import type { HumanoidBodyChannel } from "../../world/humanoid/motion-plan.js";
import {
  type HumanoidFrameSink,
  type HumanoidPersistenceSink,
  type HumanoidSkillEventSink,
  type HumanoidWorld,
  type HumanoidWorldObservation,
  type HumanoidWorldSnapshot
} from "../../world/humanoid/world.js";
import type { HumanoidPolicyFrameSink } from
  "../../world/humanoid/simulation.js";
import {
  HumanoidActionInputs,
  type HumanoidActionName
} from "./actions.js";
import { MAX_CHECKPOINT_ACTION_RECEIPTS } from "./embodied-memory.js";
import { HUMANOID_NEURAL_AGENT_IDS } from "./neural-hierarchy-contract.js";
import {
  humanoidRecoverySafetyInterruptIsCurrent
} from "./recovery-safety-authority.js";
import { normalizeHumanoidMotionCandidateBatchInput } from "./motion-candidate-input.js";
import { BlockRemovalAuthorityError } from "./block-removal-authority.js";
import {
  navigationTransitClearanceContext,
  navigationTransitClearanceFromRejection,
  navigationTransitClearanceMotionRejection,
  NavigationTransitClearanceRequirementSchema,
  refreshNavigationTransitClearanceRequirement,
  type NavigationTransitClearanceRequirement
} from "./navigation-transit-clearance.js";
import {
  bindHumanoidSkill,
  ActiveHumanoidSkillBindingSchema,
  humanoidEmbodiedSkillIdentity,
  navigableManipulationBasePlacements,
  validateSkillPlanningReference,
  type ActiveHumanoidSkillBinding
} from "./skill-binding.js";
import { alignHumanoidSkillToGoal } from "./goal-skill-alignment.js";
import {
  createHumanoidRecoveryPolicy,
  humanoidRecoverySelectionAccepted,
  HumanoidRecoveryPolicyStateSchema,
  type HumanoidRecoveryPolicyState
} from "./recovery-policy.js";
import {
  advanceHumanoidSkillPlan,
  authorizeHumanoidSkillPlanNode,
  readyHumanoidSkillPlanBindings,
  registerHumanoidSkillPlan,
  RegisteredHumanoidSkillPlanSchema,
  type RegisteredHumanoidSkillPlan
} from "./skill-plan.js";
import { planAutonomousHumanoidSkill } from "./autonomous-skill-planner.js";
import { planAutonomousHumanoidNavigation } from
  "./autonomous-navigation-planning.js";
import {
  executeHumanoidArticulationHorizon,
  isHumanoidArticulationActuation
} from "./articulation-horizon-executor.js";
import {
  HUMANOID_ARTICULATION_HORIZON,
  humanoidArticulationSegmentBudgetExhausted
} from "./articulation-control.js";
import {
  executeHumanoidNavigationHorizon,
  isHumanoidNavigationSkill,
  navigationPlanForHorizon
} from "./navigation-horizon-executor.js";
import {
  HUMANOID_NAVIGATION_HORIZON,
  humanoidNavigationSegmentBudgetExhausted
} from "./navigation-control.js";
import { HumanoidSkillEventStream } from
  "../../world/humanoid/skill-event-stream.js";
import {
  restoreHumanoidSkillEventStreamStates,
  type HumanoidSkillEventStreamState
} from "../../world/humanoid/skill-event-stream.js";
import type { HumanoidEmbodiedSkillEvent } from
  "../../world/humanoid/embodied-skill-call.js";

type HumanoidPlanningActionName = "plan_humanoid_skill"
  | "plan_whole_body_motion"
  | "plan_whole_body_motion_candidates"
  | "plan_humanoid_navigation";

type HumanoidPhysicalActionName = "execute_humanoid_skill"
  | "execute_whole_body_motion"
  | "execute_humanoid_navigation";

const LEGACY_HUMANOID_MOTION_AGENT_ID = "humanoid-motion-reference";

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

export const HumanoidActionRuntimeStateSchema = z.object({
  version: z.literal(1),
  /**
   * Cognitive authority boundary for mutable Skill plans, bindings, recovery
   * policy and grounding. Physical receipts remain durable across epochs, but
   * this hot cache may only be restored by the hierarchy epoch that produced
   * it. Missing values are accepted only for legacy checkpoint migration.
   */
  neural_hierarchy_epoch_id: z.string().uuid().optional(),
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
  }).strict()),
  navigation_transit_clearance_requirements: z.array(z.object({
    agent_id: z.string().trim().min(1),
    requirement: NavigationTransitClearanceRequirementSchema
  }).strict()).default([]),
  latest_grounding_observation: JsonValueSchema.nullable().default(null)
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

export type HumanoidActionRuntimeState = z.infer<
  typeof HumanoidActionRuntimeStateSchema
>;

const MANIPULATION_APPROACH_RADIUS_METERS = 1;
const MANIPULATION_BASE_TARGET_TOLERANCE_METERS = 0.04;
const SKILL_AUTHORITY_OBJECT_DRIFT_METERS = 0.015;
const SKILL_AUTHORITY_ORIENTATION_DRIFT_RADIANS = 0.05;
const MOTION_GROUNDING_OBJECT_LIMIT = 32;
const MOTION_GROUNDING_SOLID_LIMIT = 32;
const MOTION_GROUNDING_FRONTIER_LIMIT = 16;
const MOTION_GROUNDING_INTERACTION_POINT_LIMIT = 8;
const MOTION_GROUNDING_BASE_PLACEMENT_LIMIT = 8;

export interface HumanoidPhysicalExecutionIntent {
  transactionId: string;
  agentId: string;
  action: HumanoidPhysicalActionName;
  fingerprint: string;
  planningTransactionId: string;
  planId: string;
  decision?: ModelDecisionRef;
  toolAuthority?: HumanoidActionToolCallAuthority;
  neuralRolloutCertificate?: NeuralRolloutExecutionAdmission;
}

export interface HumanoidActionToolCallAuthority {
  tool_call_id: string;
  tool_name: string;
  arguments_sha256: string;
  normalized_arguments_sha256?: string | undefined;
  deterministic_delegation?: {
    contract_id: "grounding_monitor_v1" | "execution_gate_v1";
    source_input: JsonValue;
    action_input_sha256: string;
  };
}

export interface HumanoidActionInvocationOptions {
  signal?: AbortSignal;
  toolAuthority?: HumanoidActionToolCallAuthority;
  /** Internal crash recovery only: replay the decision persisted at admission. */
  recoveryDecision?: ModelDecisionRef;
  /** Harness-issued rollout authority consumed atomically with physical admission. */
  neuralRolloutCertificate?: NeuralRolloutExecutionAdmission;
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
    authority: HumanoidActionToolCallAuthority,
    options?: HumanoidActionInvocationOptions
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
  commitSequence?: number | undefined;
  committedAt: string;
}

export interface HumanoidActionRuntimeOptions {
  frameSink?: HumanoidFrameSink;
  policyFrameSink?: HumanoidPolicyFrameSink;
  physicalFrameSink?: HumanoidPersistenceSink;
  physicalPersistenceFrameStride?: number;
  physicalExecutionFrameOffset?: (transactionId: string) => number;
  physicalExecutionStartWorldRevision?: (transactionId: string) => number | undefined;
  completedPhysicalPlanFrameCount?: (transactionId: string) => number;
  completedPhysicalPlanCount?: (transactionId: string) => number;
  skillEventSink?: HumanoidSkillEventSink;
  receiptSink?: (receipt: HumanoidActionReceipt) => void | Promise<void>;
  beforePhysicalExecution?: (
    intent: HumanoidPhysicalExecutionIntent
  ) => HumanoidGroundingReceipt | undefined
    | Promise<HumanoidGroundingReceipt | undefined>;
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
  /**
   * Rebuild receipt-derived planning memory when restoring a normal runtime.
   * A fresh neural epoch disables this while retaining receipts as physical
   * and idempotency authority.
   */
  replayHistoricalCognitiveState?: boolean;
  neuralHierarchyEpochId?: string;
  realtimeExecution?: boolean;
  retainPhysicalTerminals?: boolean;
  requireSkillBinding?: boolean;
  activeGoal?: () => Goal | undefined;
  activeNeuralSkillCommitment?: () => {
    commitmentId: string;
    goalEpochId: string;
    invocation: HumanoidSkillInvocation;
  } | undefined;
  activeRecoverySafetyInterrupt?: () => NeuralSafetyInterrupt | undefined;
  signal?: AbortSignal;
}

export class HumanoidActionRuntime {
  readonly #world: HumanoidWorld;
  readonly #frameSink: HumanoidFrameSink | undefined;
  readonly #policyFrameSink: HumanoidPolicyFrameSink | undefined;
  readonly #physicalFrameSink: HumanoidPersistenceSink | undefined;
  readonly #physicalPersistenceFrameStride: number | undefined;
  readonly #physicalExecutionFrameOffset: NonNullable<
    HumanoidActionRuntimeOptions["physicalExecutionFrameOffset"]
  >;
  readonly #physicalExecutionStartWorldRevision: NonNullable<
    HumanoidActionRuntimeOptions["physicalExecutionStartWorldRevision"]
  >;
  readonly #completedPhysicalPlanFrameCount: NonNullable<
    HumanoidActionRuntimeOptions["completedPhysicalPlanFrameCount"]
  >;
  readonly #completedPhysicalPlanCount: NonNullable<
    HumanoidActionRuntimeOptions["completedPhysicalPlanCount"]
  >;
  readonly #skillEventSink: HumanoidSkillEventSink | undefined;
  readonly #receiptSink: HumanoidActionRuntimeOptions["receiptSink"];
  readonly #beforePhysicalExecution: HumanoidActionRuntimeOptions[
    "beforePhysicalExecution"
  ];
  readonly #receiptNormalizer: HumanoidActionRuntimeOptions["receiptNormalizer"];
  readonly #prepareBlockRemoval: HumanoidActionRuntimeOptions["prepareBlockRemoval"];
  readonly #realtimeExecution: boolean;
  readonly #retainPhysicalTerminals: boolean;
  readonly #requireSkillBinding: boolean;
  readonly #activeGoal: HumanoidActionRuntimeOptions["activeGoal"];
  readonly #activeNeuralSkillCommitment: HumanoidActionRuntimeOptions[
    "activeNeuralSkillCommitment"
  ];
  readonly #activeRecoverySafetyInterrupt: HumanoidActionRuntimeOptions[
    "activeRecoverySafetyInterrupt"
  ];
  readonly #neuralHierarchyEpochId: string | undefined;
  readonly #signal: AbortSignal | undefined;
  readonly #receipts = new Map<string, HumanoidActionReceipt>();
  readonly #transactions = new Map<string, {
    fingerprint: string;
    decisionSha256: string | undefined;
    toolAuthoritySha256: string | undefined;
    toolAuthority: HumanoidActionToolCallAuthority | undefined;
    neuralRolloutCertificateSha256: string | undefined;
    promise: Promise<HumanoidActionReceipt>;
  }>();
  readonly #receiptCommits = new Map<string, Promise<void>>();
  readonly #uncommittedReceiptIds = new Set<string>();
  readonly #planChannels = new Map<string, HumanoidBodyChannel[]>();
  readonly #inFlightTransactions = new Set<string>();
  readonly #observationRevisionByAgent = new Map<string, number>();
  readonly #observationByAgent = new Map<string, HumanoidWorldObservation>();
  #latestGroundingObservation: HumanoidWorldObservation | null = null;
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
  readonly #restoredSkillEventStateByCallId = new Map<
    string,
    HumanoidSkillEventStreamState
  >();
  readonly #skillEventStreamByCallId = new Map<
    string,
    HumanoidSkillEventStream
  >();
  #latestPhysicalExecutionRevision = 0;

  constructor(world: HumanoidWorld, options: HumanoidActionRuntimeOptions = {}) {
    this.#world = world;
    this.#frameSink = options.frameSink;
    this.#policyFrameSink = options.policyFrameSink;
    this.#physicalFrameSink = options.physicalFrameSink;
    this.#physicalPersistenceFrameStride = options.physicalPersistenceFrameStride;
    if (this.#physicalPersistenceFrameStride !== undefined
      && (!Number.isSafeInteger(this.#physicalPersistenceFrameStride)
        || this.#physicalPersistenceFrameStride <= 0)) {
      throw new Error("Physical persistence frame stride must be a positive integer");
    }
    this.#physicalExecutionFrameOffset = options.physicalExecutionFrameOffset
      ?? (() => 0);
    this.#physicalExecutionStartWorldRevision =
      options.physicalExecutionStartWorldRevision ?? (() => undefined);
    this.#completedPhysicalPlanFrameCount = options.completedPhysicalPlanFrameCount
      ?? (() => 0);
    this.#completedPhysicalPlanCount = options.completedPhysicalPlanCount
      ?? (() => 0);
    this.#skillEventSink = options.skillEventSink;
    this.#receiptSink = options.receiptSink;
    this.#beforePhysicalExecution = options.beforePhysicalExecution;
    this.#receiptNormalizer = options.receiptNormalizer;
    this.#prepareBlockRemoval = options.prepareBlockRemoval;
    this.#realtimeExecution = options.realtimeExecution ?? false;
    this.#retainPhysicalTerminals = options.retainPhysicalTerminals ?? false;
    this.#requireSkillBinding = options.requireSkillBinding ?? false;
    this.#activeGoal = options.activeGoal;
    this.#activeNeuralSkillCommitment = options.activeNeuralSkillCommitment;
    this.#activeRecoverySafetyInterrupt = options.activeRecoverySafetyInterrupt;
    this.#neuralHierarchyEpochId = options.neuralHierarchyEpochId;
    this.#signal = options.signal;
    if (options.state !== undefined && options.state !== null) {
      this.#restoreState(options.state);
    }
    const replayHistoricalCognitiveState =
      options.replayHistoricalCognitiveState ?? true;
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
        if (!isRecognizedPlanningReceiptActor(receipt.agentId)) {
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
      if (replayHistoricalCognitiveState) {
        this.#recordPlanningOutcome(receipt);
        this.#recordNavigationTransitClearance(receipt);
      }
      this.#transactions.set(transactionId, {
        fingerprint,
        decisionSha256: receipt.decision
          ? modelPayloadSha256(receipt.decision)
          : undefined,
        toolAuthoritySha256: undefined,
        toolAuthority: undefined,
        neuralRolloutCertificateSha256: undefined,
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
    for (const agentId of this.#navigationTransitClearanceRequirementByAgent.keys()) {
      this.#rebindNavigationTransitClearanceSkill(agentId);
    }
    this.#pruneTransactionHistory();
  }

  persistenceState(): JsonValue {
    return jsonValue(HumanoidActionRuntimeStateSchema.parse({
      version: 1,
      ...(this.#neuralHierarchyEpochId === undefined
        ? {}
        : { neural_hierarchy_epoch_id: this.#neuralHierarchyEpochId }),
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
        })),
      navigation_transit_clearance_requirements: [
        ...this.#navigationTransitClearanceRequirementByAgent
      ].map(([agentId, requirement]) => ({
        agent_id: agentId,
        requirement: structuredClone(requirement)
      })),
      latest_grounding_observation: this.#latestGroundingObservation
    }));
  }

  /**
   * Replaces the mutable action cache with state recovered from a verified
   * append-only action event. Receipts remain independently reconstructed
   * from their durable commit proofs.
   */
  recoverPersistenceState(rawState: JsonValue): void {
    this.#skillPlansByTransactionId.clear();
    this.#activeSkillPlanTransactionByAgent.clear();
    this.#activeSkillByAgent.clear();
    this.#skillByPlanningTransactionId.clear();
    this.#recoveryPolicyByAgent.clear();
    this.#navigationTransitClearanceRequirementByAgent.clear();
    this.#latestGroundingObservation = null;
    for (const agentId of humanoidGroundingAuthorityIds()) {
      this.#observationRevisionByAgent.delete(agentId);
      this.#observationByAgent.delete(agentId);
    }
    this.#latestPhysicalExecutionRevision = 0;
    this.#restoreState(rawState);
    for (const agentId of this.#navigationTransitClearanceRequirementByAgent.keys()) {
      this.#rebindNavigationTransitClearanceSkill(agentId);
    }
  }

  restoreSkillEventJournal(
    events: readonly HumanoidEmbodiedSkillEvent[]
  ): void {
    const restored = restoreHumanoidSkillEventStreamStates(events);
    this.#restoredSkillEventStateByCallId.clear();
    this.#skillEventStreamByCallId.clear();
    for (const [callId, state] of restored) {
      this.#restoredSkillEventStateByCallId.set(
        callId,
        structuredClone(state)
      );
    }
  }

  skillEventRecoveryCallIds(
    planningTransactionIds?: ReadonlySet<string>
  ): Set<string> {
    return new Set([...this.#skillByPlanningTransactionId]
      .filter(([planningTransactionId]) => (
        planningTransactionIds?.has(planningTransactionId) ?? true
      ))
      .map(([, binding]) => humanoidEmbodiedSkillIdentity(binding).callId));
  }

  #restoreState(rawState: JsonValue): void {
    const state = HumanoidActionRuntimeStateSchema.parse(rawState);
    if (state.neural_hierarchy_epoch_id !== undefined
      && this.#neuralHierarchyEpochId !== undefined
      && state.neural_hierarchy_epoch_id !== this.#neuralHierarchyEpochId) {
      throw new Error(
        "Humanoid action runtime cognitive state belongs to another neural hierarchy epoch"
      );
    }
    const currentWorldRevision = this.#world.snapshot().worldRevision;
    if (state.latest_physical_execution_revision > currentWorldRevision) {
      throw new Error(
        "Humanoid action runtime state is ahead of the authoritative world"
      );
    }
    this.#latestPhysicalExecutionRevision = state.latest_physical_execution_revision;
    const grounding = persistedGroundingObservation(
      state.latest_grounding_observation,
      currentWorldRevision,
      this.#latestPhysicalExecutionRevision
    );
    if (grounding) {
      this.#latestGroundingObservation = grounding;
      for (const agentId of humanoidGroundingAuthorityIds()) {
        this.#observationRevisionByAgent.set(agentId, grounding.worldRevision);
        this.#observationByAgent.set(agentId, structuredClone(grounding));
      }
    }
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
    for (const { agent_id: agentId, requirement } of (
      state.navigation_transit_clearance_requirements
    )) {
      if (requirement.observedWorldRevision > currentWorldRevision) {
        throw new Error(
          "Persisted navigation clearance state is ahead of the authoritative world"
        );
      }
      this.#navigationTransitClearanceRequirementByAgent.set(
        agentId,
        structuredClone(requirement)
      );
    }
  }

  snapshot(): HumanoidWorldSnapshot {
    return this.#world.snapshot();
  }

  receipt(transactionId: string): HumanoidActionReceipt | undefined {
    const receipt = this.#receipts.get(transactionId);
    return receipt ? structuredClone(receipt) : undefined;
  }

  toolCallAuthority(
    transactionId: string
  ): HumanoidActionToolCallAuthority | undefined {
    const authority = this.#transactions.get(transactionId.trim())?.toolAuthority;
    return authority ? structuredClone(authority) : undefined;
  }

  #invocationMatchesActiveNeuralCommitment(
    invocation: HumanoidSkillInvocation
  ): boolean {
    if (this.#activeNeuralSkillCommitment === undefined) return true;
    const commitment = this.#activeNeuralSkillCommitment();
    return commitment !== undefined
      && sameHumanoidSkillInvocation(invocation, commitment.invocation);
  }

  #skillPlanMatchesActiveNeuralCommitment(
    plan: RegisteredHumanoidSkillPlan | null
  ): boolean {
    if (!plan) return false;
    const ready = readyHumanoidSkillPlanBindings(plan);
    return ready.length === 1
      && this.#invocationMatchesActiveNeuralCommitment(ready[0]!.invocation);
  }

  isActionAvailable(name: HumanoidActionName, agentId: string): boolean {
    const normalizedAgentId = agentId.trim();
    if (!isCurrentHumanoidPlanningActor(normalizedAgentId)) return true;
    const transitClearance = this.#navigationTransitClearanceRequirementByAgent.has(
      normalizedAgentId
    );
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
            && this.#skillPlanMatchesActiveNeuralCommitment(activePlan)
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
              || activePlan.world_revision !== observation.worldRevision
              || !this.#skillPlanMatchesActiveNeuralCommitment(activePlan)))
          || hasCurrentPlanningFailure);
    }
    if (!isPlanningAction(name)) return true;
    if (observedRevision === undefined
      || observedRevision < this.#latestPhysicalExecutionRevision) {
      return false;
    }
    if (this.#requireSkillBinding) {
      const skill = this.#activeSkillByAgent.get(normalizedAgentId);
      const clearanceRecoveryAction = transitClearance
        && (name === "plan_whole_body_motion_candidates"
          || name === "plan_humanoid_navigation");
      if ((!skill || !this.#invocationMatchesActiveNeuralCommitment(
        skill.invocation
      )) && !clearanceRecoveryAction) return false;
      if (skill
        && (skill.observed_world_revision < this.#latestPhysicalExecutionRevision
          || (skill.planning_action !== name && !clearanceRecoveryAction))) {
        return false;
      }
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
    const neuralCommitment = this.#activeNeuralSkillCommitment?.();
    const storedSkill = this.#activeSkillByAgent.get(normalizedAgentId);
    const activeSkill = storedSkill
      && this.#invocationMatchesActiveNeuralCommitment(storedSkill.invocation)
      ? storedSkill
      : undefined;
    const storedPlan = this.#activeSkillPlan(normalizedAgentId);
    const activePlan = this.#skillPlanMatchesActiveNeuralCommitment(storedPlan)
      ? storedPlan
      : null;
    const readySkillBindings = activePlan && !activeSkill
      ? readyHumanoidSkillPlanBindings(activePlan)
      : [];
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
      skill_plan: activePlan,
      ready_skill_bindings: readySkillBindings,
      required_next_tool: readySkillBindings.length > 0
        ? "begin_humanoid_skill"
        : neuralCommitment && !activeSkill
          ? "submit_humanoid_skill_plan"
          : null,
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

  planningGroundingState(agentId: string): JsonValue {
    const observation = this.#renewObservationAuthority(agentId.trim());
    return observation
      ? jsonValue(motionPlanningObservation(observation, this.#activeGoal?.()))
      : null;
  }

  skillCommitmentObservation(
    agentId: string
  ): HumanoidWorldObservation | undefined {
    const observation = this.#renewObservationAuthority(agentId.trim());
    return observation ? structuredClone(observation) : undefined;
  }

  async invoke(
    name: HumanoidActionName,
    rawInput: unknown,
    transactionId: string,
    agentId: string,
    decision?: ModelDecisionRef,
    options: HumanoidActionInvocationOptions = {}
  ): Promise<HumanoidActionReceipt> {
    const normalizedTransactionId = transactionId.trim();
    const normalizedAgentId = agentId.trim();
    if (!normalizedTransactionId) throw new Error("Humanoid action transaction id is required");
    if (!normalizedAgentId) throw new Error("Humanoid action agent id is required");
    const fingerprint = humanoidActionFingerprint(name, normalizedAgentId, rawInput);
    const decisionSha256 = decision ? modelPayloadSha256(decision) : undefined;
    const toolAuthoritySha256 = options.toolAuthority
      ? modelPayloadSha256(options.toolAuthority)
      : undefined;
    const neuralRolloutCertificateSha256 = options.neuralRolloutCertificate
      ? modelPayloadSha256(options.neuralRolloutCertificate)
      : undefined;
    const existing = this.#transactions.get(normalizedTransactionId);
    if (existing) {
      if (existing.fingerprint !== fingerprint
        || existing.decisionSha256 !== decisionSha256
        || existing.toolAuthoritySha256 !== toolAuthoritySha256
        || existing.neuralRolloutCertificateSha256
          !== neuralRolloutCertificateSha256) {
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
      decision,
      options
    );
    this.#transactions.set(normalizedTransactionId, {
      fingerprint,
      decisionSha256,
      toolAuthoritySha256,
      toolAuthority: options.toolAuthority
        ? structuredClone(options.toolAuthority)
        : undefined,
      neuralRolloutCertificateSha256,
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
    decision: ModelDecisionRef | undefined,
    options: HumanoidActionInvocationOptions
  ): Promise<HumanoidActionReceipt> {
    const before = this.#world.snapshot();
    const result = await this.#execute(name, rawInput, {
      transactionId,
      agentId,
      fingerprint,
      ...(decision ? { decision } : {}),
      ...(options.toolAuthority ? { toolAuthority: options.toolAuthority } : {}),
      ...(options.neuralRolloutCertificate
        ? { neuralRolloutCertificate: options.neuralRolloutCertificate }
        : {}),
      ...(options.signal ? { signal: options.signal } : {})
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
      const planningReceipt = typeof planningTransactionId === "string"
        ? this.#receipts.get(planningTransactionId)
        : undefined;
      const transitClearanceRecovery = planningReceipt !== undefined
        && jsonObject(planningReceipt.detail)?.recovery_kind
          === "navigation_transit_clearance";
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
        this.#latestGroundingObservation = null;
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
      if (skill && skillPlan && receipt.accepted && !transitClearanceRecovery) {
        const observation = this.#world.observe();
        const advanced = advanceHumanoidSkillPlan({
          plan: skillPlan,
          binding: skill,
          worldRevision: receipt.worldAfterRevision,
          physicalAnchor: skillPlanPhysicalAnchor(
            observation,
            skillPlanObjectIds(skillPlan)
          ),
          executionSucceeded: true,
          phasePostconditionSatisfied: humanoidSkillPhasePostconditionSatisfied({
            binding: skill,
            observation,
            planningReceipt,
            executionReceipt: receipt
          })
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
    if (!isRecognizedPlanningReceiptActor(receipt.agentId)
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
    if (receipt.accepted
      && (receipt.action === "execute_humanoid_skill"
        || receipt.action === "execute_whole_body_motion"
        || receipt.action === "execute_humanoid_navigation")) {
      this.#navigationTransitClearanceRequirementByAgent.clear();
      return;
    }
    if (!isRecognizedPlanningReceiptActor(receipt.agentId) || receipt.accepted) return;
    const motionAgentId = receipt.agentId;
    const detail = jsonObject(receipt.detail);
    const failures = receipt.action === "plan_humanoid_navigation"
      ? [{
          reason: detail?.reason,
          blockingContacts: detail?.blocking_contacts
        }]
      : receipt.action === "plan_humanoid_skill"
        && detail?.autonomous_plan_kind === "navigation"
        && Array.isArray(detail.attempts)
        ? detail.attempts.map((attempt) => {
            const failure = jsonObject(attempt);
            return {
              reason: failure?.reason,
              blockingContacts: failure?.blocking_contacts
            };
          })
        : [];
    if (failures.length === 0) return;
    const worldWithObservation = this.#world as HumanoidWorld & {
      observe?: () => HumanoidWorldObservation;
    };
    const solidTokens = typeof worldWithObservation.observe === "function"
      ? worldWithObservation.observe().solidTokens
      : [];
    for (const failure of failures) {
      const requirement = navigationTransitClearanceFromRejection({
        reason: failure.reason,
        blockingContacts: failure.blockingContacts,
        transactionId: receipt.transactionId,
        blockedAction: receipt.action as "plan_humanoid_navigation" | "plan_humanoid_skill",
        worldRevision: receipt.worldAfterRevision,
        snapshot: this.#world.snapshot(),
        solidTokens,
        skillTransactionId: navigationFailureSkillTransactionId(
          receipt,
          this.#activeSkillByAgent.get(motionAgentId)
        )
      });
      if (requirement) {
        this.#navigationTransitClearanceRequirementByAgent.set(
          motionAgentId,
          requirement
        );
        return;
      }
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
    const compatible = humanoidSkillObservationCompatible(observed, current);
    this.#reconcileSkillPlanWorldAuthority(
      agentId,
      observed,
      current,
      compatible
    );
    if (!compatible) return undefined;
    // Manipulation reachability is expensive rollout-derived sensor evidence.
    // A synchronous authority renewal only refreshes the cheap world snapshot,
    // whose reachability arrays are intentionally empty.  Keep the derived
    // evidence while its strict root/object/carry compatibility contract still
    // holds; otherwise model latency silently erases the exact IK base pose and
    // turns a manipulation approach back into generic face-the-object motion.
    const renewed = retainCompatibleManipulationEvidence(observed, current);
    this.#observationRevisionByAgent.set(agentId, renewed.worldRevision);
    this.#observationByAgent.set(agentId, renewed);
    return renewed;
  }

  #reconcileSkillPlanWorldAuthority(
    agentId: string,
    previous: HumanoidWorldObservation | undefined,
    current: HumanoidWorldObservation,
    compatible = previous !== undefined
      && humanoidSkillObservationCompatible(previous, current)
  ): void {
    const transactionId = this.#activeSkillPlanTransactionByAgent.get(agentId);
    const plan = transactionId
      ? this.#skillPlansByTransactionId.get(transactionId)
      : undefined;
    if (!plan) return;
    if (plan.world_revision === current.worldRevision) return;
    if ((previous && plan.world_revision === previous.worldRevision && compatible)
      || skillPlanAnchorCompatible(plan, current)) {
      this.#skillPlansByTransactionId.set(plan.transaction_id, {
        ...plan,
        observed_frame: current.frame,
        world_revision: current.worldRevision,
        physical_anchor: skillPlanPhysicalAnchor(current, skillPlanObjectIds(plan))
      });
      return;
    }
    this.#skillPlansByTransactionId.delete(plan.transaction_id);
    this.#activeSkillPlanTransactionByAgent.delete(agentId);
    this.#activeSkillByAgent.delete(agentId);
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
    const activeGoal = this.#activeGoal?.();
    const recovery = this.#recoveryPolicyByAgent.get(agentId);
    const recoveryInterrupt = this.#activeRecoverySafetyInterrupt?.();
    const safetyRecoveryAuthorized = humanoidSafetyRecoveryAuthorized(
      recoveryInterrupt,
      observation,
      previous.invocation
    );
    const rebound = bindHumanoidSkill({
      transactionId: previous.transaction_id,
      agentId,
      request: {
        skill_plan_transaction_id: previous.skill_plan_transaction_id,
        skill_node_id: previous.skill_node_id,
        invocation: previous.invocation,
        phase: previous.phase
      },
      observation,
      ...(activeGoal ? { activeGoal } : {}),
      ...((recovery
        && humanoidRecoverySelectionAccepted(recovery, previous.invocation))
        || safetyRecoveryAuthorized
        ? { recoveryAuthorized: true }
        : {}),
      ...(safetyRecoveryAuthorized && recoveryInterrupt
        ? { recoveryInterrupt }
        : {})
    });
    if (!rebound.accepted) return rebound;
    this.#activeSkillByAgent.set(agentId, rebound.binding);
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
      decision?: ModelDecisionRef;
      toolAuthority?: HumanoidActionToolCallAuthority;
      neuralRolloutCertificate?: NeuralRolloutExecutionAdmission;
      signal?: AbortSignal;
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
      && invocation.agentId === LEGACY_HUMANOID_MOTION_AGENT_ID) {
      return {
        accepted: false,
        code: "legacy_agent_authority_retired",
        channels: [],
        detail: {
          automatic_actuation: false,
          legacy_agent_id: invocation.agentId,
          current_planning_agent_id: HUMANOID_NEURAL_AGENT_IDS.motorIntent,
          recovery: "Start a fresh neural hierarchy epoch and invoke Motor Intent through its Premotor parent. Legacy Motion receipts are read-only history."
        }
      };
    }
    if (isPlanningAction(name)
      && isCurrentHumanoidPlanningActor(invocation.agentId)) {
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
      const transitClearanceRecovery = (
        name === "plan_whole_body_motion_candidates"
          || name === "plan_humanoid_navigation"
      )
        && this.#navigationTransitClearanceRequirementByAgent.has(
          invocation.agentId
        );
      if (this.#requireSkillBinding && name !== "plan_whole_body_motion") {
        const skillReference = transitClearanceRecovery
          ? transitClearanceSkillReference(
              this.#navigationTransitClearanceRequirementByAgent.get(
                invocation.agentId
              )!,
              this.#activeSkillByAgent.get(invocation.agentId),
              name,
              rawInput
            )
          : (() => {
              const renewed = this.#renewPlanningSkillAuthority(invocation.agentId);
              return renewed.accepted
                ? validateSkillPlanningReference({
                    binding: renewed.binding,
                    action: name,
                    rawInput,
                    currentWorldRevision: this.#world.snapshot().worldRevision
                  })
                : renewed;
            })();
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
      const observation = invocation.agentId === HUMANOID_NEURAL_AGENT_IDS.sensorFusion
        && goalRequiresManipulationReachability(this.#activeGoal?.())
        ? await this.#world.observeManipulationReachability()
        : await this.#world.captureObservation();
      const authorityOwners = invocation.agentId === HUMANOID_NEURAL_AGENT_IDS.sensorFusion
        ? [invocation.agentId, HUMANOID_NEURAL_AGENT_IDS.motorIntent]
        : [invocation.agentId];
      for (const ownerId of authorityOwners) {
        const previousObservation = this.#observationByAgent.get(ownerId);
        this.#reconcileSkillPlanWorldAuthority(
          ownerId,
          previousObservation,
          observation
        );
        this.#observationRevisionByAgent.set(ownerId, observation.worldRevision);
        this.#observationByAgent.set(ownerId, structuredClone(observation));
        const clearance = this.#navigationTransitClearanceRequirementByAgent.get(ownerId);
        if (clearance) {
          this.#navigationTransitClearanceRequirementByAgent.set(
            ownerId,
            refreshNavigationTransitClearanceRequirement({
              requirement: clearance,
              worldRevision: observation.worldRevision,
              snapshot: this.#world.snapshot(),
              solidTokens: observation.solidTokens
            })
          );
        } else {
          this.#activeSkillByAgent.delete(ownerId);
        }
        const recovery = this.#recoveryPolicyByAgent.get(ownerId);
        if (recovery && recovery.world_revision !== observation.worldRevision) {
          this.#recoveryPolicyByAgent.delete(ownerId);
        }
        if (ownerId === HUMANOID_NEURAL_AGENT_IDS.motorIntent) {
          const requirement = manipulationBasePlacementRequirement(observation);
          if (requirement) {
            this.#manipulationBasePlacementRequirementByAgent.set(ownerId, requirement);
          } else {
            this.#manipulationBasePlacementRequirementByAgent.delete(ownerId);
          }
        }
      }
      if (invocation.agentId === HUMANOID_NEURAL_AGENT_IDS.sensorFusion) {
        this.#latestGroundingObservation = structuredClone(observation);
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
      const plan = registerHumanoidSkillPlan({
        transactionId: invocation.transactionId,
        agentId: invocation.agentId,
        proposal,
        observedFrame: observation.frame,
        worldRevision: observation.worldRevision,
        physicalAnchor: skillPlanPhysicalAnchor(
          observation,
          skillPlanProposalObjectIds(proposal)
        )
      });
      const readyBindings = readyHumanoidSkillPlanBindings(plan);
      const neuralCommitment = this.#activeNeuralSkillCommitment?.();
      if (this.#activeNeuralSkillCommitment !== undefined
        && (!neuralCommitment
          || readyBindings.length !== 1
          || !sameHumanoidSkillInvocation(
            readyBindings[0]!.invocation,
            neuralCommitment.invocation
          ))) {
        return {
          accepted: false,
          code: neuralCommitment
            ? "skill_plan_commitment_entry_mismatch"
            : "skill_plan_neural_commitment_required",
          channels: [],
          detail: {
            automatic_actuation: false,
            active_commitment: neuralCommitment ?? null,
            rejected_ready_nodes: readyBindings.map((binding) => ({
              skill_node_id: binding.skill_node_id,
              invocation: binding.invocation
            })),
            recovery: neuralCommitment
              ? "The dependency-ready Skill DAG entry must be exactly the one bounded invocation authorized by the active Action Selection commitment"
              : "Return to Action Selection and establish one Goal-aligned bounded Skill commitment before motor planning"
          }
        };
      }
      const activeGoal = this.#activeGoal?.();
      if (activeGoal) {
        const recovery = this.#recoveryPolicyByAgent.get(invocation.agentId);
        const recoveryInterrupt = this.#activeRecoverySafetyInterrupt?.();
        const admission = readyBindings.map((binding) => ({
          binding,
          alignment: alignHumanoidSkillToGoal({
            goal: activeGoal,
            invocation: binding.invocation,
            observation,
            ...((recovery
              && humanoidRecoverySelectionAccepted(recovery, binding.invocation))
              || humanoidSafetyRecoveryAuthorized(
                recoveryInterrupt,
                observation,
                binding.invocation
              )
              ? { recoveryAuthorized: true }
              : {})
          })
        }));
        if (!admission.some(({ alignment }) => alignment.accepted)) {
          return {
            accepted: false,
            code: "skill_plan_no_goal_aligned_entry",
            channels: [],
            detail: {
              automatic_actuation: false,
              selected_strategy_id: plan.selected_strategy_id,
              active_goal: activeGoal,
              rejected_ready_nodes: admission.map(({ binding, alignment }) => ({
                skill_node_id: binding.skill_node_id,
                invocation: binding.invocation,
                reason: alignment.accepted ? null : alignment.reason
              })),
              recovery: "Submit a materially different Skill DAG whose dependency-ready entry advances the active Goal, establishes a matching prerequisite, or is authorized by current physical recovery evidence"
            }
          };
        }
      }
      const previous = this.#activeSkillPlanTransactionByAgent.get(invocation.agentId);
      if (previous) this.#skillPlansByTransactionId.delete(previous);
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
          skill_plan_transaction_id: plan.transaction_id,
          selected_strategy_id: plan.selected_strategy_id,
          required_next_tool: "begin_humanoid_skill",
          ready_skill_bindings: readyBindings
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
      const neuralCommitment = this.#activeNeuralSkillCommitment?.();
      if (this.#activeNeuralSkillCommitment !== undefined
        && (!neuralCommitment
          || !sameHumanoidSkillInvocation(
            request.invocation,
            neuralCommitment.invocation
          ))) {
        return {
          accepted: false,
          code: neuralCommitment
            ? "skill_binding_commitment_mismatch"
            : "skill_binding_neural_commitment_required",
          channels: [],
          detail: {
            automatic_actuation: false,
            active_commitment: neuralCommitment ?? null,
            rejected_invocation: request.invocation,
            recovery: neuralCommitment
              ? "Bind exactly the bounded Skill invocation authorized by the active Action Selection commitment"
              : "Return to Action Selection and establish one Goal-aligned bounded Skill commitment before binding a motor phase"
          }
        };
      }
      const planTransactionId = request.skill_plan_transaction_id;
      const plan = planTransactionId
        ? this.#skillPlansByTransactionId.get(planTransactionId)
        : undefined;
      if (this.#requireSkillBinding || planTransactionId || request.skill_node_id) {
        const nodeAuthority = authorizeHumanoidSkillPlanNode({
          plan,
          planTransactionId,
          nodeId: request.skill_node_id,
          invocation: request.invocation,
          phase: request.phase,
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
      const recovery = this.#recoveryPolicyByAgent.get(invocation.agentId);
      const recoveryInterrupt = this.#activeRecoverySafetyInterrupt?.();
      const safetyRecoveryAuthorized = humanoidSafetyRecoveryAuthorized(
        recoveryInterrupt,
        observation,
        request.invocation
      );
      const activeGoal = this.#activeGoal?.();
      const result = bindHumanoidSkill({
        transactionId: invocation.transactionId,
        agentId: invocation.agentId,
        request,
        observation,
        ...(activeGoal ? { activeGoal } : {}),
        ...((recovery
          && humanoidRecoverySelectionAccepted(recovery, request.invocation))
          || safetyRecoveryAuthorized
          ? { recoveryAuthorized: true }
          : {}),
        ...(safetyRecoveryAuthorized && recoveryInterrupt
          ? { recoveryInterrupt }
          : {})
      });
      if (!result.accepted) {
        return {
          accepted: false,
          code: result.code,
          channels: [],
          detail: result.detail
        };
      }
      if (recovery
        && !humanoidRecoverySelectionAccepted(recovery, result.binding.invocation)
        && !safetyRecoveryAuthorized) {
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
      this.#rebindNavigationTransitClearanceSkill(invocation.agentId);
      return {
        accepted: true,
        code: "humanoid_skill_bound",
        channels: [],
        detail: {
          automatic_actuation: false,
          skill_transaction_id: result.binding.transaction_id,
          required_next_tool: "plan_humanoid_skill",
          required_next_arguments: {
            skill_transaction_id: result.binding.transaction_id
          },
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
        const activeGoal = this.#activeGoal?.();
        const recovery = this.#recoveryPolicyByAgent.get(invocation.agentId);
        const recoveryInterrupt = this.#activeRecoverySafetyInterrupt?.();
        const safetyRecoveryAuthorized = humanoidSafetyRecoveryAuthorized(
          recoveryInterrupt,
          observation,
          binding.invocation
        );
        plan = planAutonomousHumanoidSkill({
          binding,
          observation,
          ...(activeGoal ? { activeGoal } : {}),
          ...((recovery
            && humanoidRecoverySelectionAccepted(recovery, binding.invocation))
            || safetyRecoveryAuthorized
            ? { recoveryAuthorized: true }
            : {}),
          ...(safetyRecoveryAuthorized && recoveryInterrupt
            ? { recoveryInterrupt }
            : {})
        });
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
        const result = await this.#world.planWholeBodyMotionCandidates(plan.batch, {
          skillCallIdentity: humanoidEmbodiedSkillIdentity(binding)
        });
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
      if (plan.kind === "recovery") {
        const result = await this.#world.planHumanoidRecovery(plan.plan, {
          skillCallIdentity: humanoidEmbodiedSkillIdentity(binding)
        });
        if (result.accepted) this.#planChannels.set(result.planId, result.channels);
        return {
          accepted: result.accepted,
          code: result.accepted
            ? "autonomous_skill_recovery_admitted"
            : "autonomous_skill_recovery_rejected",
          channels: result.channels,
          detail: {
            ...this.#planningSkillDetail(invocation.agentId),
            automatic_actuation: false,
            autonomous_plan_kind: "recovery",
            plan_id: result.planId,
            created_revision: result.createdRevision,
            expires_revision: result.expiresRevision,
            intent_sha256: result.intentSha256,
            recovery_contract_sha256: result.contractSha256,
            safety_interrupt_id: plan.plan.contract.safetyInterrupt.interrupt_id,
            reason: result.reason ?? null
          }
        };
      }
      const { selected, attempts } = await planAutonomousHumanoidNavigation({
        world: this.#world,
        binding,
        plan
      });
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
      const skill = this.#skillByPlanningTransactionId.get(
        input.planning_transaction_id
      );
      const neuralCommitment = this.#activeNeuralSkillCommitment?.();
      if (this.#activeNeuralSkillCommitment !== undefined
        && (!neuralCommitment
          || !skill
          || !sameHumanoidSkillInvocation(
            skill.invocation,
            neuralCommitment.invocation
          ))) {
        return {
          accepted: false,
          code: "skill_execution_commitment_mismatch",
          channels: [],
          detail: {
            planning_transaction_id: input.planning_transaction_id,
            active_commitment: neuralCommitment ?? null,
            planned_skill_binding: skill ?? null,
            automatic_actuation: false,
            recovery: "Return to Action Selection; physical execution requires the planned Skill binding to exactly match the one active bounded commitment"
          }
        };
      }
      const planKind = planningReceipt
        ? jsonObject(planningReceipt.detail)?.autonomous_plan_kind
        : undefined;
      if (planKind !== "motion" && planKind !== "navigation"
        && planKind !== "recovery") {
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
      if (planKind === "recovery") {
        const recoveryInterrupt = this.#activeRecoverySafetyInterrupt?.();
        if (!skill || skill.invocation.skill !== "stabilize"
          || skill.phase !== "recover_support"
          || !skill.recovery_interrupt_id
          || recoveryInterrupt?.interrupt_id !== skill.recovery_interrupt_id
          || recoveryInterrupt.status !== "acknowledged") {
          return {
            accepted: false,
            code: "recovery_execution_authority_missing",
            channels: [],
            detail: {
              planning_transaction_id: input.planning_transaction_id,
              plan_id: reference.planId,
              automatic_actuation: false,
              reason: "Recovery execution lost its acknowledged physical safety interrupt"
            }
          };
        }
      }
      const channels = this.#planChannels.get(reference.planId)
        ?? (planKind === "navigation"
          ? ["locomotion"]
          : planKind === "recovery"
            ? [
                "locomotion",
                "left_leg",
                "right_leg",
                "torso",
                "left_arm",
                "right_arm"
              ]
            : []);
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
      const grounding = await this.#beforePhysicalExecution?.({
        transactionId: invocation.transactionId,
        agentId: invocation.agentId,
        action: name,
        fingerprint: invocation.fingerprint,
        planningTransactionId: input.planning_transaction_id,
        planId: reference.planId,
        ...(invocation.decision ? { decision: invocation.decision } : {}),
        ...(invocation.toolAuthority ? { toolAuthority: invocation.toolAuthority } : {}),
        ...(invocation.neuralRolloutCertificate
          ? { neuralRolloutCertificate: invocation.neuralRolloutCertificate }
          : {})
      });
      if (grounding && !grounding.accepted) {
        return groundingRejection({
          grounding,
          channels,
          planningTransactionId: input.planning_transaction_id,
          planId: reference.planId,
          planKind
        });
      }
      const executionSignal = combineExecutionSignals(
        this.#signal,
        invocation.signal
      );
      const skillEventStream = skill
        ? this.#skillEventStream(humanoidEmbodiedSkillIdentity(skill))
        : null;
      const articulationHorizon = isHumanoidArticulationActuation(skill)
        ? skillEventStream
        : null;
      const navigationHorizon = planKind === "navigation"
        && isHumanoidNavigationSkill(skill)
        ? skillEventStream
        : null;
      const deterministicHorizon = articulationHorizon ?? navigationHorizon;
      const skillFrameOffset = deterministicHorizon
        ? this.#physicalExecutionFrameOffset(invocation.transactionId)
        : 0;
      const completedPlanFrameCount = deterministicHorizon
        ? this.#completedPhysicalPlanFrameCount(invocation.transactionId)
        : 0;
      const completedPlanCount = deterministicHorizon
        ? this.#completedPhysicalPlanCount(invocation.transactionId)
        : 0;
      const continuationPlanId = articulationHorizon && skill
        ? this.#world.pendingWholeBodyMotionPlanIdForSkillCall(
            humanoidEmbodiedSkillIdentity(skill).callId
          )
        : undefined;
      const resumeBetweenArticulationSegments = articulationHorizon !== null
        && skillFrameOffset > 0
        && continuationPlanId === undefined;
      const articulationSegmentBudgetExhausted = articulationHorizon !== null
        && humanoidArticulationSegmentBudgetExhausted(completedPlanCount);
      const navigationContinuationPlanId = navigationHorizon && skill
        ? this.#world.pendingNavigationPlanIdForSkillCall(
            humanoidEmbodiedSkillIdentity(skill).callId
          )
        : undefined;
      const resumeBetweenNavigationSegments = navigationHorizon !== null
        && skillFrameOffset > 0
        && navigationContinuationPlanId === undefined;
      const navigationSegmentBudgetExhausted = navigationHorizon !== null
        && humanoidNavigationSegmentBudgetExhausted(completedPlanCount);
      const executionPlanId = continuationPlanId
        ?? navigationContinuationPlanId
        ?? reference.planId;
      const options = {
        realtime: this.#realtimeExecution,
        retainTerminal: this.#retainPhysicalTerminals,
        ...(this.#physicalFrameSink
          ? {
              persistenceSink: this.#physicalFrameSink,
              ...(this.#frameSink ? { progressSink: this.#frameSink } : {}),
              ...(this.#physicalPersistenceFrameStride === undefined
                ? {}
                : {
                    persistenceFrameStride:
                      this.#physicalPersistenceFrameStride,
                    persistenceStartWorldRevision:
                      this.#physicalExecutionStartWorldRevision(
                        invocation.transactionId
                      ) ?? this.#world.snapshot().worldRevision
                  })
            }
          : {}),
        ...(this.#skillEventSink
          ? { skillEventSink: this.#skillEventSink }
          : {}),
        ...(skillEventStream ? { skillEventStream } : {}),
        ...(deterministicHorizon
          ? {
              deferSkillProgress: true,
              deferSkillTerminal: true,
              deferSkillControllerOutcome: true,
              skillWindow: {
                maximumSteps: articulationHorizon
                  ? HUMANOID_ARTICULATION_HORIZON.maximum_control_steps
                  : HUMANOID_NAVIGATION_HORIZON.maximum_control_steps,
                stepOffset: completedPlanFrameCount
              }
            }
          : {}),
        ...(this.#policyFrameSink
          ? { policyFrameSink: this.#policyFrameSink }
          : {}),
        ...(executionSignal ? { signal: executionSignal } : {})
      };
      const initialRootPosition = this.#world.snapshot().robot.rootPosition;
      const initialResult = resumeBetweenArticulationSegments
        || articulationSegmentBudgetExhausted
        || resumeBetweenNavigationSegments
        || navigationSegmentBudgetExhausted
        ? {
            accepted: true,
            code: "motion_option_succeeded" as const,
            frames: 0,
            finalSnapshot: this.#world.snapshot(),
            detail: {}
          }
        : planKind === "motion"
        ? await this.#world.executeWholeBodyMotion(
            executionPlanId,
            this.#frameSink,
            options
          )
        : planKind === "recovery"
        ? await this.#world.executeHumanoidRecovery(
            executionPlanId,
            this.#frameSink,
            options
          )
        : await this.#world.executeNavigation(
            executionPlanId,
            this.#frameSink,
            options
          );
      let result: {
        accepted: boolean;
        code: string;
        frames: number;
        finalSnapshot: HumanoidWorldSnapshot;
        terminalResultSha256?: string;
        detail: unknown;
      } = initialResult;
      if (planKind === "motion" && result.accepted
        && isHumanoidArticulationActuation(skill)) {
        result = await executeHumanoidArticulationHorizon({
          world: this.#world,
          binding: skill,
          initialPlanId: resumeBetweenArticulationSegments
            || articulationSegmentBudgetExhausted
            ? null
            : executionPlanId,
          initialExecution: initialResult,
          skillEventStream: articulationHorizon!,
          initialCommittedFrames: completedPlanFrameCount,
          initialCompletedSegments: completedPlanCount,
          ...(this.#frameSink ? { frameSink: this.#frameSink } : {}),
          executionOptions: options
        });
      }
      if (planKind === "navigation" && result.accepted
        && isHumanoidNavigationSkill(skill)) {
        const horizonGoal = this.#activeGoal?.();
        const initialNavigationPlan = resumeBetweenNavigationSegments
          || navigationSegmentBudgetExhausted
          ? null
          : navigationPlanForHorizon(this.#world, executionPlanId);
        result = await executeHumanoidNavigationHorizon({
          world: this.#world,
          binding: skill,
          initialPlan: initialNavigationPlan,
          initialExecution: initialResult,
          initialRootPosition,
          skillEventStream: navigationHorizon!,
          initialCommittedFrames: completedPlanFrameCount,
          initialCompletedSegments: completedPlanCount,
          ...(this.#frameSink ? { frameSink: this.#frameSink } : {}),
          executionOptions: options,
          ...(horizonGoal ? { activeGoal: horizonGoal } : {}),
          ...(skill.recovery_authorized ? { recoveryAuthorized: true } : {})
        });
      }
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
      const activeSkill = this.#activeSkillByAgent.get(invocation.agentId);
      const result = await this.#world.planWholeBodyMotion(plan, {
        retainTerminalJointTracking: transitClearance !== undefined,
        ...(activeSkill
          ? { skillCallIdentity: humanoidEmbodiedSkillIdentity(activeSkill) }
          : {})
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
            transitClearance,
            batch.termination
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
      const activeSkill = this.#activeSkillByAgent.get(invocation.agentId);
      const result = await this.#world.planWholeBodyMotionCandidates(batch, {
        retainTerminalJointTracking: transitClearance !== undefined,
        ...(activeSkill
          ? { skillCallIdentity: humanoidEmbodiedSkillIdentity(activeSkill) }
          : {})
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
          ...(transitClearance
            ? navigationClearanceRecoveryDetail(transitClearance)
            : {}),
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
      const grounding = await this.#beforePhysicalExecution?.({
        transactionId: invocation.transactionId,
        agentId: invocation.agentId,
        action: name,
        fingerprint: invocation.fingerprint,
        planningTransactionId: input.planning_transaction_id,
        planId: reference.planId,
        ...(invocation.decision ? { decision: invocation.decision } : {}),
        ...(invocation.toolAuthority ? { toolAuthority: invocation.toolAuthority } : {}),
        ...(invocation.neuralRolloutCertificate
          ? { neuralRolloutCertificate: invocation.neuralRolloutCertificate }
          : {})
      });
      if (grounding && !grounding.accepted) {
        return groundingRejection({
          grounding,
          channels,
          planningTransactionId: input.planning_transaction_id,
          planId: reference.planId,
          planningAction: reference.planningAction
        });
      }
      const executionSignal = combineExecutionSignals(
        this.#signal,
        invocation.signal
      );
      const motionSkill = this.#skillByPlanningTransactionId.get(
        input.planning_transaction_id
      );
      const motionSkillEventStream = motionSkill
        ? this.#skillEventStream(humanoidEmbodiedSkillIdentity(motionSkill))
        : null;
      const result = await this.#world.executeWholeBodyMotion(
        reference.planId,
        this.#frameSink,
        {
          realtime: this.#realtimeExecution,
          retainTerminal: this.#retainPhysicalTerminals,
          ...(this.#physicalFrameSink
            ? {
                persistenceSink: this.#physicalFrameSink,
                ...(this.#frameSink ? { progressSink: this.#frameSink } : {}),
                ...(this.#physicalPersistenceFrameStride === undefined
                  ? {}
                  : {
                      persistenceFrameStride:
                        this.#physicalPersistenceFrameStride,
                      persistenceStartWorldRevision:
                        this.#physicalExecutionStartWorldRevision(
                          invocation.transactionId
                        ) ?? this.#world.snapshot().worldRevision
                    })
              }
            : {}),
          ...(this.#skillEventSink
            ? { skillEventSink: this.#skillEventSink }
            : {}),
          ...(motionSkillEventStream
            ? { skillEventStream: motionSkillEventStream }
            : {}),
          ...(this.#policyFrameSink
            ? { policyFrameSink: this.#policyFrameSink }
            : {}),
          ...(executionSignal ? { signal: executionSignal } : {})
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
          ...reference.recovery,
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
      const placementRequirement = this.#manipulationBasePlacementRequirementByAgent.get(
        invocation.agentId
      );
      const placementRejection = placementRequirement
        ? manipulationBasePlacementNavigationRejection(input, placementRequirement)
        : null;
      if (placementRejection) return placementRejection;
      const activeSkill = this.#activeSkillByAgent.get(invocation.agentId);
      const result = await this.#world.planNavigation(
        input.target,
        input.arrival_heading,
        null,
        activeSkill
          ? { skillCallIdentity: humanoidEmbodiedSkillIdentity(activeSkill) }
          : {}
      );
      if (result.accepted) this.#planChannels.set(result.planId, ["locomotion"]);
      return {
        accepted: result.accepted,
        code: result.accepted ? "humanoid_route_validated" : "humanoid_route_rejected",
        channels: ["locomotion"],
        detail: {
          ...this.#planningSkillDetail(invocation.agentId),
          ...(transitClearance
            ? navigationClearanceRecoveryDetail(transitClearance)
            : {}),
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
          blocking_contacts: result.blockingContacts ?? [],
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
    const grounding = await this.#beforePhysicalExecution?.({
      transactionId: invocation.transactionId,
      agentId: invocation.agentId,
      action: name,
      fingerprint: invocation.fingerprint,
      planningTransactionId: input.planning_transaction_id,
      planId: reference.planId,
      ...(invocation.decision ? { decision: invocation.decision } : {}),
      ...(invocation.toolAuthority ? { toolAuthority: invocation.toolAuthority } : {}),
      ...(invocation.neuralRolloutCertificate
        ? { neuralRolloutCertificate: invocation.neuralRolloutCertificate }
        : {})
    });
    if (grounding && !grounding.accepted) {
      return groundingRejection({
        grounding,
        channels: ["locomotion"],
        planningTransactionId: input.planning_transaction_id,
        planId: reference.planId,
        planningAction: reference.planningAction
      });
    }
    const executionSignal = combineExecutionSignals(
      this.#signal,
      invocation.signal
    );
    const navigationSkill = this.#skillByPlanningTransactionId.get(
      input.planning_transaction_id
    );
    const navigationSkillEventStream = navigationSkill
      ? this.#skillEventStream(humanoidEmbodiedSkillIdentity(navigationSkill))
      : null;
    const result = await this.#world.executeNavigation(
      reference.planId,
      this.#frameSink,
      {
        realtime: this.#realtimeExecution,
        retainTerminal: this.#retainPhysicalTerminals,
        ...(this.#physicalFrameSink
          ? {
              persistenceSink: this.#physicalFrameSink,
              ...(this.#frameSink ? { progressSink: this.#frameSink } : {}),
              ...(this.#physicalPersistenceFrameStride === undefined
                ? {}
                : {
                    persistenceFrameStride:
                      this.#physicalPersistenceFrameStride,
                    persistenceStartWorldRevision:
                      this.#physicalExecutionStartWorldRevision(
                        invocation.transactionId
                      ) ?? this.#world.snapshot().worldRevision
                  })
            }
          : {}),
        ...(this.#skillEventSink
          ? { skillEventSink: this.#skillEventSink }
          : {}),
        ...(navigationSkillEventStream
          ? { skillEventStream: navigationSkillEventStream }
          : {}),
        ...(this.#policyFrameSink
          ? { policyFrameSink: this.#policyFrameSink }
          : {}),
        ...(executionSignal ? { signal: executionSignal } : {})
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
        ...reference.recovery,
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

  #skillEventStream(
    identity: ReturnType<typeof humanoidEmbodiedSkillIdentity>
  ): HumanoidSkillEventStream {
    const existing = this.#skillEventStreamByCallId.get(identity.callId);
    if (existing) return existing;
    const stream = new HumanoidSkillEventStream(
      identity,
      this.#skillEventSink,
      this.#restoredSkillEventStateByCallId.get(identity.callId)
    );
    this.#skillEventStreamByCallId.set(identity.callId, stream);
    return stream;
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
    recovery: {
      recovery_kind: "navigation_transit_clearance";
      recovery_collision: JsonValue;
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
    const recoveryCollision = detail?.recovery_collision;
    const recovery = detail?.recovery_kind === "navigation_transit_clearance"
      && recoveryCollision !== undefined
      ? {
          recovery_kind: "navigation_transit_clearance" as const,
          recovery_collision: jsonValue(recoveryCollision)
        }
      : undefined;
    return {
      accepted: true,
      planId,
      planningAction: receipt.action as HumanoidPlanningActionName,
      candidateSelection,
      recovery
    };
  }

  #rebindNavigationTransitClearanceSkill(agentId: string): void {
    const requirement = this.#navigationTransitClearanceRequirementByAgent.get(
      agentId
    );
    const activeSkill = this.#activeSkillByAgent.get(agentId);
    if (!requirement || !activeSkill
      || requirement.skillTransactionId === activeSkill.transaction_id) {
      return;
    }
    this.#navigationTransitClearanceRequirementByAgent.set(agentId, {
      ...requirement,
      skillTransactionId: activeSkill.transaction_id
    });
  }
}

function isPlanningAction(action: HumanoidActionName): action is HumanoidPlanningActionName {
  return action === "plan_humanoid_skill"
    || action === "plan_whole_body_motion"
    || action === "plan_whole_body_motion_candidates"
    || action === "plan_humanoid_navigation";
}

/**
 * Motor Intent is the only live planning authority in neural hierarchy V3.
 */
function isCurrentHumanoidPlanningActor(agentId: string): boolean {
  return agentId === HUMANOID_NEURAL_AGENT_IDS.motorIntent;
}

/**
 * Persisted pre-V3 receipts may still rebuild bounded failure/cooldown state.
 * This historical predicate must never be used to authorize a new invocation.
 */
function isRecognizedPlanningReceiptActor(agentId: string): boolean {
  return isCurrentHumanoidPlanningActor(agentId)
    || agentId === LEGACY_HUMANOID_MOTION_AGENT_ID;
}

function humanoidGroundingAuthorityIds(): readonly string[] {
  return [
    HUMANOID_NEURAL_AGENT_IDS.sensorFusion,
    HUMANOID_NEURAL_AGENT_IDS.motorIntent,
    LEGACY_HUMANOID_MOTION_AGENT_ID
  ];
}

function transitClearanceSkillReference(
  requirement: NavigationTransitClearanceRequirement,
  activeSkill: ActiveHumanoidSkillBinding | undefined,
  action: "plan_whole_body_motion_candidates" | "plan_humanoid_navigation",
  rawInput: unknown
): { accepted: true } | { accepted: false; code: string; detail: JsonValue } {
  const suppliedValue = rawInput !== null
    && typeof rawInput === "object"
    && !Array.isArray(rawInput)
    ? (rawInput as Record<string, unknown>).skill_transaction_id
    : undefined;
  const supplied = typeof suppliedValue === "string" ? suppliedValue : null;
  const authoritativeSkillTransactionId = activeSkill?.transaction_id
    ?? requirement.skillTransactionId;
  if (authoritativeSkillTransactionId === null
    || supplied !== authoritativeSkillTransactionId) {
    return {
      accepted: false,
      code: "skill_reference_mismatch",
      detail: jsonValue({
        action,
        supplied_skill_transaction_id: supplied ?? null,
        active_skill_transaction_id: authoritativeSkillTransactionId,
        automatic_actuation: false
      })
    };
  }
  return { accepted: true };
}

function navigationFailureSkillTransactionId(
  receipt: HumanoidActionReceipt,
  activeSkill: ActiveHumanoidSkillBinding | undefined
): string | null {
  const detail = jsonObject(receipt.detail);
  const binding = detail?.skill_binding === undefined
    ? undefined
    : jsonObject(detail.skill_binding);
  const input = jsonObject(receipt.input);
  const candidate = binding?.transaction_id
    ?? input?.skill_transaction_id
    ?? activeSkill?.transaction_id;
  return typeof candidate === "string" && candidate.trim()
    ? candidate
    : null;
}

function navigationClearanceRecoveryDetail(
  requirement: NavigationTransitClearanceRequirement
): {
  recovery_kind: "navigation_transit_clearance";
  recovery_collision: JsonValue;
} {
  return {
    recovery_kind: "navigation_transit_clearance",
    recovery_collision: jsonValue({
      hand_surface: requirement.handSurface,
      target_kind: requirement.collisionTargetKind,
      target_id: requirement.collisionTargetKind === "environment"
        ? null
        : requirement.collisionTargetId,
      contact_point_world: requirement.contactPointWorld,
      separation_normal_world: requirement.separationNormalWorld
    })
  };
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
  const observedObjectTokens = observed.interaction?.object_world_model?.objects;
  const currentObjectTokens = current.interaction?.object_world_model?.objects;
  const observedCarryBindings = observed.interaction?.carrying?.bindings;
  const currentCarryBindings = current.interaction?.carrying?.bindings;
  if (!observedObjectTokens
    || !currentObjectTokens
    || !observedCarryBindings
    || !currentCarryBindings) {
    return false;
  }
  const observedObjects = new Map(observedObjectTokens
    .filter(({ status }) => status === "visible")
    .map((object) => [object.id, object]));
  const currentObjects = new Map(currentObjectTokens
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
    observedCarryBindings,
    currentCarryBindings
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

function goalRequiresManipulationReachability(goal: Goal | undefined): boolean {
  return goal?.predicates.some((predicate) => (
    predicate.type !== "robot_at"
      && predicate.type !== "robot_in_zone"
      && predicate.type !== "block_removed"
  )) ?? false;
}

function retainCompatibleManipulationEvidence(
  observed: HumanoidWorldObservation,
  current: HumanoidWorldObservation
): HumanoidWorldObservation {
  if (observed.manipulationReachability.length === 0
    && observed.manipulationBasePlacements.length === 0) return current;
  return {
    ...current,
    manipulationReachability: structuredClone(
      observed.manipulationReachability
    ),
    manipulationBasePlacements: structuredClone(
      observed.manipulationBasePlacements
    )
  };
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

function skillPlanPhysicalAnchor(
  observation: HumanoidWorldObservation,
  objectIds: readonly string[]
): NonNullable<RegisteredHumanoidSkillPlan["physical_anchor"]> {
  const objects = new Map(
    observation.interaction.object_world_model.objects.map((object) => [object.id, object])
  );
  return {
    root_position: { ...observation.robot.rootPosition },
    root_rotation: { ...observation.robot.rootRotation },
    carried_object_ids: [...new Set(
      observation.interaction.carrying.bindings.map(({ object_id }) => object_id)
    )].sort(),
    carried_bindings: observation.interaction.carrying.bindings
      .map(({ object_id: objectId, hand }) => ({ object_id: objectId, hand }))
      .sort((left, right) => (
        `${left.object_id}\0${left.hand}`.localeCompare(
          `${right.object_id}\0${right.hand}`
        )
      )),
    object_poses: [...new Set(objectIds)].sort().flatMap((objectId) => {
      const object = objects.get(objectId);
      return object?.status === "visible" ? [{
        object_id: objectId,
        position: { ...object.pose.position },
        rotation: { ...object.pose.rotation },
        articulation: object.articulation
          ? {
              joint_id: object.articulation.joint_id,
              position: object.articulation.position
            }
          : null
      }] : [];
    })
  };
}

function skillPlanObjectIds(plan: RegisteredHumanoidSkillPlan): string[] {
  return skillPlanProposalObjectIds(plan.proposal);
}

function skillPlanProposalObjectIds(
  proposal: RegisteredHumanoidSkillPlan["proposal"]
): string[] {
  const strategy = proposal.strategies.find(
    ({ strategy_id }) => strategy_id === proposal.selected_strategy_id
  );
  return [...new Set(strategy?.nodes.flatMap(({ invocation }) => {
    const objectIds = "object_id" in invocation ? [invocation.object_id] : [];
    if (invocation.skill !== "place"
      || invocation.destination.type === "semantic_zone"
      || invocation.destination.type === "world_pose") return objectIds;
    return [...objectIds, invocation.destination.object_id];
  }) ?? [])].sort();
}

function skillPlanAnchorCompatible(
  plan: RegisteredHumanoidSkillPlan,
  current: HumanoidWorldObservation
): boolean {
  const anchor = plan.physical_anchor;
  if (!anchor) return false;
  const currentCarriedObjectIds = [...new Set(
    current.interaction.carrying.bindings.map(({ object_id }) => object_id)
  )].sort();
  const currentObjects = new Map(
    current.interaction.object_world_model.objects.map((object) => [object.id, object])
  );
  const mutationDomain = skillPlanMutationDomain(plan);
  const currentBindingKeys = current.interaction.carrying.bindings
    .map(({ object_id: objectId, hand }) => `${objectId}\0${hand}`)
    .sort();
  const anchorBindingKeys = anchor.carried_bindings
    ?.map(({ object_id: objectId, hand }) => `${objectId}\0${hand}`)
    .sort();
  const rootCompatible = mutationDomain.rootPose
    || (pointDistance(anchor.root_position, current.robot.rootPosition)
        <= MANIPULATION_BASE_TARGET_TOLERANCE_METERS
      && quaternionDistance(anchor.root_rotation, current.robot.rootRotation)
        <= SKILL_AUTHORITY_ORIENTATION_DRIFT_RADIANS);
  const carryingCompatible = anchorBindingKeys
    ? JSON.stringify(anchorBindingKeys) === JSON.stringify(currentBindingKeys)
    : JSON.stringify(anchor.carried_object_ids)
      === JSON.stringify(currentCarriedObjectIds);
  return rootCompatible
    && carryingCompatible
    && anchor.object_poses.every((before) => {
      const after = currentObjects.get(before.object_id);
      if (after?.status !== "visible") return false;
      const poseCompatible = mutationDomain.objectPoseIds.has(before.object_id)
        || (pointDistance(before.position, after.pose.position)
            <= SKILL_AUTHORITY_OBJECT_DRIFT_METERS
          && quaternionDistance(before.rotation, after.pose.rotation)
            <= SKILL_AUTHORITY_ORIENTATION_DRIFT_RADIANS);
      const articulationCompatible = before.articulation === undefined
        || (before.articulation === null && after.articulation === null)
        || (before.articulation !== null
          && after.articulation !== null
          && before.articulation.joint_id === after.articulation.joint_id
          && nullableDistance(
            before.articulation.position,
            after.articulation.position
          ) <= SKILL_AUTHORITY_OBJECT_DRIFT_METERS);
      return poseCompatible && articulationCompatible;
    });
}

/**
 * A Skill DAG is durable semantic state.  Only the phase already in physical
 * progress may relax its anchor, and only for entities that phase is allowed
 * to mutate.  This prevents ordinary navigation/station-keeping motion from
 * invalidating the DAG without making manipulation anchors permissive.
 */
function skillPlanMutationDomain(plan: RegisteredHumanoidSkillPlan): {
  rootPose: boolean;
  objectPoseIds: ReadonlySet<string>;
} {
  const active = plan.in_progress_phase;
  if (!active) return { rootPose: false, objectPoseIds: new Set() };
  const strategy = plan.proposal.strategies.find(
    ({ strategy_id: strategyId }) => strategyId === plan.selected_strategy_id
  );
  const node = strategy?.nodes.find(
    ({ node_id: nodeId }) => nodeId === active.node_id
  );
  if (!node) return { rootPose: false, objectPoseIds: new Set() };
  const process = HUMANOID_SKILL_CONTRACTS[node.invocation.skill].process.find(
    ({ phase }) => phase === active.phase
  );
  if (process?.authority !== "navigation") {
    return { rootPose: false, objectPoseIds: new Set() };
  }
  const carriedObjectId = node.invocation.skill === "carry"
      || node.invocation.skill === "carry_to_zone"
      || node.invocation.skill === "bimanual_carry"
    ? node.invocation.object_id
    : undefined;
  return {
    rootPose: true,
    objectPoseIds: new Set(carriedObjectId ? [carriedObjectId] : [])
  };
}

function humanoidSkillPhasePostconditionSatisfied(input: {
  binding: ActiveHumanoidSkillBinding;
  observation: HumanoidWorldObservation;
  planningReceipt: HumanoidActionReceipt | undefined;
  executionReceipt: HumanoidActionReceipt;
}): boolean {
  const { binding, observation } = input;
  const invocation = binding.invocation;
  if (binding.phase_authority !== "navigation") return true;

  if (invocation.skill === "navigate_to_point") {
    return Math.hypot(
      observation.robot.rootPosition.x - invocation.target.x,
      observation.robot.rootPosition.z - invocation.target.z
    ) <= invocation.tolerance_m;
  }

  if (invocation.skill === "navigate_to_zone") {
    const zone = observation.interaction.zones.find(
      ({ zone_id: zoneId }) => zoneId === invocation.zone_id
    );
    return zone?.robot_inside_horizontal === true;
  }

  if (invocation.skill === "carry_to_zone") {
    const carried = observation.interaction.carrying.bindings.some(
      ({ object_id: objectId }) => objectId === invocation.object_id
    );
    const relation = observation.interaction.manipulable_objects
      .find(({ object_id: objectId }) => objectId === invocation.object_id)
      ?.zone_relations.find(({ zone_id: zoneId }) => zoneId === invocation.zone_id);
    return carried && relation !== undefined
      && relation.minimum_horizontal_clearance_m + invocation.tolerance_m >= 0
      && Math.abs(relation.support_height_error_m)
        <= Math.max(invocation.tolerance_m, 0.025);
  }

  const executionDetail = jsonObject(input.executionReceipt.detail);
  const executionResult = executionDetail?.result === undefined
    ? undefined
    : jsonObject(executionDetail.result);
  const navigationHorizon = executionResult?.navigation_horizon === undefined
    ? undefined
    : jsonObject(executionResult.navigation_horizon);
  if (navigationHorizon?.completed === true) return true;

  const remainingDistance = input.planningReceipt
    ? jsonObject(input.planningReceipt.detail)?.remaining_distance
    : undefined;
  return typeof remainingDistance !== "number"
    || !Number.isFinite(remainingDistance)
    || remainingDistance <= 1e-9;
}

function planIdFromReceipt(receipt: HumanoidActionReceipt): string | undefined {
  const planId = jsonObject(receipt.detail)?.plan_id;
  return typeof planId === "string" && planId ? planId : undefined;
}

/**
 * Model-facing projection for Motion. The full observation remains in the
 * deterministic runtime and is still used by every planner, admission check,
 * and physical execution guard. This projection only carries the semantic and
 * geometric facts needed to choose the next Skill transaction.
 */
function motionPlanningObservation(
  snapshot: HumanoidWorldObservation,
  activeGoal: Goal | undefined
): unknown {
  const robot = snapshot.robot;
  const rootYaw = yawFromQuaternion(robot.rootRotation);
  const controllerExecution = robot.controllerExecution ?? {
    protocol: "humanoid-controller-execution-v1" as const,
    mode: robot.controller.learnedPolicy
      ? "learned_policy" as const
      : "reference_control" as const,
    activeImplementation: robot.controller.implementation,
    transition: null
  };
  const relevant = groundingGoalEntities(activeGoal);
  for (const binding of snapshot.interaction.carrying?.bindings ?? []) {
    relevant.objectIds.add(binding.object_id);
  }
  const tokenById = new Map(snapshot.objectTokens.map((token) => [token.id, token]));
  const objects = [...(snapshot.interaction.object_world_model?.objects ?? [])]
    .sort((left, right) => compareGroundingObjects(
      left.id,
      right.id,
      relevant.objectIds,
      tokenById
    ))
    .slice(0, MOTION_GROUNDING_OBJECT_LIMIT)
    .map((object) => ({
      id: object.id,
      kind: object.kind,
      role: object.role,
      status: object.status,
      authority: object.authority,
      pose: object.pose,
      size: object.size,
      shape: object.shape,
      physical: {
        mass_kg: object.physical.mass_kg,
        friction: object.physical.friction,
        mobility: object.physical.mobility
      },
      belief: {
        observation_age_frames: object.belief.observation_age_frames,
        pose_confidence: object.belief.pose_confidence
      },
      affordances: object.affordances,
      interaction_points: object.interaction_points
        .slice(0, MOTION_GROUNDING_INTERACTION_POINT_LIMIT),
      container: object.container ?? null,
      support_surface: object.support_surface ?? null,
      articulation: object.articulation,
      relations: object.relations,
      current_contact_count: object.current_contact_count
    }));
  const projectedObjectIds = new Set(objects.map(({ id }) => id));
  const objectTokens = [...snapshot.objectTokens]
    .sort((left, right) => compareGroundingObjects(
      left.id,
      right.id,
      relevant.objectIds,
      tokenById
    ))
    .filter((token) => projectedObjectIds.has(token.id))
    .map((token) => ({
      id: token.id,
      role: token.role,
      kind: token.kind,
      portable: token.portable,
      status: token.status,
      state: token.state,
      authority: token.authority,
      observable: token.observable,
      pose: token.pose,
      velocity: {
        linear: token.linearVelocity,
        angular: token.angularVelocity
      },
      observation: {
        frame: token.observedFrame,
        world_revision: token.observedWorldRevision,
        age_revisions: token.ageRevisions
      },
      relation: {
        distance_to_robot_m: token.relation.distanceToRobot,
        bearing_radians: token.relation.bearingRadians,
        vertical_offset_m: token.relation.verticalOffset,
        distance_to_left_wrist_m: token.relation.distanceToLeftWrist,
        distance_to_right_wrist_m: token.relation.distanceToRightWrist
      },
      contacts: token.currentContacts
    }));
  const exposeHandGeometry = requiresHandSurfaceGeometry(snapshot);
  const geometryObjects = exposeHandGeometry
    ? objects.filter((object) => (
        object.status === "visible"
          && (object.role === "manipulable" || object.articulation !== null)
      )).map((object) => ({
        object_id: object.id,
        object_center_world: object.pose.position,
        object_size: object.size,
        reachable_base_placements: navigableManipulationBasePlacements(
          snapshot,
          object.id
        )
          .sort((left, right) => left.ikResidualMeters - right.ikResidualMeters)
          .slice(0, MOTION_GROUNDING_BASE_PLACEMENT_LIMIT)
          .map((entry) => ({
            interaction_point_id: entry.interactionPointId ?? null,
            hand_surface: entry.handSurface,
            root_world_target: entry.rootWorldTarget,
            root_yaw_radians: entry.rootYawRadians,
            wrist_world_target: entry.wristWorldTarget,
            ik_residual_m: entry.ikResidualMeters,
            navigation_validation_required: true
          })),
        hands: Object.fromEntries(([
          "left",
          "right"
        ] as const).map((hand) => {
          const wrist = robot.links[
            hand === "left" ? "left_wrist_yaw_link" : "right_wrist_yaw_link"
          ];
          return [hand, {
            current_wrist_world: wrist.position,
            interaction_alignments: modelInteractionAlignments(
              snapshot,
              object.id,
              hand,
              wrist.position
            ).slice(0, MOTION_GROUNDING_INTERACTION_POINT_LIMIT)
          }];
        }))
      }))
    : [];
  const skillCatalog = snapshot.interaction.skill_catalog;
  const currentWaypoint = snapshot.navigation.waypointIndex === null
    ? null
    : snapshot.navigation.waypoints[snapshot.navigation.waypointIndex] ?? null;
  return {
    protocol: "humanoid-motion-grounding-v1",
    frame: snapshot.frame,
    world_revision: snapshot.worldRevision,
    control_authority: {
      physics_backend: "mujoco",
      learned_policy: robot.controller.learnedPolicy
        ? {
            runtime: robot.controller.learnedPolicy.runtime,
            capabilities: robot.controller.learnedPolicy.capabilities,
            observation_space: robot.controller.learnedPolicy.observationSpace,
            action_space: robot.controller.learnedPolicy.actionSpace
          }
        : null,
      reference_control: robot.controller.capabilityRouting
        ? {
            strategy: robot.controller.capabilityRouting.strategy,
            implementation: robot.controller.capabilityRouting.fallback.implementation
          }
        : null,
      active_control: {
        mode: controllerExecution.mode,
        implementation: controllerExecution.activeImplementation,
        transition: controllerExecution.transition
          ? {
              from_implementation: controllerExecution.transition.fromImplementation,
              to_implementation: controllerExecution.transition.toImplementation,
              progress: controllerExecution.transition.progress
            }
          : null
      },
      motion_generator: snapshot.motionGenerator
    },
    sensor: {
      position: snapshot.sensor.position,
      maximum_range_m: snapshot.sensor.maximumRange,
      horizontal_field_of_view_radians: snapshot.sensor.horizontalFieldOfView,
      vertical_field_of_view_radians: snapshot.sensor.verticalFieldOfView
    },
    robot: {
      root: {
        position: robot.rootPosition,
        rotation: robot.rootRotation,
        heading: {
          yaw_radians: rootYaw,
          forward_world: { x: Math.sin(rootYaw), y: 0, z: Math.cos(rootYaw) }
        }
      },
      fallen: robot.fallen,
      balance: robot.balance,
      feet: robot.feet,
      end_effectors: Object.fromEntries(HUMANOID_END_EFFECTORS.map((endEffector) => [
        endEffector,
        {
          world_position: humanoidEndEffectorPosition(robot, endEffector, "world"),
          pelvis_relative_position: humanoidEndEffectorPosition(
            robot,
            endEffector,
            "pelvis"
          )
        }
      ])),
      non_foot_environment_contacts: robot.nonFootEnvironmentContacts,
      relevant_contacts: modelRelevantContacts(robot)
    },
    navigation: {
      plan_id: snapshot.navigation.planId,
      status: snapshot.navigation.status,
      target: snapshot.navigation.target,
      waypoint_index: snapshot.navigation.waypointIndex,
      waypoint_count: snapshot.navigation.waypoints.length,
      current_waypoint: currentWaypoint
    },
    spatial_belief: {
      ...snapshot.spatialBelief,
      frontiers: snapshot.spatialBelief.frontiers
        .slice(0, MOTION_GROUNDING_FRONTIER_LIMIT)
    },
    zones: [...snapshot.interaction.zones]
      .sort((left, right) => (
        Number(relevant.zoneIds.has(right.zone_id))
          - Number(relevant.zoneIds.has(left.zone_id))
          || left.robot_planar_distance_m - right.robot_planar_distance_m
          || left.zone_id.localeCompare(right.zone_id)
      )),
    objects: {
      world_model: objects,
      tokens: objectTokens
    },
    solids: [...snapshot.solidTokens]
      .sort((left, right) => (
        Number(relevant.solidIds.has(right.id))
          - Number(relevant.solidIds.has(left.id))
          || planarDistance(left.center, robot.rootPosition)
            - planarDistance(right.center, robot.rootPosition)
          || left.id.localeCompare(right.id)
      ))
      .slice(0, MOTION_GROUNDING_SOLID_LIMIT)
      .map((solid) => ({
        id: solid.id,
        source_id: solid.sourceId,
        kind: solid.kind,
        center: solid.center,
        size: solid.size,
        contact_count: solid.currentContacts.length
      })),
    carrying: snapshot.interaction.carrying,
    grasp: {
      contract_sha256: snapshot.grasp.contractSha256,
      assessments: snapshot.grasp.assessments
        .filter((assessment) => projectedObjectIds.has(assessment.object_id))
        .map((assessment) => ({
          object_id: assessment.object_id,
          hand: assessment.hand,
          phase: assessment.phase,
          verified: assessment.grasp_verified,
          reason: assessment.reason,
          contact: {
            status: assessment.evidence.contact.status,
            observed_contact_count:
              assessment.evidence.contact.observed_contact_count,
            force_qualified_contact_count:
              assessment.evidence.contact.force_qualified_contact_count,
            distinct_force_qualified_links:
              assessment.evidence.contact.distinct_force_qualified_links
          },
          support: assessment.evidence.support,
          relative_pose: assessment.evidence.relative_pose,
          lifted_hold_frames: assessment.evidence.lifted_hold_frames
        }))
    },
    manipulation_geometry: geometryObjects.length > 0
      ? { objects: geometryObjects }
      : null,
    skill_authority: {
      protocol: skillCatalog.protocol,
      contract_sha256: skillCatalog.contract_sha256,
      world_frame: skillCatalog.world_frame,
      world_revision: skillCatalog.world_revision,
      skills: skillCatalog.entries.map((entry) => ({
        id: entry.id,
        parameters: entry.parameters,
        required_affordances: entry.required_affordances,
        available: entry.available,
        unavailable_reasons: entry.unavailable_reasons,
        observable_target_ids: entry.observable_target_ids,
        observable_solid_ids: entry.observable_solid_ids,
        observable_zone_ids: entry.observable_zone_ids,
        remembered_target_ids: entry.remembered_target_ids,
        destination_ids: entry.destination_ids,
        learned_policy_ready: entry.learned_policy_ready,
        learned_policy_required_capabilities:
          entry.learned_policy_required_capabilities,
        learned_policy_missing_capabilities:
          entry.learned_policy_missing_capabilities
      }))
    }
  };
}

function groundingGoalEntities(goal: Goal | undefined): {
  objectIds: Set<string>;
  solidIds: Set<string>;
  zoneIds: Set<string>;
} {
  const objectIds = new Set<string>();
  const solidIds = new Set<string>();
  const zoneIds = new Set<string>();
  for (const predicate of goal?.predicates ?? []) {
    if ("object_id" in predicate) objectIds.add(predicate.object_id);
    if (predicate.type === "object_inside") objectIds.add(predicate.container_id);
    if (predicate.type === "object_on") objectIds.add(predicate.support_id);
    if (predicate.type === "block_removed") solidIds.add(predicate.block_id);
    if (predicate.type === "robot_in_zone"
      || predicate.type === "object_in_zone"
      || predicate.type === "object_placed") {
      zoneIds.add(predicate.zone_id);
    }
  }
  return { objectIds, solidIds, zoneIds };
}

function compareGroundingObjects(
  leftId: string,
  rightId: string,
  relevantIds: ReadonlySet<string>,
  tokens: ReadonlyMap<string, HumanoidWorldObservation["objectTokens"][number]>
): number {
  const left = tokens.get(leftId);
  const right = tokens.get(rightId);
  return Number(relevantIds.has(rightId)) - Number(relevantIds.has(leftId))
    || Number(right?.status === "visible") - Number(left?.status === "visible")
    || (left?.relation.distanceToRobot ?? Number.POSITIVE_INFINITY)
      - (right?.relation.distanceToRobot ?? Number.POSITIVE_INFINITY)
    || leftId.localeCompare(rightId);
}

function modelObservation(snapshot: HumanoidWorldObservation): unknown {
  const robot = snapshot.robot;
  const controllerExecution = robot.controllerExecution ?? {
    protocol: "humanoid-controller-execution-v1" as const,
    mode: robot.controller.learnedPolicy
      ? "learned_policy" as const
      : "reference_control" as const,
    activeImplementation: robot.controller.implementation,
    transition: null
  };
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
    )),
    ...(snapshot.interaction.carrying?.bindings ?? []).map(({ object_id: objectId }) => (
      objectId
    ))
  ]);
  const contactReferenceObjects = (snapshot.interaction.object_world_model?.objects ?? [])
    .filter((object) => object.status === "visible"
      && (object.role === "manipulable" || object.articulation !== null)
      && embodiedObjectIds.has(object.id));
  const manipulationGeometry = exposeHandGeometry
    ? {
        objects: contactReferenceObjects.map((object) => ({
          object_id: object.id,
          object_center_world: object.pose.position,
          object_size: object.size,
          reachable_base_placements: navigableManipulationBasePlacements(
            snapshot,
            object.id
          )
            .map((entry) => ({
              interaction_point_id: entry.interactionPointId ?? null,
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
              interaction_alignments: modelInteractionAlignments(
                snapshot,
                object.id,
                hand,
                wrist.position
              )
            }];
          }))
        }))
      }
    : null;
  return {
    frame: snapshot.frame,
    world_revision: snapshot.worldRevision,
    control_authority: {
      physics_backend: "mujoco",
      learned_policy: robot.controller.learnedPolicy
        ? {
            protocol: robot.controller.learnedPolicy.protocol,
            runtime: robot.controller.learnedPolicy.runtime,
            observation_space: robot.controller.learnedPolicy.observationSpace,
            observation_features:
              robot.controller.learnedPolicy.observationFeatures ?? [],
            action_space: robot.controller.learnedPolicy.actionSpace,
            capabilities: robot.controller.learnedPolicy.capabilities
          }
        : null,
      reference_control: robot.controller.capabilityRouting
        ? {
            protocol: robot.controller.capabilityRouting.protocol,
            strategy: robot.controller.capabilityRouting.strategy,
            mode: robot.controller.capabilityRouting.fallback.mode,
            implementation:
              robot.controller.capabilityRouting.fallback.implementation
          }
        : null,
      active_control: {
        protocol: controllerExecution.protocol,
        mode: controllerExecution.mode,
        implementation: controllerExecution.activeImplementation,
        transition: controllerExecution.transition
          ? {
              from_implementation:
                controllerExecution.transition.fromImplementation,
              to_implementation: controllerExecution.transition.toImplementation,
              progress: controllerExecution.transition.progress,
              duration_seconds: controllerExecution.transition.durationSeconds
            }
          : null
      },
      motion_generator: snapshot.motionGenerator
    },
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

function modelInteractionAlignments(
  snapshot: HumanoidWorldObservation,
  objectId: string,
  hand: "left" | "right",
  currentWristWorld: { x: number; y: number; z: number }
): unknown[] {
  const selected = new Map<
    string,
    HumanoidWorldObservation["manipulationReachability"][number]
  >();
  for (const candidate of snapshot.manipulationReachability) {
    if (candidate.objectId !== objectId
      || !candidate.handSurface.startsWith(`${hand}_`)) continue;
    const key = candidate.interactionPointId ?? "";
    const current = selected.get(key);
    if (!current || compareInteractionAlignment(candidate, current) < 0) {
      selected.set(key, candidate);
    }
  }
  return [...selected.values()]
    .sort((left, right) => (
      (left.interactionPointId ?? "").localeCompare(right.interactionPointId ?? "")
    ))
    .map((alignment) => {
      const target = alignment.wristWorldTarget;
      const delta = {
        x: target.x - currentWristWorld.x,
        y: target.y - currentWristWorld.y,
        z: target.z - currentWristWorld.z
      };
      return {
        interaction_point_id: alignment.interactionPointId ?? null,
        hand_surface: alignment.handSurface,
        wrist_world_target: target,
        wrist_world_orientation: alignment.wristWorldOrientation ?? null,
        delta_from_current_wrist_world: delta,
        distance_from_current_wrist_m: Math.hypot(delta.x, delta.y, delta.z),
        ik_reference_reachable: alignment.ikReferenceReachable,
        ik_residual_m: alignment.ikResidualMeters
      };
    });
}

function compareInteractionAlignment(
  left: HumanoidWorldObservation["manipulationReachability"][number],
  right: HumanoidWorldObservation["manipulationReachability"][number]
): number {
  if (left.ikReferenceReachable !== right.ikReferenceReachable) {
    return left.ikReferenceReachable ? -1 : 1;
  }
  const leftResidual = left.ikResidualMeters ?? Number.POSITIVE_INFINITY;
  const rightResidual = right.ikResidualMeters ?? Number.POSITIVE_INFINITY;
  return leftResidual !== rightResidual
    ? leftResidual - rightResidual
    : left.handSurface.localeCompare(right.handSurface);
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
  if ((snapshot.interaction.object_world_model?.objects ?? []).some((object) => (
    object.status === "visible" && object.articulation !== null
      && planarDistance(object.pose.position, snapshot.robot.rootPosition) <= 1.5
  ))) return true;
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
  const worldModelObjects = observation.interaction.object_world_model?.objects;
  // Transitional/lightweight observers can publish the v2 interaction envelope
  // before they have materialized any object-world-model entries.  In that
  // state the visible object tokens are still the current sensor evidence; an
  // empty envelope must not erase their IK/base-placement diagnosis.
  if (!worldModelObjects || worldModelObjects.length === 0) {
    return legacyManipulationBasePlacementRequirement(observation);
  }
  const objects = worldModelObjects.flatMap((object) => {
    if (object.status !== "visible"
      || object.role !== "manipulable" && object.articulation === null) return [];
    const reachability = observation.manipulationReachability.filter((entry) => (
      entry.objectId === object.id
    ));
    if (reachability.length === 0
      || reachability.some((entry) => entry.ikReferenceReachable)) return [];
    const observedPlacements = observation.manipulationBasePlacements.filter((entry) => (
      entry.objectId === object.id
    ));
    const placements = navigableManipulationBasePlacements(observation, object.id);
    return observedPlacements.length === 0
      ? []
      : [{
          objectId: object.id,
          objectCenterWorld: { ...object.pose.position },
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

function legacyManipulationBasePlacementRequirement(
  observation: HumanoidWorldObservation
): ManipulationBasePlacementRequirement | null {
  const objects = observation.objectTokens.flatMap((object) => {
    if (!object.portable || object.status !== "visible"
      || !object.observable || object.position === null) return [];
    const reachability = observation.manipulationReachability.filter((entry) => (
      entry.objectId === object.id
    ));
    if (reachability.length === 0
      || reachability.some((entry) => entry.ikReferenceReachable)) return [];
    const observedPlacements = observation.manipulationBasePlacements.filter((entry) => (
      entry.objectId === object.id
    ));
    const placements = navigableManipulationBasePlacements(observation, object.id);
    return observedPlacements.length === 0 ? [] : [{
      objectId: object.id,
      objectCenterWorld: { ...object.position },
      placements: structuredClone(placements)
    }];
  });
  return objects.length === 0 ? null : {
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
    interaction_point_id: placement.interactionPointId ?? null,
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

function humanoidSafetyRecoveryAuthorized(
  interrupt: NeuralSafetyInterrupt | undefined,
  observation: HumanoidWorldObservation,
  invocation: HumanoidSkillInvocation
): boolean {
  return invocation.skill === "stabilize"
    && humanoidRecoverySafetyInterruptIsCurrent(interrupt, {
      worldRevision: observation.worldRevision
    });
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

function persistedGroundingObservation(
  value: JsonValue,
  currentWorldRevision: number,
  latestPhysicalExecutionRevision: number
): HumanoidWorldObservation | null {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return null;
  const frame = value.frame;
  const worldRevision = value.worldRevision;
  if (typeof frame !== "number"
    || !Number.isSafeInteger(frame)
    || typeof worldRevision !== "number"
    || !Number.isSafeInteger(worldRevision)
    || worldRevision > currentWorldRevision
    || worldRevision < latestPhysicalExecutionRevision) {
    return null;
  }
  return structuredClone(value) as unknown as HumanoidWorldObservation;
}

function jsonObject(value: JsonValue): Record<string, JsonValue> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value
    : undefined;
}

function groundingRejection(input: {
  grounding: HumanoidGroundingReceipt;
  channels: HumanoidBodyChannel[];
  planningTransactionId: string;
  planId: string;
  planKind?: "motion" | "navigation" | "recovery";
  planningAction?: HumanoidPlanningActionName;
}): {
  accepted: false;
  code: "execution_grounding_rejected";
  channels: HumanoidBodyChannel[];
  detail: JsonValue;
} {
  return {
    accepted: false,
    code: "execution_grounding_rejected",
    channels: [...input.channels],
    detail: jsonValue({
      planning_transaction_id: input.planningTransactionId,
      plan_id: input.planId,
      ...(input.planKind ? { autonomous_plan_kind: input.planKind } : {}),
      ...(input.planningAction ? { planning_action: input.planningAction } : {}),
      automatic_actuation: false,
      failed_obligation_ids: input.grounding.failed_obligation_ids,
      grounding_receipt: input.grounding
    })
  };
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

function sameHumanoidSkillInvocation(
  left: HumanoidSkillInvocation,
  right: HumanoidSkillInvocation
): boolean {
  return stableJson(left) === stableJson(right);
}

function combineExecutionSignals(
  runtimeSignal: AbortSignal | undefined,
  invocationSignal: AbortSignal | undefined
): AbortSignal | undefined {
  if (!runtimeSignal) return invocationSignal;
  if (!invocationSignal || invocationSignal === runtimeSignal) return runtimeSignal;
  return AbortSignal.any([runtimeSignal, invocationSignal]);
}
