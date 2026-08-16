import { randomUUID } from "node:crypto";
import type {
  AgentManifest,
  HistoricalAgentAuthorityManifest
} from "../../domain/agent-manifest.js";
import type { HumanoidActionName } from "../../domain/humanoid-action.js";
import {
  authorityForToolCall,
  ModelCallLifecycleRecordSchema,
  ModelDecisionRefSchema,
  modelPayloadSha256,
  rebuildModelCallAuthorities,
  type ModelCallAuthority,
  type ModelCallLifecycleRecord,
  type ModelDecisionRef
} from "../../domain/model-call-authority.js";
import {
  sameAutonomousCycle,
  type AutonomousCycleRef
} from "../../domain/autonomous-cycle.js";
import type { HumanoidReplanModelCall } from
  "../../domain/humanoid-replan-budget.js";
import type {
  GoalHarnessValidation,
  GoalModelSource
} from "../../domain/goal-epoch.js";
import type { Scenario, TaskNode } from "../../domain/schema.js";
import {
  goalPredicateIsObservable,
  type GoalEvidenceArtifact
} from "./goal-evidence.js";
import { HUMANOID_NEURAL_AGENT_IDS } from "./neural-hierarchy-contract.js";

export interface HumanoidModelToolCallAuthority {
  tool_call_id: string;
  tool_name: string;
  arguments_sha256: string;
  normalized_arguments_sha256?: string | undefined;
  deterministic_delegation?: {
    contract_id: "grounding_monitor_v1" | "execution_gate_v1";
    source_input: unknown;
    action_input_sha256: string;
  };
}

interface ManifestAuthority {
  agentId: string;
  epochId: string;
  identitySha256: string;
}

type StartedModelCall = Extract<
  ModelCallLifecycleRecord,
  { lifecycle: "started" }
>;
type CompletedModelCall = Extract<
  ModelCallLifecycleRecord,
  { lifecycle: "completed" }
>;
type FailedModelCall = Extract<
  ModelCallLifecycleRecord,
  { lifecycle: "failed" }
>;

export class HumanoidModelAuthority {
  readonly #goalManager: ManifestAuthority;
  readonly #groundingManager: ManifestAuthority;
  readonly #executionManager: ManifestAuthority;
  readonly #authorizedGroundingManagers: readonly ManifestAuthority[];
  readonly #authorizedExecutionManagers: readonly ManifestAuthority[];
  readonly #authorizedGoalManagers: readonly ManifestAuthority[];
  readonly #actionAgents: ReadonlyMap<string, readonly ManifestAuthority[]>;
  readonly #legacyV2RuntimeAuthorityEnabled: boolean;
  readonly #appendRecord: (record: ModelCallLifecycleRecord) => Promise<void>;
  readonly #started = new Map<string, StartedModelCall>();
  readonly #terminalIds = new Set<string>();
  #authorities: Map<string, ModelCallAuthority>;

  private constructor(input: {
    goalManager: ManifestAuthority;
    groundingManager: ManifestAuthority;
    executionManager: ManifestAuthority;
    authorizedGroundingManagers: readonly ManifestAuthority[];
    authorizedExecutionManagers: readonly ManifestAuthority[];
    authorizedGoalManagers: readonly ManifestAuthority[];
    actionAgents: ReadonlyMap<string, readonly ManifestAuthority[]>;
    legacyV2RuntimeAuthorityEnabled: boolean;
    records: readonly ModelCallLifecycleRecord[];
    appendRecord: (record: ModelCallLifecycleRecord) => Promise<void>;
  }) {
    this.#goalManager = input.goalManager;
    this.#groundingManager = input.groundingManager;
    this.#executionManager = input.executionManager;
    this.#authorizedGroundingManagers = input.authorizedGroundingManagers;
    this.#authorizedExecutionManagers = input.authorizedExecutionManagers;
    this.#authorizedGoalManagers = input.authorizedGoalManagers;
    this.#actionAgents = input.actionAgents;
    this.#legacyV2RuntimeAuthorityEnabled = input.legacyV2RuntimeAuthorityEnabled;
    this.#appendRecord = input.appendRecord;
    this.#authorities = rebuildModelCallAuthorities(input.records);
    for (const record of input.records) {
      if (record.lifecycle === "started") {
        this.#started.set(record.model_call_id, record);
      } else {
        this.#started.delete(record.model_call_id);
        this.#terminalIds.add(record.model_call_id);
      }
    }
  }

  static restore(input: {
    manifest: AgentManifest;
    archivedManifests?: readonly HistoricalAgentAuthorityManifest[];
    nodes: Readonly<Record<string, TaskNode>>;
    records: readonly unknown[];
    appendRecord: (record: ModelCallLifecycleRecord) => Promise<void>;
  }): HumanoidModelAuthority {
    const goalManager = goalManagerIdentity(input.manifest);
    assertManifestNode(input.nodes, goalManager.agent_id, "Goal Manager");
    const goalAuthority = manifestAuthority(input.manifest, goalManager.agent_id);
    const authorizedGoalManagers = [
      goalAuthority,
      ...(input.archivedManifests ?? []).map((archived) => {
        return historicalManifestAuthority(
          archived,
          archived.goal_manager_agent_id
        );
      })
    ];
    const goalAuthorityKeys = new Set<string>();
    for (const authority of authorizedGoalManagers) {
      const key = `${authority.epochId}:${authority.identitySha256}:${authority.agentId}`;
      if (goalAuthorityKeys.has(key)) {
        throw new Error(`Duplicate Goal Manager manifest authority: ${authority.epochId}`);
      }
      goalAuthorityKeys.add(key);
    }
    const actionAgents = new Map<string, ManifestAuthority[]>();
    const addActionAgent = (
      sourceManifest: AgentManifest,
      agentId: string,
      role: string
    ): void => {
      assertManifestNode(input.nodes, agentId, role);
      const authority = manifestAuthority(sourceManifest, agentId);
      const existing = actionAgents.get(agentId) ?? [];
      if (!existing.some((candidate) => (
        candidate.epochId === authority.epochId
          && candidate.identitySha256 === authority.identitySha256
      ))) {
        existing.push(authority);
        actionAgents.set(agentId, existing);
      }
    };
    for (const agent of manifestAgentIdentities(input.manifest)) {
      if (agent.agent_id === goalManager.agent_id) continue;
      addActionAgent(input.manifest, agent.agent_id, "Current hierarchy node");
    }
    const groundingManager = manifestAuthority(
      input.manifest,
      groundingManagerIdentity(input.manifest).agent_id
    );
    const executionManager = manifestAuthority(
      input.manifest,
      executionManagerIdentity(input.manifest).agent_id
    );
    const authorizedGroundingManagers = [groundingManager];
    const authorizedExecutionManagers = [executionManager];
    for (const archived of input.archivedManifests ?? []) {
      for (const agentId of archived.agent_ids) {
        const authority = historicalManifestAuthority(archived, agentId);
        const existing = actionAgents.get(agentId) ?? [];
        if (!existing.some((candidate) => sameManifestAuthority(candidate, authority))) {
          existing.push(authority);
          actionAgents.set(agentId, existing);
        }
      }
      pushUniqueAuthority(
        authorizedGroundingManagers,
        historicalManifestAuthority(archived, archived.grounding_manager_agent_id)
      );
      pushUniqueAuthority(
        authorizedExecutionManagers,
        historicalManifestAuthority(archived, archived.execution_manager_agent_id)
      );
    }
    const records = input.records.map((record) => (
      ModelCallLifecycleRecordSchema.parse(record)
    ));
    return new HumanoidModelAuthority({
      goalManager: goalAuthority,
      groundingManager,
      executionManager,
      authorizedGroundingManagers,
      authorizedExecutionManagers,
      authorizedGoalManagers,
      actionAgents,
      legacyV2RuntimeAuthorityEnabled: input.manifest.version !== 3,
      records,
      appendRecord: input.appendRecord
    });
  }

  async recordStarted(
    agentId: string,
    cycle?: AutonomousCycleRef,
    modelCallId = randomUUID(),
    at = new Date().toISOString(),
    replanBudgetCall?: HumanoidReplanModelCall
  ): Promise<StartedModelCall> {
    const parsed = ModelCallLifecycleRecordSchema.parse({
      version: 1,
      lifecycle: "started",
      model_call_id: modelCallId,
      agent_id: agentId,
      ...(cycle ? { cycle } : {}),
      ...(replanBudgetCall
        ? { replan_budget_call: replanBudgetCall }
        : {}),
      at
    });
    if (parsed.lifecycle !== "started") {
      throw new Error("Model call start parsed as another lifecycle");
    }
    const record = parsed;
    if (this.#started.has(record.model_call_id)
      || this.#terminalIds.has(record.model_call_id)) {
      throw new Error(`Duplicate model call start: ${record.model_call_id}`);
    }
    await this.#appendRecord(record);
    this.#started.set(record.model_call_id, record);
    return structuredClone(record);
  }

  async recordCompleted(input: {
    modelCallId: string;
    agentId: string;
    responseId: string;
    responseOutputSha256: string;
    toolCalls: Array<{
      toolCallId: string;
      toolName: string;
      argumentsSha256: string;
    }>;
    at?: string;
  }): Promise<CompletedModelCall> {
    const origin = this.#requiredStarted(input.modelCallId, input.agentId);
    const parsed = ModelCallLifecycleRecordSchema.parse({
      version: 1,
      lifecycle: "completed",
      model_call_id: input.modelCallId,
      agent_id: input.agentId,
      response_id: input.responseId,
      response_output_sha256: input.responseOutputSha256,
      tool_calls: input.toolCalls.map((toolCall) => ({
        tool_call_id: toolCall.toolCallId,
        tool_name: toolCall.toolName,
        arguments_sha256: toolCall.argumentsSha256
      })),
      ...(origin.cycle ? { cycle: origin.cycle } : {}),
      at: input.at ?? new Date().toISOString()
    });
    if (parsed.lifecycle !== "completed") {
      throw new Error("Completed model call parsed as another lifecycle");
    }
    const record = parsed;
    this.#assertTerminalRecord(origin, record);
    const authority: ModelCallAuthority = {
      ...record,
      started_at: origin.at
    };
    await this.#appendRecord(record);
    this.#started.delete(record.model_call_id);
    this.#terminalIds.add(record.model_call_id);
    this.#authorities.set(record.model_call_id, authority);
    return structuredClone(record);
  }

  async recordFailed(
    modelCallId: string,
    agentId: string,
    at = new Date().toISOString()
  ): Promise<FailedModelCall> {
    const origin = this.#requiredStarted(modelCallId, agentId);
    const parsed = ModelCallLifecycleRecordSchema.parse({
      version: 1,
      lifecycle: "failed",
      model_call_id: modelCallId,
      agent_id: agentId,
      ...(origin.cycle ? { cycle: origin.cycle } : {}),
      at
    });
    if (parsed.lifecycle !== "failed") {
      throw new Error("Failed model call parsed as another lifecycle");
    }
    const record = parsed;
    this.#assertTerminalRecord(origin, record);
    await this.#appendRecord(record);
    this.#started.delete(record.model_call_id);
    this.#terminalIds.add(record.model_call_id);
    return structuredClone(record);
  }

  startedCalls(): StartedModelCall[] {
    return [...this.#started.values()].map((record) => structuredClone(record));
  }

  goalHarness(input: {
    evidence: ReadonlyMap<string, GoalEvidenceArtifact>;
    scenario: Scenario;
  }): GoalHarnessValidation {
    return {
      authorized_model_sources: this.#authorizedGoalManagers.map((authority) => ({
        agent_id: authority.agentId,
        agent_manifest_sha256: authority.identitySha256,
        agent_manifest_epoch_id: authority.epochId
      })),
      is_model_call_authoritative: (source, expectedToolName) => {
        const authority = this.#authorities.get(source.model_call_id);
        if (!authority
          || authority.agent_id !== source.agent_id
          || authority.response_id !== source.response_id
          || authority.response_output_sha256 !== source.response_output_sha256) {
          return false;
        }
        const matches = authority.tool_calls.filter((toolCall) => (
          toolCall.tool_call_id === source.tool_call_id
            && toolCall.arguments_sha256 === source.tool_arguments_sha256
            && toolCall.tool_name === expectedToolName
        ));
        return matches.length === 1;
      },
      evidence_by_ref: (ref) => input.evidence.get(ref)?.evidence,
      is_predicate_observable: ({
        predicate,
        world_revision: worldRevision,
        evidence_refs: evidenceRefs
      }) => goalPredicateIsObservable({
        predicate,
        worldRevision,
        evidenceRefs,
        artifacts: input.evidence,
        scenario: input.scenario
      })
    };
  }

  goalModelSource(
    toolAuthority: HumanoidModelToolCallAuthority,
    expectedToolName: string,
    normalizedInput: unknown
  ): GoalModelSource {
    const manifest = this.#goalManager;
    if (toolAuthority.tool_name !== expectedToolName) {
      throw new Error(
        `Goal tool authority mismatch: expected ${expectedToolName}, received ${toolAuthority.tool_name}`
      );
    }
    const authority = authorityForToolCall(
      this.#authorities,
      manifest.agentId,
      toolAuthority.tool_call_id,
      expectedToolName
    );
    const toolCall = authority?.tool_calls.find((entry) => (
      entry.tool_call_id === toolAuthority.tool_call_id
        && entry.tool_name === expectedToolName
    ));
    if (!authority || !toolCall
      || toolCall.arguments_sha256 !== toolAuthority.arguments_sha256) {
      throw new Error(
        `Goal tool call has no completed model response authority: ${toolAuthority.tool_call_id}`
      );
    }
    const normalizedArgumentsSha256 = modelPayloadSha256(normalizedInput);
    if ((toolAuthority.normalized_arguments_sha256
      ?? toolAuthority.arguments_sha256) !== normalizedArgumentsSha256) {
      throw new Error(
        `Goal tool arguments have no model authority: ${toolAuthority.tool_call_id}`
      );
    }
    return {
      agent_id: manifest.agentId,
      agent_manifest_sha256: manifest.identitySha256,
      agent_manifest_epoch_id: manifest.epochId,
      model_call_id: authority.model_call_id,
      response_id: authority.response_id,
      response_output_sha256: authority.response_output_sha256,
      tool_call_id: toolCall.tool_call_id,
      tool_arguments_sha256: toolCall.arguments_sha256,
      ...(normalizedArgumentsSha256 === toolCall.arguments_sha256
        ? {}
        : { normalized_tool_arguments_sha256: normalizedArgumentsSha256 })
    };
  }

  actionModelSource(input: {
    toolAuthority: HumanoidModelToolCallAuthority;
    expectedToolName: HumanoidActionName;
    actionInput: unknown;
    transactionId: string;
    agentId: string;
  }): ModelDecisionRef {
    const transactionId = input.transactionId.trim();
    const actorAgentId = input.agentId.trim();
    if (input.toolAuthority.tool_call_id !== transactionId) {
      throw new Error(`Humanoid action tool authority mismatch: ${transactionId}`);
    }
    const normalizedArgumentsSha256 = modelPayloadSha256(input.actionInput);
    const delegation = input.toolAuthority.deterministic_delegation;
    const requiredDelegation = requiredDeterministicDelegation(
      input.expectedToolName,
      actorAgentId,
      this.#legacyV2RuntimeAuthorityEnabled
    );
    if (requiredDelegation && delegation?.contract_id !== requiredDelegation) {
      throw new Error(
        `Humanoid action requires hierarchical Manager delegation: ${transactionId}`
      );
    }
    const sourceAgentId = delegation
      ? this.#deterministicDecisionSource({
          contractId: delegation.contract_id,
          sourceToolName: input.toolAuthority.tool_name,
          action: input.expectedToolName,
          actorAgentId
        })
      : actorAgentId;
    if (delegation) {
      const normalizedSourceArgumentsSha256 = modelPayloadSha256(
        delegation.source_input
      );
      if ((input.toolAuthority.normalized_arguments_sha256
        ?? input.toolAuthority.arguments_sha256) !== normalizedSourceArgumentsSha256) {
        throw new Error(
          `Deterministic humanoid delegation input has no model authority: ${transactionId}`
        );
      }
      if (delegation.action_input_sha256 !== normalizedArgumentsSha256) {
        throw new Error(
          `Deterministic humanoid action input is not bound to its delegation: ${transactionId}`
        );
      }
      this.#assertDeterministicActionMapping({
        contractId: delegation.contract_id,
        sourceToolName: input.toolAuthority.tool_name,
        sourceInput: delegation.source_input,
        action: input.expectedToolName,
        actionInput: input.actionInput,
        actorAgentId
      });
    } else {
      if (input.toolAuthority.tool_name !== input.expectedToolName) {
        throw new Error(`Humanoid action tool authority mismatch: ${transactionId}`);
      }
      if ((input.toolAuthority.normalized_arguments_sha256
          ?? input.toolAuthority.arguments_sha256) !== normalizedArgumentsSha256) {
        throw new Error(
          `Humanoid action tool arguments have no model authority: ${transactionId}`
        );
      }
    }
    const manifest = this.#actionAgents.get(sourceAgentId)?.[0];
    if (!manifest) {
      throw new Error(`Humanoid action decision source has no manifest authority: ${sourceAgentId}`);
    }
    const authority = authorityForToolCall(
      this.#authorities,
      sourceAgentId,
      transactionId,
      input.toolAuthority.tool_name
    );
    const toolCall = authority?.tool_calls.find((entry) => (
      entry.tool_call_id === transactionId
        && entry.tool_name === input.toolAuthority.tool_name
    ));
    if (!authority || !authority.cycle || !toolCall
      || toolCall.arguments_sha256 !== input.toolAuthority.arguments_sha256) {
      throw new Error(
        `Humanoid action tool call has no completed model response authority: ${transactionId}`
      );
    }
    return ModelDecisionRefSchema.parse({
      agent_id: sourceAgentId,
      agent_manifest_sha256: manifest.identitySha256,
      agent_manifest_epoch_id: manifest.epochId,
      model_call_id: authority.model_call_id,
      response_id: authority.response_id,
      response_output_sha256: authority.response_output_sha256,
      tool_call_id: toolCall.tool_call_id,
      tool_arguments_sha256: toolCall.arguments_sha256,
      ...(normalizedArgumentsSha256 === toolCall.arguments_sha256
        ? {}
        : { normalized_tool_arguments_sha256: normalizedArgumentsSha256 })
    });
  }

  assertDecisionCycleActive(
    decision: ModelDecisionRef,
    cycle: AutonomousCycleRef
  ): void {
    const authority = this.#authorities.get(decision.model_call_id);
    if (!authority?.cycle || !sameAutonomousCycle(authority.cycle, cycle)) {
      throw new Error(
        `Humanoid action model decision belongs to another autonomous cycle: ${decision.tool_call_id}`
      );
    }
  }

  assertActionDecision(input: {
    rawDecision: ModelDecisionRef;
    expectedToolName: HumanoidActionName;
    actionInput: unknown;
    transactionId: string;
    agentId: string;
    cycle: AutonomousCycleRef;
    toolAuthority?: HumanoidModelToolCallAuthority;
  }): void {
    const decision = ModelDecisionRefSchema.parse(input.rawDecision);
    const manifest = this.#actionAgents.get(decision.agent_id)?.find((candidate) => (
      candidate.identitySha256 === decision.agent_manifest_sha256
        && candidate.epochId === decision.agent_manifest_epoch_id
    ));
    const authority = this.#authorities.get(decision.model_call_id);
    const toolCalls = authority?.tool_calls.filter((entry) => (
      entry.tool_call_id === input.transactionId
        && entry.arguments_sha256 === decision.tool_arguments_sha256
    ));
    const sourceToolName = toolCalls?.length === 1 ? toolCalls[0]!.tool_name : undefined;
    const requiredDelegation = requiredDeterministicDelegation(
      input.expectedToolName,
      input.agentId,
      this.#legacyV2RuntimeAuthorityEnabled
    );
    const sourceAuthorized = requiredDelegation
      ? sourceToolName !== undefined && this.#deterministicDecisionSource({
          sourceToolName,
          action: input.expectedToolName,
          actorAgentId: input.agentId,
          contractId: requiredDelegation,
          sourceAgentId: decision.agent_id
        }) === decision.agent_id
      : sourceToolName === input.expectedToolName
        ? decision.agent_id === input.agentId
        : sourceToolName !== undefined && this.#deterministicDecisionSource({
            sourceToolName,
            action: input.expectedToolName,
            actorAgentId: input.agentId,
            sourceAgentId: decision.agent_id
          }) === decision.agent_id;
    const normalizedArgumentsSha256 = modelPayloadSha256(input.actionInput);
    if (requiredDelegation) {
      const envelope = input.toolAuthority;
      const delegation = envelope?.deterministic_delegation;
      if (!envelope || !delegation
        || envelope.tool_call_id !== input.transactionId
        || envelope.tool_name !== sourceToolName
        || envelope.arguments_sha256 !== decision.tool_arguments_sha256
        || delegation.contract_id !== requiredDelegation
        || modelPayloadSha256(delegation.source_input)
          !== (envelope.normalized_arguments_sha256
            ?? envelope.arguments_sha256)
        || delegation.action_input_sha256 !== normalizedArgumentsSha256) {
        throw new Error(
          `Humanoid action decision has no durable deterministic delegation: ${input.transactionId}`
        );
      }
      this.#assertDeterministicActionMapping({
        contractId: delegation.contract_id,
        sourceToolName: envelope.tool_name,
        sourceInput: delegation.source_input,
        action: input.expectedToolName,
        actionInput: input.actionInput,
        actorAgentId: input.agentId
      });
    }
    if (!manifest
      || decision.agent_manifest_sha256 !== manifest.identitySha256
      || decision.agent_manifest_epoch_id !== manifest.epochId
      || decision.tool_call_id !== input.transactionId
      || !authority
      || authority.agent_id !== decision.agent_id
      || authority.response_id !== decision.response_id
      || authority.response_output_sha256 !== decision.response_output_sha256
      || !sameAutonomousCycle(authority.cycle, input.cycle)
      || toolCalls?.length !== 1
      || !sourceAuthorized
      || (decision.normalized_tool_arguments_sha256
        ?? decision.tool_arguments_sha256) !== normalizedArgumentsSha256) {
      throw new Error(`Humanoid action decision is not authoritative: ${input.transactionId}`);
    }
  }

  #deterministicDecisionSource(input: {
    sourceToolName: string;
    action: HumanoidActionName;
    actorAgentId: string;
    contractId?: "grounding_monitor_v1" | "execution_gate_v1";
    sourceAgentId?: string;
  }): string {
    if ((this.#legacyV2RuntimeAuthorityEnabled
      && input.sourceToolName === "delegate_humanoid_sentry"
      && input.actorAgentId === "humanoid-sentry"
      || input.sourceToolName === "capture_sensor_fusion"
      && input.actorAgentId === HUMANOID_NEURAL_AGENT_IDS.sensorFusion)
      && input.action === "observe_humanoid"
      && (input.contractId === undefined
        || input.contractId === "grounding_monitor_v1")) {
      return selectAuthorizedManager(
        this.#authorizedGroundingManagers,
        this.#groundingManager,
        input.sourceAgentId
      );
    }
    if ((this.#legacyV2RuntimeAuthorityEnabled
      ? input.sourceToolName === "delegate_physics_executor"
      : input.sourceToolName === "execute_certified_motor_intent")
      && input.actorAgentId === HUMANOID_NEURAL_AGENT_IDS.executor
      && [
        "execute_humanoid_skill",
        "execute_whole_body_motion",
        "execute_humanoid_navigation",
        "remove_world_block"
      ].includes(input.action)
      && (input.contractId === undefined || input.contractId === "execution_gate_v1")) {
      return selectAuthorizedManager(
        this.#authorizedExecutionManagers,
        this.#executionManager,
        input.sourceAgentId
      );
    }
    throw new Error(
      `Humanoid deterministic delegation is not authoritative: ${input.sourceToolName}`
    );
  }

  #assertDeterministicActionMapping(input: {
    contractId: "grounding_monitor_v1" | "execution_gate_v1";
    sourceToolName: string;
    sourceInput: unknown;
    action: HumanoidActionName;
    actionInput: unknown;
    actorAgentId: string;
  }): void {
    if (input.contractId === "grounding_monitor_v1") {
      const validGroundingEdge = this.#legacyV2RuntimeAuthorityEnabled
        && input.sourceToolName === "delegate_humanoid_sentry"
        && input.actorAgentId === "humanoid-sentry"
        || input.sourceToolName === "capture_sensor_fusion"
        && input.actorAgentId === HUMANOID_NEURAL_AGENT_IDS.sensorFusion;
      if (!validGroundingEdge
        || input.action !== "observe_humanoid"
        || modelPayloadSha256(input.sourceInput) !== modelPayloadSha256({})
        || modelPayloadSha256(input.actionInput) !== modelPayloadSha256({})) {
        throw new Error("Grounding Monitor delegation changed its deterministic action");
      }
      return;
    }
    const source = objectRecord(input.sourceInput);
    const execution = objectRecord(source?.execution);
    const expectedExecutionTool = this.#legacyV2RuntimeAuthorityEnabled
      ? "delegate_physics_executor"
      : "execute_certified_motor_intent";
    if (input.sourceToolName !== expectedExecutionTool
      || input.actorAgentId !== HUMANOID_NEURAL_AGENT_IDS.executor
      || !execution) {
      throw new Error("Execution Gate delegation has no validated source input");
    }
    let expectedAction: HumanoidActionName | undefined;
    let expectedInput: unknown;
    if (execution.kind === "remove_world_block") {
      expectedAction = "remove_world_block";
      expectedInput = {
        solid_id: execution.solid_id,
        execution_transaction_id: execution.execution_transaction_id
      };
    } else if (execution.kind === "execute_plan") {
      const actions: Record<string, HumanoidActionName> = {
        plan_humanoid_skill: "execute_humanoid_skill",
        plan_whole_body_motion: "execute_whole_body_motion",
        plan_whole_body_motion_candidates: "execute_whole_body_motion",
        plan_humanoid_navigation: "execute_humanoid_navigation"
      };
      expectedAction = typeof execution.planning_action === "string"
        ? actions[execution.planning_action]
        : undefined;
      expectedInput = {
        planning_transaction_id: execution.planning_transaction_id
      };
    }
    if (!expectedAction
      || input.action !== expectedAction
      || modelPayloadSha256(input.actionInput) !== modelPayloadSha256(expectedInput)) {
      throw new Error("Execution Gate delegation changed its deterministic action");
    }
  }

  cycleForModelCall(modelCallId: string): AutonomousCycleRef | undefined {
    const cycle = this.#authorities.get(modelCallId)?.cycle;
    return cycle ? structuredClone(cycle) : undefined;
  }

  prune(retainModelCallIds: ReadonlySet<string>, recentTerminalLimit = 256): void {
    if (!Number.isSafeInteger(recentTerminalLimit) || recentTerminalLimit < 1) {
      throw new Error("Model authority retention limit must be positive");
    }
    const retained = new Set(retainModelCallIds);
    for (const modelCallId of this.#started.keys()) retained.add(modelCallId);
    const recentTerminalIds = [...this.#terminalIds].slice(-recentTerminalLimit);
    for (const modelCallId of recentTerminalIds) retained.add(modelCallId);
    for (const modelCallId of this.#authorities.keys()) {
      if (!retained.has(modelCallId)) this.#authorities.delete(modelCallId);
    }
    for (const modelCallId of this.#terminalIds) {
      if (!retained.has(modelCallId)) this.#terminalIds.delete(modelCallId);
    }
  }

  #requiredStarted(modelCallId: string, agentId: string): StartedModelCall {
    const origin = this.#started.get(modelCallId);
    if (!origin || origin.agent_id !== agentId || this.#terminalIds.has(modelCallId)) {
      throw new Error(`Model call terminal record has no matching start: ${modelCallId}`);
    }
    return origin;
  }

  #assertTerminalRecord(
    origin: StartedModelCall,
    record: CompletedModelCall | FailedModelCall
  ): void {
    if ((origin.cycle !== undefined || record.cycle !== undefined)
      && !sameAutonomousCycle(origin.cycle, record.cycle)) {
      throw new Error(`Model call cycle identity changed: ${record.model_call_id}`);
    }
    if (record.at < origin.at) {
      throw new Error(`Model call terminal record precedes its start: ${record.model_call_id}`);
    }
  }
}

function manifestAuthority(
  manifest: AgentManifest,
  agentId: string
): ManifestAuthority {
  return {
    agentId,
    epochId: manifest.epoch_id,
    identitySha256: manifest.identity_sha256
  };
}

function historicalManifestAuthority(
  manifest: HistoricalAgentAuthorityManifest,
  agentId: string
): ManifestAuthority {
  if (!manifest.agent_ids.includes(agentId)) {
    throw new Error(`Historical manifest has no Agent authority: ${agentId}`);
  }
  return {
    agentId,
    epochId: manifest.epoch_id,
    identitySha256: manifest.identity_sha256
  };
}

function manifestAgentIdentities(
  manifest: AgentManifest
): Array<{ agent_id: string }> {
  return Object.values(manifest.agents);
}

function goalManagerIdentity(
  manifest: AgentManifest
): { agent_id: string } {
  const identity = manifest.version === 3
    ? manifest.agents[HUMANOID_NEURAL_AGENT_IDS.goalManager]
    : manifest.agents.goal_manager;
  if (!identity || (manifest.version === 3 && identity.execution_kind !== "model_agent")) {
    throw new Error("Agent manifest has no structural Goal Valuation authority");
  }
  return identity;
}

function groundingManagerIdentity(
  manifest: AgentManifest
): { agent_id: string } {
  const identity = manifest.version === 3
    ? manifest.agents[HUMANOID_NEURAL_AGENT_IDS.perceptionManager]
    : manifest.agents.coordinator;
  if (!identity || (manifest.version === 3 && identity.execution_kind !== "model_agent")) {
    throw new Error("Agent manifest has no structural Perception Manager authority");
  }
  return identity;
}

function executionManagerIdentity(
  manifest: AgentManifest
): { agent_id: string } {
  const identity = manifest.version === 3
    ? manifest.agents[HUMANOID_NEURAL_AGENT_IDS.executionDispatcher]
    : manifest.agents.coordinator;
  if (!identity || (manifest.version === 3 && identity.execution_kind !== "model_agent")) {
    throw new Error("Agent manifest has no structural Certified Execution Dispatcher authority");
  }
  return identity;
}

function sameManifestAuthority(
  left: ManifestAuthority,
  right: ManifestAuthority
): boolean {
  return left.agentId === right.agentId
    && left.epochId === right.epochId
    && left.identitySha256 === right.identitySha256;
}

function pushUniqueAuthority(
  authorities: ManifestAuthority[],
  candidate: ManifestAuthority
): void {
  if (!authorities.some((authority) => sameManifestAuthority(authority, candidate))) {
    authorities.push(candidate);
  }
}

function selectAuthorizedManager(
  authorities: readonly ManifestAuthority[],
  current: ManifestAuthority,
  requestedAgentId?: string
): string {
  if (requestedAgentId === undefined) return current.agentId;
  const candidates = authorities.filter((authority) => (
    authority.agentId === requestedAgentId
  ));
  if (candidates.length === 0) {
    throw new Error(`Deterministic delegation has no Manager authority: ${requestedAgentId}`);
  }
  return requestedAgentId;
}

function requiredDeterministicDelegation(
  action: HumanoidActionName,
  actorAgentId: string,
  legacyV2RuntimeAuthorityEnabled: boolean
): "grounding_monitor_v1" | "execution_gate_v1" | undefined {
  if ((legacyV2RuntimeAuthorityEnabled && actorAgentId === "humanoid-sentry"
    || actorAgentId === HUMANOID_NEURAL_AGENT_IDS.sensorFusion)
    && action === "observe_humanoid") {
    return "grounding_monitor_v1";
  }
  if (actorAgentId === HUMANOID_NEURAL_AGENT_IDS.executor && [
    "execute_humanoid_skill",
    "execute_whole_body_motion",
    "execute_humanoid_navigation",
    "remove_world_block"
  ].includes(action)) {
    return "execution_gate_v1";
  }
  return undefined;
}

function assertManifestNode(
  nodes: Readonly<Record<string, TaskNode>>,
  agentId: string,
  role: string
): void {
  if (!nodes[agentId]) {
    throw new Error(`${role} manifest identity is absent from the hierarchy: ${agentId}`);
  }
}

function objectRecord(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}
