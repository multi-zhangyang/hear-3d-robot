import { randomUUID } from "node:crypto";
import type { AgentManifest } from "../../domain/agent-manifest.js";
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
import type {
  GoalHarnessValidation,
  GoalModelSource
} from "../../domain/goal-epoch.js";
import type { JsonValue, Scenario, TaskNode } from "../../domain/schema.js";
import {
  goalPredicateIsObservable,
  type GoalEvidenceArtifact
} from "./goal-evidence.js";

interface ToolCallAuthority {
  tool_call_id: string;
  tool_name: string;
  arguments_sha256: string;
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
  readonly #actionAgents: ReadonlyMap<string, ManifestAuthority>;
  readonly #appendRecord: (record: ModelCallLifecycleRecord) => Promise<void>;
  readonly #started = new Map<string, StartedModelCall>();
  readonly #terminalIds = new Set<string>();
  #authorities: Map<string, ModelCallAuthority>;

  private constructor(input: {
    goalManager: ManifestAuthority;
    actionAgents: ReadonlyMap<string, ManifestAuthority>;
    records: readonly ModelCallLifecycleRecord[];
    appendRecord: (record: ModelCallLifecycleRecord) => Promise<void>;
  }) {
    this.#goalManager = input.goalManager;
    this.#actionAgents = input.actionAgents;
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
    nodes: Readonly<Record<string, TaskNode>>;
    records: readonly JsonValue[];
    appendRecord: (record: ModelCallLifecycleRecord) => Promise<void>;
  }): HumanoidModelAuthority {
    const goalManager = input.manifest.agents.goal_manager;
    assertManifestNode(input.nodes, goalManager.agent_id, "Goal Manager");
    const goalAuthority = manifestAuthority(input.manifest, goalManager.agent_id);
    const actionAgents = new Map<string, ManifestAuthority>();
    for (const role of ["sentry", "motion", "executor"] as const) {
      const agent = input.manifest.agents[role];
      assertManifestNode(input.nodes, agent.agent_id, role);
      if (agent.agent_id === goalManager.agent_id || actionAgents.has(agent.agent_id)) {
        throw new Error(`Agent manifest reuses an authority identity: ${agent.agent_id}`);
      }
      actionAgents.set(
        agent.agent_id,
        manifestAuthority(input.manifest, agent.agent_id)
      );
    }
    const records = input.records.map((record) => (
      ModelCallLifecycleRecordSchema.parse(record)
    ));
    return new HumanoidModelAuthority({
      goalManager: goalAuthority,
      actionAgents,
      records,
      appendRecord: input.appendRecord
    });
  }

  async recordStarted(
    agentId: string,
    cycle?: AutonomousCycleRef,
    modelCallId = randomUUID(),
    at = new Date().toISOString()
  ): Promise<StartedModelCall> {
    const parsed = ModelCallLifecycleRecordSchema.parse({
      version: 1,
      lifecycle: "started",
      model_call_id: modelCallId,
      agent_id: agentId,
      ...(cycle ? { cycle } : {}),
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

  goalHarness(input: {
    evidence: ReadonlyMap<string, GoalEvidenceArtifact>;
    scenario: Scenario;
  }): GoalHarnessValidation {
    const manifest = this.#goalManager;
    return {
      authorized_model_sources: [{
        agent_id: manifest.agentId,
        agent_manifest_sha256: manifest.identitySha256,
        agent_manifest_epoch_id: manifest.epochId
      }],
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
    toolAuthority: ToolCallAuthority,
    expectedToolName: string
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
    return {
      agent_id: manifest.agentId,
      agent_manifest_sha256: manifest.identitySha256,
      agent_manifest_epoch_id: manifest.epochId,
      model_call_id: authority.model_call_id,
      response_id: authority.response_id,
      response_output_sha256: authority.response_output_sha256,
      tool_call_id: toolCall.tool_call_id,
      tool_arguments_sha256: toolCall.arguments_sha256
    };
  }

  actionModelSource(input: {
    toolAuthority: ToolCallAuthority;
    expectedToolName: HumanoidActionName;
    actionInput: unknown;
    transactionId: string;
    agentId: string;
  }): ModelDecisionRef {
    const transactionId = input.transactionId.trim();
    const agentId = input.agentId.trim();
    if (input.toolAuthority.tool_name !== input.expectedToolName
      || input.toolAuthority.tool_call_id !== transactionId) {
      throw new Error(`Humanoid action tool authority mismatch: ${transactionId}`);
    }
    const argumentsSha256 = modelPayloadSha256(input.actionInput);
    if (input.toolAuthority.arguments_sha256 !== argumentsSha256) {
      throw new Error(`Humanoid action tool arguments have no model authority: ${transactionId}`);
    }
    const manifest = this.#actionAgents.get(agentId);
    if (!manifest) {
      throw new Error(`Humanoid action Agent has no manifest authority: ${agentId}`);
    }
    const authority = authorityForToolCall(
      this.#authorities,
      agentId,
      transactionId,
      input.expectedToolName
    );
    const toolCall = authority?.tool_calls.find((entry) => (
      entry.tool_call_id === transactionId
        && entry.tool_name === input.expectedToolName
    ));
    if (!authority || !authority.cycle || !toolCall
      || toolCall.arguments_sha256 !== argumentsSha256) {
      throw new Error(
        `Humanoid action tool call has no completed model response authority: ${transactionId}`
      );
    }
    return ModelDecisionRefSchema.parse({
      agent_id: agentId,
      agent_manifest_sha256: manifest.identitySha256,
      agent_manifest_epoch_id: manifest.epochId,
      model_call_id: authority.model_call_id,
      response_id: authority.response_id,
      response_output_sha256: authority.response_output_sha256,
      tool_call_id: toolCall.tool_call_id,
      tool_arguments_sha256: toolCall.arguments_sha256
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
  }): void {
    const decision = ModelDecisionRefSchema.parse(input.rawDecision);
    const manifest = this.#actionAgents.get(input.agentId);
    const authority = this.#authorities.get(decision.model_call_id);
    const toolCalls = authority?.tool_calls.filter((entry) => (
      entry.tool_call_id === input.transactionId
        && entry.tool_name === input.expectedToolName
        && entry.arguments_sha256 === modelPayloadSha256(input.actionInput)
    ));
    if (!manifest
      || decision.agent_id !== input.agentId
      || decision.agent_manifest_sha256 !== manifest.identitySha256
      || decision.agent_manifest_epoch_id !== manifest.epochId
      || decision.tool_call_id !== input.transactionId
      || !authority
      || authority.agent_id !== input.agentId
      || authority.response_id !== decision.response_id
      || authority.response_output_sha256 !== decision.response_output_sha256
      || !sameAutonomousCycle(authority.cycle, input.cycle)
      || toolCalls?.length !== 1
      || toolCalls[0]!.arguments_sha256 !== decision.tool_arguments_sha256) {
      throw new Error(`Humanoid action decision is not authoritative: ${input.transactionId}`);
    }
  }

  cycleForModelCall(modelCallId: string): AutonomousCycleRef | undefined {
    const cycle = this.#authorities.get(modelCallId)?.cycle;
    return cycle ? structuredClone(cycle) : undefined;
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

function assertManifestNode(
  nodes: Readonly<Record<string, TaskNode>>,
  agentId: string,
  role: string
): void {
  if (!nodes[agentId]) {
    throw new Error(`${role} manifest identity is absent from the hierarchy: ${agentId}`);
  }
}
