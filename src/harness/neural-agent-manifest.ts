import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import type { Agent, Session } from "@openai/agents";
import { z } from "zod";
import {
  DEFAULT_MODEL_REQUEST_TIMEOUT_MS,
  providerConfigForAgent,
  providerConfigForProfile,
  type AgentModelProfile,
  type ModelProviderConfig,
  type ProviderConfig
} from "../config/load.js";
import { NEURAL_HIERARCHY_CONTRACT_VERSION } from
  "../domain/neural-hierarchy.js";
import {
  NeuralAgentManifestSchema,
  type NeuralAgentIdentity,
  type NeuralAgentManifest
} from "../domain/agent-manifest.js";
import {
  HUMANOID_NEURAL_AGENT_IDS,
  HUMANOID_NEURAL_NODES,
  HUMANOID_NEURAL_SIGNAL_CONTRACTS,
  type HumanoidNeuralAgentKey,
  type HumanoidNeuralNodeDescriptor
} from "./humanoid/neural-hierarchy-contract.js";

const HUMANOID_NEURAL_HARNESS_CONTRACT_VERSION = 52;
const CORE_SDK_PACKAGES = [
  "@openai/agents",
  "@openai/agents-extensions",
  "ai"
] as const;
const PROTOCOL_ADAPTER_PACKAGE = {
  openai_compatible: "@ai-sdk/openai-compatible",
  openai_responses: "@ai-sdk/openai",
  anthropic_messages: "@ai-sdk/anthropic"
} as const satisfies Record<ModelProviderConfig["protocol"], string>;

/**
 * Stateful domain tools are exclusive neural capabilities. Child delegation
 * ownership is derived from the structural tree below; this table covers the
 * non-child tools whose accidental reuse would create a hidden flat control
 * plane around that tree.
 */
const EXCLUSIVE_NEURAL_TOOL_OWNER = {
  recall_goal_history: HUMANOID_NEURAL_AGENT_IDS.goalManager,
  submit_goal_candidates: HUMANOID_NEURAL_AGENT_IDS.goalManager,
  select_goal_candidate: HUMANOID_NEURAL_AGENT_IDS.goalManager,
  retire_goal_epoch: HUMANOID_NEURAL_AGENT_IDS.goalManager,
  continue_goal_epoch: HUMANOID_NEURAL_AGENT_IDS.goalManager,
  retrieve_relevant_embodied_memory: HUMANOID_NEURAL_AGENT_IDS.memoryRetriever,
  establish_skill_commitment: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
  authorize_skill_execution: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
  complete_skill_commitment: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
  fail_skill_commitment: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
  release_skill_commitment: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
  submit_humanoid_skill_plan: HUMANOID_NEURAL_AGENT_IDS.motorIntent,
  begin_humanoid_skill: HUMANOID_NEURAL_AGENT_IDS.motorIntent,
  plan_humanoid_skill: HUMANOID_NEURAL_AGENT_IDS.motorIntent,
  plan_whole_body_motion_candidates: HUMANOID_NEURAL_AGENT_IDS.motorIntent,
  plan_humanoid_navigation: HUMANOID_NEURAL_AGENT_IDS.motorIntent,
  complete_neural_autonomous_cycle: HUMANOID_NEURAL_AGENT_IDS.executive,
  complete_neural_satisfied_goal: HUMANOID_NEURAL_AGENT_IDS.executive
} as const satisfies Readonly<Record<string, string>>;

export interface NeuralRuntimeService {
  id: string;
  name: string;
  implementationContract: string;
}

/**
 * Runtime objects for the strict ownership tree. The same structural id may
 * never appear in both maps, and every model Agent must return its own Session.
 */
export interface NeuralAgentHierarchy {
  root: Agent<any, any>;
  agents: ReadonlyMap<string, Agent<any, any>>;
  services: ReadonlyMap<string, NeuralRuntimeService>;
  session(agentId: string): Session | undefined;
  outputSchema(agentId: string): z.ZodType | undefined;
}

export function humanoidNeuralAgentToolName(
  key: HumanoidNeuralAgentKey
): string {
  const names: Readonly<Record<HumanoidNeuralAgentKey, string>> = {
    executive: "run_executive",
    goalManager: "delegate_goal_valuation",
    actionSelection: "delegate_action_selection",
    perceptionManager: "delegate_perception_manager",
    sensorFusion: "capture_sensor_fusion",
    sceneInterpreter: "delegate_scene_interpretation",
    memoryRetriever: "delegate_relevant_memory",
    sensorimotorManager: "delegate_sensorimotor_manager",
    affordance: "delegate_affordance_assessment",
    risk: "delegate_risk_interoception",
    predictive: "delegate_predictive_critic",
    premotor: "delegate_premotor_composition",
    motorIntent: "delegate_motor_intent",
    rolloutGate: "run_mujoco_rollout_gate",
    executionDispatcher: "delegate_certified_execution_dispatcher",
    executor: "execute_certified_motor_intent",
    reflex: "track_motor_reference",
    body: "step_mujoco_body",
    recovery: "run_recovery_lease_episode"
  };
  return names[key];
}

export function createHumanoidNeuralAgentManifest(input: {
  hierarchy: NeuralAgentHierarchy;
  provider: ProviderConfig;
  epochId: string;
  createdAt?: string;
  runtimeSdkIdentity?: Record<string, string>;
}): NeuralAgentManifest {
  assertRuntimeHierarchyMatchesContract(input.hierarchy);
  const agents = Object.fromEntries(HUMANOID_NEURAL_NODES.map((descriptor) => {
    const runtimeAgent = input.hierarchy.agents.get(descriptor.id);
    const runtimeService = input.hierarchy.services.get(descriptor.id);
    const parentAgentId = descriptor.parentKey === null
      ? null
      : HUMANOID_NEURAL_AGENT_IDS[descriptor.parentKey];
    const common = {
      agent_id: descriptor.id,
      agent_name: descriptor.name,
      parent_agent_id: parentAgentId,
      layer: descriptor.layer,
      pathway: descriptor.pathway,
      orchestration_kind: descriptor.orchestrationKind,
      session_mode: descriptor.sessionMode,
      cadence: descriptor.cadence,
      maximum_correction_scope: descriptor.maximumCorrectionScope,
      ...(descriptor.parallelGroup
        ? { parallel_group: descriptor.parallelGroup }
        : {}),
      parallel_safe: descriptor.parallelSafe,
      physical_write_authority: descriptor.physicalWriteAuthority,
      capabilities: [...descriptor.capabilities]
    };
    let identity: NeuralAgentIdentity;
    if (descriptor.executionKind === "model_agent") {
      if (!runtimeAgent) {
        throw new Error(`Neural model Agent is absent: ${descriptor.id}`);
      }
      const profile = modelProfileForNode(descriptor.key);
      identity = {
        ...common,
        execution_kind: "model_agent",
        provider_profile: profile,
        ...modelIdentity(
          runtimeAgent,
          providerConfigForAgent(input.provider, descriptor.id, profile),
          requiredNeuralOutputSchema(input.hierarchy, descriptor.id)
        )
      };
    } else {
      if (!runtimeService) {
        throw new Error(`Neural runtime service is absent: ${descriptor.id}`);
      }
      identity = {
        ...common,
        execution_kind: descriptor.executionKind,
        implementation_contract: runtimeService.implementationContract
      };
    }
    return [descriptor.id, identity] as const;
  }));
  const controlEdges = HUMANOID_NEURAL_NODES.flatMap((descriptor) => {
    if (descriptor.parentKey === null) return [];
    const parentId = HUMANOID_NEURAL_AGENT_IDS[descriptor.parentKey];
    const base = {
      parent_agent_id: parentId,
      child_agent_id: descriptor.id,
      orchestration_kind: descriptor.orchestrationKind,
      contract_id: controlContractId(descriptor),
      ...(descriptor.parallelGroup
        ? { parallel_group: descriptor.parallelGroup }
        : {})
    };
    if (descriptor.orchestrationKind !== "agent_tool") return [base];
    return [{
      ...base,
      tool_name: humanoidNeuralAgentToolName(descriptor.key),
      session_agent_id: descriptor.id
    }];
  });
  const identity = {
    version: 3 as const,
    runtime: "humanoid_g1" as const,
    harness_contract_version: HUMANOID_NEURAL_HARNESS_CONTRACT_VERSION,
    neural_contract_version: NEURAL_HIERARCHY_CONTRACT_VERSION,
    runtime_sdk_identity: input.runtimeSdkIdentity
      ?? installedRuntimeSdkIdentity(input.provider),
    root_agent_id: HUMANOID_NEURAL_AGENT_IDS.executive,
    agents,
    control_edges: controlEdges,
    signal_contracts: HUMANOID_NEURAL_SIGNAL_CONTRACTS.map((contract) => ({
      source_agent_id: contract.sourceAgentId,
      target_agent_id: contract.targetAgentId,
      direction: contract.direction,
      signal_kinds: [...contract.signalKinds]
    }))
  };
  return NeuralAgentManifestSchema.parse({
    ...identity,
    epoch_id: input.epochId,
    created_at: input.createdAt ?? new Date().toISOString(),
    identity_sha256: sha256(canonicalJson(identity))
  });
}

function assertRuntimeHierarchyMatchesContract(
  hierarchy: NeuralAgentHierarchy
): void {
  const expectedModelIds: ReadonlySet<string> = new Set(HUMANOID_NEURAL_NODES
    .filter((node) => node.executionKind === "model_agent")
    .map((node) => node.id));
  const expectedServiceIds: ReadonlySet<string> = new Set(HUMANOID_NEURAL_NODES
    .filter((node) => node.executionKind !== "model_agent")
    .map((node) => node.id));
  if (hierarchy.root !== hierarchy.agents.get(HUMANOID_NEURAL_AGENT_IDS.executive)) {
    throw new Error("Neural hierarchy root must be the structural Executive Agent");
  }
  if (hierarchy.agents.size !== expectedModelIds.size
    || hierarchy.services.size !== expectedServiceIds.size) {
    throw new Error("Neural runtime node count does not match the ownership contract");
  }
  const sessions = new Set<Session>();
  const runtimeAgents = new Set<Agent<any, any>>();
  for (const agentId of expectedModelIds) {
    const agent = hierarchy.agents.get(agentId);
    const session = hierarchy.session(agentId);
    if (!agent || !session) {
      throw new Error(`Neural model Agent requires its own Session: ${agentId}`);
    }
    if (runtimeAgents.has(agent)) {
      throw new Error(`Neural hierarchy identities cannot alias one Agent: ${agentId}`);
    }
    runtimeAgents.add(agent);
    if (sessions.has(session)) {
      throw new Error(`Neural model Agents cannot share one Session: ${agentId}`);
    }
    sessions.add(session);
    if (agent.handoffs.length > 0) {
      throw new Error(
        `Neural control tree forbids SDK handoffs; use the structural parent tool: ${agentId}`
      );
    }
    if (agent.mcpServers.length > 0) {
      throw new Error(
        `Neural Agents cannot attach undeclared MCP tool surfaces: ${agentId}`
      );
    }
    if (agent.outputType !== "text") {
      throw new Error(
        `Neural Agent ${agentId} must reserve final authority for terminal tools`
      );
    }
    requiredNeuralOutputSchema(hierarchy, agentId);
  }
  for (const serviceId of expectedServiceIds) {
    if (!hierarchy.services.has(serviceId) || hierarchy.session(serviceId)) {
      throw new Error(`Neural runtime service cannot own a Session: ${serviceId}`);
    }
  }
  for (const id of hierarchy.agents.keys()) {
    if (!expectedModelIds.has(id)) throw new Error(`Unknown neural model Agent: ${id}`);
  }
  for (const id of hierarchy.services.keys()) {
    if (!expectedServiceIds.has(id)) throw new Error(`Unknown neural runtime service: ${id}`);
  }
  const exposedControlChildren = HUMANOID_NEURAL_NODES.filter((descriptor) => (
    descriptor.parentKey !== null
      && (descriptor.orchestrationKind === "agent_tool"
        || descriptor.key === "sensorFusion"
        || descriptor.key === "executor"
        || descriptor.key === "recovery")
  ));
  const controlToolOwnerByName = new Map(exposedControlChildren.map((descriptor) => [
    humanoidNeuralAgentToolName(descriptor.key),
    HUMANOID_NEURAL_AGENT_IDS[descriptor.parentKey!]
  ] as const));
  const exclusiveToolOwnerByName = new Map<string, string>([
    ...controlToolOwnerByName,
    ...Object.entries(EXCLUSIVE_NEURAL_TOOL_OWNER)
  ]);
  const exclusiveToolCounts = new Map<string, number>();
  for (const [agentId, agent] of hierarchy.agents) {
    for (const candidate of agent.tools) {
      if (candidate.type !== "function") continue;
      const expectedOwner = exclusiveToolOwnerByName.get(candidate.name);
      if (expectedOwner !== undefined && expectedOwner !== agentId) {
        throw new Error(
          `Exclusive neural tool crossed its ownership boundary: ${candidate.name} on ${agentId}`
        );
      }
      if (expectedOwner !== undefined) {
        exclusiveToolCounts.set(
          candidate.name,
          (exclusiveToolCounts.get(candidate.name) ?? 0) + 1
        );
      }
    }
  }
  for (const [toolName, expectedOwner] of Object.entries(
    EXCLUSIVE_NEURAL_TOOL_OWNER
  )) {
    if (exclusiveToolCounts.get(toolName) !== 1) {
      throw new Error(
        `Exclusive neural tool ${toolName} must exist exactly once on ${expectedOwner}`
      );
    }
  }
  for (const descriptor of exposedControlChildren) {
    if (descriptor.parentKey === null) {
      throw new Error(`Exposed neural child has no structural parent: ${descriptor.id}`);
    }
    const parentId = HUMANOID_NEURAL_AGENT_IDS[descriptor.parentKey];
    const parent = hierarchy.agents.get(parentId);
    const toolName = humanoidNeuralAgentToolName(descriptor.key);
    const matches = parent?.tools.filter((tool) => (
      tool.type === "function" && tool.name === toolName
    )) ?? [];
    if (matches.length !== 1) {
      throw new Error(
        `Parent ${parentId} must expose exactly one owned child edge ${toolName}`
      );
    }
  }
}

function modelIdentity(
  agent: Agent,
  provider: ModelProviderConfig,
  outputSchema: z.ZodType
): Pick<
  Extract<NeuralAgentIdentity, { execution_kind: "model_agent" }>,
  "protocol" | "model" | "endpoint_sha256" | "instructions_sha256"
    | "tool_schema_sha256" | "output_schema_sha256" | "sdk_model_settings"
    | "sdk_output_type" | "reset_tool_choice" | "settings"
> {
  if (typeof agent.instructions !== "string") {
    throw new Error(`Dynamic instructions cannot be persisted for ${agent.name}`);
  }
  const tools = agent.tools.map((tool) => {
    if (tool.type !== "function") {
      throw new Error(`Unsupported non-function tool on neural Agent ${agent.name}`);
    }
    return {
      type: tool.type,
      name: tool.name,
      description: tool.description,
      strict: tool.strict,
      parameters: tool.parameters,
      ...(tool.outputSchema === undefined ? {} : { output_schema: tool.outputSchema })
    };
  });
  const reasoningEffort = agent.modelSettings.reasoning?.effort ?? undefined;
  return {
    protocol: provider.protocol,
    model: provider.model,
    endpoint_sha256: sha256(new URL(provider.baseUrl).href),
    instructions_sha256: sha256(agent.instructions),
    tool_schema_sha256: sha256(canonicalJson(tools)),
    output_schema_sha256: sha256(canonicalJson(outputTypeIdentity(
      agent,
      outputSchema
    ))),
    sdk_output_type: agent.outputType === "text" ? "text" : "structured",
    sdk_model_settings: modelSettingsIdentity(agent.modelSettings),
    reset_tool_choice: agent.resetToolChoice,
    settings: {
      request_timeout_ms:
        provider.requestTimeoutMs ?? DEFAULT_MODEL_REQUEST_TIMEOUT_MS,
      temperature: provider.temperature,
      ...(reasoningEffort === undefined
        ? {}
        : { reasoning_effort: reasoningEffort }),
      ...(provider.maxOutputTokens === undefined
        ? {}
        : { max_output_tokens: provider.maxOutputTokens }),
      context_window_tokens: provider.contextWindowTokens,
      compact_trigger_tokens: provider.compactTriggerTokens,
      compact_recent_model_turns: provider.compactRecentModelTurns,
      ...(provider.compactMaxOutputTokens === undefined
        ? {}
        : { compact_max_output_tokens: provider.compactMaxOutputTokens })
    }
  };
}

function outputTypeIdentity(agent: Agent, outputSchema: z.ZodType): unknown {
  try {
    return z.toJSONSchema(outputSchema);
  } catch (cause) {
    throw new Error(
      `Neural Agent ${agent.name} requires a serializable terminal output schema`,
      { cause }
    );
  }
}

function requiredNeuralOutputSchema(
  hierarchy: NeuralAgentHierarchy,
  agentId: string
): z.ZodType {
  const schema = hierarchy.outputSchema(agentId);
  if (!schema) {
    throw new Error(`Neural Agent terminal output schema is absent: ${agentId}`);
  }
  return schema;
}

function modelSettingsIdentity(
  settings: Agent["modelSettings"]
): Extract<NeuralAgentIdentity, {
  execution_kind: "model_agent";
}>["sdk_model_settings"] {
  const identity: Record<string, unknown> = { ...settings };
  if (settings.providerData !== undefined) {
    identity.providerData = {
      redacted_sha256: sha256(canonicalJson(jsonIdentity(
        settings.providerData,
        "providerData"
      )))
    };
  }
  return z.record(z.string(), z.json()).parse(identity);
}

function modelProfileForNode(key: HumanoidNeuralAgentKey): Exclude<
  AgentModelProfile,
  "compactor"
> {
  const profiles: Readonly<Record<HumanoidNeuralAgentKey, AgentModelProfile | null>> = {
    executive: "executive",
    goalManager: "executive",
    actionSelection: "executive",
    perceptionManager: "associative",
    sensorFusion: null,
    sceneInterpreter: "associative",
    memoryRetriever: "associative",
    sensorimotorManager: "sensorimotor",
    affordance: "associative",
    risk: "associative",
    predictive: "sensorimotor",
    premotor: "sensorimotor",
    motorIntent: "motor_intent",
    rolloutGate: null,
    executionDispatcher: "sensorimotor",
    executor: null,
    reflex: null,
    body: null,
    recovery: "sensorimotor"
  };
  const profile = profiles[key];
  if (!profile || profile === "compactor") {
    throw new Error(`Neural runtime node has no model profile: ${key}`);
  }
  return profile;
}

function controlContractId(descriptor: HumanoidNeuralNodeDescriptor): string {
  if (descriptor.orchestrationKind === "agent_tool") {
    return "typed_neural_agent_tool_v1";
  }
  if (descriptor.orchestrationKind === "exclusive_lease_episode") {
    return "exclusive_recovery_lease_episode_v1";
  }
  if (descriptor.orchestrationKind === "controller_loop") {
    return "continuous_controller_reflex_loop_v1";
  }
  if (descriptor.orchestrationKind === "physical_plant") {
    return "authoritative_mujoco_plant_v1";
  }
  return "deterministic_neural_runtime_edge_v1";
}

function installedRuntimeSdkIdentity(provider: ProviderConfig): Record<string, string> {
  const packages = new Set<string>(CORE_SDK_PACKAGES);
  for (const profile of [
    "executive",
    "associative",
    "sensorimotor",
    "motor_intent",
    "compactor"
  ] as const) {
    const selected = providerConfigForProfile(provider, profile);
    packages.add(PROTOCOL_ADAPTER_PACKAGE[selected.protocol]);
  }
  for (const override of Object.values(provider.agentOverrides ?? {})) {
    if (override.protocol !== undefined) {
      packages.add(PROTOCOL_ADAPTER_PACKAGE[override.protocol]);
    }
  }
  return {
    node: process.versions.node,
    ...Object.fromEntries([...packages]
      .sort(compareUnicodeCodePoints)
      .map((name) => [name, installedPackageVersion(name)]))
  };
}

function installedPackageVersion(packageName: string): string {
  const require = createRequire(import.meta.url);
  let directory = dirname(require.resolve(packageName));
  for (let level = 0; level < 8; level += 1) {
    const packagePath = resolve(directory, "package.json");
    try {
      const metadata = JSON.parse(readFileSync(packagePath, "utf8")) as {
        name?: unknown;
        version?: unknown;
      };
      if (metadata.name === packageName && typeof metadata.version === "string") {
        return metadata.version;
      }
    } catch (error) {
      if (!(error instanceof Error && "code" in error && error.code === "ENOENT")) {
        throw error;
      }
    }
    const parent = dirname(directory);
    if (parent === directory) break;
    directory = parent;
  }
  throw new Error(`Cannot resolve runtime SDK identity for ${packageName}`);
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function canonicalJson(value: unknown): string {
  return JSON.stringify(canonicalValue(value));
}

function canonicalValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalValue);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(Object.entries(value as Record<string, unknown>)
      .filter(([, entry]) => entry !== undefined)
      .sort(([left], [right]) => compareUnicodeCodePoints(left, right))
      .map(([key, entry]) => [key, canonicalValue(entry)]));
  }
  return value;
}

function jsonIdentity(value: unknown, path: string): unknown {
  if (value === null || typeof value === "string" || typeof value === "boolean") {
    return value;
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error(`${path} contains a non-finite number`);
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((entry, index) => jsonIdentity(entry, `${path}[${index}]`));
  }
  if (typeof value === "object") {
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      throw new Error(`${path} contains a non-plain object`);
    }
    return Object.fromEntries(Object.entries(value as Record<string, unknown>)
      .filter(([, entry]) => entry !== undefined)
      .map(([key, entry]) => [key, jsonIdentity(entry, `${path}.${key}`)]));
  }
  throw new Error(`${path} contains a non-serializable ${typeof value}`);
}

function compareUnicodeCodePoints(left: string, right: string): number {
  const leftPoints = Array.from(left, (character) => character.codePointAt(0)!);
  const rightPoints = Array.from(right, (character) => character.codePointAt(0)!);
  const length = Math.min(leftPoints.length, rightPoints.length);
  for (let index = 0; index < length; index += 1) {
    const difference = leftPoints[index]! - rightPoints[index]!;
    if (difference !== 0) return difference < 0 ? -1 : 1;
  }
  if (leftPoints.length === rightPoints.length) return 0;
  return leftPoints.length < rightPoints.length ? -1 : 1;
}
