import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import type {
  Agent,
  ModelSettings,
  ToolUseBehavior
} from "@openai/agents";
import {
  AGENT_MODEL_ROLES,
  DEFAULT_MODEL_REQUEST_TIMEOUT_MS,
  providerConfigForRole,
  type AgentModelRole,
  type ModelProviderConfig,
  type ProviderConfig
} from "../config/load.js";
import {
  AgentManifestSchema,
  type AgentManifest,
  type AgentModelIdentity
} from "../domain/agent-manifest.js";
import { configuredOutputTokenLimit } from "../runtime/context-budget.js";
import { contextCompactorIdentitySource } from "./context-summary-agent.js";
import {
  HUMANOID_AGENT_IDS,
  HUMANOID_AGENT_TOOL_CONTRACTS,
  type HumanoidAgentHierarchy
} from "./humanoid/agents.js";

const COMPACTOR_AGENT_ID = "humanoid-context-compactor";
const COMPACTOR_AGENT_NAME = "Context Compactor";
const HUMANOID_HARNESS_CONTRACT_VERSION = 13;
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
const RECEIPT_TERMINAL_TOOLS = {
  sentry: ["observe_humanoid"],
  motion: [
    "plan_humanoid_skill",
    "plan_whole_body_motion_candidates"
  ],
  executor: [
    "execute_humanoid_skill",
    "execute_whole_body_motion",
    "remove_world_block"
  ]
} as const;
const VERIFIED_STATUS_TERMINAL_TOOLS = {
  goal_manager: {
    select_goal_candidate: "goal_candidate_selected",
    retire_goal_epoch: "goal_epoch_retired"
  },
  coordinator: {
    complete_autonomous_cycle: "cycle_completed",
    complete_goal_transition: "goal_transition_completed",
    complete_satisfied_goal: "satisfied_goal_completed"
  }
} as const;

type ToolUseBehaviorIdentity = AgentModelIdentity["tool_use_behavior"];
type AgentToolContract = AgentManifest["agent_tool_contracts"][number];

interface AgentIdentitySource {
  agentName: string;
  instructions: string;
  modelSettings: AgentModelIdentity["sdk_model_settings"];
  resetToolChoice: boolean;
  toolUseBehavior: ToolUseBehaviorIdentity;
  tools: Array<{
    type: string;
    name?: string;
    description?: string;
    strict?: boolean;
    parameters?: unknown;
    outputSchema?: unknown;
  }>;
}

export function createHumanoidAgentManifest(input: {
  hierarchy: HumanoidAgentHierarchy;
  provider: ProviderConfig;
  epochId: string;
  createdAt?: string;
  runtimeSdkIdentity?: Record<string, string>;
}): AgentManifest {
  const sources: Record<Exclude<AgentModelRole, "compactor">, {
    agentId: string;
    source: AgentIdentitySource;
  }> = {
    goal_manager: {
      agentId: HUMANOID_AGENT_IDS.goalManager,
      source: sourceFromAgent(input.hierarchy.goalManager, "goal_manager")
    },
    coordinator: {
      agentId: HUMANOID_AGENT_IDS.coordinator,
      source: sourceFromAgent(input.hierarchy.coordinator, "coordinator")
    },
    sentry: {
      agentId: HUMANOID_AGENT_IDS.sentry,
      source: sourceFromAgent(input.hierarchy.sentry, "sentry")
    },
    motion: {
      agentId: HUMANOID_AGENT_IDS.motion,
      source: sourceFromAgent(input.hierarchy.motion, "motion")
    },
    executor: {
      agentId: HUMANOID_AGENT_IDS.executor,
      source: sourceFromAgent(input.hierarchy.executor, "executor")
    }
  };
  const compactorProvider = providerConfigForRole(input.provider, "compactor");
  const compactorIdentity = contextCompactorIdentitySource();
  const compactorTool = compactorIdentity.tools[0];
  if (compactorIdentity.tools.length !== 1 || !compactorTool) {
    throw new Error("Context Compactor manifest requires exactly one terminal tool");
  }
  const compactorOutputLimit = configuredOutputTokenLimit(
    compactorProvider.compactMaxOutputTokens,
    compactorProvider.maxOutputTokens
  );
  const compactorSource: AgentIdentitySource = {
    agentName: COMPACTOR_AGENT_NAME,
    instructions: compactorIdentity.instructions,
    modelSettings: modelSettingsIdentity({
      temperature: compactorProvider.temperature,
      ...(compactorProvider.reasoningEffort === undefined
        ? {}
        : { reasoning: { effort: compactorProvider.reasoningEffort } }),
      ...(compactorOutputLimit === undefined
        ? {}
        : { maxTokens: compactorOutputLimit }),
      parallelToolCalls: false,
      toolChoice: compactorTool.name
    }),
    resetToolChoice: false,
    toolUseBehavior: {
      kind: "harness_callback",
      contract_id: "validated_context_checkpoint_v1",
      terminal_tool_names: compactorIdentity.tools.map((tool) => tool.name)
    },
    tools: compactorIdentity.tools
  };
  const agents = {
    goal_manager: modelIdentity(
      "goal_manager",
      sources.goal_manager.agentId,
      sources.goal_manager.source,
      providerConfigForRole(input.provider, "goal_manager")
    ),
    coordinator: modelIdentity(
      "coordinator",
      sources.coordinator.agentId,
      sources.coordinator.source,
      providerConfigForRole(input.provider, "coordinator")
    ),
    sentry: modelIdentity(
      "sentry",
      sources.sentry.agentId,
      sources.sentry.source,
      providerConfigForRole(input.provider, "sentry")
    ),
    motion: modelIdentity(
      "motion",
      sources.motion.agentId,
      sources.motion.source,
      providerConfigForRole(input.provider, "motion")
    ),
    executor: modelIdentity(
      "executor",
      sources.executor.agentId,
      sources.executor.source,
      providerConfigForRole(input.provider, "executor")
    ),
    compactor: modelIdentity(
      "compactor",
      COMPACTOR_AGENT_ID,
      compactorSource,
      compactorProvider
    )
  };
  const agentToolContracts = createAgentToolContracts(input.hierarchy);
  const identity = {
    version: 1 as const,
    runtime: "humanoid_g1" as const,
    harness_contract_version: HUMANOID_HARNESS_CONTRACT_VERSION,
    runtime_sdk_identity: input.runtimeSdkIdentity
      ?? installedRuntimeSdkIdentity(input.provider),
    agents,
    agent_tool_contracts: agentToolContracts
  };
  return AgentManifestSchema.parse({
    ...identity,
    epoch_id: input.epochId,
    created_at: input.createdAt ?? new Date().toISOString(),
    identity_sha256: sha256(canonicalJson(identity))
  });
}

export function assertAgentManifestCompatible(
  persistedInput: AgentManifest,
  currentInput: AgentManifest
): void {
  const persisted = assertManifestIntegrity(persistedInput, "persisted");
  const current = assertManifestIntegrity(currentInput, "current");
  if (persisted.identity_sha256 === current.identity_sha256) return;

  const changes: string[] = [];
  if (persisted.harness_contract_version !== current.harness_contract_version) {
    changes.push("harness_contract_version");
  }
  if (canonicalJson(persisted.runtime_sdk_identity)
    !== canonicalJson(current.runtime_sdk_identity)) {
    changes.push("runtime_sdk_identity");
  }
  if (canonicalJson(stableAgentToolContracts(persisted.agent_tool_contracts))
    !== canonicalJson(stableAgentToolContracts(current.agent_tool_contracts))) {
    changes.push("agent_tool_contracts");
  }
  for (const role of AGENT_MODEL_ROLES) {
    const before = persisted.agents[role];
    const after = current.agents[role];
    for (const field of [
      "agent_id",
      "agent_name",
      "protocol",
      "model",
      "endpoint_sha256",
      "instructions_sha256",
      "tool_schema_sha256",
      "sdk_model_settings",
      "reset_tool_choice",
      "tool_use_behavior",
      "settings"
    ] as const) {
      if (canonicalJson(before[field]) !== canonicalJson(after[field])) {
        changes.push(`agents.${role}.${field}`);
      }
    }
  }
  if (changes.length === 0) return;
  throw new AgentManifestIncompatibleError(changes);
}

export class AgentManifestIncompatibleError extends Error {
  readonly code = "agent_manifest_incompatible";
  readonly changedFields: string[];

  constructor(changedFields: string[]) {
    const detail = changedFields.length > 0
      ? changedFields.join(", ")
      : "identity_sha256";
    super(
      `Agent manifest is incompatible with the persisted session epoch (${detail}); `
      + "refusing to reuse model or Session state"
    );
    this.name = "AgentManifestIncompatibleError";
    this.changedFields = [...changedFields];
  }
}

function modelIdentity(
  role: AgentModelRole,
  agentId: string,
  source: AgentIdentitySource,
  provider: ModelProviderConfig
): AgentModelIdentity {
  const toolSchema = source.tools.map((entry) => {
    return functionToolSchemaIdentity(entry, agentId);
  });
  return {
    agent_id: agentId,
    agent_name: source.agentName,
    role,
    protocol: provider.protocol,
    model: provider.model,
    endpoint_sha256: sha256(new URL(provider.baseUrl).href),
    instructions_sha256: sha256(source.instructions),
    tool_schema_sha256: sha256(canonicalJson(toolSchema)),
    sdk_model_settings: source.modelSettings,
    reset_tool_choice: source.resetToolChoice,
    tool_use_behavior: source.toolUseBehavior,
    settings: {
      request_timeout_ms:
        provider.requestTimeoutMs ?? DEFAULT_MODEL_REQUEST_TIMEOUT_MS,
      temperature: provider.temperature,
      ...(provider.reasoningEffort === undefined
        ? {}
        : { reasoning_effort: provider.reasoningEffort }),
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

function sourceFromAgent(
  agent: Agent,
  role: Exclude<AgentModelRole, "compactor">
): AgentIdentitySource {
  if (typeof agent.instructions !== "string") {
    throw new Error(`Dynamic instructions cannot be persisted for agent ${agent.name}`);
  }
  return {
    agentName: agent.name,
    instructions: agent.instructions,
    modelSettings: modelSettingsIdentity(agent.modelSettings),
    resetToolChoice: agent.resetToolChoice,
    toolUseBehavior: toolUseBehaviorIdentity(agent, role),
    tools: agent.tools.map((entry) => ({
      type: entry.type,
      ...(entry.type === "function"
        ? {
            name: entry.name,
            description: entry.description,
            strict: entry.strict,
            parameters: entry.parameters,
            ...(entry.outputSchema === undefined ? {} : { outputSchema: entry.outputSchema })
          }
        : {})
    }))
  };
}

function toolUseBehaviorIdentity(
  agent: Agent,
  role: Exclude<AgentModelRole, "compactor">
): ToolUseBehaviorIdentity {
  const behavior = agent.toolUseBehavior;
  if (typeof behavior === "string") {
    return { kind: "sdk_flag", value: behavior };
  }
  if (typeof behavior === "object") {
    return {
      kind: "stop_at_tool_names",
      tool_names: [...behavior.stopAtToolNames]
    };
  }
  if (role === "coordinator" || role === "goal_manager") {
    const terminalToolNames = probeVerifiedStatusTerminalTools(
      agent,
      behavior,
      VERIFIED_STATUS_TERMINAL_TOOLS[role]
    );
    return {
      kind: "harness_callback",
      contract_id: "verified_harness_terminal_status_v1",
      terminal_tool_names: terminalToolNames
    };
  }
  const terminalToolNames = probeReceiptTerminalTools(agent, behavior);
  if (canonicalJson(terminalToolNames)
    !== canonicalJson(RECEIPT_TERMINAL_TOOLS[role])) {
    throw new Error(
      `Agent ${agent.name} receipt callback no longer matches its declared terminal tools`
    );
  }
  return {
    kind: "harness_callback",
    contract_id: "accepted_humanoid_action_receipt_v1",
    terminal_tool_names: terminalToolNames
  };
}

function probeVerifiedStatusTerminalTools(
  agent: Agent,
  behavior: Extract<ToolUseBehavior, (...args: never[]) => unknown>,
  statusByTool: Readonly<Record<string, string>>
): string[] {
  const terminalToolNames: string[] = [];
  for (const [toolName, expectedStatus] of Object.entries(statusByTool)) {
    const tool = agent.tools.find((entry) => (
      entry.type === "function" && entry.name === toolName
    ));
    if (!tool || tool.type !== "function") {
      throw new Error(`Agent ${agent.name} is missing terminal tool ${toolName}`);
    }
    const accepted = behavior({} as never, [{
      type: "function_output",
      tool,
      output: JSON.stringify({ status: expectedStatus })
    } as never]);
    const rejected = behavior({} as never, [{
      type: "function_output",
      tool,
      output: JSON.stringify({
        accepted: false,
        code: "contract_probe_rejection"
      })
    } as never]);
    if (accepted instanceof Promise || rejected instanceof Promise) {
      throw new Error(
        `Agent ${agent.name} uses an asynchronous toolUseBehavior that cannot be `
        + "verified synchronously for recovery"
      );
    }
    if (!accepted.isFinalOutput || rejected.isFinalOutput) {
      throw new Error(
        `Agent ${agent.name} terminal callback does not enforce status ${expectedStatus}`
      );
    }
    terminalToolNames.push(toolName);
  }
  return terminalToolNames;
}

function probeReceiptTerminalTools(
  agent: Agent,
  behavior: Extract<ToolUseBehavior, (...args: never[]) => unknown>
): string[] {
  const terminalToolNames: string[] = [];
  for (const tool of agent.tools) {
    if (tool.type !== "function") continue;
    const outcome = behavior({} as never, [{
      type: "function_output",
      tool,
      output: JSON.stringify({
        transactionId: "agent-manifest-contract-probe",
        action: tool.name,
        accepted: true
      })
    } as never]);
    if (outcome instanceof Promise) {
      throw new Error(
        `Agent ${agent.name} uses an asynchronous toolUseBehavior that cannot be `
        + "verified synchronously for recovery"
      );
    }
    if (outcome.isFinalOutput) terminalToolNames.push(tool.name);
  }
  return terminalToolNames;
}

function modelSettingsIdentity(
  settings: ModelSettings
): AgentModelIdentity["sdk_model_settings"] {
  const identity: Record<string, unknown> = { ...settings };
  if (settings.providerData !== undefined) {
    identity.providerData = {
      redacted_sha256: sha256(canonicalJson(jsonIdentity(settings.providerData, "providerData")))
    };
  }
  const value = jsonIdentity(identity, "modelSettings");
  if (value === null || Array.isArray(value) || typeof value !== "object") {
    throw new Error("Agent modelSettings must be a JSON object");
  }
  return value;
}

function createAgentToolContracts(
  hierarchy: HumanoidAgentHierarchy
): AgentToolContract[] {
  const definitions = Object.values(HUMANOID_AGENT_TOOL_CONTRACTS);
  return definitions.map((definition) => {
    const targetAgent = definition.targetRole === "goal_manager"
      ? hierarchy.goalManager
      : hierarchy[definition.targetRole];
    const matches = hierarchy.coordinator.tools.filter(
      (tool) => tool.type === "function" && tool.name === definition.toolName
    );
    if (matches.length !== 1 || matches[0]?.type !== "function") {
      throw new Error(
        `Expected exactly one Agent-as-tool function named ${definition.toolName}`
      );
    }
    if (!hierarchy.session(definition.runOptions.sessionAgentId)) {
      throw new Error(
        `Agent-as-tool ${definition.toolName} requires an owned Session for `
        + definition.runOptions.sessionAgentId
      );
    }
    const toolSchema = functionToolSchemaIdentity(matches[0], definition.toolName);
    return {
      tool_name: definition.toolName,
      target_role: definition.targetRole,
      target_agent_id: definition.targetAgentId,
      target_agent_name: targetAgent.name,
      tool_schema_sha256: sha256(canonicalJson(toolSchema)),
      input_builder_contract: definition.inputBuilderContract,
      input_builder_sha256: sha256(canonicalJson({
        tool_name: definition.toolName,
        contract: definition.inputBuilderContract
      })),
      run_options: {
        session_agent_id: definition.runOptions.sessionAgentId,
        context_source: definition.runOptions.contextSource,
        max_turns: definition.runOptions.maxTurns
      },
      resume_context_strategy: definition.resumeContextStrategy,
      include_input_schema: definition.includeInputSchema,
      needs_approval: definition.needsApproval,
      output_contract: definition.outputContract
    };
  });
}

function stableAgentToolContracts(
  contracts: readonly AgentToolContract[]
): Array<Omit<AgentToolContract, "input_builder_sha256">> {
  return contracts.map(({ input_builder_sha256: _runtimeDigest, ...contract }) => contract);
}

function functionToolSchemaIdentity(
  entry: AgentIdentitySource["tools"][number],
  owner: string
): {
  type: "function";
  name: string;
  description: string;
  strict: boolean;
  parameters: unknown;
  output_schema?: unknown;
} {
  if (entry.type !== "function" || !entry.name || !entry.description
    || entry.parameters === undefined || entry.strict === undefined) {
    throw new Error(`Unsupported tool identity in agent manifest: ${owner}`);
  }
  return {
    type: "function",
    name: entry.name,
    description: entry.description,
    strict: entry.strict,
    parameters: entry.parameters,
    ...(entry.outputSchema === undefined ? {} : { output_schema: entry.outputSchema })
  };
}

function assertManifestIntegrity(
  input: AgentManifest,
  label: "persisted" | "current"
): AgentManifest {
  const manifest = AgentManifestSchema.parse(input);
  const identity = {
    version: manifest.version,
    runtime: manifest.runtime,
    harness_contract_version: manifest.harness_contract_version,
    runtime_sdk_identity: manifest.runtime_sdk_identity,
    agents: manifest.agents,
    agent_tool_contracts: manifest.agent_tool_contracts
  };
  const expected = sha256(canonicalJson(identity));
  if (manifest.identity_sha256 !== expected) {
    throw new Error(`The ${label} agent manifest identity hash is invalid`);
  }
  return manifest;
}

function installedRuntimeSdkIdentity(provider: ProviderConfig): Record<string, string> {
  const packages = new Set<string>(CORE_SDK_PACKAGES);
  for (const role of AGENT_MODEL_ROLES) {
    const selected = providerConfigForRole(provider, role);
    packages.add(PROTOCOL_ADAPTER_PACKAGE[selected.protocol]);
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
      if (!(error instanceof Error && "code" in error && error.code === "ENOENT")) throw error;
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

type JsonIdentity = null | boolean | number | string | JsonIdentity[] | {
  [key: string]: JsonIdentity;
};

function jsonIdentity(value: unknown, path: string): JsonIdentity {
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
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, entry]) => entry !== undefined)
      .map(([key, entry]) => [key, jsonIdentity(entry, `${path}.${key}`)] as const);
    return Object.fromEntries(entries);
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
