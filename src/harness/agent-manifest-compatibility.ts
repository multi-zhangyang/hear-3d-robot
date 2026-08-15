import { createHash } from "node:crypto";
import { AGENT_MODEL_ROLES } from "../config/load.js";
import {
  AgentManifestSchema,
  type AgentManifest,
  type LegacyAgentManifest,
  type NeuralAgentManifest
} from "../domain/agent-manifest.js";

type LegacyAgentToolContract = LegacyAgentManifest["agent_tool_contracts"][number];

/**
 * Verify that persisted Agent/Session state belongs to the exact executable
 * hierarchy being started. This module deliberately has no dependency on the
 * retired Coordinator hierarchy, so the neural runtime never loads that code.
 */
export function assertAgentManifestCompatible(
  persistedInput: AgentManifest,
  currentInput: AgentManifest
): void {
  const persisted = assertManifestIntegrity(persistedInput, "persisted");
  const current = assertManifestIntegrity(currentInput, "current");
  if (persisted.identity_sha256 === current.identity_sha256) return;

  if (persisted.version !== current.version) {
    throw new AgentManifestIncompatibleError(["version"]);
  }
  if (persisted.version === 3 && current.version === 3) {
    assertNeuralAgentManifestCompatible(persisted, current);
    return;
  }
  if (persisted.version !== 1 || current.version !== 1) {
    throw new AgentManifestIncompatibleError(["version"]);
  }

  const changes: string[] = [];
  if (persisted.harness_contract_version !== current.harness_contract_version) {
    changes.push("harness_contract_version");
  }
  if (canonicalJson(persisted.runtime_sdk_identity)
    !== canonicalJson(current.runtime_sdk_identity)) {
    changes.push("runtime_sdk_identity");
  }
  if (canonicalJson(stableLegacyAgentToolContracts(persisted.agent_tool_contracts))
    !== canonicalJson(stableLegacyAgentToolContracts(current.agent_tool_contracts))) {
    changes.push("agent_tool_contracts");
  }
  for (const role of AGENT_MODEL_ROLES) {
    const before = persisted.agents[role];
    const after = current.agents[role];
    for (const field of [
      "execution_kind",
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
      "settings",
      "implementation_contract",
      "decision_authority_role"
    ] as const) {
      const beforeValue = (before as unknown as Record<string, unknown>)[field];
      const afterValue = (after as unknown as Record<string, unknown>)[field];
      if (canonicalJson(beforeValue) !== canonicalJson(afterValue)) {
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

function assertNeuralAgentManifestCompatible(
  persisted: NeuralAgentManifest,
  current: NeuralAgentManifest
): void {
  const changes: string[] = [];
  for (const field of [
    "harness_contract_version",
    "neural_contract_version",
    "runtime_sdk_identity",
    "root_agent_id",
    "agents",
    "control_edges",
    "signal_contracts"
  ] as const) {
    if (canonicalJson(persisted[field]) !== canonicalJson(current[field])) {
      changes.push(field);
    }
  }
  if (changes.length > 0) throw new AgentManifestIncompatibleError(changes);
}

function assertManifestIntegrity(
  input: AgentManifest,
  label: "persisted" | "current"
): AgentManifest {
  const manifest = AgentManifestSchema.parse(input);
  const identity = manifest.version === 1
    ? {
        version: manifest.version,
        runtime: manifest.runtime,
        harness_contract_version: manifest.harness_contract_version,
        runtime_sdk_identity: manifest.runtime_sdk_identity,
        agents: manifest.agents,
        agent_tool_contracts: manifest.agent_tool_contracts
      }
    : {
        version: manifest.version,
        runtime: manifest.runtime,
        harness_contract_version: manifest.harness_contract_version,
        neural_contract_version: manifest.neural_contract_version,
        runtime_sdk_identity: manifest.runtime_sdk_identity,
        root_agent_id: manifest.root_agent_id,
        agents: manifest.agents,
        control_edges: manifest.control_edges,
        signal_contracts: manifest.signal_contracts
      };
  const expected = sha256(canonicalJson(identity));
  if (manifest.identity_sha256 !== expected) {
    throw new Error(`The ${label} agent manifest identity hash is invalid`);
  }
  return manifest;
}

function stableLegacyAgentToolContracts(
  contracts: readonly LegacyAgentToolContract[]
): Array<Omit<LegacyAgentToolContract, "input_builder_sha256">> {
  return contracts.map(({ input_builder_sha256: _runtimeDigest, ...contract }) => contract);
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
