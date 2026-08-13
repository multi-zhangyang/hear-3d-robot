import { AsyncLocalStorage } from "node:async_hooks";
import { createHash, randomUUID } from "node:crypto";
import type { UUID } from "node:crypto";
import type { FunctionTool, ToolInputParameters } from "@openai/agents";

const NODE_MARKER = "HEAR_AGENT_INVOCATION_V1";
const NODE_ID_PATTERN = /^[a-zA-Z0-9_-]+$/;

interface AgentInvocationScope {
  agentId: string;
  invocationId: UUID;
  parentAgentId: string | null;
  parentInvocationId: UUID | null;
  rootInvocationId: UUID;
  parent: AgentInvocationScope | undefined;
  recovering: boolean;
  transportInterruption: Error | undefined;
  decisionInterruption: Error | undefined;
}

const invocationScope = new AsyncLocalStorage<AgentInvocationScope>();

export function agentInvocationMarker(agentId: string): string {
  if (!NODE_ID_PATTERN.test(agentId)) throw new Error("Invalid hierarchy node identifier");
  return `${NODE_MARKER}:${agentId}`;
}

export function currentAgentInvocationId(): string | undefined {
  return invocationScope.getStore()?.agentId;
}

/**
 * Identity of one concrete hierarchy episode.  Agent identity and invocation
 * identity are deliberately separate: the same structural Agent may run many
 * bounded episodes, while parallel siblings must retain distinct invocation
 * identities under the same parent invocation.
 */
export function currentAgentHarnessInvocation(): {
  agentId: string;
  invocationId: UUID;
  parentAgentId: string | null;
  parentInvocationId: UUID | null;
  rootInvocationId: UUID;
} | undefined {
  const scope = invocationScope.getStore();
  if (!scope) return undefined;
  return {
    agentId: scope.agentId,
    invocationId: scope.invocationId,
    parentAgentId: scope.parent?.agentId ?? null,
    parentInvocationId: scope.parentInvocationId,
    rootInvocationId: scope.rootInvocationId
  };
}

/**
 * Return the process-local SDK episode ancestry from Executive to the current
 * node. Durable lease deadlines are recovery horizons, not asynchronous
 * cancellation points: while this exact lexical Agent.asTool chain is on the
 * stack, a slow model response must not make its own tools disappear between
 * model selection and SDK execution.
 */
export function currentAgentHarnessInvocationChain(): Array<{
  agentId: string;
  invocationId: UUID;
}> {
  const chain: Array<{ agentId: string; invocationId: UUID }> = [];
  let scope = invocationScope.getStore();
  while (scope) {
    chain.push({ agentId: scope.agentId, invocationId: scope.invocationId });
    scope = scope.parent;
  }
  return chain.reverse();
}

export function currentAgentInvocationIsRecovery(): boolean {
  return invocationScope.getStore()?.recovering === true;
}

/**
 * Preserve a model transport failure before Agent.asTool's default function
 * error handler converts it to model-visible text. Every nested invocation has
 * a distinct scope. The failure is promoted to the parent only if this tool
 * invocation ultimately fails, so a successful SDK-managed retry clears it
 * without contaminating a parent or concurrent sibling.
 */
export function recordAgentInvocationTransportInterruption(error: Error): boolean {
  const scope = invocationScope.getStore();
  if (!scope) return false;
  scope.transportInterruption ??= error;
  return true;
}

export function clearAgentInvocationTransportInterruption(): void {
  const scope = invocationScope.getStore();
  if (scope) scope.transportInterruption = undefined;
}

export function recordAgentInvocationDecisionInterruption(error: Error): boolean {
  const scope = invocationScope.getStore();
  if (!scope) return false;
  scope.decisionInterruption ??= error;
  return true;
}

export function withAgentInvocation<T>(
  agentId: string,
  operation: () => Promise<T>,
  recovering = false,
  invocationId?: UUID
): Promise<T> {
  return runAgentInvocation(agentId, recovering, operation, invocationId);
}

/**
 * Keep an SDK Agent.asTool invocation attributed to its concrete hierarchy
 * node and undo the SDK's model-visible conversion of transport failures.
 * Mutating the freshly-created tool preserves the SDK's private source-agent
 * registration, which is required for RunState identity and nested resume.
 */
export function scopeAgentToolInvocation<
  TContext,
  TParameters extends ToolInputParameters,
  TResult,
  TTool extends FunctionTool<TContext, TParameters, TResult>
>(agentId: string, agentTool: TTool): TTool {
  assertAgentId(agentId);
  const invoke = agentTool.invoke;
  agentTool.invoke = async (context, input, details) => runAgentInvocation(
    agentId,
    typeof details?.resumeState === "string",
    async (scope) => {
      let output: Awaited<ReturnType<typeof invoke>>;
      try {
        output = await invoke(context, input, details);
      } catch (error) {
        if (scope.transportInterruption) {
          throw promoteTransportInterruption(scope);
        }
        if (scope.decisionInterruption) {
          throw promoteDecisionInterruption(scope);
        }
        throw error;
      }
      if (scope.transportInterruption) throw promoteTransportInterruption(scope);
      if (scope.decisionInterruption) throw promoteDecisionInterruption(scope);
      return output;
    },
    stableAgentToolInvocationId(agentId, details?.toolCall?.callId)
  );
  return agentTool;
}

function promoteTransportInterruption(scope: AgentInvocationScope): Error {
  const interruption = scope.transportInterruption;
  if (!interruption) throw new Error("Agent invocation has no transport interruption");
  if (scope.parent) scope.parent.transportInterruption ??= interruption;
  return interruption;
}

function promoteDecisionInterruption(scope: AgentInvocationScope): Error {
  const interruption = scope.decisionInterruption;
  if (!interruption) throw new Error("Agent invocation has no decision interruption");
  if (scope.parent) scope.parent.decisionInterruption ??= interruption;
  return interruption;
}

function runAgentInvocation<T>(
  agentId: string,
  recovering: boolean,
  operation: (scope: AgentInvocationScope) => Promise<T>,
  invocationId: UUID = randomUUID()
): Promise<T> {
  assertAgentId(agentId);
  const parent = invocationScope.getStore();
  const scope: AgentInvocationScope = {
    agentId,
    invocationId,
    parentAgentId: parent?.agentId ?? null,
    parentInvocationId: parent?.invocationId ?? null,
    rootInvocationId: parent?.rootInvocationId ?? invocationId,
    parent,
    recovering,
    transportInterruption: undefined,
    decisionInterruption: undefined
  };
  return invocationScope.run(scope, () => operation(scope));
}

/**
 * Derive one durable structural-child invocation from the SDK tool call that
 * opened it. Model children and deterministic/runtime children share this
 * namespace so RunState recovery re-enters the same hierarchy episode.
 */
export function stableAgentToolInvocationId(
  agentId: string,
  toolCallId?: string
): UUID {
  if (!toolCallId) return randomUUID();
  const bytes = createHash("sha256")
    .update("hear-agent-tool-invocation-v1\0")
    .update(agentId)
    .update("\0")
    .update(toolCallId)
    .digest()
    .subarray(0, 16);
  // Encode an RFC 4122 variant, version-5-shaped UUID.  The namespace is
  // HEAR-private and SHA-256 based; the version bits make strict UUID parsers
  // accept the stable identifier without pretending it is an SDK call id.
  bytes[6] = (bytes[6]! & 0x0f) | 0x50;
  bytes[8] = (bytes[8]! & 0x3f) | 0x80;
  const hex = bytes.toString("hex");
  return (`${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}`
    + `-${hex.slice(16, 20)}-${hex.slice(20)}`) as UUID;
}

export function agentIdFromInstructions(
  instructions: unknown,
  rootAgentId: string
): string {
  if (typeof instructions !== "string") return rootAgentId;
  const match = /^HEAR_AGENT_INVOCATION_V1:([a-zA-Z0-9_-]+)(?:\r?\n|$)/
    .exec(instructions);
  return match?.[1] ?? rootAgentId;
}

function assertAgentId(agentId: string): void {
  if (!NODE_ID_PATTERN.test(agentId)) throw new Error("Invalid hierarchy node identifier");
}
