import { AsyncLocalStorage } from "node:async_hooks";
import type { FunctionTool, ToolInputParameters } from "@openai/agents";

const NODE_MARKER = "HEAR_AGENT_INVOCATION_V1";
const NODE_ID_PATTERN = /^[a-zA-Z0-9_-]+$/;

interface AgentInvocationScope {
  agentId: string;
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
  recovering = false
): Promise<T> {
  return runAgentInvocation(agentId, recovering, operation);
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
    }
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
  operation: (scope: AgentInvocationScope) => Promise<T>
): Promise<T> {
  assertAgentId(agentId);
  const scope: AgentInvocationScope = {
    agentId,
    parent: invocationScope.getStore(),
    recovering,
    transportInterruption: undefined,
    decisionInterruption: undefined
  };
  return invocationScope.run(scope, () => operation(scope));
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
