import { AsyncLocalStorage } from "node:async_hooks";

const NODE_MARKER = "HEAR_AGENT_INVOCATION_V1";
const NODE_ID_PATTERN = /^[a-zA-Z0-9_-]+$/;

interface AgentInvocationScope {
  agentId: string;
  parent: AgentInvocationScope | undefined;
  recovering: boolean;
  transportInterruption: Error | undefined;
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
 * a distinct scope; copying the first failure into its direct ancestor chain
 * lets a supervisor's asTool boundary recover the same original Error without
 * exposing it to an unrelated parallel sibling.
 */
export function recordAgentInvocationTransportInterruption(error: Error): boolean {
  let scope = invocationScope.getStore();
  if (!scope) return false;
  while (scope) {
    scope.transportInterruption ??= error;
    scope = scope.parent;
  }
  return true;
}

export function withAgentInvocation<T>(
  agentId: string,
  operation: () => Promise<T>,
  recovering = false
): Promise<T> {
  if (!NODE_ID_PATTERN.test(agentId)) throw new Error("Invalid hierarchy node identifier");
  const scope: AgentInvocationScope = {
    agentId,
    parent: invocationScope.getStore(),
    recovering,
    transportInterruption: undefined
  };
  return invocationScope.run(scope, operation);
}

/**
 * Resolve the concrete hierarchy node behind an SDK model request.
 *
 * Nested Agent.asTool runs retain AsyncLocalStorage across their whole model/tool
 * lifecycle. The marker is also placed in their durable input and filtered
 * system instructions so serialized state and request-level tests can recover
 * identity without relying on mutable UI focus.
 */
export function agentIdFromModelPayload(value: unknown, rootAgentId: string): string {
  const scoped = currentAgentInvocationId();
  if (scoped) return scoped;
  const marked = findMarker(value);
  return marked ?? rootAgentId;
}

function findMarker(value: unknown): string | undefined {
  const visited = new Set<object>();
  const stack: unknown[] = [value];
  let found: string | undefined;
  while (stack.length > 0) {
    const current = stack.pop();
    if (typeof current === "string") {
      const matches = current.matchAll(/HEAR_AGENT_INVOCATION_V1:([a-zA-Z0-9_-]+)/g);
      for (const match of matches) found = match[1];
      continue;
    }
    if (!current || typeof current !== "object" || visited.has(current)) continue;
    visited.add(current);
    if (Array.isArray(current)) {
      for (const item of current) stack.push(item);
    } else {
      for (const value of Object.values(current)) stack.push(value);
    }
  }
  return found;
}
