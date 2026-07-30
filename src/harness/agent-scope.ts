import { AsyncLocalStorage } from "node:async_hooks";

const NODE_MARKER = "HEAR_AGENT_INVOCATION_V1";
const NODE_ID_PATTERN = /^[a-zA-Z0-9_-]+$/;
const invocationScope = new AsyncLocalStorage<string>();

export function agentInvocationMarker(agentId: string): string {
  if (!NODE_ID_PATTERN.test(agentId)) throw new Error("Invalid hierarchy node identifier");
  return `${NODE_MARKER}:${agentId}`;
}

export function currentAgentInvocationId(): string | undefined {
  return invocationScope.getStore();
}

export function withAgentInvocation<T>(
  agentId: string,
  operation: () => Promise<T>
): Promise<T> {
  if (!NODE_ID_PATTERN.test(agentId)) throw new Error("Invalid hierarchy node identifier");
  return invocationScope.run(agentId, operation);
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
    if (Array.isArray(current)) stack.push(...current);
    else stack.push(...Object.values(current));
  }
  return found;
}
