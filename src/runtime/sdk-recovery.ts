import { createHash } from "node:crypto";
import type { RunCheckpoint } from "../domain/schema.js";

export function sdkRecoveryCheckpointFingerprint(checkpoint: RunCheckpoint): string {
  const nodes = Object.fromEntries(Object.entries(checkpoint.nodes).map(([id, node]) => {
    const { updated_at: _updatedAt, model_calls_used: _modelCalls, ...semanticNode } = node;
    return [id, semanticNode];
  }));
  const recoverySurface = {
    version: 1,
    run_id: checkpoint.run_id,
    root_id: checkpoint.root_id,
    active_agent_id: checkpoint.active_agent_id,
    active_agent_ids: checkpoint.active_agent_ids.toSorted(),
    nodes,
    world: checkpoint.world,
    inflight_action: checkpoint.inflight_action,
    inflight_actions: checkpoint.inflight_actions,
    committed_actions: checkpoint.committed_actions,
    spatial_memory: checkpoint.spatial_memory,
    context_memory: checkpoint.context_memory,
    checker: checkpoint.checker
  };
  return createHash("sha256").update(canonicalJson(recoverySurface)).digest("hex");
}

/**
 * A hierarchy child can only still be executing when the root SDK turn still
 * owns the function call that created it. Completed tool outputs cancel the
 * matching call even though both objects remain in serialized RunState history.
 */
export function assertRunStateMatchesOpenRootDelegations(
  checkpoint: RunCheckpoint,
  serializedRunState: unknown
): void {
  const root = checkpoint.nodes[checkpoint.root_id];
  if (!root) throw new Error("Checkpoint hierarchy root is missing");

  const pendingCallIds = pendingFunctionCallIds(serializedRunState);
  const missing = root.child_ids.flatMap((childId) => {
    const child = checkpoint.nodes[childId];
    if (!child) return [`${childId}:missing_node`];
    if (!isOpen(child.status)) return [];
    if (!child.source_call_id) return [`${child.id}:missing_call_id`];
    return pendingCallIds.has(child.source_call_id)
      ? []
      : [`${child.id}:${child.source_call_id}`];
  });

  if (missing.length > 0) {
    throw new Error(
      "SDK RunState does not own every unfinished root delegation: "
      + missing.join(", ")
    );
  }
}

export function hasOpenRootDelegations(checkpoint: RunCheckpoint): boolean {
  const root = checkpoint.nodes[checkpoint.root_id];
  if (!root) throw new Error("Checkpoint hierarchy root is missing");
  return root.child_ids.some((childId) => {
    const child = checkpoint.nodes[childId];
    return child !== undefined && isOpen(child.status);
  });
}

export function pendingFunctionCallIds(serializedRunState: unknown): Set<string> {
  const calls = new Set<string>();
  const results = new Set<string>();
  const visited = new Set<object>();
  const pending: unknown[] = [serializedRunState];

  while (pending.length > 0) {
    const value = pending.pop();
    if (!value || typeof value !== "object" || visited.has(value)) continue;
    visited.add(value);
    if (Array.isArray(value)) {
      pending.push(...value);
      continue;
    }

    const record = value as Record<string, unknown>;
    const callId = functionCallId(record);
    if (record.type === "function_call" && callId) calls.add(callId);
    if (record.type === "function_call_result" && callId) results.add(callId);
    pending.push(...Object.values(record));
  }

  return new Set([...calls].filter((callId) => !results.has(callId)));
}

function functionCallId(record: Record<string, unknown>): string | undefined {
  if (typeof record.callId === "string") return record.callId;
  return typeof record.call_id === "string" ? record.call_id : undefined;
}

function isOpen(status: string): boolean {
  return status === "ready" || status === "active" || status === "waiting";
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record).toSorted().flatMap((key) => {
    const item = record[key];
    return item === undefined
      ? []
      : [`${JSON.stringify(key)}:${canonicalJson(item)}`];
  }).join(",")}}`;
}
