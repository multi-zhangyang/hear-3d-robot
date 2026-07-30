import type {
  ActionReceipt,
  CheckerResult,
  ContextMemoryState,
  ProviderActivity,
  RunListItem,
  RuntimeEvent,
  TaskNode,
  WorldSnapshot
} from "./types";

export function nextRuntimeEventCursor(
  current: string | undefined,
  event: RuntimeEvent
): string | undefined {
  const durable = event.durable ?? event.type !== "world_frames";
  return durable ? event.event_id : current;
}

export function mergeWorldFrames(
  current: WorldSnapshot[],
  incoming: WorldSnapshot[],
  limit: number
): WorldSnapshot[] {
  if (incoming.length === 0) return current;
  let next = current;
  let writable = false;
  const ensureWritable = (): void => {
    if (writable) return;
    next = [...next];
    writable = true;
  };

  for (const frame of incoming) {
    const last = next.at(-1);
    if (!last || frame.frame > last.frame) {
      ensureWritable();
      next.push(frame);
      continue;
    }
    if (frame.frame === last.frame) {
      ensureWritable();
      next[next.length - 1] = frame;
      continue;
    }

    let lower = 0;
    let upper = next.length;
    while (lower < upper) {
      const middle = (lower + upper) >>> 1;
      if (next[middle]!.frame < frame.frame) lower = middle + 1;
      else upper = middle;
    }
    ensureWritable();
    if (next[lower]?.frame === frame.frame) next[lower] = frame;
    else next.splice(lower, 0, frame);
  }

  return next.length > limit ? next.slice(-limit) : next;
}

export function upsertAction(actions: ActionReceipt[], receipt: ActionReceipt, limit: number): ActionReceipt[] {
  const index = actions.findIndex((action) => action.transaction_id === receipt.transaction_id);
  if (index < 0) return appendRecent(actions, receipt, limit);
  const next = [...actions];
  next[index] = receipt;
  return next;
}

export function appendRecent<T>(current: T[], entry: T, limit: number): T[] {
  if (current.length < limit) return [...current, entry];
  return [...current.slice(current.length - limit + 1), entry];
}

export function updateRunListStatus(
  runs: RunListItem[],
  runId: string,
  status: RunListItem["status"],
  error: string | null,
  updatedAt: string
): RunListItem[] {
  return runs.map((run) => run.run_id === runId
    ? { ...run, status, error, updated_at: updatedAt }
    : run);
}

export function completeRootNode(
  nodes: Record<string, TaskNode>,
  rootId: string,
  finalOutput: string | null,
  checker: CheckerResult | null,
  updatedAt: string
): Record<string, TaskNode> {
  const root = nodes[rootId];
  if (!root) return nodes;
  return {
    ...nodes,
    [rootId]: {
      ...root,
      status: "completed",
      ...(finalOutput !== null && checker !== null
        ? { last_result: { output: finalOutput, checker } }
        : {}),
      updated_at: updatedAt
    }
  };
}

export function failOpenNodes(
  nodes: Record<string, TaskNode>,
  error: string,
  updatedAt: string
): Record<string, TaskNode> {
  return Object.fromEntries(Object.entries(nodes).map(([id, node]) => {
    if (node.status !== "ready" && node.status !== "active" && node.status !== "waiting") {
      return [id, node];
    }
    return [id, {
      ...node,
      status: "failed" as const,
      last_result: { error },
      updated_at: updatedAt
    }];
  }));
}

export function latestProviderActivity(entries: unknown[]): ProviderActivity | null {
  for (let index = entries.length - 1; index >= 0; index -= 1) {
    const activity = providerActivityFrom(entries[index]);
    if (activity) return activity;
  }
  return null;
}

export function providerActivityFrom(value: unknown): ProviderActivity | null {
  const record = asRecord(value);
  if (!record || typeof record.status !== "string") return null;
  return {
    status: record.status,
    at: typeof record.at === "string" ? record.at : null,
    source: typeof record.source === "string" ? record.source : null
  };
}

export function worldSnapshotsFrom(value: unknown): WorldSnapshot[] {
  if (isWorldSnapshot(value)) return [value];
  const record = asRecord(value);
  if (!record) return [];
  const candidates: WorldSnapshot[] = [];
  for (const key of ["world", "snapshot"] as const) {
    if (isWorldSnapshot(record[key])) candidates.push(record[key]);
  }
  if (Array.isArray(record.frames)) {
    for (const frame of record.frames) if (isWorldSnapshot(frame)) candidates.push(frame);
  }
  return candidates;
}

export function taskNodesFrom(value: unknown): Record<string, TaskNode> | null {
  const record = asRecord(value);
  if (!record) return null;
  for (const node of Object.values(record)) {
    const candidate = asRecord(node);
    if (!candidate || typeof candidate.id !== "string" || typeof candidate.name !== "string"
      || typeof candidate.status !== "string" || !Array.isArray(candidate.child_ids)) return null;
  }
  return record as unknown as Record<string, TaskNode>;
}

export function actionReceiptFrom(value: unknown): ActionReceipt | null {
  const record = asRecord(value);
  if (!record || typeof record.transaction_id !== "string" || typeof record.agent_id !== "string"
    || typeof record.agent_name !== "string"
    || typeof record.name !== "string" || typeof record.accepted !== "boolean"
    || typeof record.code !== "string") return null;
  return record as unknown as ActionReceipt;
}

export function checkerFrom(value: unknown): CheckerResult | null {
  const record = asRecord(value);
  if (!record || typeof record.success !== "boolean" || !Array.isArray(record.checks)) return null;
  return record as unknown as CheckerResult;
}

export function contextMemoryFrom(value: unknown): ContextMemoryState | null {
  const record = asRecord(value);
  if (!record || record.version !== 1
    || typeof record.context_window_tokens !== "number"
    || typeof record.compact_trigger_tokens !== "number"
    || typeof record.active_estimated_tokens !== "number"
    || typeof record.total_compactions !== "number"
    || asRecord(record.scopes) === null) return null;
  return record as unknown as ContextMemoryState;
}

export function isWorldSnapshot(value: unknown): value is WorldSnapshot {
  const record = asRecord(value);
  return record !== null
    && typeof record.frame === "number"
    && typeof record.simulated_time === "number"
    && asRecord(record.robot) !== null
    && Array.isArray(record.objects)
    && Array.isArray(record.zones)
    && Array.isArray(record.obstacles);
}

export function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

export function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}
