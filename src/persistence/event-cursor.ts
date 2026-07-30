import { createHash } from "node:crypto";

const VERSIONED_CURSOR = /^v1:(0|[1-9]\d*):([a-f0-9]{64})$/;

export type ParsedRuntimeEventCursor =
  | { kind: "versioned"; index: number; hash: string }
  | { kind: "legacy" }
  | { kind: "invalid" };

/**
 * A durable SSE cursor identifies the exact journal row, not merely an event
 * payload ID. The run identity is included in the proof so a cursor copied
 * from another run cannot accidentally address the same ordinal and UUID.
 */
export function runtimeEventCursor(
  runId: string,
  eventId: string,
  index: number
): string {
  if (!Number.isSafeInteger(index) || index < 0) {
    throw new Error("Runtime event journal index must be a nonnegative safe integer");
  }
  const hash = runtimeEventCursorHash(runId, eventId);
  return `v1:${index}:${hash}`;
}

export function parseRuntimeEventCursor(value: string): ParsedRuntimeEventCursor {
  if (!value.startsWith("v1:")) return { kind: "legacy" };
  const match = VERSIONED_CURSOR.exec(value);
  if (!match) return { kind: "invalid" };
  const index = Number(match[1]);
  if (!Number.isSafeInteger(index)) return { kind: "invalid" };
  return { kind: "versioned", index, hash: match[2]! };
}

export function runtimeEventCursorMatches(
  parsed: Extract<ParsedRuntimeEventCursor, { kind: "versioned" }>,
  runId: string,
  eventId: string
): boolean {
  return parsed.hash === runtimeEventCursorHash(runId, eventId);
}

function runtimeEventCursorHash(runId: string, eventId: string): string {
  return createHash("sha256")
    .update(runId)
    .update("\0")
    .update(eventId)
    .digest("hex");
}
