export type EventStreamFailureDecision = "refresh_snapshot" | "retry_stream" | "stop";

const INITIAL_EVENT_STREAM_RETRY_MS = 800;
const MAXIMUM_EVENT_STREAM_RETRY_MS = 15_000;

export function eventStreamFailureDecision(error: Error): EventStreamFailureDecision {
  const status = "status" in error && typeof error.status === "number"
    ? error.status
    : undefined;
  if (status === 409) return "refresh_snapshot";
  if (status !== undefined && status < 500) return "stop";
  if (error instanceof SyntaxError) return "stop";
  return "retry_stream";
}

export function eventStreamRetryDelay(attempt: number): number {
  if (!Number.isSafeInteger(attempt) || attempt < 1) {
    throw new Error("Event stream retry attempt must be a positive safe integer");
  }
  return Math.min(
    INITIAL_EVENT_STREAM_RETRY_MS * 2 ** Math.min(attempt - 1, 30),
    MAXIMUM_EVENT_STREAM_RETRY_MS
  );
}
