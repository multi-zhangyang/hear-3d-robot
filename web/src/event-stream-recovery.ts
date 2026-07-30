export type EventStreamFailureDecision = "refresh_snapshot" | "retry_stream" | "stop";

export function eventStreamFailureDecision(error: Error): EventStreamFailureDecision {
  const status = "status" in error && typeof error.status === "number"
    ? error.status
    : undefined;
  if (status === 409) return "refresh_snapshot";
  if (status !== undefined && status < 500) return "stop";
  if (error instanceof SyntaxError) return "stop";
  return "retry_stream";
}
