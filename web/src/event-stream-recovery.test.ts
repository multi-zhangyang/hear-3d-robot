import { describe, expect, it } from "vitest";
import { eventStreamFailureDecision } from "./event-stream-recovery";

describe("event stream recovery", () => {
  it("refreshes the authoritative snapshot when a durable cursor is no longer known", () => {
    const unknownCursor = Object.assign(new Error("Unknown event cursor"), { status: 409 });
    expect(eventStreamFailureDecision(unknownCursor)).toBe("refresh_snapshot");
  });

  it("retries transient failures and stops on permanent client or payload errors", () => {
    expect(eventStreamFailureDecision(Object.assign(new Error("Unavailable"), { status: 503 })))
      .toBe("retry_stream");
    expect(eventStreamFailureDecision(Object.assign(new Error("Not found"), { status: 404 })))
      .toBe("stop");
    expect(eventStreamFailureDecision(new SyntaxError("Invalid event"))).toBe("stop");
  });
});
