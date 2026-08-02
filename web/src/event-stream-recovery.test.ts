import { describe, expect, it } from "vitest";
import {
  eventStreamFailureDecision,
  eventStreamRetryDelay
} from "./event-stream-recovery";

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

  it("backs off repeated disconnects without allowing an unbounded wait", () => {
    expect([1, 2, 3, 4, 5, 20].map(eventStreamRetryDelay))
      .toEqual([800, 1_600, 3_200, 6_400, 12_800, 15_000]);
    expect(() => eventStreamRetryDelay(0)).toThrow("positive safe integer");
  });
});
