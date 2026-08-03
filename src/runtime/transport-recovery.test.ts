import { describe, expect, it } from "vitest";
import {
  ConsecutiveTransportRecovery,
  isTransportInterruption,
  transportRetryPlan
} from "./transport-recovery.js";

describe("isTransportInterruption", () => {
  it("recognizes the socket drop that ended a real mission mid-stream", () => {
    // A mid-stream socket drop is surfaced as a bare TypeError; the transport
    // reason is carried one level down in `cause`.
    const error = Object.assign(new TypeError("terminated"), {
      cause: Object.assign(new Error("other side closed"), {
        name: "SocketError",
        code: "UND_ERR_SOCKET"
      })
    });
    expect(isTransportInterruption(error)).toBe(true);
  });

  it("recognizes a connection reset carried directly on the error", () => {
    expect(isTransportInterruption(Object.assign(new Error("read ECONNRESET"), {
      code: "ECONNRESET"
    }))).toBe(true);
  });

  it("treats throttling and gateway failures as continuable", () => {
    for (const statusCode of [408, 409, 425, 429, 500, 502, 503, 504, 599]) {
      expect(isTransportInterruption(Object.assign(new Error("upstream"), { statusCode })))
        .toBe(true);
    }
    expect(isTransportInterruption(Object.assign(new Error("busy"), { status: 429 }))).toBe(true);
    expect(isTransportInterruption({
      error: { statusCode: 503, message: "nested compatible-provider failure" }
    })).toBe(true);
    expect(isTransportInterruption(Object.assign(new Error("gateway timeout"), {
      name: "ModelTransportError",
      code: 504
    }))).toBe(true);
  });

  it("does not reinterpret an arbitrary numeric business code as HTTP status", () => {
    expect(isTransportInterruption(Object.assign(new Error("domain failure"), {
      code: 503
    }))).toBe(false);
  });

  it.each([
    ["statusCode", 400, false],
    ["status", 401, false],
    ["statusCode", 408, true],
    ["status", 409, true],
    ["statusCode", 425, true],
    ["status", 429, true],
    ["statusCode", 503, true],
    ["status", 503, true]
  ] as const)(
    "uses %s=%i instead of blindly retrying an undici response-status error",
    (field, status, expected) => {
      const error = Object.assign(new Error(`HTTP ${status}`), {
        code: "UND_ERR_RESPONSE_STATUS_CODE",
        [field]: status
      });
      expect(isTransportInterruption(error)).toBe(expected);
    }
  );

  it("does not let a terminal response status fall through to a nested socket code", () => {
    const error = Object.assign(new Error("HTTP 400"), {
      code: "UND_ERR_RESPONSE_STATUS_CODE",
      statusCode: 400,
      cause: Object.assign(new Error("socket closed"), { code: "ECONNRESET" })
    });
    expect(isTransportInterruption(error)).toBe(false);
  });

  it("does not continue through failures that would repeat identically", () => {
    // Replaying these loops on an error only a human can clear, so they must
    // end the run rather than consume the recovery budget.
    expect(isTransportInterruption(Object.assign(new Error("unauthorized"), { status: 401 })))
      .toBe(false);
    expect(isTransportInterruption(Object.assign(new Error("bad request"), { statusCode: 400 })))
      .toBe(false);
    expect(isTransportInterruption(new Error("Mission Coordinator returned no text"))).toBe(false);
    expect(isTransportInterruption(new TypeError("x is not a function"))).toBe(false);
    expect(isTransportInterruption(null)).toBe(false);
    expect(isTransportInterruption("boom")).toBe(false);
  });

  it("finds the transport failure inside an aggregate of retried attempts", () => {
    const aggregate = new AggregateError(
      [new Error("first"), Object.assign(new Error("second"), { code: "ETIMEDOUT" })],
      "all attempts failed"
    );
    expect(isTransportInterruption(aggregate)).toBe(true);
  });

  it("terminates on a self-referencing cause chain instead of hanging", () => {
    const error = new Error("looping") as Error & { cause?: unknown };
    error.cause = error;
    expect(isTransportInterruption(error)).toBe(false);
  });

});

describe("transportRetryPlan", () => {
  it("uses bounded exponential backoff when the server provides no hint", () => {
    expect(transportRetryPlan(new Error("offline"), 1)).toEqual({
      backoffMs: 2_000,
      retryAfterMs: null,
      waitMs: 2_000
    });
    expect(transportRetryPlan(new Error("offline"), 5)).toEqual({
      backoffMs: 30_000,
      retryAfterMs: null,
      waitMs: 30_000
    });
    expect(transportRetryPlan(new Error("offline"), 40).backoffMs).toBe(30_000);
  });

  it("honors a standard Retry-After delay without shortening local backoff", () => {
    const longer = Object.assign(new Error("throttled"), {
      statusCode: 429,
      responseHeaders: { "Retry-After": "75" }
    });
    expect(transportRetryPlan(longer, 1)).toEqual({
      backoffMs: 2_000,
      retryAfterMs: 75_000,
      waitMs: 75_000
    });

    const shorter = Object.assign(new Error("throttled"), {
      response: { headers: new Headers({ "retry-after": "1" }) }
    });
    expect(transportRetryPlan(shorter, 3)).toEqual({
      backoffMs: 8_000,
      retryAfterMs: 1_000,
      waitMs: 8_000
    });
  });

  it("accepts HTTP-date hints, ignores malformed values and caps extreme waits", () => {
    const now = Date.parse("2026-07-30T16:00:00.000Z");
    const dated = Object.assign(new Error("maintenance"), {
      headers: { "retry-after": "Wed, 30 Jul 2026 16:01:30 GMT" }
    });
    expect(transportRetryPlan(dated, 1, now).retryAfterMs).toBe(90_000);

    const malformed = Object.assign(new Error("maintenance"), {
      responseHeaders: { "retry-after": "later" }
    });
    expect(transportRetryPlan(malformed, 1, now).retryAfterMs).toBeNull();

    const extreme = Object.assign(new Error("maintenance"), {
      responseHeaders: { "retry-after": "86400" }
    });
    expect(transportRetryPlan(extreme, 1, now).retryAfterMs).toBe(300_000);
  });

  it("rejects invalid attempt counters", () => {
    expect(() => transportRetryPlan(new Error("offline"), 0))
      .toThrow("positive safe integer");
  });
});

describe("ConsecutiveTransportRecovery", () => {
  it("bounds one outage and opens a fresh window after a completed response", () => {
    const recovery = new ConsecutiveTransportRecovery(3);

    expect(recovery.nextAttempt()).toBe(1);
    expect(recovery.nextAttempt()).toBe(2);
    expect(recovery.responseCompleted()).toBe(2);
    expect(recovery.responseCompleted()).toBe(0);
    expect(recovery.nextAttempt()).toBe(1);
    expect(recovery.nextAttempt()).toBe(2);
    expect(recovery.nextAttempt()).toBe(3);
    expect(recovery.nextAttempt()).toBeNull();
  });

  it("rejects invalid recovery limits", () => {
    expect(() => new ConsecutiveTransportRecovery(0)).toThrow("positive safe integer");
    expect(() => new ConsecutiveTransportRecovery(Number.POSITIVE_INFINITY))
      .toThrow("positive safe integer");
  });
});
