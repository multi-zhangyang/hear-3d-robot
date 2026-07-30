import { describe, expect, it } from "vitest";
import { summarizeFailure } from "./failure-summary";

describe("summarizeFailure", () => {
  it("leaves a plain sentence alone", () => {
    const summary = summarizeFailure("Mission budget exhausted after 40 model calls");
    expect(summary.structured).toBe(false);
    expect(summary.headline).toBe("Mission budget exhausted after 40 model calls");
    expect(summary.facts).toEqual([]);
  });

  it("reaches the innermost cause of a socket failure", () => {
    // This is the exact shape a dropped upstream connection produced in a real
    // run: the outer message says nothing, and the reason is two levels down.
    const summary = summarizeFailure(
      JSON.stringify({
        name: "TypeError",
        message: "terminated",
        cause: {
          name: "SocketError",
          message: "other side closed",
          code: "UND_ERR_SOCKET",
          bytesWritten: 21045,
          bytesRead: 24305
        }
      })
    );
    expect(summary.headline).toBe("other side closed");
    expect(summary.facts).toContainEqual({ label: "代码", value: "UND_ERR_SOCKET" });
    expect(summary.structured).toBe(true);
  });

  it("surfaces the status and message of an HTTP-shaped error", () => {
    const summary = summarizeFailure(
      JSON.stringify({ status: 503, error: { message: "upstream unavailable", type: "provider_error" } })
    );
    expect(summary.headline).toBe("upstream unavailable");
    expect(summary.facts).toContainEqual({ label: "状态", value: "503" });
  });

  it("falls back to the error name when no message survives", () => {
    const summary = summarizeFailure(JSON.stringify({ name: "AbortError", cause: { code: "ABORT_ERR" } }));
    expect(summary.headline).toBe("AbortError");
    expect(summary.facts).toContainEqual({ label: "代码", value: "ABORT_ERR" });
  });

  it("keeps the raw text so the evidence is still reachable", () => {
    const raw = JSON.stringify({ message: "terminated", cause: { message: "other side closed" } });
    expect(summarizeFailure(raw).raw).toBe(raw);
  });

  it("treats prose that starts with a brace as prose", () => {
    const summary = summarizeFailure("{not json after all");
    expect(summary.structured).toBe(false);
    expect(summary.headline).toBe("{not json after all");
  });
});
