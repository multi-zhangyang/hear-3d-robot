import { describe, expect, it, vi } from "vitest";
import { isTransportInterruption } from "../runtime/transport-recovery.js";
import { requestTimedFetch } from "./factory.js";

describe("model request timeout", () => {
  it("turns an unbounded provider request into a retryable transport timeout", async () => {
    const implementation = vi.fn((_input: URL | RequestInfo, init?: RequestInit) => (
      new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          reject(new DOMException("aborted", "AbortError"));
        }, { once: true });
      })
    )) as typeof fetch;
    const timed = requestTimedFetch(10, implementation);

    const error = await timed("https://example.test/model").catch((failure: unknown) => failure);

    expect(error).toBeInstanceOf(TypeError);
    expect(isTransportInterruption(error)).toBe(true);
    expect(implementation).toHaveBeenCalledOnce();
  });

  it("preserves a caller cancellation instead of relabeling it as a timeout", async () => {
    const implementation = vi.fn((_input: URL | RequestInfo, init?: RequestInit) => (
      new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true });
      })
    )) as typeof fetch;
    const controller = new AbortController();
    const reason = new Error("operator stopped");
    const request = requestTimedFetch(1_000, implementation)("https://example.test/model", {
      signal: controller.signal
    });
    controller.abort(reason);

    await expect(request).rejects.toBe(reason);
  });
});
