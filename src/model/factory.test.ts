import { describe, expect, it, vi } from "vitest";
import { isTransportInterruption } from "../runtime/transport-recovery.js";
import { providerIdentity, requestTimedFetch } from "./factory.js";

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

describe("public provider identity", () => {
  it("never publishes the configured endpoint or credential", () => {
    const identity = providerIdentity({
      protocol: "openai_compatible",
      baseUrl: "https://private.example.test/v1",
      model: "configured-model",
      apiKey: "private-credential",
      requestTimeoutMs: 90_000,
      temperature: 0.2,
      maxOutputTokens: 2048,
      contextWindowTokens: 32_768,
      compactTriggerTokens: 8_192,
      compactRecentModelTurns: 4,
      compactMaxOutputTokens: 1_024
    });

    expect(identity).toEqual({
      protocol: "openai_compatible",
      model: "configured-model"
    });
    expect(JSON.stringify(identity)).not.toContain("private.example.test");
    expect(JSON.stringify(identity)).not.toContain("private-credential");
  });
});
