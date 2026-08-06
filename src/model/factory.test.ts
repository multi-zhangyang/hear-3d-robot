import { describe, expect, it, vi } from "vitest";
import { setTracingDisabled, withTrace, type Model } from "@openai/agents";
import { isTransportInterruption } from "../runtime/transport-recovery.js";
import {
  createConfiguredModel,
  openAICompatibleRequestFetch,
  promptCacheAffinityKey,
  promptCacheProviderData,
  providerIdentity,
  requestTimedFetch,
  withPromptCacheAffinity
} from "./factory.js";

setTracingDisabled(true);

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

  it("times out a response body that becomes silent after opening the stream", async () => {
    const implementation = vi.fn((_input: URL | RequestInfo, init?: RequestInit) => (
      Promise.resolve(new Response(new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(new TextEncoder().encode("first"));
          init?.signal?.addEventListener("abort", () => {
            controller.error(init.signal?.reason);
          }, { once: true });
        }
      })))
    )) as typeof fetch;
    const response = await requestTimedFetch(10, implementation)(
      "https://example.test/model"
    );
    const reader = response.body!.getReader();

    expect(new TextDecoder().decode((await reader.read()).value)).toBe("first");
    const error = await reader.read().catch((failure: unknown) => failure);

    expect(error).toBeInstanceOf(TypeError);
    expect(isTransportInterruption(error)).toBe(true);
  });

  it("preserves caller cancellation while reading a streamed response body", async () => {
    const implementation = vi.fn((_input: URL | RequestInfo, init?: RequestInit) => (
      Promise.resolve(new Response(new ReadableStream<Uint8Array>({
        start(controller) {
          init?.signal?.addEventListener("abort", () => {
            controller.error(init.signal?.reason);
          }, { once: true });
        }
      })))
    )) as typeof fetch;
    const controller = new AbortController();
    const reason = new Error("operator stopped streamed response");
    const response = await requestTimedFetch(1_000, implementation)(
      "https://example.test/model",
      { signal: controller.signal }
    );
    const reading = response.body!.getReader().read();
    controller.abort(reason);

    await expect(reading).rejects.toBe(reason);
  });

});

describe("OpenAI-compatible request normalization", () => {
  it("merges a reasoning-only assistant item into its following tool call", async () => {
    const bodies: unknown[] = [];
    const implementation = vi.fn(async (_input: URL | RequestInfo, init?: RequestInit) => {
      bodies.push(JSON.parse(String(init?.body)));
      return new Response(null, { status: 200 });
    });
    const compatible = openAICompatibleRequestFetch(implementation as typeof fetch);

    await compatible("https://example.test/chat/completions", {
      method: "POST",
      body: JSON.stringify({
        model: "configured-model",
        messages: [
          { role: "user", content: "use the tool" },
          { role: "assistant", content: "", reasoning_content: "private reasoning" },
          {
            role: "assistant",
            content: null,
            tool_calls: [{
              id: "call-1",
              type: "function",
              function: { name: "observe", arguments: "{}" }
            }]
          },
          { role: "tool", tool_call_id: "call-1", content: "ok" }
        ]
      })
    });

    expect(bodies).toEqual([{
      model: "configured-model",
      messages: [
        { role: "user", content: "use the tool" },
        {
          role: "assistant",
          content: null,
          tool_calls: [{
            id: "call-1",
            type: "function",
            function: { name: "observe", arguments: "{}" }
          }],
          reasoning_content: "private reasoning"
        },
        { role: "tool", tool_call_id: "call-1", content: "ok" }
      ]
    }]);
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

describe("prompt cache affinity", () => {
  const provider = {
    protocol: "openai_compatible" as const,
    baseUrl: "https://private.example.test/v1",
    model: "configured-model",
    apiKey: "private-credential",
    requestTimeoutMs: 90_000,
    temperature: 0.2,
    contextWindowTokens: 262_144,
    compactTriggerTokens: 222_822,
    compactRecentModelTurns: 4
  };

  it("derives a stable opaque key per cache namespace, Agent, and credential", () => {
    const first = promptCacheAffinityKey({
      namespace: "run:epoch",
      agentId: "humanoid-coordinator",
      provider
    });
    const repeated = promptCacheAffinityKey({
      namespace: "run:epoch",
      agentId: "humanoid-coordinator",
      provider
    });
    const sibling = promptCacheAffinityKey({
      namespace: "run:epoch",
      agentId: "humanoid-motion-reference",
      provider
    });
    const otherCredential = promptCacheAffinityKey({
      namespace: "run:epoch",
      agentId: "humanoid-coordinator",
      provider: { ...provider, apiKey: "another-private-credential" }
    });

    expect(first).toMatch(/^[a-f0-9]{64}$/);
    expect(repeated).toBe(first);
    expect(sibling).not.toBe(first);
    expect(otherCredential).not.toBe(first);
    expect(first).not.toContain("private.example.test");
    expect(first).not.toContain("configured-model");
    expect(first).not.toContain("private-credential");
  });

  it("merges protocol-native cache settings without dropping caller options", () => {
    expect(promptCacheProviderData("openai_compatible", "cache-key", {
      providerOptions: {
        "configured-openai-compatible": { reasoning_effort: "high" }
      }
    })).toEqual({
      providerOptions: {
        "configured-openai-compatible": {
          reasoning_effort: "high",
          prompt_cache_key: "cache-key"
        }
      }
    });
    expect(promptCacheProviderData("openai_responses", "cache-key")).toEqual({
      providerOptions: { openai: { promptCacheKey: "cache-key" } }
    });
    expect(promptCacheProviderData("anthropic_messages", "cache-key")).toEqual({
      providerOptions: {
        anthropic: { cacheControl: { type: "ephemeral" } }
      }
    });
  });

  it("injects the same affinity key into every request on one model facade", async () => {
    const requests: Array<Parameters<Model["getResponse"]>[0]> = [];
    const underlying = {
      getResponse: async (request) => {
        requests.push(request);
        throw new Error("captured request");
      },
      getStreamedResponse: () => {
        throw new Error("outside test");
      }
    } satisfies Model;
    const model = withPromptCacheAffinity(
      underlying,
      "openai_compatible",
      "stable-key"
    );
    const request = {
      input: [{ role: "user" as const, content: "first" }],
      modelSettings: {},
      tools: [],
      outputType: "text" as const,
      handoffs: [],
      tracing: false as const
    };

    await expect(model.getResponse(request)).rejects.toThrow("captured request");
    await expect(model.getResponse({
      ...request,
      input: [{ role: "user", content: "second" }]
    })).rejects.toThrow("captured request");

    expect(requests).toHaveLength(2);
    expect(requests.map((captured) => captured.modelSettings.providerData)).toEqual([
      {
        providerOptions: {
          "configured-openai-compatible": { prompt_cache_key: "stable-key" }
        }
      },
      {
        providerOptions: {
          "configured-openai-compatible": { prompt_cache_key: "stable-key" }
        }
      }
    ]);
  });

  it("reports exact request-prefix stability without recording prompt content", async () => {
    const traces: Array<{
      requestSha256: string;
      messageCount: number;
      previousMessageCount: number | null;
      commonMessagePrefixCount: number;
      commonMessagePrefixBytes: number;
      appendOnlyMessagePrefix: boolean;
      toolCount: number;
      toolsStable: boolean;
      settingsStable: boolean;
      cacheAffinityPresent: boolean;
    }> = [];
    const implementation = vi.fn(async () => new Response(null, { status: 200 }));
    const timed = requestTimedFetch(
      1_000,
      implementation as typeof fetch,
      (trace) => { traces.push(trace); }
    );
    const sharedMessages = [
      { role: "system", content: "private-system-prefix" },
      { role: "user", content: "private-user-prefix" }
    ];
    const tools = [{
      type: "function",
      function: { name: "observe", parameters: { type: "object" } }
    }];
    const request = (messages: unknown[]) => timed("https://example.test/model", {
      method: "POST",
      body: JSON.stringify({
        model: "configured-model",
        messages,
        tools,
        prompt_cache_key: "opaque-affinity"
      })
    });

    await request(sharedMessages);
    await request([...sharedMessages, { role: "assistant", content: "new suffix" }]);

    expect(traces).toHaveLength(2);
    expect(traces[0]).toMatchObject({
      messageCount: 2,
      previousMessageCount: null,
      commonMessagePrefixCount: 0,
      appendOnlyMessagePrefix: false,
      toolCount: 1,
      toolsStable: true,
      settingsStable: true,
      cacheAffinityPresent: true
    });
    expect(traces[1]).toMatchObject({
      messageCount: 3,
      previousMessageCount: 2,
      commonMessagePrefixCount: 2,
      appendOnlyMessagePrefix: true,
      toolCount: 1,
      toolsStable: true,
      settingsStable: true,
      cacheAffinityPresent: true
    });
    const serialized = JSON.stringify(traces);
    expect(serialized).not.toContain("private-system-prefix");
    expect(serialized).not.toContain("private-user-prefix");
    expect(serialized).not.toContain("new suffix");
    expect(traces.every((trace) => /^[a-f0-9]{64}$/.test(trace.requestSha256))).toBe(true);
  });

  it("sends the affinity key on the OpenAI-compatible HTTP request", async () => {
    const requestBodies: unknown[] = [];
    const fetchMock = vi.fn(async (_input: URL | RequestInfo, init?: RequestInit) => {
      requestBodies.push(JSON.parse(String(init?.body)));
      return new Response(JSON.stringify({
        id: "chatcmpl-cache-wire",
        object: "chat.completion",
        created: 1,
        model: provider.model,
        choices: [{
          index: 0,
          message: { role: "assistant", content: "ok" },
          finish_reason: "stop"
        }],
        usage: {
          prompt_tokens: 12,
          completion_tokens: 1,
          total_tokens: 13,
          prompt_tokens_details: { cached_tokens: 8 }
        }
      }), {
        status: 200,
        headers: { "content-type": "application/json" }
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    try {
      const model = createConfiguredModel(provider, {
        promptCacheKey: "stable-wire-key"
      });
      await withTrace("prompt-cache-wire-test", () => model.getResponse({
        input: [{ role: "user", content: "hello" }],
        modelSettings: {},
        tools: [],
        outputType: "text",
        handoffs: [],
        tracing: false
      }));
    } finally {
      vi.unstubAllGlobals();
    }

    expect(requestBodies).toHaveLength(1);
    expect(requestBodies[0]).toMatchObject({
      model: provider.model,
      prompt_cache_key: "stable-wire-key"
    });
  });

  it("negotiates an unsupported cache parameter once and retries transparently", async () => {
    const requests: Array<Parameters<Model["getResponse"]>[0]> = [];
    const statuses: Array<{ status: string; compatibilityRetry: boolean }> = [];
    const underlying = {
      getResponse: async (request) => {
        requests.push(request);
        if (request.modelSettings.providerData) {
          throw Object.assign(new Error(
            "Validation: Unsupported parameter(s): `prompt_cache_key`"
          ), { statusCode: 400 });
        }
        return {
          usage: {
            requests: 1,
            inputTokens: 1,
            outputTokens: 1,
            totalTokens: 2,
            inputTokensDetails: [],
            outputTokensDetails: []
          },
          output: []
        };
      },
      getStreamedResponse: () => {
        throw new Error("outside test");
      }
    } satisfies Model;
    const model = withPromptCacheAffinity(
      underlying,
      "openai_compatible",
      "stable-key",
      (status) => { statuses.push(status); }
    );
    const request = {
      input: [{ role: "user" as const, content: "first" }],
      modelSettings: {},
      tools: [],
      outputType: "text" as const,
      handoffs: [],
      tracing: false as const
    };

    await expect(model.getResponse(request)).resolves.toMatchObject({ output: [] });
    await expect(model.getResponse(request)).resolves.toMatchObject({ output: [] });

    expect(requests).toHaveLength(3);
    expect(requests[0]?.modelSettings.providerData).toBeDefined();
    expect(requests[1]?.modelSettings.providerData).toBeUndefined();
    expect(requests[2]?.modelSettings.providerData).toBeUndefined();
    expect(statuses).toEqual([{
      status: "unsupported",
      compatibilityRetry: true
    }]);
  });
});
