import { createAnthropic } from "@ai-sdk/anthropic";
import { createOpenAI } from "@ai-sdk/openai";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import type { Model } from "@openai/agents";
import { aisdk } from "@openai/agents-extensions/ai-sdk";
import type {
  ModelProviderConfig,
  ProviderConfig
} from "../config/load.js";

export interface ProviderIdentity {
  protocol: ModelProviderConfig["protocol"];
  model: string;
}

export function createConfiguredModel(config: ModelProviderConfig): Model {
  const timedFetch = requestTimedFetch(config.requestTimeoutMs ?? 90_000);
  if (config.protocol === "openai_compatible") {
    const provider = createOpenAICompatible({
      name: "configured-openai-compatible",
      baseURL: config.baseUrl,
      apiKey: config.apiKey,
      fetch: timedFetch
    });
    return aisdk(provider.chatModel(config.model));
  }
  if (config.protocol === "openai_responses") {
    const provider = createOpenAI({
      name: "configured-openai-responses",
      baseURL: config.baseUrl,
      apiKey: config.apiKey,
      fetch: timedFetch
    });
    return aisdk(provider.responses(config.model));
  }
  const provider = createAnthropic({
    name: "configured-anthropic-messages",
    baseURL: config.baseUrl,
    apiKey: config.apiKey,
    fetch: timedFetch
  });
  return aisdk(provider.messages(config.model));
}

export function requestTimedFetch(
  timeoutMs: number,
  implementation: typeof fetch = fetch
): typeof fetch {
  return async (input, init) => {
    const timeout = new AbortController();
    const timer = setTimeout(() => timeout.abort(), timeoutMs);
    const upstream = init?.signal;
    const signal = upstream
      ? AbortSignal.any([upstream, timeout.signal])
      : timeout.signal;
    try {
      return await implementation(input, { ...init, signal });
    } catch (error) {
      if (timeout.signal.aborted && !upstream?.aborted) {
        const cause = Object.assign(new Error(`Model request exceeded ${timeoutMs}ms`), {
          code: "ETIMEDOUT"
        });
        throw new TypeError("fetch failed", { cause });
      }
      throw error;
    } finally {
      clearTimeout(timer);
    }
  };
}

export function providerIdentity(config: ProviderConfig | ModelProviderConfig): ProviderIdentity {
  return {
    protocol: config.protocol,
    model: config.model
  };
}
