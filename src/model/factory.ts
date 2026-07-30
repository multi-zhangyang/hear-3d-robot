import { createAnthropic } from "@ai-sdk/anthropic";
import { createOpenAI } from "@ai-sdk/openai";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import type { Model } from "@openai/agents";
import { aisdk } from "@openai/agents-extensions/ai-sdk";
import type { ProviderConfig } from "../config/load.js";

export interface ProviderIdentity {
  protocol: ProviderConfig["protocol"];
  model: string;
  endpoint: string;
}

export function createConfiguredModel(config: ProviderConfig): Model {
  if (config.protocol === "openai_compatible") {
    const provider = createOpenAICompatible({
      name: "configured-openai-compatible",
      baseURL: config.baseUrl,
      apiKey: config.apiKey
    });
    return aisdk(provider.chatModel(config.model));
  }
  if (config.protocol === "openai_responses") {
    const provider = createOpenAI({
      name: "configured-openai-responses",
      baseURL: config.baseUrl,
      apiKey: config.apiKey
    });
    return aisdk(provider.responses(config.model));
  }
  const provider = createAnthropic({
    name: "configured-anthropic-messages",
    baseURL: config.baseUrl,
    apiKey: config.apiKey
  });
  return aisdk(provider.messages(config.model));
}

export function providerIdentity(config: ProviderConfig): ProviderIdentity {
  return {
    protocol: config.protocol,
    model: config.model,
    endpoint: new URL(config.baseUrl).origin
  };
}
