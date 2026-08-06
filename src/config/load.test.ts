import { describe, expect, it } from "vitest";
import { loadProviderConfig, loadServerConfig } from "./load.js";

const required = {
  AI_PROVIDER: "openai_compatible",
  AI_BASE_URL: "https://example.test/v1",
  AI_MODEL: "model-name",
  AI_API_KEY: "test-key"
};

describe("provider context budget", () => {
  it("derives the compaction trigger from the configured context window", () => {
    const config = loadProviderConfig({
      ...required,
      AI_CONTEXT_WINDOW_TOKENS: "32768",
      AI_MAX_OUTPUT_TOKENS: "4096",
      AI_COMPACT_MAX_OUTPUT_TOKENS: "1024"
    });
    expect(config).toMatchObject({
      contextWindowTokens: 32768,
      compactTriggerTokens: Math.floor(32768 * 0.85),
      compactRecentModelTurns: 4,
      compactMaxOutputTokens: 1024
    });
  });

  it("defaults to a 262k context without imposing model output limits", () => {
    const config = loadProviderConfig({
      ...required,
      AI_MAX_OUTPUT_TOKENS: "",
      AI_COMPACT_MAX_OUTPUT_TOKENS: ""
    });

    expect(config.contextWindowTokens).toBe(262_144);
    expect(config.requestTimeoutMs).toBe(300_000);
    expect(config.streamEventIdleTimeoutMs).toBe(300_000);
    expect(config.compactTriggerTokens).toBe(Math.floor(262_144 * 0.85));
    expect(config.maxOutputTokens).toBeUndefined();
    expect(config.compactMaxOutputTokens).toBeUndefined();
    expect(config.agentModels?.executor.maxOutputTokens).toBeUndefined();
    expect(config.agentModels?.executor.requestTimeoutMs).toBe(300_000);
    expect(config.agentModels?.executor.streamEventIdleTimeoutMs).toBe(300_000);
    expect(config.agentModels?.compactor.compactMaxOutputTokens).toBeUndefined();
  });

  it("rejects a trigger that conflicts with an explicitly configured output limit", () => {
    expect(() => loadProviderConfig({
      ...required,
      AI_CONTEXT_WINDOW_TOKENS: "16000",
      AI_COMPACT_TRIGGER_TOKENS: "11000",
      AI_MAX_OUTPUT_TOKENS: "5000",
      AI_COMPACT_MAX_OUTPUT_TOKENS: "2000"
    })).toThrow("explicitly configured AI_MAX_OUTPUT_TOKENS");
  });

  it("reserves the model window for every turn in a fresh compactor attempt", () => {
    expect(() => loadProviderConfig({
      ...required,
      AI_CONTEXT_WINDOW_TOKENS: "16000",
      AI_COMPACT_TRIGGER_TOKENS: "1000",
      AI_MAX_OUTPUT_TOKENS: "1000",
      AI_COMPACT_MAX_OUTPUT_TOKENS: "8000"
    })).toThrow("compactor repair turns");
  });

  it("inherits AI defaults while resolving independent provider-neutral agent profiles", () => {
    const config = loadProviderConfig({
      ...required,
      AI_GOAL_MANAGER_MODEL: "goal-manager-model",
      AI_COORDINATOR_TEMPERATURE: "0.1",
      AI_SENTRY_MAX_OUTPUT_TOKENS: "2048",
      AI_MOTION_PROVIDER: "anthropic_messages",
      AI_MOTION_BASE_URL: "https://motion.example.test/messages",
      AI_MOTION_MODEL: "motion-model",
      AI_MOTION_API_KEY: "motion-key",
      AI_EXECUTOR_CONTEXT_WINDOW_TOKENS: "131072",
      AI_EXECUTOR_COMPACT_TRIGGER_TOKENS: "24000",
      AI_COMPACTOR_MODEL: "compactor-model",
      AI_COMPACTOR_COMPACT_MAX_OUTPUT_TOKENS: "2048"
    });

    expect(config.agentModels?.goal_manager).toMatchObject({
      model: "goal-manager-model"
    });
    expect(config.agentModels?.coordinator).toMatchObject({
      model: "model-name",
      temperature: 0.1
    });
    expect(config.agentModels?.sentry).toMatchObject({
      protocol: "openai_compatible",
      baseUrl: "https://example.test/v1",
      apiKey: "test-key",
      maxOutputTokens: 2048
    });
    expect(config.agentModels?.motion).toMatchObject({
      protocol: "anthropic_messages",
      baseUrl: "https://motion.example.test/messages",
      model: "motion-model",
      apiKey: "motion-key"
    });
    expect(config.agentModels?.executor).toMatchObject({
      contextWindowTokens: 131072,
      compactTriggerTokens: 24000
    });
    expect(config.agentModels?.compactor).toMatchObject({
      model: "compactor-model",
      compactMaxOutputTokens: 2048
    });
  });

  it("derives an independent 85% trigger when a role overrides its window", () => {
    const config = loadProviderConfig({
      ...required,
      AI_EXECUTOR_CONTEXT_WINDOW_TOKENS: "131072"
    });

    expect(config.agentModels?.executor).toMatchObject({
      contextWindowTokens: 131072,
      compactTriggerTokens: Math.floor(131072 * 0.85)
    });
  });

  it("uses the same 85% high-water mark for every configured window", () => {
    const config = loadProviderConfig({
      ...required,
      AI_CONTEXT_WINDOW_TOKENS: "16000"
    });

    expect(config.compactTriggerTokens).toBe(Math.floor(16000 * 0.85));
  });

  it("validates every agent profile before the runtime can create a model", () => {
    expect(() => loadProviderConfig({
      ...required,
      AI_SENTRY_CONTEXT_WINDOW_TOKENS: "16000",
      AI_SENTRY_COMPACT_TRIGGER_TOKENS: "11000",
      AI_SENTRY_MAX_OUTPUT_TOKENS: "5000",
      AI_SENTRY_COMPACT_MAX_OUTPUT_TOKENS: "2000"
    })).toThrow("explicitly configured AI_MAX_OUTPUT_TOKENS");
  });
});

describe("operator network boundary", () => {
  it("keeps an unprotected operator on a loopback address", () => {
    expect(loadServerConfig({ HEAR_HOST: "127.23.4.5" })).toMatchObject({
      host: "127.23.4.5",
      password: ""
    });
    expect(loadServerConfig({ HEAR_HOST: "::1" })).toMatchObject({
      host: "::1",
      password: ""
    });
  });

  it("requires authentication before binding beyond loopback", () => {
    expect(() => loadServerConfig({ HEAR_HOST: "0.0.0.0" }))
      .toThrow("HEAR_OPERATOR_PASSWORD is required");
    expect(() => loadServerConfig({ HEAR_HOST: "192.168.1.20" }))
      .toThrow("HEAR_OPERATOR_PASSWORD is required");
    expect(loadServerConfig({
      HEAR_HOST: "0.0.0.0",
      HEAR_OPERATOR_PASSWORD: "operator-secret"
    })).toMatchObject({ host: "0.0.0.0", password: "operator-secret" });
  });
});
