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
      compactTriggerTokens: 28_672,
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
    expect(config.toolChoice).toBe("auto");
    expect(config.requestTimeoutMs).toBe(300_000);
    expect(config.streamEventIdleTimeoutMs).toBe(300_000);
    expect(config.compactTriggerTokens).toBe(249_037);
    expect(config.maxOutputTokens).toBeUndefined();
    expect(config.compactMaxOutputTokens).toBeUndefined();
    expect(config.agentModels?.motor_intent.maxOutputTokens).toBeUndefined();
    expect(config.agentModels?.motor_intent.requestTimeoutMs).toBe(300_000);
    expect(config.agentModels?.motor_intent.streamEventIdleTimeoutMs).toBe(300_000);
    expect(config.agentModels?.compactor.compactMaxOutputTokens).toBeUndefined();
  });

  it("supports provider-neutral automatic tool selection without a model special case", () => {
    const config = loadProviderConfig({
      ...required,
      AI_TOOL_CHOICE: "auto",
      AI_MOTOR_INTENT_TOOL_CHOICE: "required"
    });

    expect(config.toolChoice).toBe("auto");
    expect(config.agentModels?.associative.toolChoice).toBe("auto");
    expect(config.agentModels?.motor_intent.toolChoice).toBe("required");
  });

  it("applies provider-neutral reasoning effort and one-million-token context to every role", () => {
    const config = loadProviderConfig({
      ...required,
      AI_CONTEXT_WINDOW_TOKENS: "1000000",
      AI_REASONING_EFFORT: "high"
    });

    expect(config.contextWindowTokens).toBe(1_000_000);
    expect(config.compactTriggerTokens).toBe(983_616);
    expect(config.reasoningEffort).toBe("high");
    for (const profile of Object.values(config.agentModels ?? {})) {
      expect(profile.contextWindowTokens).toBe(1_000_000);
      expect(profile.compactTriggerTokens).toBe(983_616);
      expect(profile.reasoningEffort).toBe("high");
    }
  });

  it("rejects unknown reasoning effort values", () => {
    expect(() => loadProviderConfig({
      ...required,
      AI_REASONING_EFFORT: "ultra"
    })).toThrow("AI_REASONING_EFFORT must be");
  });

  it("rejects unknown tool choice values", () => {
    expect(() => loadProviderConfig({
      ...required,
      AI_TOOL_CHOICE: "sometimes"
    })).toThrow("AI_TOOL_CHOICE must be auto, required, or none");
  });

  it("rejects disabling formal tool decisions for any autonomous role", () => {
    expect(() => loadProviderConfig({
      ...required,
      AI_TOOL_CHOICE: "none"
    })).toThrow("cannot run an autonomous Harness");
    expect(() => loadProviderConfig({
      ...required,
      AI_MOTOR_INTENT_TOOL_CHOICE: "none"
    })).toThrow("cannot run an autonomous Harness");
  });

  it("rejects a trigger that conflicts with an explicitly configured output limit", () => {
    expect(() => loadProviderConfig({
      ...required,
      AI_CONTEXT_WINDOW_TOKENS: "16000",
      AI_COMPACT_TRIGGER_TOKENS: "11001",
      AI_MAX_OUTPUT_TOKENS: "5000",
      AI_COMPACT_MAX_OUTPUT_TOKENS: "2000"
    })).toThrow("explicitly configured AI_MAX_OUTPUT_TOKENS");
  });

  it("rejects a compactor output budget that consumes its entire model window", () => {
    expect(() => loadProviderConfig({
      ...required,
      AI_CONTEXT_WINDOW_TOKENS: "16000",
      AI_COMPACT_TRIGGER_TOKENS: "1000",
      AI_MAX_OUTPUT_TOKENS: "1000",
      AI_COMPACT_MAX_OUTPUT_TOKENS: "16000"
    })).toThrow("one compactor request");
  });

  it("inherits AI defaults while resolving independent provider-neutral agent profiles", () => {
    const config = loadProviderConfig({
      ...required,
      AI_EXECUTIVE_MODEL: "executive-model",
      AI_EXECUTIVE_TEMPERATURE: "0.1",
      AI_ASSOCIATIVE_MAX_OUTPUT_TOKENS: "2048",
      AI_MOTOR_INTENT_PROVIDER: "anthropic_messages",
      AI_MOTOR_INTENT_BASE_URL: "https://motion.example.test/messages",
      AI_MOTOR_INTENT_MODEL: "motion-model",
      AI_MOTOR_INTENT_API_KEY: "motion-key",
      AI_SENSORIMOTOR_CONTEXT_WINDOW_TOKENS: "131072",
      AI_SENSORIMOTOR_COMPACT_TRIGGER_TOKENS: "24000",
      AI_COMPACTOR_MODEL: "compactor-model",
      AI_COMPACTOR_COMPACT_MAX_OUTPUT_TOKENS: "2048"
    });

    expect(config.agentModels?.executive).toMatchObject({
      model: "executive-model",
      temperature: 0.1
    });
    expect(config.agentModels?.associative).toMatchObject({
      protocol: "openai_compatible",
      baseUrl: "https://example.test/v1",
      apiKey: "test-key",
      maxOutputTokens: 2048
    });
    expect(config.agentModels?.motor_intent).toMatchObject({
      protocol: "anthropic_messages",
      baseUrl: "https://motion.example.test/messages",
      model: "motion-model",
      apiKey: "motion-key"
    });
    expect(config.agentModels?.sensorimotor).toMatchObject({
      contextWindowTokens: 131072,
      compactTriggerTokens: 24000
    });
    expect(config.agentModels?.compactor).toMatchObject({
      model: "compactor-model",
      compactMaxOutputTokens: 2048
    });
  });

  it("derives an independent output-reserved trigger when a role overrides its window", () => {
    const config = loadProviderConfig({
      ...required,
      AI_SENSORIMOTOR_CONTEXT_WINDOW_TOKENS: "131072"
    });

    expect(config.agentModels?.sensorimotor).toMatchObject({
      contextWindowTokens: 131072,
      compactTriggerTokens: 124_519
    });
  });

  it("reserves output capacity for every configured window", () => {
    const config = loadProviderConfig({
      ...required,
      AI_CONTEXT_WINDOW_TOKENS: "16000"
    });

    expect(config.compactTriggerTokens).toBe(11_904);
  });

  it("validates every agent profile before the runtime can create a model", () => {
    expect(() => loadProviderConfig({
      ...required,
      AI_ASSOCIATIVE_CONTEXT_WINDOW_TOKENS: "16000",
      AI_ASSOCIATIVE_COMPACT_TRIGGER_TOKENS: "11001",
      AI_ASSOCIATIVE_MAX_OUTPUT_TOKENS: "5000",
      AI_ASSOCIATIVE_COMPACT_MAX_OUTPUT_TOKENS: "2000"
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
