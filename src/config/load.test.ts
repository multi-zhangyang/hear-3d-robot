import { describe, expect, it } from "vitest";
import { loadProviderConfig } from "./load.js";

const required = {
  AI_PROVIDER: "openai_compatible",
  AI_BASE_URL: "https://example.test/v1",
  AI_MODEL: "model-name",
  AI_API_KEY: "test-key"
};

describe("provider context budget", () => {
  it("derives a bounded compaction trigger from the configured context window", () => {
    const config = loadProviderConfig({
      ...required,
      AI_CONTEXT_WINDOW_TOKENS: "32768",
      AI_MAX_OUTPUT_TOKENS: "4096",
      AI_COMPACT_MAX_OUTPUT_TOKENS: "1024"
    });
    expect(config).toMatchObject({
      contextWindowTokens: 32768,
      compactTriggerTokens: Math.min(18_000, Math.floor(32768 * 0.4)),
      compactRecentModelTurns: 4,
      compactMaxOutputTokens: 1024
    });
  });

  it("rejects a trigger that leaves no room for model and compactor output", () => {
    expect(() => loadProviderConfig({
      ...required,
      AI_CONTEXT_WINDOW_TOKENS: "16000",
      AI_COMPACT_TRIGGER_TOKENS: "10000",
      AI_MAX_OUTPUT_TOKENS: "5000",
      AI_COMPACT_MAX_OUTPUT_TOKENS: "2000"
    })).toThrow("output reserves");
  });

  it("reserves the model window for every bounded compactor repair turn", () => {
    expect(() => loadProviderConfig({
      ...required,
      AI_CONTEXT_WINDOW_TOKENS: "16000",
      AI_COMPACT_TRIGGER_TOKENS: "1000",
      AI_MAX_OUTPUT_TOKENS: "1000",
      AI_COMPACT_MAX_OUTPUT_TOKENS: "2000"
    })).toThrow("compactor repair turns");
  });
});
