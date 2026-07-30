import { describe, expect, it } from "vitest";
import { errorMessage } from "./error-message.js";

describe("errorMessage", () => {
  it("preserves structured provider errors instead of collapsing them", () => {
    expect(errorMessage({
      status: 503,
      error: { message: "upstream unavailable", type: "provider_error" }
    })).toBe(
      '{"status":503,"error":{"message":"upstream unavailable","type":"provider_error"}}'
    );
  });

  it("redacts credential fields recursively", () => {
    expect(errorMessage({
      message: "request failed",
      authorization: "Bearer private",
      cause: { api_key: "private", status: 401 }
    })).toBe(
      '{"message":"request failed","authorization":"[redacted]","cause":{"api_key":"[redacted]","status":401}}'
    );
  });

  it("does not serialize an SDK Error's full agent state", () => {
    const error = new Error("Max turns exceeded") as Error & {
      status: number;
      state: { prompt: string; generatedItems: string[] };
    };
    error.name = "MaxTurnsExceededError";
    error.status = 429;
    error.state = {
      prompt: "large private working context",
      generatedItems: ["reasoning", "reasoning"]
    };

    expect(errorMessage(error)).toBe(
      '{"name":"MaxTurnsExceededError","message":"Max turns exceeded","status":429}'
    );
  });
});
