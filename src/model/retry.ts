import {
  retryPolicies,
  type ModelRetrySettings
} from "@openai/agents";

const transientModelFailure = retryPolicies.any(
  retryPolicies.providerSuggested(),
  retryPolicies.retryAfter(),
  retryPolicies.networkError(),
  retryPolicies.httpStatus([408, 409, 425, 429, 500, 502, 503, 504])
);

/**
 * Request-level recovery owned by the Agents SDK. Its replay-safety checks
 * prevent retries after visible streamed output and preserve provider vetoes.
 * Mission/robot recovery remains a separate Harness responsibility.
 */
export function agentsModelRetrySettings(): ModelRetrySettings {
  return {
    maxRetries: 4,
    backoff: {
      initialDelayMs: 500,
      maxDelayMs: 5_000,
      multiplier: 2,
      jitter: true
    },
    policy: transientModelFailure
  };
}
