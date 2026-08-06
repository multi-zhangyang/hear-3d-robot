import { z } from "zod";

const ModelUsageTotalsSchema = z.object({
  requests: z.number().int().nonnegative(),
  reported_requests: z.number().int().nonnegative(),
  input_tokens: z.number().int().nonnegative(),
  output_tokens: z.number().int().nonnegative(),
  total_tokens: z.number().int().nonnegative(),
  cached_input_tokens: z.number().int().nonnegative(),
  reasoning_tokens: z.number().int().nonnegative()
}).strict().superRefine((usage, context) => {
  if (usage.reported_requests > usage.requests) {
    context.addIssue({
      code: "custom",
      path: ["reported_requests"],
      message: "Reported model-usage requests cannot exceed all requests"
    });
  }
});

type ModelUsageTotals = z.infer<typeof ModelUsageTotalsSchema>;

export const ModelUsageStateSchema = z.object({
  version: z.literal(1),
  total: ModelUsageTotalsSchema,
  by_agent: z.record(z.string().trim().min(1), ModelUsageTotalsSchema),
  updated_at: z.string().datetime().nullable()
}).strict();

export type ModelUsageState = z.infer<typeof ModelUsageStateSchema>;

const EmptyModelUsageTotals: ModelUsageTotals = {
  requests: 0,
  reported_requests: 0,
  input_tokens: 0,
  output_tokens: 0,
  total_tokens: 0,
  cached_input_tokens: 0,
  reasoning_tokens: 0
};

export const EmptyModelUsageState: ModelUsageState = {
  version: 1,
  total: { ...EmptyModelUsageTotals },
  by_agent: {},
  updated_at: null
};

export interface ModelUsageDelta {
  agentId: string;
  usage: ModelUsageTotals;
}

/** Reads provider-neutral AI SDK usage without assuming a vendor payload. */
export function modelUsageDeltaFromProviderEvent(
  value: unknown,
  attributedAgentId?: string
): ModelUsageDelta | undefined {
  const event = record(value);
  const usage = record(event?.usage);
  if (!usage) return undefined;
  const requests = integer(usage.requests) ?? 1;
  if (requests === 0) return undefined;
  const agentId = attributedAgentId?.trim() || text(event?.agent_id);
  if (!agentId) return undefined;
  const inputTokens = integer(usage.inputTokens) ?? integer(usage.input_tokens);
  const outputTokens = integer(usage.outputTokens) ?? integer(usage.output_tokens);
  const totalTokens = integer(usage.totalTokens) ?? integer(usage.total_tokens);
  const inputDetails = detailRecords(
    usage.inputTokensDetails ?? usage.input_tokens_details
  );
  const outputDetails = detailRecords(
    usage.outputTokensDetails ?? usage.output_tokens_details
  );
  const reported = inputTokens !== undefined
    || outputTokens !== undefined
    || totalTokens !== undefined;
  return {
    agentId,
    usage: ModelUsageTotalsSchema.parse({
      requests,
      reported_requests: reported ? requests : 0,
      input_tokens: inputTokens ?? 0,
      output_tokens: outputTokens ?? 0,
      total_tokens: totalTokens ?? ((inputTokens ?? 0) + (outputTokens ?? 0)),
      cached_input_tokens: detailTotal(inputDetails, [
        "cachedTokens",
        "cached_tokens",
        "cacheReadTokens",
        "cache_read_tokens"
      ]),
      reasoning_tokens: detailTotal(outputDetails, [
        "reasoningTokens",
        "reasoning_tokens"
      ])
    })
  };
}

export function addModelUsage(
  state: ModelUsageState,
  delta: ModelUsageDelta,
  at = new Date().toISOString()
): ModelUsageState {
  const current = state.by_agent[delta.agentId] ?? EmptyModelUsageTotals;
  return ModelUsageStateSchema.parse({
    version: 1,
    total: addTotals(state.total, delta.usage),
    by_agent: {
      ...state.by_agent,
      [delta.agentId]: addTotals(current, delta.usage)
    },
    updated_at: at
  });
}

function addTotals(left: ModelUsageTotals, right: ModelUsageTotals): ModelUsageTotals {
  return ModelUsageTotalsSchema.parse(Object.fromEntries(
    Object.keys(EmptyModelUsageTotals).map((key) => [
      key,
      left[key as keyof ModelUsageTotals] + right[key as keyof ModelUsageTotals]
    ])
  ));
}

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

function detailRecords(value: unknown): Record<string, unknown>[] {
  if (Array.isArray(value)) {
    return value.flatMap((item) => {
      const parsed = record(item);
      return parsed ? [parsed] : [];
    });
  }
  const parsed = record(value);
  return parsed ? [parsed] : [];
}

function detailTotal(
  details: Record<string, unknown>[],
  aliases: string[]
): number {
  return details.reduce((total, detail) => {
    for (const alias of aliases) {
      const value = integer(detail[alias]);
      if (value !== undefined) return total + value;
    }
    return total;
  }, 0);
}

function integer(value: unknown): number | undefined {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0
    ? value
    : undefined;
}

function text(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}
