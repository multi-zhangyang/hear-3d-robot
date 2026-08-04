/**
 * A compatible endpoint may ignore a required tool choice and emit reasoning
 * only. Correction turns stay bounded, but they are split across fresh SDK
 * runs so failed prose from one attempt cannot inflate every later request.
 */
export const CONTEXT_COMPACTOR_MAX_ATTEMPTS = 4;
export const CONTEXT_COMPACTOR_TURNS_PER_ATTEMPT = 2;

export function defaultOutputTokenReserve(contextWindowTokens: number): number {
  return Math.min(
    16_384,
    Math.max(4_096, Math.floor(contextWindowTokens * 0.05))
  );
}

export function configuredOutputTokenLimit(
  ...limits: Array<number | undefined>
): number | undefined {
  const configured = limits.filter((limit): limit is number => limit !== undefined);
  return configured.length > 0 ? Math.min(...configured) : undefined;
}

export function effectiveContextSummaryOutputTokens(
  sourceCompactMaxOutputTokens: number | undefined,
  compactorCompactMaxOutputTokens: number | undefined,
  compactorMaxOutputTokens: number | undefined,
  compactorContextWindowTokens: number
): number {
  return configuredOutputTokenLimit(
    sourceCompactMaxOutputTokens,
    compactorCompactMaxOutputTokens,
    compactorMaxOutputTokens
  ) ?? defaultOutputTokenReserve(compactorContextWindowTokens);
}

export function compactorInputTokenLimit(
  contextWindowTokens: number,
  compactMaxOutputTokens: number
): number {
  return contextWindowTokens
    - compactMaxOutputTokens * CONTEXT_COMPACTOR_TURNS_PER_ATTEMPT;
}
