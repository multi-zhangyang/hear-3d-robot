/**
 * A compatible endpoint may ignore a required tool choice and emit reasoning
 * only. Correction turns stay bounded, but they are split across fresh SDK
 * runs so failed prose from one attempt cannot inflate every later request.
 */
export const CONTEXT_COMPACTOR_MAX_ATTEMPTS = 4;
export const CONTEXT_COMPACTOR_TURNS_PER_ATTEMPT = 2;
export const CONTEXT_COMPACTOR_MAX_TURNS =
  CONTEXT_COMPACTOR_MAX_ATTEMPTS * CONTEXT_COMPACTOR_TURNS_PER_ATTEMPT;

export function compactorInputTokenLimit(
  contextWindowTokens: number,
  compactMaxOutputTokens: number
): number {
  return contextWindowTokens
    - compactMaxOutputTokens * CONTEXT_COMPACTOR_MAX_TURNS;
}
