import type { Model, ModelRequest } from "@openai/agents";

/**
 * Strengthens the standard tool-decision contract after a model completes a
 * branch without one. The recovery stays attached to the same model facade and
 * expires automatically when the authoritative Harness lifecycle changes.
 */
export class ModelDecisionProtocolRecovery {
  readonly #authorityFingerprint: () => string;
  readonly #requiredAtAuthority = new Map<string, string>();
  readonly #requiredUnsupported = new Set<string>();

  constructor(authorityFingerprint: () => string) {
    this.#authorityFingerprint = authorityFingerprint;
  }

  requireToolDecision(agentId: string): boolean {
    if (this.#requiredUnsupported.has(agentId)) return false;
    const authorityFingerprint = this.#authorityFingerprint();
    const activated = this.#requiredAtAuthority.get(agentId) !== authorityFingerprint;
    this.#requiredAtAuthority.set(agentId, authorityFingerprint);
    return activated;
  }

  requiresToolDecision(agentId: string): boolean {
    if (this.#requiredUnsupported.has(agentId)) return false;
    const requiredAt = this.#requiredAtAuthority.get(agentId);
    if (requiredAt === undefined) return false;
    if (requiredAt === this.#authorityFingerprint()) return true;
    this.#requiredAtAuthority.delete(agentId);
    return false;
  }

  rejectRequiredToolChoice(agentId: string): void {
    this.#requiredUnsupported.add(agentId);
    this.#requiredAtAuthority.delete(agentId);
  }

  clear(): void {
    this.#requiredAtAuthority.clear();
  }
}

export function withModelDecisionProtocolRecovery(
  model: Model,
  agentId: string,
  recovery: ModelDecisionProtocolRecovery,
  onRequiredToolChoiceUnsupported?: (agentId: string) => void | Promise<void>
): Model {
  const prepare = (request: ModelRequest): ModelRequest => (
    recovery.requiresToolDecision(agentId)
      ? {
          ...request,
          modelSettings: {
            ...request.modelSettings,
            toolChoice: "required"
          }
        }
      : request
  );
  const rejectRequired = async (): Promise<void> => {
    recovery.rejectRequiredToolChoice(agentId);
    await onRequiredToolChoiceUnsupported?.(agentId);
  };
  return {
    getResponse: async (request) => {
      const prepared = prepare(request);
      try {
        return await model.getResponse(prepared);
      } catch (error) {
        if (prepared === request || !isUnsupportedRequiredToolChoiceError(error)) throw error;
        await rejectRequired();
        return model.getResponse(request);
      }
    },
    getStreamedResponse: (request) => ({
      async *[Symbol.asyncIterator]() {
        const prepared = prepare(request);
        let yielded = false;
        try {
          for await (const event of model.getStreamedResponse(prepared)) {
            yielded = true;
            yield event;
          }
        } catch (error) {
          if (prepared === request || yielded
            || !isUnsupportedRequiredToolChoiceError(error)) throw error;
          await rejectRequired();
          yield* model.getStreamedResponse(request);
        }
      }
    }),
    ...(model.getRetryAdvice
      ? {
          getRetryAdvice: (input) => model.getRetryAdvice!({
            ...input,
            request: prepare(input.request)
          })
        }
      : {})
  };
}

function isUnsupportedRequiredToolChoiceError(error: unknown): boolean {
  const pending: unknown[] = [error];
  const seen = new Set<object>();
  const messages: string[] = [];
  while (pending.length > 0 && seen.size < 16) {
    const current = pending.shift();
    if (typeof current === "string") {
      messages.push(current);
      continue;
    }
    if (current === null || typeof current !== "object" || seen.has(current)) continue;
    seen.add(current);
    const record = current as Record<string, unknown>;
    if (typeof record.message === "string") messages.push(record.message);
    for (const key of ["cause", "error", "responseBody", "data"]) {
      if (record[key] !== undefined) pending.push(record[key]);
    }
    if (Array.isArray(record.errors)) pending.push(...record.errors);
  }
  const detail = messages.join(" ").toLowerCase();
  const namesToolChoice = detail.includes("tool_choice")
    || detail.includes("tool choice");
  const rejectsSetting = detail.includes("not support")
    || detail.includes("unsupported")
    || detail.includes("invalid_request")
    || detail.includes("invalid request")
    || detail.includes("not allowed");
  return namesToolChoice && rejectsSetting;
}
