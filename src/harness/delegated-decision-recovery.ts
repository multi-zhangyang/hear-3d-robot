import {
  ModelDecisionStallError,
  modelDecisionStallFrom
} from "./model-telemetry.js";

export class DelegatedDecisionRecovery {
  readonly #attempts = new Map<string, number>();

  invocationInput(agentId: string, authorityInput: string): string {
    const attempt = this.#attempts.get(agentId) ?? 0;
    if (attempt === 0) return authorityInput;
    return [
      authorityInput,
      "",
      "SPECIALIST DECISION RECOVERY V1",
      `恢复轮次：${attempt}`,
      "上一次该专职节点响应没有返回终止合约要求的正式工具结果。",
      "根据末尾 CURRENT HARNESS AUTHORITY 立即调用当前可用的正式工具；普通文字、工具名说明或参数说明都不是结果。",
      "不得从上一条文字中恢复、猜测或代填动作，也不得复用旧 frame、revision 或 transactionId。"
    ].join("\n");
  }

  accept<T>(
    agentId: string,
    output: T,
    isFormalDecision: (output: T) => boolean
  ): T {
    if (isFormalDecision(output)) {
      this.#attempts.delete(agentId);
      return output;
    }
    this.#increment(agentId);
    throw new ModelDecisionStallError(
      agentId,
      `${agentId} did not return its required terminal tool result`
    );
  }

  recordFailure(agentId: string, error: unknown): void {
    if (modelDecisionStallFrom(error)) this.#increment(agentId);
  }

  attempt(agentId: string): number {
    return this.#attempts.get(agentId) ?? 0;
  }

  #increment(agentId: string): void {
    this.#attempts.set(agentId, (this.#attempts.get(agentId) ?? 0) + 1);
  }
}
