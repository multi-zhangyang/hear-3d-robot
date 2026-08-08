import { describe, expect, it } from "vitest";
import { ModelDecisionStallError } from "./model-telemetry.js";
import { DelegatedDecisionRecovery } from "./delegated-decision-recovery.js";

describe("delegated decision recovery", () => {
  it("adds targeted authority feedback after prose and clears it after a tool result", () => {
    const recovery = new DelegatedDecisionRecovery();
    const agentId = "humanoid-motion-reference";
    const authority = "CURRENT HARNESS AUTHORITY";

    expect(() => recovery.accept(agentId, "普通说明", () => false)).toThrow(
      ModelDecisionStallError
    );
    expect(recovery.attempt(agentId)).toBe(1);
    expect(recovery.invocationInput(agentId, authority)).toContain(
      "SPECIALIST DECISION RECOVERY V1"
    );
    expect(recovery.accept(agentId, "正式回执", () => true)).toBe("正式回执");
    expect(recovery.invocationInput(agentId, authority)).toBe(authority);
  });

  it("tracks a nested model decision stall without treating transport errors as decisions", () => {
    const recovery = new DelegatedDecisionRecovery();
    recovery.recordFailure(
      "humanoid-sentry",
      new ModelDecisionStallError("humanoid-sentry", "missing tool")
    );
    recovery.recordFailure("humanoid-sentry", new TypeError("network interrupted"));

    expect(recovery.attempt("humanoid-sentry")).toBe(1);
  });

});
