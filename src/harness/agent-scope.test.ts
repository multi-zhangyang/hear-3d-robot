import { describe, expect, it } from "vitest";
import {
  agentIdFromModelPayload,
  agentInvocationMarker,
  currentAgentInvocationId,
  currentAgentInvocationIsRecovery,
  withAgentInvocation
} from "./agent-scope.js";

describe("agent invocation scope", () => {
  it("keeps overlapping hierarchy model runs attributed to their own node", async () => {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    let firstReady!: () => void;
    const ready = new Promise<void>((resolve) => {
      firstReady = resolve;
    });

    const first = withAgentInvocation("agent_a", async () => {
      firstReady();
      await gate;
      return {
        scoped: currentAgentInvocationId(),
        resolved: agentIdFromModelPayload([agentInvocationMarker("agent_b")], "root")
      };
    });
    await ready;
    const second = withAgentInvocation("agent_b", async () => {
      await Promise.resolve();
      return {
        scoped: currentAgentInvocationId(),
        resolved: agentIdFromModelPayload([], "root")
      };
    });
    release();

    await expect(Promise.all([first, second])).resolves.toEqual([
      { scoped: "agent_a", resolved: "agent_a" },
      { scoped: "agent_b", resolved: "agent_b" }
    ]);
    expect(currentAgentInvocationId()).toBeUndefined();
  });

  it("recovers a serialized marker without accepting an invalid node id", () => {
    expect(agentIdFromModelPayload({ input: agentInvocationMarker("worker_2") }, "root"))
      .toBe("worker_2");
    expect(() => agentInvocationMarker("worker:2")).toThrow("Invalid hierarchy node identifier");
  });

  it("keeps nested recovery identity scoped to the resumed agent invocation", async () => {
    await withAgentInvocation("supervisor", async () => {
      expect(currentAgentInvocationIsRecovery()).toBe(false);
      await withAgentInvocation("resumed_child", async () => {
        expect(currentAgentInvocationIsRecovery()).toBe(true);
      }, true);
      expect(currentAgentInvocationIsRecovery()).toBe(false);
    });
    expect(currentAgentInvocationIsRecovery()).toBe(false);
  });

  it("scans a large serialized Session payload without expanding it into call arguments", () => {
    const input = Array.from({ length: 150_000 }, (_, index) =>
      index === 0 ? agentInvocationMarker("worker_large") : `history-${index}`
    );

    expect(agentIdFromModelPayload(input, "root")).toBe("worker_large");
  });
});
