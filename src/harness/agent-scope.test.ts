import { describe, expect, it } from "vitest";
import {
  agentIdFromModelPayload,
  agentInvocationMarker,
  currentAgentInvocationId,
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
});
