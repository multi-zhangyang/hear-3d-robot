import { describe, expect, it } from "vitest";
import {
  authorityForToolCall,
  rebuildModelCallAuthorities
} from "./model-call-authority.js";

const CALL_ID = "00000000-0000-4000-8000-000000000010";
const CYCLE = {
  cycle_id: "autonomous-cycle:00000000-0000-4000-8000-000000000001",
  cycle_index: 1,
  goal_epoch_id: `goal-epoch:${"c".repeat(64)}`
} as const;

describe("model call authority", () => {
  it("binds a real request lifecycle to its response and SDK tool call", () => {
    const authorities = rebuildModelCallAuthorities([
      started(),
      {
        version: 1,
        lifecycle: "completed",
        model_call_id: CALL_ID,
        agent_id: "humanoid-goal-manager",
        response_id: "response-10",
        response_output_sha256: "a".repeat(64),
        tool_calls: [{
          tool_call_id: "tool-10",
          tool_name: "submit_goal_candidates",
          arguments_sha256: "b".repeat(64)
        }],
        at: "2026-08-03T00:00:01.000Z"
      }
    ]);

    expect(authorityForToolCall(
      authorities,
      "humanoid-goal-manager",
      "tool-10",
      "submit_goal_candidates"
    )).toMatchObject({
      model_call_id: CALL_ID,
      response_id: "response-10",
      started_at: "2026-08-03T00:00:00.000Z"
    });
    expect(authorityForToolCall(
      authorities,
      "humanoid-goal-manager",
      "invented",
      "submit_goal_candidates"
    )).toBeUndefined();
  });

  it("rejects a response, failure or duplicate terminal record without one exact start", () => {
    expect(() => rebuildModelCallAuthorities([{
      version: 1,
      lifecycle: "completed",
      model_call_id: CALL_ID,
      agent_id: "humanoid-goal-manager",
      response_id: "response-10",
      response_output_sha256: "a".repeat(64),
      tool_calls: [],
      at: "2026-08-03T00:00:01.000Z"
    }])).toThrow("no matching start");
    expect(() => rebuildModelCallAuthorities([
      started(),
      {
        version: 1,
        lifecycle: "failed",
        model_call_id: CALL_ID,
        agent_id: "another-agent",
        at: "2026-08-03T00:00:01.000Z"
      }
    ])).toThrow("no matching start");
  });

  it("rejects a terminal model record rebound to another autonomous cycle", () => {
    expect(() => rebuildModelCallAuthorities([
      { ...started(), cycle: CYCLE },
      {
        version: 1,
        lifecycle: "completed",
        model_call_id: CALL_ID,
        agent_id: "humanoid-goal-manager",
        response_id: "response-10",
        response_output_sha256: "a".repeat(64),
        tool_calls: [],
        cycle: { ...CYCLE, cycle_index: 2 },
        at: "2026-08-03T00:00:01.000Z"
      }
    ])).toThrow("cycle identity changed");
  });
});

function started() {
  return {
    version: 1 as const,
    lifecycle: "started" as const,
    model_call_id: CALL_ID,
    agent_id: "humanoid-goal-manager",
    at: "2026-08-03T00:00:00.000Z"
  };
}
