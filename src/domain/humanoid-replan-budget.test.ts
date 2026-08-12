import { describe, expect, it } from "vitest";
import {
  beginHumanoidReplanModelCall,
  createHumanoidReplanBudget,
  finishHumanoidReplanModelCall,
  humanoidReplanBudgetAuthority,
  restoreHumanoidReplanModelCall
} from "./humanoid-replan-budget.js";

describe("humanoid replan budget", () => {
  it("bounds compact replans, specialist calls and one Goal escalation", () => {
    let budget = createHumanoidReplanBudget();
    const first = beginHumanoidReplanModelCall(budget, {
      modelCallId: "00000000-0000-4000-8000-000000000001",
      agentId: "humanoid-coordinator",
      role: "coordinator",
      at: "2026-08-11T00:00:00.000Z"
    });
    budget = first.budget;
    expect(first.call).toMatchObject({
      recovery_tier: "compact_replan",
      role: "replan_decision"
    });

    const specialist = beginHumanoidReplanModelCall(budget, {
      modelCallId: "00000000-0000-4000-8000-000000000002",
      agentId: "humanoid-motion-reference",
      role: "motion",
      at: "2026-08-11T00:00:01.000Z"
    });
    budget = finishHumanoidReplanModelCall(specialist.budget, {
      modelCallId: specialist.call.model_call_id,
      status: "completed",
      at: "2026-08-11T00:00:32.000Z"
    }).budget;
    expect(budget.model_calls[1]).toMatchObject({
      latency_ms: 31_000,
      slo_violated: true
    });

    for (let index = 3; index <= 4; index += 1) {
      budget = beginHumanoidReplanModelCall(budget, {
        modelCallId: `00000000-0000-4000-8000-${String(index).padStart(12, "0")}`,
        agentId: "humanoid-coordinator",
        role: "coordinator",
        at: `2026-08-11T00:00:0${index}.000Z`
      }).budget;
    }
    const escalation = beginHumanoidReplanModelCall(budget, {
      modelCallId: "00000000-0000-4000-8000-000000000005",
      agentId: "humanoid-coordinator",
      role: "coordinator",
      at: "2026-08-11T00:00:04.000Z"
    });
    budget = escalation.budget;
    expect(escalation.call.recovery_tier).toBe("goal_re_evaluation");
    expect(() => beginHumanoidReplanModelCall(budget, {
      modelCallId: "00000000-0000-4000-8000-000000000006",
      agentId: "humanoid-coordinator",
      role: "coordinator",
      at: "2026-08-11T00:00:05.000Z"
    })).toThrow("exhausted after Goal re-evaluation");
  });

  it("escalates on the recovery deadline without spending a stale compact replan", () => {
    const initial = createHumanoidReplanBudget();
    const first = beginHumanoidReplanModelCall(initial, {
      modelCallId: "00000000-0000-4000-8000-000000000011",
      agentId: "humanoid-coordinator",
      role: "coordinator",
      at: "2026-08-11T00:00:00.000Z"
    });
    const escalated = beginHumanoidReplanModelCall(first.budget, {
      modelCallId: "00000000-0000-4000-8000-000000000012",
      agentId: "humanoid-coordinator",
      role: "coordinator",
      at: "2026-08-11T00:02:00.001Z"
    });
    expect(escalated.budget.compact_replans_started).toBe(1);
    expect(escalated.call.recovery_tier).toBe("goal_re_evaluation");
    expect(humanoidReplanBudgetAuthority(
      escalated.budget,
      "2026-08-11T00:02:00.001Z"
    )).toMatchObject({
      status: "goal_re_evaluation_in_progress",
      recovery_deadline_exceeded: true,
      compact_replans: { used: 1, remaining: 2, available: false }
    });
  });

  it("restores a journaled replan call that was not checkpointed before a crash", () => {
    const initial = createHumanoidReplanBudget();
    const started = beginHumanoidReplanModelCall(initial, {
      modelCallId: "00000000-0000-4000-8000-000000000021",
      agentId: "humanoid-coordinator",
      role: "coordinator",
      at: "2026-08-11T00:00:00.000Z"
    });

    const restored = restoreHumanoidReplanModelCall(initial, started.call);

    expect(restored).toEqual(started.budget);
    expect(restoreHumanoidReplanModelCall(restored, started.call)).toEqual(restored);
  });
});
