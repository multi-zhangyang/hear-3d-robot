import { describe, expect, it } from "vitest";
import {
  autonomyContentSha256,
  autonomyPlanningBehaviorSha256,
  normalizeAutonomyPlanningInput,
  quantizeAutonomyVec3
} from "./autonomy-signature.js";

describe("autonomy behavior signatures", () => {
  it("ignores textual and identity noise but retains continuous physical choices", () => {
    const first = {
      id: "model-plan-a",
      intent: "向前探索",
      option_id: "option-a",
      target: { x: 2, y: 0, z: 4 },
      contact_constraints: [{ object_id: "crate-a", body: "left_hand" }]
    };
    const renamed = {
      ...first,
      id: "model-plan-b",
      intent: "用另一种文本描述",
      option_id: "option-b"
    };
    expect(normalizeAutonomyPlanningInput(renamed)).toEqual(
      normalizeAutonomyPlanningInput(first)
    );
    expect(autonomyPlanningBehaviorSha256("plan_humanoid_navigation", renamed))
      .toBe(autonomyPlanningBehaviorSha256("plan_humanoid_navigation", first));
    expect(autonomyPlanningBehaviorSha256("plan_humanoid_navigation", {
      ...renamed,
      target: { x: 3, y: 0, z: 4 }
    })).not.toBe(autonomyPlanningBehaviorSha256(
      "plan_humanoid_navigation",
      first
    ));
  });

  it("keeps model preference order and semantic object identities", () => {
    const leftFirst = {
      candidates: [{ object_id: "left" }, { object_id: "right" }]
    };
    const rightFirst = {
      candidates: [{ object_id: "right" }, { object_id: "left" }]
    };
    expect(autonomyContentSha256(leftFirst)).not.toBe(
      autonomyContentSha256(rightFirst)
    );
  });

  it("quantizes physical endpoints to millimetres", () => {
    expect(quantizeAutonomyVec3({ x: 1.0004, y: 0, z: 2.1236 })).toEqual({
      x: 1,
      y: 0,
      z: 2.124
    });
    expect(quantizeAutonomyVec3(null)).toBeNull();
  });
});
