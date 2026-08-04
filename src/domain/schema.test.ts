import { describe, expect, it } from "vitest";
import {
  ContextMemoryStateSchema,
  EmptyContextMemoryState,
  GoalSchema
} from "./schema.js";

const legacyEndEffectorGoal = {
  summary: "保持左腕位置",
  predicates: [{
    type: "end_effector_at" as const,
    end_effector: "left_wrist" as const,
    frame: "pelvis" as const,
    target: { x: 0.25, y: 0.3, z: 0.1 },
    tolerance: 0.05,
    stable_frames: 4
  }]
};

describe("end-effector goal schema", () => {
  it("keeps legacy position-only goals unchanged", () => {
    expect(GoalSchema.parse(legacyEndEffectorGoal)).toEqual(legacyEndEffectorGoal);
  });

  it("accepts an optional quaternion orientation as one complete constraint", () => {
    expect(GoalSchema.parse({
      ...legacyEndEffectorGoal,
      predicates: [{
        ...legacyEndEffectorGoal.predicates[0],
        orientation: { x: 0, y: 0.2, z: 0, w: 0.98 },
        orientation_tolerance_rad: 0.12
      }]
    }).predicates[0]).toMatchObject({
      orientation: { x: 0, y: 0.2, z: 0, w: 0.98 },
      orientation_tolerance_rad: 0.12
    });
  });

  it("rejects partial and zero-magnitude orientation constraints", () => {
    expect(GoalSchema.safeParse({
      ...legacyEndEffectorGoal,
      predicates: [{
        ...legacyEndEffectorGoal.predicates[0],
        orientation: { x: 0, y: 0, z: 0, w: 1 }
      }]
    }).success).toBe(false);
    expect(GoalSchema.safeParse({
      ...legacyEndEffectorGoal,
      predicates: [{
        ...legacyEndEffectorGoal.predicates[0],
        orientation: { x: 0, y: 0, z: 0, w: 0 },
        orientation_tolerance_rad: 0.1
      }]
    }).success).toBe(false);
  });
});

describe("object grasp Goal schema", () => {
  it("accepts a hand choice without exposing the authority contract hash", () => {
    const goal = GoalSchema.parse({
      summary: "抓住箱体",
      predicates: [{
        type: "object_grasped",
        object_id: "crate",
        hand: "either"
      }]
    });
    expect(goal.predicates[0]).toEqual({
      type: "object_grasped",
      object_id: "crate",
      hand: "either"
    });
    expect(GoalSchema.safeParse({
      summary: "不得修改抓取权威契约",
      predicates: [{
        type: "object_grasped",
        object_id: "crate",
        hand: "left",
        grasp_contract_sha256: "a".repeat(64)
      }]
    }).success).toBe(false);
  });

  it("rejects empty objects and unsupported hand names", () => {
    expect(GoalSchema.safeParse({
      summary: "无效抓取",
      predicates: [{ type: "object_grasped", object_id: "", hand: "left" }]
    }).success).toBe(false);
    expect(GoalSchema.safeParse({
      summary: "无效抓取",
      predicates: [{ type: "object_grasped", object_id: "crate", hand: "both" }]
    }).success).toBe(false);
  });
});

describe("object placed Goal schema", () => {
  it("accepts only object, zone and geometric tolerance from the caller", () => {
    const predicate = GoalSchema.parse({
      summary: "将箱体稳放到区域",
      predicates: [{
        type: "object_placed",
        object_id: "crate",
        zone_id: "arrival",
        tolerance: 0.04
      }]
    }).predicates[0];

    expect(predicate).toEqual({
      type: "object_placed",
      object_id: "crate",
      zone_id: "arrival",
      tolerance: 0.04
    });
    expect(GoalSchema.safeParse({
      summary: "不得修改物理放置阈值",
      predicates: [{
        type: "object_placed",
        object_id: "crate",
        zone_id: "arrival",
        tolerance: 0.04,
        maximum_linear_speed_mps: 99,
        minimum_support_normal_force_n: 0
      }]
    }).success).toBe(false);
  });
});

describe("context scope budget schema", () => {
  it("restores an old v1 scope without inventing per-agent budgets", () => {
    const legacy = contextMemoryWithScope();
    const parsed = ContextMemoryStateSchema.parse(legacy);

    expect(parsed.scopes.worker).not.toHaveProperty("context_window_tokens");
    expect(parsed.scopes.worker).not.toHaveProperty("compact_trigger_tokens");
    expect(parsed.scopes.worker).not.toHaveProperty("compact_recent_model_turns");
    expect(parsed.scopes.worker).not.toHaveProperty("compact_max_output_tokens");
  });

  it("accepts only a complete per-agent budget envelope", () => {
    const legacy = contextMemoryWithScope();
    expect(ContextMemoryStateSchema.parse({
      ...legacy,
      scopes: {
        worker: {
          ...legacy.scopes.worker,
          context_window_tokens: 32_768,
          compact_trigger_tokens: 8_000,
          compact_recent_model_turns: 2,
          compact_max_output_tokens: 768
        }
      }
    }).scopes.worker).toMatchObject({
      context_window_tokens: 32_768,
      compact_trigger_tokens: 8_000,
      compact_recent_model_turns: 2,
      compact_max_output_tokens: 768
    });
    expect(ContextMemoryStateSchema.safeParse({
      ...legacy,
      scopes: {
        worker: {
          ...legacy.scopes.worker,
          compact_trigger_tokens: 8_000
        }
      }
    }).success).toBe(false);
  });
});

function contextMemoryWithScope() {
  return {
    ...structuredClone(EmptyContextMemoryState),
    active_scope_id: "worker",
    scopes: {
      worker: {
        scope_id: "worker",
        agent_id: "worker",
        agent_name: "工作智能体",
        raw_item_count: 0,
        raw_chain_hash: null,
        compacted_item_count: 0,
        retained_item_count: 0,
        retained_chain_hash: null,
        active_estimated_tokens: 0,
        compaction_count: 0,
        summary: null,
        summary_origin: null,
        summary_world_revision: null,
        last_compacted_at: null
      }
    }
  };
}
