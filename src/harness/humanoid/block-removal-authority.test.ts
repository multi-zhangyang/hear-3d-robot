import { describe, expect, it } from "vitest";
import { createActiveAutonomousCycle, autonomousCycleRef } from "../../domain/autonomous-cycle.js";
import { ScenarioSchema } from "../../domain/schema.js";
import { createScenarioChunkDeltaState } from "../../domain/scenario-chunk-delta-schema.js";
import type { HumanoidMotionOptionContract } from "../../world/humanoid/motion-option.js";
import type { HumanoidWorldSnapshot } from "../../world/humanoid/world.js";
import {
  BlockRemovalAuthorityError,
  prepareAuthorizedBlockRemoval
} from "./block-removal-authority.js";
import type { HumanoidActionReceipt } from "./runtime.js";

const scenario = ScenarioSchema.parse({
  title: "Block authority fixture",
  seed: 97,
  bounds: { width: 20, depth: 20 },
  visibility_radius: 8,
  robot: { x: 2, z: 2, yaw: 0 },
  obstacles: [{
    id: "block-a",
    center: { x: 3, y: 0.5, z: 3 },
    size: { x: 1, y: 1, z: 1 }
  }],
  objects: [],
  zones: [],
  default_goal: {
    summary: "Reach the block.",
    predicates: [{
      type: "robot_at",
      target: { x: 3, y: 0, z: 3 },
      tolerance: 0.3
    }]
  }
});
const cycle = autonomousCycleRef(createActiveAutonomousCycle({
  cycleIndex: 2,
  goalEpochId: `goal-epoch:${"a".repeat(64)}`,
  worldFrame: 80,
  worldRevision: 80,
  cycleUuid: "11111111-1111-4111-8111-111111111111",
  startedAt: "2026-08-04T00:00:00.000Z"
}));

describe("block-removal physical authority", () => {
  it("binds a removal transaction to the required stable contact predicate", () => {
    const transaction = prepareAuthorizedBlockRemoval(authorityInput());

    expect(transaction).toMatchObject({
      transaction_id: "remove-call",
      solid_id: "block-a",
      execution_transaction_id: "execute-call",
      planning_transaction_id: "plan-call",
      source_world_frame: 100,
      source_world_revision: 100,
      contact_evidence: {
        predicate_type: "hand_contact_solid",
        surface: "left_hand_palm_link",
        planned_stable_frames: 10,
        observed_stable_frames: 10,
        observed_maximum_normal_force_n: 9
      }
    });
  });

  it("rejects a contact predicate that is optional in a terminal any branch", () => {
    const input = authorityInput();
    const plan = input.committedActions["plan-call"]!;
    const contract = optionContract();
    contract.phases = {
      precondition: null,
      during: null,
      terminal: {
        condition: {
          op: "any",
          conditions: [
            { op: "predicate", predicate_index: 0 },
            { op: "predicate", predicate_index: 1 }
          ]
        }
      }
    };
    contract.predicates.push({
      type: "root_near_point",
      target: { x: 2, y: 0.8, z: 2 },
      tolerance_m: 0.2
    });
    plan.detail = {
      ...plan.detail as object,
      termination: contract,
      option: { contract }
    };

    expectAuthorityCode(input, "block_removal_contact_contract_missing");
  });

  it("rejects model-lowered stability and later physical actuation", () => {
    const brief = authorityInput();
    const plan = brief.committedActions["plan-call"]!;
    const contract = optionContract();
    contract.stable_steps = 2;
    plan.detail = {
      ...plan.detail as object,
      termination: contract,
      option: { contract }
    };
    const execution = brief.committedActions["execute-call"]!;
    execution.detail = executionDetail(contract, 2, 9);
    expectAuthorityCode(brief, "block_removal_contact_too_brief");

    const superseded = authorityInput();
    superseded.committedActions["later-navigation"] = receipt({
      transactionId: "later-navigation",
      agentId: "humanoid-executor",
      action: "execute_humanoid_navigation",
      input: { planning_transaction_id: "another-plan" },
      accepted: true,
      code: "navigation_completed",
      worldBeforeRevision: 100,
      worldAfterRevision: 104,
      frameCount: 4,
      detail: {}
    });
    expectAuthorityCode(superseded, "block_removal_execution_superseded");
  });
});

function authorityInput() {
  const contract = optionContract();
  const plan = receipt({
    transactionId: "plan-call",
    agentId: "humanoid-motion-reference",
    action: "plan_whole_body_motion_candidates",
    input: {},
    accepted: true,
    code: "whole_body_candidates_validated",
    worldBeforeRevision: 80,
    worldAfterRevision: 80,
    frameCount: 0,
    detail: {
      plan_id: "candidate-plan",
      termination: contract,
      option: { contract }
    }
  });
  const execute = receipt({
    transactionId: "execute-call",
    agentId: "humanoid-executor",
    action: "execute_whole_body_motion",
    input: { planning_transaction_id: "plan-call" },
    accepted: true,
    code: "motion_option_succeeded",
    worldBeforeRevision: 90,
    worldAfterRevision: 100,
    frameCount: 10,
    detail: executionDetail(contract, 10, 9)
  });
  return {
    scenario,
    chunks: createScenarioChunkDeltaState(scenario),
    currentWorld: { frame: 100, worldRevision: 100 } as HumanoidWorldSnapshot,
    activeCycle: cycle,
    removalTransactionId: "remove-call",
    agentId: "humanoid-executor",
    solidId: "block-a",
    executionTransactionId: "execute-call",
    committedActions: {
      "plan-call": plan,
      "execute-call": execute
    } as Record<string, HumanoidActionReceipt>
  };
}

function optionContract(): HumanoidMotionOptionContract {
  return {
    option_id: "touch-block-a",
    predicates: [{
      type: "hand_contact_solid",
      hand_surface: "left_hand_palm_link",
      solid_id: "block-a",
      minimum_normal_force: 7
    }],
    stable_steps: 10,
    phases: null
  };
}

function executionDetail(
  contract: HumanoidMotionOptionContract,
  stableFrames: number,
  force: number
) {
  return {
    planning_transaction_id: "plan-call",
    planning_action: "plan_whole_body_motion_candidates",
    plan_id: "candidate-plan",
    result: {
      option: {
        option_id: contract.option_id,
        status: "succeeded",
        termination_reason: "physical_success",
        evidence: {
          monitor: { terminalStableSteps: stableFrames },
          predicates: [{
            predicateIndex: 0,
            type: "hand_contact_solid",
            status: "satisfied",
            handSurface: "left_hand_palm_link",
            solidId: "block-a",
            solidObservable: true,
            maximumNormalForce: force,
            minimumNormalForce: 7
          }]
        }
      }
    }
  };
}

function receipt(input: Omit<HumanoidActionReceipt, "fingerprint" | "channels" | "committedAt" | "cycle">): HumanoidActionReceipt {
  return {
    ...input,
    cycle,
    fingerprint: `fingerprint:${input.transactionId}`,
    channels: [],
    committedAt: "2026-08-04T00:00:01.000Z"
  };
}

function expectAuthorityCode(
  input: ReturnType<typeof authorityInput>,
  code: BlockRemovalAuthorityError["code"]
): void {
  try {
    prepareAuthorizedBlockRemoval(input);
    throw new Error("Expected block-removal authority rejection");
  } catch (error) {
    expect(error).toBeInstanceOf(BlockRemovalAuthorityError);
    expect((error as BlockRemovalAuthorityError).code).toBe(code);
  }
}
