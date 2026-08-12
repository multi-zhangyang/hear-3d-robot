import { describe, expect, it } from "vitest";
import type { AutonomousCycleRef } from "../../domain/autonomous-cycle.js";
import { createScenarioBlockRemovalTransaction } from "../../domain/scenario-block-removal.js";
import { createScenarioChunkDeltaState } from "../../domain/scenario-chunk-delta-schema.js";
import { ScenarioSchema } from "../../domain/schema.js";
import {
  resolveHumanoidCycleCompletionReadiness,
  validateHumanoidCycleCausalEvidence,
  type HumanoidCycleCausalEvidenceInput
} from "./cycle-causal-evidence.js";
import type { HumanoidActionReceipt } from "./runtime.js";

const activeCycle: AutonomousCycleRef = {
  cycle_id: "autonomous-cycle:00000000-0000-4000-8000-000000000001",
  cycle_index: 3,
  goal_epoch_id: `goal-epoch:${"a".repeat(64)}`
};

const scenario = ScenarioSchema.parse({
  title: "Cycle 因果验证场",
  seed: 9,
  bounds: { width: 8, depth: 8 },
  visibility_radius: 6,
  robot: { x: 1, z: 1, yaw: 0 },
  obstacles: [{
    id: "block-a",
    center: { x: 2, y: 0.5, z: 2 },
    size: { x: 1, y: 1, z: 1 }
  }],
  objects: [],
  zones: [],
  default_goal: {
    summary: "拆除方块",
    predicates: [{ type: "block_removed", block_id: "block-a" }]
  }
});

describe("humanoid Cycle causal evidence", () => {
  it("returns the latest physical execution and every causally ordered world mutation", () => {
    const input = validInput();
    const result = validateHumanoidCycleCausalEvidence(input);

    expect(result.execution.transactionId).toBe("execute-a");
    expect(result.worldMutations.map(({ transactionId }) => transactionId))
      .toEqual(["remove-a"]);
    result.execution.code = "mutated-outside-authority";
    expect(input.committedActions["execute-a"]?.code).toBe("motion_option_succeeded");
  });

  it("uses the durable commit sequence after checkpoint key reordering", () => {
    const input = validInput();
    input.committedActions["execute-a"]!.commitSequence = 1;
    input.committedActions["remove-a"]!.commitSequence = 2;
    input.committedActions["observe-after"]!.commitSequence = 3;
    input.committedActions = {
      "observe-after": input.committedActions["observe-after"]!,
      "remove-a": input.committedActions["remove-a"]!,
      "execute-a": input.committedActions["execute-a"]!
    };

    expect(validateHumanoidCycleCausalEvidence(input)).toMatchObject({
      execution: { transactionId: "execute-a" },
      worldMutations: [{ transactionId: "remove-a" }]
    });
    const { evidenceTransactionIds: _evidenceTransactionIds, ...state } = input;
    expect(resolveHumanoidCycleCompletionReadiness(state)).toMatchObject({
      status: "ready",
      observed_after_execution: true
    });
  });

  it("publishes the exact completion evidence and detects post-execution observation", () => {
    const input = validInput();
    const { evidenceTransactionIds: _evidenceTransactionIds, ...state } = input;
    const { "observe-after": _observation, ...unobserved } = state.committedActions;
    state.committedActions = unobserved;
    expect(resolveHumanoidCycleCompletionReadiness(state)).toMatchObject({
      status: "ready",
      evidence_transaction_ids: ["execute-a", "remove-a"],
      execution_transaction_id: "execute-a",
      observed_after_execution: false,
      reason: null
    });

    state.committedActions = {
      ...state.committedActions,
      "observe-after": receipt({
        transactionId: "observe-after",
        action: "observe_humanoid",
        accepted: true,
        code: "humanoid_observed",
        frameCount: 0,
        worldBeforeRevision: 6,
        worldAfterRevision: 6
      })
    };
    expect(resolveHumanoidCycleCompletionReadiness(state)).toMatchObject({
      status: "ready",
      observed_after_execution: true
    });
  });

  it("requires the completion to include every committed world mutation", () => {
    const input = validInput();
    input.evidenceTransactionIds = ["execute-a"];

    expect(() => validateHumanoidCycleCausalEvidence(input)).toThrow(
      "omits world mutation evidence"
    );
  });

  it("rejects a mutation recorded before the physical execution", () => {
    const input = validInput();
    input.committedActions = {
      "remove-a": input.committedActions["remove-a"]!,
      "execute-a": input.committedActions["execute-a"]!
    };

    expect(() => validateHumanoidCycleCausalEvidence(input)).toThrow(
      "world mutation precedes its physical execution"
    );
  });

  it("rejects a completed Cycle after another physical actuation", () => {
    const input = validInput();
    input.committedActions = {
      ...input.committedActions,
      "execute-later": receipt({
        transactionId: "execute-later",
        action: "execute_whole_body_motion",
        accepted: false,
        code: "motion_failed",
        frameCount: 2,
        worldBeforeRevision: 5,
        worldAfterRevision: 7
      })
    };

    expect(() => validateHumanoidCycleCausalEvidence(input)).toThrow(
      "superseded by later physical actuation"
    );
  });

  it("rejects an accepted plan that has not been consumed", () => {
    const input = validInput();
    input.committedActions = {
      ...input.committedActions,
      "plan-later": receipt({
        transactionId: "plan-later",
        action: "plan_humanoid_navigation",
        accepted: true,
        code: "navigation_planned",
        frameCount: 0,
        worldBeforeRevision: 5,
        worldAfterRevision: 5
      })
    };

    expect(() => validateHumanoidCycleCausalEvidence(input)).toThrow(
      "unconsumed accepted plan"
    );
  });

  it("requires a Sentry observation after execution and every causal mutation", () => {
    const missing = validInput();
    const { "observe-after": _observation, ...unobserved } = missing.committedActions;
    missing.committedActions = unobserved;
    expect(() => validateHumanoidCycleCausalEvidence(missing)).toThrow(
      "requires an accepted Sentry observation"
    );

    const stale = validInput();
    stale.committedActions = {
      "execute-a": stale.committedActions["execute-a"]!,
      "observe-after": stale.committedActions["observe-after"]!,
      "remove-a": stale.committedActions["remove-a"]!
    };
    expect(() => validateHumanoidCycleCausalEvidence(stale)).toThrow(
      "requires an accepted Sentry observation"
    );
    expect(resolveHumanoidCycleCompletionReadiness({
      committedActions: stale.committedActions,
      previousCycle: stale.previousCycle,
      activeCycle: stale.activeCycle,
      currentWorld: stale.currentWorld
    })).toMatchObject({
      status: "ready",
      observed_after_execution: false
    });

    const motionObservation = validInput();
    motionObservation.committedActions["observe-after"] = {
      ...motionObservation.committedActions["observe-after"]!,
      agentId: "humanoid-motion-reference"
    };
    expect(() => validateHumanoidCycleCausalEvidence(motionObservation)).toThrow(
      "requires an accepted Sentry observation"
    );
    expect(resolveHumanoidCycleCompletionReadiness({
      committedActions: motionObservation.committedActions,
      previousCycle: motionObservation.previousCycle,
      activeCycle: motionObservation.activeCycle,
      currentWorld: motionObservation.currentWorld
    })).toMatchObject({ observed_after_execution: false });
  });

  it("accepts only the exact unique evidence chain in causal order", () => {
    const unrelated = validInput();
    unrelated.evidenceTransactionIds = ["execute-a", "observe-after", "remove-a"];
    expect(() => validateHumanoidCycleCausalEvidence(unrelated)).toThrow(
      "unrelated evidence"
    );

    const reversed = validInput();
    reversed.evidenceTransactionIds = ["remove-a", "execute-a"];
    expect(() => validateHumanoidCycleCausalEvidence(reversed)).toThrow(
      "canonical causal order"
    );

    const duplicate = validInput();
    duplicate.evidenceTransactionIds = ["execute-a", "remove-a", "remove-a"];
    expect(() => validateHumanoidCycleCausalEvidence(duplicate)).toThrow(
      "must be unique"
    );
  });

  it("does not complete a Cycle from consumed evidence or a fallen world", () => {
    const consumed = validInput();
    consumed.previousCycle = { evidence_transaction_ids: ["execute-a"] };
    expect(() => validateHumanoidCycleCausalEvidence(consumed)).toThrow("already consumed");

    const fallen = validInput();
    fallen.currentWorld.robot.fallen = true;
    expect(() => validateHumanoidCycleCausalEvidence(fallen)).toThrow("has fallen");
  });
});

function validInput(): HumanoidCycleCausalEvidenceInput {
  const execution = receipt({
    transactionId: "execute-a",
    action: "execute_whole_body_motion",
    accepted: true,
    code: "motion_option_succeeded",
    frameCount: 20,
    worldBeforeRevision: 1,
    worldAfterRevision: 5
  });
  const transaction = createScenarioBlockRemovalTransaction({
    scenario,
    chunks: createScenarioChunkDeltaState(scenario),
    transactionId: "remove-a",
    solidId: "block-a",
    executionTransactionId: execution.transactionId,
    planningTransactionId: "plan-a",
    sourceWorldFrame: 20,
    sourceWorldRevision: execution.worldAfterRevision,
    contactEvidence: {
      predicate_index: 0,
      predicate_type: "body_contact_solid",
      surface_kind: "body",
      surface: "left_wrist_yaw_link",
      planned_stable_frames: 8,
      observed_stable_frames: 8,
      planned_minimum_normal_force_n: 5,
      observed_maximum_normal_force_n: 5
    }
  });
  const removal = receipt({
    transactionId: transaction.transaction_id,
    action: "remove_world_block",
    accepted: true,
    code: "world_block_removal_authorized",
    frameCount: 0,
    worldBeforeRevision: 5,
    worldAfterRevision: 6,
    detail: { removal_transaction: transaction }
  });
  const observation = receipt({
    transactionId: "observe-after",
    action: "observe_humanoid",
    accepted: true,
    code: "humanoid_observed",
    frameCount: 0,
    worldBeforeRevision: 6,
    worldAfterRevision: 6
  });
  return {
    evidenceTransactionIds: [execution.transactionId, removal.transactionId],
    committedActions: {
      [execution.transactionId]: execution,
      [removal.transactionId]: removal,
      [observation.transactionId]: observation
    },
    previousCycle: null,
    activeCycle,
    currentWorld: { worldRevision: 6, robot: { fallen: false } }
  };
}

function receipt(input: {
  transactionId: string;
  action: HumanoidActionReceipt["action"];
  accepted: boolean;
  code: string;
  frameCount: number;
  worldBeforeRevision: number;
  worldAfterRevision: number;
  detail?: HumanoidActionReceipt["detail"];
}): HumanoidActionReceipt {
  return {
    transactionId: input.transactionId,
    agentId: input.action === "observe_humanoid"
      ? "humanoid-sentry"
      : input.action.startsWith("plan_")
        ? "humanoid-motion-reference"
        : "humanoid-executor",
    cycle: activeCycle,
    action: input.action,
    input: {},
    fingerprint: `fingerprint:${input.transactionId}`,
    accepted: input.accepted,
    code: input.code,
    worldBeforeRevision: input.worldBeforeRevision,
    worldAfterRevision: input.worldAfterRevision,
    frameCount: input.frameCount,
    channels: [],
    detail: input.detail ?? {},
    committedAt: "2026-08-04T00:00:00.000Z"
  };
}
