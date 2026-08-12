import { describe, expect, it } from "vitest";
import {
  createHumanoidGroundingObligation,
  createHumanoidGroundingReceipt
} from "../domain/humanoid-grounding.js";
import {
  beginHumanoidReplanModelCall,
  createHumanoidReplanBudget,
  finishHumanoidReplanModelCall,
  humanoidReplanBudgetAuthority
} from "../domain/humanoid-replan-budget.js";
import type { JsonValue } from "../domain/schema.js";
import { attributeHumanoidDualLayerRun } from
  "./humanoid-dual-layer-attribution.js";

describe("humanoid dual-layer attribution", () => {
  it("separates Skill-use, Harness containment and controller outcomes", () => {
    const acceptedGrounding = grounding("execute-success", []);
    const boundaryGrounding = grounding("execute-boundary", ["active_goal"]);
    const complianceGrounding = grounding("execute-compliance", ["target_evidence"]);
    const actions = [
      plan("plan-success"),
      execution({
        transactionId: "execute-success",
        planningTransactionId: "plan-success",
        accepted: true,
        code: "motion_option_succeeded",
        frameCount: 20,
        grounding: acceptedGrounding,
        detail: {
          grounding_receipt: acceptedGrounding,
          result: {
            controller_routing: {
              execution: {
                route: "primary",
                attribution: {
                  primarySteps: 15,
                  fallbackSteps: 5,
                  upperBodyOverlaySteps: 0,
                  memoryBridgeSteps: 0
                }
              }
            }
          }
        }
      }),
      plan("plan-boundary"),
      execution({
        transactionId: "execute-boundary",
        planningTransactionId: "plan-boundary",
        accepted: false,
        code: "execution_grounding_rejected",
        frameCount: 0,
        grounding: boundaryGrounding
      }),
      plan("plan-compliance"),
      execution({
        transactionId: "execute-compliance",
        planningTransactionId: "plan-compliance",
        accepted: false,
        code: "execution_grounding_rejected",
        frameCount: 0,
        grounding: complianceGrounding
      }),
      plan("plan-never-triggered"),
      execution({
        transactionId: "execute-controller-failed",
        planningTransactionId: "legacy-plan",
        action: "execute_whole_body_motion",
        accepted: false,
        code: "motion_goal_unmet",
        frameCount: 8,
        detail: {
          result: {
            online_replans: [{ accepted: false }],
            online_replan_budget: {
              terminal_failure_class: "budget_exhausted",
              model_calls_consumed: 0
            }
          }
        }
      })
    ];
    const event = completedReplanEvent();

    const measured = attributeHumanoidDualLayerRun({
      status: "failed",
      actions,
      events: [event]
    });

    expect(measured).toMatchObject({
      coverage: {
        semantic_skill_call_count: 4,
        semantic_execution_count: 3,
        grounded_semantic_execution_count: 3,
        fully_attributed_semantic_execution_count: 1
      },
      harness: {
        semantic_planning: {
          skill_call_count: 4,
          accepted_skill_call_count: 4
        },
        skill_use: {
          trigger_expected: 4,
          trigger_observed: 3,
          trigger_missed: 1,
          compliance_evaluated: 3,
          compliant: 2,
          compliance_violations: 1,
          failed_obligation_counts: {
            active_goal: 1,
            target_evidence: 1
          },
          boundary_attempts: 1,
          boundary_contained: 1,
          boundary_escaped: 0,
          boundary_containment_rate: 1
        },
        dispatch: {
          attempts: 4,
          admitted: 2,
          rejected_before_actuation: 2,
          grounding_rejections: 2
        },
        recovery: {
          compact_replan_decisions: 1,
          completed_model_calls: 1,
          model_call_slo_violations: 1,
          local_navigation_replans: 1,
          local_navigation_budget_exhaustions: 1,
          local_navigation_model_calls: 0
        }
      },
      controller: {
        admitted_execution_count: 2,
        physical_execution_count: 2,
        succeeded_execution_count: 1,
        failed_execution_count: 1,
        success_rate_after_admission: 0.5,
        attributed_steps: {
          primary: 15,
          fallback: 5
        },
        primary_only_successes: 0,
        fallback_rescued_successes: 1,
        failure_class_counts: { contract_unmet: 1 }
      },
      joint: {
        successful_semantic_skills: 1,
        failed_semantic_skills: 2,
        semantic_skill_success_rate: 0.333333,
        harness_rejected_no_actuation: 2,
        harness_admitted_controller_succeeded: 1,
        harness_admitted_controller_failed: 1,
        primary_failure_layer_counts: {
          semantic_planner: 0,
          harness_gate: 3,
          controller: 1,
          none: 1,
          unattributed: 0
        },
        mission_failure_unattributed: false
      }
    });
  });
});

function plan(transactionId: string): unknown {
  return action({
    transactionId,
    action: "plan_humanoid_skill",
    accepted: true,
    code: "autonomous_skill_route_validated",
    frameCount: 0,
    input: {},
    detail: {}
  });
}

function execution(input: {
  transactionId: string;
  planningTransactionId: string;
  action?: string;
  accepted: boolean;
  code: string;
  frameCount: number;
  grounding?: ReturnType<typeof grounding>;
  detail?: JsonValue;
}): unknown {
  return action({
    transactionId: input.transactionId,
    action: input.action ?? "execute_humanoid_skill",
    accepted: input.accepted,
    code: input.code,
    frameCount: input.frameCount,
    input: { planning_transaction_id: input.planningTransactionId },
    detail: input.detail ?? {
      ...(input.grounding ? { grounding_receipt: input.grounding } : {})
    }
  });
}

function action(input: {
  transactionId: string;
  action: string;
  accepted: boolean;
  code: string;
  frameCount: number;
  input: JsonValue;
  detail: JsonValue;
}): unknown {
  return {
    transactionId: input.transactionId,
    agentId: input.action.startsWith("plan_")
      ? "humanoid-motion-reference"
      : "humanoid-executor",
    action: input.action,
    input: input.input,
    fingerprint: input.transactionId,
    accepted: input.accepted,
    code: input.code,
    worldBeforeRevision: 1,
    worldAfterRevision: 1 + input.frameCount,
    frameCount: input.frameCount,
    channels: ["left_arm"],
    detail: input.detail,
    committedAt: "2026-08-11T00:00:00.000Z"
  };
}

function grounding(
  transactionId: string,
  failed: Array<"active_goal" | "target_evidence">
) {
  const obligation = (
    id: "active_goal" | "semantic_preconditions" | "target_evidence",
    scope: "goal" | "skill" | "object"
  ) => createHumanoidGroundingObligation({
    id,
    scope,
    required: true,
    status: failed.includes(id as "active_goal" | "target_evidence")
      ? "failed"
      : "satisfied",
    code: failed.includes(id as "active_goal" | "target_evidence")
      ? `${id}_failed`
      : `${id}_satisfied`,
    detail: { id }
  });
  return createHumanoidGroundingReceipt({
    protocol: "humanoid-grounding-receipt-v1",
    receipt_id: `grounding:${transactionId}`,
    transaction_id: transactionId,
    planning_transaction_id: transactionId.replace("execute", "plan"),
    plan_id: `plan-id:${transactionId}`,
    call_id: `skill-call:${transactionId}`,
    world_frame: 10,
    world_revision: 10,
    authority_state_sha256: "a".repeat(64),
    obligations: [
      obligation("active_goal", "goal"),
      obligation("semantic_preconditions", "skill"),
      obligation("target_evidence", "object")
    ]
  });
}

function completedReplanEvent(): unknown {
  const started = beginHumanoidReplanModelCall(createHumanoidReplanBudget(), {
    modelCallId: "00000000-0000-4000-8000-000000000031",
    agentId: "humanoid-coordinator",
    role: "coordinator",
    at: "2026-08-11T00:00:00.000Z"
  });
  const completed = finishHumanoidReplanModelCall(started.budget, {
    modelCallId: started.call.model_call_id,
    status: "completed",
    at: "2026-08-11T00:00:31.000Z"
  });
  return {
    type: "model_request_completed",
    data: {
      replan_budget_evidence: {
        call: completed.call,
        authority: humanoidReplanBudgetAuthority(
          completed.budget,
          "2026-08-11T00:00:31.000Z"
        )
      }
    }
  };
}
