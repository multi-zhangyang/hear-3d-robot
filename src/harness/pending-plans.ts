import type { ActionReceipt, JsonValue } from "../domain/schema.js";
import type { PlanKind } from "../world/plan-registry.js";

interface PlannerContract {
  kind: PlanKind;
  executor: "execute_base_plan" | "execute_joint_plan";
}

export interface RequiredPlanHandoff {
  planning_transaction_id: string;
  planning_action: string;
  required_execution_action: PlannerContract["executor"];
}

const PLANNERS: Record<string, PlannerContract | undefined> = {
  plan_base_path: { kind: "base", executor: "execute_base_plan" },
  plan_joint_targets: { kind: "arm", executor: "execute_joint_plan" },
  solve_end_effector_position: { kind: "arm", executor: "execute_joint_plan" },
  solve_end_effector_pose: { kind: "arm", executor: "execute_joint_plan" }
};

/**
 * Find a still-valid plan authored by a leaf that cannot execute it.
 *
 * That plan is a hierarchy phase boundary: continuing to expose perception or
 * planning tools lets the leaf reason about a future body pose that does not
 * exist yet. The model must end its assignment and hand the accepted receipt
 * back to a parent, which can grant it to an executor. This function only
 * detects that state; it never chooses or executes the plan.
 */
export function requiredPlanHandoff(input: {
  receipts: readonly ActionReceipt[];
  agentId: string;
  capabilities: readonly string[];
  status: (
    kind: PlanKind,
    planId: string
  ) => "valid" | "unknown" | "consumed" | "stale";
}): RequiredPlanHandoff | null {
  for (let index = input.receipts.length - 1; index >= 0; index -= 1) {
    const receipt = input.receipts[index]!;
    const contract = PLANNERS[receipt.name];
    if (!contract
      || !receipt.accepted
      || receipt.kind !== "tool"
      || receipt.agent_id !== input.agentId
      || input.capabilities.includes(contract.executor)) continue;

    const detail = objectValue(receipt.detail);
    const planId = detail?.plan_id;
    if (typeof planId !== "string" || input.status(contract.kind, planId) !== "valid") continue;
    return {
      planning_transaction_id: receipt.transaction_id,
      planning_action: receipt.name,
      required_execution_action: contract.executor
    };
  }
  return null;
}

/**
 * Project still-valid plans into fresh model context after resume/compaction.
 * This is an authority view, not a scheduler: it names executable receipts
 * and explicitly leaves the decision and actuation to a model-run agent.
 */
export function pendingPlanReceipts(input: {
  receipts: readonly ActionReceipt[];
  visible: (receipt: ActionReceipt) => boolean;
  status: (
    kind: PlanKind,
    planId: string
  ) => "valid" | "unknown" | "consumed" | "stale";
  limit?: number;
}): JsonValue[] {
  const limit = Math.max(0, Math.trunc(input.limit ?? 6));
  const pending: JsonValue[] = [];
  const seenPlanIds = new Set<string>();

  for (let index = input.receipts.length - 1; index >= 0; index -= 1) {
    if (pending.length >= limit) break;
    const receipt = input.receipts[index]!;
    const contract = PLANNERS[receipt.name];
    if (!contract || !receipt.accepted || receipt.kind !== "tool" || !input.visible(receipt)) {
      continue;
    }
    const detail = objectValue(receipt.detail);
    if (!detail) continue;
    const planId = detail.plan_id;
    if (typeof planId !== "string" || seenPlanIds.has(planId)) continue;
    seenPlanIds.add(planId);
    if (input.status(contract.kind, planId) !== "valid") continue;

    pending.push({
      planning_transaction_id: receipt.transaction_id,
      planning_action: receipt.name,
      required_execution_action: contract.executor,
      planning_agent_id: receipt.agent_id,
      planning_agent_name: receipt.agent_name,
      world_revision: receipt.world_revision,
      target: detail.target ?? detail.resolved_target ?? null,
      reference: { transaction_id: receipt.transaction_id },
      plan_status: "valid",
      automatic_actuation: false,
      decision_owner: "model",
      usage: `A model may delegate this receipt by reference to an agent with ${contract.executor}; the harness never executes it automatically.`
    });
  }
  return pending;
}

function objectValue(value: JsonValue): Record<string, JsonValue> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value
    : null;
}
