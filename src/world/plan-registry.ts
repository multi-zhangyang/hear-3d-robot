import type { JsonValue } from "../domain/schema.js";

export type PlanKind = "base" | "arm";

/**
 * A plan is computed from one robot pose and stays valid exactly as long as the
 * robot has not left that pose. World revision, not the physics frame counter,
 * expresses that lifetime: reading the world does not invalidate a plan, and a
 * plan survives arbitrarily many physics steps until a body command commits.
 */
export type PlanLookup<T> =
  | { status: "valid"; plan: T }
  | { status: "unknown" }
  | { status: "consumed"; consumedAtRevision: number }
  | { status: "stale"; createdRevision: number };

const PLAN_HISTORY_LIMIT = 256;
const CONSUMED_HISTORY_LIMIT = 256;

export class PlanRegistry<T extends { id: string; createdRevision: number }> {
  readonly #plans = new Map<string, T>();
  readonly #consumed = new Map<string, number>();

  /** Plans still valid at the given revision. Superseded plans are retained so
   * a later execution can be told apart from an unknown identifier. */
  valid(currentRevision: number): T[] {
    return [...this.#plans.values()].filter((plan) => plan.createdRevision === currentRevision);
  }

  set(plan: T): void {
    this.#plans.set(plan.id, plan);
    evictOldest(this.#plans, PLAN_HISTORY_LIMIT);
  }

  /**
   * Resolves a plan id against the current world revision. A superseded plan is
   * reported as stale rather than forgotten, so the caller can name the exact
   * revision mismatch and the tool to call to recover.
   */
  lookup(planId: string, currentRevision: number): PlanLookup<T> {
    const plan = this.#plans.get(planId);
    if (!plan) {
      const consumedAtRevision = this.#consumed.get(planId);
      return consumedAtRevision === undefined
        ? { status: "unknown" }
        : { status: "consumed", consumedAtRevision };
    }
    return plan.createdRevision === currentRevision
      ? { status: "valid", plan }
      : { status: "stale", createdRevision: plan.createdRevision };
  }

  /** Marks a plan as spent so a second execution is distinguishable from expiry. */
  consume(planId: string, currentRevision: number): void {
    this.#plans.delete(planId);
    this.#consumed.set(planId, currentRevision);
    evictOldest(this.#consumed, CONSUMED_HISTORY_LIMIT);
  }
}

function evictOldest(entries: Map<string, unknown>, limit: number): void {
  while (entries.size > limit) {
    const oldest = entries.keys().next();
    if (oldest.done) return;
    entries.delete(oldest.value);
  }
}

export function planDenialDetail(
  lookup: Exclude<PlanLookup<unknown>, { status: "valid" }>,
  planId: string,
  kind: PlanKind,
  currentRevision: number
): { code: string; detail: JsonValue } {
  const replanTool = kind === "base"
    ? "plan_base_path"
    : "plan_joint_targets for a relative posture, or solve_end_effector_position/solve_end_effector_pose for a fixed world-space target";
  if (lookup.status === "unknown") {
    return {
      code: kind === "base" ? "unknown_base_plan" : "unknown_arm_plan",
      detail: {
        plan_id: planId,
        recovery: `No plan with this identifier exists. Call ${replanTool} to create one.`
      }
    };
  }
  if (lookup.status === "consumed") {
    return {
      code: "plan_already_consumed",
      detail: {
        plan_id: planId,
        consumed_at_world_revision: lookup.consumedAtRevision,
        current_world_revision: currentRevision,
        recovery: `This plan already executed once and cannot be replayed. Observe the current state, then call ${replanTool} for a new plan.`
      }
    };
  }
  return {
    code: "stale_plan_revision",
    detail: {
      plan_id: planId,
      planned_world_revision: lookup.createdRevision,
      current_world_revision: currentRevision,
      recovery: `The world changed after this plan was created. Re-observe the current state, then call ${replanTool} to plan from the current pose.`
    }
  };
}
