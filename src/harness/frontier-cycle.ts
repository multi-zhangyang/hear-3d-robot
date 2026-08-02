import type {
  ActionReceipt,
  AgentReference
} from "../domain/schema.js";
import { TerrainSurveyDetailSchema } from "./frontier-navigation.js";

export type FrontierCycleState = {
  phase: "survey_required" | "choice_required";
  enabled_action: "survey_terrain" | "navigate_frontier";
  decision_owner: "model";
  automatic_actuation: false;
  current_world_revision: number;
  current_survey_transaction_id: string | null;
  available_choice_ids: string[];
  reason: "no_current_survey" | "no_reachable_choices" | "current_survey_ready";
};

interface FrontierCycleAgent {
  id: string;
  capabilities: string[];
  references: AgentReference[];
}

/**
 * Projects the valid phase of a model-owned frontier cycle. The projection
 * never ranks or selects a choice; it only prevents the SDK from advertising a
 * state transition that the harness would deterministically reject.
 */
export function frontierCycleState(input: {
  agent: FrontierCycleAgent;
  receipts: readonly ActionReceipt[];
  currentWorldRevision: number;
}): FrontierCycleState | null {
  if (!input.agent.capabilities.includes("survey_terrain")
    || !input.agent.capabilities.includes("navigate_frontier")) return null;

  const granted = new Set(input.agent.references
    .filter((reference) => reference.name === "survey_terrain")
    .map((reference) => reference.transaction_id));
  for (let index = input.receipts.length - 1; index >= 0; index -= 1) {
    const receipt = input.receipts[index]!;
    if (receipt.agent_id !== input.agent.id && !granted.has(receipt.transaction_id)) continue;
    if (!receipt.accepted
      || receipt.kind !== "tool"
      || receipt.name !== "survey_terrain"
      || receipt.code !== "terrain_survey"
      || receipt.world_revision !== input.currentWorldRevision) continue;
    const detail = TerrainSurveyDetailSchema.safeParse(receipt.detail);
    if (!detail.success) continue;
    const choiceIds = detail.data.frontier.map((choice) => choice.choice_id);
    if (choiceIds.length === 0) {
      return state(
        "survey_required",
        input.currentWorldRevision,
        receipt.transaction_id,
        choiceIds,
        "no_reachable_choices"
      );
    }
    return state(
      "choice_required",
      input.currentWorldRevision,
      receipt.transaction_id,
      choiceIds,
      "current_survey_ready"
    );
  }
  return state(
    "survey_required",
    input.currentWorldRevision,
    null,
    [],
    "no_current_survey"
  );
}

export function frontierCycleActionEnabled(
  name: string,
  cycle: FrontierCycleState | null
): boolean {
  if (!cycle) return true;
  if (name === "survey_terrain") return cycle.phase === "survey_required";
  if (name === "navigate_frontier") return cycle.phase === "choice_required";
  return true;
}

function state(
  phase: FrontierCycleState["phase"],
  currentWorldRevision: number,
  currentSurveyTransactionId: string | null,
  availableChoiceIds: string[],
  reason: FrontierCycleState["reason"]
): FrontierCycleState {
  return {
    phase,
    enabled_action: phase === "survey_required" ? "survey_terrain" : "navigate_frontier",
    decision_owner: "model",
    automatic_actuation: false,
    current_world_revision: currentWorldRevision,
    current_survey_transaction_id: currentSurveyTransactionId,
    available_choice_ids: availableChoiceIds,
    reason
  };
}
