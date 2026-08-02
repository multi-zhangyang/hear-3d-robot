import { z } from "zod";
import {
  Vec3Schema,
  type ActionReceipt,
  type AgentReference,
  type JsonValue
} from "../domain/schema.js";
import { AgentSkillInputs } from "../runtime/actions.js";
import type { CommandResult } from "../world/rapier-world.js";

interface FrontierAgent {
  id: string;
  references: AgentReference[];
}

const FrontierChoiceSchema = z.object({
  choice_id: z.string().trim().min(1),
  target: Vec3Schema,
  face_point: Vec3Schema
}).passthrough();

export const TerrainSurveyDetailSchema = z.object({
  frontier: z.array(FrontierChoiceSchema)
}).passthrough();

export function resolveFrontierNavigation(input: {
  rawInput: unknown;
  agent: FrontierAgent;
  currentWorldRevision: number;
  lookupReceipt: (transactionId: string) => ActionReceipt | undefined;
}): { ok: true; input: unknown } | { ok: false; result: CommandResult } {
  const parsed = AgentSkillInputs.navigate_frontier.safeParse(input.rawInput);
  if (!parsed.success) {
    return failure("invalid_skill_input", {
      name: "navigate_frontier",
      issues: parsed.error.issues.map((issue) => ({
        path: issue.path.map(String),
        message: issue.message
      }))
    });
  }

  const transactionId = parsed.data.survey_transaction_id;
  const receipt = input.lookupReceipt(transactionId);
  if (!receipt) {
    return failure("unknown_survey_transaction", {
      survey_transaction_id: transactionId,
      recovery: "Call survey_terrain in this agent, then pass the exact transaction_id and one returned choice_id to navigate_frontier."
    });
  }

  const explicitlyGranted = input.agent.references.some((reference) =>
    reference.transaction_id === transactionId && reference.name === "survey_terrain"
  );
  if (receipt.agent_id !== input.agent.id && !explicitlyGranted) {
    return failure("survey_transaction_not_granted", {
      survey_transaction_id: transactionId,
      agent_id: input.agent.id,
      owning_agent_id: receipt.agent_id,
      owning_agent_name: receipt.agent_name,
      recovery: "Call survey_terrain in this agent or have the parent grant the accepted survey receipt by reference."
    });
  }

  if (!receipt.accepted || receipt.kind !== "tool" || receipt.name !== "survey_terrain"
    || receipt.code !== "terrain_survey") {
    return failure("invalid_survey_transaction", {
      survey_transaction_id: transactionId,
      actual_kind: receipt.kind,
      actual_action: receipt.name,
      actual_code: receipt.code,
      accepted: receipt.accepted
    });
  }

  if (receipt.world_revision !== input.currentWorldRevision) {
    return failure("stale_survey_revision", {
      survey_transaction_id: transactionId,
      surveyed_world_revision: receipt.world_revision,
      current_world_revision: input.currentWorldRevision,
      recovery: "The body changed after this survey. Call survey_terrain again and make a fresh model choice from the new frontier set."
    });
  }

  const survey = TerrainSurveyDetailSchema.safeParse(receipt.detail);
  if (!survey.success) {
    return failure("invalid_survey_frontier", {
      survey_transaction_id: transactionId,
      recovery: "The referenced receipt has no canonical frontier set. Call survey_terrain again."
    });
  }
  const matches = survey.data.frontier.filter((choice) =>
    choice.choice_id === parsed.data.choice_id
  );
  if (matches.length !== 1) {
    return failure(matches.length === 0 ? "unknown_frontier_choice" : "ambiguous_frontier_choice", {
      survey_transaction_id: transactionId,
      choice_id: parsed.data.choice_id,
      available_choice_ids: survey.data.frontier.map((choice) => choice.choice_id),
      recovery: matches.length === 0
        ? "Choose one exact choice_id from the referenced survey receipt; no substitute was selected."
        : "The survey is malformed because the choice identifier is not unique. Call survey_terrain again."
    });
  }

  const choice = matches[0]!;
  return {
    ok: true,
    input: {
      survey_transaction_id: transactionId,
      survey_world_revision: receipt.world_revision,
      choice_id: choice.choice_id,
      target: choice.target,
      face_point: choice.face_point,
      ...(parsed.data.options !== undefined ? { options: parsed.data.options } : {})
    }
  };
}

function failure(code: string, detail: JsonValue): { ok: false; result: CommandResult } {
  return { ok: false, result: { accepted: false, code, detail } };
}
