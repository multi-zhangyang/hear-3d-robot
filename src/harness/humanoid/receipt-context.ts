import type { JsonValue } from "../../domain/schema.js";
import type { HumanoidActionReceipt } from "./runtime.js";

const MODEL_RECEIPT_DETAIL_KEYS = [
  "frame",
  "world_revision",
  "planning_transaction_id",
  "planning_action",
  "plan_id",
  "objective",
  "intent_sha256",
  "target",
  "chunk_target",
  "requested_arrival_heading",
  "arrival_heading",
  "distance",
  "remaining_distance",
  "partial_endpoint",
  "reachable_base_placements",
  "repeated_action",
  "repeated_failure_count",
  "previous_code",
  "physical_execution_revision",
  "preview_frames",
  "preview_travelled_m",
  "blocking_contacts",
  "selected_candidate_id",
  "selected_rank",
  "candidate_count",
  "autonomous_plan_kind",
  "failure_class",
  "recovery_kind",
  "recovery_collision",
  "termination",
  "frames",
  "travelledDistance",
  "reason",
  "terminal_result_sha256",
  "solid_id",
  "source_id",
  "revalidation",
  "automatic_actuation",
  "recovery",
  "supplied_skill_transaction_id",
  "active_skill_transaction_id"
] as const;

export function recentReceiptContext(receipt: HumanoidActionReceipt): JsonValue {
  return json({
    transaction_id: receipt.transactionId,
    agent_id: receipt.agentId,
    action: receipt.action,
    accepted: receipt.accepted,
    code: receipt.code,
    world_before_revision: receipt.worldBeforeRevision,
    world_after_revision: receipt.worldAfterRevision,
    frame_count: receipt.frameCount,
    detail: modelReceiptDetail(receipt.detail)
  });
}

export function modelReceiptDetail(value: JsonValue): Record<string, unknown> {
  const source = record(value);
  if (!source) return {};
  const projected: Record<string, unknown> = {};
  for (const key of MODEL_RECEIPT_DETAIL_KEYS) {
    if (source[key] !== undefined) projected[key] = source[key];
  }
  const validation = record(source.validation);
  if (validation) projected.validation = validationSummary(validation);
  if (Array.isArray(source.candidates)) {
    projected.candidates = source.candidates.map((candidate) => {
      const candidateRecord = record(candidate);
      if (!candidateRecord) return {};
      const summary: Record<string, unknown> = {};
      for (const key of ["rank", "id", "plan_id", "intent", "selected"] as const) {
        if (candidateRecord[key] !== undefined) summary[key] = candidateRecord[key];
      }
      const candidateValidation = record(candidateRecord.validation);
      if (candidateValidation) {
        summary.validation = validationSummary(candidateValidation);
      }
      return summary;
    });
  }
  if (Array.isArray(source.attempts)) {
    projected.attempts = source.attempts.slice(0, 8).map((attempt) => {
      const attemptRecord = record(attempt);
      return attemptRecord
        ? projectRecord(attemptRecord, [
            "target",
            "score",
            "accepted",
            "reason",
            "blocking_contacts"
          ])
        : {};
    });
  }
  const skillBinding = record(source.skill_binding);
  if (skillBinding) {
    projected.skill_binding = projectRecord(skillBinding, [
      "transaction_id",
      "skill_plan_transaction_id",
      "skill_node_id",
      "invocation",
      "phase",
      "phase_authority",
      "planning_action",
      "observed_world_revision",
      "target_position",
      "control_mode"
    ]);
  }
  const binding = record(source.binding);
  if (binding) {
    projected.binding = projectRecord(binding, [
      "transaction_id",
      "skill_plan_transaction_id",
      "skill_node_id",
      "invocation",
      "phase",
      "phase_authority",
      "planning_action",
      "observed_world_revision"
    ]);
  }
  const result = record(source.result);
  if (result) {
    projected.result = projectRecord(result, [
      "travelledDistance",
      "terminationReason",
      "reason",
      "blocking_contacts",
      "carry",
      "revalidation"
    ]);
  }
  const final = record(source.final);
  if (final) {
    projected.final = projectRecord(final, [
      "simulated_time",
      "root_position",
      "fallen",
      "balance"
    ]);
  }
  return projected;
}

export function modelToolReceiptDetail(
  receipt: HumanoidActionReceipt
): Record<string, unknown> {
  if (receipt.action !== "observe_humanoid"
    || receipt.agentId !== "humanoid-motion-reference") {
    return modelReceiptDetail(receipt.detail);
  }
  const source = record(receipt.detail);
  if (!source) return {};
  return projectRecord(source, [
    "frame",
    "world_revision",
    "sensor",
    "root",
    "fallen",
    "balance",
    "feet",
    "key_links",
    "end_effectors",
    "manipulation_geometry",
    "hand_coordination",
    "hand_surfaces",
    "object_tokens",
    "solid_tokens",
    "grasp",
    "interaction",
    "contacts",
    "non_foot_environment_contacts",
    "navigation"
  ]);
}

function validationSummary(validation: Record<string, unknown>): Record<string, unknown> {
  return projectRecord(validation, ["feasible", "failures"]);
}

function projectRecord(
  source: Record<string, unknown>,
  keys: readonly string[]
): Record<string, unknown> {
  return Object.fromEntries(keys.flatMap((key) => (
    source[key] === undefined ? [] : [[key, source[key]]]
  )));
}

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

function json(value: unknown): JsonValue {
  return JSON.parse(JSON.stringify(value)) as JsonValue;
}
