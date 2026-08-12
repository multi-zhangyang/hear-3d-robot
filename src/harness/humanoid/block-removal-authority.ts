import {
  sameAutonomousCycle,
  type AutonomousCycleRef
} from "../../domain/autonomous-cycle.js";
import {
  createScenarioBlockRemovalTransaction,
  MINIMUM_BLOCK_REMOVAL_NORMAL_FORCE_N,
  MINIMUM_BLOCK_REMOVAL_STABLE_FRAMES,
  type ScenarioBlockRemovalTransaction
} from "../../domain/scenario-block-removal.js";
import type { Scenario } from "../../domain/schema.js";
import type { ScenarioChunkDeltaState } from "../../domain/scenario-chunk-delta-schema.js";
import {
  HumanoidMotionOptionContractSchema,
  humanoidMotionOptionContractSha256,
  type HumanoidMotionOptionCondition,
  type HumanoidMotionOptionContract
} from "../../world/humanoid/motion-option.js";
import type { HumanoidWorldSnapshot } from "../../world/humanoid/world.js";
import type { HumanoidActionReceipt } from "./runtime.js";
import { humanoidActionReceiptsInCommitOrder } from "../../domain/humanoid-run.js";

export type BlockRemovalAuthorityCode =
  | "block_removal_execution_missing"
  | "block_removal_execution_invalid"
  | "block_removal_execution_superseded"
  | "block_removal_execution_consumed"
  | "block_removal_plan_invalid"
  | "block_removal_contact_contract_missing"
  | "block_removal_contact_evidence_missing"
  | "block_removal_contact_too_brief"
  | "block_removal_contact_force_insufficient"
  | "block_removal_target_invalid";

export class BlockRemovalAuthorityError extends Error {
  readonly code: BlockRemovalAuthorityCode;

  constructor(code: BlockRemovalAuthorityCode, message: string) {
    super(message);
    this.name = "BlockRemovalAuthorityError";
    this.code = code;
  }
}

export function prepareAuthorizedBlockRemoval(input: {
  scenario: Scenario;
  chunks: ScenarioChunkDeltaState;
  currentWorld: HumanoidWorldSnapshot;
  activeCycle: AutonomousCycleRef;
  removalTransactionId: string;
  agentId: string;
  solidId: string;
  executionTransactionId: string;
  committedActions: Readonly<Record<string, HumanoidActionReceipt>>;
}): ScenarioBlockRemovalTransaction {
  const execution = input.committedActions[input.executionTransactionId];
  if (!execution) {
    reject(
      "block_removal_execution_missing",
      `Block removal execution receipt is unavailable: ${input.executionTransactionId}`
    );
  }
  if (execution.agentId !== input.agentId
    || execution.action !== "execute_whole_body_motion"
    || !execution.accepted
    || execution.code !== "motion_option_succeeded"
    || execution.frameCount <= 0
    || execution.worldAfterRevision - execution.worldBeforeRevision !== execution.frameCount
    || execution.worldAfterRevision > input.currentWorld.worldRevision
    || !sameAutonomousCycle(execution.cycle, input.activeCycle)) {
    reject(
      "block_removal_execution_invalid",
      "Block removal requires a successful same-Agent whole-body execution from the active autonomous cycle"
    );
  }

  const receipts = humanoidActionReceiptsInCommitOrder(input.committedActions);
  const latestActuation = receipts.findLast((receipt) => (
    receipt.frameCount > 0
      && (receipt.action === "execute_whole_body_motion"
        || receipt.action === "execute_humanoid_navigation")
  ));
  if (latestActuation?.transactionId !== execution.transactionId) {
    reject(
      "block_removal_execution_superseded",
      "Block removal contact evidence was superseded by later physical actuation"
    );
  }
  const consumed = receipts.find((receipt) => {
    if (!receipt.accepted || receipt.action !== "remove_world_block") return false;
    const transaction = record(record(receipt.detail).removal_transaction);
    return transaction.execution_transaction_id === execution.transactionId;
  });
  if (consumed) {
    reject(
      "block_removal_execution_consumed",
      `Physical contact execution was already consumed by ${consumed.transactionId}`
    );
  }

  const executionInput = record(execution.input);
  const executionDetail = record(execution.detail);
  const planningTransactionId = stringValue(
    executionInput.planning_transaction_id
      ?? executionDetail.planning_transaction_id
  );
  if (!planningTransactionId) {
    reject(
      "block_removal_plan_invalid",
      "Block removal execution has no planning transaction identity"
    );
  }
  const planning = input.committedActions[planningTransactionId];
  if (!planning
    || planning.action !== "plan_whole_body_motion_candidates"
    || !planning.accepted
    || !sameAutonomousCycle(planning.cycle, input.activeCycle)
    || executionDetail.planning_action !== planning.action
    || executionDetail.plan_id !== record(planning.detail).plan_id) {
    reject(
      "block_removal_plan_invalid",
      "Block removal requires the exact accepted multi-candidate plan consumed by its execution"
    );
  }

  const planDetail = record(planning.detail);
  const termination = parseContract(planDetail.termination);
  const certifiedContract = parseContract(record(planDetail.option).contract);
  if (humanoidMotionOptionContractSha256(termination)
    !== humanoidMotionOptionContractSha256(certifiedContract)) {
    reject(
      "block_removal_plan_invalid",
      "Block removal planning receipt has inconsistent termination and certificate contracts"
    );
  }
  const contact = requiredSolidContact(certifiedContract, input.solidId);
  if (!contact) {
    reject(
      "block_removal_contact_contract_missing",
      `The certified terminal contract does not require contact with ${input.solidId}`
    );
  }

  const executionResult = record(executionDetail.result);
  const option = record(executionResult.option);
  if (option.status !== "succeeded"
    || option.termination_reason !== "physical_success"
    || option.option_id !== certifiedContract.option_id) {
    reject(
      "block_removal_execution_invalid",
      "Block removal execution does not contain a successful certified physical Option"
    );
  }
  const evidence = record(option.evidence);
  const monitor = record(evidence.monitor);
  const observedStableFrames = integerValue(monitor.terminalStableSteps);
  if (certifiedContract.stable_steps < MINIMUM_BLOCK_REMOVAL_STABLE_FRAMES
    || observedStableFrames === undefined
    || observedStableFrames < certifiedContract.stable_steps) {
    reject(
      "block_removal_contact_too_brief",
      `Block removal requires at least ${MINIMUM_BLOCK_REMOVAL_STABLE_FRAMES} stable physical frames`
    );
  }
  const predicates = Array.isArray(evidence.predicates) ? evidence.predicates : [];
  const predicateEvidence = record(predicates[contact.predicateIndex]);
  const maximumNormalForce = finiteNumber(predicateEvidence.maximumNormalForce);
  if (predicateEvidence.status !== "satisfied"
    || predicateEvidence.type !== contact.predicate.type
    || predicateEvidence.solidId !== input.solidId
    || maximumNormalForce === undefined) {
    reject(
      "block_removal_contact_evidence_missing",
      `Successful execution has no terminal contact evidence for ${input.solidId}`
    );
  }
  const requiredNormalForce = Math.max(
    MINIMUM_BLOCK_REMOVAL_NORMAL_FORCE_N,
    contact.predicate.minimum_normal_force
  );
  if (maximumNormalForce < requiredNormalForce) {
    reject(
      "block_removal_contact_force_insufficient",
      `Block removal contact force is below the Harness threshold for ${input.solidId}`
    );
  }

  const revisionLag = input.currentWorld.worldRevision - execution.worldAfterRevision;
  const sourceWorldFrame = input.currentWorld.frame - revisionLag;
  if (!Number.isSafeInteger(sourceWorldFrame) || sourceWorldFrame < 0) {
    reject(
      "block_removal_execution_invalid",
      "Block removal cannot resolve the physical source frame"
    );
  }
  try {
    return createScenarioBlockRemovalTransaction({
      scenario: input.scenario,
      chunks: input.chunks,
      transactionId: input.removalTransactionId,
      solidId: input.solidId,
      executionTransactionId: execution.transactionId,
      planningTransactionId,
      sourceWorldFrame,
      sourceWorldRevision: execution.worldAfterRevision,
      contactEvidence: {
        predicate_index: contact.predicateIndex,
        predicate_type: contact.predicate.type,
        surface_kind: contact.predicate.type === "body_contact_solid"
          ? "body"
          : "hand_surface",
        surface: contact.predicate.type === "body_contact_solid"
          ? contact.predicate.body
          : contact.predicate.hand_surface,
        planned_stable_frames: certifiedContract.stable_steps,
        observed_stable_frames: observedStableFrames,
        planned_minimum_normal_force_n: contact.predicate.minimum_normal_force,
        observed_maximum_normal_force_n: maximumNormalForce
      }
    });
  } catch (error) {
    reject(
      "block_removal_target_invalid",
      error instanceof Error ? error.message : String(error)
    );
  }
}

function requiredSolidContact(
  contract: HumanoidMotionOptionContract,
  solidId: string
): {
  predicateIndex: number;
  predicate: Extract<HumanoidMotionOptionContract["predicates"][number], {
    type: "body_contact_solid" | "hand_contact_solid";
  }>;
} | undefined {
  const terminal = contract.phases?.terminal.condition;
  for (let predicateIndex = 0; predicateIndex < contract.predicates.length; predicateIndex += 1) {
    const predicate = contract.predicates[predicateIndex]!;
    if ((predicate.type !== "body_contact_solid"
      && predicate.type !== "hand_contact_solid")
      || predicate.solid_id !== solidId) continue;
    if (!terminal || conditionRequiresPredicate(terminal, predicateIndex)) {
      return { predicateIndex, predicate };
    }
  }
  return undefined;
}

function conditionRequiresPredicate(
  condition: HumanoidMotionOptionCondition,
  predicateIndex: number
): boolean {
  if (condition.op === "predicate") {
    return condition.predicate_index === predicateIndex;
  }
  if (condition.op === "not") return false;
  const requirements = condition.conditions.map((nested) => (
    conditionRequiresPredicate(nested, predicateIndex)
  ));
  return condition.op === "all"
    ? requirements.some(Boolean)
    : requirements.every(Boolean);
}

function parseContract(value: unknown): HumanoidMotionOptionContract {
  const parsed = HumanoidMotionOptionContractSchema.safeParse(value);
  if (!parsed.success) {
    reject(
      "block_removal_plan_invalid",
      "Block removal planning receipt has no valid physical Option contract"
    );
  }
  return parsed.data;
}

function reject(code: BlockRemovalAuthorityCode, message: string): never {
  throw new BlockRemovalAuthorityError(code, message);
}

function record(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function integerValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isSafeInteger(value) ? value : undefined;
}
