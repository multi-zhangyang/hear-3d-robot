import {
  sameAutonomousCycle,
  type AutonomousCycleRef
} from "../../domain/autonomous-cycle.js";
import { ScenarioBlockRemovalTransactionSchema } from "../../domain/scenario-block-removal.js";
import type { JsonValue } from "../../domain/schema.js";
import type { HumanoidActionReceipt } from "./runtime.js";
import {
  completedPhysicalExecution,
  object,
  physicalActuationReceipt,
  previousCycleEvidence
} from "./run-runtime-persistence.js";

export interface HumanoidCycleCausalEvidence {
  execution: HumanoidActionReceipt;
  worldMutations: HumanoidActionReceipt[];
}

export interface HumanoidCycleCausalEvidenceInput {
  evidenceTransactionIds: readonly string[];
  committedActions: Readonly<Record<string, HumanoidActionReceipt>>;
  previousCycle: JsonValue;
  activeCycle: AutonomousCycleRef | undefined;
  currentWorld: {
    worldRevision: number;
    robot: { fallen: boolean };
  };
}

export interface HumanoidCycleCompletionReadiness {
  status: "ready" | "not_ready";
  evidence_transaction_ids: string[];
  execution_transaction_id: string | null;
  observed_after_execution: boolean;
  reason: string | null;
}

export function resolveHumanoidCycleCompletionReadiness(
  input: Omit<HumanoidCycleCausalEvidenceInput, "evidenceTransactionIds">
): HumanoidCycleCompletionReadiness {
  const activeCycle = input.activeCycle;
  if (!activeCycle) return notReady("No autonomous cycle is active");
  const receipts = Object.values(input.committedActions);
  const executionIndex = receipts.findLastIndex((receipt) => (
    completedPhysicalExecution(receipt)
      && sameAutonomousCycle(receipt.cycle, activeCycle)
      && receipt.worldAfterRevision <= input.currentWorld.worldRevision
  ));
  const execution = receipts[executionIndex];
  if (!execution) {
    return notReady("The active cycle has no accepted physical execution");
  }
  const mutations = receipts.slice(executionIndex + 1).flatMap((receipt, offset) => {
    if (!receipt.accepted
      || receipt.action !== "remove_world_block"
      || receipt.code !== "world_block_removal_authorized"
      || !sameAutonomousCycle(receipt.cycle, activeCycle)) return [];
    const transaction = ScenarioBlockRemovalTransactionSchema.parse(
      object(receipt.detail).removal_transaction
    );
    return transaction.execution_transaction_id === execution.transactionId
      ? [{ receipt, index: executionIndex + offset + 1 }]
      : [];
  });
  const evidenceTransactionIds = [
    execution.transactionId,
    ...mutations.map(({ receipt }) => receipt.transactionId)
  ];
  try {
    validateHumanoidCycleCausalEvidenceCore({
      ...input,
      evidenceTransactionIds
    }, false);
  } catch (error) {
    return notReady(error instanceof Error ? error.message : String(error));
  }
  const observationBarrierIndex = mutations.at(-1)?.index ?? executionIndex;
  const observedAfterExecution = receipts.slice(observationBarrierIndex + 1).some((receipt) => (
    receipt.accepted
      && receipt.action === "observe_humanoid"
      && sameAutonomousCycle(receipt.cycle, activeCycle)
  ));
  return {
    status: "ready",
    evidence_transaction_ids: evidenceTransactionIds,
    execution_transaction_id: execution.transactionId,
    observed_after_execution: observedAfterExecution,
    reason: null
  };
}

export function validateHumanoidCycleCausalEvidence(
  input: HumanoidCycleCausalEvidenceInput
): HumanoidCycleCausalEvidence {
  return validateHumanoidCycleCausalEvidenceCore(input, true);
}

function validateHumanoidCycleCausalEvidenceCore(
  input: HumanoidCycleCausalEvidenceInput,
  requirePostExecutionObservation: boolean
): HumanoidCycleCausalEvidence {
  const evidence = input.evidenceTransactionIds.map((transactionId) => {
    const receipt = input.committedActions[transactionId];
    if (!receipt) throw new Error(`Unknown humanoid cycle evidence: ${transactionId}`);
    return receipt;
  });
  const previouslyConsumed = previousCycleEvidence(input.previousCycle);
  const repeated = evidence.find((receipt) => previouslyConsumed.has(receipt.transactionId));
  if (repeated) {
    throw new Error(`Humanoid execution evidence was already consumed: ${repeated.transactionId}`);
  }
  const activeCycle = input.activeCycle;
  if (!activeCycle) throw new Error("No autonomous cycle is active");

  const currentRevision = input.currentWorld.worldRevision;
  const execution = evidence.findLast((receipt) => (
    completedPhysicalExecution(receipt)
    && sameAutonomousCycle(receipt.cycle, activeCycle)
    && receipt.worldAfterRevision <= currentRevision
  ));
  if (!execution) {
    throw new Error(
      `Autonomous cycle requires accepted physical execution evidence no newer than world revision ${currentRevision}`
    );
  }

  const receipts = Object.values(input.committedActions);
  const latestActuation = receipts.findLast(physicalActuationReceipt);
  if (latestActuation?.transactionId !== execution.transactionId) {
    throw new Error("Autonomous cycle evidence was superseded by later physical actuation");
  }
  if (input.currentWorld.robot.fallen) {
    throw new Error("Autonomous cycle cannot complete after the humanoid has fallen");
  }

  const executionIndex = receipts.findIndex((receipt) => (
    receipt.transactionId === execution.transactionId
  ));
  const pendingPlan = receipts.slice(executionIndex + 1).find((receipt) => (
    receipt.accepted
    && sameAutonomousCycle(receipt.cycle, activeCycle)
    && (receipt.action === "plan_humanoid_skill"
      || receipt.action === "plan_whole_body_motion"
      || receipt.action === "plan_whole_body_motion_candidates"
      || receipt.action === "plan_humanoid_navigation")
  ));
  if (pendingPlan) {
    throw new Error(`Autonomous cycle has an unconsumed accepted plan: ${pendingPlan.transactionId}`);
  }

  const successfulRemovals = receipts.filter((receipt) => (
    receipt.accepted
      && receipt.action === "remove_world_block"
      && receipt.code === "world_block_removal_authorized"
      && sameAutonomousCycle(receipt.cycle, activeCycle)
  ));
  const worldMutations = successfulRemovals.filter((receipt) => {
    const transaction = ScenarioBlockRemovalTransactionSchema.parse(
      object(receipt.detail).removal_transaction
    );
    const receiptIndex = receipts.findIndex((candidate) => (
      candidate.transactionId === receipt.transactionId
    ));
    if (receiptIndex <= executionIndex) {
      throw new Error("Autonomous cycle world mutation precedes its physical execution");
    }
    if (transaction.execution_transaction_id !== execution.transactionId) {
      throw new Error(
        "Autonomous cycle contains a world mutation superseded by later physical actuation"
      );
    }
    return true;
  });

  const evidenceIds = new Set(input.evidenceTransactionIds);
  const omittedMutation = worldMutations.find((receipt) => (
    !evidenceIds.has(receipt.transactionId)
  ));
  if (omittedMutation) {
    throw new Error(
      `Autonomous cycle completion omits world mutation evidence: ${omittedMutation.transactionId}`
    );
  }
  const authorizedMutationIds = new Set(worldMutations.map((receipt) => (
    receipt.transactionId
  )));
  const invalidMutation = evidence.find((receipt) => (
    receipt.action === "remove_world_block"
      && !authorizedMutationIds.has(receipt.transactionId)
  ));
  if (invalidMutation) {
    throw new Error(
      `Autonomous cycle references invalid world mutation evidence: ${invalidMutation.transactionId}`
    );
  }

  if (requirePostExecutionObservation) {
    const mutationIndexes = worldMutations.map((mutation) => receipts.findIndex((receipt) => (
      receipt.transactionId === mutation.transactionId
    )));
    const observationBarrierIndex = Math.max(executionIndex, ...mutationIndexes);
    const observation = receipts.slice(observationBarrierIndex + 1).find((receipt) => (
      receipt.accepted
        && receipt.action === "observe_humanoid"
        && sameAutonomousCycle(receipt.cycle, activeCycle)
    ));
    if (!observation) {
      throw new Error(
        "Autonomous cycle completion requires an accepted Sentry observation after the latest physical execution and world mutation"
      );
    }
  }

  return {
    execution: structuredClone(execution),
    worldMutations: structuredClone(worldMutations)
  };
}

function notReady(reason: string): HumanoidCycleCompletionReadiness {
  return {
    status: "not_ready",
    evidence_transaction_ids: [],
    execution_transaction_id: null,
    observed_after_execution: false,
    reason
  };
}
