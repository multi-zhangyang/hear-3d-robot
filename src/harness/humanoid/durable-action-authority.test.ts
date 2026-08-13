import { describe, expect, it } from "vitest";
import {
  EmptyActionCommitOutbox,
  stageActionCommit
} from "../../domain/action-commit-outbox.js";
import {
  createActionTransactionIdentity
} from "../../domain/action-transaction-identity.js";
import type { JsonValue } from "../../domain/schema.js";
import type { JournalName, RunStore } from "../../persistence/run-store.js";
import { modelPayloadSha256 } from "../../domain/model-call-authority.js";
import { humanoidActionFingerprint, type HumanoidActionReceipt } from "./runtime.js";
import {
  recoverDurableHumanoidActionRuntimeState,
  verifyDurableHumanoidActionWindow
} from "./durable-action-authority.js";

describe("durable humanoid action authority", () => {
  const hierarchyEpochId = "11111111-1111-4111-8111-111111111111";

  it("rebuilds a hot receipt only from its identity, action row and runtime event", async () => {
    const proof = fixture();
    const assertions: string[] = [];
    const verified = await verifyDurableHumanoidActionWindow({
      store: journalStore(proof.journals),
      runId: "run-1",
      receipts: { [proof.receipt.transactionId]: proof.receipt },
      identities: new Map([[proof.identity.transaction_id, proof.identity]]),
      assertDecisionAuthority: (receipt, toolAuthority) => {
        expect(toolAuthority).toEqual(proof.actionToolAuthority);
        assertions.push(receipt.transactionId);
      }
    });

    expect(verified.get(proof.receipt.transactionId)?.receipt).toEqual(proof.receipt);
    expect(assertions).toEqual([proof.receipt.transactionId]);
  });

  it("rejects a checkpoint-only committed receipt", async () => {
    const proof = fixture();
    await expect(verifyDurableHumanoidActionWindow({
      store: journalStore({}),
      runId: "run-1",
      receipts: { [proof.receipt.transactionId]: proof.receipt },
      identities: new Map(),
      assertDecisionAuthority: () => undefined
    })).rejects.toThrow("no durable transaction identity");
  });

  it("rejects tampered checkpoint receipts and rebound runtime events", async () => {
    const proof = fixture();
    await expect(verifyDurableHumanoidActionWindow({
      store: journalStore(proof.journals),
      runId: "run-1",
      receipts: {
        [proof.receipt.transactionId]: {
          ...proof.receipt,
          code: "forged_success"
        }
      },
      identities: new Map([[proof.identity.transaction_id, proof.identity]]),
      assertDecisionAuthority: () => undefined
    })).rejects.toThrow("conflicts with durable authority");

    const rebound = structuredClone(proof.journals);
    rebound.events!.push(structuredClone(rebound.events![0]!));
    await expect(verifyDurableHumanoidActionWindow({
      store: journalStore(rebound),
      runId: "run-1",
      receipts: { [proof.receipt.transactionId]: proof.receipt },
      identities: new Map([[proof.identity.transaction_id, proof.identity]]),
      assertDecisionAuthority: () => undefined
    })).rejects.toThrow("Duplicate durable action runtime event identity");
  });

  it("requires a completed model decision authority after journal verification", async () => {
    const proof = fixture();
    await expect(verifyDurableHumanoidActionWindow({
      store: journalStore(proof.journals),
      runId: "run-1",
      receipts: { [proof.receipt.transactionId]: proof.receipt },
      identities: new Map([[proof.identity.transaction_id, proof.identity]]),
      assertDecisionAuthority: () => {
        throw new Error("model lifecycle is not authoritative");
      }
    })).rejects.toThrow("model lifecycle is not authoritative");
  });

  it("anchors mutable Skill runtime state to the latest committed event", async () => {
    const proof = fixture();
    const verified = await verifyDurableHumanoidActionWindow({
      store: journalStore(proof.journals),
      runId: "run-1",
      receipts: { [proof.receipt.transactionId]: proof.receipt },
      identities: new Map([[proof.identity.transaction_id, proof.identity]]),
      assertDecisionAuthority: () => undefined
    });
    expect(recoverDurableHumanoidActionRuntimeState({
      receipts: { [proof.receipt.transactionId]: proof.receipt },
      proofs: verified,
      checkpointState: proof.actionRuntimeState,
      neuralHierarchyEpochId: hierarchyEpochId
    })).toMatchObject({
      state: proof.actionRuntimeState,
      checkpointRecovered: false
    });
    expect(recoverDurableHumanoidActionRuntimeState({
      receipts: { [proof.receipt.transactionId]: proof.receipt },
      proofs: verified,
      checkpointState: {
        ...proof.actionRuntimeState,
        latest_physical_execution_revision: 1
      },
      neuralHierarchyEpochId: hierarchyEpochId
    })).toMatchObject({
      state: proof.actionRuntimeState,
      checkpointRecovered: true
    });
  });

  it("does not restore a previous epoch's cognitive hot cache", async () => {
    const proof = fixture();
    const verified = await verifyDurableHumanoidActionWindow({
      store: journalStore(proof.journals),
      runId: "run-1",
      receipts: { [proof.receipt.transactionId]: proof.receipt },
      identities: new Map([[proof.identity.transaction_id, proof.identity]]),
      assertDecisionAuthority: () => undefined
    });
    const freshEpochId = "55555555-5555-4555-8555-555555555555";
    const freshState = {
      version: 1 as const,
      neural_hierarchy_epoch_id: freshEpochId,
      latest_physical_execution_revision: 7,
      skill_plans: [],
      active_skill_plan_transactions: {},
      active_skills: [],
      planning_skill_bindings: [],
      recovery_policies: [],
      navigation_transit_clearance_requirements: [],
      latest_grounding_observation: null
    };

    expect(recoverDurableHumanoidActionRuntimeState({
      receipts: { [proof.receipt.transactionId]: proof.receipt },
      proofs: verified,
      checkpointState: freshState,
      neuralHierarchyEpochId: freshEpochId
    })).toEqual({
      state: freshState,
      checkpointRecovered: false
    });
  });
});

function fixture() {
  const actionRuntimeState = {
    version: 1 as const,
    neural_hierarchy_epoch_id: "11111111-1111-4111-8111-111111111111",
    latest_physical_execution_revision: 0,
    skill_plans: [],
    active_skill_plan_transactions: {},
    active_skills: [],
    planning_skill_bindings: [],
    recovery_policies: [],
    navigation_transit_clearance_requirements: []
  };
  const input = { id: "durable-plan", intent: "hold", duration_seconds: 0.1 };
  const actionToolAuthority = {
    tool_call_id: "transaction-1",
    tool_name: "plan_whole_body_motion",
    arguments_sha256: modelPayloadSha256(input)
  };
  const decision = {
    agent_id: "humanoid-motion",
    agent_manifest_sha256: "a".repeat(64),
    agent_manifest_epoch_id: "11111111-1111-4111-8111-111111111111",
    model_call_id: "22222222-2222-4222-8222-222222222222",
    response_id: "response-1",
    response_output_sha256: "b".repeat(64),
    tool_call_id: "transaction-1",
    tool_arguments_sha256: modelPayloadSha256(input)
  };
  const receipt: HumanoidActionReceipt = {
    transactionId: "transaction-1",
    agentId: "humanoid-motion",
    decision,
    cycle: {
      cycle_id: "autonomous-cycle:33333333-3333-4333-8333-333333333333",
      cycle_index: 1,
      goal_epoch_id: `goal-epoch:${"4".repeat(64)}`
    },
    action: "plan_whole_body_motion",
    input,
    fingerprint: humanoidActionFingerprint(
      "plan_whole_body_motion",
      "humanoid-motion",
      input
    ),
    accepted: true,
    code: "whole_body_plan_validated",
    worldBeforeRevision: 0,
    worldAfterRevision: 0,
    frameCount: 0,
    channels: [],
    detail: { automatic_actuation: false },
    commitSequence: 1,
    committedAt: "2026-08-12T00:00:00.000Z"
  };
  const actionRecord = {
    ...receipt,
    runtime_event_id: "event-1"
  };
  const outbox = stageActionCommit(EmptyActionCommitOutbox, {
    transactionId: receipt.transactionId,
    runtimeEventId: "event-1",
    actionRecord: json(actionRecord),
    goalEvidenceRef: "action:transaction-1",
    goalEvidenceRecord: {
      evidence: { ref: "action:transaction-1", kind: "action_receipt" },
      payload: { transaction_id: "transaction-1", receipt }
    },
    runtimeEvent: {
      event_id: "event-1",
      run_id: "run-1",
      type: "humanoid_action_committed",
      at: "2026-08-12T00:00:00.000Z",
      data: {
        receipt: actionRecord,
        action_tool_authority: actionToolAuthority,
        action_runtime_state: actionRuntimeState
      }
    },
    stagedAt: "2026-08-12T00:00:00.000Z"
  });
  const commit = outbox.pending[receipt.transactionId]!;
  const event = {
    ...commit.runtime_event,
    cursor: "event-cursor-1"
  };
  return {
    receipt,
    actionRuntimeState,
    actionToolAuthority,
    identity: createActionTransactionIdentity(commit),
    journals: {
      actions: [commit.action_record],
      events: [json(event)],
      goal_evidence: [commit.goal_evidence_record]
    } satisfies Partial<Record<JournalName, JsonValue[]>>
  };
}

function journalStore(journals: Partial<Record<JournalName, JsonValue[]>>): RunStore {
  return {
    scanJournal: async (
      name: JournalName,
      visit: (entry: JsonValue, index: number) => void | Promise<void>
    ) => {
      for (const [index, entry] of (journals[name] ?? []).entries()) {
        await visit(structuredClone(entry), index);
      }
    }
  } as RunStore;
}

function json(value: unknown): JsonValue {
  return JSON.parse(JSON.stringify(value)) as JsonValue;
}
