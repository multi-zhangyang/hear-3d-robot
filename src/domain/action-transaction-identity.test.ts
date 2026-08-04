import { describe, expect, it } from "vitest";
import {
  EmptyActionCommitOutbox,
  stageActionCommit
} from "./action-commit-outbox.js";
import {
  ActionTransactionIdentitySchema,
  actionTransactionFingerprintSha256,
  createActionTransactionIdentity,
  rebuildActionTransactionIdentities
} from "./action-transaction-identity.js";

describe("action transaction identity", () => {
  it("creates a permanent lightweight identity from the canonical commit", () => {
    const identity = createActionTransactionIdentity(commit());
    expect(identity).toMatchObject({
      version: 1,
      run_id: "run-1",
      transaction_id: "transaction-1",
      agent_id: "executor",
      action: "execute_whole_body_motion",
      action_fingerprint_sha256: actionTransactionFingerprintSha256("fingerprint"),
      runtime_event_id: "event-1",
      goal_evidence_ref: "action:transaction-1"
    });
    expect(identity.identity_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(rebuildActionTransactionIdentities([identity], "run-1").get(
      "transaction-1"
    )).toEqual(identity);
  });

  it("rejects tampering, duplicate tombstones and cross-run reuse", () => {
    const identity = createActionTransactionIdentity(commit());
    const tampered = { ...identity, agent_id: "other" };
    expect(ActionTransactionIdentitySchema.safeParse(tampered).success).toBe(false);
    expect(() => rebuildActionTransactionIdentities(
      [identity, identity],
      "run-1"
    )).toThrow(/Duplicate/);
    expect(() => rebuildActionTransactionIdentities(
      [identity],
      "run-2"
    )).toThrow(/another run/);
  });
});

function commit() {
  const receipt = {
    transactionId: "transaction-1",
    agentId: "executor",
    action: "execute_whole_body_motion" as const,
    input: { planning_transaction_id: "planning-1" },
    fingerprint: "fingerprint",
    accepted: true,
    code: "motion_completed",
    worldBeforeRevision: 1,
    worldAfterRevision: 2,
    frameCount: 1,
    channels: ["locomotion"],
    detail: { plan_id: "plan-1" },
    committedAt: "2026-08-03T10:00:00.000Z"
  };
  const actionRecord = { ...receipt, runtime_event_id: "event-1" };
  return stageActionCommit(EmptyActionCommitOutbox, {
    transactionId: receipt.transactionId,
    runtimeEventId: "event-1",
    actionRecord,
    goalEvidenceRef: "action:transaction-1",
    goalEvidenceRecord: {
      evidence: { ref: "action:transaction-1", kind: "action_receipt" },
      payload: { transaction_id: "transaction-1", receipt }
    },
    runtimeEvent: {
      event_id: "event-1",
      run_id: "run-1",
      type: "humanoid_action_committed",
      at: "2026-08-03T10:00:00.000Z",
      data: { receipt: actionRecord }
    }
  }).pending[receipt.transactionId]!;
}
