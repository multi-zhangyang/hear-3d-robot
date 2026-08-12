import { describe, expect, it } from "vitest";
import {
  EmptyActionCommitOutbox,
  stageActionCommit
} from "./action-commit-outbox.js";
import {
  ActionExecutionLedgerSchema,
  EmptyActionExecutionLedger,
  acknowledgeTerminalActionExecution,
  activeActionExecutions,
  recordActionExecutionProgress,
  restoreActionExecutionLedger,
  stageActionExecutionIntent,
  terminalizeActionExecution
} from "./action-execution-ledger.js";

const HASH_A = "a".repeat(64);
const HASH_B = "b".repeat(64);
const HASH_C = "c".repeat(64);
const MODEL_CALL_ID = "00000000-0000-4000-8000-000000000099";
const CYCLE = {
  cycle_id: "autonomous-cycle:00000000-0000-4000-8000-000000000001",
  cycle_index: 1,
  goal_epoch_id: `goal-epoch:${"d".repeat(64)}`
} as const;

describe("action execution ledger", () => {
  it("durably admits one immutable transaction before its first physical frame", () => {
    const admitted = admittedLedger();
    expect(activeActionExecutions(admitted)).toEqual([
      expect.objectContaining({
        run_id: "run-1",
        transaction_id: "execute-1",
        cycle: CYCLE,
        action: "execute_whole_body_motion",
        status: "admitted",
        admission: expect.objectContaining({
          planning_transaction_id: "plan-1",
          plan_id: "physical-plan-1",
          world_frame: 90,
          world_revision: 10
        }),
        progress: expect.objectContaining({
          committed_frame_count: 0,
          world_frame: 90,
          world_revision: 10
        })
      })
    ]);

    expect(stageIntent(admitted)).toEqual(admitted);
    expect(() => stageIntent(admitted, { actionFingerprint: "different" }))
      .toThrow(/transaction conflict/);
    expect(() => stageIntent(admitted, {
      transactionId: "execute-2",
      planId: "physical-plan-1"
    })).toThrow(/already admitted/);
  });

  it("advances only a monotonic checkpoint-bound physical prefix", () => {
    const admitted = admittedLedger();
    const executing = recordActionExecutionProgress(admitted, {
      transactionId: "execute-1",
      committedFrameCount: 2,
      worldFrame: 92,
      worldRevision: 12,
      authorityStateSha256: HASH_B,
      physicalCheckpointSha256: HASH_C,
      updatedAt: "2026-08-03T10:00:02.000Z"
    });
    expect(executing.active["execute-1"]).toMatchObject({
      status: "executing",
      progress: {
        committed_frame_count: 2,
        world_frame: 92,
        world_revision: 12,
        authority_state_sha256: HASH_B,
        physical_checkpoint_sha256: HASH_C
      }
    });
    expect(recordActionExecutionProgress(executing, {
      transactionId: "execute-1",
      committedFrameCount: 2,
      worldFrame: 92,
      worldRevision: 12,
      authorityStateSha256: HASH_B,
      physicalCheckpointSha256: HASH_C
    })).toEqual(executing);
    expect(recordActionExecutionProgress(executing, {
      transactionId: "execute-1",
      committedFrameCount: 2,
      worldFrame: 92,
      worldRevision: 12,
      authorityStateSha256: HASH_B,
      physicalCheckpointSha256: HASH_A,
      updatedAt: "2026-08-03T10:00:03.000Z"
    }).active["execute-1"]!.progress.physical_checkpoint_sha256).toBe(HASH_A);
    expect(() => recordActionExecutionProgress(executing, {
      transactionId: "execute-1",
      committedFrameCount: 2,
      worldFrame: 92,
      worldRevision: 12,
      authorityStateSha256: HASH_A,
      physicalCheckpointSha256: HASH_A
    })).toThrow(/identity conflict/);
    expect(() => recordActionExecutionProgress(executing, {
      transactionId: "execute-1",
      committedFrameCount: 1,
      worldFrame: 91,
      worldRevision: 11,
      authorityStateSha256: HASH_A,
      physicalCheckpointSha256: HASH_A
    })).toThrow(/regressed/);
    expect(() => recordActionExecutionProgress(executing, {
      transactionId: "execute-1",
      committedFrameCount: 3,
      worldFrame: 94,
      worldRevision: 13,
      authorityStateSha256: HASH_A,
      physicalCheckpointSha256: HASH_A
    })).toThrow(/aligned with its committed frame count/);
  });

  it("binds a terminal receipt to the admitted intent and exact durable envelope", () => {
    const executing = progressedLedger();
    const commit = terminalCommit();
    const terminal = terminalizeActionExecution(executing, {
      transactionId: "execute-1",
      commit,
      terminalAt: "2026-08-03T10:00:03.000Z"
    });
    const identity = terminal.active["execute-1"]!.terminal!;
    expect(terminal.active["execute-1"]).toMatchObject({
      status: "terminal",
      terminal: {
        runtime_event_id: "event-1",
        goal_evidence_ref: "action:execute-1"
      }
    });
    expect(terminalizeActionExecution(terminal, {
      transactionId: "execute-1",
      commit,
      terminalAt: "2026-08-03T10:00:03.000Z"
    })).toEqual(terminal);
    expect(acknowledgeTerminalActionExecution(
      terminal,
      "execute-1",
      identity
    )).toEqual(EmptyActionExecutionLedger);

    const wrongRun = structuredClone(commit);
    wrongRun.runtime_event.run_id = "run-other";
    expect(() => terminalizeActionExecution(executing, {
      transactionId: "execute-1",
      commit: wrongRun
    })).toThrow(/conflicts with admitted intent/);
  });

  it("detects identity, progress and map-key tampering during restore", () => {
    const executing = progressedLedger();
    const changedIntent = structuredClone(executing);
    changedIntent.active["execute-1"]!.agent_id = "another-agent";
    expect(() => restoreActionExecutionLedger(changedIntent)).toThrow(/intent integrity/);

    const changedCycle = structuredClone(executing);
    changedCycle.active["execute-1"]!.cycle!.cycle_index = 2;
    expect(() => restoreActionExecutionLedger(changedCycle)).toThrow(/intent integrity/);

    const changedProgress = structuredClone(executing);
    changedProgress.active["execute-1"]!.progress.world_revision += 1;
    expect(() => restoreActionExecutionLedger(changedProgress)).toThrow(/committed frame count/);

    const changedDelegation = structuredClone(executing);
    changedDelegation.active["execute-1"]!.admission
      .tool_call_authority!.deterministic_delegation.source_input = {
        objective: "execute a different plan"
      };
    expect(() => restoreActionExecutionLedger(changedDelegation))
      .toThrow(/intent integrity/);

    const unboundDelegation = structuredClone(executing);
    unboundDelegation.active["execute-1"]!.admission
      .tool_call_authority!.deterministic_delegation.action_input_sha256 = HASH_A;
    expect(() => restoreActionExecutionLedger(unboundDelegation))
      .toThrow(/not bound to its model decision/);

    expect(ActionExecutionLedgerSchema.safeParse({
      version: 1,
      active: { other: executing.active["execute-1"] }
    }).success).toBe(false);
  });
});

function admittedLedger() {
  return stageIntent(EmptyActionExecutionLedger);
}

function stageIntent(
  ledger: typeof EmptyActionExecutionLedger,
  override: Partial<Parameters<typeof stageActionExecutionIntent>[1]> = {}
) {
  return stageActionExecutionIntent(ledger, {
    runId: "run-1",
    transactionId: "execute-1",
    agentId: "humanoid-executor",
    action: "execute_whole_body_motion",
    actionFingerprint: "execute_whole_body_motion\nhumanoid-executor\n{}",
    cycle: CYCLE,
    planningTransactionId: "plan-1",
    planId: "physical-plan-1",
    worldFrame: 90,
    worldRevision: 10,
    authorityStateSha256: HASH_A,
    physicalCheckpointSha256: HASH_A,
    decision: {
      agent_id: "humanoid-coordinator",
      agent_manifest_sha256: HASH_A,
      agent_manifest_epoch_id: "00000000-0000-4000-8000-000000000098",
      model_call_id: MODEL_CALL_ID,
      response_id: "response-execute-1",
      response_output_sha256: HASH_B,
      tool_call_id: "execute-1",
      tool_arguments_sha256: HASH_C,
      normalized_tool_arguments_sha256: HASH_B
    },
    toolCallAuthority: {
      tool_call_id: "execute-1",
      tool_name: "delegate_physics_executor",
      arguments_sha256: HASH_C,
      deterministic_delegation: {
        contract_id: "execution_gate_v1",
        source_input: {
          objective: "execute accepted plan",
          execution: {
            kind: "execute_plan",
            planning_action: "plan_whole_body_motion",
            planning_transaction_id: "plan-1"
          }
        },
        action_input_sha256: HASH_B
      }
    },
    admittedAt: "2026-08-03T10:00:00.000Z",
    ...override
  });
}

function progressedLedger() {
  return recordActionExecutionProgress(admittedLedger(), {
    transactionId: "execute-1",
    committedFrameCount: 2,
    worldFrame: 92,
    worldRevision: 12,
    authorityStateSha256: HASH_B,
    physicalCheckpointSha256: HASH_C,
    updatedAt: "2026-08-03T10:00:02.000Z"
  });
}

function terminalCommit() {
  const receipt = {
    transactionId: "execute-1",
    agentId: "humanoid-executor",
    cycle: CYCLE,
    action: "execute_whole_body_motion" as const,
    input: { planning_transaction_id: "plan-1" },
    fingerprint: "execute_whole_body_motion\nhumanoid-executor\n{}",
    accepted: true,
    code: "motion_completed",
    worldBeforeRevision: 10,
    worldAfterRevision: 12,
    frameCount: 2,
    channels: ["locomotion"],
    detail: { plan_id: "physical-plan-1" },
    committedAt: "2026-08-03T10:00:03.000Z"
  };
  const actionRecord = { ...receipt, runtime_event_id: "event-1" };
  const outbox = stageActionCommit(EmptyActionCommitOutbox, {
    transactionId: "execute-1",
    runtimeEventId: "event-1",
    actionRecord,
    goalEvidenceRef: "action:execute-1",
    goalEvidenceRecord: {
      evidence: { ref: "action:execute-1", kind: "action_receipt" },
      payload: { transaction_id: "execute-1", receipt }
    },
    runtimeEvent: {
      event_id: "event-1",
      run_id: "run-1",
      type: "humanoid_action_committed",
      at: "2026-08-03T10:00:03.000Z",
      data: { receipt: actionRecord }
    },
    stagedAt: "2026-08-03T10:00:03.000Z"
  });
  return outbox.pending["execute-1"]!;
}
