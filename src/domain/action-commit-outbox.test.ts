import { describe, expect, it } from "vitest";
import {
  ActionCommitOutboxSchema,
  EmptyActionCommitOutbox,
  acknowledgeActionCommit,
  pendingActionCommits,
  restoreActionCommitOutbox,
  stageActionCommit
} from "./action-commit-outbox.js";
import { createScenarioPhysicalWorldDelta } from "./scenario-physical-delta.js";
import { createScenarioBlockRemovalTransaction } from "./scenario-block-removal.js";
import { createScenarioChunkDeltaState } from "./scenario-chunk-delta-schema.js";
import { ScenarioSchema } from "./schema.js";

const actionRecord = {
  transactionId: "call-1",
  agentId: "humanoid-executor",
  action: "execute_whole_body_motion",
  runtime_event_id: "event-1"
} as const;
const actionReceipt = {
  transactionId: "call-1",
  agentId: "humanoid-executor",
  action: "execute_whole_body_motion"
} as const;

const runtimeEvent = {
  event_id: "event-1",
  run_id: "run-1",
  type: "humanoid_action_committed" as const,
  at: "2026-08-03T10:00:00.000Z",
  data: { receipt: actionRecord }
};
const goalEvidenceInput = {
  goalEvidenceRef: "action:call-1",
  goalEvidenceRecord: {
    evidence: { ref: "action:call-1", kind: "action_receipt" },
    payload: {
      transaction_id: "call-1",
      receipt: actionReceipt
    }
  }
} as const;

describe("action commit outbox", () => {
  it("stages an integrity-bound action and acknowledges it only after reconciliation", () => {
    const staged = stageActionCommit(EmptyActionCommitOutbox, {
      transactionId: "call-1",
      runtimeEventId: "event-1",
      actionRecord,
      ...goalEvidenceInput,
      runtimeEvent,
      stagedAt: "2026-08-03T10:00:00.000Z"
    });

    expect(pendingActionCommits(staged)).toHaveLength(1);
    expect(pendingActionCommits(staged)[0]).toMatchObject({
      transaction_id: "call-1",
      runtime_event_id: "event-1"
    });
    expect(acknowledgeActionCommit(staged, "call-1")).toEqual(EmptyActionCommitOutbox);
  });

  it("is idempotent for the exact same durable commit", () => {
    const staged = stageActionCommit(EmptyActionCommitOutbox, {
      transactionId: "call-1",
      runtimeEventId: "event-1",
      actionRecord,
      ...goalEvidenceInput,
      runtimeEvent,
      stagedAt: "2026-08-03T10:00:00.000Z"
    });
    expect(stageActionCommit(staged, {
      transactionId: "call-1",
      runtimeEventId: "event-1",
      actionRecord,
      ...goalEvidenceInput,
      runtimeEvent,
      stagedAt: "2026-08-03T10:00:01.000Z"
    })).toEqual(staged);
  });

  it("rejects transaction reuse with another event or record", () => {
    const staged = stageActionCommit(EmptyActionCommitOutbox, {
      transactionId: "call-1",
      runtimeEventId: "event-1",
      actionRecord,
      ...goalEvidenceInput,
      runtimeEvent
    });
    expect(() => stageActionCommit(staged, {
      transactionId: "call-1",
      runtimeEventId: "event-2",
      actionRecord: { ...actionRecord, runtime_event_id: "event-2" },
      ...goalEvidenceInput,
      runtimeEvent: {
        ...runtimeEvent,
        event_id: "event-2",
        data: { receipt: { ...actionRecord, runtime_event_id: "event-2" } }
      }
    })).toThrow(/transaction conflict/);
  });

  it("detects tampering during restore", () => {
    const staged = stageActionCommit(EmptyActionCommitOutbox, {
      transactionId: "call-1",
      runtimeEventId: "event-1",
      actionRecord,
      ...goalEvidenceInput,
      runtimeEvent
    });
    const corrupted = structuredClone(staged);
    corrupted.pending["call-1"]!.action_record = {
      ...actionRecord,
      agentId: "another-agent"
    };

    expect(() => restoreActionCommitOutbox(corrupted)).toThrow(/integrity hash/);

    const corruptedEvidence = structuredClone(staged);
    corruptedEvidence.pending["call-1"]!.goal_evidence_record = {
      evidence: { ref: "action:call-1" },
      transaction_id: "another-call"
    };
    expect(() => restoreActionCommitOutbox(corruptedEvidence)).toThrow(/integrity hash/);
  });

  it("rejects mismatched map, action and event identities", () => {
    const staged = stageActionCommit(EmptyActionCommitOutbox, {
      transactionId: "call-1",
      runtimeEventId: "event-1",
      actionRecord,
      ...goalEvidenceInput,
      runtimeEvent
    });
    const entry = staged.pending["call-1"]!;
    expect(ActionCommitOutboxSchema.safeParse({
      version: 1,
      pending: { "call-other": entry }
    }).success).toBe(false);
  });

  it("rejects individually hashable records that disagree about the receipt", () => {
    expect(() => stageActionCommit(EmptyActionCommitOutbox, {
      transactionId: "call-1",
      runtimeEventId: "event-1",
      actionRecord,
      goalEvidenceRef: "action:call-1",
      goalEvidenceRecord: {
        evidence: { ref: "action:call-1", kind: "action_receipt" },
        payload: {
          transaction_id: "call-1",
          receipt: { ...actionReceipt, agentId: "another-agent" }
        }
      },
      runtimeEvent
    })).toThrow(/Goal evidence receipt/);

    expect(() => stageActionCommit(EmptyActionCommitOutbox, {
      transactionId: "call-1",
      runtimeEventId: "event-1",
      actionRecord,
      ...goalEvidenceInput,
      runtimeEvent: {
        ...runtimeEvent,
        data: { receipt: { ...actionRecord, transactionId: "call-other" } }
      }
    })).toThrow(/runtime event receipt/);
  });

  it("binds a physical chunk transition to the exact execution receipt and event", () => {
    const physicalActionRecord = {
      ...actionRecord,
      worldAfterRevision: 84
    } as const;
    const physicalReceipt = {
      ...actionReceipt,
      worldAfterRevision: 84
    } as const;
    const physicalWorldDelta = createScenarioPhysicalWorldDelta({
      version: 1,
      transaction_id: "call-1",
      source_world_frame: 84,
      source_world_revision: 84,
      base_chunk_revision: 0,
      entities: [{
        id: "crate",
        position: { x: 6, y: 0.25, z: 7 },
        rotation: { x: 0, y: 0, z: 0, w: 1 },
        linear_velocity: { x: 0, y: 0, z: 0 },
        angular_velocity: { x: 0, y: 0, z: 0 },
        physical_authority: {
          source: "humanoid_mujoco",
          transaction_id: "call-1",
          world_frame: 84,
          world_revision: 84
        }
      }]
    });
    const staged = stageActionCommit(EmptyActionCommitOutbox, {
      transactionId: "call-1",
      runtimeEventId: "event-1",
      actionRecord: physicalActionRecord,
      goalEvidenceRef: "action:call-1",
      goalEvidenceRecord: {
        evidence: { ref: "action:call-1", kind: "action_receipt" },
        payload: { transaction_id: "call-1", receipt: physicalReceipt }
      },
      physicalWorldDelta,
      runtimeEvent: {
        ...runtimeEvent,
        data: {
          receipt: physicalActionRecord,
          physical_world_delta: physicalWorldDelta
        }
      }
    });

    expect(staged.pending["call-1"]).toMatchObject({
      physical_world_delta: {
        transaction_id: "call-1",
        source_world_revision: 84
      }
    });
    expect(() => stageActionCommit(EmptyActionCommitOutbox, {
      transactionId: "call-1",
      runtimeEventId: "event-1",
      actionRecord: { ...physicalActionRecord, worldAfterRevision: 85 },
      goalEvidenceRef: "action:call-1",
      goalEvidenceRecord: {
        evidence: { ref: "action:call-1", kind: "action_receipt" },
        payload: {
          transaction_id: "call-1",
          receipt: { ...physicalReceipt, worldAfterRevision: 85 }
        }
      },
      physicalWorldDelta,
      runtimeEvent: {
        ...runtimeEvent,
        data: {
          receipt: { ...physicalActionRecord, worldAfterRevision: 85 },
          physical_world_delta: physicalWorldDelta
        }
      }
    })).toThrow(/physical world delta does not match/);
  });

  it("binds block removal to the accepted semantic action and its event", () => {
    const scenario = ScenarioSchema.parse({
      title: "Outbox block fixture",
      seed: 9,
      bounds: { width: 16, depth: 16 },
      visibility_radius: 8,
      robot: { x: 2, z: 2, yaw: 0 },
      obstacles: [{
        id: "block-a",
        center: { x: 3, y: 0.5, z: 3 },
        size: { x: 1, y: 1, z: 1 }
      }],
      objects: [],
      zones: [],
      default_goal: {
        summary: "Reach.",
        predicates: [{
          type: "robot_at",
          target: { x: 3, y: 0, z: 3 },
          tolerance: 0.2
        }]
      }
    });
    const blockRemoval = createScenarioBlockRemovalTransaction({
      scenario,
      chunks: createScenarioChunkDeltaState(scenario),
      transactionId: "remove-call",
      solidId: "block-a",
      executionTransactionId: "execute-call",
      planningTransactionId: "plan-call",
      sourceWorldFrame: 40,
      sourceWorldRevision: 40,
      contactEvidence: {
        predicate_index: 0,
        predicate_type: "hand_contact_solid",
        surface_kind: "hand_surface",
        surface: "left_hand_palm_link",
        planned_stable_frames: 8,
        observed_stable_frames: 8,
        planned_minimum_normal_force_n: 5,
        observed_maximum_normal_force_n: 8
      }
    });
    const removalReceipt = {
      transactionId: "remove-call",
      agentId: "humanoid-executor",
      action: "remove_world_block",
      accepted: true,
      detail: { removal_transaction: blockRemoval }
    } as const;
    const removalRecord = {
      ...removalReceipt,
      runtime_event_id: "remove-event"
    } as const;
    const staged = stageActionCommit(EmptyActionCommitOutbox, {
      transactionId: "remove-call",
      runtimeEventId: "remove-event",
      actionRecord: removalRecord,
      goalEvidenceRef: "action:remove-call",
      goalEvidenceRecord: {
        evidence: { ref: "action:remove-call", kind: "action_receipt" },
        payload: {
          transaction_id: "remove-call",
          receipt: removalReceipt
        }
      },
      blockRemoval,
      runtimeEvent: {
        event_id: "remove-event",
        run_id: "run-1",
        type: "humanoid_action_committed",
        at: "2026-08-04T00:00:00.000Z",
        data: {
          receipt: removalRecord,
          block_removal: blockRemoval
        }
      }
    });

    expect(staged.pending["remove-call"]).toMatchObject({
      block_removal: {
        block_id: "block-a",
        execution_transaction_id: "execute-call"
      }
    });
  });
});
