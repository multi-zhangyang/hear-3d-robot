import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { goalSha256 } from "./goal-identity.js";
import {
  completeGoalEpoch,
  createGoalDAG,
  goalCandidateBySequence,
  proposeGoalCandidate,
  selectGoalCandidate,
  type GoalDAG,
  type GoalHarnessValidation,
  type GoalModelSource,
  type GoalPhysicalEvidence
} from "./goal-epoch.js";
import {
  applyGoalHistoryArchiveRecord,
  createGoalHistoryArchiveRecord,
  type GoalHistoryArchiveRecord
} from "./goal-history-archive.js";

const MANIFEST_SHA256 = "a".repeat(64);
const MANIFEST_EPOCH_ID = "00000000-0000-4000-8000-000000000001";

describe("Goal history archive", () => {
  it("keeps the checkpoint bounded across hundreds of completed model Goals", () => {
    const state = fixture();
    let dag = createGoalDAG();
    const records: GoalHistoryArchiveRecord[] = [];
    const candidateIds: string[] = [];
    for (let index = 0; index < 300; index += 1) {
      const completed = completeOneGoal(dag, state, index);
      dag = completed.dag;
      candidateIds.push(completed.candidateId);
      while (dag.epochs.length > 12) {
        const record = createGoalHistoryArchiveRecord(dag);
        records.push(record);
        dag = applyGoalHistoryArchiveRecord(dag, record);
      }
    }

    expect(dag.epochs).toHaveLength(12);
    expect(Object.keys(dag.candidates)).toHaveLength(12);
    expect(Object.keys(dag.evidence)).toHaveLength(36);
    expect(dag.archive).toMatchObject({ record_count: 288 });
    expect(dag.next_epoch_index).toBe(300);
    expect(dag.next_candidate_sequence).toBe(301);
    expect(records).toHaveLength(288);
    expect(records[0]).toMatchObject({
      sequence: 1,
      candidate_sequence: 1,
      previous_record_sha256: null,
      candidate: { candidate_id: candidateIds[0] }
    });
    expect(records.at(-1)).toMatchObject({
      sequence: 288,
      candidate_sequence: 288,
      previous_record_sha256: records.at(-2)!.record_sha256,
      candidate: { candidate_id: candidateIds[287] }
    });
  }, 30_000);

  it("retains an archived completed dependency until its proposed child activates", () => {
    const state = fixture();
    let dag = completeOneGoal(createGoalDAG(), state, 0).dag;
    const parentId = goalCandidateBySequence(dag, 1)!.candidate_id;
    dag = proposeOnly(dag, state, 1, [parentId]);
    const childId = goalCandidateBySequence(dag, 2)!.candidate_id;
    for (let index = 2; index < 15; index += 1) {
      dag = completeOneGoal(dag, state, index).dag;
    }
    const record = createGoalHistoryArchiveRecord(dag);
    dag = applyGoalHistoryArchiveRecord(dag, record);

    expect(dag.archive.retained_candidate_ids).toEqual([parentId]);
    expect(dag.candidates[parentId]?.status).toBe("completed");
    expect(dag.candidates[childId]?.status).toBe("proposed");

    const selectionRevision = 1000;
    const selectionEvidence = evidence(`selection:${selectionRevision}`, selectionRevision);
    state.evidence.set(selectionEvidence.ref, selectionEvidence);
    expect(() => selectGoalCandidate(dag, {
      candidate_id: childId,
      selected_by: source("select-child"),
      selection_evidence_refs: [selectionEvidence.ref],
      created_world_revision: selectionRevision
    }, state.harness)).not.toThrow();
  });

  it("replays an append-before-checkpoint crash window without changing identity", () => {
    const state = fixture();
    let checkpoint = createGoalDAG();
    for (let index = 0; index < 13; index += 1) {
      checkpoint = completeOneGoal(checkpoint, state, index).dag;
    }
    const appendedRecord = createGoalHistoryArchiveRecord(checkpoint);

    const firstRecovery = applyGoalHistoryArchiveRecord(checkpoint, appendedRecord);
    const repeatedRecovery = applyGoalHistoryArchiveRecord(checkpoint, appendedRecord);
    expect(repeatedRecovery).toEqual(firstRecovery);
    expect(firstRecovery).toMatchObject({
      archive: {
        record_count: 1,
        last_record_sha256: appendedRecord.record_sha256
      },
      next_epoch_index: 13
    });
    expect(firstRecovery.epochs).toHaveLength(12);
    expect(() => applyGoalHistoryArchiveRecord(firstRecovery, appendedRecord)).toThrow(
      "cannot advance checkpoint"
    );
  });
});

function completeOneGoal(
  dag: GoalDAG,
  state: ReturnType<typeof fixture>,
  index: number
): { dag: GoalDAG; candidateId: string } {
  let next = proposeOnly(dag, state, index, []);
  const candidate = goalCandidateBySequence(next, next.next_candidate_sequence - 1)!;
  const selectionRevision = index * 3 + 2;
  const selectionEvidence = evidence(`selection:${index}`, selectionRevision);
  state.evidence.set(selectionEvidence.ref, selectionEvidence);
  next = selectGoalCandidate(next, {
    candidate_id: candidate.candidate_id,
    selected_by: source(`select:${index}`),
    selection_evidence_refs: [selectionEvidence.ref],
    created_world_revision: selectionRevision
  }, state.harness);
  const completionRevision = index * 3 + 3;
  const completionEvidence = evidence(
    `evaluation:${index}`,
    completionRevision,
    "goal_evaluation",
    candidate.content_sha256
  );
  state.evidence.set(completionEvidence.ref, completionEvidence);
  next = completeGoalEpoch(next, {
    resolution_evidence_refs: [completionEvidence.ref],
    resolved_world_revision: completionRevision
  }, state.harness);
  return { dag: next, candidateId: candidate.candidate_id };
}

function proposeOnly(
  dag: GoalDAG,
  state: ReturnType<typeof fixture>,
  index: number,
  dependencies: string[]
): GoalDAG {
  const revision = index * 3 + 1;
  const proposalEvidence = evidence(`observation:${index}`, revision);
  state.evidence.set(proposalEvidence.ref, proposalEvidence);
  return proposeGoalCandidate(dag, {
    proposal_id: `proposal-${index}`,
    source: source(`propose:${index}`),
    goal: {
      summary: `自主目标 ${index}`,
      predicates: [{
        type: "robot_at",
        target: { x: index + 1, y: 0, z: index + 2 },
        tolerance: 0.3
      }]
    },
    mission_link: "持续自主活动",
    dependency_candidate_ids: dependencies,
    proposal_evidence_refs: [proposalEvidence.ref],
    created_world_revision: revision
  }, state.harness);
}

function fixture() {
  const artifacts = new Map<string, GoalPhysicalEvidence>();
  const harness: GoalHarnessValidation = {
    authorized_model_sources: [{
      agent_id: "humanoid-goal-manager",
      agent_manifest_sha256: MANIFEST_SHA256,
      agent_manifest_epoch_id: MANIFEST_EPOCH_ID
    }],
    is_model_call_authoritative: () => true,
    evidence_by_ref: (ref) => artifacts.get(ref),
    is_predicate_observable: () => true
  };
  return { evidence: artifacts, harness };
}

function source(identity: string): GoalModelSource {
  const digest = createHash("sha256").update(identity).digest("hex");
  return {
    agent_id: "humanoid-goal-manager",
    agent_manifest_sha256: MANIFEST_SHA256,
    agent_manifest_epoch_id: MANIFEST_EPOCH_ID,
    model_call_id: [
      digest.slice(0, 8),
      digest.slice(8, 12),
      `4${digest.slice(13, 16)}`,
      `8${digest.slice(17, 20)}`,
      digest.slice(20, 32)
    ].join("-"),
    response_id: `response-${identity}`,
    response_output_sha256: createHash("sha256").update(`response-${identity}`).digest("hex"),
    tool_call_id: `tool-${identity}`,
    tool_arguments_sha256: createHash("sha256").update(`args-${identity}`).digest("hex")
  };
}

function evidence(
  ref: string,
  revision: number,
  kind: GoalPhysicalEvidence["kind"] = "world_observation",
  goalContentSha256 = goalSha256({
    summary: "unused",
    predicates: [{
      type: "robot_at",
      target: { x: 0, y: 0, z: 0 },
      tolerance: 0.1
    }]
  })
): GoalPhysicalEvidence {
  const base = {
    ref,
    content_sha256: createHash("sha256").update(`${ref}:${revision}`).digest("hex"),
    world_frame: revision * 4,
    world_revision: revision
  };
  return kind === "goal_evaluation"
    ? { ...base, kind, goal_content_sha256: goalContentSha256 }
    : { ...base, kind };
}
