import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { modelPayloadSha256 } from "./model-call-authority.js";
import {
  GoalDAGSchema,
  GoalDAGValidationError,
  completeGoalEpoch,
  createGoalDAG,
  proposeGoalCandidate,
  restoreGoalDAG,
  selectGoalCandidate,
  type GoalDAG,
  type GoalHarnessValidation,
  type GoalModelSource,
  type GoalPhysicalEvidence
} from "./goal-epoch.js";

const MANIFEST_SHA256 = "a".repeat(64);
const MANIFEST_EPOCH_ID = "00000000-0000-4000-8000-000000000001";

function modelSource(modelCallId: string): GoalModelSource {
  const digest = createHash("sha256").update(modelCallId).digest("hex");
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
    response_id: `response-${modelCallId}`,
    response_output_sha256: createHash("sha256")
      .update(`response-${modelCallId}`)
      .digest("hex"),
    tool_call_id: `tool-${modelCallId}`,
    tool_arguments_sha256: createHash("sha256")
      .update(`arguments-${modelCallId}`)
      .digest("hex")
  };
}

function physicalEvidence(
  ref: string,
  worldRevision: number,
  kind: GoalPhysicalEvidence["kind"] = "world_observation",
  goalContentSha256?: string
): GoalPhysicalEvidence {
  const common = {
    ref,
    content_sha256: Buffer.from(`${ref}:${worldRevision}`)
      .toString("hex")
      .padEnd(64, "0")
      .slice(0, 64),
    world_frame: worldRevision * 4,
    world_revision: worldRevision
  };
  return kind === "goal_evaluation"
    ? {
        ...common,
        kind,
        goal_content_sha256: goalContentSha256 ?? "0".repeat(64)
      }
    : { ...common, kind };
}

function testHarness(input: {
  evidence?: GoalPhysicalEvidence[];
  observable?: boolean;
  manifestSha256?: string;
  authoritativeModelCall?: boolean;
  authoritativeToolName?:
    | "submit_goal_candidates"
    | "select_goal_candidate"
    | "retire_goal_epoch";
} = {}): {
  harness: GoalHarnessValidation;
  evidence: Map<string, GoalPhysicalEvidence>;
} {
  const evidence = new Map((input.evidence ?? []).map((entry) => [entry.ref, entry]));
  return {
    evidence,
    harness: {
      authorized_model_sources: [{
        agent_id: "humanoid-goal-manager",
        agent_manifest_sha256: input.manifestSha256 ?? MANIFEST_SHA256,
        agent_manifest_epoch_id: MANIFEST_EPOCH_ID
      }],
      is_model_call_authoritative: (_source, expectedToolName) => (
        (input.authoritativeModelCall ?? true)
          && (input.authoritativeToolName === undefined
            || input.authoritativeToolName === expectedToolName)
      ),
      evidence_by_ref: (ref) => evidence.get(ref),
      is_predicate_observable: () => input.observable ?? true
    }
  };
}

function propose(input: {
  dag?: GoalDAG;
  harness: GoalHarnessValidation;
  proposalId?: string;
  modelCallId?: string;
  evidenceRef: string;
  revision: number;
  dependencies?: string[];
  source?: GoalModelSource;
}): GoalDAG {
  return proposeGoalCandidate(input.dag ?? createGoalDAG(), {
    proposal_id: input.proposalId ?? "candidate-a",
    source: input.source ?? modelSource(input.modelCallId ?? "call-propose-a"),
    goal: {
      summary: "前往当前可见位置",
      predicates: [{
        type: "robot_at",
        target: { x: 3, y: 0, z: 4 },
        tolerance: 0.3
      }]
    },
    mission_link: "推进长期任务",
    dependency_candidate_ids: input.dependencies ?? [],
    proposal_evidence_refs: [input.evidenceRef],
    created_world_revision: input.revision
  }, input.harness);
}

function onlyCandidate(dag: GoalDAG) {
  const candidates = Object.values(dag.candidates);
  expect(candidates).toHaveLength(1);
  return candidates[0]!;
}

describe("persistent model goal epochs", () => {
  it("upgrades the complete v1 DAG into the bounded v2 working format", () => {
    const current = createGoalDAG();
    const {
      candidate_sequences: _candidateSequences,
      next_candidate_sequence: _nextCandidateSequence,
      archive: _archive,
      state_sha256: _stateSha256,
      ...legacyContents
    } = current;
    const legacy = { ...legacyContents, version: 1 as const };
    expect(GoalDAGSchema.parse({
      ...legacy,
      state_sha256: modelPayloadSha256(legacy)
    })).toMatchObject({
      version: 2,
      candidate_sequences: {},
      next_candidate_sequence: 1,
      archive: {
        record_count: 0,
        last_record_sha256: null,
        last_epoch_id: null,
        retained_candidate_ids: []
      }
    });
  });

  it("persists model provenance, candidate hashes, dependencies and physical evidence", () => {
    const proposalEvidence = physicalEvidence("observation:12", 12);
    const { harness } = testHarness({ evidence: [proposalEvidence] });
    const empty = createGoalDAG();
    const dag = propose({ harness, evidenceRef: proposalEvidence.ref, revision: 12 });
    const candidate = onlyCandidate(dag);

    expect(empty).toMatchObject({
      status: "awaiting_model_selection",
      candidates: {},
      epochs: [],
      current_epoch_id: null
    });
    expect(dag.status).toBe("awaiting_model_selection");
    expect(dag.epochs).toEqual([]);
    expect(candidate).toMatchObject({
      proposal_id: "candidate-a",
      source: modelSource("call-propose-a"),
      dependency_candidate_ids: [],
      status: "proposed",
      physical_evidence_refs: {
        proposal: [proposalEvidence.ref],
        resolution: []
      },
      created_world_revision: 12,
      resolved_world_revision: null
    });
    expect(candidate.candidate_id).toMatch(/^goal-candidate:[a-f0-9]{64}$/);
    expect(candidate.identity_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(candidate.content_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(candidate.integrity_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(dag.evidence[proposalEvidence.ref]).toEqual(proposalEvidence);
    expect(GoalDAGSchema.parse(dag)).toEqual(dag);
  });

  it("rejects invalid or unobservable model goals without substituting a goal", () => {
    const evidence = physicalEvidence("observation:2", 2);
    const observable = testHarness({ evidence: [evidence] });
    const empty = createGoalDAG();

    expect(() => proposeGoalCandidate(empty, {
      proposal_id: "invalid",
      source: modelSource("call-invalid"),
      goal: { summary: "", predicates: [] },
      mission_link: "推进长期任务",
      dependency_candidate_ids: [],
      proposal_evidence_refs: [evidence.ref],
      created_world_revision: 2
    }, observable.harness)).toThrow();

    const hidden = testHarness({ evidence: [evidence], observable: false });
    expect(() => propose({
      harness: hidden.harness,
      evidenceRef: evidence.ref,
      revision: 2
    })).toThrowError(expect.objectContaining({ code: "predicate_not_observable" }));
    expect(empty).toEqual(createGoalDAG());
  });

  it("rejects a source outside the exact Agent manifest recovery domain", () => {
    const evidence = physicalEvidence("observation:3", 3);
    const { harness } = testHarness({ evidence: [evidence] });
    const foreign = {
      ...modelSource("call-foreign"),
      agent_manifest_sha256: "b".repeat(64)
    };

    expect(() => propose({
      harness,
      source: foreign,
      evidenceRef: evidence.ref,
      revision: 3
    })).toThrowError(expect.objectContaining({ code: "unauthorized_model_source" }));
  });

  it("requires the Agent runtime to attest the originating model call", () => {
    const evidence = physicalEvidence("observation:call", 3);
    const { harness } = testHarness({
      evidence: [evidence],
      authoritativeModelCall: false
    });

    expect(() => propose({
      harness,
      evidenceRef: evidence.ref,
      revision: 3
    })).toThrowError(expect.objectContaining({ code: "unauthorized_model_source" }));
  });

  it("binds each model source to the exact Goal tool lifecycle", () => {
    const evidence = physicalEvidence("observation:wrong-tool", 3);
    const { harness } = testHarness({
      evidence: [evidence],
      authoritativeToolName: "select_goal_candidate"
    });

    expect(() => propose({
      harness,
      evidenceRef: evidence.ref,
      revision: 3
    })).toThrowError(expect.objectContaining({ code: "unauthorized_model_source" }));
  });

  it("requires every model-declared dependency to exist", () => {
    const evidence = physicalEvidence("observation:4", 4);
    const { harness } = testHarness({ evidence: [evidence] });

    expect(() => propose({
      harness,
      evidenceRef: evidence.ref,
      revision: 4,
      dependencies: [`goal-candidate:${"b".repeat(64)}`]
    })).toThrowError(expect.objectContaining({ code: "missing_dependency" }));
  });

  it("does not activate a candidate until a model explicitly selects it", () => {
    const proposalEvidence = physicalEvidence("observation:5", 5);
    const selectionEvidence = physicalEvidence("checkpoint:6", 6, "world_checkpoint");
    const setup = testHarness({ evidence: [proposalEvidence, selectionEvidence] });
    const proposed = propose({
      harness: setup.harness,
      evidenceRef: proposalEvidence.ref,
      revision: 5
    });
    const candidate = onlyCandidate(proposed);

    expect(proposed.status).toBe("awaiting_model_selection");
    expect(proposed.current_epoch_id).toBeNull();
    const active = selectGoalCandidate(proposed, {
      candidate_id: candidate.candidate_id,
      selected_by: modelSource("call-select-a"),
      selection_evidence_refs: [selectionEvidence.ref],
      created_world_revision: 6
    }, setup.harness);

    expect(active).toMatchObject({
      status: "active",
      next_epoch_index: 1
    });
    expect(active.current_epoch_id).toBe(active.epochs[0]?.epoch_id);
    expect(active.epochs[0]).toMatchObject({
      epoch_index: 0,
      previous_epoch_id: null,
      candidate_id: candidate.candidate_id,
      candidate_source: candidate.source,
      selected_by: modelSource("call-select-a"),
      candidate_identity_sha256: candidate.identity_sha256,
      candidate_content_sha256: candidate.content_sha256,
      dependency_candidate_ids: [],
      status: "active",
      physical_evidence_refs: {
        selection: [selectionEvidence.ref],
        resolution: []
      },
      created_world_revision: 6,
      resolved_world_revision: null
    });
  });

  it("resolves one model proposal slate as one transactional selection", () => {
    const proposalEvidence = physicalEvidence("observation:slate", 7);
    const selectionEvidence = physicalEvidence("checkpoint:slate", 8, "world_checkpoint");
    const setup = testHarness({ evidence: [proposalEvidence, selectionEvidence] });
    const slateSource = modelSource("call-propose-slate");
    let dag = createGoalDAG();
    for (const proposalId of ["slate-a", "slate-b", "slate-c"]) {
      dag = propose({
        dag,
        harness: setup.harness,
        proposalId,
        source: slateSource,
        evidenceRef: proposalEvidence.ref,
        revision: 7
      });
    }
    const selected = Object.values(dag.candidates).find(
      (candidate) => candidate.proposal_id === "slate-b"
    )!;
    dag = selectGoalCandidate(dag, {
      candidate_id: selected.candidate_id,
      selected_by: modelSource("call-select-slate"),
      selection_evidence_refs: [selectionEvidence.ref],
      created_world_revision: 8
    }, setup.harness);

    expect(dag.candidates[selected.candidate_id]?.status).toBe("active");
    expect(Object.values(dag.candidates).filter(
      (candidate) => candidate.status === "expired"
    )).toEqual(expect.arrayContaining([
      expect.objectContaining({
        proposal_id: "slate-a",
        resolved_world_revision: 8,
        physical_evidence_refs: expect.objectContaining({
          resolution: [selectionEvidence.ref]
        })
      }),
      expect.objectContaining({
        proposal_id: "slate-c",
        resolved_world_revision: 8,
        physical_evidence_refs: expect.objectContaining({
          resolution: [selectionEvidence.ref]
        })
      })
    ]));
    expect(restoreGoalDAG(dag, setup.harness)).toEqual(dag);
  });

  it("waits for another model selection after completion instead of choosing a goal", () => {
    const evidence = [
      physicalEvidence("observation:first", 10),
      physicalEvidence("checkpoint:first", 11, "world_checkpoint"),
      physicalEvidence("observation:next", 14),
      physicalEvidence("receipt:first", 13, "action_receipt"),
      physicalEvidence("checkpoint:next", 15, "world_checkpoint")
    ];
    const setup = testHarness({ evidence });
    let dag = propose({
      harness: setup.harness,
      evidenceRef: "observation:first",
      revision: 10,
      proposalId: "first"
    });
    const firstCandidate = onlyCandidate(dag);
    setup.evidence.set(
      "evaluation:first",
      physicalEvidence(
        "evaluation:first",
        13,
        "goal_evaluation",
        firstCandidate.content_sha256
      )
    );
    dag = selectGoalCandidate(dag, {
      candidate_id: firstCandidate.candidate_id,
      selected_by: modelSource("call-select-first"),
      selection_evidence_refs: ["checkpoint:first"],
      created_world_revision: 11
    }, setup.harness);
    expect(() => selectGoalCandidate(dag, {
      candidate_id: firstCandidate.candidate_id,
      selected_by: modelSource("call-select-too-early"),
      selection_evidence_refs: ["observation:next"],
      created_world_revision: 14
    }, setup.harness)).toThrowError(expect.objectContaining({ code: "goal_epoch_active" }));

    expect(() => completeGoalEpoch(dag, {
      resolution_evidence_refs: ["receipt:first"],
      resolved_world_revision: 13
    }, setup.harness)).toThrowError(expect.objectContaining({
      code: "goal_evaluation_missing"
    }));

    dag = completeGoalEpoch(dag, {
      resolution_evidence_refs: ["evaluation:first"],
      resolved_world_revision: 13
    }, setup.harness);

    expect(dag.status).toBe("awaiting_model_selection");
    expect(dag.current_epoch_id).toBeNull();
    expect(dag.epochs).toHaveLength(1);
    expect(dag.epochs[0]).toMatchObject({
      status: "completed",
      resolved_world_revision: 13,
      physical_evidence_refs: { resolution: ["evaluation:first"] }
    });
    dag = propose({
      dag,
      harness: setup.harness,
      proposalId: "next",
      modelCallId: "call-propose-next",
      evidenceRef: "observation:next",
      revision: 14,
      dependencies: [firstCandidate.candidate_id]
    });
    const nextCandidate = Object.values(dag.candidates)
      .find((candidate) => candidate.proposal_id === "next")!;
    expect(dag.candidates[nextCandidate.candidate_id]?.status).toBe("proposed");

    dag = selectGoalCandidate(dag, {
      candidate_id: nextCandidate.candidate_id,
      selected_by: modelSource("call-select-next"),
      selection_evidence_refs: ["checkpoint:next"],
      created_world_revision: 15
    }, setup.harness);
    expect(dag.epochs[1]).toMatchObject({
      epoch_index: 1,
      previous_epoch_id: dag.epochs[0]?.epoch_id,
      candidate_id: nextCandidate.candidate_id,
      dependency_candidate_ids: [firstCandidate.candidate_id],
      status: "active"
    });
  });

  it("refuses to propose a candidate before all dependencies complete", () => {
    const evidence = [
      physicalEvidence("observation:parent", 20),
      physicalEvidence("observation:child", 20),
      physicalEvidence("checkpoint:child", 21, "world_checkpoint")
    ];
    const setup = testHarness({ evidence });
    let dag = propose({
      harness: setup.harness,
      proposalId: "parent",
      evidenceRef: "observation:parent",
      revision: 20
    });
    const parent = onlyCandidate(dag);
    expect(() => propose({
      dag,
      harness: setup.harness,
      proposalId: "child",
      modelCallId: "call-child",
      evidenceRef: "observation:child",
      revision: 20,
      dependencies: [parent.candidate_id]
    })).toThrowError(expect.objectContaining({
      code: "dependency_not_completed"
    }));
  });

  it("rejects candidate or DAG hash tampering during recovery", () => {
    const evidence = physicalEvidence("observation:30", 30);
    const setup = testHarness({ evidence: [evidence] });
    const dag = propose({
      harness: setup.harness,
      evidenceRef: evidence.ref,
      revision: 30
    });
    const candidate = onlyCandidate(dag);
    const changedCandidate = structuredClone(dag);
    changedCandidate.candidates[candidate.candidate_id]!.goal.summary = "被篡改的目标";
    expect(() => restoreGoalDAG(changedCandidate, setup.harness)).toThrow();

    const changedState = structuredClone(dag);
    changedState.next_epoch_index = 9;
    expect(() => restoreGoalDAG(changedState, setup.harness)).toThrow();
  });

  it("requires exact external physical evidence on recovery", () => {
    const evidence = physicalEvidence("observation:40", 40);
    const setup = testHarness({ evidence: [evidence] });
    const dag = propose({
      harness: setup.harness,
      evidenceRef: evidence.ref,
      revision: 40
    });

    setup.evidence.set(evidence.ref, {
      ...evidence,
      content_sha256: "f".repeat(64)
    });
    expect(() => restoreGoalDAG(dag, setup.harness)).toThrowError(
      expect.objectContaining({ code: "evidence_mismatch" })
    );
    setup.evidence.delete(evidence.ref);
    expect(() => restoreGoalDAG(dag, setup.harness)).toThrowError(
      expect.objectContaining({ code: "evidence_unavailable" })
    );
  });

  it("rejects recovery under a different Agent manifest", () => {
    const evidence = physicalEvidence("observation:50", 50);
    const original = testHarness({ evidence: [evidence] });
    const dag = propose({
      harness: original.harness,
      evidenceRef: evidence.ref,
      revision: 50
    });
    const changed = testHarness({
      evidence: [evidence],
      manifestSha256: "c".repeat(64)
    });

    expect(() => restoreGoalDAG(dag, changed.harness)).toThrowError(
      expect.objectContaining({ code: "unauthorized_model_source" })
    );
  });

  it("requires evidence from the exact lifecycle world revision", () => {
    const stale = physicalEvidence("observation:stale", 59);
    const setup = testHarness({ evidence: [stale] });

    expect(() => propose({
      harness: setup.harness,
      evidenceRef: stale.ref,
      revision: 60
    })).toThrowError(expect.objectContaining({ code: "evidence_revision_mismatch" }));
  });

  it("uses typed validation errors for harness policy failures", () => {
    const evidence = physicalEvidence("observation:70", 70);
    const setup = testHarness({ evidence: [evidence], observable: false });
    try {
      propose({ harness: setup.harness, evidenceRef: evidence.ref, revision: 70 });
      throw new Error("expected proposal rejection");
    } catch (error) {
      expect(error).toBeInstanceOf(GoalDAGValidationError);
    }
  });
});
