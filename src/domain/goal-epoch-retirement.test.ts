import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import {
  createGoalDAG,
  proposeGoalCandidate,
  selectGoalCandidate,
  type GoalDAG,
  type GoalHarnessValidation,
  type GoalModelSource,
  type GoalPhysicalEvidence
} from "./goal-epoch.js";
import {
  GOAL_RETIREMENT_STATUSES,
  retireGoalEpoch
} from "./goal-epoch-retirement.js";

const MANIFEST = "d".repeat(64);
const MANIFEST_EPOCH = "00000000-0000-4000-8000-000000000002";

describe("Goal epoch retirement", () => {
  it.each(GOAL_RETIREMENT_STATUSES)(
    "retires an active epoch as %s without selecting a replacement",
    (status) => {
      const fixture = activeFixture(status === "blocked" ? "action_receipt" : "world_checkpoint");
      const retired = retireGoalEpoch(fixture.dag, {
        status,
        retired_by: source(`retire-${status}`),
        reason: `physical evidence says ${status}`,
        resolution_evidence_refs: [fixture.retirementEvidence.ref],
        resolved_world_revision: 3
      }, fixture.harness);

      expect(retired.status).toBe("awaiting_model_selection");
      expect(retired.current_epoch_id).toBeNull();
      expect(retired.epochs).toHaveLength(1);
      expect(retired.epochs[0]).toMatchObject({
        status,
        retired_by: source(`retire-${status}`),
        retirement_reason: `physical evidence says ${status}`,
        resolved_world_revision: 3,
        physical_evidence_refs: { resolution: [fixture.retirementEvidence.ref] }
      });
      expect(Object.values(retired.candidates)).toHaveLength(1);
      expect(Object.values(retired.candidates)[0]?.status).toBe(status);
    }
  );

  it("refuses to mark a Goal blocked without an exact physical action receipt", () => {
    const fixture = activeFixture("world_checkpoint");
    expect(() => retireGoalEpoch(fixture.dag, {
      status: "blocked",
      retired_by: source("retire-without-action"),
      reason: "not reachable",
      resolution_evidence_refs: [fixture.retirementEvidence.ref],
      resolved_world_revision: 3
    }, fixture.harness)).toThrowError(expect.objectContaining({
      code: "blocked_evidence_missing"
    }));
  });
});

function activeFixture(retirementKind: GoalPhysicalEvidence["kind"]): {
  dag: GoalDAG;
  harness: GoalHarnessValidation;
  retirementEvidence: GoalPhysicalEvidence;
} {
  const records = [
    evidence("world:1", "world_checkpoint", 1),
    evidence("world:2", "world_checkpoint", 2),
    evidence("retirement:3", retirementKind, 3)
  ];
  const byRef = new Map(records.map((entry) => [entry.ref, entry]));
  const harness: GoalHarnessValidation = {
    authorized_model_sources: [{
      agent_id: "humanoid-goal-manager",
      agent_manifest_sha256: MANIFEST,
      agent_manifest_epoch_id: MANIFEST_EPOCH
    }],
    is_model_call_authoritative: () => true,
    evidence_by_ref: (ref) => byRef.get(ref),
    is_predicate_observable: () => true
  };
  let dag = proposeGoalCandidate(createGoalDAG(), {
    proposal_id: "candidate",
    source: source("propose"),
    goal: {
      summary: "前往目标",
      predicates: [{
        type: "robot_at",
        target: { x: 1, y: 0, z: 2 },
        tolerance: 0.3
      }]
    },
    mission_link: "推进长期任务",
    dependency_candidate_ids: [],
    proposal_evidence_refs: ["world:1"],
    created_world_revision: 1
  }, harness);
  const candidate = Object.values(dag.candidates)[0]!;
  dag = selectGoalCandidate(dag, {
    candidate_id: candidate.candidate_id,
    selected_by: source("select"),
    selection_evidence_refs: ["world:2"],
    created_world_revision: 2
  }, harness);
  return { dag, harness, retirementEvidence: records[2]! };
}

function source(label: string): GoalModelSource {
  const digest = createHash("sha256").update(label).digest("hex");
  return {
    agent_id: "humanoid-goal-manager",
    agent_manifest_sha256: MANIFEST,
    agent_manifest_epoch_id: MANIFEST_EPOCH,
    model_call_id: [
      digest.slice(0, 8),
      digest.slice(8, 12),
      `4${digest.slice(13, 16)}`,
      `8${digest.slice(17, 20)}`,
      digest.slice(20, 32)
    ].join("-"),
    response_id: `response-${label}`,
    response_output_sha256: createHash("sha256")
      .update(`response-${label}`)
      .digest("hex"),
    tool_call_id: `tool-${label}`,
    tool_arguments_sha256: createHash("sha256")
      .update(`arguments-${label}`)
      .digest("hex")
  };
}

function evidence(
  ref: string,
  kind: Exclude<GoalPhysicalEvidence["kind"], "goal_evaluation">,
  revision: number
): GoalPhysicalEvidence {
  return {
    ref,
    kind,
    content_sha256: createHash("sha256").update(ref).digest("hex"),
    world_frame: revision * 4,
    world_revision: revision
  };
}
