import { createHash } from "node:crypto";
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

const MANIFEST_SHA256 = "a".repeat(64);
const MANIFEST_EPOCH_ID = "00000000-0000-4000-8000-000000000001";

export function createCompletedGoalDAG(
  count: number,
  candidatesPerGoal = 1
): GoalDAG {
  const state = goalHistoryFixture();
  let dag = createGoalDAG();
  for (let index = 0; index < count; index += 1) {
    dag = completeFixtureGoal(dag, state, index, candidatesPerGoal).dag;
  }
  return dag;
}

function completeFixtureGoal(
  dag: GoalDAG,
  state: ReturnType<typeof goalHistoryFixture>,
  index: number,
  candidatesPerGoal: number
): { dag: GoalDAG; candidateId: string } {
  let next = dag;
  for (let variant = 0; variant < candidatesPerGoal; variant += 1) {
    next = proposeFixtureGoal(next, state, index, variant, []);
  }
  const candidate = goalCandidateBySequence(
    next,
    next.next_candidate_sequence - candidatesPerGoal
  )!;
  const selectionRevision = index * 3 + 2;
  const selectionEvidence = fixtureEvidence(`selection:${index}`, selectionRevision);
  state.evidence.set(selectionEvidence.ref, selectionEvidence);
  next = selectGoalCandidate(next, {
    candidate_id: candidate.candidate_id,
    selected_by: fixtureModelSource(`select:${index}`),
    selection_evidence_refs: [selectionEvidence.ref],
    created_world_revision: selectionRevision
  }, state.harness);
  const completionRevision = index * 3 + 3;
  const completionEvidence = fixtureEvidence(
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

function proposeFixtureGoal(
  dag: GoalDAG,
  state: ReturnType<typeof goalHistoryFixture>,
  index: number,
  variant: number,
  dependencies: string[]
): GoalDAG {
  const revision = index * 3 + 1;
  const proposalEvidence = fixtureEvidence(`observation:${index}`, revision);
  state.evidence.set(proposalEvidence.ref, proposalEvidence);
  return proposeGoalCandidate(dag, {
    proposal_id: `proposal-${index}-${variant}`,
    source: fixtureModelSource(`propose:${index}`),
    goal: {
      summary: `自主目标 ${index}-${variant}`,
      predicates: [{
        type: "robot_at",
        target: { x: index + 1 + variant / 10, y: 0, z: index + 2 },
        tolerance: 0.3
      }]
    },
    mission_link: "持续自主活动",
    dependency_candidate_ids: dependencies,
    proposal_evidence_refs: [proposalEvidence.ref],
    created_world_revision: revision
  }, state.harness);
}

function goalHistoryFixture() {
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

function fixtureModelSource(identity: string): GoalModelSource {
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

function fixtureEvidence(
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
