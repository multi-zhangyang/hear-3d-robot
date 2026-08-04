import { describe, expect, it } from "vitest";
import { goalSha256 } from "../../domain/goal-identity.js";
import type { HumanoidRunCheckpoint } from "../../domain/humanoid-run.js";
import type { Goal } from "../../domain/schema.js";
import { createGoalEvaluationEvidence } from "./goal-evidence.js";
import { resolveHumanoidMissionCompletion } from "./mission-completion-evidence.js";

const goal: Goal = {
  summary: "到达庭院出口",
  predicates: [{
    type: "robot_at",
    target: { x: 4, y: 0, z: 7 },
    tolerance: 0.3
  }]
};

describe("durable Mission Goal completion", () => {
  it("resolves the exact completed Mission Goal through its physical evaluation artifact", () => {
    const { checkpoint, artifact } = completedMission();

    expect(resolveHumanoidMissionCompletion(checkpoint, [artifact])).toEqual({
      epoch_id: "goal-epoch:mission",
      candidate_id: "goal-candidate:mission",
      goal_evaluation_ref: artifact.evidence.ref,
      world_frame: 24,
      world_revision: 6,
      checker: expect.objectContaining({ success: true, goal })
    });
  });

  it("accepts a different summary only when every physical predicate is identical", () => {
    const candidateGoal: Goal = {
      ...goal,
      summary: "以真实导航进入庭院出口"
    };
    const { checkpoint, artifact } = completedMission(candidateGoal);

    expect(resolveHumanoidMissionCompletion(checkpoint, [artifact])).toMatchObject({
      checker: { success: true, goal: candidateGoal }
    });
  });

  it("fails closed when a completed Mission Goal lost its durable evaluation", () => {
    const { checkpoint } = completedMission();

    expect(() => resolveHumanoidMissionCompletion(checkpoint, []))
      .toThrow(/no durable successful evaluation/);
  });

  it("does not treat a completed subgoal as the Mission Goal", () => {
    const { checkpoint, artifact } = completedMission();
    const otherMission: Goal = {
      summary: "将物体放入目标区",
      predicates: [{
        type: "object_at",
        object_id: "crate",
        target: { x: 5, y: 0.5, z: 5 },
        tolerance: 0.2
      }]
    };

    expect(resolveHumanoidMissionCompletion({
      ...checkpoint,
      mission_goal: otherMission
    }, [artifact])).toBeNull();
  });
});

function completedMission(candidateGoal: Goal = goal): {
  checkpoint: Pick<HumanoidRunCheckpoint, "mission_goal" | "goal_dag">;
  artifact: ReturnType<typeof createGoalEvaluationEvidence>;
} {
  const epochId = "goal-epoch:mission";
  const candidateId = "goal-candidate:mission";
  const checker = {
    success: true,
    goal: candidateGoal,
    worldFrame: 24,
    worldRevision: 6,
    checks: [{ name: "1:robot_at", passed: true, actual: { distance: 0.1 } }],
    checkedAt: "2026-08-04T00:00:00.000Z"
  };
  const artifact = createGoalEvaluationEvidence({
    epochId,
    goalContentSha256: goalSha256(candidateGoal),
    worldFrame: checker.worldFrame,
    worldRevision: checker.worldRevision,
    evaluation: checker
  });
  const checkpoint = {
    mission_goal: goal,
    goal_dag: {
      candidates: {
        [candidateId]: {
          candidate_id: candidateId,
          status: "completed",
          content_sha256: goalSha256(candidateGoal),
          goal: candidateGoal
        }
      },
      epochs: [{
        epoch_id: epochId,
        candidate_id: candidateId,
        status: "completed",
        resolved_world_revision: checker.worldRevision,
        physical_evidence_refs: {
          selection: [],
          resolution: [artifact.evidence.ref]
        }
      }],
      evidence: {
        [artifact.evidence.ref]: artifact.evidence
      }
    }
  } as unknown as Pick<HumanoidRunCheckpoint, "mission_goal" | "goal_dag">;
  return { checkpoint, artifact };
}
