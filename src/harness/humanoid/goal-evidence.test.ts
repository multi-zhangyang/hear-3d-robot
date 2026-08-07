import { describe, expect, it } from "vitest";
import { ScenarioSchema } from "../../domain/schema.js";
import { goalSha256 } from "../../domain/goal-identity.js";
import { createGoalDAG } from "../../domain/goal-epoch.js";
import {
  GoalEvidenceArtifactSchema,
  createActionGoalEvidence,
  createGoalEvaluationEvidence,
  createWorldGoalEvidence,
  goalPredicateIsObservable
} from "./goal-evidence.js";
import type { HumanoidGraspAssessment } from "../../world/humanoid/grasp-tracker.js";
import { createHumanoidAutonomyContext } from "./autonomy-context.js";

const scenario = ScenarioSchema.parse({
  title: "Goal evidence test world",
  seed: 1,
  bounds: { width: 12, depth: 12 },
  visibility_radius: 6,
  robot: { x: 2, z: 2, yaw: 0 },
  obstacles: [{
    id: "visible-block",
    center: { x: 2.5, y: 0.5, z: 2.5 },
    size: { x: 1, y: 1, z: 1 }
  }],
  objects: [{
    id: "visible-object",
    kind: "cube",
    color: "red",
    position: { x: 3, y: 0.5, z: 3 },
    size: { x: 1, y: 1, z: 1 },
    portable: true
  }],
  zones: [{
    id: "placement-zone",
    color: "green",
    center: { x: 4, y: 0.01, z: 4 },
    size: { x: 2, y: 0.02, z: 2 }
  }],
  default_goal: {
    summary: "Remain observable",
    predicates: [{
      type: "robot_at",
      target: { x: 2, y: 0, z: 2 },
      tolerance: 0.5
    }]
  }
});

describe("Goal evidence artifacts", () => {
  it("binds a world observation and its physical identity to one content hash", () => {
    const artifact = createWorldGoalEvidence({
      world: {
        frame: 20,
        worldRevision: 12,
        robot: {
          rootPosition: { x: 2, y: 0, z: 2 }
        },
        grasp: graspState(20)
      },
      observation: observation(20, 12, "visible"),
      scenario
    });
    expect(GoalEvidenceArtifactSchema.parse(artifact)).toEqual(artifact);
    expect(artifact).toMatchObject({
      version: 4,
      observation: {
        objects: [{
          id: "visible-object",
          role: "manipulable",
          portable: true,
          position: { x: 3, y: 0.5, z: 3 }
        }],
        zones: [{
          id: "placement-zone",
          center: { x: 4, y: 0.01, z: 4 }
        }],
        solids: [{
          id: "visible-block",
          kind: "block",
          source_id: "visible-block"
        }]
      }
    });

    const autonomy = createHumanoidAutonomyContext({
      goalDAG: createGoalDAG(),
      worldEvidence: artifact
    });
    expect(autonomy).toMatchObject({
      source_world_revision: 12,
      selection_authority: "goal_manager_model",
      harness_selection: "none",
      object_frontier: [{
        object_id: "visible-object",
        portable: true,
        prior_goal_outcomes: { total: 0 }
      }],
      zone_frontier: [{
        zone_id: "placement-zone",
        prior_goal_outcomes: { total: 0 }
      }],
      solid_frontier: [{
        solid_id: "visible-block",
        kind: "block",
        supported_goal_predicates: ["block_removed"],
        prior_goal_outcomes: { total: 0 }
      }]
    });

    const reboundObservation = structuredClone(artifact);
    reboundObservation.observation!.visible_object_ids = ["invented-object"];
    expect(GoalEvidenceArtifactSchema.safeParse(reboundObservation).success).toBe(false);

    const reboundRevision = structuredClone(artifact);
    reboundRevision.evidence.world_revision += 1;
    expect(GoalEvidenceArtifactSchema.safeParse(reboundRevision).success).toBe(false);
  });

  it("uses only aligned visible tokens and never promotes remembered objects", () => {
    const hidden = createWorldGoalEvidence({
      world: {
        frame: 21,
        worldRevision: 13,
        robot: { rootPosition: { x: 2, y: 0, z: 2 } },
        grasp: graspState(21)
      },
      observation: observation(21, 13, "remembered"),
      scenario
    });
    const hiddenArtifacts = new Map([[hidden.evidence.ref, hidden]]);
    const objectAt = {
      type: "object_at" as const,
      object_id: "visible-object",
      target: { x: 3, y: 0.5, z: 3 },
      tolerance: 0.2
    };
    const objectGrasped = {
      type: "object_grasped" as const,
      object_id: "visible-object",
      hand: "either" as const
    };
    const objectPlaced = {
      type: "object_placed" as const,
      object_id: "visible-object",
      zone_id: "placement-zone",
      tolerance: 0.05
    };

    expect(hidden.observation?.visible_object_ids).toEqual([]);
    if (hidden.version !== 4) throw new Error("Expected version 4 world evidence");
    expect(hidden.observation?.objects).toEqual([]);
    expect(hidden.observation?.grasp.assessments).toEqual([]);
    expect(goalPredicateIsObservable({
      predicate: objectAt,
      worldRevision: 13,
      evidenceRefs: [hidden.evidence.ref],
      artifacts: hiddenArtifacts,
      scenario
    })).toBe(false);
    expect(goalPredicateIsObservable({
      predicate: objectGrasped,
      worldRevision: 13,
      evidenceRefs: [hidden.evidence.ref],
      artifacts: hiddenArtifacts,
      scenario
    })).toBe(false);
    expect(goalPredicateIsObservable({
      predicate: objectPlaced,
      worldRevision: 13,
      evidenceRefs: [hidden.evidence.ref],
      artifacts: hiddenArtifacts,
      scenario
    })).toBe(false);

    const nonObservable = createWorldGoalEvidence({
      world: {
        frame: 21,
        worldRevision: 13,
        robot: { rootPosition: { x: 2, y: 0, z: 2 } },
        grasp: graspState(21)
      },
      observation: observation(21, 13, "visible", false),
      scenario
    });
    expect(nonObservable.observation?.visible_object_ids).toEqual([]);

    const visible = createWorldGoalEvidence({
      world: {
        frame: 21,
        worldRevision: 13,
        robot: { rootPosition: { x: 2, y: 0, z: 2 } },
        grasp: graspState(21)
      },
      observation: observation(21, 13, "visible"),
      scenario
    });
    const visibleArtifacts = new Map([[visible.evidence.ref, visible]]);
    expect(goalPredicateIsObservable({
      predicate: objectAt,
      worldRevision: 13,
      evidenceRefs: [visible.evidence.ref],
      artifacts: visibleArtifacts,
      scenario
    })).toBe(true);

    const blockRemoved = {
      type: "block_removed" as const,
      block_id: "visible-block"
    };
    expect(goalPredicateIsObservable({
      predicate: blockRemoved,
      worldRevision: 13,
      evidenceRefs: [visible.evidence.ref],
      artifacts: visibleArtifacts,
      scenario
    })).toBe(true);

    const noSolid = createWorldGoalEvidence({
      world: {
        frame: 21,
        worldRevision: 13,
        robot: { rootPosition: { x: 2, y: 0, z: 2 } },
        grasp: graspState(21)
      },
      observation: observation(21, 13, "visible", true, false),
      scenario
    });
    expect(goalPredicateIsObservable({
      predicate: blockRemoved,
      worldRevision: 13,
      evidenceRefs: [noSolid.evidence.ref],
      artifacts: new Map([[noSolid.evidence.ref, noSolid]]),
      scenario
    })).toBe(false);
    expect(goalPredicateIsObservable({
      predicate: objectGrasped,
      worldRevision: 13,
      evidenceRefs: [visible.evidence.ref],
      artifacts: visibleArtifacts,
      scenario
    })).toBe(true);
    expect(goalPredicateIsObservable({
      predicate: objectPlaced,
      worldRevision: 13,
      evidenceRefs: [visible.evidence.ref],
      artifacts: visibleArtifacts,
      scenario
    })).toBe(true);
  });

  it("rejects an observation or grasp assessment from another authority frame", () => {
    expect(() => createWorldGoalEvidence({
      world: {
        frame: 22,
        worldRevision: 14,
        robot: { rootPosition: { x: 2, y: 0, z: 2 } },
        grasp: graspState(22)
      },
      observation: observation(21, 14, "visible"),
      scenario
    })).toThrow("not aligned");

    expect(() => createWorldGoalEvidence({
      world: {
        frame: 22,
        worldRevision: 14,
        robot: { rootPosition: { x: 2, y: 0, z: 2 } },
        grasp: graspState(21)
      },
      observation: observation(22, 14, "visible"),
      scenario
    })).toThrow("grasp assessment");
  });

  it("binds action and Goal evaluation metadata to their durable payloads", () => {
    const action = createActionGoalEvidence({
      transactionId: "tool-call-1",
      worldFrame: 30,
      worldRevision: 22,
      receipt: {
        transactionId: "tool-call-1",
        worldAfterRevision: 22,
        accepted: false,
        code: "physical_rejection"
      }
    });
    const goalContentSha256 = goalSha256(scenario.default_goal);
    const evaluation = createGoalEvaluationEvidence({
      epochId: `goal-epoch:${"a".repeat(64)}`,
      goalContentSha256,
      worldFrame: 31,
      worldRevision: 23,
      evaluation: {
        success: true,
        goal: scenario.default_goal,
        worldFrame: 31,
        worldRevision: 23
      }
    });
    expect(GoalEvidenceArtifactSchema.parse(action)).toEqual(action);
    expect(GoalEvidenceArtifactSchema.parse(evaluation)).toEqual(evaluation);

    const reboundAction = structuredClone(action);
    reboundAction.evidence.world_revision += 1;
    expect(GoalEvidenceArtifactSchema.safeParse(reboundAction).success).toBe(false);

    const reboundEvaluation = structuredClone(evaluation);
    if (reboundEvaluation.evidence.kind !== "goal_evaluation") {
      throw new Error("Expected Goal evaluation evidence");
    }
    reboundEvaluation.evidence.goal_content_sha256 = "c".repeat(64);
    expect(GoalEvidenceArtifactSchema.safeParse(reboundEvaluation).success).toBe(false);
  });
});

function observation(
  frame: number,
  worldRevision: number,
  status: "visible" | "remembered",
  observable = status === "visible",
  includeSolid = true
) {
  return {
    frame,
    worldRevision,
    objectTokens: [{
      id: "visible-object",
      role: "manipulable" as const,
      kind: "cube",
      color: "red",
      size: { x: 1, y: 1, z: 1 },
      portable: true,
      status,
      state: status === "visible" ? "active" as const : "historical" as const,
      authority: status === "visible" ? "mujoco_exact" as const : "sensor_history" as const,
      exact: status === "visible",
      observable,
      pose: {
        position: { x: 3, y: 0.5, z: 3 },
        rotation: { x: 0, y: 0, z: 0, w: 1 }
      },
      observedFrame: status === "visible" ? frame : frame - 1,
      observedWorldRevision: status === "visible" ? worldRevision : worldRevision - 1,
      position: { x: 3, y: 0.5, z: 3 },
      rotation: { x: 0, y: 0, z: 0, w: 1 },
      linearVelocity: { x: 0, y: 0, z: 0 },
      angularVelocity: { x: 0, y: 0, z: 0 },
      firstSeenRevision: 1,
      lastSeenRevision: status === "visible" ? worldRevision : worldRevision - 1,
      lastSeenFrame: status === "visible" ? frame : frame - 1,
      observationCount: 2,
      ageRevisions: status === "visible" ? 0 : 1,
      relation: {
        distanceToRobot: Math.sqrt(2),
        bearingRadians: Math.PI / 4,
        verticalOffset: 0.5,
        distanceToLeftWrist: 1,
        distanceToRightWrist: 1.1
      },
      currentContacts: []
    }],
    solidTokens: includeSolid
      ? [{
          id: "visible-block",
          sourceId: "visible-block",
          kind: "block" as const,
          center: { x: 2.5, y: 0.5, z: 2.5 },
          size: { x: 1, y: 1, z: 1 },
          currentContacts: []
        }]
      : []
  };
}

function graspState(frame: number) {
  return {
    contractSha256: "a".repeat(64),
    assessments: [
      graspAssessment(frame, "left"),
      graspAssessment(frame, "right")
    ]
  };
}

function graspAssessment(
  frame: number,
  hand: "left" | "right"
): HumanoidGraspAssessment {
  return {
    protocol: "humanoid-grasp-assessment-v1",
    frame,
    object_id: "visible-object",
    hand,
    phase: "idle",
    grasp_verified: false,
    reason: "contact_missing",
    reset_reason: "contact_lost",
    evidence: {
      contact: {
        status: "missing",
        observed_contact_count: 0,
        force_qualified_contact_count: 0,
        distinct_force_qualified_links: [],
        distinct_normal_qualified_links: [],
        opposing_pair: null
      },
      support: {
        status: "supported",
        candidate_contact_count: 1,
        force_qualified_contact_count: 1,
        upward_contact_count: 1,
        baseline_projection_m: 0.5,
        current_projection_m: 0.5,
        lift_m: 0
      },
      relative_pose: {
        stable_frames: 0,
        translation_drift_m: null,
        rotation_drift_rad: null
      },
      lifted_hold_frames: 0
    }
  };
}
