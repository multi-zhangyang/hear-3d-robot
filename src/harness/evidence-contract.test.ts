import { describe, expect, it } from "vitest";
import type { ActionReceipt, EvidenceRequirement, JsonValue } from "../domain/schema.js";
import {
  assertEvidenceRequirementsJointlySatisfiable,
  assertReceiptRequirementDefinition,
  hierarchyNeedsEvidenceContractRotation,
  receiptEvidenceRequirement,
  type ReceiptEvidenceProvenanceContext,
  verifyBlockerEvidence,
  verifyReceiptEvidence
} from "./evidence-contract.js";

type ReceiptWithBeforeRevision = ActionReceipt & {
  world_before_revision?: number;
};

function receipt(overrides: Partial<ReceiptWithBeforeRevision> = {}): ActionReceipt {
  const value: ReceiptWithBeforeRevision = {
    transaction_id: "agent_a:inspect_1",
    agent_id: "agent_a",
    agent_name: "Inspector",
    kind: "tool",
    name: "inspect_entity",
    input: { entity_id: "red_block" },
    accepted: true,
    code: "entity_state",
    detail: { entity_id: "red_block" },
    world_before_frame: 4,
    world_after_frame: 4,
    frame_count: 0,
    world_before_revision: 7,
    world_revision: 7,
    channels: [],
    gates: [],
    committed_at: "2026-07-30T00:00:00.000Z",
    ...overrides
  };
  return value;
}

const baseTarget = { x: 4, y: 0, z: 8 };

function basePlanReceipt(overrides: Partial<ReceiptWithBeforeRevision> = {}): ActionReceipt {
  return receipt({
    transaction_id: "agent_a:plan_base",
    name: "plan_base_path",
    kind: "tool",
    input: { target: baseTarget, face_point: { x: 5, y: 0, z: 8 } },
    code: "base_path_planned",
    detail: { plan_id: "base_plan_1", resolved_target: baseTarget },
    world_before_revision: 7,
    world_revision: 7,
    ...overrides
  });
}

function baseExecutionReceipt(
  overrides: Partial<ReceiptWithBeforeRevision> = {}
): ActionReceipt {
  return receipt({
    transaction_id: "agent_a:execute_base",
    name: "execute_base_plan",
    kind: "skill",
    input: { planning_transaction_id: "agent_a:plan_base", options: {} },
    code: "base_plan_completed",
    detail: { plan_id: "base_plan_1", final_position: baseTarget },
    world_before_revision: 7,
    world_revision: 8,
    channels: ["base"],
    ...overrides
  });
}

function provenance(
  source: ActionReceipt | undefined,
  authorized = true
): ReceiptEvidenceProvenanceContext {
  return {
    lookupReceipt: () => source,
    isSourceAuthorized: () => authorized
  };
}

describe("typed hierarchy evidence", () => {
  const requirement = receiptEvidenceRequirement(0, "inspect_entity", {
    kind: "entity",
    entity_id: "red_block"
  });

  it("derives action, result, effect, target, and freshness from the receipt", () => {
    expect(verifyReceiptEvidence(0, requirement, receipt(), 7)).toMatchObject({
      criterion_index: 0,
      action: "inspect_entity",
      result_code: "entity_state",
      effect: "observation",
      target: { kind: "entity", entity_id: "red_block" },
      freshness: "current_world",
      world_revision: 7
    });
  });

  it("rejects unrelated accepted receipts, false result codes, wrong targets, and stale state", () => {
    expect(() => verifyReceiptEvidence(0, requirement, receipt({
      name: "sense_scene",
      input: {},
      code: "scene_observation"
    }), 7)).toThrow("expected one of inspect_entity");
    expect(() => verifyReceiptEvidence(0, requirement, receipt({ code: "scene_observation" }), 7))
      .toThrow("accepted inspect_entity evidence requires entity_state");
    expect(() => verifyReceiptEvidence(0, requirement, receipt({
      input: { entity_id: "blue_block" }
    }), 7)).toThrow("not {\"kind\":\"entity\",\"entity_id\":\"red_block\"}");
    expect(() => verifyReceiptEvidence(0, requirement, receipt(), 8))
      .toThrow("current revision is 8");
  });

  it("rejects evidence definitions outside the grant or action policy", () => {
    expect(() => assertReceiptRequirementDefinition(requirement, ["sense_scene"]))
      .toThrow("outside the child capability grant");
    expect(() => assertReceiptRequirementDefinition({
      ...requirement,
      effect: "body_motion"
    }, ["inspect_entity"])).toThrow("produces observation, not body_motion");
  });

  it("revalidates restored requirement authority at receipt verification time", () => {
    expect(() => verifyReceiptEvidence(1, requirement, receipt(), 7))
      .toThrow("does not match requirement 0");
    expect(() => verifyReceiptEvidence(0, {
      ...requirement,
      freshness: "historical_record"
    }, receipt(), 99)).toThrow("requires current_world, not historical_record");
    expect(() => verifyReceiptEvidence(0, {
      ...requirement,
      effect: "body_motion"
    }, receipt(), 7)).toThrow("produces observation, not body_motion");
  });

  it("rejects internally inconsistent blocker receipts", () => {
    expect(() => verifyBlockerEvidence(0, requirement, receipt({
      accepted: true,
      code: "entity_not_visible"
    }), 7)).toThrow("requires a rejected transaction");
    expect(() => verifyBlockerEvidence(0, requirement, receipt({
      accepted: false,
      code: "entity_state"
    }), 7)).toThrow("uses successful result code");
  });

  it.each([
    "repeated_accepted_action",
    "body_channel_busy",
    "repeated_denied_action"
  ])("rejects non-terminal blocker source %s", (code) => {
    expect(() => verifyBlockerEvidence(0, requirement, receipt({
      accepted: false,
      code
    }), 7)).toThrow(`non-terminal code ${code}`);
  });

  it("accepts a current rejected terminal blocker", () => {
    expect(verifyBlockerEvidence(0, requirement, receipt({
      accepted: false,
      code: "entity_not_visible"
    }), 7)).toMatchObject({
      accepted: false,
      result_code: "entity_not_visible",
      world_revision: 7
    });
  });

  it("rotates only unfinished legacy branches that cannot provide typed evidence", () => {
    const node = {
      id: "worker",
      name: "Worker",
      parent_id: "root",
      child_ids: [],
      objective: "Observe the entity.",
      success_criteria: ["The entity is observed."],
      evidence_requirements: [],
      goal_predicate_indexes: [],
      capabilities: ["inspect_entity"],
      may_delegate: false,
      references: [],
      depth: 1,
      status: "active" as const,
      steps_used: 0,
      model_calls_used: 1,
      created_at: "2026-07-30T00:00:00.000Z",
      updated_at: "2026-07-30T00:00:00.000Z"
    };

    expect(hierarchyNeedsEvidenceContractRotation({ worker: node }, "root")).toBe(true);
    expect(hierarchyNeedsEvidenceContractRotation({
      worker: { ...node, status: "completed" }
    }, "root")).toBe(false);
    expect(hierarchyNeedsEvidenceContractRotation({
      worker: { ...node, evidence_requirements: [requirement] }
    }, "root")).toBe(false);
    expect(hierarchyNeedsEvidenceContractRotation({
      worker: {
        ...node,
        success_criteria: ["Plan exists.", "Plan executed."],
        capabilities: ["plan_base_path", "execute_base_plan"],
        evidence_requirements: [
          receiptEvidenceRequirement(0, "plan_base_path", {
            kind: "position",
            position: baseTarget
          }),
          receiptEvidenceRequirement(1, "execute_base_plan", {
            kind: "body",
            channel: "base"
          })
        ]
      }
    }, "root")).toBe(true);
  });
});

describe("joint evidence satisfiability", () => {
  it("accepts observation, planning, and one terminal physical criterion when compatible", () => {
    const requirements: EvidenceRequirement[] = [
      receiptEvidenceRequirement(0, "survey_terrain", { kind: "terrain" }),
      receiptEvidenceRequirement(1, "plan_base_path", {
        kind: "position",
        position: baseTarget
      }),
      receiptEvidenceRequirement(2, "set_head_target", {
        kind: "body",
        channel: "head"
      }),
      { kind: "goal_predicate", criterion_index: 3, predicate_index: 0 }
    ];

    expect(() => assertEvidenceRequirementsJointlySatisfiable(requirements)).not.toThrow();
  });

  it("rejects base planning and base execution as simultaneous terminal criteria", () => {
    const requirements = [
      receiptEvidenceRequirement(0, "plan_base_path", {
        kind: "position",
        position: baseTarget
      }),
      receiptEvidenceRequirement(1, "execute_base_plan", {
        kind: "body",
        channel: "base"
      })
    ];

    expect(() => assertEvidenceRequirementsJointlySatisfiable(requirements))
      .toThrow("cannot declare plan_base_path together with execute_base_plan");
  });

  it.each([
    ["plan_joint_targets", { kind: "body", channel: "arm" }],
    ["solve_end_effector_position", {
      kind: "position",
      position: { x: 0.2, y: 1, z: 0.1 }
    }],
    ["solve_end_effector_pose", {
      kind: "position",
      position: { x: 0.2, y: 1, z: 0.1 }
    }]
  ] as const)("rejects %s together with joint execution", (planner, target) => {
    const requirements = [
      receiptEvidenceRequirement(0, planner, target),
      receiptEvidenceRequirement(1, "execute_joint_plan", {
        kind: "body",
        channel: "arm"
      })
    ];

    expect(() => assertEvidenceRequirementsJointlySatisfiable(requirements))
      .toThrow(`cannot declare ${planner} together with execute_joint_plan`);
  });

  it("detects conflicting planner/executor alternatives inside multi-action requirements", () => {
    const planning = receiptEvidenceRequirement(0, "plan_base_path", {
      kind: "position",
      position: baseTarget
    });
    const executing = receiptEvidenceRequirement(1, "execute_base_plan", {
      kind: "body",
      channel: "base"
    });

    expect(() => assertEvidenceRequirementsJointlySatisfiable([
      { ...planning, actions: ["sense_scene", "plan_base_path"] },
      executing
    ])).toThrow("plan_base_path together with execute_base_plan");
  });

  it("rejects any two terminal body-motion or world-mutation criteria", () => {
    expect(() => assertEvidenceRequirementsJointlySatisfiable([
      receiptEvidenceRequirement(0, "set_head_target", {
        kind: "body",
        channel: "head"
      }),
      receiptEvidenceRequirement(1, "break_voxel", {
        kind: "voxel",
        coordinate: { column: 1, level: 0, row: 2 }
      })
    ])).toThrow("multiple terminal physical receipt criteria");
  });
});

describe("execution evidence provenance", () => {
  const baseRequirement = receiptEvidenceRequirement(0, "execute_base_plan", {
    kind: "body",
    channel: "base"
  });

  it("returns the authorized base plan source and canonical source target", () => {
    const source = basePlanReceipt();
    expect(verifyReceiptEvidence(
      0,
      baseRequirement,
      baseExecutionReceipt(),
      8,
      provenance(source)
    )).toMatchObject({
      source_transaction_id: "agent_a:plan_base",
      source_action: "plan_base_path",
      source_target: { kind: "position", position: baseTarget }
    });
  });

  it("requires provenance context and an explicit execution-start revision", () => {
    expect(() => verifyReceiptEvidence(
      0,
      baseRequirement,
      baseExecutionReceipt(),
      8
    )).toThrow("requires receipt provenance context");
    expect(() => verifyReceiptEvidence(
      0,
      baseRequirement,
      baseExecutionReceipt({ world_before_revision: undefined }),
      8,
      provenance(basePlanReceipt())
    )).toThrow("no canonical world_before_revision");
  });

  it("requires the exact planning transaction named by the execution input", () => {
    expect(() => verifyReceiptEvidence(
      0,
      baseRequirement,
      baseExecutionReceipt({ input: {} }),
      8,
      provenance(basePlanReceipt())
    )).toThrow("no canonical planning_transaction_id");
    expect(() => verifyReceiptEvidence(
      0,
      baseRequirement,
      baseExecutionReceipt(),
      8,
      provenance(undefined)
    )).toThrow("references unknown planning transaction");
    expect(() => verifyReceiptEvidence(
      0,
      baseRequirement,
      baseExecutionReceipt(),
      8,
      provenance(basePlanReceipt({ transaction_id: "agent_a:other_plan" }))
    )).toThrow("returned transaction agent_a:other_plan");
  });

  it("rejects rejected, non-tool, wrong-action, and unauthorized plan sources", () => {
    expect(() => verifyReceiptEvidence(
      0,
      baseRequirement,
      baseExecutionReceipt(),
      8,
      provenance(basePlanReceipt({ accepted: false, code: "base_path_collision" }))
    )).toThrow("was rejected");
    expect(() => verifyReceiptEvidence(
      0,
      baseRequirement,
      baseExecutionReceipt(),
      8,
      provenance(basePlanReceipt({ kind: "skill" }))
    )).toThrow("requires a tool receipt");
    expect(() => verifyReceiptEvidence(
      0,
      baseRequirement,
      baseExecutionReceipt(),
      8,
      provenance(basePlanReceipt({
        name: "plan_joint_targets",
        code: "joint_target_plan"
      }))
    )).toThrow("expected planning action plan_base_path");
    expect(() => verifyReceiptEvidence(
      0,
      baseRequirement,
      baseExecutionReceipt(),
      8,
      provenance(basePlanReceipt(), false)
    )).toThrow("not authorized for this evidence branch");
  });

  it("rejects a stale source, invalid source result, or a different internal plan", () => {
    expect(() => verifyReceiptEvidence(
      0,
      baseRequirement,
      baseExecutionReceipt(),
      8,
      provenance(basePlanReceipt({ world_revision: 6 }))
    )).toThrow("execution began at revision 7");
    expect(() => verifyReceiptEvidence(
      0,
      baseRequirement,
      baseExecutionReceipt(),
      8,
      provenance(basePlanReceipt({ code: "scene_observation" }))
    )).toThrow("invalid accepted code scene_observation");
    expect(() => verifyReceiptEvidence(
      0,
      baseRequirement,
      baseExecutionReceipt({ detail: { plan_id: "base_plan_other" } }),
      8,
      provenance(basePlanReceipt())
    )).toThrow("not source plan base_plan_1");
  });

  it.each([
    [
      "plan_joint_targets",
      { targets: { shoulder: 0.2 } },
      "joint_target_plan",
      { kind: "body", channel: "arm" }
    ],
    [
      "solve_end_effector_position",
      { position: { x: 0.2, y: 1, z: 0.1 } },
      "end_effector_solution",
      { kind: "position", position: { x: 0.2, y: 1, z: 0.1 } }
    ],
    [
      "solve_end_effector_pose",
      { position: { x: 0.2, y: 1, z: 0.1 } },
      "end_effector_solution",
      { kind: "position", position: { x: 0.2, y: 1, z: 0.1 } }
    ]
  ] as const)("accepts authorized %s provenance for joint execution", (
    action,
    input,
    code,
    expectedTarget
  ) => {
    const source = receipt({
      transaction_id: "agent_a:plan_joint",
      name: action,
      kind: "tool",
      input: input as unknown as JsonValue,
      code,
      detail: { plan_id: "arm_plan_1" },
      world_before_revision: 7,
      world_revision: 7
    });
    const execution = receipt({
      transaction_id: "agent_a:execute_joint",
      name: "execute_joint_plan",
      kind: "skill",
      input: { planning_transaction_id: "agent_a:plan_joint" },
      code: "joint_targets_reached",
      detail: { plan_id: "arm_plan_1" },
      world_before_revision: 7,
      world_revision: 8,
      channels: ["arm"]
    });
    const requirement = receiptEvidenceRequirement(0, "execute_joint_plan", {
      kind: "body",
      channel: "arm"
    });

    expect(verifyReceiptEvidence(
      0,
      requirement,
      execution,
      8,
      provenance(source)
    )).toMatchObject({
      source_transaction_id: "agent_a:plan_joint",
      source_action: action,
      source_target: expectedTarget
    });
  });
});
