import { describe, expect, it } from "vitest";
import { createHash } from "node:crypto";
import {
  HumanoidMotionOptionContractSchema,
  advanceHumanoidMotionOptionMonitor,
  createHumanoidMotionOptionMonitorState,
  detectHumanoidMotionOption,
  humanoidMotionOptionContractSha256,
  type HumanoidMotionOptionCondition,
  type HumanoidMotionOptionContract,
  type HumanoidMotionOptionDetectorInput,
  type HumanoidMotionOptionRobotSnapshot
} from "./motion-option.js";

const robot: HumanoidMotionOptionRobotSnapshot = {
  rootPosition: { x: 1, y: 0.8, z: 2 },
  links: {
    pelvis: {
      position: { x: 1, y: 0.8, z: 2 },
      rotation: {
        x: 0,
        y: Math.SQRT1_2,
        z: 0,
        w: Math.SQRT1_2
      }
    },
    left_wrist_yaw_link: {
      position: { x: 1.2, y: 1.1, z: 2.4 }
    },
    right_wrist_yaw_link: {
      position: { x: 1.4, y: 1.1, z: 1.8 }
    }
  },
  contacts: [{
    normalForce: 14,
    firstBody: "left_wrist_yaw_link",
    secondBody: null,
    firstObject: null,
    secondObject: "crate"
  }]
};

const observableCrate = {
  id: "crate",
  position: { x: 1, y: 0.3, z: 3 },
  size: { x: 0.5, y: 0.5, z: 0.5 }
};

const destination = {
  id: "destination",
  center: { x: 1, y: 0, z: 3 },
  size: { x: 2, y: 0.1, z: 2 }
};

const contract: HumanoidMotionOptionContract = {
  option_id: "reach-and-place",
  stable_steps: 5,
  predicates: [
    {
      type: "root_near_point",
      target: { x: 1, y: 0.8, z: 2 },
      tolerance_m: 0.05
    },
    {
      type: "body_near_point",
      body: "left_wrist_yaw_link",
      target: { x: 1.2, y: 1.1, z: 2.4 },
      tolerance_m: 0.05
    },
    {
      type: "body_contact_object",
      body: "left_wrist_yaw_link",
      object_id: "crate",
      minimum_normal_force: 10
    },
    {
      type: "object_near_point",
      object_id: "crate",
      target: { x: 1, y: 0.3, z: 3 },
      tolerance_m: 0.05
    },
    {
      type: "object_in_zone",
      object_id: "crate",
      zone_id: "destination",
      expected: true,
      tolerance_m: 0.01
    }
  ]
};

describe("humanoid motion option detector", () => {
  it("validates physical predicate contracts and stable steps", () => {
    expect(HumanoidMotionOptionContractSchema.parse(contract)).toEqual({
      ...contract,
      phases: null
    });
    expect(humanoidMotionOptionContractSha256(contract)).toBe(
      createHash("sha256").update(JSON.stringify({
        option_id: contract.option_id,
        predicates: contract.predicates,
        stable_steps: contract.stable_steps
      })).digest("hex")
    );
    expect(() => HumanoidMotionOptionContractSchema.parse({
      ...contract,
      stable_steps: 0
    })).toThrow();
    expect(() => HumanoidMotionOptionContractSchema.parse({
      ...contract,
      predicates: [{
        type: "body_near_point",
        body: "invented_link",
        target: { x: 0, y: 0, z: 0 },
        tolerance_m: 0.1
      }]
    })).toThrow();
  });

  it("evaluates named end effectors in world and pelvis frames", () => {
    const endEffectorContract: HumanoidMotionOptionContract = {
      option_id: "right-hand-frame-check",
      stable_steps: 2,
      predicates: [
        {
          type: "end_effector_near_point",
          end_effector: "right_wrist",
          frame: "world",
          target: { x: 1.4, y: 1.1, z: 1.8 },
          tolerance_m: 0.001
        },
        {
          type: "end_effector_near_point",
          end_effector: "right_wrist",
          frame: "pelvis",
          target: { x: 0.2, y: 0.3, z: 0.4 },
          tolerance_m: 0.001
        }
      ]
    };

    const detection = detectHumanoidMotionOption(endEffectorContract, {
      snapshot: robot,
      observableObjects: [],
      zones: []
    });

    expect(detection.allSatisfied).toBe(true);
    expect(detection.evidence).toEqual([
      expect.objectContaining({
        type: "end_effector_near_point",
        frame: "world",
        endEffector: "right_wrist",
        actualPosition: { x: 1.4, y: 1.1, z: 1.8 },
        status: "satisfied"
      }),
      expect.objectContaining({
        type: "end_effector_near_point",
        frame: "pelvis",
        endEffector: "right_wrist",
        actualPosition: {
          x: expect.closeTo(0.2, 10),
          y: expect.closeTo(0.3, 10),
          z: expect.closeTo(0.4, 10)
        },
        status: "satisfied"
      })
    ]);
  });

  it("fails closed when a pelvis-relative frame cannot be observed", () => {
    const endEffectorContract: HumanoidMotionOptionContract = {
      option_id: "missing-pelvis-frame",
      stable_steps: 1,
      predicates: [{
        type: "end_effector_near_point",
        end_effector: "left_wrist",
        frame: "pelvis",
        target: { x: 0, y: 0, z: 0 },
        tolerance_m: 0.1
      }]
    };
    const detection = detectHumanoidMotionOption(endEffectorContract, {
      snapshot: { ...robot, links: { left_wrist_yaw_link: robot.links.left_wrist_yaw_link! } },
      observableObjects: [],
      zones: []
    });

    expect(detection.status).toBe("uncertain");
    expect(detection.evidence[0]).toMatchObject({
      actualPosition: null,
      reason: "end_effector_snapshot_missing"
    });
  });

  it("accepts only bounded predicate-index condition ASTs", () => {
    const terminal = {
      precondition: null,
      during: null,
      terminal: {
        condition: {
          op: "all" as const,
          conditions: [
            { op: "predicate" as const, predicate_index: 0 },
            {
              op: "not" as const,
              condition: { op: "predicate" as const, predicate_index: 1 }
            }
          ]
        }
      }
    };
    expect(HumanoidMotionOptionContractSchema.parse({
      ...contract,
      phases: terminal
    }).phases).toEqual(terminal);

    expect(() => HumanoidMotionOptionContractSchema.parse({
      ...contract,
      phases: {
        ...terminal,
        terminal: {
          condition: { op: "predicate", predicate_index: 15 }
        }
      }
    })).toThrow(/missing predicate 15/);
    expect(() => HumanoidMotionOptionContractSchema.parse({
      ...contract,
      phases: {
        ...terminal,
        terminal: {
          condition: {
            op: "predicate",
            predicate_index: 0,
            javascript: "return true"
          }
        }
      }
    })).toThrow();

    let tooDeep: HumanoidMotionOptionCondition = {
      op: "predicate",
      predicate_index: 0
    };
    for (let level = 0; level < 8; level += 1) {
      tooDeep = { op: "not", condition: tooDeep };
    }
    expect(() => HumanoidMotionOptionContractSchema.parse({
      ...contract,
      phases: {
        ...terminal,
        terminal: { condition: tooDeep }
      }
    })).toThrow(/eight AST levels/);

    const tooManyNodes: HumanoidMotionOptionCondition = {
      op: "all",
      conditions: Array.from({ length: 16 }, () => ({
        op: "all" as const,
        conditions: Array.from({ length: 4 }, () => ({
          op: "predicate" as const,
          predicate_index: 0
        }))
      }))
    };
    expect(() => HumanoidMotionOptionContractSchema.parse({
      ...contract,
      phases: {
        ...terminal,
        terminal: { condition: tooManyNodes }
      }
    })).toThrow(/64 AST nodes/);
  });

  it("returns ordered evidence when every observable physical relation holds", () => {
    const detection = detectHumanoidMotionOption(contract, {
      snapshot: robot,
      observableObjects: [observableCrate],
      zones: [destination]
    });

    expect(detection.allSatisfied).toBe(true);
    expect(detection.hasUncertain).toBe(false);
    expect(detection.evidence.map((entry) => ({
      index: entry.predicateIndex,
      type: entry.type,
      status: entry.status
    }))).toEqual([
      { index: 0, type: "root_near_point", status: "satisfied" },
      { index: 1, type: "body_near_point", status: "satisfied" },
      { index: 2, type: "body_contact_object", status: "satisfied" },
      { index: 3, type: "object_near_point", status: "satisfied" },
      { index: 4, type: "object_in_zone", status: "satisfied" }
    ]);
    expect(detection.evidence[2]).toMatchObject({
      maximumNormalForce: 14,
      minimumNormalForce: 10,
      objectObservable: true
    });
  });

  it("never uses hidden snapshot objects or contacts as observable success", () => {
    const snapshotWithHiddenObject = {
      ...robot,
      objects: {
        hidden: {
          id: "hidden",
          position: { x: 4, y: 0.25, z: 4 }
        }
      },
      contacts: [{
        normalForce: 50,
        firstBody: "left_wrist_yaw_link" as const,
        secondBody: null,
        firstObject: null,
        secondObject: "hidden"
      }]
    };
    const hiddenContract = HumanoidMotionOptionContractSchema.parse({
      option_id: "hidden-object-must-not-pass",
      stable_steps: 2,
      predicates: [
        {
          type: "body_contact_object",
          body: "left_wrist_yaw_link",
          object_id: "hidden",
          minimum_normal_force: 1
        },
        {
          type: "object_near_point",
          object_id: "hidden",
          target: { x: 4, y: 0.25, z: 4 },
          tolerance_m: 0.01
        },
        {
          type: "object_in_zone",
          object_id: "hidden",
          zone_id: "destination",
          expected: true,
          tolerance_m: 0.01
        }
      ]
    });

    const detection = detectHumanoidMotionOption(hiddenContract, {
      snapshot: snapshotWithHiddenObject,
      observableObjects: [],
      zones: [destination]
    });

    expect(detection.allSatisfied).toBe(false);
    expect(detection.hasUncertain).toBe(true);
    expect(detection.evidence).toHaveLength(3);
    expect(detection.evidence.every((entry) => entry.status === "uncertain")).toBe(true);
    expect(detection.evidence.every((entry) => (
      "reason" in entry && entry.reason === "object_not_observable"
    ))).toBe(true);
  });

  it("distinguishes measured failure from unavailable body or zone evidence", () => {
    const input: HumanoidMotionOptionDetectorInput = {
      snapshot: {
        ...robot,
        rootPosition: { x: 4, y: 0.8, z: 4 },
        links: {},
        contacts: []
      },
      observableObjects: [observableCrate],
      zones: []
    };
    const detection = detectHumanoidMotionOption(contract, input);

    expect(detection.allSatisfied).toBe(false);
    expect(detection.hasUncertain).toBe(true);
    expect(detection.evidence[0]).toMatchObject({ status: "unsatisfied" });
    expect(detection.evidence[1]).toMatchObject({
      status: "uncertain",
      reason: "body_snapshot_missing"
    });
    expect(detection.evidence[2]).toMatchObject({
      status: "unsatisfied",
      maximumNormalForce: 0
    });
    expect(detection.evidence[4]).toMatchObject({
      status: "uncertain",
      reason: "zone_not_found"
    });
  });

  it("supports an explicit expectation that an observed object is outside a zone", () => {
    const outside = { ...observableCrate, position: { x: 5, y: 0.3, z: 5 } };
    const outsideContract = HumanoidMotionOptionContractSchema.parse({
      option_id: "outside-zone",
      stable_steps: 1,
      predicates: [{
        type: "object_in_zone",
        object_id: "crate",
        zone_id: "destination",
        expected: false,
        tolerance_m: 0
      }]
    });
    const detection = detectHumanoidMotionOption(outsideContract, {
      snapshot: robot,
      observableObjects: [outside],
      zones: [destination]
    });

    expect(detection).toMatchObject({
      allSatisfied: true,
      hasUncertain: false,
      evidence: [{ status: "satisfied", inside: false, expected: false }]
    });
  });

  it("evaluates all, any, and not with three-valued physical logic", () => {
    const predicates = [
      {
        type: "root_near_point" as const,
        target: { ...robot.rootPosition },
        tolerance_m: 0.01
      },
      {
        type: "object_near_point" as const,
        object_id: "hidden",
        target: { x: 0, y: 0, z: 0 },
        tolerance_m: 0.01
      },
      {
        type: "root_near_point" as const,
        target: { x: 9, y: 0.8, z: 9 },
        tolerance_m: 0.01
      }
    ];
    const input = {
      snapshot: robot,
      observableObjects: [observableCrate],
      zones: [destination]
    };
    const detectionFor = (condition: unknown) => detectHumanoidMotionOption(
      HumanoidMotionOptionContractSchema.parse({
        option_id: "three-value-logic",
        predicates,
        stable_steps: 1,
        phases: {
          precondition: null,
          during: null,
          terminal: { condition }
        }
      }),
      input
    );

    const resolvedAny = detectionFor({
      op: "any",
      conditions: [
        { op: "predicate", predicate_index: 0 },
        { op: "predicate", predicate_index: 1 }
      ]
    });
    expect(resolvedAny).toMatchObject({
      status: "satisfied",
      allSatisfied: true,
      hasUncertain: false,
      phases: { terminal: { status: "satisfied", predicateIndexes: [0, 1] } }
    });

    const resolvedAll = detectionFor({
      op: "all",
      conditions: [
        { op: "predicate", predicate_index: 2 },
        { op: "predicate", predicate_index: 1 }
      ]
    });
    expect(resolvedAll).toMatchObject({
      status: "unsatisfied",
      allSatisfied: false,
      hasUncertain: false
    });

    const unresolvedNot = detectionFor({
      op: "not",
      condition: { op: "predicate", predicate_index: 1 }
    });
    expect(unresolvedNot).toMatchObject({
      status: "uncertain",
      allSatisfied: false,
      hasUncertain: true
    });
  });

  it("monitors precondition, during invariant, and terminal stability windows", () => {
    const phasedContract = HumanoidMotionOptionContractSchema.parse({
      option_id: "phased-reach",
      predicates: [
        {
          type: "root_near_point",
          target: { ...robot.rootPosition },
          tolerance_m: 0.01
        },
        {
          type: "body_near_point",
          body: "left_wrist_yaw_link",
          target: { x: 1.2, y: 1.1, z: 2.4 },
          tolerance_m: 0.05
        },
        {
          type: "object_near_point",
          object_id: "crate",
          target: { x: 4, y: 0.3, z: 4 },
          tolerance_m: 0.05
        }
      ],
      stable_steps: 2,
      phases: {
        precondition: {
          condition: { op: "predicate", predicate_index: 0 },
          stable_steps: 2
        },
        during: {
          condition: { op: "predicate", predicate_index: 1 }
        },
        terminal: {
          condition: { op: "predicate", predicate_index: 2 }
        }
      }
    });
    const inputWith = (
      snapshot: HumanoidMotionOptionRobotSnapshot,
      observableObjects = [observableCrate]
    ): HumanoidMotionOptionDetectorInput => ({
      snapshot,
      observableObjects,
      zones: [destination]
    });
    const targetCrate = {
      ...observableCrate,
      position: { x: 4, y: 0.3, z: 4 }
    };

    let state = createHumanoidMotionOptionMonitorState(phasedContract);
    let update = advanceHumanoidMotionOptionMonitor(
      phasedContract,
      state,
      inputWith(robot, [targetCrate])
    );
    expect(update.state).toMatchObject({
      phase: "awaiting_precondition",
      preconditionStableSteps: 1,
      terminalStableSteps: 0
    });
    state = update.state;
    update = advanceHumanoidMotionOptionMonitor(
      phasedContract,
      state,
      inputWith(robot, [targetCrate])
    );
    expect(update.state).toMatchObject({
      phase: "running",
      preconditionStableSteps: 2,
      terminalStableSteps: 0
    });

    const afterPrecondition = {
      ...robot,
      rootPosition: { x: 8, y: 0.8, z: 8 }
    };
    state = update.state;
    update = advanceHumanoidMotionOptionMonitor(
      phasedContract,
      state,
      inputWith(afterPrecondition, [targetCrate])
    );
    expect(update.state).toMatchObject({
      phase: "running",
      terminalStableSteps: 1
    });

    state = update.state;
    update = advanceHumanoidMotionOptionMonitor(
      phasedContract,
      state,
      inputWith(afterPrecondition, [])
    );
    expect(update).toMatchObject({
      observationStatus: "uncertain",
      state: { phase: "running", terminalStableSteps: 0 }
    });

    for (let step = 0; step < 2; step += 1) {
      update = advanceHumanoidMotionOptionMonitor(
        phasedContract,
        update.state,
        inputWith(afterPrecondition, [targetCrate])
      );
    }
    expect(update.state).toMatchObject({
      phase: "succeeded",
      terminalStableSteps: 2
    });

    let invariantState = createHumanoidMotionOptionMonitorState(phasedContract);
    invariantState = advanceHumanoidMotionOptionMonitor(
      phasedContract,
      invariantState,
      inputWith(robot)
    ).state;
    invariantState = advanceHumanoidMotionOptionMonitor(
      phasedContract,
      invariantState,
      inputWith(robot)
    ).state;
    const missingBody = {
      ...robot,
      links: {}
    };
    const uncertainInvariant = advanceHumanoidMotionOptionMonitor(
      phasedContract,
      invariantState,
      inputWith(missingBody)
    );
    expect(uncertainInvariant).toMatchObject({
      observationStatus: "uncertain",
      state: { phase: "indeterminate", terminalStableSteps: 0 }
    });
    const stillIndeterminate = advanceHumanoidMotionOptionMonitor(
      phasedContract,
      uncertainInvariant.state,
      inputWith(robot, [targetCrate])
    );
    expect(stillIndeterminate).toMatchObject({
      observationStatus: "uncertain",
      state: { phase: "indeterminate", terminalStableSteps: 0 }
    });
    const violatedInvariant = advanceHumanoidMotionOptionMonitor(
      phasedContract,
      invariantState,
      inputWith({
        ...robot,
        links: {
          left_wrist_yaw_link: { position: { x: 9, y: 9, z: 9 } }
        }
      })
    );
    expect(violatedInvariant.state.phase).toBe("violated");
  });
});
