import { describe, expect, it } from "vitest";
import {
  HumanoidMotionOptionContractSchema,
  advanceHumanoidMotionOptionMonitor,
  createHumanoidMotionOptionMonitorState,
  detectHumanoidMotionOption,
  type HumanoidMotionOptionContract,
  type HumanoidMotionOptionDetectorInput
} from "./motion-option.js";
import {
  HUMANOID_OBJECT_SETTLED_SUPPORT_AUTHORITY,
  assessHumanoidObjectSettledOnSupport,
  type HumanoidObjectSettledSupportContact,
  type HumanoidObjectSettledSupportSnapshot
} from "./object-settled-support.js";

const still = {
  linearVelocity: { x: 0.005, y: 0, z: -0.004 },
  angularVelocity: { x: 0, y: 0.02, z: 0 }
};

const upwardEnvironmentContact: HumanoidObjectSettledSupportContact = {
  normal: { x: 0, y: 1, z: 0 },
  normalForce: 3,
  firstBody: null,
  secondBody: null,
  firstObject: null,
  secondObject: "crate",
  firstHandLink: null,
  secondHandLink: null
};

function snapshot(
  overrides: Partial<HumanoidObjectSettledSupportSnapshot> = {}
): HumanoidObjectSettledSupportSnapshot {
  return {
    objects: { crate: still },
    contacts: [upwardEnvironmentContact],
    ...overrides
  };
}

describe("object settled on physical support", () => {
  it("keeps every physical threshold under immutable Harness authority", () => {
    expect(HUMANOID_OBJECT_SETTLED_SUPPORT_AUTHORITY).toEqual({
      protocol: "humanoid-object-settled-support-authority-v1",
      worldUp: { x: 0, y: 1, z: 0 },
      minimumUpwardNormalDot: 0.7,
      minimumSupportNormalForceN: 2,
      maximumLinearSpeedMps: 0.03,
      maximumAngularSpeedRadps: 0.1
    });
    expect(Object.isFrozen(HUMANOID_OBJECT_SETTLED_SUPPORT_AUTHORITY)).toBe(true);
    expect(Object.isFrozen(HUMANOID_OBJECT_SETTLED_SUPPORT_AUTHORITY.worldUp)).toBe(true);
    expect(() => HumanoidMotionOptionContractSchema.parse({
      option_id: "model-owned-threshold",
      stable_steps: 2,
      predicates: [{
        type: "object_settled_on_support",
        object_id: "crate",
        maximum_linear_speed_mps: 99
      }]
    })).toThrow();
  });

  it("requires current observability before reading privileged dynamics", () => {
    const assessment = assessHumanoidObjectSettledOnSupport({
      objectId: "crate",
      objectObservable: false,
      snapshot: snapshot()
    });

    expect(assessment).toMatchObject({
      status: "uncertain",
      reason: "object_not_observable",
      evidence: {
        objectObservable: false,
        supportContactCount: null,
        linearVelocity: null,
        angularVelocity: null
      }
    });
  });

  it("keeps missing object dynamics uncertain", () => {
    expect(assessHumanoidObjectSettledOnSupport({
      objectId: "crate",
      objectObservable: true,
      snapshot: snapshot({ objects: {} })
    })).toMatchObject({
      status: "uncertain",
      reason: "object_dynamics_missing",
      evidence: {
        supportContactCount: null,
        linearSpeedMps: null,
        angularSpeedRadps: null
      }
    });
  });

  it("orients MuJoCo normals toward either object side and excludes humanoid support", () => {
    const assessment = assessHumanoidObjectSettledOnSupport({
      objectId: "crate",
      objectObservable: true,
      snapshot: snapshot({
        contacts: [
          upwardEnvironmentContact,
          {
            normal: { x: 0, y: -2, z: 0 },
            normalForce: 4,
            firstBody: null,
            secondBody: null,
            firstObject: "crate",
            secondObject: "table",
            firstHandLink: null,
            secondHandLink: null
          },
          {
            normal: { x: 0, y: 1, z: 0 },
            normalForce: 100,
            firstBody: "left_wrist_yaw_link",
            secondBody: null,
            firstObject: null,
            secondObject: "crate",
            firstHandLink: null,
            secondHandLink: null
          },
          {
            normal: { x: 0, y: 1, z: 0 },
            normalForce: 100,
            firstBody: null,
            secondBody: null,
            firstObject: null,
            secondObject: "crate",
            firstHandLink: "left_hand_palm_link",
            secondHandLink: null
          }
        ]
      })
    });

    expect(assessment).toMatchObject({
      status: "satisfied",
      reason: "object_settled_on_support",
      evidence: {
        supportContactCount: 2,
        upwardSupportContactCount: 2,
        forceQualifiedSupportContactCount: 2,
        maximumUpwardNormalDot: 1,
        maximumNormalForce: 4,
        totalNormalForce: 7,
        totalUpwardSupportForceN: 7,
        linearVelocity: still.linearVelocity,
        angularVelocity: still.angularVelocity
      }
    });
  });

  it("does not count a hand or body holding the object as settled support", () => {
    const assessment = assessHumanoidObjectSettledOnSupport({
      objectId: "crate",
      objectObservable: true,
      snapshot: snapshot({
        contacts: [
          {
            ...upwardEnvironmentContact,
            firstBody: "pelvis"
          },
          {
            ...upwardEnvironmentContact,
            firstHandLink: "right_hand_index_1_link"
          }
        ]
      })
    });

    expect(assessment).toMatchObject({
      status: "unsatisfied",
      reason: "support_contact_missing",
      evidence: {
        supportContactCount: 0,
        upwardSupportContactCount: 0,
        forceQualifiedSupportContactCount: 0
      }
    });
  });

  it("distinguishes missing measurements from measured direction and force failures", () => {
    expect(assessHumanoidObjectSettledOnSupport({
      objectId: "crate",
      objectObservable: true,
      snapshot: snapshot({
        contacts: [{ ...upwardEnvironmentContact, normal: undefined }]
      })
    })).toMatchObject({
      status: "uncertain",
      reason: "support_contact_normal_missing"
    });

    expect(assessHumanoidObjectSettledOnSupport({
      objectId: "crate",
      objectObservable: true,
      snapshot: snapshot({
        contacts: [{
          ...upwardEnvironmentContact,
          normal: { x: 1, y: 0, z: 0 }
        }]
      })
    })).toMatchObject({
      status: "unsatisfied",
      reason: "support_direction_insufficient"
    });

    expect(assessHumanoidObjectSettledOnSupport({
      objectId: "crate",
      objectObservable: true,
      snapshot: snapshot({
        contacts: [{ ...upwardEnvironmentContact, normalForce: 1.99 }]
      })
    })).toMatchObject({
      status: "unsatisfied",
      reason: "support_force_insufficient"
    });
  });

  it("aggregates four upward contact-point forces without requiring one 2N point", () => {
    const assessFourPoints = (normalForce: number) => (
      assessHumanoidObjectSettledOnSupport({
        objectId: "crate",
        objectObservable: true,
        snapshot: snapshot({
          contacts: Array.from({ length: 4 }, () => ({
            ...upwardEnvironmentContact,
            normalForce
          }))
        })
      })
    );

    expect(assessFourPoints(0.55)).toMatchObject({
      status: "satisfied",
      reason: "object_settled_on_support",
      evidence: {
        supportContactCount: 4,
        upwardSupportContactCount: 4,
        forceQualifiedSupportContactCount: 4,
        maximumNormalForce: 0.55,
        totalNormalForce: expect.closeTo(2.2, 12),
        totalUpwardSupportForceN: expect.closeTo(2.2, 12)
      }
    });
    expect(assessFourPoints(0.49)).toMatchObject({
      status: "unsatisfied",
      reason: "support_force_insufficient",
      evidence: {
        supportContactCount: 4,
        upwardSupportContactCount: 4,
        forceQualifiedSupportContactCount: 4,
        maximumNormalForce: 0.49,
        totalNormalForce: expect.closeTo(1.96, 12),
        totalUpwardSupportForceN: expect.closeTo(1.96, 12)
      }
    });
    expect(assessHumanoidObjectSettledOnSupport({
      objectId: "crate",
      objectObservable: true,
      snapshot: snapshot({
        contacts: Array.from({ length: 4 }, () => ({
          ...upwardEnvironmentContact,
          normal: { x: 0.6, y: 0.8, z: 0 },
          normalForce: 0.55
        }))
      })
    })).toMatchObject({
      status: "unsatisfied",
      reason: "support_force_insufficient",
      evidence: {
        totalNormalForce: expect.closeTo(2.2, 12),
        totalUpwardSupportForceN: expect.closeTo(1.76, 12)
      }
    });
  });

  it("requires both linear and angular motion to settle", () => {
    expect(assessHumanoidObjectSettledOnSupport({
      objectId: "crate",
      objectObservable: true,
      snapshot: snapshot({
        objects: {
          crate: {
            ...still,
            linearVelocity: { x: 0.031, y: 0, z: 0 }
          }
        }
      })
    })).toMatchObject({
      status: "unsatisfied",
      reason: "linear_velocity_exceeded"
    });

    expect(assessHumanoidObjectSettledOnSupport({
      objectId: "crate",
      objectObservable: true,
      snapshot: snapshot({
        objects: {
          crate: {
            ...still,
            angularVelocity: { x: 0, y: 0.101, z: 0 }
          }
        }
      })
    })).toMatchObject({
      status: "unsatisfied",
      reason: "angular_velocity_exceeded"
    });
  });

  it("publishes raw support and velocity evidence through the stable Option monitor", () => {
    const contract: HumanoidMotionOptionContract = {
      option_id: "settled-crate",
      stable_steps: 2,
      predicates: [{
        type: "object_settled_on_support",
        object_id: "crate"
      }]
    };
    const input: HumanoidMotionOptionDetectorInput = {
      snapshot: {
        rootPosition: { x: 0, y: 0.8, z: 0 },
        links: {},
        objects: { crate: still },
        contacts: [upwardEnvironmentContact]
      },
      observableObjects: [{
        id: "crate",
        position: { x: 1, y: 0.25, z: 1 },
        rotation: { x: 0, y: 0, z: 0, w: 1 },
        size: { x: 0.5, y: 0.5, z: 0.5 }
      }],
      zones: []
    };
    const detection = detectHumanoidMotionOption(contract, input);
    expect(detection).toMatchObject({
      status: "satisfied",
      evidence: [{
        type: "object_settled_on_support",
        objectId: "crate",
        status: "satisfied",
        reason: "object_settled_on_support",
        supportContactCount: 1,
        upwardSupportContactCount: 1,
        forceQualifiedSupportContactCount: 1,
        totalUpwardSupportForceN: 3,
        linearVelocity: still.linearVelocity,
        angularVelocity: still.angularVelocity,
        authority: HUMANOID_OBJECT_SETTLED_SUPPORT_AUTHORITY
      }]
    });
    expect(detectHumanoidMotionOption(contract, {
      ...input,
      observableObjects: []
    })).toMatchObject({
      status: "uncertain",
      evidence: [{
        reason: "object_not_observable",
        supportContactCount: null,
        linearVelocity: null,
        angularVelocity: null
      }]
    });

    let state = createHumanoidMotionOptionMonitorState(contract);
    state = advanceHumanoidMotionOptionMonitor(contract, state, input).state;
    expect(state).toMatchObject({ phase: "running", terminalStableSteps: 1 });
    state = advanceHumanoidMotionOptionMonitor(contract, state, input).state;
    expect(state).toMatchObject({ phase: "succeeded", terminalStableSteps: 2 });
  });
});
