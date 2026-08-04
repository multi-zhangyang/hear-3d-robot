import { describe, expect, it } from "vitest";
import {
  HUMANOID_BODY_NAMES,
  HUMANOID_JOINT_NAMES
} from "./model.js";
import {
  accumulateHumanoidPhysicalSafetyFrame,
  captureHumanoidPhysicalSafetyFrame,
  completeHumanoidPhysicalSafetyEvidence,
  createHumanoidPhysicalSafetyAccumulator,
  HumanoidPhysicalSafetyAccumulatorSchema,
  HumanoidPhysicalSafetyEvidenceSchema,
  signedHorizontalSupportMargin
} from "./physical-safety.js";
import {
  HumanoidSimulation,
  type HumanoidContactSnapshot,
  type HumanoidSimulationSnapshot
} from "./simulation.js";
import { neutralHumanoidReference } from "./reference.js";

describe("humanoid physical safety metrics", () => {
  it("accepts an authoritative MuJoCo snapshot without synthesizing missing metrics", async () => {
    const simulation = await HumanoidSimulation.create();
    try {
      await simulation.step(neutralHumanoidReference());
      const snapshot = simulation.snapshot();
      const evidence = captureHumanoidPhysicalSafetyFrame(0, snapshot);
      expect(evidence.simulated_time_seconds).toBe(snapshot.simulatedTime);
      expect(evidence.contacts.count).toBe(snapshot.contacts.length);
      expect(evidence.joints.minimum_limit_margin_rad.value).toBeTypeOf("number");
      expect(snapshot.joints.left_hip_pitch_joint.effort).toBeDefined();
      expect(evidence.joints.maximum_effort_utilization).toMatchObject({
        requested_utilization: expect.any(Number),
        applied_utilization: expect.any(Number),
        saturated: false
      });
    } finally {
      await simulation.dispose();
    }
  }, 30_000);

  it("computes a yaw-invariant signed margin against the contact convex hull", () => {
    const points = [
      vector(-1, 0, -1),
      vector(1, 0, -1),
      vector(1, 0, 1),
      vector(-1, 0, 1),
      vector(0, 0, 0),
      vector(-1, 0, -1)
    ];
    const center = vector(0.25, 0.8, 0);
    const initial = signedHorizontalSupportMargin(center, points);
    const angle = 0.713;
    const rotated = signedHorizontalSupportMargin(
      yaw(center, angle),
      points.map((point) => yaw(point, angle))
    );

    expect(initial.convexHull).toHaveLength(4);
    expect(initial.signedMarginMeters).toBeCloseTo(0.75, 12);
    expect(rotated.signedMarginMeters).toBeCloseTo(initial.signedMarginMeters!, 12);
    expect(signedHorizontalSupportMargin(vector(1.5, 0.8, 0), points)
      .signedMarginMeters).toBeCloseTo(-0.5, 12);
  });

  it("keeps empty and degenerate support observations explicit", () => {
    expect(signedHorizontalSupportMargin(vector(0, 1, 0), [])).toEqual({
      signedMarginMeters: null,
      convexHull: []
    });
    expect(signedHorizontalSupportMargin(vector(1, 1, 0), [vector(0, 0, 0)]))
      .toMatchObject({ signedMarginMeters: -1 });
    expect(signedHorizontalSupportMargin(
      vector(0, 1, 0.5),
      [vector(0, 0, 0), vector(0, 0, 1)]
    ).signedMarginMeters).toBe(0);
  });

  it("captures relative foot slip, model joint margins, velocity, and contact force", () => {
    const snapshot = safetySnapshot();
    snapshot.links.left_ankle_roll_link.linearVelocity = vector(3, 4, 0);
    snapshot.links.right_ankle_roll_link.linearVelocity = vector(3, 0, 0);
    snapshot.links.right_ankle_roll_link.angularVelocity = vector(0, 1, 0);
    snapshot.objects.platform = objectSnapshot(
      "platform",
      vector(1, 0, 0),
      vector(0, 0.5, 0)
    );
    snapshot.joints.left_hip_pitch_joint.position = 0.95;
    snapshot.joints.right_elbow_joint.velocity = -4.2;
    snapshot.contacts = [
      contact({
        normalForce: 120,
        firstBody: "left_ankle_roll_link"
      }),
      contact({
        position: vector(0, 0, 1),
        normalForce: 80,
        firstBody: "right_ankle_roll_link",
        secondObject: "platform"
      })
    ];
    snapshot.contactCount = snapshot.contacts.length;
    const evidence = captureHumanoidPhysicalSafetyFrame(7, snapshot);

    expect(evidence.support).toMatchObject({
      contact_point_count: 4,
      signed_margin_m: 0.5
    });
    expect(evidence.foot_slip).toEqual({
      left: { contact_count: 1, maximum_tangential_speed_mps: 3 },
      right: { contact_count: 1, maximum_tangential_speed_mps: 2.5 },
      maximum: { foot: "left", tangential_speed_mps: 3 }
    });
    expect(evidence.joints).toEqual({
      minimum_limit_margin_rad: {
        joint: "left_hip_pitch_joint",
        value: 0.050000000000000044
      },
      peak_absolute_velocity_rad_s: {
        joint: "right_elbow_joint",
        value: 4.2
      },
      maximum_effort_utilization: null
    });
    expect(evidence.contacts).toMatchObject({
      count: 2,
      total_normal_force_n: 200,
      peak: {
        contact_index: 0,
        normal_force_n: 120,
        first_body: "left_ankle_roll_link"
      }
    });
  });

  it("accumulates chronological extrema and serializes evidence without sentinels", () => {
    const first = safetySnapshot();
    first.simulatedTime = 1;
    first.links.left_ankle_roll_link.linearVelocity.x = 0.4;
    first.joints.left_hip_pitch_joint.position = 0.8;
    first.joints.right_elbow_joint.velocity = 2;
    first.contacts = [contact({ normalForce: 100, firstBody: "left_ankle_roll_link" })];
    first.contactCount = 1;

    const second = structuredClone(first);
    second.simulatedTime = 1.02;
    second.balance.centerOfMass.x = 0.45;
    second.links.left_ankle_roll_link.linearVelocity.x = 1.6;
    second.joints.left_hip_pitch_joint.position = 1.1;
    second.joints.right_elbow_joint.velocity = -5;
    second.contacts[0]!.normalForce = 160;

    const empty = createHumanoidPhysicalSafetyAccumulator();
    const afterFirst = accumulateHumanoidPhysicalSafetyFrame(empty, 10, first);
    const afterSecond = accumulateHumanoidPhysicalSafetyFrame(afterFirst, 11, second);
    const evidence = completeHumanoidPhysicalSafetyEvidence(afterSecond);

    expect(empty.frame_count).toBe(0);
    expect(afterFirst.frame_count).toBe(1);
    expect(evidence).toMatchObject({
      protocol: "humanoid-physical-safety-evidence-v1",
      frame_count: 2,
      first_frame: 10,
      last_frame: 11,
      first_simulated_time_seconds: 1,
      last_simulated_time_seconds: 1.02,
      minimum_signed_support_margin: {
        frame: 11,
        signed_margin_m: expect.any(Number)
      },
      maximum_foot_tangential_speed: {
        frame: 11,
        foot: "left",
        tangential_speed_mps: 1.6
      },
      minimum_joint_limit_margin: {
        frame: 11,
        joint: "left_hip_pitch_joint",
        margin_rad: expect.any(Number)
      },
      maximum_joint_velocity: {
        frame: 11,
        joint: "right_elbow_joint",
        absolute_velocity_rad_s: 5
      },
      peak_contact_normal_force: {
        frame: 11,
        contact: { normal_force_n: 160 }
      },
      peak_total_normal_force: { frame: 11, total_normal_force_n: 160 },
      peak_total_normal_force_rise_rate: {
        frame: 11,
        previous_frame: 10,
        rise_rate_nps: expect.any(Number)
      }
    });
    expect(evidence.minimum_signed_support_margin!.signed_margin_m).toBeCloseTo(0.05, 12);
    expect(evidence.minimum_joint_limit_margin.margin_rad).toBeCloseTo(-0.1, 12);
    expect(evidence.peak_total_normal_force_rise_rate!.rise_rate_nps)
      .toBeCloseTo(3000, 9);
    expect(HumanoidPhysicalSafetyAccumulatorSchema.parse(
      JSON.parse(JSON.stringify(afterSecond))
    )).toEqual(afterSecond);
    expect(HumanoidPhysicalSafetyEvidenceSchema.parse(
      JSON.parse(JSON.stringify(evidence))
    )).toEqual(evidence);
  });

  it("does not invent contact, support, slip, or impact observations", () => {
    const snapshot = safetySnapshot();
    snapshot.feet.left = emptyFoot();
    snapshot.feet.right = emptyFoot();
    snapshot.contacts = [];
    snapshot.contactCount = 0;
    const frame = captureHumanoidPhysicalSafetyFrame(0, snapshot);
    const evidence = completeHumanoidPhysicalSafetyEvidence(
      accumulateHumanoidPhysicalSafetyFrame(
        createHumanoidPhysicalSafetyAccumulator(),
        0,
        snapshot
      )
    );

    expect(frame.support.signed_margin_m).toBeNull();
    expect(frame.foot_slip.maximum).toBeNull();
    expect(frame.contacts.peak).toBeNull();
    expect(evidence.minimum_signed_support_margin).toBeNull();
    expect(evidence.maximum_foot_tangential_speed).toBeNull();
    expect(evidence.peak_contact_normal_force).toBeNull();
    expect(evidence.peak_total_normal_force_rise_rate).toBeNull();
  });

  it("rejects non-finite, inconsistent, and non-chronological evidence", () => {
    const snapshot = safetySnapshot();
    const invalidCenter = structuredClone(snapshot);
    invalidCenter.balance.centerOfMass.x = Number.NaN;
    expect(() => captureHumanoidPhysicalSafetyFrame(0, invalidCenter)).toThrow("finite");

    const invalidJoint = structuredClone(snapshot);
    invalidJoint.joints.left_knee_joint.velocity = Number.POSITIVE_INFINITY;
    expect(() => captureHumanoidPhysicalSafetyFrame(0, invalidJoint)).toThrow(
      "non-finite joint evidence"
    );

    const invalidRange = structuredClone(snapshot);
    invalidRange.joints.left_knee_joint.minimum = 1;
    invalidRange.joints.left_knee_joint.maximum = 1;
    expect(() => captureHumanoidPhysicalSafetyFrame(0, invalidRange)).toThrow(
      "invalid joint range"
    );

    const invalidNormal = structuredClone(snapshot);
    invalidNormal.contacts[0]!.normal = vector(0, 0, 0);
    expect(() => captureHumanoidPhysicalSafetyFrame(0, invalidNormal)).toThrow(
      "zero contact normal"
    );

    const invalidForce = structuredClone(snapshot);
    invalidForce.contacts[0]!.normalForce = Number.NaN;
    expect(() => captureHumanoidPhysicalSafetyFrame(0, invalidForce)).toThrow(
      "invalid contact force"
    );

    const inconsistentFoot = structuredClone(snapshot);
    inconsistentFoot.feet.left.contactCount = 3;
    expect(() => captureHumanoidPhysicalSafetyFrame(0, inconsistentFoot)).toThrow(
      "inconsistent left foot"
    );

    const incompleteEffort = structuredClone(snapshot);
    incompleteEffort.joints.left_knee_joint.effort = {
      requestedNewtonMeters: 2,
      appliedNewtonMeters: 2,
      minimumNewtonMeters: -10,
      maximumNewtonMeters: 10,
      requestedUtilization: 0.2,
      appliedUtilization: 0.2,
      saturated: false
    };
    expect(() => captureHumanoidPhysicalSafetyFrame(0, incompleteEffort)).toThrow(
      "incomplete actuator effort evidence"
    );

    const first = structuredClone(snapshot);
    first.simulatedTime = 2;
    const state = accumulateHumanoidPhysicalSafetyFrame(
      createHumanoidPhysicalSafetyAccumulator(),
      4,
      first
    );
    const next = structuredClone(snapshot);
    next.simulatedTime = 2.02;
    expect(() => accumulateHumanoidPhysicalSafetyFrame(state, 6, next)).toThrow(
      "frames must be consecutive"
    );
    next.simulatedTime = 2;
    expect(() => accumulateHumanoidPhysicalSafetyFrame(state, 5, next)).toThrow(
      "time must increase"
    );
    const forgedRange = structuredClone(state);
    forgedRange.minimum_joint_limit_margin!.frame = 99;
    expect(() => HumanoidPhysicalSafetyAccumulatorSchema.parse(forgedRange)).toThrow(
      "outside its evidence range"
    );
    expect(() => completeHumanoidPhysicalSafetyEvidence(
      createHumanoidPhysicalSafetyAccumulator()
    )).toThrow("without observed frames");
  });
});

function safetySnapshot(): HumanoidSimulationSnapshot {
  const points = [
    vector(-0.5, 0, -0.5),
    vector(0.5, 0, -0.5),
    vector(0.5, 0, 0.5),
    vector(-0.5, 0, 0.5)
  ];
  const feet = {
    left: {
      touching: true,
      contactCount: 2,
      normalForce: 100,
      points: points.slice(0, 2)
    },
    right: {
      touching: true,
      contactCount: 2,
      normalForce: 100,
      points: points.slice(2)
    }
  };
  const contacts = [contact({ normalForce: 100, firstBody: "left_ankle_roll_link" })];
  return {
    simulatedTime: 0,
    controller: {
      protocol: "humanoid-controller-v1",
      implementation: "test",
      actuation: "joint_position_pd",
      controlStepSeconds: 0.02,
      physicsStepSeconds: 0.005
    },
    rootPosition: vector(0, 0.8, 0),
    rootRotation: { x: 0, y: 0, z: 0, w: 1 },
    joints: Object.fromEntries(HUMANOID_JOINT_NAMES.map((name) => [name, {
      position: 0,
      velocity: 0,
      minimum: -1,
      maximum: 1
    }])) as HumanoidSimulationSnapshot["joints"],
    links: Object.fromEntries(HUMANOID_BODY_NAMES.map((name) => [name, {
      position: vector(0, 0, 0),
      rotation: { x: 0, y: 0, z: 0, w: 1 },
      linearVelocity: vector(0, 0, 0),
      angularVelocity: vector(0, 0, 0)
    }])) as HumanoidSimulationSnapshot["links"],
    objects: {},
    contactCount: contacts.length,
    contacts,
    feet,
    balance: {
      centerOfMass: vector(0, 0.8, 0),
      support: "double",
      supportMargin: 0.5,
      upright: 1
    },
    nonFootEnvironmentContacts: [],
    fallen: false
  };
}

function contact(overrides: Partial<HumanoidContactSnapshot>): HumanoidContactSnapshot {
  return {
    position: vector(0, 0, 0),
    normal: vector(0, 1, 0),
    normalForce: 0,
    firstBody: null,
    secondBody: null,
    firstObject: null,
    secondObject: null,
    ...overrides
  };
}

function emptyFoot(): HumanoidSimulationSnapshot["feet"]["left"] {
  return { touching: false, contactCount: 0, normalForce: 0, points: [] };
}

function objectSnapshot(
  id: string,
  linearVelocity: ReturnType<typeof vector>,
  angularVelocity = vector(0, 0, 0)
): HumanoidSimulationSnapshot["objects"][string] {
  return {
    id,
    position: vector(0, 0, 0),
    rotation: { x: 0, y: 0, z: 0, w: 1 },
    linearVelocity,
    angularVelocity
  };
}

function vector(x: number, y: number, z: number) {
  return { x, y, z };
}

function yaw(point: ReturnType<typeof vector>, radians: number) {
  return {
    x: point.x * Math.cos(radians) - point.z * Math.sin(radians),
    y: point.y,
    z: point.x * Math.sin(radians) + point.z * Math.cos(radians)
  };
}
