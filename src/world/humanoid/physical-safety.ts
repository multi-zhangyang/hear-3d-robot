import { z } from "zod";
import { Vec3Schema, type Vec3 } from "../../domain/schema.js";
import {
  HUMANOID_BODY_NAMES,
  HUMANOID_JOINT_NAMES,
  type HumanoidBodyName,
  type HumanoidJointName
} from "./model.js";
import type {
  HumanoidContactSnapshot,
  HumanoidSimulationSnapshot
} from "./simulation.js";

const FootSchema = z.enum(["left", "right"]);
const JointNameSchema = z.enum(HUMANOID_JOINT_NAMES);
const BodyNameSchema = z.enum(HUMANOID_BODY_NAMES);
const ProjectedPointSchema = z.object({
  x: z.number().finite(),
  z: z.number().finite()
}).strict();

const ExtremumFrameSchema = z.object({
  frame: z.number().int().nonnegative(),
  simulated_time_seconds: z.number().finite().nonnegative()
}).strict();

const SupportExtremumSchema = ExtremumFrameSchema.extend({
  signed_margin_m: z.number().finite()
}).strict();

const FootSlipExtremumSchema = ExtremumFrameSchema.extend({
  foot: FootSchema,
  tangential_speed_mps: z.number().finite().nonnegative()
}).strict();

const JointLimitExtremumSchema = ExtremumFrameSchema.extend({
  joint: JointNameSchema,
  margin_rad: z.number().finite()
}).strict();

const JointVelocityExtremumSchema = ExtremumFrameSchema.extend({
  joint: JointNameSchema,
  absolute_velocity_rad_s: z.number().finite().nonnegative()
}).strict();

const ActuatorEffortMetricSchema = z.object({
  joint: JointNameSchema,
  requested_newton_meters: z.number().finite(),
  applied_newton_meters: z.number().finite(),
  requested_utilization: z.number().finite().nonnegative(),
  applied_utilization: z.number().finite().nonnegative().max(1 + 1e-9),
  saturated: z.boolean()
}).strict();

const ActuatorEffortExtremumSchema = ExtremumFrameSchema.extend({
  ...ActuatorEffortMetricSchema.shape
}).strict();

const PeakContactSchema = z.object({
  contact_index: z.number().int().nonnegative(),
  normal_force_n: z.number().finite().nonnegative(),
  position: Vec3Schema,
  first_body: BodyNameSchema.nullable(),
  second_body: BodyNameSchema.nullable(),
  first_object: z.string().min(1).nullable(),
  second_object: z.string().min(1).nullable()
}).strict();

const ContactForceExtremumSchema = ExtremumFrameSchema.extend({
  contact: PeakContactSchema
}).strict();

const TotalForceExtremumSchema = ExtremumFrameSchema.extend({
  total_normal_force_n: z.number().finite().nonnegative()
}).strict();

const ForceRiseExtremumSchema = ExtremumFrameSchema.extend({
  previous_frame: z.number().int().nonnegative(),
  rise_rate_nps: z.number().finite().nonnegative()
}).strict();

const JointFrameMetricSchema = z.object({
  joint: JointNameSchema,
  value: z.number().finite()
}).strict();

const FootSlipFrameMetricSchema = z.object({
  contact_count: z.number().int().nonnegative(),
  maximum_tangential_speed_mps: z.number().finite().nonnegative().nullable()
}).strict().superRefine((metric, context) => {
  if ((metric.contact_count === 0) !== (metric.maximum_tangential_speed_mps === null)) {
    context.addIssue({
      code: "custom",
      message: "Foot slip speed must exist exactly when a foot has a measured contact"
    });
  }
});

const HumanoidPhysicalSafetyFrameEvidenceSchema = z.object({
  frame: z.number().int().nonnegative(),
  simulated_time_seconds: z.number().finite().nonnegative(),
  support: z.object({
    contact_point_count: z.number().int().nonnegative(),
    convex_hull: z.array(ProjectedPointSchema),
    signed_margin_m: z.number().finite().nullable()
  }).strict(),
  foot_slip: z.object({
    left: FootSlipFrameMetricSchema,
    right: FootSlipFrameMetricSchema,
    maximum: z.object({
      foot: FootSchema,
      tangential_speed_mps: z.number().finite().nonnegative()
    }).strict().nullable()
  }).strict(),
  joints: z.object({
    minimum_limit_margin_rad: JointFrameMetricSchema,
    peak_absolute_velocity_rad_s: JointFrameMetricSchema.extend({
      value: z.number().finite().nonnegative()
    }).strict(),
    maximum_effort_utilization: ActuatorEffortMetricSchema.nullable()
  }).strict(),
  contacts: z.object({
    count: z.number().int().nonnegative(),
    total_normal_force_n: z.number().finite().nonnegative(),
    peak: PeakContactSchema.nullable()
  }).strict()
}).strict().superRefine((evidence, context) => {
  const hasSupport = evidence.support.contact_point_count > 0;
  if (hasSupport !== (evidence.support.signed_margin_m !== null)
    || hasSupport !== (evidence.support.convex_hull.length > 0)) {
    context.addIssue({
      code: "custom",
      path: ["support"],
      message: "Support evidence must preserve observed contact geometry"
    });
  }
  const footCandidates = ([
    ["left", evidence.foot_slip.left.maximum_tangential_speed_mps],
    ["right", evidence.foot_slip.right.maximum_tangential_speed_mps]
  ] as const).filter(
    (entry): entry is readonly ["left" | "right", number] => entry[1] !== null
  );
  const expectedMaximum = footCandidates.length === 0 ? null : footCandidates.reduce(
    (best, entry) => entry[1] > best[1] ? entry : best
  );
  if ((expectedMaximum === null) !== (evidence.foot_slip.maximum === null)
    || (expectedMaximum !== null && evidence.foot_slip.maximum !== null
      && (evidence.foot_slip.maximum.foot !== expectedMaximum[0]
        || evidence.foot_slip.maximum.tangential_speed_mps !== expectedMaximum[1]))) {
    context.addIssue({
      code: "custom",
      path: ["foot_slip", "maximum"],
      message: "Maximum foot slip does not match the measured foot contacts"
    });
  }
  if ((evidence.contacts.count === 0) !== (evidence.contacts.peak === null)
    || (evidence.contacts.peak !== null
      && evidence.contacts.peak.contact_index >= evidence.contacts.count)) {
    context.addIssue({
      code: "custom",
      path: ["contacts", "peak"],
      message: "Peak contact evidence does not match the observed contact count"
    });
  }
});

export type HumanoidPhysicalSafetyFrameEvidence = z.infer<
  typeof HumanoidPhysicalSafetyFrameEvidenceSchema
>;

export const HumanoidPhysicalSafetyAccumulatorSchema = z.object({
  protocol: z.literal("humanoid-physical-safety-accumulator-v1"),
  frame_count: z.number().int().nonnegative(),
  first_frame: z.number().int().nonnegative().nullable(),
  last_frame: z.number().int().nonnegative().nullable(),
  first_simulated_time_seconds: z.number().finite().nonnegative().nullable(),
  last_simulated_time_seconds: z.number().finite().nonnegative().nullable(),
  previous_total_normal_force_n: z.number().finite().nonnegative().nullable(),
  minimum_signed_support_margin: SupportExtremumSchema.nullable(),
  maximum_foot_tangential_speed: FootSlipExtremumSchema.nullable(),
  minimum_joint_limit_margin: JointLimitExtremumSchema.nullable(),
  maximum_joint_velocity: JointVelocityExtremumSchema.nullable(),
  maximum_actuator_effort_utilization: ActuatorEffortExtremumSchema.nullable(),
  peak_contact_normal_force: ContactForceExtremumSchema.nullable(),
  peak_total_normal_force: TotalForceExtremumSchema.nullable(),
  peak_total_normal_force_rise_rate: ForceRiseExtremumSchema.nullable()
}).strict().superRefine((state, context) => {
  const empty = state.frame_count === 0;
  const requiredNullableFields = [
    state.first_frame,
    state.last_frame,
    state.first_simulated_time_seconds,
    state.last_simulated_time_seconds,
    state.previous_total_normal_force_n,
    state.minimum_joint_limit_margin,
    state.maximum_joint_velocity,
    state.peak_total_normal_force
  ];
  if (empty && requiredNullableFields.some((value) => value !== null)) {
    context.addIssue({ code: "custom", message: "An empty safety accumulator cannot contain frame evidence" });
  }
  if (!empty && requiredNullableFields.some((value) => value === null)) {
    context.addIssue({ code: "custom", message: "A populated safety accumulator is missing frame evidence" });
  }
  if (!empty && state.first_frame !== null && state.last_frame !== null
    && state.last_frame - state.first_frame + 1 !== state.frame_count) {
    context.addIssue({ code: "custom", message: "Safety accumulator frames must be consecutive" });
  }
  if (!empty && state.first_simulated_time_seconds !== null
    && state.last_simulated_time_seconds !== null
    && state.last_simulated_time_seconds < state.first_simulated_time_seconds) {
    context.addIssue({ code: "custom", message: "Safety accumulator time range is invalid" });
  }
  validateExtremumFrames(state, context);
});

export type HumanoidPhysicalSafetyAccumulator = z.infer<
  typeof HumanoidPhysicalSafetyAccumulatorSchema
>;

export const HumanoidPhysicalSafetyEvidenceSchema = z.object({
  protocol: z.literal("humanoid-physical-safety-evidence-v1"),
  frame_count: z.number().int().positive(),
  first_frame: z.number().int().nonnegative(),
  last_frame: z.number().int().nonnegative(),
  first_simulated_time_seconds: z.number().finite().nonnegative(),
  last_simulated_time_seconds: z.number().finite().nonnegative(),
  minimum_signed_support_margin: SupportExtremumSchema.nullable(),
  maximum_foot_tangential_speed: FootSlipExtremumSchema.nullable(),
  minimum_joint_limit_margin: JointLimitExtremumSchema,
  maximum_joint_velocity: JointVelocityExtremumSchema,
  maximum_actuator_effort_utilization: ActuatorEffortExtremumSchema.nullable(),
  peak_contact_normal_force: ContactForceExtremumSchema.nullable(),
  peak_total_normal_force: TotalForceExtremumSchema,
  peak_total_normal_force_rise_rate: ForceRiseExtremumSchema.nullable()
}).strict().superRefine((evidence, context) => {
  if (evidence.last_frame - evidence.first_frame + 1 !== evidence.frame_count) {
    context.addIssue({ code: "custom", message: "Physical safety evidence frames must be consecutive" });
  }
  if (evidence.last_simulated_time_seconds < evidence.first_simulated_time_seconds) {
    context.addIssue({ code: "custom", message: "Physical safety evidence time range is invalid" });
  }
  validateExtremumFrames(evidence, context);
});

export type HumanoidPhysicalSafetyEvidence = z.infer<
  typeof HumanoidPhysicalSafetyEvidenceSchema
>;

interface ProjectedPoint {
  x: number;
  z: number;
}

const FOOT_LINKS = {
  left: "left_ankle_roll_link",
  right: "right_ankle_roll_link"
} as const satisfies Record<"left" | "right", HumanoidBodyName>;
const HUMANOID_BODY_NAME_SET = new Set<string>(HUMANOID_BODY_NAMES);

export function signedHorizontalSupportMargin(
  centerOfMass: Vec3,
  contactPoints: readonly Vec3[]
): { signedMarginMeters: number | null; convexHull: ProjectedPoint[] } {
  assertVector(centerOfMass, "center of mass");
  contactPoints.forEach((point, index) => assertVector(point, `contact point ${index}`));
  const center = { x: centerOfMass.x, z: centerOfMass.z };
  const hull = convexHull(contactPoints.map(({ x, z }) => ({ x, z })));
  if (hull.length === 0) return { signedMarginMeters: null, convexHull: [] };
  if (hull.length === 1) {
    const distance = pointDistance(center, hull[0]!);
    return {
      signedMarginMeters: distance === 0 ? 0 : -distance,
      convexHull: hull
    };
  }
  if (hull.length === 2) {
    const distance = pointSegmentDistance(center, hull[0]!, hull[1]!);
    return {
      signedMarginMeters: distance === 0 ? 0 : -distance,
      convexHull: hull
    };
  }
  const inside = hull.every((point, index) => (
    cross(point, hull[(index + 1) % hull.length]!, center) >= -1e-12
  ));
  const distance = Math.min(...hull.map((point, index) => (
    pointSegmentDistance(center, point, hull[(index + 1) % hull.length]!)
  )));
  return {
    signedMarginMeters: distance === 0 ? 0 : inside ? distance : -distance,
    convexHull: hull
  };
}

export function captureHumanoidPhysicalSafetyFrame(
  frame: number,
  snapshot: HumanoidSimulationSnapshot
): HumanoidPhysicalSafetyFrameEvidence {
  assertFrame(frame);
  assertSnapshotInputs(snapshot);
  const contactPoints = [...snapshot.feet.left.points, ...snapshot.feet.right.points];
  const support = signedHorizontalSupportMargin(snapshot.balance.centerOfMass, contactPoints);
  const leftSlip = footSlipMetric("left", snapshot);
  const rightSlip = footSlipMetric("right", snapshot);
  const maximumSlip = maximumFootSlip(leftSlip, rightSlip);
  const jointMetrics = jointSafetyMetrics(snapshot);
  const peakContact = peakContactMetric(snapshot.contacts);
  return HumanoidPhysicalSafetyFrameEvidenceSchema.parse({
    frame,
    simulated_time_seconds: snapshot.simulatedTime,
    support: {
      contact_point_count: contactPoints.length,
      convex_hull: support.convexHull,
      signed_margin_m: support.signedMarginMeters
    },
    foot_slip: {
      left: leftSlip,
      right: rightSlip,
      maximum: maximumSlip
    },
    joints: jointMetrics,
    contacts: {
      count: snapshot.contacts.length,
      total_normal_force_n: snapshot.contacts.reduce(
        (sum, contact) => sum + contact.normalForce,
        0
      ),
      peak: peakContact
    }
  });
}

export function createHumanoidPhysicalSafetyAccumulator(): HumanoidPhysicalSafetyAccumulator {
  return HumanoidPhysicalSafetyAccumulatorSchema.parse({
    protocol: "humanoid-physical-safety-accumulator-v1",
    frame_count: 0,
    first_frame: null,
    last_frame: null,
    first_simulated_time_seconds: null,
    last_simulated_time_seconds: null,
    previous_total_normal_force_n: null,
    minimum_signed_support_margin: null,
    maximum_foot_tangential_speed: null,
    minimum_joint_limit_margin: null,
    maximum_joint_velocity: null,
    maximum_actuator_effort_utilization: null,
    peak_contact_normal_force: null,
    peak_total_normal_force: null,
    peak_total_normal_force_rise_rate: null
  });
}

export function accumulateHumanoidPhysicalSafetyFrame(
  accumulator: HumanoidPhysicalSafetyAccumulator,
  frame: number,
  snapshot: HumanoidSimulationSnapshot
): HumanoidPhysicalSafetyAccumulator {
  const state = HumanoidPhysicalSafetyAccumulatorSchema.parse(accumulator);
  const evidence = captureHumanoidPhysicalSafetyFrame(frame, snapshot);
  assertNextFrame(state, evidence);
  const extremumFrame = {
    frame: evidence.frame,
    simulated_time_seconds: evidence.simulated_time_seconds
  };
  const support = evidence.support.signed_margin_m === null ? null : {
    ...extremumFrame,
    signed_margin_m: evidence.support.signed_margin_m
  };
  const slip = evidence.foot_slip.maximum === null ? null : {
    ...extremumFrame,
    foot: evidence.foot_slip.maximum.foot,
    tangential_speed_mps: evidence.foot_slip.maximum.tangential_speed_mps
  };
  const jointLimit = {
    ...extremumFrame,
    joint: evidence.joints.minimum_limit_margin_rad.joint,
    margin_rad: evidence.joints.minimum_limit_margin_rad.value
  };
  const jointVelocity = {
    ...extremumFrame,
    joint: evidence.joints.peak_absolute_velocity_rad_s.joint,
    absolute_velocity_rad_s: evidence.joints.peak_absolute_velocity_rad_s.value
  };
  const actuatorEffort = evidence.joints.maximum_effort_utilization === null
    ? null
    : {
        ...extremumFrame,
        ...evidence.joints.maximum_effort_utilization
      };
  const contactForce = evidence.contacts.peak === null ? null : {
    ...extremumFrame,
    contact: evidence.contacts.peak
  };
  const totalForce = {
    ...extremumFrame,
    total_normal_force_n: evidence.contacts.total_normal_force_n
  };
  const forceRise = forceRiseEvidence(state, evidence);
  return HumanoidPhysicalSafetyAccumulatorSchema.parse({
    protocol: state.protocol,
    frame_count: state.frame_count + 1,
    first_frame: state.first_frame ?? evidence.frame,
    last_frame: evidence.frame,
    first_simulated_time_seconds: state.first_simulated_time_seconds
      ?? evidence.simulated_time_seconds,
    last_simulated_time_seconds: evidence.simulated_time_seconds,
    previous_total_normal_force_n: evidence.contacts.total_normal_force_n,
    minimum_signed_support_margin: minimumBy(
      state.minimum_signed_support_margin,
      support,
      (entry) => entry.signed_margin_m
    ),
    maximum_foot_tangential_speed: maximumBy(
      state.maximum_foot_tangential_speed,
      slip,
      (entry) => entry.tangential_speed_mps
    ),
    minimum_joint_limit_margin: minimumBy(
      state.minimum_joint_limit_margin,
      jointLimit,
      (entry) => entry.margin_rad
    ),
    maximum_joint_velocity: maximumBy(
      state.maximum_joint_velocity,
      jointVelocity,
      (entry) => entry.absolute_velocity_rad_s
    ),
    maximum_actuator_effort_utilization: maximumBy(
      state.maximum_actuator_effort_utilization,
      actuatorEffort,
      (entry) => entry.requested_utilization
    ),
    peak_contact_normal_force: maximumBy(
      state.peak_contact_normal_force,
      contactForce,
      (entry) => entry.contact.normal_force_n
    ),
    peak_total_normal_force: maximumBy(
      state.peak_total_normal_force,
      totalForce,
      (entry) => entry.total_normal_force_n
    ),
    peak_total_normal_force_rise_rate: maximumBy(
      state.peak_total_normal_force_rise_rate,
      forceRise,
      (entry) => entry.rise_rate_nps
    )
  });
}

export function completeHumanoidPhysicalSafetyEvidence(
  accumulator: HumanoidPhysicalSafetyAccumulator
): HumanoidPhysicalSafetyEvidence {
  const state = HumanoidPhysicalSafetyAccumulatorSchema.parse(accumulator);
  if (state.frame_count === 0 || state.first_frame === null || state.last_frame === null
    || state.first_simulated_time_seconds === null
    || state.last_simulated_time_seconds === null
    || state.minimum_joint_limit_margin === null || state.maximum_joint_velocity === null
    || state.peak_total_normal_force === null) {
    throw new Error("Cannot complete physical safety evidence without observed frames");
  }
  return HumanoidPhysicalSafetyEvidenceSchema.parse({
    protocol: "humanoid-physical-safety-evidence-v1",
    frame_count: state.frame_count,
    first_frame: state.first_frame,
    last_frame: state.last_frame,
    first_simulated_time_seconds: state.first_simulated_time_seconds,
    last_simulated_time_seconds: state.last_simulated_time_seconds,
    minimum_signed_support_margin: state.minimum_signed_support_margin,
    maximum_foot_tangential_speed: state.maximum_foot_tangential_speed,
    minimum_joint_limit_margin: state.minimum_joint_limit_margin,
    maximum_joint_velocity: state.maximum_joint_velocity,
    maximum_actuator_effort_utilization: state.maximum_actuator_effort_utilization,
    peak_contact_normal_force: state.peak_contact_normal_force,
    peak_total_normal_force: state.peak_total_normal_force,
    peak_total_normal_force_rise_rate: state.peak_total_normal_force_rise_rate
  });
}

function assertSnapshotInputs(snapshot: HumanoidSimulationSnapshot): void {
  if (!Number.isFinite(snapshot.simulatedTime) || snapshot.simulatedTime < 0) {
    throw new Error("Humanoid safety frame has an invalid simulated time");
  }
  assertVector(snapshot.balance.centerOfMass, "center of mass");
  for (const [side, foot] of Object.entries(snapshot.feet)) {
    if (!Number.isInteger(foot.contactCount) || foot.contactCount < 0
      || !Number.isFinite(foot.normalForce) || foot.normalForce < 0
      || foot.contactCount !== foot.points.length
      || foot.touching !== (foot.contactCount > 0)) {
      throw new Error(`Humanoid safety frame has inconsistent ${side} foot contact evidence`);
    }
    foot.points.forEach((point, index) => assertVector(point, `${side} foot point ${index}`));
  }
  for (const side of ["left", "right"] as const) {
    const link = snapshot.links[FOOT_LINKS[side]];
    if (!link) throw new Error(`Humanoid safety frame is missing the ${side} foot link`);
    assertVector(link.position, `${side} foot position`);
    assertVector(link.linearVelocity, `${side} foot linear velocity`);
    assertVector(link.angularVelocity, `${side} foot angular velocity`);
  }
  if (!Number.isInteger(snapshot.contactCount) || snapshot.contactCount < 0
    || snapshot.contactCount !== snapshot.contacts.length) {
    throw new Error("Humanoid safety frame has inconsistent contact count evidence");
  }
  snapshot.contacts.forEach((contact, index) => assertContact(contact, index, snapshot));
  for (const name of HUMANOID_JOINT_NAMES) {
    const joint = snapshot.joints[name];
    if (!joint || ![joint.position, joint.velocity, joint.minimum, joint.maximum]
      .every(Number.isFinite)) {
      throw new Error(`Humanoid safety frame has non-finite joint evidence: ${name}`);
    }
    if (joint.minimum >= joint.maximum) {
      throw new Error(`Humanoid safety frame has an invalid joint range: ${name}`);
    }
  }
  const effortCount = HUMANOID_JOINT_NAMES.reduce(
    (count, name) => count + (snapshot.joints[name].effort ? 1 : 0),
    0
  );
  if (effortCount !== 0 && effortCount !== HUMANOID_JOINT_NAMES.length) {
    throw new Error("Humanoid safety frame has incomplete actuator effort evidence");
  }
  for (const name of HUMANOID_JOINT_NAMES) {
    const effort = snapshot.joints[name].effort;
    if (!effort) continue;
    const values = [
      effort.requestedNewtonMeters,
      effort.appliedNewtonMeters,
      effort.minimumNewtonMeters,
      effort.maximumNewtonMeters,
      effort.requestedUtilization,
      effort.appliedUtilization
    ];
    const limit = Math.max(
      Math.abs(effort.minimumNewtonMeters),
      Math.abs(effort.maximumNewtonMeters)
    );
    const expectedSaturation = effort.requestedNewtonMeters < effort.minimumNewtonMeters
      || effort.requestedNewtonMeters > effort.maximumNewtonMeters;
    if (!values.every(Number.isFinite)
      || effort.minimumNewtonMeters >= effort.maximumNewtonMeters
      || limit <= 0
      || effort.requestedUtilization < 0
      || effort.appliedUtilization < 0
      || effort.appliedUtilization > 1 + 1e-9
      || effort.appliedNewtonMeters < effort.minimumNewtonMeters - 1e-9
      || effort.appliedNewtonMeters > effort.maximumNewtonMeters + 1e-9
      || Math.abs(
        effort.requestedUtilization - Math.abs(effort.requestedNewtonMeters) / limit
      ) > 1e-9
      || Math.abs(
        effort.appliedUtilization - Math.abs(effort.appliedNewtonMeters) / limit
      ) > 1e-9
      || effort.saturated !== expectedSaturation) {
      throw new Error(`Humanoid safety frame has invalid actuator effort evidence: ${name}`);
    }
  }
}

function assertContact(
  contact: HumanoidContactSnapshot,
  index: number,
  snapshot: HumanoidSimulationSnapshot
): void {
  assertVector(contact.position, `contact ${index} position`);
  assertVector(contact.normal, `contact ${index} normal`);
  const normalMagnitude = vectorMagnitude(contact.normal);
  if (!Number.isFinite(normalMagnitude) || normalMagnitude <= 1e-12) {
    throw new Error(`Humanoid safety frame has a zero contact normal at index ${index}`);
  }
  if (!Number.isFinite(contact.normalForce) || contact.normalForce < 0) {
    throw new Error(`Humanoid safety frame has an invalid contact force at index ${index}`);
  }
  for (const body of [contact.firstBody, contact.secondBody]) {
    if (body !== null && !HUMANOID_BODY_NAME_SET.has(body)) {
      throw new Error(`Humanoid safety contact references an unknown body: ${body}`);
    }
  }
  for (const objectId of [contact.firstObject, contact.secondObject]) {
    if (objectId !== null && (typeof objectId !== "string" || objectId.trim().length === 0)) {
      throw new Error(`Humanoid safety contact has an invalid object identity at index ${index}`);
    }
    if (objectId !== null && !snapshot.objects[objectId]) {
      throw new Error(`Humanoid safety contact references a missing object: ${objectId}`);
    }
    if (objectId !== null) {
      const object = snapshot.objects[objectId]!;
      if (object.id !== objectId) {
        throw new Error(`Humanoid safety object identity does not match its key: ${objectId}`);
      }
      assertVector(object.position, `object ${objectId} position`);
      assertVector(object.linearVelocity, `object ${objectId} linear velocity`);
      assertVector(object.angularVelocity, `object ${objectId} angular velocity`);
    }
  }
}

function footSlipMetric(
  side: "left" | "right",
  snapshot: HumanoidSimulationSnapshot
): z.infer<typeof FootSlipFrameMetricSchema> {
  const body = FOOT_LINKS[side];
  const foot = snapshot.links[body];
  const speeds = snapshot.contacts.flatMap((contact) => {
    const footIsFirst = contact.firstBody === body && contact.secondBody === null;
    const footIsSecond = contact.secondBody === body && contact.firstBody === null;
    if (!footIsFirst && !footIsSecond) return [];
    const supportObject = footIsFirst ? contact.secondObject : contact.firstObject;
    const footVelocity = pointVelocity(foot, contact.position);
    const supportVelocity = supportObject === null
      ? { x: 0, y: 0, z: 0 }
      : pointVelocity(snapshot.objects[supportObject]!, contact.position);
    const relative = {
      x: footVelocity.x - supportVelocity.x,
      y: footVelocity.y - supportVelocity.y,
      z: footVelocity.z - supportVelocity.z
    };
    const normalMagnitude = vectorMagnitude(contact.normal);
    const normal = {
      x: contact.normal.x / normalMagnitude,
      y: contact.normal.y / normalMagnitude,
      z: contact.normal.z / normalMagnitude
    };
    const normalSpeed = dot3(relative, normal);
    return [vectorMagnitude({
      x: relative.x - normalSpeed * normal.x,
      y: relative.y - normalSpeed * normal.y,
      z: relative.z - normalSpeed * normal.z
    })];
  });
  return FootSlipFrameMetricSchema.parse({
    contact_count: speeds.length,
    maximum_tangential_speed_mps: speeds.length === 0 ? null : Math.max(...speeds)
  });
}

function maximumFootSlip(
  left: z.infer<typeof FootSlipFrameMetricSchema>,
  right: z.infer<typeof FootSlipFrameMetricSchema>
): { foot: "left" | "right"; tangential_speed_mps: number } | null {
  const candidates = ([
    ["left", left.maximum_tangential_speed_mps],
    ["right", right.maximum_tangential_speed_mps]
  ] as const).filter(
    (candidate): candidate is readonly ["left" | "right", number] => candidate[1] !== null
  );
  if (candidates.length === 0) return null;
  const [foot, speed] = candidates.reduce((best, candidate) => (
    candidate[1] > best[1] ? candidate : best
  ));
  return { foot, tangential_speed_mps: speed };
}

function jointSafetyMetrics(snapshot: HumanoidSimulationSnapshot): {
  minimum_limit_margin_rad: { joint: HumanoidJointName; value: number };
  peak_absolute_velocity_rad_s: { joint: HumanoidJointName; value: number };
  maximum_effort_utilization: z.infer<typeof ActuatorEffortMetricSchema> | null;
} {
  const [first, ...remaining] = HUMANOID_JOINT_NAMES;
  const initial = snapshot.joints[first];
  let minimum: { joint: HumanoidJointName; value: number } = {
    joint: first,
    value: Math.min(initial.position - initial.minimum, initial.maximum - initial.position)
  };
  let peak: { joint: HumanoidJointName; value: number } = {
    joint: first,
    value: Math.abs(initial.velocity)
  };
  for (const name of remaining) {
    const joint = snapshot.joints[name];
    const margin = Math.min(joint.position - joint.minimum, joint.maximum - joint.position);
    const velocity = Math.abs(joint.velocity);
    if (margin < minimum.value) minimum = { joint: name, value: margin };
    if (velocity > peak.value) peak = { joint: name, value: velocity };
  }
  return {
    minimum_limit_margin_rad: minimum,
    peak_absolute_velocity_rad_s: peak,
    maximum_effort_utilization: maximumActuatorEffortMetric(snapshot)
  };
}

function maximumActuatorEffortMetric(
  snapshot: HumanoidSimulationSnapshot
): z.infer<typeof ActuatorEffortMetricSchema> | null {
  let maximum: z.infer<typeof ActuatorEffortMetricSchema> | null = null;
  for (const name of HUMANOID_JOINT_NAMES) {
    const effort = snapshot.joints[name].effort;
    if (!effort) continue;
    const metric = {
      joint: name,
      requested_newton_meters: effort.requestedNewtonMeters,
      applied_newton_meters: effort.appliedNewtonMeters,
      requested_utilization: effort.requestedUtilization,
      applied_utilization: effort.appliedUtilization,
      saturated: effort.saturated
    };
    if (!maximum || metric.requested_utilization > maximum.requested_utilization) {
      maximum = metric;
    }
  }
  return maximum;
}

function peakContactMetric(
  contacts: readonly HumanoidContactSnapshot[]
): z.infer<typeof PeakContactSchema> | null {
  if (contacts.length === 0) return null;
  let peakIndex = 0;
  for (let index = 1; index < contacts.length; index += 1) {
    if (contacts[index]!.normalForce > contacts[peakIndex]!.normalForce) peakIndex = index;
  }
  const contact = contacts[peakIndex]!;
  return PeakContactSchema.parse({
    contact_index: peakIndex,
    normal_force_n: contact.normalForce,
    position: contact.position,
    first_body: contact.firstBody,
    second_body: contact.secondBody,
    first_object: contact.firstObject,
    second_object: contact.secondObject
  });
}

function forceRiseEvidence(
  state: HumanoidPhysicalSafetyAccumulator,
  evidence: HumanoidPhysicalSafetyFrameEvidence
): z.infer<typeof ForceRiseExtremumSchema> | null {
  if (state.last_frame === null || state.last_simulated_time_seconds === null
    || state.previous_total_normal_force_n === null) return null;
  const elapsed = evidence.simulated_time_seconds - state.last_simulated_time_seconds;
  return {
    frame: evidence.frame,
    previous_frame: state.last_frame,
    simulated_time_seconds: evidence.simulated_time_seconds,
    rise_rate_nps: Math.max(
      0,
      evidence.contacts.total_normal_force_n - state.previous_total_normal_force_n
    ) / elapsed
  };
}

function assertNextFrame(
  state: HumanoidPhysicalSafetyAccumulator,
  evidence: HumanoidPhysicalSafetyFrameEvidence
): void {
  if (state.last_frame !== null && evidence.frame !== state.last_frame + 1) {
    throw new Error("Humanoid physical safety frames must be consecutive");
  }
  if (state.last_simulated_time_seconds !== null
    && evidence.simulated_time_seconds <= state.last_simulated_time_seconds) {
    throw new Error("Humanoid physical safety frame time must increase");
  }
}

function validateExtremumFrames(
  state: {
    first_frame: number | null;
    last_frame: number | null;
    first_simulated_time_seconds: number | null;
    last_simulated_time_seconds: number | null;
    minimum_signed_support_margin: ExtremumReference | null;
    maximum_foot_tangential_speed: ExtremumReference | null;
    minimum_joint_limit_margin: ExtremumReference | null;
    maximum_joint_velocity: ExtremumReference | null;
    maximum_actuator_effort_utilization: ExtremumReference | null;
    peak_contact_normal_force: ExtremumReference | null;
    peak_total_normal_force: ExtremumReference | null;
    peak_total_normal_force_rise_rate: (ExtremumReference & { previous_frame: number }) | null;
  },
  context: { addIssue(issue: { code: "custom"; message: string }): void }
): void {
  const extrema = [
    state.minimum_signed_support_margin,
    state.maximum_foot_tangential_speed,
    state.minimum_joint_limit_margin,
    state.maximum_joint_velocity,
    state.maximum_actuator_effort_utilization,
    state.peak_contact_normal_force,
    state.peak_total_normal_force,
    state.peak_total_normal_force_rise_rate
  ];
  for (const extremum of extrema) {
    if (extremum === null) continue;
    if (state.first_frame === null || state.last_frame === null
      || extremum.frame < state.first_frame || extremum.frame > state.last_frame) {
      context.addIssue({
        code: "custom",
        message: "Physical safety extremum references a frame outside its evidence range"
      });
    }
    if (state.first_simulated_time_seconds === null
      || state.last_simulated_time_seconds === null
      || extremum.simulated_time_seconds < state.first_simulated_time_seconds
      || extremum.simulated_time_seconds > state.last_simulated_time_seconds) {
      context.addIssue({
        code: "custom",
        message: "Physical safety extremum references a time outside its evidence range"
      });
    }
  }
  const rise = state.peak_total_normal_force_rise_rate;
  if (rise !== null && rise.previous_frame !== rise.frame - 1) {
    context.addIssue({
      code: "custom",
      message: "Physical safety force-rise evidence must reference the preceding frame"
    });
  }
}

interface ExtremumReference {
  frame: number;
  simulated_time_seconds: number;
}

function convexHull(points: readonly ProjectedPoint[]): ProjectedPoint[] {
  const unique = [...new Map(points.map((point) => [`${point.x}:${point.z}`, point])).values()]
    .sort((left, right) => left.x - right.x || left.z - right.z);
  if (unique.length <= 1) return unique.map((point) => ({ ...point }));
  const lower: ProjectedPoint[] = [];
  for (const point of unique) {
    while (lower.length >= 2
      && cross(lower[lower.length - 2]!, lower[lower.length - 1]!, point) <= 0) {
      lower.pop();
    }
    lower.push(point);
  }
  const upper: ProjectedPoint[] = [];
  for (let index = unique.length - 1; index >= 0; index -= 1) {
    const point = unique[index]!;
    while (upper.length >= 2
      && cross(upper[upper.length - 2]!, upper[upper.length - 1]!, point) <= 0) {
      upper.pop();
    }
    upper.push(point);
  }
  return [...lower.slice(0, -1), ...upper.slice(0, -1)].map((point) => ({ ...point }));
}

function minimumBy<T>(
  current: T | null,
  candidate: T | null,
  value: (entry: T) => number
): T | null {
  if (candidate === null) return current;
  if (current === null || value(candidate) < value(current)) return candidate;
  return current;
}

function maximumBy<T>(
  current: T | null,
  candidate: T | null,
  value: (entry: T) => number
): T | null {
  if (candidate === null) return current;
  if (current === null || value(candidate) > value(current)) return candidate;
  return current;
}

function assertFrame(frame: number): void {
  if (!Number.isInteger(frame) || frame < 0) {
    throw new Error("Humanoid physical safety frame index must be a non-negative integer");
  }
}

function assertVector(vector: Vec3, label: string): void {
  if (![vector.x, vector.y, vector.z].every(Number.isFinite)) {
    throw new Error(`Humanoid safety ${label} must be finite`);
  }
}

function cross(origin: ProjectedPoint, end: ProjectedPoint, point: ProjectedPoint): number {
  const value = (end.x - origin.x) * (point.z - origin.z)
    - (end.z - origin.z) * (point.x - origin.x);
  if (!Number.isFinite(value)) {
    throw new Error("Humanoid support hull geometry exceeds finite numeric range");
  }
  return value;
}

function pointDistance(left: ProjectedPoint, right: ProjectedPoint): number {
  const distance = Math.hypot(left.x - right.x, left.z - right.z);
  if (!Number.isFinite(distance)) {
    throw new Error("Humanoid support distance exceeds finite numeric range");
  }
  return distance;
}

function pointSegmentDistance(
  point: ProjectedPoint,
  start: ProjectedPoint,
  end: ProjectedPoint
): number {
  const deltaX = end.x - start.x;
  const deltaZ = end.z - start.z;
  const length = Math.hypot(deltaX, deltaZ);
  if (!Number.isFinite(length)) {
    throw new Error("Humanoid support segment exceeds finite numeric range");
  }
  if (length === 0) return pointDistance(point, start);
  const fromStartX = point.x - start.x;
  const fromStartZ = point.z - start.z;
  const projection = fromStartX * deltaX / length + fromStartZ * deltaZ / length;
  if (!Number.isFinite(projection)) {
    throw new Error("Humanoid support projection exceeds finite numeric range");
  }
  const clamped = Math.max(0, Math.min(length, projection));
  const distance = Math.hypot(
    fromStartX - clamped * deltaX / length,
    fromStartZ - clamped * deltaZ / length
  );
  if (!Number.isFinite(distance)) {
    throw new Error("Humanoid support distance exceeds finite numeric range");
  }
  return distance;
}

function vectorMagnitude(vector: Vec3): number {
  return Math.hypot(vector.x, vector.y, vector.z);
}

function pointVelocity(
  body: { position: Vec3; linearVelocity: Vec3; angularVelocity: Vec3 },
  point: Vec3
): Vec3 {
  const offset = {
    x: point.x - body.position.x,
    y: point.y - body.position.y,
    z: point.z - body.position.z
  };
  const rotational = cross3(body.angularVelocity, offset);
  return {
    x: body.linearVelocity.x + rotational.x,
    y: body.linearVelocity.y + rotational.y,
    z: body.linearVelocity.z + rotational.z
  };
}

function cross3(left: Vec3, right: Vec3): Vec3 {
  return {
    x: left.y * right.z - left.z * right.y,
    y: left.z * right.x - left.x * right.z,
    z: left.x * right.y - left.y * right.x
  };
}

function dot3(left: Vec3, right: Vec3): number {
  return left.x * right.x + left.y * right.y + left.z * right.z;
}
