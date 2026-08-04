import type { Vec3 } from "../../domain/schema.js";

export interface HumanoidObjectSettledSupportAuthority {
  protocol: "humanoid-object-settled-support-authority-v1";
  worldUp: Readonly<Vec3>;
  minimumUpwardNormalDot: number;
  minimumSupportNormalForceN: number;
  maximumLinearSpeedMps: number;
  maximumAngularSpeedRadps: number;
}

const worldUp = Object.freeze({ x: 0, y: 1, z: 0 });

export const HUMANOID_OBJECT_SETTLED_SUPPORT_AUTHORITY: Readonly<
  HumanoidObjectSettledSupportAuthority
> = Object.freeze({
  protocol: "humanoid-object-settled-support-authority-v1",
  worldUp,
  minimumUpwardNormalDot: 0.7,
  minimumSupportNormalForceN: 2,
  maximumLinearSpeedMps: 0.03,
  maximumAngularSpeedRadps: 0.1
});

export type HumanoidObjectSettledSupportReason =
  | "object_not_observable"
  | "object_dynamics_missing"
  | "support_contact_metadata_missing"
  | "support_contact_missing"
  | "support_contact_normal_missing"
  | "support_contact_force_missing"
  | "support_direction_insufficient"
  | "support_force_insufficient"
  | "linear_velocity_exceeded"
  | "angular_velocity_exceeded"
  | "object_settled_on_support";

export interface HumanoidObjectSettledSupportContact {
  normal?: Vec3 | null;
  normalForce: number;
  firstBody: string | null;
  secondBody: string | null;
  firstObject: string | null;
  secondObject: string | null;
  firstHandLink?: string | null;
  secondHandLink?: string | null;
}

export interface HumanoidObjectSettledSupportSnapshot {
  objects?: Readonly<Record<string, {
    linearVelocity?: Vec3;
    angularVelocity?: Vec3;
  }>>;
  contacts: readonly HumanoidObjectSettledSupportContact[];
}

export interface HumanoidObjectSettledSupportEvidence {
  objectObservable: boolean;
  supportContactCount: number | null;
  upwardSupportContactCount: number | null;
  forceQualifiedSupportContactCount: number | null;
  missingCounterpartMetadataCount: number | null;
  missingNormalContactCount: number | null;
  missingForceContactCount: number | null;
  maximumUpwardNormalDot: number | null;
  maximumNormalForce: number | null;
  totalNormalForce: number | null;
  totalUpwardSupportForceN: number | null;
  linearVelocity: Vec3 | null;
  linearSpeedMps: number | null;
  angularVelocity: Vec3 | null;
  angularSpeedRadps: number | null;
  authority: Readonly<HumanoidObjectSettledSupportAuthority>;
}

export interface HumanoidObjectSettledSupportAssessment {
  protocol: "humanoid-object-settled-support-assessment-v1";
  objectId: string;
  status: "satisfied" | "unsatisfied" | "uncertain";
  reason: HumanoidObjectSettledSupportReason;
  evidence: HumanoidObjectSettledSupportEvidence;
}

export function assessHumanoidObjectSettledOnSupport(input: {
  objectId: string;
  objectObservable: boolean;
  snapshot: HumanoidObjectSettledSupportSnapshot;
}): HumanoidObjectSettledSupportAssessment {
  if (!input.objectObservable) {
    return assessment(input.objectId, "uncertain", "object_not_observable", {
      objectObservable: false,
      ...unavailableMeasurements()
    });
  }

  const dynamics = input.snapshot.objects?.[input.objectId];
  if (!dynamics
    || !finiteVector(dynamics.linearVelocity)
    || !finiteVector(dynamics.angularVelocity)) {
    return assessment(input.objectId, "uncertain", "object_dynamics_missing", {
      objectObservable: true,
      ...unavailableMeasurements()
    });
  }

  const linearVelocity = cloneVector(dynamics.linearVelocity);
  const angularVelocity = cloneVector(dynamics.angularVelocity);
  const linearSpeedMps = vectorLength(linearVelocity);
  const angularSpeedRadps = vectorLength(angularVelocity);
  const support = supportMetrics(input.snapshot.contacts, input.objectId);
  const evidence: HumanoidObjectSettledSupportEvidence = {
    objectObservable: true,
    ...support,
    linearVelocity,
    linearSpeedMps,
    angularVelocity,
    angularSpeedRadps,
    authority: HUMANOID_OBJECT_SETTLED_SUPPORT_AUTHORITY
  };

  if (support.supportContactCount === 0) {
    return support.missingCounterpartMetadataCount > 0
      ? assessment(
          input.objectId,
          "uncertain",
          "support_contact_metadata_missing",
          evidence
        )
      : assessment(
          input.objectId,
          "unsatisfied",
          "support_contact_missing",
          evidence
        );
  }
  if (support.upwardSupportContactCount === 0) {
    return support.missingNormalContactCount > 0
      ? assessment(
          input.objectId,
          "uncertain",
          "support_contact_normal_missing",
          evidence
        )
      : assessment(
          input.objectId,
          "unsatisfied",
          "support_direction_insufficient",
          evidence
        );
  }
  if (support.forceQualifiedSupportContactCount === 0) {
    return assessment(
      input.objectId,
      support.missingForceContactCount > 0 ? "uncertain" : "unsatisfied",
      support.missingForceContactCount > 0
        ? "support_contact_force_missing"
        : "support_force_insufficient",
      evidence
    );
  }
  if (support.missingForceContactCount > 0
    || support.totalUpwardSupportForceN === null) {
    return assessment(
      input.objectId,
      "uncertain",
      "support_contact_force_missing",
      evidence
    );
  }
  if (support.totalUpwardSupportForceN
    < HUMANOID_OBJECT_SETTLED_SUPPORT_AUTHORITY.minimumSupportNormalForceN) {
    return assessment(
      input.objectId,
      "unsatisfied",
      "support_force_insufficient",
      evidence
    );
  }
  if (linearSpeedMps
    > HUMANOID_OBJECT_SETTLED_SUPPORT_AUTHORITY.maximumLinearSpeedMps) {
    return assessment(
      input.objectId,
      "unsatisfied",
      "linear_velocity_exceeded",
      evidence
    );
  }
  if (angularSpeedRadps
    > HUMANOID_OBJECT_SETTLED_SUPPORT_AUTHORITY.maximumAngularSpeedRadps) {
    return assessment(
      input.objectId,
      "unsatisfied",
      "angular_velocity_exceeded",
      evidence
    );
  }
  return assessment(
    input.objectId,
    "satisfied",
    "object_settled_on_support",
    evidence
  );
}

function assessment(
  objectId: string,
  status: HumanoidObjectSettledSupportAssessment["status"],
  reason: HumanoidObjectSettledSupportReason,
  evidence: Omit<HumanoidObjectSettledSupportEvidence, "authority"> & {
    authority?: Readonly<HumanoidObjectSettledSupportAuthority>;
  }
): HumanoidObjectSettledSupportAssessment {
  return {
    protocol: "humanoid-object-settled-support-assessment-v1",
    objectId,
    status,
    reason,
    evidence: {
      ...evidence,
      authority: evidence.authority ?? HUMANOID_OBJECT_SETTLED_SUPPORT_AUTHORITY
    }
  };
}

function unavailableMeasurements(): Omit<
  HumanoidObjectSettledSupportEvidence,
  "objectObservable" | "authority"
> {
  return {
    supportContactCount: null,
    upwardSupportContactCount: null,
    forceQualifiedSupportContactCount: null,
    missingCounterpartMetadataCount: null,
    missingNormalContactCount: null,
    missingForceContactCount: null,
    maximumUpwardNormalDot: null,
    maximumNormalForce: null,
    totalNormalForce: null,
    totalUpwardSupportForceN: null,
    linearVelocity: null,
    linearSpeedMps: null,
    angularVelocity: null,
    angularSpeedRadps: null
  };
}

function supportMetrics(
  contacts: readonly HumanoidObjectSettledSupportContact[],
  objectId: string
): Pick<
  HumanoidObjectSettledSupportEvidence,
  | "supportContactCount"
  | "upwardSupportContactCount"
  | "forceQualifiedSupportContactCount"
  | "missingCounterpartMetadataCount"
  | "missingNormalContactCount"
  | "missingForceContactCount"
  | "maximumUpwardNormalDot"
  | "maximumNormalForce"
  | "totalNormalForce"
  | "totalUpwardSupportForceN"
> & {
  supportContactCount: number;
  upwardSupportContactCount: number;
  forceQualifiedSupportContactCount: number;
  missingCounterpartMetadataCount: number;
  missingNormalContactCount: number;
  missingForceContactCount: number;
} {
  let supportContactCount = 0;
  let upwardSupportContactCount = 0;
  let forceQualifiedSupportContactCount = 0;
  let missingCounterpartMetadataCount = 0;
  let missingNormalContactCount = 0;
  let missingForceContactCount = 0;
  let maximumUpwardNormalDot: number | null = null;
  let maximumNormalForce: number | null = null;
  let upwardNormalForceTotal = 0;
  let upwardSupportForceTotal = 0;

  for (const contact of contacts) {
    const objectIsFirst = contact.firstObject === objectId;
    const objectIsSecond = contact.secondObject === objectId;
    if (objectIsFirst === objectIsSecond) continue;

    const counterpartObject = objectIsFirst
      ? contact.secondObject
      : contact.firstObject;
    const counterpartBody = objectIsFirst
      ? contact.secondBody
      : contact.firstBody;
    const counterpartHand = objectIsFirst
      ? contact.secondHandLink
      : contact.firstHandLink;
    if (counterpartBody !== null || counterpartHand !== null
      && counterpartHand !== undefined) continue;
    if (counterpartObject === objectId) continue;
    if (counterpartObject === null && counterpartHand === undefined) {
      missingCounterpartMetadataCount += 1;
      continue;
    }

    supportContactCount += 1;
    const normalTowardObject = orientedNormalTowardObject(
      contact.normal,
      objectIsSecond
    );
    if (!normalTowardObject) {
      missingNormalContactCount += 1;
      continue;
    }
    const upwardDot = dot(
      normalTowardObject,
      HUMANOID_OBJECT_SETTLED_SUPPORT_AUTHORITY.worldUp
    );
    maximumUpwardNormalDot = Math.max(maximumUpwardNormalDot ?? -1, upwardDot);
    if (upwardDot
      < HUMANOID_OBJECT_SETTLED_SUPPORT_AUTHORITY.minimumUpwardNormalDot) continue;
    upwardSupportContactCount += 1;
    const forceValid = Number.isFinite(contact.normalForce)
      && contact.normalForce >= 0;
    if (!forceValid) {
      missingForceContactCount += 1;
      continue;
    }
    if (contact.normalForce > 0) forceQualifiedSupportContactCount += 1;
    maximumNormalForce = Math.max(maximumNormalForce ?? 0, contact.normalForce);
    upwardNormalForceTotal += contact.normalForce;
    upwardSupportForceTotal += contact.normalForce * upwardDot;
  }

  return {
    supportContactCount,
    upwardSupportContactCount,
    forceQualifiedSupportContactCount,
    missingCounterpartMetadataCount,
    missingNormalContactCount,
    missingForceContactCount,
    maximumUpwardNormalDot,
    maximumNormalForce,
    totalNormalForce: upwardSupportContactCount === 0
      ? 0
      : missingForceContactCount > 0 ? null : upwardNormalForceTotal,
    totalUpwardSupportForceN: upwardSupportContactCount === 0
      ? 0
      : missingForceContactCount > 0 ? null : upwardSupportForceTotal
  };
}

function orientedNormalTowardObject(
  normal: Vec3 | null | undefined,
  objectIsSecond: boolean
): Vec3 | null {
  if (!finiteVector(normal)) return null;
  const magnitude = vectorLength(normal);
  if (magnitude === 0) return null;
  const direction = objectIsSecond ? 1 : -1;
  return {
    x: normalizedComponent(direction * normal.x / magnitude),
    y: normalizedComponent(direction * normal.y / magnitude),
    z: normalizedComponent(direction * normal.z / magnitude)
  };
}

function finiteVector(value: Vec3 | null | undefined): value is Vec3 {
  return value !== null
    && value !== undefined
    && [value.x, value.y, value.z].every(Number.isFinite);
}

function cloneVector(value: Vec3): Vec3 {
  return { x: value.x, y: value.y, z: value.z };
}

function vectorLength(value: Vec3): number {
  return Math.hypot(value.x, value.y, value.z);
}

function dot(left: Readonly<Vec3>, right: Readonly<Vec3>): number {
  return left.x * right.x + left.y * right.y + left.z * right.z;
}

function normalizedComponent(value: number): number {
  return Object.is(value, -0) ? 0 : value;
}
