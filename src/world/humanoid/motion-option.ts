import { z } from "zod";
import {
  type HumanoidEndEffector,
  type Quaternion,
  type Vec3
} from "../../domain/schema.js";
import { type HumanoidBodyName } from "./model.js";
import type { G1HandContactSurfaceName } from "./morphology.js";
import {
  humanoidEndEffectorPosition,
  humanoidEndEffectorRotation
} from "./end-effectors.js";
import {
  HumanoidGraspAssessmentSchema,
  type HumanoidGraspAssessment
} from "./grasp-tracker.js";
import {
  assessHumanoidObjectSettledOnSupport,
  type HumanoidObjectSettledSupportEvidence,
  type HumanoidObjectSettledSupportReason
} from "./object-settled-support.js";
import { assessHumanoidObjectReleased } from "./object-release.js";
import {
  normalizeQuaternion,
  quaternionAngularDistance
} from "../geometry.js";
import { humanoidObjectInsideZone } from "./object-zone-relation.js";
import {
  humanoidObjectInsideContainerGeometry,
  humanoidObjectOnSupportGeometry
} from "./object-world-model.js";
import type { HumanoidObjectCapabilityDescriptor } from "./object-capability.js";
import {
  HumanoidMotionOptionContractSchema,
  humanoidMotionOptionConditionMetrics,
  humanoidMotionOptionContractSha256,
  type HumanoidMotionOptionCondition,
  type HumanoidMotionOptionContract,
  type HumanoidMotionOptionPredicate
} from "./motion-option-contract.js";

export {
  HumanoidMotionOptionContractSchema,
  humanoidMotionOptionContractSha256,
  type HumanoidMotionOptionCondition,
  type HumanoidMotionOptionContract
} from "./motion-option-contract.js";

export interface HumanoidMotionOptionRobotSnapshot {
  rootPosition: Vec3;
  fallen?: boolean;
  balance?: {
    supportMargin: number | null;
  };
  links: Readonly<Partial<Record<HumanoidBodyName, {
    position: Vec3;
    rotation?: Quaternion;
  }>>>;
  objects?: Readonly<Record<string, {
    linearVelocity?: Vec3;
    angularVelocity?: Vec3;
  }>>;
  contacts: ReadonlyArray<{
    normal?: Vec3 | null;
    normalForce: number;
    firstBody: HumanoidBodyName | null;
    secondBody: HumanoidBodyName | null;
    firstObject: string | null;
    secondObject: string | null;
    firstSolid?: string | null | undefined;
    secondSolid?: string | null | undefined;
    firstHandLink?: string | null;
    secondHandLink?: string | null;
  }>;
}

export interface HumanoidMotionOptionObservableObject {
  id: string;
  position: Vec3;
  rotation: Quaternion;
  size: Vec3;
  articulation?: {
    jointId: string;
    position: number;
    velocity: number;
    closedPosition: number;
    openPosition: number;
  };
  container?: HumanoidObjectCapabilityDescriptor["container"];
  supportSurface?: HumanoidObjectCapabilityDescriptor["supportSurface"];
}

interface HumanoidMotionOptionZone {
  id: string;
  center: Vec3;
  size: Vec3;
}

export interface HumanoidMotionOptionDetectorInput {
  snapshot: HumanoidMotionOptionRobotSnapshot;
  observableObjects: readonly HumanoidMotionOptionObservableObject[];
  observableSolidIds?: readonly string[];
  zones: readonly HumanoidMotionOptionZone[];
  graspAssessments?: readonly HumanoidMotionOptionGraspAssessmentBinding[];
}

export interface HumanoidMotionOptionGraspAssessmentBinding {
  predicate_index: number;
  contract_sha256: string;
  assessment: HumanoidGraspAssessment;
}

const HumanoidMotionOptionGraspAssessmentBindingSchema = z.object({
  predicate_index: z.number().int().min(0).max(15),
  contract_sha256: z.string().regex(/^[a-f0-9]{64}$/),
  assessment: HumanoidGraspAssessmentSchema
}).strict();

type HumanoidMotionOptionTruth =
  "satisfied" | "unsatisfied" | "uncertain";
type PredicateUncertainty = "body_snapshot_missing"
  | "end_effector_snapshot_missing"
  | "object_not_observable"
  | "solid_not_observable"
  | "zone_not_found"
  | "grasp_assessment_missing"
  | "articulation_not_observable"
  | "relation_target_not_observable"
  | "relation_capability_missing"
  | "balance_snapshot_missing";

interface PredicateEvidenceBase {
  predicateIndex: number;
  status: HumanoidMotionOptionTruth;
}

type HumanoidMotionOptionPredicateEvidence =
  | PredicateEvidenceBase & {
      type: "root_near_point";
      actualPosition: Vec3;
      target: Vec3;
      distanceMeters: number;
      toleranceMeters: number;
    }
  | PredicateEvidenceBase & {
      type: "body_near_point";
      body: HumanoidBodyName;
      actualPosition: Vec3 | null;
      target: Vec3;
      distanceMeters: number | null;
      toleranceMeters: number;
      actualOrientation?: Quaternion | null;
      targetOrientation?: Quaternion;
      orientationErrorRadians?: number | null;
      orientationToleranceRadians?: number;
      reason?: Extract<PredicateUncertainty, "body_snapshot_missing">;
    }
  | PredicateEvidenceBase & {
      type: "end_effector_near_point";
      endEffector: HumanoidEndEffector;
      frame: "world" | "pelvis";
      actualPosition: Vec3 | null;
      target: Vec3;
      distanceMeters: number | null;
      toleranceMeters: number;
      reason?: Extract<
        PredicateUncertainty,
        "end_effector_snapshot_missing"
      >;
    }
  | PredicateEvidenceBase & {
      type: "body_contact_object";
      body: HumanoidBodyName;
      objectId: string;
      objectObservable: boolean;
      maximumNormalForce: number | null;
      minimumNormalForce: number;
      reason?: Extract<PredicateUncertainty, "object_not_observable">;
    }
  | PredicateEvidenceBase & {
      type: "hand_contact_object";
      handSurface: G1HandContactSurfaceName;
      objectId: string;
      objectObservable: boolean;
      maximumNormalForce: number | null;
      minimumNormalForce: number;
      reason?: Extract<PredicateUncertainty, "object_not_observable">;
    }
  | PredicateEvidenceBase & {
      type: "body_contact_solid";
      body: HumanoidBodyName;
      solidId: string;
      solidObservable: boolean;
      maximumNormalForce: number | null;
      minimumNormalForce: number;
      reason?: Extract<PredicateUncertainty, "solid_not_observable">;
    }
  | PredicateEvidenceBase & {
      type: "hand_contact_solid";
      handSurface: G1HandContactSurfaceName;
      solidId: string;
      solidObservable: boolean;
      maximumNormalForce: number | null;
      minimumNormalForce: number;
      reason?: Extract<PredicateUncertainty, "solid_not_observable">;
    }
  | PredicateEvidenceBase & {
      type: "object_near_point";
      objectId: string;
      objectObservable: boolean;
      actualPosition: Vec3 | null;
      target: Vec3;
      distanceMeters: number | null;
      toleranceMeters: number;
      reason?: Extract<PredicateUncertainty, "object_not_observable">;
    }
  | PredicateEvidenceBase & {
      type: "object_in_zone";
      objectId: string;
      zoneId: string;
      objectObservable: boolean;
      actualPosition: Vec3 | null;
      inside: boolean | null;
      expected: boolean;
      toleranceMeters: number;
      reason?: Extract<
        PredicateUncertainty,
        "object_not_observable" | "zone_not_found"
      >;
    }
  | PredicateEvidenceBase & {
      type: "articulation_state";
      objectId: string;
      requestedJointId: string;
      observedJointId: string | null;
      requestedState: "open" | "closed";
      objectObservable: boolean;
      articulationObservable: boolean;
      jointPosition: number | null;
      jointVelocity: number | null;
      openFraction: number | null;
      tolerance: number;
      reason?: Extract<
        PredicateUncertainty,
        "object_not_observable" | "articulation_not_observable"
      > | "joint_id_mismatch";
    }
  | PredicateEvidenceBase & {
      type: "object_inside" | "object_on";
      objectId: string;
      relationTargetId: string;
      objectObservable: boolean;
      relationTargetObservable: boolean;
      relation: boolean | null;
      expected: boolean;
      toleranceMeters: number;
      reason?: Extract<
        PredicateUncertainty,
        "object_not_observable" | "relation_target_not_observable"
          | "relation_capability_missing"
      >;
    }
  | PredicateEvidenceBase & {
      type: "object_displaced";
      objectId: string;
      objectObservable: boolean;
      origin: Vec3;
      actualPosition: Vec3 | null;
      directionWorld: Vec3;
      projectedDistanceMeters: number | null;
      lateralErrorMeters: number | null;
      minimumDistanceMeters: number;
      maximumLateralErrorMeters: number;
      reason?: Extract<PredicateUncertainty, "object_not_observable">;
    }
  | PredicateEvidenceBase & {
      type: "articulation_displaced";
      objectId: string;
      requestedJointId: string;
      observedJointId: string | null;
      objectObservable: boolean;
      articulationObservable: boolean;
      originPosition: number;
      actualPosition: number | null;
      signedDelta: number | null;
      direction: "increasing" | "decreasing";
      minimumDelta: number;
      reason?: Extract<
        PredicateUncertainty,
        "object_not_observable" | "articulation_not_observable"
      > | "joint_id_mismatch";
    }
  | PredicateEvidenceBase & {
      type: "balance_stable";
      fallen: boolean | null;
      supportMarginMeters: number | null;
      minimumSupportMarginMeters: number;
      reason?: Extract<PredicateUncertainty, "balance_snapshot_missing">;
    }
  | PredicateEvidenceBase & {
      type: "grasp_verified";
      objectId: string;
      hand: "left" | "right";
      contractSha256: string;
      assessment: HumanoidGraspAssessment | null;
      reason: HumanoidGraspAssessment["reason"]
        | Extract<PredicateUncertainty, "grasp_assessment_missing">;
    }
  | PredicateEvidenceBase & {
      type: "object_released";
      objectId: string;
      hand: "left" | "right";
      reason: "object_released" | "hand_contact_present" | "object_not_observable";
      objectObservable: boolean;
      handContactCount: number | null;
      contactSurfaces: string[];
      totalNormalForceN: number | null;
    }
  | PredicateEvidenceBase & {
      type: "object_settled_on_support";
      objectId: string;
      reason: HumanoidObjectSettledSupportReason;
    } & HumanoidObjectSettledSupportEvidence;

interface HumanoidMotionOptionConditionEvaluation {
  status: HumanoidMotionOptionTruth;
  predicateIndexes: number[];
}

export interface HumanoidMotionOptionDetection {
  evidence: HumanoidMotionOptionPredicateEvidence[];
  status: HumanoidMotionOptionTruth;
  phases: {
    precondition: HumanoidMotionOptionConditionEvaluation | null;
    during: HumanoidMotionOptionConditionEvaluation | null;
    terminal: HumanoidMotionOptionConditionEvaluation;
  };
  allSatisfied: boolean;
  hasUncertain: boolean;
}

export function detectHumanoidMotionOption(
  rawContract: HumanoidMotionOptionContract,
  input: HumanoidMotionOptionDetectorInput
): HumanoidMotionOptionDetection {
  const contract = HumanoidMotionOptionContractSchema.parse(rawContract);
  const objects = uniqueById(input.observableObjects, "observable humanoid object");
  const solids = new Set(input.observableSolidIds ?? []);
  if (solids.size !== (input.observableSolidIds?.length ?? 0)) {
    throw new Error("Duplicate observable humanoid solid");
  }
  const zones = uniqueById(input.zones, "humanoid option zone");
  const graspAssessments = graspAssessmentsForContract(
    contract.predicates,
    input.graspAssessments ?? []
  );
  const evidence = contract.predicates.map((predicate, predicateIndex) => (
    detectPredicate(
      predicate,
      predicateIndex,
      input.snapshot,
      objects,
      solids,
      zones,
      graspAssessments
    )
  ));
  const conditions = conditionsForContract(contract);
  const precondition = conditions.precondition
    ? evaluateCondition(conditions.precondition, evidence)
    : null;
  const during = conditions.during
    ? evaluateCondition(conditions.during, evidence)
    : null;
  const terminal = evaluateCondition(conditions.terminal, evidence);
  const status = everyTruth([
    ...(during ? [during.status] : []),
    terminal.status
  ]);
  return {
    evidence,
    status,
    phases: { precondition, during, terminal },
    allSatisfied: status === "satisfied",
    hasUncertain: contract.phases
      ? [precondition, during, terminal].some((phase) => (
          phase?.status === "uncertain"
        ))
      : evidence.some((entry) => entry.status === "uncertain")
  };
}

function conditionsForContract(contract: HumanoidMotionOptionContract): {
  precondition: HumanoidMotionOptionCondition | null;
  during: HumanoidMotionOptionCondition | null;
  terminal: HumanoidMotionOptionCondition;
} {
  if (contract.phases) {
    return {
      precondition: contract.phases.precondition?.condition ?? null,
      during: contract.phases.during?.condition ?? null,
      terminal: contract.phases.terminal.condition
    };
  }
  return {
    precondition: null,
    during: null,
    terminal: {
      op: "all",
      conditions: contract.predicates.map((_, predicateIndex) => ({
        op: "predicate",
        predicate_index: predicateIndex
      }))
    }
  };
}

function evaluateCondition(
  condition: HumanoidMotionOptionCondition,
  evidence: readonly HumanoidMotionOptionPredicateEvidence[]
): HumanoidMotionOptionConditionEvaluation {
  const metrics = humanoidMotionOptionConditionMetrics(condition);
  return {
    status: evaluateConditionTruth(condition, evidence),
    predicateIndexes: [...new Set(metrics.predicateIndexes)]
  };
}

function evaluateConditionTruth(
  condition: HumanoidMotionOptionCondition,
  evidence: readonly HumanoidMotionOptionPredicateEvidence[]
): HumanoidMotionOptionTruth {
  if (condition.op === "predicate") {
    const predicateEvidence = evidence[condition.predicate_index];
    if (!predicateEvidence) {
      throw new Error(
        `Missing humanoid predicate evidence ${condition.predicate_index}`
      );
    }
    return predicateEvidence.status;
  }
  if (condition.op === "not") {
    const nested = evaluateConditionTruth(condition.condition, evidence);
    return nested === "uncertain"
      ? "uncertain"
      : nested === "satisfied" ? "unsatisfied" : "satisfied";
  }
  const values = condition.conditions.map((nested) => (
    evaluateConditionTruth(nested, evidence)
  ));
  return condition.op === "all" ? everyTruth(values) : someTruth(values);
}

function everyTruth(
  values: readonly HumanoidMotionOptionTruth[]
): HumanoidMotionOptionTruth {
  if (values.some((value) => value === "unsatisfied")) return "unsatisfied";
  return values.some((value) => value === "uncertain")
    ? "uncertain"
    : "satisfied";
}

function someTruth(
  values: readonly HumanoidMotionOptionTruth[]
): HumanoidMotionOptionTruth {
  if (values.some((value) => value === "satisfied")) return "satisfied";
  return values.some((value) => value === "uncertain")
    ? "uncertain"
    : "unsatisfied";
}

export const HumanoidMotionOptionMonitorStateSchema = z.object({
  contractSha256: z.string().regex(/^[a-f0-9]{64}$/),
  phase: z.enum([
    "awaiting_precondition",
    "running",
    "succeeded",
    "violated",
    "indeterminate"
  ]),
  preconditionStableSteps: z.number().int().min(0).max(500),
  terminalStableSteps: z.number().int().min(0).max(500)
}).strict();

export type HumanoidMotionOptionMonitorState = z.infer<
  typeof HumanoidMotionOptionMonitorStateSchema
>;

export interface HumanoidMotionOptionMonitorUpdate {
  state: HumanoidMotionOptionMonitorState;
  detection: HumanoidMotionOptionDetection;
  observationStatus: HumanoidMotionOptionTruth;
}

export function createHumanoidMotionOptionMonitorState(
  rawContract: HumanoidMotionOptionContract
): HumanoidMotionOptionMonitorState {
  const contract = HumanoidMotionOptionContractSchema.parse(rawContract);
  return {
    contractSha256: humanoidMotionOptionContractSha256(contract),
    phase: contract.phases?.precondition
      ? "awaiting_precondition"
      : "running",
    preconditionStableSteps: 0,
    terminalStableSteps: 0
  };
}

export function advanceHumanoidMotionOptionMonitor(
  rawContract: HumanoidMotionOptionContract,
  rawState: HumanoidMotionOptionMonitorState,
  input: HumanoidMotionOptionDetectorInput
): HumanoidMotionOptionMonitorUpdate {
  const contract = HumanoidMotionOptionContractSchema.parse(rawContract);
  const state = HumanoidMotionOptionMonitorStateSchema.parse(rawState);
  if (state.contractSha256 !== humanoidMotionOptionContractSha256(contract)) {
    throw new Error("Humanoid motion option monitor contract does not match");
  }
  const detection = detectHumanoidMotionOption(contract, input);
  if (state.phase === "succeeded"
    || state.phase === "violated"
    || state.phase === "indeterminate") {
    return {
      state,
      detection,
      observationStatus: state.phase === "succeeded"
        ? "satisfied"
        : state.phase === "violated" ? "unsatisfied" : "uncertain"
    };
  }

  let phase: HumanoidMotionOptionMonitorState["phase"] = state.phase;
  let preconditionStableSteps = state.preconditionStableSteps;
  let terminalStableSteps = state.terminalStableSteps;
  if (phase === "awaiting_precondition") {
    const precondition = detection.phases.precondition;
    if (!precondition || !contract.phases?.precondition) {
      throw new Error("Humanoid motion option monitor has no precondition");
    }
    preconditionStableSteps = precondition.status === "satisfied"
      ? Math.min(preconditionStableSteps + 1, 500)
      : 0;
    const requiredStableSteps = contract.phases.precondition.stable_steps ?? 1;
    if (preconditionStableSteps < requiredStableSteps) {
      return {
        state: {
          ...state,
          preconditionStableSteps,
          terminalStableSteps: 0
        },
        detection,
        observationStatus: precondition.status
      };
    }
    return {
      state: {
        ...state,
        phase: "running",
        preconditionStableSteps,
        terminalStableSteps: 0
      },
      detection,
      observationStatus: "satisfied"
    };
  }

  const duringStatus = detection.phases.during?.status ?? "satisfied";
  if (duringStatus === "unsatisfied") {
    return {
      state: {
        ...state,
        phase: "violated",
        preconditionStableSteps,
        terminalStableSteps: 0
      },
      detection,
      observationStatus: "unsatisfied"
    };
  }
  if (duringStatus === "uncertain") {
    return {
      state: {
        ...state,
        phase: "indeterminate",
        preconditionStableSteps,
        terminalStableSteps: 0
      },
      detection,
      observationStatus: "uncertain"
    };
  }

  terminalStableSteps = detection.phases.terminal.status === "satisfied"
    ? Math.min(terminalStableSteps + 1, contract.stable_steps)
    : 0;
  if (terminalStableSteps >= contract.stable_steps) phase = "succeeded";
  return {
    state: {
      ...state,
      phase,
      preconditionStableSteps,
      terminalStableSteps
    },
    detection,
    observationStatus: detection.phases.terminal.status
  };
}

function detectPredicate(
  predicate: HumanoidMotionOptionPredicate,
  predicateIndex: number,
  snapshot: HumanoidMotionOptionRobotSnapshot,
  objects: ReadonlyMap<string, HumanoidMotionOptionObservableObject>,
  solids: ReadonlySet<string>,
  zones: ReadonlyMap<string, HumanoidMotionOptionZone>,
  graspAssessments: ReadonlyMap<
    number,
    HumanoidMotionOptionGraspAssessmentBinding
  >
): HumanoidMotionOptionPredicateEvidence {
  if (predicate.type === "root_near_point") {
    const distanceMeters = distance(snapshot.rootPosition, predicate.target);
    return {
      predicateIndex,
      type: predicate.type,
      status: distanceMeters <= predicate.tolerance_m ? "satisfied" : "unsatisfied",
      actualPosition: { ...snapshot.rootPosition },
      target: { ...predicate.target },
      distanceMeters,
      toleranceMeters: predicate.tolerance_m
    };
  }
  if (predicate.type === "body_near_point") {
    const body = snapshot.links[predicate.body];
    if (!body) {
      return {
        predicateIndex,
        type: predicate.type,
        status: "uncertain",
        body: predicate.body,
        actualPosition: null,
        target: { ...predicate.target },
        distanceMeters: null,
        toleranceMeters: predicate.tolerance_m,
        reason: "body_snapshot_missing"
      };
    }
    const distanceMeters = distance(body.position, predicate.target);
    return {
      predicateIndex,
      type: predicate.type,
      status: distanceMeters <= predicate.tolerance_m ? "satisfied" : "unsatisfied",
      body: predicate.body,
      actualPosition: { ...body.position },
      target: { ...predicate.target },
      distanceMeters,
      toleranceMeters: predicate.tolerance_m
    };
  }
  if (predicate.type === "end_effector_near_point") {
    const actualPosition = humanoidEndEffectorPosition(
      snapshot,
      predicate.end_effector,
      predicate.frame
    );
    const actualOrientation = predicate.target_orientation
      ? humanoidEndEffectorRotation(
          snapshot,
          predicate.end_effector,
          predicate.frame
        )
      : undefined;
    if (!actualPosition || predicate.target_orientation && !actualOrientation) {
      return {
        predicateIndex,
        type: predicate.type,
        status: "uncertain",
        endEffector: predicate.end_effector,
        frame: predicate.frame,
        actualPosition: null,
        target: { ...predicate.target },
        distanceMeters: null,
        toleranceMeters: predicate.tolerance_m,
        ...(predicate.target_orientation
          ? {
              actualOrientation: null,
              targetOrientation: normalizeQuaternion(predicate.target_orientation),
              orientationErrorRadians: null,
              orientationToleranceRadians: predicate.orientation_tolerance_rad
            }
          : {}),
        reason: "end_effector_snapshot_missing"
      };
    }
    const distanceMeters = distance(actualPosition, predicate.target);
    const targetOrientation = predicate.target_orientation
      ? normalizeQuaternion(predicate.target_orientation)
      : undefined;
    const orientationErrorRadians = targetOrientation && actualOrientation
      ? quaternionAngularDistance(actualOrientation, targetOrientation)
      : undefined;
    const positionSatisfied = distanceMeters <= predicate.tolerance_m;
    const orientationSatisfied = orientationErrorRadians === undefined
      || predicate.orientation_tolerance_rad !== undefined
        && orientationErrorRadians <= predicate.orientation_tolerance_rad;
    return {
      predicateIndex,
      type: predicate.type,
      status: positionSatisfied && orientationSatisfied
        ? "satisfied"
        : "unsatisfied",
      endEffector: predicate.end_effector,
      frame: predicate.frame,
      actualPosition,
      target: { ...predicate.target },
      distanceMeters,
      toleranceMeters: predicate.tolerance_m,
      ...(targetOrientation && actualOrientation
        && orientationErrorRadians !== undefined
        && predicate.orientation_tolerance_rad !== undefined
        ? {
            actualOrientation,
            targetOrientation,
            orientationErrorRadians,
            orientationToleranceRadians: predicate.orientation_tolerance_rad
          }
        : {})
    };
  }
  if (predicate.type === "grasp_verified") {
    const binding = graspAssessments.get(predicateIndex);
    if (!binding) {
      return {
        predicateIndex,
        type: predicate.type,
        status: "uncertain",
        objectId: predicate.object_id,
        hand: predicate.hand,
        contractSha256: predicate.grasp_contract_sha256,
        assessment: null,
        reason: "grasp_assessment_missing"
      };
    }
    const assessment = structuredClone(binding.assessment);
    return {
      predicateIndex,
      type: predicate.type,
      status: assessment.grasp_verified
        ? "satisfied"
        : graspAssessmentIsPhysicallyUncertain(assessment)
          ? "uncertain"
          : "unsatisfied",
      objectId: predicate.object_id,
      hand: predicate.hand,
      contractSha256: predicate.grasp_contract_sha256,
      assessment,
      reason: assessment.reason
    };
  }
  if (predicate.type === "object_released") {
    const assessment = assessHumanoidObjectReleased({
      objectId: predicate.object_id,
      hand: predicate.hand,
      objectObservable: objects.has(predicate.object_id),
      contacts: snapshot.contacts
    });
    return {
      predicateIndex,
      type: predicate.type,
      status: assessment.status,
      objectId: predicate.object_id,
      hand: predicate.hand,
      reason: assessment.reason,
      objectObservable: assessment.objectObservable,
      handContactCount: assessment.handContactCount,
      contactSurfaces: assessment.contactSurfaces,
      totalNormalForceN: assessment.totalNormalForceN
    };
  }
  if (predicate.type === "object_settled_on_support") {
    const assessment = assessHumanoidObjectSettledOnSupport({
      objectId: predicate.object_id,
      objectObservable: objects.has(predicate.object_id),
      snapshot
    });
    return {
      predicateIndex,
      type: predicate.type,
      status: assessment.status,
      objectId: predicate.object_id,
      reason: assessment.reason,
      ...assessment.evidence
    };
  }
  if (predicate.type === "balance_stable") {
    if (snapshot.fallen === undefined || snapshot.balance === undefined) {
      return {
        predicateIndex,
        type: predicate.type,
        status: "uncertain",
        fallen: snapshot.fallen ?? null,
        supportMarginMeters: snapshot.balance?.supportMargin ?? null,
        minimumSupportMarginMeters: predicate.minimum_support_margin_m,
        reason: "balance_snapshot_missing"
      };
    }
    const satisfied = !snapshot.fallen
      && snapshot.balance.supportMargin !== null
      && snapshot.balance.supportMargin >= predicate.minimum_support_margin_m;
    return {
      predicateIndex,
      type: predicate.type,
      status: satisfied ? "satisfied" : "unsatisfied",
      fallen: snapshot.fallen,
      supportMarginMeters: snapshot.balance.supportMargin,
      minimumSupportMarginMeters: predicate.minimum_support_margin_m
    };
  }
  if (predicate.type === "body_contact_solid") {
    if (!solids.has(predicate.solid_id)) {
      return {
        predicateIndex,
        type: predicate.type,
        status: "uncertain",
        body: predicate.body,
        solidId: predicate.solid_id,
        solidObservable: false,
        maximumNormalForce: null,
        minimumNormalForce: predicate.minimum_normal_force,
        reason: "solid_not_observable"
      };
    }
    const maximumNormalForce = maximumSolidBodyContactForce(
      snapshot,
      predicate.body,
      predicate.solid_id
    );
    return {
      predicateIndex,
      type: predicate.type,
      status: maximumNormalForce >= predicate.minimum_normal_force
        ? "satisfied"
        : "unsatisfied",
      body: predicate.body,
      solidId: predicate.solid_id,
      solidObservable: true,
      maximumNormalForce,
      minimumNormalForce: predicate.minimum_normal_force
    };
  }
  if (predicate.type === "hand_contact_solid") {
    if (!solids.has(predicate.solid_id)) {
      return {
        predicateIndex,
        type: predicate.type,
        status: "uncertain",
        handSurface: predicate.hand_surface,
        solidId: predicate.solid_id,
        solidObservable: false,
        maximumNormalForce: null,
        minimumNormalForce: predicate.minimum_normal_force,
        reason: "solid_not_observable"
      };
    }
    const maximumNormalForce = maximumSolidHandContactForce(
      snapshot,
      predicate.hand_surface,
      predicate.solid_id
    );
    return {
      predicateIndex,
      type: predicate.type,
      status: maximumNormalForce >= predicate.minimum_normal_force
        ? "satisfied"
        : "unsatisfied",
      handSurface: predicate.hand_surface,
      solidId: predicate.solid_id,
      solidObservable: true,
      maximumNormalForce,
      minimumNormalForce: predicate.minimum_normal_force
    };
  }
  const object = objects.get(predicate.object_id);
  if (!object) return unobservableObjectEvidence(predicate, predicateIndex);
  if (predicate.type === "hand_contact_object") {
    const maximumNormalForce = maximumHandContactForce(
      snapshot,
      predicate.hand_surface,
      predicate.object_id
    );
    return {
      predicateIndex,
      type: predicate.type,
      status: maximumNormalForce >= predicate.minimum_normal_force
        ? "satisfied"
        : "unsatisfied",
      handSurface: predicate.hand_surface,
      objectId: predicate.object_id,
      objectObservable: true,
      maximumNormalForce,
      minimumNormalForce: predicate.minimum_normal_force
    };
  }
  if (predicate.type === "body_contact_object") {
    const maximumNormalForce = maximumContactForce(
      snapshot,
      predicate.body,
      predicate.object_id
    );
    return {
      predicateIndex,
      type: predicate.type,
      status: maximumNormalForce >= predicate.minimum_normal_force
        ? "satisfied"
        : "unsatisfied",
      body: predicate.body,
      objectId: predicate.object_id,
      objectObservable: true,
      maximumNormalForce,
      minimumNormalForce: predicate.minimum_normal_force
    };
  }
  if (predicate.type === "object_near_point") {
    const distanceMeters = distance(object.position, predicate.target);
    return {
      predicateIndex,
      type: predicate.type,
      status: distanceMeters <= predicate.tolerance_m ? "satisfied" : "unsatisfied",
      objectId: predicate.object_id,
      objectObservable: true,
      actualPosition: { ...object.position },
      target: { ...predicate.target },
      distanceMeters,
      toleranceMeters: predicate.tolerance_m
    };
  }
  if (predicate.type === "object_displaced") {
    const delta = subtractVector(object.position, predicate.origin);
    const projectedDistanceMeters = dotVector(delta, predicate.direction_world);
    const lateral = subtractVector(
      delta,
      scaleVector(predicate.direction_world, projectedDistanceMeters)
    );
    const lateralErrorMeters = vectorLength(lateral);
    return {
      predicateIndex,
      type: predicate.type,
      status: projectedDistanceMeters >= predicate.minimum_distance_m
        && lateralErrorMeters <= predicate.maximum_lateral_error_m
        ? "satisfied"
        : "unsatisfied",
      objectId: predicate.object_id,
      objectObservable: true,
      origin: { ...predicate.origin },
      actualPosition: { ...object.position },
      directionWorld: { ...predicate.direction_world },
      projectedDistanceMeters,
      lateralErrorMeters,
      minimumDistanceMeters: predicate.minimum_distance_m,
      maximumLateralErrorMeters: predicate.maximum_lateral_error_m
    };
  }
  if (predicate.type === "articulation_state") {
    const articulation = object.articulation;
    if (!articulation) {
      return {
        predicateIndex,
        type: predicate.type,
        status: "uncertain",
        objectId: predicate.object_id,
        requestedJointId: predicate.joint_id,
        observedJointId: null,
        requestedState: predicate.state,
        objectObservable: true,
        articulationObservable: false,
        jointPosition: null,
        jointVelocity: null,
        openFraction: null,
        tolerance: predicate.tolerance,
        reason: "articulation_not_observable"
      };
    }
    if (articulation.jointId !== predicate.joint_id) {
      return {
        predicateIndex,
        type: predicate.type,
        status: "unsatisfied",
        objectId: predicate.object_id,
        requestedJointId: predicate.joint_id,
        observedJointId: articulation.jointId,
        requestedState: predicate.state,
        objectObservable: true,
        articulationObservable: true,
        jointPosition: articulation.position,
        jointVelocity: articulation.velocity,
        openFraction: null,
        tolerance: predicate.tolerance,
        reason: "joint_id_mismatch"
      };
    }
    const openFraction = clamp01(
      (articulation.position - articulation.closedPosition)
        / (articulation.openPosition - articulation.closedPosition)
    );
    const satisfied = predicate.state === "open"
      ? openFraction >= 1 - predicate.tolerance
      : openFraction <= predicate.tolerance;
    return {
      predicateIndex,
      type: predicate.type,
      status: satisfied ? "satisfied" : "unsatisfied",
      objectId: predicate.object_id,
      requestedJointId: predicate.joint_id,
      observedJointId: articulation.jointId,
      requestedState: predicate.state,
      objectObservable: true,
      articulationObservable: true,
      jointPosition: articulation.position,
      jointVelocity: articulation.velocity,
      openFraction,
      tolerance: predicate.tolerance
    };
  }
  if (predicate.type === "articulation_displaced") {
    const articulation = object.articulation;
    if (!articulation) {
      return {
        predicateIndex,
        type: predicate.type,
        status: "uncertain",
        objectId: predicate.object_id,
        requestedJointId: predicate.joint_id,
        observedJointId: null,
        objectObservable: true,
        articulationObservable: false,
        originPosition: predicate.origin_position,
        actualPosition: null,
        signedDelta: null,
        direction: predicate.direction,
        minimumDelta: predicate.minimum_delta,
        reason: "articulation_not_observable"
      };
    }
    if (articulation.jointId !== predicate.joint_id) {
      return {
        predicateIndex,
        type: predicate.type,
        status: "unsatisfied",
        objectId: predicate.object_id,
        requestedJointId: predicate.joint_id,
        observedJointId: articulation.jointId,
        objectObservable: true,
        articulationObservable: true,
        originPosition: predicate.origin_position,
        actualPosition: articulation.position,
        signedDelta: null,
        direction: predicate.direction,
        minimumDelta: predicate.minimum_delta,
        reason: "joint_id_mismatch"
      };
    }
    const signedDelta = predicate.direction === "increasing"
      ? articulation.position - predicate.origin_position
      : predicate.origin_position - articulation.position;
    return {
      predicateIndex,
      type: predicate.type,
      status: signedDelta >= predicate.minimum_delta ? "satisfied" : "unsatisfied",
      objectId: predicate.object_id,
      requestedJointId: predicate.joint_id,
      observedJointId: articulation.jointId,
      objectObservable: true,
      articulationObservable: true,
      originPosition: predicate.origin_position,
      actualPosition: articulation.position,
      signedDelta,
      direction: predicate.direction,
      minimumDelta: predicate.minimum_delta
    };
  }
  if (predicate.type === "object_inside" || predicate.type === "object_on") {
    const relationTargetId = predicate.type === "object_inside"
      ? predicate.container_id
      : predicate.support_id;
    const relationTarget = objects.get(relationTargetId);
    if (!relationTarget) {
      return {
        predicateIndex,
        type: predicate.type,
        status: "uncertain",
        objectId: predicate.object_id,
        relationTargetId,
        objectObservable: true,
        relationTargetObservable: false,
        relation: null,
        expected: predicate.expected,
        toleranceMeters: predicate.tolerance_m,
        reason: "relation_target_not_observable"
      };
    }
    const descriptor = predicate.type === "object_inside"
      ? relationTarget.container
      : relationTarget.supportSurface;
    if (!descriptor) {
      return {
        predicateIndex,
        type: predicate.type,
        status: "uncertain",
        objectId: predicate.object_id,
        relationTargetId,
        objectObservable: true,
        relationTargetObservable: true,
        relation: null,
        expected: predicate.expected,
        toleranceMeters: predicate.tolerance_m,
        reason: "relation_capability_missing"
      };
    }
    const relation = predicate.type === "object_inside"
      ? humanoidObjectInsideContainerGeometry({
          object,
          container: relationTarget,
          descriptor: relationTarget.container!,
          tolerance: predicate.tolerance_m
        })
      : humanoidObjectOnSupportGeometry({
          object,
          support: relationTarget,
          descriptor: relationTarget.supportSurface!,
          tolerance: predicate.tolerance_m
        });
    return {
      predicateIndex,
      type: predicate.type,
      status: relation === predicate.expected ? "satisfied" : "unsatisfied",
      objectId: predicate.object_id,
      relationTargetId,
      objectObservable: true,
      relationTargetObservable: true,
      relation,
      expected: predicate.expected,
      toleranceMeters: predicate.tolerance_m
    };
  }
  const zone = zones.get(predicate.zone_id);
  if (!zone) {
    return {
      predicateIndex,
      type: predicate.type,
      status: "uncertain",
      objectId: predicate.object_id,
      zoneId: predicate.zone_id,
      objectObservable: true,
      actualPosition: { ...object.position },
      inside: null,
      expected: predicate.expected,
      toleranceMeters: predicate.tolerance_m,
      reason: "zone_not_found"
    };
  }
  const inside = humanoidObjectInsideZone({
    object,
    zone,
    tolerance: predicate.tolerance_m
  });
  return {
    predicateIndex,
    type: predicate.type,
    status: inside === predicate.expected ? "satisfied" : "unsatisfied",
    objectId: predicate.object_id,
    zoneId: predicate.zone_id,
    objectObservable: true,
    actualPosition: { ...object.position },
    inside,
    expected: predicate.expected,
    toleranceMeters: predicate.tolerance_m
  };
}

function graspAssessmentsForContract(
  predicates: readonly HumanoidMotionOptionPredicate[],
  rawBindings: readonly HumanoidMotionOptionGraspAssessmentBinding[]
): ReadonlyMap<number, HumanoidMotionOptionGraspAssessmentBinding> {
  const bindings = new Map<number, HumanoidMotionOptionGraspAssessmentBinding>();
  const objectHands = new Set<string>();
  for (const rawBinding of rawBindings) {
    const binding = HumanoidMotionOptionGraspAssessmentBindingSchema.parse(rawBinding);
    if (bindings.has(binding.predicate_index)) {
      throw new Error(
        `Duplicate humanoid grasp assessment predicate index: ${binding.predicate_index}`
      );
    }
    const predicate = predicates[binding.predicate_index];
    if (predicate?.type !== "grasp_verified") {
      throw new Error(
        `Humanoid grasp assessment ${binding.predicate_index} does not reference a grasp predicate`
      );
    }
    if (binding.assessment.object_id !== predicate.object_id) {
      throw new Error(
        `Humanoid grasp assessment ${binding.predicate_index} object does not match`
      );
    }
    if (binding.assessment.hand !== predicate.hand) {
      throw new Error(
        `Humanoid grasp assessment ${binding.predicate_index} hand does not match`
      );
    }
    if (binding.contract_sha256 !== predicate.grasp_contract_sha256) {
      throw new Error(
        `Humanoid grasp assessment ${binding.predicate_index} contract does not match`
      );
    }
    const objectHand = `${binding.assessment.object_id}\0${binding.assessment.hand}`;
    if (objectHands.has(objectHand)) {
      throw new Error(
        `Duplicate humanoid grasp assessment object and hand: ${binding.assessment.object_id}/${binding.assessment.hand}`
      );
    }
    objectHands.add(objectHand);
    bindings.set(binding.predicate_index, binding);
  }
  return bindings;
}

function graspAssessmentIsPhysicallyUncertain(
  assessment: HumanoidGraspAssessment
): boolean {
  return assessment.evidence.contact.status === "insufficient_normal"
    || assessment.evidence.support.status === "insufficient_normal"
    || assessment.reason === "support_baseline_missing";
}

function unobservableObjectEvidence(
  predicate: Extract<HumanoidMotionOptionPredicate, {
    type: "body_contact_object" | "hand_contact_object"
      | "object_near_point" | "object_in_zone" | "articulation_state"
      | "object_inside" | "object_on" | "object_displaced"
      | "articulation_displaced";
  }>,
  predicateIndex: number
): HumanoidMotionOptionPredicateEvidence {
  if (predicate.type === "body_contact_object") {
    return {
      predicateIndex,
      type: predicate.type,
      status: "uncertain",
      body: predicate.body,
      objectId: predicate.object_id,
      objectObservable: false,
      maximumNormalForce: null,
      minimumNormalForce: predicate.minimum_normal_force,
      reason: "object_not_observable"
    };
  }
  if (predicate.type === "hand_contact_object") {
    return {
      predicateIndex,
      type: predicate.type,
      status: "uncertain",
      handSurface: predicate.hand_surface,
      objectId: predicate.object_id,
      objectObservable: false,
      maximumNormalForce: null,
      minimumNormalForce: predicate.minimum_normal_force,
      reason: "object_not_observable"
    };
  }
  if (predicate.type === "object_near_point") {
    return {
      predicateIndex,
      type: predicate.type,
      status: "uncertain",
      objectId: predicate.object_id,
      objectObservable: false,
      actualPosition: null,
      target: { ...predicate.target },
      distanceMeters: null,
      toleranceMeters: predicate.tolerance_m,
      reason: "object_not_observable"
    };
  }
  if (predicate.type === "object_displaced") {
    return {
      predicateIndex,
      type: predicate.type,
      status: "uncertain",
      objectId: predicate.object_id,
      objectObservable: false,
      origin: { ...predicate.origin },
      actualPosition: null,
      directionWorld: { ...predicate.direction_world },
      projectedDistanceMeters: null,
      lateralErrorMeters: null,
      minimumDistanceMeters: predicate.minimum_distance_m,
      maximumLateralErrorMeters: predicate.maximum_lateral_error_m,
      reason: "object_not_observable"
    };
  }
  if (predicate.type === "articulation_state") {
    return {
      predicateIndex,
      type: predicate.type,
      status: "uncertain",
      objectId: predicate.object_id,
      requestedJointId: predicate.joint_id,
      observedJointId: null,
      requestedState: predicate.state,
      objectObservable: false,
      articulationObservable: false,
      jointPosition: null,
      jointVelocity: null,
      openFraction: null,
      tolerance: predicate.tolerance,
      reason: "object_not_observable"
    };
  }
  if (predicate.type === "articulation_displaced") {
    return {
      predicateIndex,
      type: predicate.type,
      status: "uncertain",
      objectId: predicate.object_id,
      requestedJointId: predicate.joint_id,
      observedJointId: null,
      objectObservable: false,
      articulationObservable: false,
      originPosition: predicate.origin_position,
      actualPosition: null,
      signedDelta: null,
      direction: predicate.direction,
      minimumDelta: predicate.minimum_delta,
      reason: "object_not_observable"
    };
  }
  if (predicate.type === "object_inside" || predicate.type === "object_on") {
    return {
      predicateIndex,
      type: predicate.type,
      status: "uncertain",
      objectId: predicate.object_id,
      relationTargetId: predicate.type === "object_inside"
        ? predicate.container_id
        : predicate.support_id,
      objectObservable: false,
      relationTargetObservable: false,
      relation: null,
      expected: predicate.expected,
      toleranceMeters: predicate.tolerance_m,
      reason: "object_not_observable"
    };
  }
  return {
    predicateIndex,
    type: predicate.type,
    status: "uncertain",
    objectId: predicate.object_id,
    zoneId: predicate.zone_id,
    objectObservable: false,
    actualPosition: null,
    inside: null,
    expected: predicate.expected,
    toleranceMeters: predicate.tolerance_m,
    reason: "object_not_observable"
  };
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function maximumContactForce(
  snapshot: HumanoidMotionOptionRobotSnapshot,
  body: HumanoidBodyName,
  objectId: string
): number {
  let maximum = 0;
  for (const contact of snapshot.contacts) {
    const matches = (contact.firstBody === body && contact.secondObject === objectId)
      || (contact.secondBody === body && contact.firstObject === objectId);
    if (matches) maximum = Math.max(maximum, contact.normalForce);
  }
  return maximum;
}

function maximumHandContactForce(
  snapshot: HumanoidMotionOptionRobotSnapshot,
  handSurface: G1HandContactSurfaceName,
  objectId: string
): number {
  let maximum = 0;
  for (const contact of snapshot.contacts) {
    const matches = (contact.firstHandLink === handSurface
      && contact.secondObject === objectId)
      || (contact.secondHandLink === handSurface
        && contact.firstObject === objectId);
    if (matches) maximum = Math.max(maximum, contact.normalForce);
  }
  return maximum;
}

function maximumSolidBodyContactForce(
  snapshot: HumanoidMotionOptionRobotSnapshot,
  body: HumanoidBodyName,
  solidId: string
): number {
  let maximum = 0;
  for (const contact of snapshot.contacts) {
    const matches = (contact.firstBody === body && contact.secondSolid === solidId)
      || (contact.secondBody === body && contact.firstSolid === solidId);
    if (matches) maximum = Math.max(maximum, contact.normalForce);
  }
  return maximum;
}

function maximumSolidHandContactForce(
  snapshot: HumanoidMotionOptionRobotSnapshot,
  handSurface: G1HandContactSurfaceName,
  solidId: string
): number {
  let maximum = 0;
  for (const contact of snapshot.contacts) {
    const matches = (contact.firstHandLink === handSurface
      && contact.secondSolid === solidId)
      || (contact.secondHandLink === handSurface
        && contact.firstSolid === solidId);
    if (matches) maximum = Math.max(maximum, contact.normalForce);
  }
  return maximum;
}

function uniqueById<T extends { id: string }>(
  values: readonly T[],
  label: string
): ReadonlyMap<string, T> {
  const result = new Map<string, T>();
  for (const value of values) {
    if (result.has(value.id)) throw new Error(`Duplicate ${label}: ${value.id}`);
    result.set(value.id, value);
  }
  return result;
}

function distance(left: Vec3, right: Vec3): number {
  return Math.hypot(left.x - right.x, left.y - right.y, left.z - right.z);
}

function subtractVector(left: Vec3, right: Vec3): Vec3 {
  return { x: left.x - right.x, y: left.y - right.y, z: left.z - right.z };
}

function scaleVector(value: Vec3, scalar: number): Vec3 {
  return { x: value.x * scalar, y: value.y * scalar, z: value.z * scalar };
}

function dotVector(left: Vec3, right: Vec3): number {
  return left.x * right.x + left.y * right.y + left.z * right.z;
}

function vectorLength(value: Vec3): number {
  return Math.hypot(value.x, value.y, value.z);
}
