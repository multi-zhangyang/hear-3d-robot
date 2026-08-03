import { z } from "zod";
import { createHash } from "node:crypto";
import {
  HumanoidEndEffectorSchema,
  Vec3Schema,
  type HumanoidEndEffector,
  type Quaternion,
  type Vec3
} from "../../domain/schema.js";
import {
  HUMANOID_BODY_NAMES,
  type HumanoidBodyName
} from "./model.js";
import { humanoidEndEffectorPosition } from "./end-effectors.js";

const PositionToleranceSchema = z.number().finite().positive().max(5);

const RootNearPointPredicateSchema = z.object({
  type: z.literal("root_near_point"),
  target: Vec3Schema,
  tolerance_m: PositionToleranceSchema
}).strict();

const BodyNearPointPredicateSchema = z.object({
  type: z.literal("body_near_point"),
  body: z.enum(HUMANOID_BODY_NAMES),
  target: Vec3Schema,
  tolerance_m: PositionToleranceSchema
}).strict();

const EndEffectorNearPointPredicateSchema = z.object({
  type: z.literal("end_effector_near_point"),
  end_effector: HumanoidEndEffectorSchema,
  frame: z.enum(["world", "pelvis"]),
  target: Vec3Schema,
  tolerance_m: PositionToleranceSchema
}).strict();

const BodyContactObjectPredicateSchema = z.object({
  type: z.literal("body_contact_object"),
  body: z.enum(HUMANOID_BODY_NAMES),
  object_id: z.string().trim().min(1),
  minimum_normal_force: z.number().finite().positive()
}).strict();

const ObjectNearPointPredicateSchema = z.object({
  type: z.literal("object_near_point"),
  object_id: z.string().trim().min(1),
  target: Vec3Schema,
  tolerance_m: PositionToleranceSchema
}).strict();

const ObjectInZonePredicateSchema = z.object({
  type: z.literal("object_in_zone"),
  object_id: z.string().trim().min(1),
  zone_id: z.string().trim().min(1),
  expected: z.boolean(),
  tolerance_m: z.number().finite().nonnegative().max(5)
}).strict();

const HumanoidMotionOptionPredicateSchema = z.discriminatedUnion("type", [
  RootNearPointPredicateSchema,
  BodyNearPointPredicateSchema,
  EndEffectorNearPointPredicateSchema,
  BodyContactObjectPredicateSchema,
  ObjectNearPointPredicateSchema,
  ObjectInZonePredicateSchema
]);

export type HumanoidMotionOptionCondition =
  | {
      op: "predicate";
      predicate_index: number;
    }
  | {
      op: "all" | "any";
      conditions: HumanoidMotionOptionCondition[];
    }
  | {
      op: "not";
      condition: HumanoidMotionOptionCondition;
    };

const HumanoidMotionOptionConditionSchema:
  z.ZodType<HumanoidMotionOptionCondition, HumanoidMotionOptionCondition> =
    z.lazy(() => z.discriminatedUnion("op", [
      z.object({
        op: z.literal("predicate"),
        predicate_index: z.number().int().min(0).max(15)
      }).strict(),
      z.object({
        op: z.literal("all"),
        conditions: z.array(HumanoidMotionOptionConditionSchema).min(1).max(16)
      }).strict(),
      z.object({
        op: z.literal("any"),
        conditions: z.array(HumanoidMotionOptionConditionSchema).min(1).max(16)
      }).strict(),
      z.object({
        op: z.literal("not"),
        condition: HumanoidMotionOptionConditionSchema
      }).strict()
    ]));

const StableStepsSchema = z.number().int().min(1).max(500);

const HumanoidMotionOptionPhasesSchema = z.object({
  precondition: z.object({
    condition: HumanoidMotionOptionConditionSchema,
    stable_steps: StableStepsSchema.nullable().default(null)
  }).strict().nullable().default(null),
  during: z.object({
    condition: HumanoidMotionOptionConditionSchema
  }).strict().nullable().default(null),
  terminal: z.object({
    condition: HumanoidMotionOptionConditionSchema
  }).strict()
}).strict();

const HumanoidMotionOptionContractShapeSchema = z.object({
  option_id: z.string().trim().min(1),
  predicates: z.array(HumanoidMotionOptionPredicateSchema).min(1).max(16),
  stable_steps: StableStepsSchema,
  phases: HumanoidMotionOptionPhasesSchema.nullable().default(null)
}).strict();

export const HumanoidMotionOptionContractSchema =
  HumanoidMotionOptionContractShapeSchema.superRefine((contract, context) => {
    if (!contract.phases) return;
    const phaseConditions: Array<{
      path: Array<string | number>;
      condition: HumanoidMotionOptionCondition;
    }> = [
      ...(contract.phases.precondition
        ? [{
            path: ["phases", "precondition", "condition"],
            condition: contract.phases.precondition.condition
          }]
        : []),
      ...(contract.phases.during
        ? [{
            path: ["phases", "during", "condition"],
            condition: contract.phases.during.condition
          }]
        : []),
      {
        path: ["phases", "terminal", "condition"],
        condition: contract.phases.terminal.condition
      }
    ];
    for (const phase of phaseConditions) {
      const metrics = conditionMetrics(phase.condition);
      if (metrics.depth > 8) {
        context.addIssue({
          code: "custom",
          path: phase.path,
          message: "A humanoid option condition cannot exceed eight AST levels"
        });
      }
      if (metrics.nodes > 64) {
        context.addIssue({
          code: "custom",
          path: phase.path,
          message: "A humanoid option condition cannot exceed 64 AST nodes"
        });
      }
      for (const predicateIndex of metrics.predicateIndexes) {
        if (predicateIndex >= contract.predicates.length) {
          context.addIssue({
            code: "custom",
            path: phase.path,
            message: `Condition references missing predicate ${predicateIndex}`
          });
        }
      }
    }
  });

type HumanoidMotionOptionPredicate = z.infer<
  typeof HumanoidMotionOptionPredicateSchema
>;
export type HumanoidMotionOptionContract = z.input<
  typeof HumanoidMotionOptionContractSchema
>;

function conditionMetrics(condition: HumanoidMotionOptionCondition): {
  depth: number;
  nodes: number;
  predicateIndexes: number[];
} {
  if (condition.op === "predicate") {
    return {
      depth: 1,
      nodes: 1,
      predicateIndexes: [condition.predicate_index]
    };
  }
  if (condition.op === "not") {
    const child = conditionMetrics(condition.condition);
    return {
      depth: child.depth + 1,
      nodes: child.nodes + 1,
      predicateIndexes: child.predicateIndexes
    };
  }
  let depth = 0;
  let nodes = 1;
  const predicateIndexes: number[] = [];
  for (const nested of condition.conditions) {
    const child = conditionMetrics(nested);
    depth = Math.max(depth, child.depth);
    nodes += child.nodes;
    predicateIndexes.push(...child.predicateIndexes);
  }
  return { depth: depth + 1, nodes, predicateIndexes };
}

export function humanoidMotionOptionContractSha256(
  contract: HumanoidMotionOptionContract
): string {
  const parsed = HumanoidMotionOptionContractSchema.parse(contract);
  const canonical = parsed.phases === null
    ? {
        option_id: parsed.option_id,
        predicates: parsed.predicates,
        stable_steps: parsed.stable_steps
      }
    : parsed;
  return createHash("sha256").update(JSON.stringify(canonical)).digest("hex");
}

export interface HumanoidMotionOptionRobotSnapshot {
  rootPosition: Vec3;
  links: Readonly<Partial<Record<HumanoidBodyName, {
    position: Vec3;
    rotation?: Quaternion;
  }>>>;
  contacts: ReadonlyArray<{
    normalForce: number;
    firstBody: HumanoidBodyName | null;
    secondBody: HumanoidBodyName | null;
    firstObject: string | null;
    secondObject: string | null;
  }>;
}

export interface HumanoidMotionOptionObservableObject {
  id: string;
  position: Vec3;
  size: Vec3;
}

interface HumanoidMotionOptionZone {
  id: string;
  center: Vec3;
  size: Vec3;
}

export interface HumanoidMotionOptionDetectorInput {
  snapshot: HumanoidMotionOptionRobotSnapshot;
  observableObjects: readonly HumanoidMotionOptionObservableObject[];
  zones: readonly HumanoidMotionOptionZone[];
}

type HumanoidMotionOptionTruth =
  "satisfied" | "unsatisfied" | "uncertain";
type PredicateUncertainty = "body_snapshot_missing"
  | "end_effector_snapshot_missing"
  | "object_not_observable"
  | "zone_not_found";

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
    };

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
  const zones = uniqueById(input.zones, "humanoid option zone");
  const evidence = contract.predicates.map((predicate, predicateIndex) => (
    detectPredicate(predicate, predicateIndex, input.snapshot, objects, zones)
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
  const metrics = conditionMetrics(condition);
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
  zones: ReadonlyMap<string, HumanoidMotionOptionZone>
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
    if (!actualPosition) {
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
        reason: "end_effector_snapshot_missing"
      };
    }
    const distanceMeters = distance(actualPosition, predicate.target);
    return {
      predicateIndex,
      type: predicate.type,
      status: distanceMeters <= predicate.tolerance_m
        ? "satisfied"
        : "unsatisfied",
      endEffector: predicate.end_effector,
      frame: predicate.frame,
      actualPosition,
      target: { ...predicate.target },
      distanceMeters,
      toleranceMeters: predicate.tolerance_m
    };
  }
  const object = objects.get(predicate.object_id);
  if (!object) return unobservableObjectEvidence(predicate, predicateIndex);
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
  const inside = objectInsideZone(object, zone, predicate.tolerance_m);
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

function unobservableObjectEvidence(
  predicate: Extract<HumanoidMotionOptionPredicate, { object_id: string }>,
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

function objectInsideZone(
  object: HumanoidMotionOptionObservableObject,
  zone: HumanoidMotionOptionZone,
  tolerance: number
): boolean {
  const bottom = object.position.y - object.size.y / 2;
  const surface = zone.center.y + zone.size.y / 2;
  return Math.abs(object.position.x - zone.center.x) + object.size.x / 2
      <= zone.size.x / 2 + tolerance
    && Math.abs(object.position.z - zone.center.z) + object.size.z / 2
      <= zone.size.z / 2 + tolerance
    && Math.abs(bottom - surface) <= Math.max(tolerance, 0.025);
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
