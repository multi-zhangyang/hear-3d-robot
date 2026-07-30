import { isDeepStrictEqual } from "node:util";
import type {
  ActionReceipt,
  EvidenceEffect,
  EvidenceFreshness,
  EvidenceRequirement,
  EvidenceTarget,
  JsonValue,
  TaskNode
} from "../domain/schema.js";

interface ReceiptEvidencePolicy {
  effect: EvidenceEffect;
  freshness: EvidenceFreshness;
  targetKind: EvidenceTarget["kind"];
  acceptedCodes: readonly string[];
  target: (receipt: ActionReceipt) => EvidenceTarget;
}

export interface VerifiedReceiptEvidence {
  criterion_index: number;
  transaction_id: string;
  action: string;
  result_code: string;
  effect: EvidenceEffect;
  target: EvidenceTarget;
  freshness: EvidenceFreshness;
  world_revision: number;
  source_transaction_id?: string;
  source_action?: string;
  source_target?: EvidenceTarget;
}

export interface VerifiedBlockerEvidence extends VerifiedReceiptEvidence {
  accepted: boolean;
}

export interface ReceiptEvidenceProvenanceContext {
  lookupReceipt: (transactionId: string) => ActionReceipt | undefined;
  isSourceAuthorized: (source: ActionReceipt) => boolean;
}

const BASE_PLANNERS: readonly string[] = ["plan_base_path"];
const JOINT_PLANNERS: readonly string[] = [
  "plan_joint_targets",
  "solve_end_effector_position",
  "solve_end_effector_pose"
];
const TERMINAL_EFFECTS: ReadonlySet<EvidenceEffect> = new Set([
  "body_motion",
  "world_mutation"
]);
const NON_TERMINAL_BLOCKER_CODES: ReadonlySet<string> = new Set([
  "repeated_accepted_action",
  "body_channel_busy",
  "repeated_denied_action"
]);

const robot = (): EvidenceTarget => ({ kind: "robot" });
const world = (): EvidenceTarget => ({ kind: "world" });
const terrain = (): EvidenceTarget => ({ kind: "terrain" });
const body = (channel: "base" | "head" | "arm" | "gripper") =>
  (): EvidenceTarget => ({ kind: "body", channel });
const entity = (receipt: ActionReceipt): EvidenceTarget => ({
  kind: "entity",
  entity_id: stringField(receipt.input, "entity_id", receipt.name)
});
const voxel = (receipt: ActionReceipt): EvidenceTarget => ({
  kind: "voxel",
  coordinate: voxelField(receipt.input, "coordinate", receipt.name)
});
const targetPosition = (receipt: ActionReceipt): EvidenceTarget => ({
  kind: "position",
  position: vec3Field(receipt.input, "target", receipt.name)
});
const endEffectorPosition = (receipt: ActionReceipt): EvidenceTarget => ({
  kind: "position",
  position: vec3Field(receipt.input, "position", receipt.name)
});

const CURRENT_OBSERVATION = {
  effect: "observation",
  freshness: "current_world"
} as const;
const CURRENT_PLAN = { effect: "plan", freshness: "current_world" } as const;
const CURRENT_MOTION = { effect: "body_motion", freshness: "current_world" } as const;
const CURRENT_MUTATION = { effect: "world_mutation", freshness: "current_world" } as const;

const POLICIES: Readonly<Record<string, ReceiptEvidencePolicy>> = {
  read_proprioception: policy(CURRENT_OBSERVATION, "robot", ["proprioception"], robot),
  sense_scene: policy(CURRENT_OBSERVATION, "world", ["scene_observation"], world),
  survey_terrain: policy(CURRENT_OBSERVATION, "terrain", ["terrain_survey"], terrain),
  scan_voxels: policy(CURRENT_OBSERVATION, "world", ["voxel_scan"], world),
  inspect_voxel: policy(CURRENT_OBSERVATION, "voxel", ["voxel_state"], voxel),
  recall_spatial_memory: policy(
    { effect: "memory", freshness: "historical_record" },
    "world",
    ["spatial_memory_recalled"],
    world
  ),
  inspect_entity: policy(CURRENT_OBSERVATION, "entity", ["entity_state"], entity),
  query_contacts: policy(CURRENT_OBSERVATION, "robot", ["contact_state"], robot),
  inspect_command: policy(CURRENT_OBSERVATION, "robot", ["command_state"], robot),
  plan_base_path: policy(CURRENT_PLAN, "position", ["base_path_planned"], targetPosition),
  plan_arm_retraction: policy(
    CURRENT_PLAN,
    "position",
    ["arm_retraction_not_required", "arm_retraction_options"],
    targetPosition
  ),
  plan_joint_targets: policy(CURRENT_PLAN, "body", ["joint_target_plan"], body("arm")),
  solve_end_effector_position: policy(
    CURRENT_PLAN,
    "position",
    ["end_effector_solution"],
    endEffectorPosition
  ),
  solve_end_effector_pose: policy(
    CURRENT_PLAN,
    "position",
    ["end_effector_solution"],
    endEffectorPosition
  ),
  execute_base_plan: policy(CURRENT_MOTION, "body", ["base_plan_completed"], body("base")),
  execute_joint_plan: policy(CURRENT_MOTION, "body", ["joint_targets_reached"], body("arm")),
  drive_base: policy(CURRENT_MOTION, "body", ["base_motion_completed"], body("base")),
  set_head_target: policy(CURRENT_MOTION, "body", ["head_target_reached"], body("head")),
  set_joint_targets: policy(CURRENT_MOTION, "body", ["joint_targets_reached"], body("arm")),
  set_gripper_target: policy(
    CURRENT_MOTION,
    "body",
    ["gripper_target_reached"],
    body("gripper")
  ),
  break_voxel: policy(CURRENT_MUTATION, "voxel", ["voxel_broken"], voxel),
  place_voxel: policy(CURRENT_MUTATION, "voxel", ["voxel_placed"], voxel)
};

export function evidenceContractGuide(): string {
  return JSON.stringify(Object.fromEntries(
    Object.entries(POLICIES).map(([action, contract]) => {
      const planningSources = executionPlanningActions(action);
      return [action, {
        effect: contract.effect,
        target_kind: contract.targetKind,
        freshness: contract.freshness,
        ...(planningSources ? { planning_source_actions: planningSources } : {})
      }];
    })
  ));
}

/**
 * Older checkpoints predate typed criterion evidence. Their physical state and
 * receipts remain authoritative, but an unfinished SDK branch cannot safely
 * invent the missing contract while completing. The caller must rotate only
 * that disposable conversation branch and let the root model delegate again.
 */
export function hierarchyNeedsEvidenceContractRotation(
  nodes: Record<string, TaskNode>,
  rootId: string
): boolean {
  return Object.values(nodes).some((node) => {
    if (node.id === rootId) return false;
    if (node.status !== "ready" && node.status !== "active" && node.status !== "waiting") {
      return false;
    }
    if (node.evidence_requirements.length !== node.success_criteria.length) return true;
    const indexes = node.evidence_requirements
      .map((requirement) => requirement.criterion_index)
      .toSorted((left, right) => left - right);
    if (indexes.some((index, position) => index !== position)) return true;
    try {
      assertEvidenceRequirementsJointlySatisfiable(node.evidence_requirements);
      return false;
    } catch {
      return true;
    }
  });
}

export function receiptEvidenceRequirement(
  criterionIndex: number,
  action: string,
  target: EvidenceTarget
): Extract<EvidenceRequirement, { kind: "receipt" }> {
  const policy = POLICIES[action];
  if (!policy) throw new Error(`Action ${action} has no harness evidence policy`);
  if (policy.targetKind !== target.kind) {
    throw new Error(`Action ${action} requires a ${policy.targetKind} evidence target`);
  }
  return {
    kind: "receipt",
    criterion_index: criterionIndex,
    actions: [action],
    effect: policy.effect,
    target: structuredClone(target),
    freshness: policy.freshness
  };
}

export function assertReceiptRequirementDefinition(
  requirement: Extract<EvidenceRequirement, { kind: "receipt" }>,
  capabilities: readonly string[]
): void {
  for (const action of requirement.actions) {
    if (!capabilities.includes(action)) {
      throw new Error(`Evidence action ${action} is outside the child capability grant`);
    }
    const policy = POLICIES[action];
    if (!policy) throw new Error(`Evidence action ${action} has no harness evidence policy`);
    assertRequirementPolicy(action, requirement, policy);
  }
}

/**
 * Reject a set of individually valid receipt requirements that cannot all be
 * true at one terminal world revision.
 *
 * Planning receipts describe the world before their matching executor moves
 * it, so a node cannot claim both as independent current-world outcomes. A
 * node also cannot claim two physical effects as terminal facts: the later
 * effect necessarily makes the earlier receipt stale.
 */
export function assertEvidenceRequirementsJointlySatisfiable(
  requirements: readonly EvidenceRequirement[]
): void {
  const receiptRequirements = requirements.filter(
    (requirement): requirement is Extract<EvidenceRequirement, { kind: "receipt" }> =>
      requirement.kind === "receipt"
  );
  const declaredActions = new Set(receiptRequirements.flatMap(
    (requirement) => requirement.actions
  ));

  assertPlannerExecutorPairAbsent(
    receiptRequirements,
    declaredActions,
    BASE_PLANNERS,
    "execute_base_plan"
  );
  assertPlannerExecutorPairAbsent(
    receiptRequirements,
    declaredActions,
    JOINT_PLANNERS,
    "execute_joint_plan"
  );

  const physicalRequirements = receiptRequirements.filter((requirement) =>
    TERMINAL_EFFECTS.has(requirement.effect)
  );
  if (physicalRequirements.length > 1) {
    throw new Error(
      "Evidence contract has multiple terminal physical receipt criteria: "
      + physicalRequirements.map(describeRequirement).join(", ")
      + ". A later body motion or world mutation makes every earlier physical receipt stale."
    );
  }
}

export function verifyReceiptEvidence(
  criterionIndex: number,
  requirement: Extract<EvidenceRequirement, { kind: "receipt" }>,
  receipt: ActionReceipt,
  currentWorldRevision: number,
  provenance?: ReceiptEvidenceProvenanceContext
): VerifiedReceiptEvidence {
  assertCriterionIndex(criterionIndex, requirement);
  if (!receipt.accepted) {
    throw new Error(`Completed evidence references a rejected transaction: ${receipt.transaction_id}`);
  }
  if (!requirement.actions.includes(receipt.name)) {
    throw new Error(
      `Evidence transaction ${receipt.transaction_id} used ${receipt.name}; expected one of ${requirement.actions.join(", ")}`
    );
  }
  const policy = POLICIES[receipt.name];
  if (!policy) throw new Error(`Action ${receipt.name} has no harness evidence policy`);
  assertRequirementPolicy(receipt.name, requirement, policy);
  if (!policy.acceptedCodes.includes(receipt.code)) {
    throw new Error(
      `Evidence transaction ${receipt.transaction_id} returned ${receipt.code}; accepted ${receipt.name} evidence requires ${policy.acceptedCodes.join(" or ")}`
    );
  }
  if (requirement.freshness === "current_world"
    && receipt.world_revision !== currentWorldRevision) {
    throw new Error(
      `Evidence transaction ${receipt.transaction_id} is from world revision ${receipt.world_revision}; current revision is ${currentWorldRevision}`
    );
  }
  const actualTarget = policy.target(receipt);
  if (!isDeepStrictEqual(actualTarget, requirement.target)) {
    throw new Error(
      `Evidence transaction ${receipt.transaction_id} targets ${JSON.stringify(actualTarget)}, not ${JSON.stringify(requirement.target)}`
    );
  }
  const source = verifyExecutionSource(receipt, provenance);
  return {
    criterion_index: criterionIndex,
    transaction_id: receipt.transaction_id,
    action: receipt.name,
    result_code: receipt.code,
    effect: policy.effect,
    target: actualTarget,
    freshness: policy.freshness,
    world_revision: receipt.world_revision,
    ...(source ?? {})
  };
}

export function verifyBlockerEvidence(
  criterionIndex: number,
  requirement: Extract<EvidenceRequirement, { kind: "receipt" }>,
  receipt: ActionReceipt,
  currentWorldRevision: number
): VerifiedBlockerEvidence {
  assertCriterionIndex(criterionIndex, requirement);
  if (receipt.accepted) {
    throw new Error(
      `Blocker evidence requires a rejected transaction; ${receipt.transaction_id} was accepted`
    );
  }
  if (NON_TERMINAL_BLOCKER_CODES.has(receipt.code)) {
    throw new Error(
      `Blocker transaction ${receipt.transaction_id} returned non-terminal code ${receipt.code}`
    );
  }
  if (!requirement.actions.includes(receipt.name)) {
    throw new Error(
      `Blocker transaction ${receipt.transaction_id} used ${receipt.name}; expected one of ${requirement.actions.join(", ")}`
    );
  }
  const policy = POLICIES[receipt.name];
  if (!policy) throw new Error(`Action ${receipt.name} has no harness evidence policy`);
  assertRequirementPolicy(receipt.name, requirement, policy);
  if (policy.acceptedCodes.includes(receipt.code)) {
    throw new Error(
      `Rejected blocker transaction ${receipt.transaction_id} uses successful result code ${receipt.code}`
    );
  }
  if (requirement.freshness === "current_world"
    && receipt.world_revision !== currentWorldRevision) {
    throw new Error(
      `Blocker transaction ${receipt.transaction_id} is from world revision ${receipt.world_revision}; current revision is ${currentWorldRevision}`
    );
  }
  const actualTarget = policy.target(receipt);
  if (!isDeepStrictEqual(actualTarget, requirement.target)) {
    throw new Error(
      `Blocker transaction ${receipt.transaction_id} targets ${JSON.stringify(actualTarget)}, not ${JSON.stringify(requirement.target)}`
    );
  }
  return {
    criterion_index: criterionIndex,
    transaction_id: receipt.transaction_id,
    action: receipt.name,
    result_code: receipt.code,
    accepted: receipt.accepted,
    effect: policy.effect,
    target: actualTarget,
    freshness: policy.freshness,
    world_revision: receipt.world_revision
  };
}

function assertPlannerExecutorPairAbsent(
  requirements: readonly Extract<EvidenceRequirement, { kind: "receipt" }>[],
  declaredActions: ReadonlySet<string>,
  planners: readonly string[],
  executor: string
): void {
  if (!declaredActions.has(executor)) return;
  const declaredPlanners = planners.filter((planner) => declaredActions.has(planner));
  if (declaredPlanners.length === 0) return;
  const involved = requirements.filter((requirement) =>
    requirement.actions.some((action) => action === executor || declaredPlanners.includes(action))
  );
  throw new Error(
    `Evidence contract cannot declare ${declaredPlanners.join(" or ")} together with ${executor} `
    + "as terminal receipt criteria: executing the plan advances the world revision and makes "
    + `the planning receipt stale (${involved.map(describeRequirement).join(", ")})`
  );
}

function describeRequirement(
  requirement: Extract<EvidenceRequirement, { kind: "receipt" }>
): string {
  return `criterion ${requirement.criterion_index} [${requirement.actions.join("|")}]`;
}

function verifyExecutionSource(
  receipt: ActionReceipt,
  provenance: ReceiptEvidenceProvenanceContext | undefined
): Pick<
  VerifiedReceiptEvidence,
  "source_transaction_id" | "source_action" | "source_target"
> | undefined {
  const expectedPlanningActions = executionPlanningActions(receipt.name);
  if (!expectedPlanningActions) return undefined;
  if (!provenance) {
    throw new Error(
      `Execution evidence ${receipt.transaction_id} requires receipt provenance context`
    );
  }
  const planningTransactionId = stringField(
    receipt.input,
    "planning_transaction_id",
    receipt.name
  );
  const beforeRevision = receiptWorldBeforeRevision(receipt);
  if (beforeRevision === null) {
    throw new Error(
      `Execution evidence ${receipt.transaction_id} has no canonical world_before_revision`
    );
  }
  const source = provenance.lookupReceipt(planningTransactionId);
  if (!source) {
    throw new Error(
      `Execution evidence ${receipt.transaction_id} references unknown planning transaction ${planningTransactionId}`
    );
  }
  if (source.transaction_id !== planningTransactionId) {
    throw new Error(
      `Planning lookup for ${planningTransactionId} returned transaction ${source.transaction_id}`
    );
  }
  if (!source.accepted) {
    throw new Error(`Planning transaction ${planningTransactionId} was rejected`);
  }
  if (source.kind !== "tool") {
    throw new Error(
      `Planning transaction ${planningTransactionId} is ${source.kind}; execution evidence requires a tool receipt`
    );
  }
  if (!expectedPlanningActions.includes(source.name)) {
    throw new Error(
      `Execution evidence ${receipt.transaction_id} used ${source.name}; expected planning action ${expectedPlanningActions.join(" or ")}`
    );
  }
  if (!provenance.isSourceAuthorized(source)) {
    throw new Error(
      `Planning transaction ${planningTransactionId} is not authorized for this evidence branch`
    );
  }
  if (source.world_revision !== beforeRevision) {
    throw new Error(
      `Planning transaction ${planningTransactionId} is from world revision ${source.world_revision}; `
      + `execution began at revision ${beforeRevision}`
    );
  }
  const sourcePolicy = POLICIES[source.name];
  if (!sourcePolicy || !sourcePolicy.acceptedCodes.includes(source.code)) {
    throw new Error(
      `Planning transaction ${planningTransactionId} returned invalid accepted code ${source.code}`
    );
  }
  const sourcePlanId = planId(source, "planning");
  const executionPlanId = planId(receipt, "execution");
  if (sourcePlanId !== executionPlanId) {
    throw new Error(
      `Execution evidence ${receipt.transaction_id} completed plan ${executionPlanId}, `
      + `not source plan ${sourcePlanId}`
    );
  }
  return {
    source_transaction_id: source.transaction_id,
    source_action: source.name,
    source_target: sourcePolicy.target(source)
  };
}

function executionPlanningActions(action: string): readonly string[] | null {
  if (action === "execute_base_plan") return BASE_PLANNERS;
  if (action === "execute_joint_plan") return JOINT_PLANNERS;
  return null;
}

function receiptWorldBeforeRevision(receipt: ActionReceipt): number | null {
  const value = (receipt as ActionReceipt & { world_before_revision?: unknown })
    .world_before_revision;
  return Number.isSafeInteger(value) && typeof value === "number" && value >= 0
    ? value
    : null;
}

function planId(receipt: ActionReceipt, role: "planning" | "execution"): string {
  const value = record(receipt.detail)?.plan_id;
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(
      `${role === "planning" ? "Planning" : "Execution"} transaction ${receipt.transaction_id} has no canonical plan_id`
    );
  }
  return value;
}

function policy(
  contract: Pick<ReceiptEvidencePolicy, "effect" | "freshness">,
  targetKind: EvidenceTarget["kind"],
  acceptedCodes: readonly string[],
  target: (receipt: ActionReceipt) => EvidenceTarget
): ReceiptEvidencePolicy {
  return { ...contract, targetKind, acceptedCodes, target };
}

function assertCriterionIndex(
  criterionIndex: number,
  requirement: Extract<EvidenceRequirement, { kind: "receipt" }>
): void {
  if (criterionIndex !== requirement.criterion_index) {
    throw new Error(
      `Evidence criterion ${criterionIndex} does not match requirement ${requirement.criterion_index}`
    );
  }
}

function assertRequirementPolicy(
  action: string,
  requirement: Extract<EvidenceRequirement, { kind: "receipt" }>,
  policy: ReceiptEvidencePolicy
): void {
  if (policy.effect !== requirement.effect) {
    throw new Error(
      `Evidence action ${action} produces ${policy.effect}, not ${requirement.effect}`
    );
  }
  if (policy.freshness !== requirement.freshness) {
    throw new Error(
      `Evidence action ${action} requires ${policy.freshness}, not ${requirement.freshness}`
    );
  }
  if (policy.targetKind !== requirement.target.kind) {
    throw new Error(
      `Evidence action ${action} targets ${policy.targetKind}, not ${requirement.target.kind}`
    );
  }
}

function record(value: JsonValue): Record<string, JsonValue> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value
    : null;
}

function stringField(value: JsonValue, key: string, action: string): string {
  const field = record(value)?.[key];
  if (typeof field !== "string" || field.length === 0) {
    throw new Error(`Receipt ${action} has no canonical ${key}`);
  }
  return field;
}

function vec3Field(
  value: JsonValue,
  key: string,
  action: string
): { x: number; y: number; z: number } {
  const field = record(record(value)?.[key] ?? null);
  if (typeof field?.x !== "number" || typeof field.y !== "number" || typeof field.z !== "number") {
    throw new Error(`Receipt ${action} has no canonical ${key}`);
  }
  return { x: field.x, y: field.y, z: field.z };
}

function voxelField(
  value: JsonValue,
  key: string,
  action: string
): { column: number; level: number; row: number } {
  const field = record(record(value)?.[key] ?? null);
  if (typeof field?.column !== "number"
    || typeof field.level !== "number"
    || typeof field.row !== "number") {
    throw new Error(`Receipt ${action} has no canonical ${key}`);
  }
  return { column: field.column, level: field.level, row: field.row };
}
