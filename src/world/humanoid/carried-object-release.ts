import {
  humanoidCarriedObjectBindingSetSha256,
  type HumanoidCarriedObjectBindingSet
} from "./carried-object-binding.js";
import {
  HumanoidMotionOptionContractSchema,
  type HumanoidMotionOptionCondition,
  type HumanoidMotionOptionContract
} from "./motion-option.js";

interface ConditionLiteral {
  predicateIndex: number;
  negated: boolean;
}

interface HumanoidCarriedObjectReleaseBindingAuthority {
  objectId: string;
  hand: "left" | "right";
  graspPredicateIndex: number;
  releasedPredicateIndex: number;
  settledPredicateIndex: number;
  destinationPredicateIndex: number;
}

export interface HumanoidCarriedObjectReleaseAuthority {
  protocol: "humanoid-carried-object-release-authority-v1";
  bindingSetSha256: string;
  bindings: HumanoidCarriedObjectReleaseBindingAuthority[];
}

export function authorizeHumanoidCarriedObjectRelease(input: {
  contract: HumanoidMotionOptionContract;
  bindingSet: HumanoidCarriedObjectBindingSet;
}): HumanoidCarriedObjectReleaseAuthority | null {
  const contract = HumanoidMotionOptionContractSchema.parse(input.contract);
  const terminalCondition = contract.phases?.terminal.condition;
  const releasePredicateIndexes = contract.predicates.flatMap((predicate, index) => (
    predicate.type === "object_released" ? [index] : []
  ));
  const terminalReferences = terminalCondition
    ? conditionLiterals(terminalCondition)
    : [];
  const terminalNegatesGrasp = terminalReferences.some((literal) => (
    literal.negated
      && contract.predicates[literal.predicateIndex]?.type === "grasp_verified"
  ));
  if (releasePredicateIndexes.length === 0 && !terminalNegatesGrasp) return null;
  if (input.bindingSet.bindings.length === 0) {
    throw new Error("Object release requires a current carried-object binding");
  }
  if (!terminalCondition) {
    throw new Error("Object release requires an explicit phased terminal condition");
  }
  const terminal = conjunctiveLiterals(terminalCondition);
  if (!terminal) {
    throw new Error("Object release terminal evidence must be a conjunction");
  }
  const preconditionCondition = contract.phases?.precondition?.condition;
  const precondition = preconditionCondition
    ? conjunctiveLiterals(preconditionCondition)
    : null;
  if (!precondition) {
    throw new Error("Object release requires a conjunctive verified-grasp precondition");
  }
  const terminalPositive = new Set(terminal.filter((entry) => !entry.negated)
    .map((entry) => entry.predicateIndex));
  const terminalNegative = new Set(terminal.filter((entry) => entry.negated)
    .map((entry) => entry.predicateIndex));
  const preconditionPositive = new Set(precondition.filter((entry) => !entry.negated)
    .map((entry) => entry.predicateIndex));
  const activeKeys = new Set(input.bindingSet.bindings.map((binding) => (
    bindingKey(binding.object_id, binding.hand)
  )));
  for (const index of releasePredicateIndexes) {
    const predicate = contract.predicates[index]!;
    if (predicate.type !== "object_released") {
      throw new Error("Object release predicate identity changed during validation");
    }
    if (!activeKeys.has(bindingKey(predicate.object_id, predicate.hand))) {
      throw new Error(
        `Object release references an unbound object and hand: ${predicate.object_id}/${predicate.hand}`
      );
    }
  }

  const bindings = input.bindingSet.bindings.map((binding) => {
    const graspPredicateIndex = uniquePredicateIndex(
      contract.predicates,
      (predicate) => predicate.type === "grasp_verified"
        && predicate.object_id === binding.object_id
        && predicate.hand === binding.hand,
      `verified grasp for ${binding.object_id}/${binding.hand}`
    );
    const releasedPredicateIndex = uniquePredicateIndex(
      contract.predicates,
      (predicate) => predicate.type === "object_released"
        && predicate.object_id === binding.object_id
        && predicate.hand === binding.hand,
      `physical separation for ${binding.object_id}/${binding.hand}`
    );
    const settledPredicateIndex = uniquePredicateIndex(
      contract.predicates,
      (predicate) => predicate.type === "object_settled_on_support"
        && predicate.object_id === binding.object_id,
      `settled support for ${binding.object_id}`
    );
    const destinationPredicateIndex = destinationRelationPredicateIndex(
      contract.predicates,
      binding.object_id
    );
    if (!preconditionPositive.has(graspPredicateIndex)) {
      throw new Error(
        `Object release precondition must require the current grasp: ${binding.object_id}/${binding.hand}`
      );
    }
    if (!terminalNegative.has(graspPredicateIndex)) {
      throw new Error(
        `Object release terminal must negate the current grasp: ${binding.object_id}/${binding.hand}`
      );
    }
    for (const [index, label] of [
      [releasedPredicateIndex, "physical separation"],
      [settledPredicateIndex, "settled support"],
      [destinationPredicateIndex, "destination relation"]
    ] as const) {
      if (!terminalPositive.has(index)) {
        throw new Error(
          `Object release terminal must require ${label}: ${binding.object_id}/${binding.hand}`
        );
      }
    }
    return {
      objectId: binding.object_id,
      hand: binding.hand,
      graspPredicateIndex,
      releasedPredicateIndex,
      settledPredicateIndex,
      destinationPredicateIndex
    };
  }).sort((left, right) => bindingKey(left.objectId, left.hand)
    .localeCompare(bindingKey(right.objectId, right.hand)));

  const during = contract.phases?.during?.condition;
  if (during) {
    const forbidden = new Set(bindings.flatMap((binding) => [
      binding.graspPredicateIndex,
      binding.releasedPredicateIndex,
      binding.settledPredicateIndex
    ]));
    if (conditionLiterals(during).some((literal) => (
      forbidden.has(literal.predicateIndex)
    ))) {
      throw new Error(
        "Object release during-condition cannot require grasp, separation, or settled state"
      );
    }
  }
  return {
    protocol: "humanoid-carried-object-release-authority-v1",
    bindingSetSha256: humanoidCarriedObjectBindingSetSha256(input.bindingSet),
    bindings
  };
}

function conjunctiveLiterals(
  condition: HumanoidMotionOptionCondition
): ConditionLiteral[] | null {
  if (condition.op === "predicate") {
    return [{ predicateIndex: condition.predicate_index, negated: false }];
  }
  if (condition.op === "not") {
    return condition.condition.op === "predicate"
      ? [{ predicateIndex: condition.condition.predicate_index, negated: true }]
      : null;
  }
  if (condition.op === "any") return null;
  const literals: ConditionLiteral[] = [];
  for (const nested of condition.conditions) {
    const entries = conjunctiveLiterals(nested);
    if (!entries) return null;
    literals.push(...entries);
  }
  return literals;
}

function conditionLiterals(
  condition: HumanoidMotionOptionCondition,
  negated = false
): ConditionLiteral[] {
  if (condition.op === "predicate") {
    return [{ predicateIndex: condition.predicate_index, negated }];
  }
  if (condition.op === "not") {
    return conditionLiterals(condition.condition, !negated);
  }
  return condition.conditions.flatMap((nested) => (
    conditionLiterals(nested, negated)
  ));
}

function uniquePredicateIndex(
  predicates: ReturnType<typeof HumanoidMotionOptionContractSchema.parse>["predicates"],
  matches: (predicate: ReturnType<
    typeof HumanoidMotionOptionContractSchema.parse
  >["predicates"][number]) => boolean,
  label: string
): number {
  const indexes = predicates.flatMap((predicate, index) => (
    matches(predicate) ? [index] : []
  ));
  if (indexes.length !== 1) {
    throw new Error(`Object release requires exactly one ${label} predicate`);
  }
  return indexes[0]!;
}

function destinationRelationPredicateIndex(
  predicates: ReturnType<typeof HumanoidMotionOptionContractSchema.parse>["predicates"],
  objectId: string
): number {
  const destinationTypes = [
    "object_in_zone",
    "object_inside",
    "object_on",
    "object_near_point"
  ] as const;
  for (const type of destinationTypes) {
    const indexes = predicates.flatMap((predicate, index) => (
      "object_id" in predicate
        && predicate.object_id === objectId
        && predicate.type === type
        && (!("expected" in predicate) || predicate.expected)
        ? [index] : []
    ));
    if (indexes.length === 1) return indexes[0]!;
    if (indexes.length > 1) {
      throw new Error(
        `Object release requires exactly one destination relation for ${objectId}`
      );
    }
  }
  throw new Error(
    `Object release requires exactly one destination relation for ${objectId}`
  );
}

function bindingKey(objectId: string, hand: "left" | "right"): string {
  return `${objectId}\0${hand}`;
}
