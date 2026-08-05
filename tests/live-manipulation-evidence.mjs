import assert from "node:assert/strict";

import { assessHumanoidObjectSettledOnSupport } from
  "../dist/world/humanoid/object-settled-support.js";
import { humanoidObjectZoneRelation } from
  "../dist/world/humanoid/object-zone-relation.js";
import { inspectLiveRunEvidence } from "./live-run-evidence.mjs";

export async function inspectLiveManipulationEvidence(input) {
  const base = await inspectLiveRunEvidence({
    store: input.store,
    scenario: input.scenario,
    expectedStatus: "succeeded",
    requireMissionCompletion: true,
    minimumTravelledDistance: 0.3
  });
  const predicate = input.scenario.default_goal.predicates.find((candidate) => (
    candidate.type === "object_placed"
  ));
  assert.ok(predicate, "Manipulation scenario has no object_placed mission predicate");
  const [checkpoint, actions] = await Promise.all([
    input.store.readHumanoidCheckpoint(),
    input.store.readJournal("actions")
  ]);
  const plans = new Map(actions.filter((receipt) => (
    isRecord(receipt)
      && receipt.action === "plan_whole_body_motion_candidates"
      && receipt.accepted === true
  )).map((receipt) => [receipt.transactionId, receipt]));
  const executions = actions.filter((receipt) => (
    isRecord(receipt)
      && receipt.action === "execute_whole_body_motion"
      && receipt.accepted === true
      && receipt.code === "motion_option_succeeded"
  ));
  const stageExecutions = executions.flatMap((execution) => {
    const planningTransactionId = record(execution.input)?.planning_transaction_id;
    const plan = typeof planningTransactionId === "string"
      ? plans.get(planningTransactionId)
      : undefined;
    const termination = record(plan?.detail)?.termination;
    const predicates = Array.isArray(record(termination)?.predicates)
      ? record(termination).predicates.filter(isRecord)
      : [];
    return plan ? [{ execution, plan, predicates }] : [];
  });
  const acquisition = stageExecutions.find(({ predicates }) => (
    predicates.some((candidate) => candidate.type === "grasp_verified"
      && candidate.object_id === predicate.object_id)
      && !predicates.some((candidate) => candidate.type === "object_released")
  ));
  assert.ok(acquisition, "Live manipulation never completed a verified physical grasp");

  const release = stageExecutions.find(({ predicates }) => (
    predicates.some((candidate) => candidate.type === "object_released"
      && candidate.object_id === predicate.object_id)
      && predicates.some((candidate) => candidate.type === "object_settled_on_support"
        && candidate.object_id === predicate.object_id)
      && predicates.some((candidate) => candidate.type === "object_in_zone"
        && candidate.object_id === predicate.object_id
        && candidate.zone_id === predicate.zone_id)
  ));
  assert.ok(release, "Live manipulation never completed physical release and settling");

  const carriedNavigation = actions.find((receipt) => {
    if (!isRecord(receipt)
      || receipt.action !== "execute_humanoid_navigation"
      || receipt.accepted !== true
      || receipt.code !== "navigation_completed") return false;
    const carry = record(record(record(receipt.detail)?.result)?.carry);
    const bindings = record(carry?.binding_set)?.bindings;
    return Array.isArray(bindings) && bindings.some((binding) => (
      isRecord(binding) && binding.object_id === predicate.object_id
    ));
  });
  assert.ok(carriedNavigation, "Live manipulation never navigated with a current grasp binding");

  const object = checkpoint.world.robot.objects[predicate.object_id];
  const descriptor = input.scenario.objects.find(({ id }) => id === predicate.object_id);
  const zone = input.scenario.zones.find(({ id }) => id === predicate.zone_id);
  assert.ok(object && descriptor && zone, "Final manipulation geometry is incomplete");
  const zoneRelation = humanoidObjectZoneRelation({
    object: {
      position: object.position,
      rotation: object.rotation,
      size: descriptor.size
    },
    zone,
    tolerance: predicate.tolerance
  });
  assert.equal(zoneRelation.inside, true, "Final object is outside its target zone");
  const settled = assessHumanoidObjectSettledOnSupport({
    objectId: predicate.object_id,
    objectObservable: true,
    snapshot: checkpoint.world.robot
  });
  assert.equal(settled.status, "satisfied", "Final object is not physically settled");
  assert.equal(checkpoint.world.grasp.assessments.some((assessment) => (
    assessment.frame === checkpoint.world.frame
      && assessment.object_id === predicate.object_id
      && assessment.grasp_verified
  )), false, "Final object is still physically grasped");
  assert.equal(checkpoint.world_checkpoint.carriedObjectLifecycle.phase, "released",
    "Carried-object lifecycle did not reach released");
  assert.equal(
    checkpoint.world_checkpoint.carriedObjectLifecycle.active_binding_set,
    null,
    "Released object retained an active carry binding"
  );

  return {
    ...base,
    manipulation_verified: true,
    manipulated_object_id: predicate.object_id,
    target_zone_id: predicate.zone_id,
    grasp_execution_transaction_id: acquisition.execution.transactionId,
    carry_navigation_transaction_id: carriedNavigation.transactionId,
    release_execution_transaction_id: release.execution.transactionId,
    final_object_position: object.position,
    final_zone_clearance_m: zoneRelation.horizontalClearance.minimum,
    final_support_height_error_m: zoneRelation.supportHeightError,
    final_settled_support: settled
  };
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function record(value) {
  return isRecord(value) ? value : undefined;
}
