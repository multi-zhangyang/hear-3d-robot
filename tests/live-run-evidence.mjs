import assert from "node:assert/strict";

import { PhysicalTrajectorySummarySchema } from
  "../dist/domain/physical-trajectory.js";
import { ScenarioBlockRemovalTransactionSchema } from
  "../dist/domain/scenario-block-removal.js";
import { resolveHumanoidMissionCompletion } from
  "../dist/harness/humanoid/mission-completion-evidence.js";
import {
  autonomyContentSha256,
  autonomyPlanningBehaviorSha256
} from "../dist/runtime/autonomy-signature.js";

export async function inspectLiveRunEvidence(input) {
  const [checkpoint, actions, modelCalls, goalEvidence, episodes, manifest] =
    await Promise.all([
      input.store.readHumanoidCheckpoint(),
      input.store.readJournal("actions"),
      input.store.readJournal("model_calls"),
      input.store.readJournal("goal_evidence"),
      input.store.readJournal("episodes"),
      input.store.readAgentManifest()
    ]);

  assert.equal(checkpoint.status, input.expectedStatus,
    `Live run ended as ${checkpoint.status}, expected ${input.expectedStatus}`);
  if (checkpoint.status === "paused") {
    assert.equal(checkpoint.error, null, "Operator pause was recorded as a runtime failure");
  }
  const missionCompletion = resolveHumanoidMissionCompletion(checkpoint, goalEvidence);
  if (input.requireMissionCompletion) {
    assert.ok(missionCompletion, "Live mission has no durable successful Goal evaluation");
    assert.equal(checkpoint.active_cycle, null,
      "Completed live mission retained an active autonomous Cycle");
    assert.equal(checkpoint.goal_dag.status, "awaiting_model_selection",
      "Completed live mission retained an active Goal DAG");
    assert.equal(checkpoint.goal_dag.current_epoch_id, null,
      "Completed live mission retained an active Goal epoch");
  }
  assert.ok(checkpoint.world.frame > 0, "Live run committed no physical frame");
  assert.ok(checkpoint.world.worldRevision > 0, "Live run committed no world revision");
  assert.equal(checkpoint.world.robot.fallen, false, "Humanoid finished the live run fallen");
  assert.ok(checkpoint.total_model_calls > 0, "Live run recorded no model call");
  assert.ok(checkpoint.model_usage.total.requests > 0,
    "Live run recorded no provider usage request");
  assert.ok(Object.keys(checkpoint.model_usage.by_agent).length > 0,
    "Live run did not attribute provider usage to an Agent");

  const manifestAgents = Object.values(manifest.agents);
  assert.equal(new Set(manifestAgents.map((agent) => agent.agent_id)).size,
    manifestAgents.length, "Agent manifest does not contain one identity per role");
  assert.ok(manifestAgents.every((agent) => typeof agent.endpoint_sha256 === "string"),
    "Agent manifest is missing endpoint identity hashes");
  assert.ok(manifestAgents.every((agent) => !Object.hasOwn(agent, "endpoint")
    && !Object.hasOwn(agent, "api_key")),
  "Agent manifest contains raw provider configuration");

  const startedCalls = modelCalls.filter((record) => (
    isRecord(record) && record.lifecycle === "started"
  ));
  const completedCalls = modelCalls.filter((record) => (
    isRecord(record) && record.lifecycle === "completed"
  ));
  const completedCallIds = new Set(completedCalls.map((record) => record.model_call_id));
  const completedCallsById = new Map(completedCalls.map((record) => [
    record.model_call_id,
    record
  ]));
  const modelToolCalls = new Map(completedCalls.flatMap((record) => (
    Array.isArray(record.tool_calls)
      ? record.tool_calls.flatMap((toolCall) => (
          isRecord(toolCall) && typeof toolCall.tool_call_id === "string"
            ? [[toolCall.tool_call_id, {
                agentId: record.agent_id,
                toolName: toolCall.tool_name
              }]]
            : []
        ))
      : []
  )));
  assert.ok(startedCalls.length > 0, "Live run journal contains no model request");
  assert.ok(startedCalls.some((record) => completedCallIds.has(record.model_call_id)),
    "Live run contains no completed model request");
  for (const record of completedCalls) {
    assert.equal(typeof record.response_id, "string",
      `Completed model call has no response identity: ${record.model_call_id}`);
    assert.notEqual(record.response_id, "FAKE_ID",
      `Completed model call retained the SDK placeholder identity: ${record.model_call_id}`);
  }

  const planningActions = new Map(actions
    .filter((receipt) => isRecord(receipt) && isPlanningAction(receipt.action))
    .map((receipt) => [receipt.transactionId, receipt]));
  const executions = actions.filter((receipt) => (
    isRecord(receipt)
      && (receipt.action === "execute_whole_body_motion"
        || receipt.action === "execute_humanoid_navigation")
      && receipt.accepted === true
      && Number.isSafeInteger(receipt.frameCount)
      && receipt.frameCount > 0
      && Number.isSafeInteger(receipt.worldBeforeRevision)
      && Number.isSafeInteger(receipt.worldAfterRevision)
      && receipt.worldAfterRevision > receipt.worldBeforeRevision
  ));
  assert.ok(executions.length > 0, "Live run contains no accepted physical execution");

  for (const receipt of actions.filter(isRecord)) {
    assert.ok(isRecord(receipt.cycle),
      `Action has no autonomous cycle identity: ${receipt.transactionId}`);
    assert.ok(isRecord(receipt.decision),
      `Action has no model decision identity: ${receipt.transactionId}`);
    const decision = receipt.decision;
    const modelCall = completedCallsById.get(decision.model_call_id);
    assert.ok(modelCall,
      `Action references no completed model call: ${receipt.transactionId}`);
    assert.equal(modelCall.agent_id, receipt.agentId,
      `Action model call belongs to another Agent: ${receipt.transactionId}`);
    assert.deepEqual(modelCall.cycle, receipt.cycle,
      `Action model call belongs to another cycle: ${receipt.transactionId}`);
    assert.equal(decision.agent_id, receipt.agentId,
      `Action decision belongs to another Agent: ${receipt.transactionId}`);
    assert.equal(decision.agent_manifest_sha256, manifest.identity_sha256,
      `Action decision uses another Agent manifest: ${receipt.transactionId}`);
    assert.equal(decision.agent_manifest_epoch_id, manifest.epoch_id,
      `Action decision uses another manifest epoch: ${receipt.transactionId}`);
    assert.equal(decision.response_id, modelCall.response_id,
      `Action response identity changed: ${receipt.transactionId}`);
    assert.equal(decision.response_output_sha256, modelCall.response_output_sha256,
      `Action response content identity changed: ${receipt.transactionId}`);
    const matchingTools = Array.isArray(modelCall.tool_calls)
      ? modelCall.tool_calls.filter((toolCall) => (
          isRecord(toolCall)
            && toolCall.tool_call_id === receipt.transactionId
            && toolCall.tool_name === receipt.action
            && toolCall.arguments_sha256 === decision.tool_arguments_sha256
        ))
      : [];
    assert.equal(matchingTools.length, 1,
      `Action has no exact real-model tool decision: ${receipt.transactionId}`);
    assert.equal(decision.tool_call_id, receipt.transactionId,
      `Action tool identity changed: ${receipt.transactionId}`);
  }

  for (const execution of executions) {
    assert.ok(isRecord(execution.input), "Physical execution has no structured input");
    const planningTransactionId = execution.input.planning_transaction_id;
    assert.equal(typeof planningTransactionId, "string",
      "Physical execution is not bound to a planning transaction");
    const plan = planningActions.get(planningTransactionId);
    assert.ok(plan, `Physical execution references an unknown plan: ${planningTransactionId}`);
    assert.equal(plan.accepted, true, "Physical execution references a rejected plan");
    assert.equal(plan.agentId, "humanoid-motion-reference",
      "Physical plan did not originate from the motion Agent");
    assert.equal(execution.agentId, "humanoid-executor",
      "Physical execution did not originate from the executor Agent");
    assert.deepEqual(plan.cycle, execution.cycle,
      "Physical execution and plan belong to different autonomous cycles");
    assert.deepEqual(modelToolCalls.get(planningTransactionId), {
      agentId: "humanoid-motion-reference",
      toolName: plan.action
    }, "Physical plan has no matching real-model tool call");
    assert.deepEqual(modelToolCalls.get(execution.transactionId), {
      agentId: "humanoid-executor",
      toolName: execution.action
    }, "Physical execution has no matching real-model tool call");
  }

  const evidenceRefs = new Set(goalEvidence.flatMap((record) => (
    isRecord(record) && isRecord(record.evidence) && typeof record.evidence.ref === "string"
      ? [record.evidence.ref]
      : []
  )));
  for (const execution of executions) {
    assert.ok(evidenceRefs.has(`action:${execution.transactionId}`),
      `Physical execution has no durable Goal evidence: ${execution.transactionId}`);
  }

  const episodeByExecution = new Map(episodes
    .filter((episode) => isRecord(episode) && isRecord(episode.causal_trace))
    .map((episode) => [episode.causal_trace.execution_transaction_id, episode]));
  for (const execution of executions) {
    const episode = episodeByExecution.get(execution.transactionId);
    assert.ok(episode,
      `Physical execution has no causal embodied memory: ${execution.transactionId}`);
    assert.equal(episode.causal_trace.planning_transaction_id,
      execution.input.planning_transaction_id,
      `Embodied memory references another plan: ${execution.transactionId}`);
    assert.deepEqual(episode.causal_trace.cycle, execution.cycle,
      `Embodied memory references another cycle: ${execution.transactionId}`);
    assert.deepEqual(episode.causal_trace.execution_decision, execution.decision,
      `Embodied memory references another model decision: ${execution.transactionId}`);
    assert.ok(episode.causal_trace.goal_evidence_refs.includes(
      `action:${execution.transactionId}`
    ), `Embodied memory has no action evidence: ${execution.transactionId}`);
    assert.equal(typeof episode.causal_trace.memory_id, "string",
      `Embodied memory has no durable identity: ${execution.transactionId}`);
  }

  const exercisedAgents = [...new Set(completedCalls.flatMap((record) => (
    typeof record.agent_id === "string" ? [record.agent_id] : []
  )))].sort();
  for (const agentId of [
    "humanoid-goal-manager",
    "humanoid-coordinator",
    "humanoid-sentry",
    "humanoid-motion-reference",
    "humanoid-executor"
  ]) {
    assert.ok(exercisedAgents.includes(agentId), `Live run never exercised ${agentId}`);
  }

  const travelledDistance = Math.hypot(
    checkpoint.world.robot.rootPosition.x - input.scenario.robot.x,
    checkpoint.world.robot.rootPosition.z - input.scenario.robot.z
  );
  assert.ok(travelledDistance >= (input.minimumTravelledDistance ?? 0.25),
    "Live humanoid did not produce meaningful physical displacement");

  const acceptedPlanningReceipts = actions.filter((receipt) => (
    isRecord(receipt) && receipt.accepted === true && isPlanningAction(receipt.action)
  ));
  const planningBehaviorHashes = acceptedPlanningReceipts.map((receipt) => (
    autonomyPlanningBehaviorSha256(receipt.action, receipt.input)
  ));
  assert.ok(planningBehaviorHashes.length > 0,
    "Live run contains no accepted model planning behavior");

  const selectedGoalHashes = selectedGoals(checkpoint).map(autonomyContentSha256);
  assert.ok(selectedGoalHashes.length > 0,
    "Live run contains no model-selected Goal content");
  const modelResponseHashes = completedCalls.flatMap((record) => (
    typeof record.response_output_sha256 === "string"
      ? [record.response_output_sha256]
      : []
  ));
  assert.ok(modelResponseHashes.length > 0,
    "Live run contains no model response content identity");
  const actionSequence = actions.flatMap((receipt) => (
    isRecord(receipt) && typeof receipt.action === "string" ? [receipt.action] : []
  ));
  const physicalTrajectories = executions.map((execution) => {
    const detail = isRecord(execution.detail) ? execution.detail : {};
    const trajectory = PhysicalTrajectorySummarySchema.parse(detail.physical_trajectory);
    assert.equal(trajectory.complete_from_admission, true,
      `Physical trajectory lost its admitted prefix: ${execution.transactionId}`);
    assert.equal(trajectory.start_world_revision, execution.worldBeforeRevision,
      `Physical trajectory starts at another revision: ${execution.transactionId}`);
    assert.equal(trajectory.end_world_revision, execution.worldAfterRevision,
      `Physical trajectory ends at another revision: ${execution.transactionId}`);
    assert.equal(trajectory.observed_frame_count, execution.frameCount + 1,
      `Physical trajectory omits committed frames: ${execution.transactionId}`);
    return trajectory;
  });
  const physicalFrameCount = physicalTrajectories.reduce((total, trajectory) => (
    total + trajectory.observed_frame_count - 1
  ), 0);
  const worldMutationHashes = actions.flatMap((receipt) => {
    if (!isRecord(receipt) || receipt.accepted !== true
      || receipt.action !== "remove_world_block" || !isRecord(receipt.detail)) return [];
    const transaction = ScenarioBlockRemovalTransactionSchema.parse(
      receipt.detail.removal_transaction
    );
    return [autonomyContentSha256({
      block_id: transaction.block_id,
      source_world_frame: transaction.source_world_frame,
      source_world_revision: transaction.source_world_revision,
      expected_block_state_sha256: transaction.expected_block_state_sha256,
      projected_chunk_state_sha256: transaction.projected_chunk_state_sha256,
      contact_evidence: transaction.contact_evidence
    })];
  });
  const physicalBehaviorSignature = autonomyContentSha256({
    trajectories: physicalTrajectories,
    world_mutation_hashes: worldMutationHashes
  });
  const completedGoalCount = checkpoint.goal_dag.epochs.filter((epoch) => (
    epoch.status === "completed"
  )).length;

  return {
    version: 3,
    run_id: checkpoint.run_id,
    scenario_id: checkpoint.scenario_id,
    seed: input.scenario.seed,
    run_mode: input.store.definition.run_mode,
    status: checkpoint.status,
    physical_verified: true,
    world_frame: checkpoint.world.frame,
    world_revision: checkpoint.world.worldRevision,
    cycle_count: checkpoint.cycle_index,
    completed_goal_count: completedGoalCount,
    model_call_count: startedCalls.length,
    model_usage: checkpoint.model_usage,
    exercised_agents: exercisedAgents,
    physical_execution_count: executions.length,
    physical_frame_count: physicalFrameCount,
    travelled_distance_m: travelledDistance,
    evidence_count: evidenceRefs.size,
    context_compaction_count: checkpoint.context_memory.total_compactions,
    embodied_episode_count: checkpoint.embodied_memory.total_episodes,
    causal_episode_count: episodeByExecution.size,
    ...(missionCompletion
      ? { mission_goal_evaluation_ref: missionCompletion.goal_evaluation_ref }
      : {}),
    selected_goal_hashes: selectedGoalHashes,
    planning_behavior_hashes: planningBehaviorHashes,
    model_response_hashes: modelResponseHashes,
    action_sequence: actionSequence,
    goal_signature: autonomyContentSha256(selectedGoalHashes),
    planning_signature: autonomyContentSha256(planningBehaviorHashes),
    model_decision_signature: autonomyContentSha256(completedCalls.map((record) => ({
      agent_id: record.agent_id,
      response_output_sha256: record.response_output_sha256,
      tool_names: Array.isArray(record.tool_calls)
        ? record.tool_calls.flatMap((toolCall) => (
            isRecord(toolCall) && typeof toolCall.tool_name === "string"
              ? [toolCall.tool_name]
              : []
          ))
        : []
    }))),
    physical_behavior_signature: physicalBehaviorSignature,
    physical_trajectories: physicalTrajectories,
    world_mutation_hashes: worldMutationHashes
  };
}

function isPlanningAction(action) {
  return action === "plan_whole_body_motion"
    || action === "plan_whole_body_motion_candidates"
    || action === "plan_humanoid_navigation";
}

function selectedGoals(checkpoint) {
  return checkpoint.goal_dag.epochs.flatMap((epoch) => {
    const candidate = checkpoint.goal_dag.candidates[epoch.candidate_id];
    return candidate ? [candidate.goal] : [];
  });
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
