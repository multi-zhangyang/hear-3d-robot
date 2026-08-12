import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import type { AgentManifest } from "../../domain/agent-manifest.js";
import { modelPayloadSha256 } from "../../domain/model-call-authority.js";
import { ScenarioSchema } from "../../domain/schema.js";
import { RunStore } from "../../persistence/run-store.js";
import { HumanoidWorld } from "../../world/humanoid/world.js";
import { HUMANOID_AGENT_IDS } from "./agents.js";
import { createHumanoidRunCheckpoint } from "./run-checkpoint.js";
import { HumanoidRunRuntime } from "./run-runtime.js";

const scenario = ScenarioSchema.parse({
  title: "动作来源验证场",
  seed: 91,
  bounds: { width: 8, depth: 8 },
  visibility_radius: 5,
  robot: { x: 2, z: 2, yaw: 0 },
  obstacles: [],
  objects: [],
  zones: [],
  default_goal: {
    summary: "保持当前站立位置",
    predicates: [{
      type: "robot_at",
      target: { x: 2, y: 0, z: 2 },
      tolerance: 0.3
    }]
  }
});

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((directory) => (
    rm(directory, { recursive: true, force: true })
  )));
});

describe("humanoid action model authority", () => {
  it("rejects invented provenance and recovers only the exact durable decision", async () => {
    const root = await mkdtemp(join(tmpdir(), "hear-action-authority-"));
    temporaryDirectories.push(root);
    const store = await RunStore.create(root, {
      mission: "验证真实模型动作来源",
      scenarioId: "action-authority-test",
      scenario,
      goal: scenario.default_goal,
      runtime: "humanoid_g1"
    });
    const world = await HumanoidWorld.create(scenario);
    let resumedWorld: HumanoidWorld | undefined;
    try {
      const checkpoint = createHumanoidRunCheckpoint({
        store,
        goal: scenario.default_goal,
        world
      });
      await store.writeCheckpoint(checkpoint);
      const runtime = new HumanoidRunRuntime({
        store,
        goal: scenario.default_goal,
        world,
        checkpoint
      });
      const manifest = await activateGoal(runtime);

      const incompleteTransaction = "observe-without-response";
      const incompleteModelCall = await runtime.recordModelCallStarted(
        HUMANOID_AGENT_IDS.motion
      );
      await expect(runtime.invoke(
        "observe_humanoid",
        {},
        incompleteTransaction,
        HUMANOID_AGENT_IDS.motion,
        {
          tool_call_id: incompleteTransaction,
          tool_name: "observe_humanoid",
          arguments_sha256: modelPayloadSha256({})
        }
      )).rejects.toThrow("no completed model response authority");
      await runtime.recordModelCallFailed(
        incompleteModelCall,
        HUMANOID_AGENT_IDS.motion
      );

      const transactionId = "observe-authoritative";
      const directSentryAuthority = await authorizeAction(
        runtime,
        transactionId,
        HUMANOID_AGENT_IDS.sentry
      );
      await expect(runtime.invoke(
        "observe_humanoid",
        {},
        transactionId,
        HUMANOID_AGENT_IDS.sentry,
        directSentryAuthority
      )).rejects.toThrow("requires Coordinator deterministic delegation");

      const delegatedTransactionId = "observe-coordinator-delegation";
      const authority = await authorizeSentryDelegation(
        runtime,
        delegatedTransactionId
      );
      await expect(runtime.invoke(
        "observe_humanoid",
        {},
        delegatedTransactionId,
        HUMANOID_AGENT_IDS.sentry,
        { ...authority, arguments_sha256: "0".repeat(64) }
      )).rejects.toThrow("delegation input has no model authority");
      await expect(runtime.invoke(
        "observe_humanoid",
        {},
        delegatedTransactionId,
        HUMANOID_AGENT_IDS.sentry,
        { ...authority, tool_name: "plan_humanoid_navigation" }
      )).rejects.toThrow("deterministic delegation is not authoritative");
      await expect(runtime.invoke(
        "observe_humanoid",
        {},
        delegatedTransactionId,
        HUMANOID_AGENT_IDS.sentry,
        { ...authority, tool_call_id: "another-tool-call" }
      )).rejects.toThrow("tool authority mismatch");
      await expect(runtime.invoke(
        "observe_humanoid",
        {},
        delegatedTransactionId,
        HUMANOID_AGENT_IDS.motion,
        authority
      )).rejects.toThrow("deterministic delegation is not authoritative");

      const crossRoleTransaction = "executor-invented-observation";
      const crossRoleAuthority = await authorizeAction(
        runtime,
        crossRoleTransaction,
        HUMANOID_AGENT_IDS.executor
      );
      await expect(runtime.invoke(
        "observe_humanoid",
        {},
        crossRoleTransaction,
        HUMANOID_AGENT_IDS.executor,
        crossRoleAuthority
      )).rejects.toThrow("outside Agent role authority");

      const directExecutorTransaction = "executor-direct-physical-response";
      const directExecutorInput = { planning_transaction_id: "invented-plan" };
      const directExecutorAuthority = await authorizeDirectAction(
        runtime,
        "execute_whole_body_motion",
        directExecutorInput,
        directExecutorTransaction,
        HUMANOID_AGENT_IDS.executor
      );
      await expect(runtime.invoke(
        "execute_whole_body_motion",
        directExecutorInput,
        directExecutorTransaction,
        HUMANOID_AGENT_IDS.executor,
        directExecutorAuthority
      )).rejects.toThrow("requires Coordinator deterministic delegation");

      const receipt = await runtime.invoke(
        "observe_humanoid",
        {},
        delegatedTransactionId,
        HUMANOID_AGENT_IDS.sentry,
        authority
      );
      expect(receipt).toMatchObject({
        transactionId: delegatedTransactionId,
        agentId: HUMANOID_AGENT_IDS.sentry,
        accepted: true,
        decision: {
          agent_id: HUMANOID_AGENT_IDS.coordinator,
          tool_call_id: delegatedTransactionId,
          tool_arguments_sha256: modelPayloadSha256({})
        },
        cycle: {
          cycle_id: runtime.checkpoint.active_cycle!.cycle_id,
          cycle_index: 1
        }
      });
      const actionRecords = await store.readJournal("actions");
      expect(actionRecords).toContainEqual(expect.objectContaining({
        transactionId: delegatedTransactionId,
        decision: receipt.decision,
        cycle: receipt.cycle
      }));

      const persisted = await store.readHumanoidCheckpoint();
      resumedWorld = await HumanoidWorld.create(scenario, persisted.world_checkpoint);
      const resumed = new HumanoidRunRuntime({
        store,
        goal: scenario.default_goal,
        world: resumedWorld,
        checkpoint: persisted
      });
      await resumed.initializeGoalAutonomy(manifest);
      expect(await resumed.invoke(
        "observe_humanoid",
        {},
        delegatedTransactionId,
        HUMANOID_AGENT_IDS.sentry,
        authority
      )).toEqual(receipt);
      await expect(resumed.invoke(
        "observe_humanoid",
        { changed: true },
        delegatedTransactionId,
        HUMANOID_AGENT_IDS.sentry,
        authority
      )).rejects.toThrow("transaction conflict");

      await authorizeSentryDelegation(
        resumed,
        delegatedTransactionId
      );
      await expect(resumed.invoke(
        "observe_humanoid",
        {},
        delegatedTransactionId,
        HUMANOID_AGENT_IDS.sentry,
        authority
      )).rejects.toThrow("no completed model response authority");
    } finally {
      await resumedWorld?.dispose();
      await world.dispose();
    }
  }, 30_000);
});

async function activateGoal(runtime: HumanoidRunRuntime): Promise<AgentManifest> {
  const manifest = {
    epoch_id: "11111111-1111-4111-8111-111111111111",
    identity_sha256: "a".repeat(64),
    agents: {
      goal_manager: { agent_id: HUMANOID_AGENT_IDS.goalManager },
      coordinator: { agent_id: HUMANOID_AGENT_IDS.coordinator },
      sentry: { agent_id: HUMANOID_AGENT_IDS.sentry },
      motion: { agent_id: HUMANOID_AGENT_IDS.motion },
      executor: { agent_id: HUMANOID_AGENT_IDS.executor }
    }
  } as AgentManifest;
  await runtime.initializeGoalAutonomy(manifest);
  runtime.contextAnchor(HUMANOID_AGENT_IDS.goalManager);
  const proposalInput = {
    candidates: [
      {
        proposal_id: "authority-primary",
        mission_link: "验证动作来源",
        goal: scenario.default_goal,
        dependency_candidate_ids: []
      },
      {
        proposal_id: "authority-alternative",
        mission_link: "保留另一个真实模型候选",
        goal: {
          ...scenario.default_goal,
          summary: "保持当前站立位置（候选）"
        },
        dependency_candidate_ids: []
      }
    ]
  };
  const proposalAuthority = await authorizeGoalTool(
    runtime,
    "submit_goal_candidates",
    proposalInput
  );
  const submitted = await runtime.submitGoalCandidates(
    proposalInput,
    proposalAuthority
  ) as { candidates: Array<{ candidate_sequence: number }> };
  runtime.contextAnchor(HUMANOID_AGENT_IDS.goalManager);
  const selectionInput = {
    candidate_sequence: submitted.candidates[0]!.candidate_sequence
  };
  await runtime.selectGoalCandidate(
    selectionInput,
    await authorizeGoalTool(runtime, "select_goal_candidate", selectionInput)
  );
  return manifest;
}

async function authorizeAction(
  runtime: HumanoidRunRuntime,
  transactionId: string,
  agentId: string
) {
  const modelCallId = await runtime.recordModelCallStarted(agentId);
  const argumentsSha256 = modelPayloadSha256({});
  await runtime.recordModelCallCompleted({
    modelCallId,
    agentId,
    responseId: `response-${modelCallId}`,
    responseOutputSha256: modelPayloadSha256({ modelCallId, transactionId }),
    toolCalls: [{
      toolCallId: transactionId,
      toolName: "observe_humanoid",
      argumentsSha256
    }]
  });
  return {
    tool_call_id: transactionId,
    tool_name: "observe_humanoid",
    arguments_sha256: argumentsSha256
  };
}

async function authorizeDirectAction(
  runtime: HumanoidRunRuntime,
  action: Parameters<HumanoidRunRuntime["invoke"]>[0],
  input: unknown,
  transactionId: string,
  agentId: string
) {
  const modelCallId = await runtime.recordModelCallStarted(agentId);
  const argumentsSha256 = modelPayloadSha256(input);
  await runtime.recordModelCallCompleted({
    modelCallId,
    agentId,
    responseId: `response-${modelCallId}`,
    responseOutputSha256: modelPayloadSha256({ modelCallId, transactionId }),
    toolCalls: [{
      toolCallId: transactionId,
      toolName: action,
      argumentsSha256
    }]
  });
  return {
    tool_call_id: transactionId,
    tool_name: action,
    arguments_sha256: argumentsSha256
  };
}

async function authorizeSentryDelegation(
  runtime: HumanoidRunRuntime,
  transactionId: string
) {
  const sourceInput = {};
  const actionInput = {};
  const modelCallId = await runtime.recordModelCallStarted(
    HUMANOID_AGENT_IDS.coordinator
  );
  const argumentsSha256 = modelPayloadSha256(sourceInput);
  await runtime.recordModelCallCompleted({
    modelCallId,
    agentId: HUMANOID_AGENT_IDS.coordinator,
    responseId: `response-${modelCallId}`,
    responseOutputSha256: modelPayloadSha256({ modelCallId, transactionId }),
    toolCalls: [{
      toolCallId: transactionId,
      toolName: "delegate_humanoid_sentry",
      argumentsSha256
    }]
  });
  return {
    tool_call_id: transactionId,
    tool_name: "delegate_humanoid_sentry",
    arguments_sha256: argumentsSha256,
    deterministic_delegation: {
      contract_id: "grounding_monitor_v1" as const,
      source_input: sourceInput,
      action_input_sha256: modelPayloadSha256(actionInput)
    }
  };
}

async function authorizeGoalTool(
  runtime: HumanoidRunRuntime,
  toolName: string,
  input: unknown
) {
  const modelCallId = await runtime.recordModelCallStarted(
    HUMANOID_AGENT_IDS.goalManager
  );
  const toolCallId = `${toolName}-${modelCallId}`;
  const argumentsSha256 = modelPayloadSha256(input);
  await runtime.recordModelCallCompleted({
    modelCallId,
    agentId: HUMANOID_AGENT_IDS.goalManager,
    responseId: `response-${modelCallId}`,
    responseOutputSha256: modelPayloadSha256({ modelCallId, toolCallId }),
    toolCalls: [{
      toolCallId,
      toolName,
      argumentsSha256
    }]
  });
  return {
    tool_call_id: toolCallId,
    tool_name: toolName,
    arguments_sha256: argumentsSha256
  };
}
