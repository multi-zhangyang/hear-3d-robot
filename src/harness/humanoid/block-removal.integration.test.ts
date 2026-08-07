import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  autonomousCycleRef,
  type AutonomousCycleRef
} from "../../domain/autonomous-cycle.js";
import type { AgentManifest } from "../../domain/agent-manifest.js";
import {
  modelPayloadSha256,
  type ModelDecisionRef
} from "../../domain/model-call-authority.js";
import { resolveScenarioChunkDeltaEntity } from "../../domain/scenario-chunk-delta.js";
import {
  ScenarioSchema,
  type Goal,
  type JsonValue
} from "../../domain/schema.js";
import { RunStore } from "../../persistence/run-store.js";
import type { RuntimeEvent } from "../../runtime/events.js";
import {
  createHumanoidGoalProgress,
  inspectHumanoidGoal
} from "../../runtime/humanoid-checker.js";
import type { HumanoidMotionOptionContract } from "../../world/humanoid/motion-option.js";
import { HumanoidWorld } from "../../world/humanoid/world.js";
import { HUMANOID_AGENT_IDS } from "./agents.js";
import { createActionGoalEvidence } from "./goal-evidence.js";
import { createHumanoidRunCheckpoint } from "./run-checkpoint.js";
import { HumanoidRunRuntime } from "./run-runtime.js";
import {
  humanoidActionFingerprint,
  type HumanoidActionReceipt,
  type HumanoidActionToolCallAuthority
} from "./runtime.js";

const BLOCK_ID = "frontier-block";
const PLAN_TRANSACTION_ID = "contact-plan";
const EXECUTION_TRANSACTION_ID = "contact-execution";
const REMOVAL_TRANSACTION_ID = "remove-frontier-block";
const OBSERVATION_TRANSACTION_ID = "observe-after-frontier-block-removal";
const CONTACT_STABLE_FRAMES = 8;
const CONTACT_FORCE_N = 9;

const scenario = ScenarioSchema.parse({
  title: "方块事务集成场",
  seed: 103,
  bounds: { width: 8, depth: 8 },
  visibility_radius: 6,
  robot: { x: 2, z: 2, yaw: 0 },
  obstacles: [{
    id: BLOCK_ID,
    center: { x: 2, y: 0.5, z: 4 },
    size: { x: 1, y: 1, z: 1 }
  }],
  objects: [],
  zones: [],
  default_goal: {
    summary: "拆除前方已接触方块",
    predicates: [{ type: "block_removed", block_id: BLOCK_ID }]
  }
});

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((directory) => (
    rm(directory, { recursive: true, force: true })
  )));
});

describe("block-removal autonomous transaction", () => {
  it("persists model-selected removal through world synchronization, Cycle memory and restart", async () => {
    const root = await mkdtemp(join(tmpdir(), "hear-block-removal-integration-"));
    temporaryDirectories.push(root);
    const store = await RunStore.create(root, {
      mission: "自主选择并物理拆除一个可观察方块",
      scenarioId: "block-removal-integration",
      scenario,
      goal: scenario.default_goal,
      runtime: "humanoid_g1"
    });
    const world = await HumanoidWorld.create(scenario);
    let resumedWorld: HumanoidWorld | undefined;
    try {
      const initial = createHumanoidRunCheckpoint({
        store,
        goal: scenario.default_goal,
        world
      });
      await store.writeCheckpoint(initial);
      const events: RuntimeEvent[] = [];
      const bootstrap = new HumanoidRunRuntime({
        store,
        goal: scenario.default_goal,
        world,
        checkpoint: initial
      });
      const manifest = await activateGoal(bootstrap, scenario.default_goal);
      expect(world.observe().solidTokens.map(({ id }) => id)).toContain(BLOCK_ID);

      const seeded = await seedCertifiedContactExecution({
        runtime: bootstrap,
        world,
        store,
        manifest
      });
      const runtime = new HumanoidRunRuntime({
        store,
        goal: scenario.default_goal,
        world,
        checkpoint: seeded.checkpoint,
        eventSink: (event) => {
          events.push(event);
        }
      });
      await runtime.initializeGoalAutonomy(manifest);
      expect(runtime.executorDelegationAvailable()).toBe(true);

      const removalInput = {
        solid_id: BLOCK_ID,
        execution_transaction_id: EXECUTION_TRANSACTION_ID
      };
      const removalAuthority = await authorizeAction(
        runtime,
        "remove_world_block",
        removalInput,
        REMOVAL_TRANSACTION_ID,
        HUMANOID_AGENT_IDS.executor
      );
      const removal = await runtime.invoke(
        "remove_world_block",
        removalInput,
        REMOVAL_TRANSACTION_ID,
        HUMANOID_AGENT_IDS.executor,
        removalAuthority
      );

      expect(removal).toMatchObject({
        transactionId: REMOVAL_TRANSACTION_ID,
        accepted: true,
        code: "world_block_removal_authorized",
        frameCount: 0,
        cycle: seeded.cycle,
        detail: {
          solid_id: BLOCK_ID,
          execution_transaction_id: EXECUTION_TRANSACTION_ID,
          automatic_actuation: false,
          removal_transaction: {
            transaction_id: REMOVAL_TRANSACTION_ID,
            solid_id: BLOCK_ID,
            execution_transaction_id: EXECUTION_TRANSACTION_ID,
            planning_transaction_id: PLAN_TRANSACTION_ID,
            base_chunk_revision: 0,
            projected_chunk_revision: 1,
            contact_evidence: {
              planned_stable_frames: CONTACT_STABLE_FRAMES,
              observed_stable_frames: CONTACT_STABLE_FRAMES,
              observed_maximum_normal_force_n: CONTACT_FORCE_N
            }
          }
        }
      });
      expect(runtime.checkpoint.action_commit_outbox.pending).toEqual({});

      const chunks = await store.readScenarioChunkDeltaState();
      expect(chunks.revision).toBe(1);
      expect(resolveScenarioChunkDeltaEntity(scenario, chunks, BLOCK_ID)).toMatchObject({
        category: "block",
        state: {
          present: false,
          properties: {
            hear_block_removal_v1: {
              transaction_id: REMOVAL_TRANSACTION_ID,
              execution_transaction_id: EXECUTION_TRANSACTION_ID
            }
          }
        }
      });
      expect(world.observe().solidTokens.map(({ id }) => id)).not.toContain(BLOCK_ID);
      const synchronizedWorldRevision = world.snapshot().worldRevision;
      expect(synchronizedWorldRevision).toBe(removal.worldAfterRevision + 1);
      expect(runtime.checkpoint.checker).toMatchObject({ success: true });
      expect(runtime.executorDelegationAvailable()).toBe(false);
      expect(events).toEqual(expect.arrayContaining([
        expect.objectContaining({
          type: "humanoid_action_committed",
          data: expect.objectContaining({
            block_removal: expect.objectContaining({ solid_id: BLOCK_ID }),
            scenario_chunks: expect.objectContaining({ revision: 1 })
          })
        }),
        expect.objectContaining({
          type: "humanoid_scenario_synchronized",
          data: expect.objectContaining({
            scenario_chunks: expect.objectContaining({ revision: 1 }),
            synchronization: expect.objectContaining({ changed: true })
          })
        })
      ]));

      const observationAuthority = await authorizeAction(
        runtime,
        "observe_humanoid",
        {},
        OBSERVATION_TRANSACTION_ID,
        HUMANOID_AGENT_IDS.sentry
      );
      const observation = await runtime.invoke(
        "observe_humanoid",
        {},
        OBSERVATION_TRANSACTION_ID,
        HUMANOID_AGENT_IDS.sentry,
        observationAuthority
      );
      expect(observation).toMatchObject({
        accepted: true,
        code: "humanoid_observed",
        worldBeforeRevision: synchronizedWorldRevision
      });

      expect(runtime.validateCycleEvidence([
        EXECUTION_TRANSACTION_ID,
        REMOVAL_TRANSACTION_ID
      ])).toMatchObject({
        transactionId: EXECUTION_TRANSACTION_ID,
        code: "motion_option_succeeded"
      });
      expect(await runtime.completeCycle(JSON.stringify({
        summary: "稳定接触后提交方块拆除事务",
        evidence_transaction_ids: [
          EXECUTION_TRANSACTION_ID,
          REMOVAL_TRANSACTION_ID
        ]
      }))).toBe(true);

      const completed = await store.readHumanoidCheckpoint();
      expect(completed).toMatchObject({
        status: "succeeded",
        active_cycle: null,
        final_output: expect.stringContaining("稳定接触后提交方块拆除事务")
      });
      expect(completed.goal_dag.status).toBe("awaiting_model_selection");
      await expect(runtime.recordModelCallStarted(
        HUMANOID_AGENT_IDS.goalManager
      )).rejects.toThrow("cannot accept new model decisions while succeeded");
      await expect(runtime.ensureAutonomousCycle()).rejects.toThrow(
        "cannot accept new model decisions while succeeded"
      );
      await expect(runtime.recallGoalHistory({ limit: 1 })).rejects.toThrow(
        "cannot accept new model decisions while succeeded"
      );
      await expect(runtime.submitGoalCandidates({
        candidates: []
      } as never, {
        tool_call_id: "terminal-goal-submit",
        tool_name: "submit_goal_candidates",
        arguments_sha256: "0".repeat(64)
      })).rejects.toThrow("cannot accept new model decisions while succeeded");
      expect(completed.embodied_memory.recent_episodes.at(-1)).toMatchObject({
        transaction_id: EXECUTION_TRANSACTION_ID,
        goal_success: true,
        result_world_revision: synchronizedWorldRevision,
        causal_trace: {
          execution_transaction_id: EXECUTION_TRANSACTION_ID,
          world_mutation_transaction_ids: [REMOVAL_TRANSACTION_ID],
          goal_evidence_refs: expect.arrayContaining([
            `action:${EXECUTION_TRANSACTION_ID}`,
            `action:${REMOVAL_TRANSACTION_ID}`
          ])
        },
        world_mutations: [{
          transaction_id: REMOVAL_TRANSACTION_ID,
          action: "remove_world_block",
          execution_transaction_id: EXECUTION_TRANSACTION_ID,
          solid_id: BLOCK_ID,
          chunk_before_revision: 0,
          chunk_after_revision: 1
        }]
      });

      const resumedStore = await RunStore.open(store.runDir);
      const resumedChunks = await resumedStore.readScenarioChunkDeltaState();
      resumedWorld = await HumanoidWorld.create(
        scenario,
        completed.world_checkpoint,
        { scenarioChunks: resumedChunks }
      );
      const resumed = new HumanoidRunRuntime({
        store: resumedStore,
        goal: scenario.default_goal,
        world: resumedWorld,
        checkpoint: completed
      });
      await resumed.initializeGoalAutonomy(manifest);
      const beforeReplay = resumedWorld.snapshot();
      expect(resumedWorld.observe().solidTokens.map(({ id }) => id)).not.toContain(BLOCK_ID);
      expect(await resumed.invoke(
        "remove_world_block",
        removalInput,
        REMOVAL_TRANSACTION_ID,
        HUMANOID_AGENT_IDS.executor,
        removalAuthority
      )).toEqual(removal);
      expect(resumedWorld.snapshot()).toEqual(beforeReplay);
      expect((await resumedStore.readScenarioChunkDeltaState()).revision).toBe(1);

      const actions = await resumedStore.readJournal("actions");
      expect(actions.filter((entry) => field(entry, "transactionId")
        === REMOVAL_TRANSACTION_ID)).toHaveLength(1);
      const durableEvents = await resumedStore.readJournal("events");
      expect(durableEvents.filter((entry) => (
        field(entry, "type") === "run_succeeded"
      ))).toHaveLength(1);
      expect(durableEvents.filter((entry) => (
        field(entry, "type") === "humanoid_action_committed"
          && nestedField(entry, "data", "receipt", "transactionId")
            === REMOVAL_TRANSACTION_ID
      ))).toHaveLength(1);
      expect(await resumedStore.readJournal("episodes")).toHaveLength(1);
    } finally {
      await resumedWorld?.dispose();
      await world.dispose();
    }
  }, 45_000);
});

async function seedCertifiedContactExecution(input: {
  runtime: HumanoidRunRuntime;
  world: HumanoidWorld;
  store: RunStore;
  manifest: AgentManifest;
}): Promise<{
  checkpoint: ReturnType<HumanoidRunRuntime["checkpoint"]>;
  cycle: AutonomousCycleRef;
}> {
  const activeCycle = input.runtime.checkpoint.active_cycle;
  if (!activeCycle) throw new Error("Test Goal selection did not create an active Cycle");
  const cycle = autonomousCycleRef(activeCycle);
  const contract = contactContract();
  const planInput = {
    objective: "以左掌稳定接触模型选择的前方方块",
    termination: contract,
    candidates: [{
      id: "left-palm-contact-primary",
      intent: "左掌连续接触目标方块",
      duration_seconds: 0.16,
      keyframes: [{ at_seconds: 0 }, { at_seconds: 0.16 }]
    }, {
      id: "left-palm-contact-alternative",
      intent: "调整躯干后连续接触目标方块",
      duration_seconds: 0.16,
      keyframes: [{ at_seconds: 0 }, { at_seconds: 0.16, torso_yaw: 0.01 }]
    }]
  };
  const planningDecision = await recordDecision({
    runtime: input.runtime,
    manifest: input.manifest,
    agentId: HUMANOID_AGENT_IDS.motion,
    toolName: "plan_whole_body_motion_candidates",
    toolCallId: PLAN_TRANSACTION_ID,
    toolInput: planInput
  });
  const executionInput = { planning_transaction_id: PLAN_TRANSACTION_ID };
  const executionDecision = await recordDecision({
    runtime: input.runtime,
    manifest: input.manifest,
    agentId: HUMANOID_AGENT_IDS.executor,
    toolName: "execute_whole_body_motion",
    toolCallId: EXECUTION_TRANSACTION_ID,
    toolInput: executionInput
  });
  const before = input.world.snapshot();
  for (let frame = 0; frame < CONTACT_STABLE_FRAMES; frame += 1) {
    await input.world.advanceStationary();
  }
  const cut = await input.world.capturePersistenceState();
  expect(cut.world.worldRevision - before.worldRevision).toBe(CONTACT_STABLE_FRAMES);

  const plan = receipt({
    transactionId: PLAN_TRANSACTION_ID,
    agentId: HUMANOID_AGENT_IDS.motion,
    decision: planningDecision,
    cycle,
    action: "plan_whole_body_motion_candidates",
    input: asJson(planInput),
    accepted: true,
    code: "whole_body_candidates_validated",
    worldBeforeRevision: before.worldRevision,
    worldAfterRevision: before.worldRevision,
    frameCount: 0,
    detail: {
      plan_id: "certified-contact-plan",
      termination: contract,
      option: { contract }
    }
  });
  const execution = receipt({
    transactionId: EXECUTION_TRANSACTION_ID,
    agentId: HUMANOID_AGENT_IDS.executor,
    decision: executionDecision,
    cycle,
    action: "execute_whole_body_motion",
    input: executionInput,
    accepted: true,
    code: "motion_option_succeeded",
    worldBeforeRevision: before.worldRevision,
    worldAfterRevision: cut.world.worldRevision,
    frameCount: CONTACT_STABLE_FRAMES,
    detail: {
      planning_transaction_id: PLAN_TRANSACTION_ID,
      planning_action: "plan_whole_body_motion_candidates",
      plan_id: "certified-contact-plan",
      candidate_count: 2,
      selected_rank: 1,
      selected_candidate_id: "left-palm-contact-primary",
      result: {
        option: {
          option_id: contract.option_id,
          status: "succeeded",
          termination_reason: "physical_success",
          full_frame_count: CONTACT_STABLE_FRAMES,
          executed_prefix_frame_count: CONTACT_STABLE_FRAMES,
          predicted_termination_frame: CONTACT_STABLE_FRAMES,
          actual_termination_frame: CONTACT_STABLE_FRAMES,
          artifact_sha256: "b".repeat(64),
          evidence: {
            monitor: { terminalStableSteps: CONTACT_STABLE_FRAMES },
            predicates: [{
              predicateIndex: 0,
              type: "hand_contact_solid",
              status: "satisfied",
              handSurface: "left_hand_palm_link",
              solidId: BLOCK_ID,
              solidObservable: true,
              maximumNormalForce: CONTACT_FORCE_N,
              minimumNormalForce: 7
            }]
          }
        }
      }
    }
  });
  const checkpoint = input.runtime.checkpoint;
  checkpoint.world = cut.world;
  checkpoint.world_checkpoint = cut.worldCheckpoint;
  checkpoint.goal_progress = createHumanoidGoalProgress(
    scenario.default_goal,
    cut.world
  );
  checkpoint.checker = inspectHumanoidGoal(
    scenario.default_goal,
    scenario,
    cut.world,
    checkpoint.goal_progress
  );
  checkpoint.committed_actions[PLAN_TRANSACTION_ID] = plan;
  checkpoint.committed_actions[EXECUTION_TRANSACTION_ID] = execution;
  await input.store.append("goal_evidence", asJson(createActionGoalEvidence({
    transactionId: EXECUTION_TRANSACTION_ID,
    worldFrame: cut.world.frame,
    worldRevision: cut.world.worldRevision,
    receipt: asJson(execution)
  })));
  await input.store.writeCheckpoint(checkpoint);
  return { checkpoint, cycle };
}

function contactContract(): HumanoidMotionOptionContract {
  return {
    option_id: "remove-frontier-block-contact",
    predicates: [{
      type: "hand_contact_solid",
      hand_surface: "left_hand_palm_link",
      solid_id: BLOCK_ID,
      minimum_normal_force: 7
    }],
    stable_steps: CONTACT_STABLE_FRAMES,
    phases: null
  };
}

function receipt(input: Omit<
  HumanoidActionReceipt,
  "fingerprint" | "channels" | "committedAt"
>): HumanoidActionReceipt {
  return {
    ...input,
    fingerprint: humanoidActionFingerprint(input.action, input.agentId, input.input),
    channels: [],
    committedAt: new Date().toISOString()
  };
}

async function activateGoal(
  runtime: HumanoidRunRuntime,
  goal: Goal
): Promise<AgentManifest> {
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
    candidates: [{
      proposal_id: "remove-visible-block",
      mission_link: "自主拆除当前可观察方块",
      goal,
      dependency_candidate_ids: []
    }, {
      proposal_id: "inspect-visible-block",
      mission_link: "保留另一个由模型提出的真实候选",
      goal: { ...structuredClone(goal), summary: "观察后再拆除当前方块" },
      dependency_candidate_ids: []
    }]
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

async function authorizeGoalTool(
  runtime: HumanoidRunRuntime,
  toolName: string,
  toolInput: unknown
) {
  const modelCallId = await runtime.recordModelCallStarted(
    HUMANOID_AGENT_IDS.goalManager
  );
  const toolCallId = `${toolName}-${modelCallId}`;
  const argumentsSha256 = modelPayloadSha256(toolInput);
  await runtime.recordModelCallCompleted({
    modelCallId,
    agentId: HUMANOID_AGENT_IDS.goalManager,
    responseId: `response-${modelCallId}`,
    responseOutputSha256: modelPayloadSha256({ modelCallId, toolCallId }),
    toolCalls: [{ toolCallId, toolName, argumentsSha256 }]
  });
  return {
    tool_call_id: toolCallId,
    tool_name: toolName,
    arguments_sha256: argumentsSha256
  };
}

async function authorizeAction(
  runtime: HumanoidRunRuntime,
  action: Parameters<HumanoidRunRuntime["invoke"]>[0],
  actionInput: unknown,
  transactionId: string,
  agentId: string
): Promise<HumanoidActionToolCallAuthority> {
  const modelCallId = await runtime.recordModelCallStarted(agentId);
  const argumentsSha256 = modelPayloadSha256(actionInput);
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

async function recordDecision(input: {
  runtime: HumanoidRunRuntime;
  manifest: AgentManifest;
  agentId: string;
  toolName: string;
  toolCallId: string;
  toolInput: unknown;
}): Promise<ModelDecisionRef> {
  const modelCallId = await input.runtime.recordModelCallStarted(input.agentId);
  const responseId = `response-${modelCallId}`;
  const responseOutputSha256 = modelPayloadSha256({
    modelCallId,
    toolCallId: input.toolCallId
  });
  const toolArgumentsSha256 = modelPayloadSha256(input.toolInput);
  await input.runtime.recordModelCallCompleted({
    modelCallId,
    agentId: input.agentId,
    responseId,
    responseOutputSha256,
    toolCalls: [{
      toolCallId: input.toolCallId,
      toolName: input.toolName,
      argumentsSha256: toolArgumentsSha256
    }]
  });
  return {
    agent_id: input.agentId,
    agent_manifest_sha256: input.manifest.identity_sha256,
    agent_manifest_epoch_id: input.manifest.epoch_id,
    model_call_id: modelCallId,
    response_id: responseId,
    response_output_sha256: responseOutputSha256,
    tool_call_id: input.toolCallId,
    tool_arguments_sha256: toolArgumentsSha256
  };
}

function asJson(value: unknown): JsonValue {
  return JSON.parse(JSON.stringify(value)) as JsonValue;
}

function field(value: unknown, key: string): unknown {
  return nestedField(value, key);
}

function nestedField(value: unknown, ...path: string[]): unknown {
  let current = value;
  for (const key of path) {
    if (current === null || typeof current !== "object" || Array.isArray(current)) {
      return undefined;
    }
    current = (current as Record<string, unknown>)[key];
  }
  return current;
}
