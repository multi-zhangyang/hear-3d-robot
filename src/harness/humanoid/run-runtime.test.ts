import { mkdtemp, rm, unlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { AgentManifest } from "../../domain/agent-manifest.js";
import type { HumanoidSkillInvocation } from "../../domain/humanoid-skill.js";
import { modelPayloadSha256 } from "../../domain/model-call-authority.js";
import { createScenarioBlockRemovalTransaction } from "../../domain/scenario-block-removal.js";
import { createScenarioChunkDeltaState } from "../../domain/scenario-chunk-delta-schema.js";
import {
  GoalSchema,
  ScenarioSchema,
  type Goal
} from "../../domain/schema.js";
import type { RuntimeEvent } from "../../runtime/events.js";
import { RunStore } from "../../persistence/run-store.js";
import { serializeHumanoidReference } from "../../world/humanoid/motion-artifact.js";
import type {
  HumanoidMotionGenerator,
  HumanoidMotionGeneratorInput
} from "../../world/humanoid/motion-plan.js";
import { HumanoidWorld } from "../../world/humanoid/world.js";
import { HUMANOID_AGENT_IDS } from "./agents.js";
import { createHumanoidRunCheckpoint } from "./run-checkpoint.js";
import { HumanoidRunRuntime } from "./run-runtime.js";
import { HumanoidActionRuntime, humanoidActionFingerprint } from "./runtime.js";

const scenario = ScenarioSchema.parse({
  title: "人形持久运行场",
  seed: 47,
  bounds: { width: 10, depth: 10 },
  visibility_radius: 6,
  robot: { x: 2, z: 2, yaw: 0 },
  obstacles: [{
    id: "lifecycle-block",
    center: { x: 8, y: 0.5, z: 8 },
    size: { x: 1, y: 1, z: 1 }
  }],
  objects: [],
  zones: [],
  default_goal: {
    summary: "保持站立",
    predicates: [{
      type: "robot_at",
      target: { x: 2, y: 0, z: 2 },
      tolerance: 0.4
    }]
  }
});

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((directory) => (
    rm(directory, { recursive: true, force: true })
  )));
});

describe("HumanoidRunRuntime", () => {
  it("keeps a model-selected exploration Goal active after the bootstrap Goal in continuous mode", async () => {
    const root = await mkdtemp(join(tmpdir(), "hear-continuous-goal-authority-"));
    temporaryDirectories.push(root);
    const store = await RunStore.create(root, {
      mission: "完成初始目标后继续自主探索",
      scenarioId: "humanoid-continuous-goal-authority-test",
      scenario,
      goal: scenario.default_goal,
      runtime: "humanoid_g1",
      runMode: "continuous"
    });
    const world = await HumanoidWorld.create(scenario);
    let runtime: HumanoidRunRuntime | undefined;
    try {
      const initial = createHumanoidRunCheckpoint({
        store,
        goal: scenario.default_goal,
        world
      });
      await store.writeCheckpoint(initial);
      runtime = new HumanoidRunRuntime({
        store,
        goal: scenario.default_goal,
        world,
        checkpoint: initial
      });
      const explorationGoal = GoalSchema.parse({
        summary: "探索新的物理位置",
        predicates: [{
          type: "robot_at",
          target: { x: 6, y: 0, z: 6 },
          tolerance: 0.3
        }]
      });
      await activateGoal(runtime, explorationGoal);
      await runtime.start(false);
      await runtime.stopContinuousPhysics();

      expect(runtime.coordinatorPhase()).toBe("observe_or_plan");
      expect(runtime.goalRetirementDelegationAvailable()).toBe(false);
      expect(runtime.checkpoint.goal_dag.status).toBe("active");
      expect(runtime.checkpoint.checker?.success).toBe(false);
    } finally {
      await runtime?.stopContinuousPhysics();
      await world.dispose();
    }
  }, 30_000);

  it("completes a newly selected Goal that the current physical state already satisfies", async () => {
    const root = await mkdtemp(join(tmpdir(), "hear-satisfied-goal-"));
    temporaryDirectories.push(root);
    const store = await RunStore.create(root, {
      mission: "验证已经由真实物理状态满足的目标",
      scenarioId: "humanoid-satisfied-goal-test",
      scenario,
      goal: scenario.default_goal,
      runtime: "humanoid_g1",
      runMode: "mission"
    });
    const world = await HumanoidWorld.create(scenario);
    let runtime: HumanoidRunRuntime | undefined;
    try {
      const initial = createHumanoidRunCheckpoint({
        store,
        goal: scenario.default_goal,
        world
      });
      await store.writeCheckpoint(initial);
      runtime = new HumanoidRunRuntime({
        store,
        goal: scenario.default_goal,
        world,
        checkpoint: initial
      });
      await activateGoal(runtime, scenario.default_goal);
      await runtime.start(false);
      await runtime.stopContinuousPhysics();

      expect(runtime.coordinatorPhase()).toBe("complete_satisfied_goal");
      expect(runtime.validateSatisfiedGoal()).toMatchObject({
        physical_execution_required: false,
        checker: { success: true }
      });
      await expect(runtime.completeSatisfiedGoal(JSON.stringify({
        status: "satisfied_goal_completed",
        summary: "当前 MuJoCo 状态已经满足选中的目标"
      }))).resolves.toBe(true);

      const checkpoint = runtime.checkpoint;
      expect(checkpoint.status).toBe("succeeded");
      expect(checkpoint.cycle_index).toBe(1);
      expect(checkpoint.active_cycle).toBeNull();
      expect(checkpoint.goal_dag.status).toBe("awaiting_model_selection");
      expect(checkpoint.goal_dag.epochs.at(-1)?.status).toBe("completed");
      expect(checkpoint.committed_actions).toEqual({});
      expect(await store.readJournal("episodes")).toEqual([]);
      expect(await store.readJournal("checker")).toEqual([
        expect.objectContaining({ success: true })
      ]);
    } finally {
      await runtime?.stopContinuousPhysics();
      await world.dispose();
    }
  }, 30_000);

  it("restores pending Skill authority and advances its DAG after restart", async () => {
    const root = await mkdtemp(join(tmpdir(), "hear-skill-restart-"));
    temporaryDirectories.push(root);
    const store = await RunStore.create(root, {
      mission: "恢复待执行的人形 Skill 计划",
      scenarioId: "humanoid-skill-restart",
      scenario,
      goal: scenario.default_goal,
      runtime: "humanoid_g1",
      runMode: "continuous"
    });
    const world = await HumanoidWorld.create(scenario);
    let tamperedWorld: HumanoidWorld | undefined;
    let resumedWorld: HumanoidWorld | undefined;
    try {
      const initial = createHumanoidRunCheckpoint({
        store,
        goal: scenario.default_goal,
        world
      });
      await store.writeCheckpoint(initial);
      const runtime = new HumanoidRunRuntime({
        store,
        goal: scenario.default_goal,
        world,
        checkpoint: initial
      });
      const manifest = await activateGoal(runtime, scenario.default_goal);
      await runtime.start(false);
      await runtime.stopContinuousPhysics();

      const target = {
        ...world.snapshot().robot.rootPosition,
        z: world.snapshot().robot.rootPosition.z + 0.35
      };
      const skillTransactionId = await bindPlanningSkill(
        runtime,
        "restart-retreat",
        {
          skill: "retreat",
          target,
          minimum_obstacle_clearance_m: 0.2
        },
        "route"
      );
      const planned = await invokeModelAction(
        runtime,
        "plan_humanoid_skill",
        { skill_transaction_id: skillTransactionId },
        "restart-retreat-navigation",
        HUMANOID_AGENT_IDS.motion
      );
      expect(planned.accepted, JSON.stringify(planned)).toBe(true);

      const persisted = await store.readHumanoidCheckpoint();
      expect(persisted.action_runtime_state).toMatchObject({
        version: 1,
        skill_plans: [{
          transaction_id: "restart-retreat-skill-plan",
          completed_node_ids: [],
          completed_phases_by_node: {}
        }],
        active_skills: [{ transaction_id: skillTransactionId }],
        planning_skill_bindings: [{
          planning_transaction_id: planned.transactionId,
          binding: { transaction_id: skillTransactionId }
        }]
      });

      tamperedWorld = await HumanoidWorld.create(
        scenario,
        persisted.world_checkpoint
      );
      const tamperedState = structuredClone(persisted.action_runtime_state) as {
        planning_skill_bindings: unknown[];
      };
      tamperedState.planning_skill_bindings = [];
      const ungrounded = new HumanoidActionRuntime(tamperedWorld, {
        receipts: persisted.committed_actions,
        state: tamperedState as never,
        requireSkillBinding: true
      });
      const beforeRejectedExecution = tamperedWorld.snapshot();
      await expect(ungrounded.invoke(
        "execute_humanoid_skill",
        { planning_transaction_id: planned.transactionId },
        "missing-restored-skill-execution",
        HUMANOID_AGENT_IDS.executor
      )).resolves.toMatchObject({
        accepted: false,
        code: "planning_skill_authority_missing",
        frameCount: 0
      });
      expect(tamperedWorld.snapshot()).toEqual(beforeRejectedExecution);
      await tamperedWorld.dispose();
      tamperedWorld = undefined;

      resumedWorld = await HumanoidWorld.create(
        scenario,
        persisted.world_checkpoint
      );
      const resumed = new HumanoidRunRuntime({
        store,
        goal: scenario.default_goal,
        world: resumedWorld,
        checkpoint: persisted
      });
      await resumed.initializeGoalAutonomy(manifest);
      const executed = await invokeModelAction(
        resumed,
        "execute_humanoid_skill",
        { planning_transaction_id: planned.transactionId },
        "restored-skill-execution",
        HUMANOID_AGENT_IDS.executor
      );
      expect(executed).toMatchObject({
        accepted: true,
        code: "navigation_completed",
        frameCount: expect.any(Number)
      });
      expect(executed.frameCount).toBeGreaterThan(0);

      const advanced = await store.readHumanoidCheckpoint();
      expect(advanced.action_runtime_state).toMatchObject({
        latest_physical_execution_revision: executed.worldAfterRevision,
        skill_plans: [{
          transaction_id: "restart-retreat-skill-plan",
          completed_node_ids: ["restart-retreat-skill-node"],
          completed_phases_by_node: {
            "restart-retreat-skill-node": ["route"]
          }
        }],
        active_skill_plan_transactions: {
          [HUMANOID_AGENT_IDS.motion]: "restart-retreat-skill-plan"
        },
        active_skills: [],
        planning_skill_bindings: []
      });
    } finally {
      await tamperedWorld?.dispose();
      await resumedWorld?.dispose();
      await world.dispose();
    }
  }, 45_000);

  it("persists lifecycle, physical frames, receipts and restart idempotency", async () => {
    const root = await mkdtemp(join(tmpdir(), "hear-humanoid-run-"));
    temporaryDirectories.push(root);
    const store = await RunStore.create(root, {
      mission: "自主保持平衡并改变双臂姿态",
      scenarioId: "humanoid-runtime-test",
      scenario,
      goal: scenario.default_goal,
      runtime: "humanoid_g1",
      runMode: "continuous"
    });
    const events: RuntimeEvent[] = [];
    const world = await HumanoidWorld.create(scenario);
    let resumedWorld: HumanoidWorld | undefined;
    try {
      const initial = createHumanoidRunCheckpoint({
        store,
        goal: scenario.default_goal,
        world
      });
      initial.capability_catalog = initial.capability_catalog.filter((capability) => (
        capability !== "plan_whole_body_motion_candidates"
      ));
      initial.nodes[HUMANOID_AGENT_IDS.motion]!.capabilities = [
        "observe_humanoid",
        "plan_whole_body_motion",
        "plan_humanoid_navigation"
      ];
      await store.writeCheckpoint(initial);
      const runtime = new HumanoidRunRuntime({
        store,
        goal: scenario.default_goal,
        world,
        checkpoint: initial,
        eventSink: (event) => {
          events.push(event);
        }
      });
      const goalManifest = await activateGoal(runtime, scenario.default_goal);

      await runtime.start(false);
      await new Promise((resolve) => setTimeout(resolve, 30));
      await runtime.recordProvider({
        status: "usable_stream",
        usage: {
          inputTokens: 480,
          outputTokens: 120,
          totalTokens: 600,
          inputTokensDetails: { cached_tokens: 320 },
          outputTokensDetails: { reasoning_tokens: 40 }
        }
      }, HUMANOID_AGENT_IDS.coordinator);
      expect(runtime.checkpoint.model_usage).toMatchObject({
        total: {
          requests: 1,
          reported_requests: 1,
          input_tokens: 480,
          output_tokens: 120,
          total_tokens: 600,
          cached_input_tokens: 320,
          reasoning_tokens: 40
        },
        by_agent: {
          [HUMANOID_AGENT_IDS.coordinator]: { requests: 1, total_tokens: 600 }
        }
      });
      expect((await store.readJournalTail("provider", 1)).entries[0]).toMatchObject({
        agent_id: HUMANOID_AGENT_IDS.coordinator,
        model_usage: {
          total: { requests: 1, total_tokens: 600 }
        }
      });
      await runtime.stopContinuousPhysics();
      await runtime.setActiveAgent(HUMANOID_AGENT_IDS.sentry);
      const meteredCheckpoint = await store.readHumanoidCheckpoint();
      expect(meteredCheckpoint.model_usage.total).toMatchObject({
        requests: 1,
        total_tokens: 600
      });
      expect(meteredCheckpoint.world.frame).toBe(
        meteredCheckpoint.world_checkpoint.frame
      );
      const stationaryFrameCount = events.filter(
        (event) => event.type === "humanoid_world_frame"
      ).length;
      const observation = await invokeModelAction(runtime,
        "observe_humanoid",
        {},
        "observe-persisted",
        HUMANOID_AGENT_IDS.sentry
      );
      const planInput = {
        id: "persistent-arm-motion",
        intent: "保持双脚支撑并连续改变双臂姿态",
        duration_seconds: 0.6,
        keyframes: [
          { at_seconds: 0 },
          {
            at_seconds: 0.6,
            torso_yaw: 0.04,
            left_hand: {
              frame: "world" as const,
              position: {
                ...world.snapshot().robot.links.left_wrist_yaw_link.position,
                y: world.snapshot().robot.links.left_wrist_yaw_link.position.y + 0.01
              },
              tolerance_m: 0.045
            },
            right_hand: {
              frame: "world" as const,
              position: {
                ...world.snapshot().robot.links.right_wrist_yaw_link.position,
                y: world.snapshot().robot.links.right_wrist_yaw_link.position.y + 0.01
              },
              tolerance_m: 0.045
            }
          }
        ]
      };
      const plan = await invokeModelAction(runtime,
        "plan_whole_body_motion",
        planInput,
        "plan-persisted",
        HUMANOID_AGENT_IDS.motion
      );
      expect(plan.accepted, JSON.stringify(plan)).toBe(true);
      expect(runtime.contextAnchor(HUMANOID_AGENT_IDS.coordinator)).toMatchObject({
        coordinator_phase: "execute_plan",
        robot: {
          root_heading: {
            yaw_radians: expect.any(Number),
            forward_world: {
              x: expect.any(Number),
              y: 0,
              z: expect.any(Number)
            },
            left_world: {
              x: expect.any(Number),
              y: 0,
              z: expect.any(Number)
            }
          }
        },
        execution_authority: {
          task: "execute_plan",
          planning_action: "plan_whole_body_motion",
          planning_transaction_id: plan.transactionId,
          executor_action: "execute_whole_body_motion",
          accepted_world_revision: plan.worldAfterRevision,
          remaining_lease_revisions: expect.any(Number)
        }
      });
      const executionInput = { planning_transaction_id: plan.transactionId };
      const execution = await invokeModelAction(runtime,
        "execute_whole_body_motion",
        executionInput,
        "execute-persisted",
        HUMANOID_AGENT_IDS.executor
      );
      expect(execution.accepted).toBe(true);
      expect(execution.frameCount).toBeGreaterThan(10);
      expect(execution.detail).toMatchObject({
        physical_trajectory: {
          version: 1,
          complete_from_admission: true,
          observed_frame_count: execution.frameCount + 1,
          start_world_revision: execution.worldBeforeRevision,
          end_world_revision: execution.worldAfterRevision,
          joint_names: expect.arrayContaining([
            "left_hip_pitch_joint",
            "right_hand_index_1_joint"
          ]),
          trajectory_sha256: expect.stringMatching(/^[a-f0-9]{64}$/)
        }
      });

      expect(() => runtime.validateCycleEvidence([execution.transactionId])).toThrow(
        "requires accepted physical execution evidence"
      );

      const pendingBefore = world.snapshot();
      const pendingTarget = {
        ...pendingBefore.robot.rootPosition,
        z: pendingBefore.robot.rootPosition.z + 0.25
      };
      const pendingSkillTransactionId = await bindPlanningSkill(
        runtime,
        "plan-unconsumed",
        {
          skill: "retreat",
          target: pendingTarget,
          minimum_obstacle_clearance_m: 0.1
        },
        "route"
      );
      const pendingPlan = await invokeModelAction(runtime,
        "plan_humanoid_skill",
        { skill_transaction_id: pendingSkillTransactionId },
        "plan-unconsumed",
        HUMANOID_AGENT_IDS.motion
      );
      expect(pendingPlan.accepted, JSON.stringify(pendingPlan)).toBe(true);
      expect(pendingPlan.code).toBe("autonomous_skill_route_validated");
      const pendingExecution = await invokeModelAction(runtime,
        "execute_humanoid_skill",
        { planning_transaction_id: pendingPlan.transactionId },
        "execute-unconsumed",
        HUMANOID_AGENT_IDS.executor
      );
      expect(pendingExecution.accepted).toBe(true);
      expect(() => runtime.validateCycleEvidence([
        pendingExecution.transactionId
      ])).toThrow("requires an accepted Sentry observation");
      await invokeModelAction(runtime,
        "observe_humanoid",
        {},
        "observe-after-unconsumed-execution",
        HUMANOID_AGENT_IDS.sentry
      );
      expect(runtime.validateCycleEvidence([pendingExecution.transactionId])).toMatchObject({
        transactionId: pendingExecution.transactionId,
        code: "navigation_completed"
      });
      const removalInput = {
        solid_id: "lifecycle-block",
        execution_transaction_id: pendingExecution.transactionId
      };
      const removalTransaction = createScenarioBlockRemovalTransaction({
        scenario,
        chunks: createScenarioChunkDeltaState(scenario),
        transactionId: "remove-lifecycle-block",
        solidId: "lifecycle-block",
        executionTransactionId: pendingExecution.transactionId,
        planningTransactionId: pendingPlan.transactionId,
        sourceWorldFrame: world.snapshot().frame,
        sourceWorldRevision: pendingExecution.worldAfterRevision,
        contactEvidence: {
          predicate_index: 0,
          predicate_type: "body_contact_solid",
          surface_kind: "body",
          surface: "left_wrist_yaw_link",
          planned_stable_frames: 8,
          observed_stable_frames: 8,
          planned_minimum_normal_force_n: 5,
          observed_maximum_normal_force_n: 5
        }
      });
      const lifecycleCheckpoint = runtime.checkpoint;
      lifecycleCheckpoint.committed_actions[removalTransaction.transaction_id] = {
        transactionId: removalTransaction.transaction_id,
        agentId: pendingExecution.agentId,
        decision: {
          ...pendingExecution.decision!,
          tool_call_id: removalTransaction.transaction_id,
          tool_arguments_sha256: modelPayloadSha256(removalInput)
        },
        cycle: pendingExecution.cycle,
        action: "remove_world_block",
        input: removalInput,
        fingerprint: humanoidActionFingerprint(
          "remove_world_block",
          pendingExecution.agentId,
          removalInput
        ),
        accepted: true,
        code: "world_block_removal_authorized",
        worldBeforeRevision: world.snapshot().worldRevision,
        worldAfterRevision: world.snapshot().worldRevision,
        frameCount: 0,
        channels: [],
        detail: {
          solid_id: "lifecycle-block",
          execution_transaction_id: pendingExecution.transactionId,
          automatic_actuation: false,
          removal_transaction: removalTransaction
        },
        committedAt: new Date().toISOString()
      };
      const lifecycleRuntime = new HumanoidRunRuntime({
        store,
        goal: scenario.default_goal,
        world,
        checkpoint: lifecycleCheckpoint
      });
      expect(() => lifecycleRuntime.validateCycleEvidence([
        pendingExecution.transactionId
      ])).toThrow("omits world mutation evidence");
      expect(() => lifecycleRuntime.validateCycleEvidence([
        pendingExecution.transactionId,
        removalTransaction.transaction_id
      ])).toThrow("requires an accepted Sentry observation");
      lifecycleCheckpoint.committed_actions["observe-after-lifecycle-removal"] = {
        transactionId: "observe-after-lifecycle-removal",
        agentId: HUMANOID_AGENT_IDS.sentry,
        cycle: pendingExecution.cycle,
        action: "observe_humanoid",
        input: {},
        fingerprint: humanoidActionFingerprint(
          "observe_humanoid",
          HUMANOID_AGENT_IDS.sentry,
          {}
        ),
        accepted: true,
        code: "humanoid_observed",
        worldBeforeRevision: world.snapshot().worldRevision,
        worldAfterRevision: world.snapshot().worldRevision,
        frameCount: 0,
        channels: [],
        detail: {},
        committedAt: new Date().toISOString()
      };
      const observedLifecycleRuntime = new HumanoidRunRuntime({
        store,
        goal: scenario.default_goal,
        world,
        checkpoint: lifecycleCheckpoint
      });
      expect(observedLifecycleRuntime.validateCycleEvidence([
        pendingExecution.transactionId,
        removalTransaction.transaction_id
      ])).toMatchObject({ transactionId: pendingExecution.transactionId });
      await runtime.completeCycle(JSON.stringify({
        status: "cycle_completed",
        evidence_transaction_ids: [pendingExecution.transactionId]
      }));
      const anchor = runtime.contextAnchor(HUMANOID_AGENT_IDS.coordinator) as {
        recent_physical_episodes: Array<{
          transaction_id: string;
          historical_only: boolean;
          model_narrative_omitted: boolean;
          model_summary?: string;
        }>;
      };
      expect(anchor.recent_physical_episodes).toEqual([
        expect.objectContaining({
          transaction_id: pendingExecution.transactionId,
          historical_only: true,
          model_narrative_omitted: true
        })
      ]);
      expect(anchor.recent_physical_episodes[0]).not.toHaveProperty("model_summary");
      expect(() => runtime.validateCycleEvidence([pendingExecution.transactionId])).toThrow(
        "already consumed"
      );
      await activateGoal(runtime, scenario.default_goal);
      expect(await runtime.recallGoalHistory({
        statuses: ["completed"],
        limit: 4
      })).toMatchObject({
        historical_only: true,
        total_matches: 1,
        candidates: [{ status: "completed", goal: scenario.default_goal }]
      });
      const rejectedExecution = await invokeModelAction(runtime,
        "execute_whole_body_motion",
        { planning_transaction_id: "missing-historical-plan" },
        "execute-rejected-history",
        HUMANOID_AGENT_IDS.executor
      );
      expect(rejectedExecution).toMatchObject({
        accepted: false,
        code: "planning_receipt_missing",
        frameCount: 0
      });
      const beforeRecall = runtime.snapshot();
      const explicitHistory = await runtime.recallEmbodiedHistory({
        source_refs: [
          "episode:1",
          `action:${pendingExecution.transactionId}`,
          `action:${rejectedExecution.transactionId}`
        ],
        before_sequence: 2,
        limit: 3
      });
      expect(explicitHistory).toMatchObject({
        historical_only: true,
        episodes: [{
          source_ref: "episode:1",
          sequence: 1,
          transaction_id: pendingExecution.transactionId
        }],
        actions: expect.arrayContaining([
          expect.objectContaining({
            source_ref: `action:${pendingExecution.transactionId}`,
            historical_only: true,
            transactionId: pendingExecution.transactionId,
            accepted: true,
            code: "navigation_completed",
            frameCount: pendingExecution.frameCount,
            worldBeforeRevision: pendingExecution.worldBeforeRevision,
            worldAfterRevision: pendingExecution.worldAfterRevision,
            detail: expect.objectContaining({ result: expect.any(Object) })
          }),
          expect.objectContaining({
            source_ref: `action:${rejectedExecution.transactionId}`,
            historical_only: true,
            transactionId: rejectedExecution.transactionId,
            accepted: false,
            code: "planning_receipt_missing",
            frameCount: 0,
            worldBeforeRevision: rejectedExecution.worldBeforeRevision,
            worldAfterRevision: rejectedExecution.worldAfterRevision
          })
        ]),
        missing_source_refs: []
      });
      expect(JSON.stringify(explicitHistory)).not.toContain("physical_trajectory");
      expect(JSON.stringify(explicitHistory).length).toBeLessThan(12_000);
      expect(await runtime.recallEmbodiedHistory({ limit: 3 })).toMatchObject({
        historical_only: true,
        ordered_source_refs: expect.arrayContaining([
          "episode:1",
          `action:${pendingExecution.transactionId}`,
          `action:${rejectedExecution.transactionId}`
        ]),
        episodes: [expect.objectContaining({ source_ref: "episode:1" })],
        actions: expect.arrayContaining([
          expect.objectContaining({
            source_ref: `action:${rejectedExecution.transactionId}`,
            accepted: false,
            code: "planning_receipt_missing"
          })
        ])
      });
      expect(await runtime.recallEmbodiedHistory({
        outcomes: ["rejected"],
        predicate_types: ["robot_at"],
        limit: 3
      })).toMatchObject({
        historical_only: true,
        semantic_query: {
          outcomes: ["rejected"],
          predicate_types: ["robot_at"]
        },
        ordered_source_refs: [`action:${rejectedExecution.transactionId}`],
        experiences: [expect.objectContaining({
          source_ref: `action:${rejectedExecution.transactionId}`,
          outcome: "rejected",
          historical_only: true
        })],
        actions: [expect.objectContaining({
          transactionId: rejectedExecution.transactionId,
          code: "planning_receipt_missing",
          historical_only: true
        })]
      });
      const olderHistory = await runtime.recallEmbodiedHistory({
        before_sequence: 1,
        limit: 3
      }) as {
        ordered_source_refs: string[];
      };
      expect(olderHistory.ordered_source_refs).not.toContain("episode:1");
      expect(olderHistory.ordered_source_refs).not.toContain(
        `action:${pendingExecution.transactionId}`
      );
      expect(olderHistory.ordered_source_refs).not.toContain(
        `action:${rejectedExecution.transactionId}`
      );
      expect(runtime.snapshot()).toEqual(beforeRecall);

      const persisted = await store.readHumanoidCheckpoint();
      expect(persisted.status).toBe("running");
      expect(persisted.pending_lifecycle_events).toEqual([]);
      expect(persisted.capability_catalog).toContain("plan_whole_body_motion_candidates");
      expect(persisted.nodes[HUMANOID_AGENT_IDS.motion]!.capabilities).toEqual([
        "observe_humanoid",
        "recall_embodied_history",
        "submit_humanoid_skill_plan",
        "begin_humanoid_skill",
        "plan_whole_body_motion_candidates",
        "plan_humanoid_navigation"
      ]);
      expect(persisted.world.frame).toBe(world.snapshot().frame);
      expect(persisted.world_checkpoint.frame).toBe(persisted.world.frame);
      expect(persisted.world_checkpoint.worldRevision).toBe(persisted.world.worldRevision);
      expect(persisted.committed_actions[observation.transactionId]).toEqual(observation);
      expect(persisted.committed_actions[execution.transactionId]).toEqual(execution);
      expect(persisted.embodied_memory).toMatchObject({
        version: 2,
        total_episodes: 1,
        pruned_episodes: 0,
        total_experiences: 3,
        pruned_experiences: 0,
        outcome_counts: {
          succeeded: 2,
          rejected: 1,
          physically_failed: 0
        },
        recent_experiences: expect.arrayContaining([
          expect.objectContaining({
            source_ref: `action:${pendingExecution.transactionId}`,
            outcome: "succeeded",
            goal_content_sha256: expect.stringMatching(/^[a-f0-9]{64}$/)
          }),
          expect.objectContaining({
            source_ref: `action:${rejectedExecution.transactionId}`,
            outcome: "rejected"
          })
        ]),
        recent_episodes: [{
          sequence: 1,
          causal_trace: {
            cycle: pendingExecution.cycle,
            planning_transaction_id: pendingPlan.transactionId,
            execution_transaction_id: pendingExecution.transactionId,
            execution_decision: pendingExecution.decision,
            goal_evidence_refs: expect.arrayContaining([
              `action:${pendingExecution.transactionId}`
            ]),
            memory_id: expect.stringMatching(/^embodied-memory:/)
          },
          transaction_id: pendingExecution.transactionId,
          action: "execute_humanoid_skill",
          planning_action: "plan_humanoid_skill",
          code: "navigation_completed",
          world_after_revision: pendingExecution.worldAfterRevision,
          goal_success: true
        }]
      });
      expect(await store.readJournal("experiences")).toEqual([
        expect.objectContaining({
          source_ref: `action:${execution.transactionId}`,
          outcome: "succeeded"
        }),
        expect.objectContaining({
          source_ref: `action:${pendingExecution.transactionId}`,
          outcome: "succeeded"
        }),
        expect.objectContaining({
          source_ref: `action:${rejectedExecution.transactionId}`,
          outcome: "rejected"
        })
      ]);
      expect(pendingPlan.cycle).toEqual(pendingExecution.cycle);
      const completedModelCalls = (await store.readJournal("model_calls")).filter(
        (record) => journalField(record, "lifecycle") === "completed"
      );
      expect(completedModelCalls).toContainEqual(expect.objectContaining({
        agent_id: HUMANOID_AGENT_IDS.executor,
        cycle: pendingExecution.cycle,
        tool_calls: expect.arrayContaining([expect.objectContaining({
          tool_call_id: pendingExecution.transactionId,
          tool_name: pendingExecution.action
        })])
      }));
      const memoryEvent = events.find((event) => event.type === "embodied_episode_recorded");
      expect(memoryEvent).toBeDefined();
      expect((memoryEvent!.data as {
        embodied_memory: { total_experiences: number };
      }).embodied_memory.total_experiences).toBe(2);
      expect(memoryEvent!.cursor).toBeTruthy();
      const rejectedMemoryEvent = events.find((event) => {
        if (event.type !== "humanoid_action_committed") return false;
        const data = event.data as {
          experience?: { transaction_id?: string; outcome?: string };
        };
        return data.experience?.transaction_id === rejectedExecution.transactionId;
      });
      expect(rejectedMemoryEvent?.data).toMatchObject({
        experience: {
          source_ref: `action:${rejectedExecution.transactionId}`,
          outcome: "rejected"
        },
        embodied_memory: {
          total_experiences: 3
        }
      });
      expect(events.some((event) => event.type === "run_started" && event.cursor)).toBe(true);
      expect(events.filter((event) => event.type === "humanoid_world_frame")).toHaveLength(
        stationaryFrameCount + execution.frameCount + pendingExecution.frameCount
      );
      expect(events.filter((event) => event.type === "humanoid_world_frame")
        .every((event) => event.durable === false && event.cursor === undefined)).toBe(true);

      resumedWorld = await HumanoidWorld.create(scenario, persisted.world_checkpoint);
      const resumed = new HumanoidRunRuntime({
        store,
        goal: scenario.default_goal,
        world: resumedWorld,
        checkpoint: persisted
      });
      await resumed.initializeGoalAutonomy(goalManifest);
      const beforeReplay = resumed.snapshot();
      const replayed = await invokeModelAction(resumed,
        "execute_whole_body_motion",
        executionInput,
        "execute-persisted",
        HUMANOID_AGENT_IDS.executor
      );
      expect(replayed).toEqual(execution);
      expect(resumed.snapshot()).toEqual(beforeReplay);
      await expect(invokeModelAction(resumed,
        "execute_whole_body_motion",
        { planning_transaction_id: "different-plan" },
        "execute-persisted",
        HUMANOID_AGENT_IDS.executor
      )).rejects.toThrow("transaction conflict");

      const corrupted = structuredClone(persisted);
      corrupted.committed_actions[execution.transactionId]!.fingerprint = "corrupted";
      expect(() => new HumanoidRunRuntime({
        store,
        goal: scenario.default_goal,
        world: resumedWorld!,
        checkpoint: corrupted
      })).toThrow("fingerprint mismatch");
    } finally {
      await resumedWorld?.dispose();
      await world.dispose();
    }
  }, 45_000);

  it("recovers newer cumulative model usage from the provider journal", async () => {
    const root = await mkdtemp(join(tmpdir(), "hear-model-usage-recovery-"));
    temporaryDirectories.push(root);
    const store = await RunStore.create(root, {
      mission: "恢复尚未进入检查点的模型用量",
      scenarioId: "humanoid-model-usage-recovery",
      scenario,
      goal: scenario.default_goal,
      runtime: "humanoid_g1",
      runMode: "continuous"
    });
    const world = await HumanoidWorld.create(scenario);
    try {
      const initial = createHumanoidRunCheckpoint({
        store,
        goal: scenario.default_goal,
        world
      });
      await store.writeCheckpoint(initial);
      const manifest = humanoidTestManifest();
      const runtime = new HumanoidRunRuntime({
        store,
        goal: scenario.default_goal,
        world,
        checkpoint: initial
      });
      await runtime.initializeGoalAutonomy(manifest);
      await runtime.recordProvider({
        status: "usable_stream",
        usage: {
          inputTokens: 900,
          outputTokens: 100,
          totalTokens: 1_000
        }
      }, HUMANOID_AGENT_IDS.motion);
      await store.appendMany("provider", Array.from({ length: 300 }, (_, index) => ({
        status: "non_usage_telemetry",
        sequence: index
      })));

      expect((await store.readHumanoidCheckpoint()).model_usage.total.requests).toBe(0);

      const recovered = new HumanoidRunRuntime({
        store,
        goal: scenario.default_goal,
        world,
        checkpoint: await store.readHumanoidCheckpoint()
      });
      await recovered.initializeGoalAutonomy(manifest);
      expect(recovered.checkpoint.model_usage).toMatchObject({
        total: {
          requests: 1,
          input_tokens: 900,
          output_tokens: 100,
          total_tokens: 1_000
        },
        by_agent: {
          [HUMANOID_AGENT_IDS.motion]: { requests: 1, total_tokens: 1_000 }
        }
      });

      await recovered.setActiveAgent(HUMANOID_AGENT_IDS.motion);
      expect((await store.readHumanoidCheckpoint()).model_usage.total).toMatchObject({
        requests: 1,
        total_tokens: 1_000
      });
    } finally {
      await world.dispose();
    }
  });

  it("reconciles a staged action commit without repeating the action", async () => {
    const root = await mkdtemp(join(tmpdir(), "hear-action-commit-recovery-"));
    temporaryDirectories.push(root);
    const store = await RunStore.create(root, {
      mission: "验证动作提交恢复",
      scenarioId: "humanoid-action-commit-recovery",
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
      const runtime = new HumanoidRunRuntime({
        store,
        goal: scenario.default_goal,
        world,
        checkpoint: initial
      });
      const goalManifest = await activateGoal(runtime, scenario.default_goal);
      await runtime.start(false);
      await runtime.stopContinuousPhysics();

      await authorizeModelAction(
        runtime,
        "observe_humanoid",
        {},
        "recover-action-commit",
        HUMANOID_AGENT_IDS.sentry
      );
      vi.spyOn(store, "appendRuntimeEvents")
        .mockRejectedValueOnce(new Error("event journal unavailable"));
      await expect(invokeModelAction(runtime,
        "observe_humanoid",
        {},
        "recover-action-commit",
        HUMANOID_AGENT_IDS.sentry
      )).rejects.toThrow("event journal unavailable");

      const staged = await store.readHumanoidCheckpoint();
      expect(Object.keys(staged.action_commit_outbox.pending)).toEqual([
        "recover-action-commit"
      ]);
      await expect(invokeModelAction(runtime,
        "observe_humanoid",
        {},
        "blocked-until-reconciled",
        HUMANOID_AGENT_IDS.sentry
      )).rejects.toThrow("retry transaction recover-action-commit");

      resumedWorld = await HumanoidWorld.create(scenario, staged.world_checkpoint);
      const resumed = new HumanoidRunRuntime({
        store,
        goal: scenario.default_goal,
        world: resumedWorld,
        checkpoint: staged
      });
      await resumed.initializeGoalAutonomy(goalManifest);
      const recovered = await invokeModelAction(resumed,
        "observe_humanoid",
        {},
        "recover-action-commit",
        HUMANOID_AGENT_IDS.sentry
      );
      expect(recovered.transactionId).toBe("recover-action-commit");
      const checkpoint = await store.readHumanoidCheckpoint();
      expect(checkpoint.action_commit_outbox.pending).toEqual({});

      const actions = await store.readJournal("actions");
      expect(actions.filter((entry) => journalField(entry, "transactionId")
        === recovered.transactionId)).toHaveLength(1);
      const evidence = await store.readJournal("goal_evidence");
      expect(evidence.filter((entry) => journalNestedField(
        entry,
        "evidence",
        "ref"
      ) === `action:${recovered.transactionId}`)).toHaveLength(1);
      const events = await store.readJournal("events");
      expect(events.filter((entry) => (
        journalField(entry, "type") === "humanoid_action_committed"
          && journalNestedField(entry, "data", "receipt", "transactionId")
            === recovered.transactionId
      ))).toHaveLength(1);
    } finally {
      vi.restoreAllMocks();
      await resumedWorld?.dispose();
      await world.dispose();
    }
  }, 30_000);

  it("persists physical admission before actuation and tolerates live frame sink failure", async () => {
    const root = await mkdtemp(join(tmpdir(), "hear-physical-admission-"));
    temporaryDirectories.push(root);
    const store = await RunStore.create(root, {
      mission: "验证首帧前动作意图持久化",
      scenarioId: "humanoid-physical-admission",
      scenario,
      goal: scenario.default_goal,
      runtime: "humanoid_g1"
    });
    const world = await HumanoidWorld.create(scenario);
    try {
      const initial = createHumanoidRunCheckpoint({
        store,
        goal: scenario.default_goal,
        world
      });
      await store.writeCheckpoint(initial);
      const runtime = new HumanoidRunRuntime({
        store,
        goal: scenario.default_goal,
        world,
        checkpoint: initial,
        eventSink: (event) => {
          if (event.type === "humanoid_world_frame") {
            throw new Error("live frame consumer unavailable");
          }
        }
      });
      await activateGoal(runtime, scenario.default_goal);
      await runtime.start(false);
      await runtime.stopContinuousPhysics();
      const planned = await invokeModelAction(runtime,
        "plan_whole_body_motion",
        {
          id: "admission-before-frame",
          intent: "保持站立并完成短姿态调整",
          duration_seconds: 0.1,
          keyframes: [{ at_seconds: 0 }, { at_seconds: 0.1, torso_yaw: 0.01 }]
        },
        "admission-plan",
        HUMANOID_AGENT_IDS.motion
      );
      expect(planned.accepted).toBe(true);
      const before = world.snapshot();
      const executionInput = { planning_transaction_id: planned.transactionId };
      await authorizeModelAction(
        runtime,
        "execute_whole_body_motion",
        executionInput,
        "admission-execute",
        HUMANOID_AGENT_IDS.executor
      );
      vi.spyOn(store, "writeCheckpoint")
        .mockRejectedValueOnce(new Error("admission checkpoint unavailable"));
      await expect(invokeModelAction(runtime,
        "execute_whole_body_motion",
        executionInput,
        "admission-execute",
        HUMANOID_AGENT_IDS.executor
      )).rejects.toThrow("admission checkpoint unavailable");
      expect(world.snapshot().worldRevision).toBe(before.worldRevision);
      expect(world.checkpoint().motions[0]!.progress.nextFrameIndex).toBe(0);

      const executed = await invokeModelAction(runtime,
        "execute_whole_body_motion",
        executionInput,
        "admission-execute",
        HUMANOID_AGENT_IDS.executor
      );
      expect(executed).toMatchObject({ accepted: true, code: "motion_completed" });
      expect(executed.frameCount).toBeGreaterThan(0);
      const persisted = await store.readHumanoidCheckpoint();
      expect(persisted.action_execution_ledger.active).toEqual({});
      expect(persisted.action_commit_outbox.pending).toEqual({});
      expect(persisted.world_checkpoint.motions).toEqual([]);
    } finally {
      vi.restoreAllMocks();
      await world.dispose();
    }
  }, 30_000);

  it("stops an SDK-cancelled MuJoCo command and resumes only its durable transaction", async () => {
    const root = await mkdtemp(join(tmpdir(), "hear-tool-signal-recovery-"));
    temporaryDirectories.push(root);
    const store = await RunStore.create(root, {
      mission: "验证工具取消后的同事务物理恢复",
      scenarioId: "humanoid-tool-signal-recovery",
      scenario,
      goal: scenario.default_goal,
      runtime: "humanoid_g1"
    });
    const world = await HumanoidWorld.create(scenario);
    const toolCall = new AbortController();
    let executionFrames = 0;
    let actuationStarted = false;
    let runtime: HumanoidRunRuntime | undefined;
    try {
      const initial = createHumanoidRunCheckpoint({
        store,
        goal: scenario.default_goal,
        world
      });
      await store.writeCheckpoint(initial);
      runtime = new HumanoidRunRuntime({
        store,
        goal: scenario.default_goal,
        world,
        checkpoint: initial,
        eventSink: (event) => {
          if (!actuationStarted || event.type !== "humanoid_world_frame") return;
          executionFrames += 1;
          if (executionFrames === 3) {
            toolCall.abort(new Error("SDK tool branch cancelled"));
          }
        }
      });
      await activateGoal(runtime, scenario.default_goal);
      await runtime.start(false);
      await runtime.stopContinuousPhysics();
      const planned = await invokeModelAction(runtime,
        "plan_whole_body_motion",
        {
          id: "tool-signal-motion",
          intent: "执行一段可在取消后续行的全身保持动作",
          duration_seconds: 0.3,
          keyframes: [{ at_seconds: 0 }, { at_seconds: 0.3 }]
        },
        "tool-signal-plan",
        HUMANOID_AGENT_IDS.motion
      );
      const executionInput = { planning_transaction_id: planned.transactionId };
      const transactionId = "tool-signal-execute";
      const authority = await authorizeModelAction(
        runtime,
        "execute_whole_body_motion",
        executionInput,
        transactionId,
        HUMANOID_AGENT_IDS.executor
      );

      actuationStarted = true;
      await expect(runtime.invoke(
        "execute_whole_body_motion",
        executionInput,
        transactionId,
        HUMANOID_AGENT_IDS.executor,
        authority,
        { signal: toolCall.signal }
      )).rejects.toThrow("SDK tool branch cancelled");
      expect(executionFrames).toBe(3);
      const stoppedRevision = world.snapshot().worldRevision;
      await new Promise((resolvePromise) => setTimeout(resolvePromise, 80));
      expect(world.snapshot().worldRevision).toBe(stoppedRevision);
      expect(runtime.pendingPhysicalExecutionTransactionId()).toBe(transactionId);
      expect(runtime.receipt(transactionId)).toBeUndefined();

      const unrelatedAuthority = await authorizeModelAction(
        runtime,
        "observe_humanoid",
        {},
        "tool-signal-unrelated",
        HUMANOID_AGENT_IDS.sentry
      );
      await expect(runtime.invoke(
        "observe_humanoid",
        {},
        "tool-signal-unrelated",
        HUMANOID_AGENT_IDS.sentry,
        unrelatedAuthority
      )).rejects.toThrow(
        `Physical execution ${transactionId} must be recovered before tool-signal-unrelated`
      );

      const recovered = await runtime.recoverPendingPhysicalExecution();
      expect(recovered).toMatchObject({
        transactionId,
        accepted: true,
        code: "motion_completed"
      });
      expect(runtime.pendingPhysicalExecutionTransactionId()).toBeUndefined();
      const matchingReceipts = (await store.readJournal("actions")).filter((entry) => (
        journalField(entry, "transactionId") === transactionId
      ));
      expect(matchingReceipts).toHaveLength(1);
    } finally {
      await runtime?.stopContinuousPhysics();
      await world.dispose();
    }
  }, 45_000);

  it.each([1, 9])(
    "automatically recovers on startup from process loss after physical frame %i",
    async (crashFrame) => {
      const root = await mkdtemp(join(tmpdir(), `hear-frame-${crashFrame}-recovery-`));
      temporaryDirectories.push(root);
      const store = await RunStore.create(root, {
        mission: "验证未落盘物理尾部的执行游标恢复",
        scenarioId: `humanoid-frame-${crashFrame}-recovery`,
        scenario,
        goal: scenario.default_goal,
        runtime: "humanoid_g1"
      });
      const world = await HumanoidWorld.create(scenario);
      let resumedWorld: HumanoidWorld | undefined;
      const controller = new AbortController();
      let physicalFrames = 0;
      let actuationStarted = false;
      try {
        const initial = createHumanoidRunCheckpoint({
          store,
          goal: scenario.default_goal,
          world
        });
        await store.writeCheckpoint(initial);
        const runtime = new HumanoidRunRuntime({
          store,
          goal: scenario.default_goal,
          world,
          checkpoint: initial,
          signal: controller.signal,
          eventSink: (event) => {
            if (!actuationStarted || event.type !== "humanoid_world_frame") return;
            physicalFrames += 1;
            if (physicalFrames === crashFrame) {
              controller.abort(new Error(`simulated process loss at frame ${crashFrame}`));
            }
          }
        });
        const manifest = await activateGoal(runtime, scenario.default_goal);
        await runtime.start(false);
        await runtime.stopContinuousPhysics();
        const planned = await invokeModelAction(runtime,
          "plan_whole_body_motion",
          {
            id: `frame-${crashFrame}-recovery-motion`,
            intent: "执行一段超过十帧的连续全身保持动作",
            duration_seconds: 0.3,
            keyframes: [{ at_seconds: 0 }, { at_seconds: 0.3 }]
          },
          `frame-${crashFrame}-plan`,
          HUMANOID_AGENT_IDS.motion
        );
        actuationStarted = true;
        await expect(invokeModelAction(runtime,
          "execute_whole_body_motion",
          { planning_transaction_id: planned.transactionId },
          `frame-${crashFrame}-execute`,
          HUMANOID_AGENT_IDS.executor
        )).rejects.toThrow(`simulated process loss at frame ${crashFrame}`);
        expect(physicalFrames).toBe(crashFrame);

        const crashCheckpoint = await store.readHumanoidCheckpoint();
        const execution = crashCheckpoint.action_execution_ledger.active[
          `frame-${crashFrame}-execute`
        ]!;
        const expectedFrames = crashCheckpoint.world_checkpoint.motions[0]!
          .artifact.frames.length;
        expect(execution).toMatchObject({
          status: "admitted",
          progress: { committed_frame_count: 0 }
        });
        expect(crashCheckpoint.world_checkpoint.motions[0]!.progress.nextFrameIndex).toBe(0);

        resumedWorld = await HumanoidWorld.create(
          scenario,
          crashCheckpoint.world_checkpoint
        );
        const resumed = new HumanoidRunRuntime({
          store,
          goal: scenario.default_goal,
          world: resumedWorld,
          checkpoint: crashCheckpoint
        });
        await resumed.initializeGoalAutonomy(manifest);
        await resumed.start(true);
        await resumed.stopContinuousPhysics();
        const recovered = resumed.receipt(`frame-${crashFrame}-execute`)!;
        expect(recovered).toMatchObject({
          accepted: true,
          code: "motion_completed",
          worldBeforeRevision: execution.admission.world_revision,
          frameCount: expectedFrames
        });
        expect(recovered.worldAfterRevision).toBe(
          execution.admission.world_revision + expectedFrames
        );
        const recoveredCheckpoint = await store.readHumanoidCheckpoint();
        expect(recoveredCheckpoint.action_execution_ledger.active).toEqual({});
        expect(recoveredCheckpoint.action_commit_outbox.pending).toEqual({});
        const actions = await store.readJournal("actions");
        expect(actions.filter((entry) => (
          journalField(entry, "transactionId") === recovered.transactionId
        ))).toHaveLength(1);
      } finally {
        await resumedWorld?.dispose();
        await world.dispose();
      }
    },
    45_000
  );

  it("uses the permanent transaction tombstone after checkpoint receipt pruning", async () => {
    const root = await mkdtemp(join(tmpdir(), "hear-transaction-tombstone-"));
    temporaryDirectories.push(root);
    const store = await RunStore.create(root, {
      mission: "验证长期事务幂等身份",
      scenarioId: "humanoid-transaction-tombstone",
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
      const runtime = new HumanoidRunRuntime({
        store,
        goal: scenario.default_goal,
        world,
        checkpoint: initial
      });
      const manifest = await activateGoal(runtime, scenario.default_goal);
      await runtime.start(false);
      await runtime.stopContinuousPhysics();
      const planned = await invokeModelAction(runtime,
        "plan_whole_body_motion",
        {
          id: "tombstone-motion",
          intent: "完成一次不可重复领取的短动作",
          duration_seconds: 0.1,
          keyframes: [{ at_seconds: 0 }, { at_seconds: 0.1 }]
        },
        "tombstone-plan",
        HUMANOID_AGENT_IDS.motion
      );
      const executionInput = { planning_transaction_id: planned.transactionId };
      const executed = await invokeModelAction(runtime,
        "execute_whole_body_motion",
        executionInput,
        "tombstone-execute",
        HUMANOID_AGENT_IDS.executor
      );
      const pruned = await store.readHumanoidCheckpoint();
      delete pruned.committed_actions[executed.transactionId];
      delete pruned.committed_actions[planned.transactionId];
      await store.writeCheckpoint(pruned);

      resumedWorld = await HumanoidWorld.create(scenario, pruned.world_checkpoint);
      const resumed = new HumanoidRunRuntime({
        store,
        goal: scenario.default_goal,
        world: resumedWorld,
        checkpoint: pruned
      });
      await resumed.initializeGoalAutonomy(manifest);
      const beforeReplay = resumedWorld.snapshot();
      expect(await invokeModelAction(resumed,
        "execute_whole_body_motion",
        executionInput,
        "tombstone-execute",
        HUMANOID_AGENT_IDS.executor
      )).toEqual(executed);
      expect(resumedWorld.snapshot()).toEqual(beforeReplay);
      await expect(invokeModelAction(resumed,
        "execute_whole_body_motion",
        { planning_transaction_id: "rebound-plan" },
        "tombstone-execute",
        HUMANOID_AGENT_IDS.executor
      )).rejects.toThrow("transaction conflict");
    } finally {
      await resumedWorld?.dispose();
      await world.dispose();
    }
  }, 30_000);

  it("recovers a terminal physical checkpoint without replaying its committed frames", async () => {
    const root = await mkdtemp(join(tmpdir(), "hear-terminal-cut-recovery-"));
    temporaryDirectories.push(root);
    const store = await RunStore.create(root, {
      mission: "验证终帧与动作提交之间的崩溃恢复",
      scenarioId: "humanoid-terminal-cut-recovery",
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
      const runtime = new HumanoidRunRuntime({
        store,
        goal: scenario.default_goal,
        world,
        checkpoint: initial
      });
      const manifest = await activateGoal(runtime, scenario.default_goal);
      await runtime.start(false);
      await runtime.stopContinuousPhysics();
      const planned = await invokeModelAction(runtime,
        "plan_whole_body_motion",
        {
          id: "terminal-cut-motion",
          intent: "执行十帧全身保持动作",
          duration_seconds: 0.2,
          keyframes: [{ at_seconds: 0 }, { at_seconds: 0.2 }]
        },
        "terminal-cut-plan",
        HUMANOID_AGENT_IDS.motion
      );
      expect(planned.accepted).toBe(true);

      const executionInput = { planning_transaction_id: planned.transactionId };
      await authorizeModelAction(
        runtime,
        "execute_whole_body_motion",
        executionInput,
        "terminal-cut-execute",
        HUMANOID_AGENT_IDS.executor
      );
      const originalWrite = store.writeCheckpoint.bind(store);
      vi.spyOn(store, "writeCheckpoint").mockImplementation(async (checkpoint) => {
        if (checkpoint.action_commit_outbox.pending["terminal-cut-execute"]) {
          throw new Error("crash before action outbox");
        }
        await originalWrite(checkpoint);
      });
      await expect(invokeModelAction(runtime,
        "execute_whole_body_motion",
        executionInput,
        "terminal-cut-execute",
        HUMANOID_AGENT_IDS.executor
      )).rejects.toThrow("crash before action outbox");

      const crashCheckpoint = await store.readHumanoidCheckpoint();
      const executionLedger = crashCheckpoint.action_execution_ledger.active[
        "terminal-cut-execute"
      ]!;
      expect(executionLedger).toMatchObject({
        status: "executing",
        progress: { committed_frame_count: 10 }
      });
      expect(crashCheckpoint.committed_actions["terminal-cut-execute"]).toBeUndefined();
      expect(crashCheckpoint.world_checkpoint.motions[0]!.terminal).not.toBeNull();

      vi.restoreAllMocks();
      resumedWorld = await HumanoidWorld.create(
        scenario,
        crashCheckpoint.world_checkpoint
      );
      const beforeRecovery = resumedWorld.snapshot();
      const resumed = new HumanoidRunRuntime({
        store,
        goal: scenario.default_goal,
        world: resumedWorld,
        checkpoint: crashCheckpoint
      });
      await resumed.initializeGoalAutonomy(manifest);
      const recovered = await invokeModelAction(resumed,
        "execute_whole_body_motion",
        executionInput,
        "terminal-cut-execute",
        HUMANOID_AGENT_IDS.executor
      );
      expect(recovered).toMatchObject({
        accepted: true,
        code: "motion_completed",
        worldBeforeRevision: executionLedger.admission.world_revision,
        frameCount: 10
      });
      expect(resumedWorld.snapshot().worldRevision).toBe(beforeRecovery.worldRevision);
      const persisted = await store.readHumanoidCheckpoint();
      expect(persisted.action_execution_ledger.active).toEqual({});
      expect(persisted.world_checkpoint.motions).toEqual([]);
      const actions = await store.readJournal("actions");
      expect(actions.filter((entry) => (
        journalField(entry, "transactionId") === recovered.transactionId
      ))).toHaveLength(1);
    } finally {
      vi.restoreAllMocks();
      await resumedWorld?.dispose();
      await world.dispose();
    }
  }, 45_000);

  it.each([
    "actions",
    "goal_evidence",
    "action_identities",
    "events",
    "ack_checkpoint"
  ] as const)(
    "recovers the physical action outbox after %s persistence fails",
    async (failureStage) => {
      const root = await mkdtemp(join(tmpdir(), `hear-outbox-${failureStage}-`));
      temporaryDirectories.push(root);
      const store = await RunStore.create(root, {
        mission: "验证物理动作提交各阶段恢复",
        scenarioId: `humanoid-outbox-${failureStage}`,
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
        const runtime = new HumanoidRunRuntime({
          store,
          goal: scenario.default_goal,
          world,
          checkpoint: initial
        });
        const manifest = await activateGoal(runtime, scenario.default_goal);
        await runtime.start(false);
        await runtime.stopContinuousPhysics();
        const planned = await invokeModelAction(runtime,
          "plan_whole_body_motion",
          {
            id: `outbox-${failureStage}-motion`,
            intent: "执行一次可恢复提交的短全身动作",
            duration_seconds: 0.1,
            keyframes: [{ at_seconds: 0 }, { at_seconds: 0.1 }]
          },
          `outbox-${failureStage}-plan`,
          HUMANOID_AGENT_IDS.motion
        );
        const executionInput = { planning_transaction_id: planned.transactionId };
        const transactionId = `outbox-${failureStage}-execute`;
        const failure = new Error(`injected ${failureStage} failure`);
        await authorizeModelAction(
          runtime,
          "execute_whole_body_motion",
          executionInput,
          transactionId,
          HUMANOID_AGENT_IDS.executor
        );
        if (failureStage === "events") {
          vi.spyOn(store, "appendRuntimeEvents").mockRejectedValueOnce(failure);
        } else if (failureStage === "ack_checkpoint") {
          const originalWrite = store.writeCheckpoint.bind(store);
          let stagedOutboxPersisted = false;
          vi.spyOn(store, "writeCheckpoint").mockImplementation(async (checkpoint) => {
            if (checkpoint.action_commit_outbox.pending[transactionId]) {
              stagedOutboxPersisted = true;
            } else if (stagedOutboxPersisted
              && checkpoint.action_execution_ledger.active[transactionId] === undefined) {
              throw failure;
            }
            await originalWrite(checkpoint);
          });
        } else {
          const originalAppend = store.append.bind(store);
          let failed = false;
          vi.spyOn(store, "append").mockImplementation(async (name, value) => {
            if (!failed && name === failureStage) {
              failed = true;
              throw failure;
            }
            await originalAppend(name, value);
          });
        }
        await expect(invokeModelAction(runtime,
          "execute_whole_body_motion",
          executionInput,
          transactionId,
          HUMANOID_AGENT_IDS.executor
        )).rejects.toThrow(failure.message);
        const staged = await store.readHumanoidCheckpoint();
        expect(staged.action_commit_outbox.pending[transactionId]).toBeDefined();
        expect(staged.action_execution_ledger.active[transactionId]).toMatchObject({
          status: "terminal"
        });
        expect(staged.world_checkpoint.motions[0]!.terminal).not.toBeNull();

        vi.restoreAllMocks();
        const resumedStore = await RunStore.open(store.runDir);
        resumedWorld = await HumanoidWorld.create(scenario, staged.world_checkpoint);
        const beforeRecovery = resumedWorld.snapshot();
        const resumed = new HumanoidRunRuntime({
          store: resumedStore,
          goal: scenario.default_goal,
          world: resumedWorld,
          checkpoint: staged
        });
        await resumed.initializeGoalAutonomy(manifest);
        const recovered = await invokeModelAction(resumed,
          "execute_whole_body_motion",
          executionInput,
          transactionId,
          HUMANOID_AGENT_IDS.executor
        );
        expect(recovered).toMatchObject({ accepted: true, code: "motion_completed" });
        expect(resumedWorld.snapshot()).toEqual(beforeRecovery);
        const persisted = await resumedStore.readHumanoidCheckpoint();
        expect(persisted.action_commit_outbox.pending).toEqual({});
        expect(persisted.action_execution_ledger.active).toEqual({});
        expect(persisted.world_checkpoint.motions).toEqual([]);
        const actions = await resumedStore.readJournal("actions");
        expect(actions.filter((entry) => (
          journalField(entry, "transactionId") === transactionId
        ))).toHaveLength(1);
      } finally {
        vi.restoreAllMocks();
        await resumedWorld?.dispose();
        await world.dispose();
      }
    },
    45_000
  );

  it("advances end-effector stability only on committed physical frames and preserves it on resume", async () => {
    const root = await mkdtemp(join(tmpdir(), "hear-humanoid-goal-progress-"));
    temporaryDirectories.push(root);
    const world = await HumanoidWorld.create(scenario);
    let resumedWorld: HumanoidWorld | undefined;
    try {
      const initialWorld = world.snapshot();
      const goal = GoalSchema.parse({
        summary: "持续保持左腕世界位置",
        predicates: [{
          type: "end_effector_at",
          end_effector: "left_wrist",
          frame: "world",
          target: initialWorld.robot.links.left_wrist_yaw_link.position,
          tolerance: 5,
          stable_frames: 500
        }]
      });
      const store = await RunStore.create(root, {
        mission: "保持左腕末端位置",
        scenarioId: "humanoid-goal-progress-test",
        scenario,
        goal,
        runtime: "humanoid_g1"
      });
      const initial = createHumanoidRunCheckpoint({ store, goal, world });
      await store.writeCheckpoint(initial);
      const runtime = new HumanoidRunRuntime({ store, goal, world, checkpoint: initial });
      const goalManifest = await activateGoal(runtime, goal);
      await runtime.start(false);
      const continuousStartFrame = runtime.snapshot().frame;
      await waitForWorldFrame(runtime, continuousStartFrame);
      await runtime.stopContinuousPhysics();

      const beforeReads = runtime.checkpoint.goal_progress;
      await invokeModelAction(runtime,
        "observe_humanoid",
        {},
        "goal-observe-before-1",
        HUMANOID_AGENT_IDS.sentry
      );
      runtime.contextAnchor(HUMANOID_AGENT_IDS.coordinator);
      await invokeModelAction(runtime,
        "observe_humanoid",
        {},
        "goal-observe-before-2",
        HUMANOID_AGENT_IDS.sentry
      );
      expect(runtime.checkpoint.goal_progress).toEqual(beforeReads);

      const plan = await invokeModelAction(runtime,
        "plan_whole_body_motion",
        {
          id: "goal-progress-hold",
          intent: "在短物理窗口内保持当前全身参考",
          duration_seconds: 0.12,
          keyframes: [{ at_seconds: 0 }, { at_seconds: 0.12 }]
        },
        "goal-progress-plan",
        HUMANOID_AGENT_IDS.motion
      );
      expect(plan.accepted).toBe(true);
      expect(runtime.checkpoint.goal_progress).toEqual(beforeReads);

      const execution = await invokeModelAction(runtime,
        "execute_whole_body_motion",
        { planning_transaction_id: plan.transactionId },
        "goal-progress-execute",
        HUMANOID_AGENT_IDS.executor
      );
      expect(execution).toMatchObject({ accepted: true, code: "motion_completed" });
      expect(execution.frameCount).toBeGreaterThan(0);
      const afterExecution = runtime.checkpoint.goal_progress;
      expect(afterExecution).toMatchObject({
        last_world_frame: runtime.snapshot().frame,
        last_world_revision: runtime.snapshot().worldRevision,
        predicate_streaks: [beforeReads.predicate_streaks[0]! + execution.frameCount]
      });

      runtime.contextAnchor(HUMANOID_AGENT_IDS.coordinator);
      await invokeModelAction(runtime,
        "observe_humanoid",
        {},
        "goal-observe-after",
        HUMANOID_AGENT_IDS.sentry
      );
      expect(runtime.checkpoint.goal_progress).toEqual(afterExecution);

      const persisted = await store.readHumanoidCheckpoint();
      expect(persisted.goal_progress).toEqual(afterExecution);
      resumedWorld = await HumanoidWorld.create(scenario, persisted.world_checkpoint);
      const resumed = new HumanoidRunRuntime({
        store,
        goal,
        world: resumedWorld,
        checkpoint: persisted
      });
      await resumed.initializeGoalAutonomy(goalManifest);
      resumed.contextAnchor(HUMANOID_AGENT_IDS.coordinator);
      expect(resumed.checkpoint.goal_progress).toEqual(afterExecution);
      await invokeModelAction(resumed,
        "observe_humanoid",
        {},
        "goal-observe-resumed",
        HUMANOID_AGENT_IDS.sentry
      );
      expect(resumed.checkpoint.goal_progress).toEqual(afterExecution);
    } finally {
      await resumedWorld?.dispose();
      await world.dispose();
    }
  }, 45_000);

  it("projects a live context anchor across an asynchronous stationary publication", async () => {
    const root = await mkdtemp(join(tmpdir(), "hear-humanoid-context-cut-"));
    temporaryDirectories.push(root);
    const world = await HumanoidWorld.create(scenario);
    try {
      const store = await RunStore.create(root, {
        mission: "读取持续物理世界中的一致上下文",
        scenarioId: "humanoid-context-cut-test",
        scenario,
        goal: scenario.default_goal,
        runtime: "humanoid_g1"
      });
      const initial = createHumanoidRunCheckpoint({
        store,
        goal: scenario.default_goal,
        world
      });
      await store.writeCheckpoint(initial);
      const runtime = new HumanoidRunRuntime({
        store,
        goal: scenario.default_goal,
        world,
        checkpoint: initial
      });
      await activateGoal(runtime, scenario.default_goal);
      const durableProgress = runtime.checkpoint.goal_progress;
      const live = await world.advanceStationary();
      expect(live).not.toBeNull();

      const anchor = runtime.contextAnchor(HUMANOID_AGENT_IDS.sentry) as {
        world_frame: number;
        world_revision: number;
        goal_state: { worldFrame: number; worldRevision: number };
      };

      expect(anchor).toMatchObject({
        world_frame: live!.frame,
        world_revision: live!.worldRevision,
        goal_state: {
          worldFrame: live!.frame,
          worldRevision: live!.worldRevision
        }
      });
      expect(runtime.contextWorldIdentity()).toEqual({
        worldRevision: live!.worldRevision
      });
      expect(runtime.checkpoint.goal_progress).toEqual(durableProgress);
    } finally {
      await world.dispose();
    }
  }, 45_000);

  it("completes a cycle only after queued stationary frames reach Goal progress", async () => {
    const root = await mkdtemp(join(tmpdir(), "hear-humanoid-cycle-frame-cut-"));
    temporaryDirectories.push(root);
    const target = { x: 2, y: 0, z: 2.55 };
    const goal = GoalSchema.parse({
      summary: "自主行走到前方目标点",
      predicates: [{ type: "robot_at", target, tolerance: 0.3 }]
    });
    const store = await RunStore.create(root, {
      mission: "行走后在持续物理世界中完成自主周期",
      scenarioId: "humanoid-cycle-frame-cut-test",
      scenario,
      goal,
      runtime: "humanoid_g1",
      runMode: "continuous"
    });
    const world = await HumanoidWorld.create(scenario);
    const publicationEntered = deferredSignal();
    const releasePublication = deferredSignal();
    const episodeAppendEntered = deferredSignal();
    const releaseEpisodeAppend = deferredSignal();
    let blockNextStationaryPublication = false;
    let publicationBlocked = false;
    let runtime: HumanoidRunRuntime | undefined;
    try {
      const initial = createHumanoidRunCheckpoint({ store, goal, world });
      await store.writeCheckpoint(initial);
      runtime = new HumanoidRunRuntime({
        store,
        goal,
        world,
        checkpoint: initial,
        eventSink: async (event) => {
          if (!blockNextStationaryPublication
            || publicationBlocked
            || event.type !== "humanoid_world_frame"
            || journalField(event.data, "frame_source") !== "stationary") return;
          publicationBlocked = true;
          publicationEntered.resolve();
          await releasePublication.promise;
        }
      });
      await activateGoal(runtime, goal);
      await runtime.start(false);

      const navigationSkillTransactionId = await bindPlanningSkill(
        runtime,
        "cycle-frame-cut-plan",
        {
          skill: "retreat",
          target,
          minimum_obstacle_clearance_m: 0.1
        },
        "route"
      );
      const planned = await invokeModelAction(runtime,
        "plan_humanoid_skill",
        { skill_transaction_id: navigationSkillTransactionId },
        "cycle-frame-cut-plan",
        HUMANOID_AGENT_IDS.motion
      );
      expect(planned.accepted, JSON.stringify(planned)).toBe(true);
      const executed = await invokeModelAction(runtime,
        "execute_humanoid_skill",
        { planning_transaction_id: planned.transactionId },
        "cycle-frame-cut-execute",
        HUMANOID_AGENT_IDS.executor
      );
      expect(executed).toMatchObject({ accepted: true, code: "navigation_completed" });
      await invokeModelAction(runtime,
        "observe_humanoid",
        {},
        "cycle-frame-cut-observe-after-execution",
        HUMANOID_AGENT_IDS.sentry
      );

      blockNextStationaryPublication = true;
      await publicationEntered.promise;
      let queuedFrameObserved = false;
      for (let attempt = 0; attempt < 100; attempt += 1) {
        const progress = runtime.checkpoint.goal_progress;
        if (progress && runtime.snapshot().frame > progress.last_world_frame) {
          queuedFrameObserved = true;
          break;
        }
        await new Promise((resolve) => setTimeout(resolve, 10));
      }
      expect(queuedFrameObserved).toBe(true);

      const append = store.append.bind(store);
      vi.spyOn(store, "append").mockImplementation(async (name, value) => {
        if (name === "episodes") {
          episodeAppendEntered.resolve();
          await releaseEpisodeAppend.promise;
        }
        await append(name, value);
      });
      const completion = runtime.completeCycle(JSON.stringify({
        summary: "完成真实导航并核验连续物理终态",
        evidence_transaction_ids: [executed.transactionId]
      }));
      const earlyResult = await Promise.race([
        completion.then(() => "resolved" as const, () => "rejected" as const),
        new Promise<"pending">((resolve) => setTimeout(() => resolve("pending"), 100))
      ]);
      expect(earlyResult).toBe("pending");
      releasePublication.resolve();
      await episodeAppendEntered.promise;

      await expect(runtime.setActiveAgent(HUMANOID_AGENT_IDS.sentry)).resolves.toBeUndefined();
      releaseEpisodeAppend.resolve();

      await expect(completion).resolves.toBe(true);
      const checkpoint = runtime.checkpoint;
      expect(checkpoint.goal_progress).toBeNull();
      expect(checkpoint.checker).toBeNull();
      expect(checkpoint.world.frame).toBe(checkpoint.world_checkpoint.frame);
      expect(checkpoint.world.worldRevision).toBe(
        checkpoint.world_checkpoint.worldRevision
      );
    } finally {
      releasePublication.resolve();
      releaseEpisodeAppend.resolve();
      await runtime?.stopContinuousPhysics();
      await world.dispose();
    }
  }, 60_000);

  it("persists a safe pause after the process signal reaches a stationary frame", async () => {
    const root = await mkdtemp(join(tmpdir(), "hear-humanoid-signal-pause-"));
    temporaryDirectories.push(root);
    const store = await RunStore.create(root, {
      mission: "在持续物理帧期间安全暂停",
      scenarioId: "humanoid-signal-pause-test",
      scenario,
      goal: scenario.default_goal,
      runtime: "humanoid_g1"
    });
    const world = await HumanoidWorld.create(scenario);
    const controller = new AbortController();
    let runtime: HumanoidRunRuntime | undefined;
    try {
      const initial = createHumanoidRunCheckpoint({
        store,
        goal: scenario.default_goal,
        world
      });
      await store.writeCheckpoint(initial);
      runtime = new HumanoidRunRuntime({
        store,
        goal: scenario.default_goal,
        world,
        checkpoint: initial,
        signal: controller.signal
      });
      await activateGoal(runtime, scenario.default_goal);
      await runtime.start(false);

      const frameBeforeSignal = runtime.snapshot().frame;
      controller.abort(new Error("simulated process signal"));
      await waitForWorldFrame(runtime, frameBeforeSignal);

      await expect(runtime.pause("simulated process signal")).resolves.toBeUndefined();
      const persisted = await store.readHumanoidCheckpoint();
      expect(persisted).toMatchObject({
        status: "paused",
        error: null,
        active_agent_id: null,
        active_agent_ids: []
      });
      expect(persisted.world.frame).toBe(persisted.world_checkpoint.frame);
      expect(persisted.world.worldRevision).toBe(
        persisted.world_checkpoint.worldRevision
      );
      expect((await store.readJournal("events")).filter((event) => (
        journalField(event, "type") === "run_paused"
      ))).toHaveLength(1);
    } finally {
      await runtime?.stopContinuousPhysics();
      await world.dispose();
    }
  }, 45_000);

  it("retires only through a recorded Goal Manager response and rejects missing recovery evidence", async () => {
    const root = await mkdtemp(join(tmpdir(), "hear-humanoid-goal-retirement-"));
    temporaryDirectories.push(root);
    const store = await RunStore.create(root, {
      mission: "持续选择并执行可验证的人形目标",
      scenarioId: "humanoid-goal-retirement-test",
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
      const manifest = await activateGoal(runtime, scenario.default_goal);
      const anchor = runtime.contextAnchor(HUMANOID_AGENT_IDS.goalManager) as {
        goal_context: { evidence_ref: string };
      };
      const retirementInput = {
        status: "abandoned" as const,
        reason: "目标管理模型根据当前物理观察决定终止这一阶段",
        evidence_refs: [anchor.goal_context.evidence_ref]
      };
      const authority = await modelToolAuthority(
        runtime,
        "retire_goal_epoch",
        retirementInput,
        "response-retire-goal"
      );
      await expect(runtime.retireGoalEpoch(retirementInput, {
        ...authority,
        arguments_sha256: "0".repeat(64)
      })).rejects.toThrow("no completed model response authority");
      await runtime.retireGoalEpoch(retirementInput, authority);

      expect(runtime.checkpoint).toMatchObject({
        goal_dag: {
          status: "awaiting_model_selection",
          current_epoch_id: null,
          epochs: [expect.objectContaining({ status: "abandoned" })]
        },
        goal_progress: null,
        checker: null
      });
      expect(runtime.validateGoalTransition()).toMatchObject({
        status: "abandoned",
        reason: retirementInput.reason
      });

      const persisted = await store.readHumanoidCheckpoint();
      await unlink(join(store.runDir, "goal_evidence.jsonl"));
      resumedWorld = await HumanoidWorld.create(scenario, persisted.world_checkpoint);
      const resumed = new HumanoidRunRuntime({
        store,
        goal: scenario.default_goal,
        world: resumedWorld,
        checkpoint: persisted
      });
      await expect(resumed.initializeGoalAutonomy(manifest)).rejects.toThrow(
        /no longer verifiable|Physical evidence is unavailable/u
      );
    } finally {
      await resumedWorld?.dispose();
      await world.dispose();
    }
  }, 45_000);

  it("keeps continuous authority running during planning and records only causal action frames", async () => {
    const root = await mkdtemp(join(tmpdir(), "hear-humanoid-live-planning-"));
    temporaryDirectories.push(root);
    const generator = new SlowFirstMotionGenerator();
    const world = await HumanoidWorld.create(scenario, undefined, {
      motionGeneratorFactory: async () => generator
    });
    let runtime: HumanoidRunRuntime | undefined;
    try {
      const store = await RunStore.create(root, {
        mission: "规划期间保持真实物理时间",
        scenarioId: "humanoid-live-planning-test",
        scenario,
        goal: scenario.default_goal,
        runtime: "humanoid_g1"
      });
      const checkpoint = createHumanoidRunCheckpoint({
        store,
        goal: scenario.default_goal,
        world
      });
      await store.writeCheckpoint(checkpoint);
      runtime = new HumanoidRunRuntime({
        store,
        goal: scenario.default_goal,
        world,
        checkpoint
      });
      await activateGoal(runtime, scenario.default_goal);
      await runtime.start(false);

      const before = runtime.snapshot();
      const planning = invokeModelAction(runtime,
        "plan_whole_body_motion",
        {
          id: "live-planning-hold",
          intent: "规划时保持平衡并让权威世界持续推进",
          duration_seconds: 0.1,
          keyframes: [{ at_seconds: 0 }, { at_seconds: 0.1 }]
        },
        "live-planning-transaction",
        HUMANOID_AGENT_IDS.motion
      );
      await generator.firstCallEntered;
      const planningStarted = runtime.snapshot();
      await waitForWorldFrame(runtime, planningStarted.frame);
      expect(runtime.snapshot().worldRevision).toBeGreaterThan(before.worldRevision);

      generator.releaseFirstCall();
      const planned = await planning;
      expect(planned.accepted).toBe(true);
      expect(planned.frameCount).toBe(0);
      expect(planned.worldAfterRevision).toBeGreaterThan(planned.worldBeforeRevision);

      const executionStartedAt = performance.now();
      const executed = await invokeModelAction(runtime,
        "execute_whole_body_motion",
        { planning_transaction_id: planned.transactionId },
        "live-planning-execution",
        HUMANOID_AGENT_IDS.executor
      );
      const executionElapsed = performance.now() - executionStartedAt;
      expect(executed.accepted).toBe(true);
      expect(executed.frameCount).toBe(5);
      expect(executionElapsed).toBeGreaterThanOrEqual(70);
      expect(executed.detail).toMatchObject({
        result: {
          revalidation: {
            performed: true,
            accepted: true
          }
        }
      });
    } finally {
      generator.releaseFirstCall();
      await runtime?.stopContinuousPhysics();
      await world.dispose();
    }
  }, 45_000);
});

async function activateGoal(
  runtime: HumanoidRunRuntime,
  goal: Goal
): Promise<AgentManifest> {
  const manifest = humanoidTestManifest();
  await runtime.initializeGoalAutonomy(manifest);
  runtime.contextAnchor(HUMANOID_AGENT_IDS.goalManager);
  const proposalInput = {
    candidates: [
      {
        proposal_id: "runtime-test-primary",
        mission_link: "验证运行时选择的主目标",
        goal,
        dependency_candidate_ids: []
      },
      {
        proposal_id: "runtime-test-alternative",
        mission_link: "保留由同一模型提出但未自动选择的替代目标",
        goal: {
          ...structuredClone(goal),
          summary: `${goal.summary}（替代候选）`
        },
        dependency_candidate_ids: []
      }
    ]
  };
  const proposalAuthority = await modelToolAuthority(
    runtime,
    "submit_goal_candidates",
    proposalInput,
    "response-submit-goal"
  );
  const submitted = await runtime.submitGoalCandidates(
    proposalInput,
    proposalAuthority
  ) as { candidates: Array<{ candidate_sequence: number }> };
  runtime.contextAnchor(HUMANOID_AGENT_IDS.goalManager);
  const selectionInput = {
    candidate_sequence: submitted.candidates[0]!.candidate_sequence
  };
  const selectionAuthority = await modelToolAuthority(
    runtime,
    "select_goal_candidate",
    selectionInput,
    "response-select-goal"
  );
  await runtime.selectGoalCandidate(selectionInput, selectionAuthority);
  return manifest;
}

function humanoidTestManifest(): AgentManifest {
  return {
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
}

const actionAuthorities = new Map<string, {
  tool_call_id: string;
  tool_name: string;
  arguments_sha256: string;
}>();

async function invokeModelAction(
  runtime: HumanoidRunRuntime,
  action: Parameters<HumanoidRunRuntime["invoke"]>[0],
  input: unknown,
  transactionId: string,
  agentId: string
) {
  if (agentId === HUMANOID_AGENT_IDS.motion
    && action !== "observe_humanoid"
    && !runtime.isActionAvailable(action, agentId)) {
    await invokeModelAction(
      runtime,
      "observe_humanoid",
      {},
      `${transactionId}-motion-observation`,
      agentId
    );
  }
  const authority = await authorizeModelAction(
    runtime,
    action,
    input,
    transactionId,
    agentId
  );
  return runtime.invoke(action, input, transactionId, agentId, authority);
}

async function bindPlanningSkill(
  runtime: HumanoidRunRuntime,
  prefix: string,
  invocation: HumanoidSkillInvocation,
  phase: string
): Promise<string> {
  const planTransactionId = `${prefix}-skill-plan`;
  const nodeId = `${prefix}-skill-node`;
  const proposal = {
    objective: `Authorize ${prefix}`,
    strategies: [{
      strategy_id: `${prefix}-strategy`,
      rationale: "Exercise the production Skill authority path in this runtime test",
      nodes: [{
        node_id: nodeId,
        invocation,
        depends_on_node_ids: []
      }]
    }],
    selected_strategy_id: `${prefix}-strategy`
  };
  const registered = await invokeModelAction(
    runtime,
    "submit_humanoid_skill_plan",
    proposal,
    planTransactionId,
    HUMANOID_AGENT_IDS.motion
  );
  expect(registered.accepted, JSON.stringify(registered)).toBe(true);
  const binding = await invokeModelAction(
    runtime,
    "begin_humanoid_skill",
    {
      skill_plan_transaction_id: planTransactionId,
      skill_node_id: nodeId,
      invocation,
      phase
    },
    `${prefix}-skill-binding`,
    HUMANOID_AGENT_IDS.motion
  );
  expect(binding.accepted, JSON.stringify(binding)).toBe(true);
  return binding.transactionId;
}

async function authorizeModelAction(
  runtime: HumanoidRunRuntime,
  action: Parameters<HumanoidRunRuntime["invoke"]>[0],
  input: unknown,
  transactionId: string,
  agentId: string
) {
  const key = `${runtime.runId}\0${transactionId}`;
  const existing = actionAuthorities.get(key);
  if (existing) return existing;
  const modelCallId = await runtime.recordModelCallStarted(agentId);
  const authority = {
    tool_call_id: transactionId,
    tool_name: action,
    arguments_sha256: modelPayloadSha256(input)
  };
  await runtime.recordModelCallCompleted({
    modelCallId,
    agentId,
    responseId: `response-${modelCallId}`,
    responseOutputSha256: modelPayloadSha256({ modelCallId, transactionId }),
    toolCalls: [{
      toolCallId: transactionId,
      toolName: action,
      argumentsSha256: authority.arguments_sha256
    }]
  });
  actionAuthorities.set(key, authority);
  return authority;
}

async function modelToolAuthority(
  runtime: HumanoidRunRuntime,
  toolName: string,
  input: unknown,
  responseId: string
) {
  const modelCallId = await runtime.recordModelCallStarted(
    HUMANOID_AGENT_IDS.goalManager
  );
  const toolCallId = `${toolName}-${modelCallId}`;
  const argumentsSha256 = modelPayloadSha256(input);
  await runtime.recordModelCallCompleted({
    modelCallId,
    agentId: HUMANOID_AGENT_IDS.goalManager,
    responseId,
    responseOutputSha256: modelPayloadSha256({ responseId, toolCallId }),
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

function journalField(value: unknown, key: string): unknown {
  return journalNestedField(value, key);
}

function journalNestedField(value: unknown, ...path: string[]): unknown {
  let current = value;
  for (const key of path) {
    if (typeof current !== "object" || current === null || Array.isArray(current)) {
      return undefined;
    }
    current = (current as Record<string, unknown>)[key];
  }
  return current;
}

async function waitForWorldFrame(
  runtime: HumanoidRunRuntime,
  initialFrame: number
): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (runtime.snapshot().frame > initialFrame) return;
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  throw new Error("Continuous humanoid physics did not advance a world frame");
}

class SlowFirstMotionGenerator implements HumanoidMotionGenerator {
  readonly descriptor = {
    protocol: "humanoid-motion-generator-v1" as const,
    implementation: "slow_live_planning_test",
    motionClass: "constraint_solver" as const,
    sampling: "deterministic" as const
  };
  readonly #entered = deferredSignal();
  readonly #release = deferredSignal();
  #blocked = false;

  get firstCallEntered(): Promise<void> {
    return this.#entered.promise;
  }

  releaseFirstCall(): void {
    this.#release.resolve();
  }

  async generate(input: HumanoidMotionGeneratorInput) {
    if (!this.#blocked) {
      this.#blocked = true;
      this.#entered.resolve();
      await this.#release.promise;
    }
    const frameCount = Math.ceil(
      input.plan.duration_seconds / input.controlStepSeconds
    );
    return {
      version: 1 as const,
      protocol: "humanoid-motion-v1" as const,
      generator: this.descriptor.implementation,
      controlStepSeconds: input.controlStepSeconds,
      durationSeconds: input.plan.duration_seconds,
      frames: Array.from({ length: frameCount }, (_, index) => ({
        atSeconds: Math.min(
          (index + 1) * input.controlStepSeconds,
          input.plan.duration_seconds
        ),
        reference: serializeHumanoidReference(input.baseline)
      }))
    };
  }

  async dispose(): Promise<void> {}
}

function deferredSignal(): { promise: Promise<void>; resolve(): void } {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
