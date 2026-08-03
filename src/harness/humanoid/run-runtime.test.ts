import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { ScenarioSchema } from "../../domain/schema.js";
import type { RuntimeEvent } from "../../runtime/events.js";
import { RunStore } from "../../persistence/run-store.js";
import { HumanoidWorld } from "../../world/humanoid/world.js";
import { HUMANOID_AGENT_IDS } from "./agents.js";
import { createHumanoidRunCheckpoint } from "./run-checkpoint.js";
import { HumanoidRunRuntime } from "./run-runtime.js";

const scenario = ScenarioSchema.parse({
  title: "人形持久运行场",
  seed: 47,
  bounds: { width: 10, depth: 10 },
  visibility_radius: 6,
  robot: { x: 2, z: 2, yaw: 0 },
  obstacles: [],
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
  it("persists lifecycle, physical frames, receipts and restart idempotency", async () => {
    const root = await mkdtemp(join(tmpdir(), "hear-humanoid-run-"));
    temporaryDirectories.push(root);
    const store = await RunStore.create(root, {
      mission: "自主保持平衡并改变双臂姿态",
      scenarioId: "humanoid-runtime-test",
      scenario,
      goal: scenario.default_goal,
      runtime: "humanoid_g1"
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

      await runtime.start(false);
      const observation = await runtime.invoke(
        "observe_humanoid",
        {},
        "observe-persisted",
        HUMANOID_AGENT_IDS.sentry
      );
      const planInput = {
        id: "persistent-arm-motion",
        intent: "保持双脚支撑并连续改变双臂姿态",
        duration_seconds: 0.4,
        keyframes: [
          { at_seconds: 0 },
          {
            at_seconds: 0.4,
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
      const plan = await runtime.invoke(
        "plan_whole_body_motion",
        planInput,
        "plan-persisted",
        HUMANOID_AGENT_IDS.motion
      );
      expect(plan.accepted).toBe(true);
      const executionInput = { planning_transaction_id: plan.transactionId };
      const execution = await runtime.invoke(
        "execute_whole_body_motion",
        executionInput,
        "execute-persisted",
        HUMANOID_AGENT_IDS.executor
      );
      expect(execution.accepted).toBe(true);
      expect(execution.frameCount).toBeGreaterThan(10);

      expect(() => runtime.validateCycleEvidence([execution.transactionId])).toThrow(
        "requires accepted execution evidence"
      );

      const pendingBefore = world.snapshot();
      const pendingTarget = {
        ...pendingBefore.robot.rootPosition,
        z: pendingBefore.robot.rootPosition.z + 0.08
      };
      const pendingPlan = await runtime.invoke(
        "plan_whole_body_motion_candidates",
        {
          objective: "比较下一次连续全身动作候选",
          termination: {
            option_id: "persisted-forward-option",
            predicates: [{
              type: "root_near_point",
              body: null,
              object_id: null,
              zone_id: null,
              target: pendingTarget,
              tolerance_m: 0.035,
              minimum_normal_force: null,
              expected: null
            }],
            stable_steps: 2,
            phases: null
          },
          candidates: [
            {
              id: "unconsumed-noop-motion",
              intent: "没有实现前进目标",
              duration_seconds: 0.8,
              keyframes: [{ at_seconds: 0 }, { at_seconds: 0.8 }]
            },
            {
              id: "unconsumed-balanced-motion",
              intent: "保持双足支撑并连续前进",
              duration_seconds: 0.8,
              keyframes: [
                {
                  at_seconds: 0,
                  root_velocity: { forward_mps: 0.2, lateral_mps: 0 }
                },
                {
                  at_seconds: 0.8,
                  root_velocity: { forward_mps: 0.2, lateral_mps: 0 }
                }
              ]
            }
          ]
        },
        "plan-unconsumed",
        HUMANOID_AGENT_IDS.motion
      );
      expect(pendingPlan.accepted).toBe(true);
      expect(pendingPlan.code).toBe("whole_body_candidates_validated");
      const pendingExecution = await runtime.invoke(
        "execute_whole_body_motion",
        { planning_transaction_id: pendingPlan.transactionId },
        "execute-unconsumed",
        HUMANOID_AGENT_IDS.executor
      );
      expect(pendingExecution.accepted).toBe(true);
      expect(runtime.validateCycleEvidence([pendingExecution.transactionId])).toMatchObject({
        transactionId: pendingExecution.transactionId,
        code: "motion_option_succeeded"
      });
      await runtime.completeCycle(JSON.stringify({
        status: "cycle_completed",
        evidence_transaction_ids: [pendingExecution.transactionId]
      }));
      expect(() => runtime.validateCycleEvidence([pendingExecution.transactionId])).toThrow(
        "already consumed"
      );
      const rejectedExecution = await runtime.invoke(
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
      expect(await runtime.recallEmbodiedHistory({
        source_refs: [
          "episode:1",
          `action:${pendingExecution.transactionId}`,
          `action:${rejectedExecution.transactionId}`
        ],
        before_sequence: 2,
        limit: 3
      })).toMatchObject({
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
            code: "motion_option_succeeded",
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
        "plan_whole_body_motion_candidates",
        "plan_humanoid_navigation"
      ]);
      expect(persisted.world.frame).toBe(world.snapshot().frame);
      expect(persisted.world_checkpoint.frame).toBe(persisted.world.frame);
      expect(persisted.world_checkpoint.worldRevision).toBe(persisted.world.worldRevision);
      expect(persisted.committed_actions[observation.transactionId]).toEqual(observation);
      expect(persisted.committed_actions[execution.transactionId]).toEqual(execution);
      expect(persisted.embodied_memory).toMatchObject({
        version: 1,
        total_episodes: 1,
        pruned_episodes: 0,
        recent_episodes: [{
          sequence: 1,
          transaction_id: pendingExecution.transactionId,
          action: "execute_whole_body_motion",
          planning_action: "plan_whole_body_motion_candidates",
          candidate_count: 2,
          selected_rank: 2,
          selected_candidate_id: "unconsumed-balanced-motion",
          code: "motion_option_succeeded",
          motion_option: {
            option_id: "persisted-forward-option",
            status: "succeeded",
            termination_reason: "physical_success"
          },
          world_after_revision: pendingExecution.worldAfterRevision,
          goal_success: true
        }]
      });
      const memoryEvent = events.find((event) => event.type === "embodied_episode_recorded");
      expect(memoryEvent).toBeDefined();
      expect((memoryEvent!.data as { embodied_memory: unknown }).embodied_memory).toEqual(
        persisted.embodied_memory
      );
      expect(memoryEvent!.cursor).toBeTruthy();
      expect(events.some((event) => event.type === "run_started" && event.cursor)).toBe(true);
      expect(events.filter((event) => event.type === "humanoid_world_frame")).toHaveLength(
        execution.frameCount + pendingExecution.frameCount
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
      const beforeReplay = resumed.snapshot();
      const replayed = await resumed.invoke(
        "execute_whole_body_motion",
        executionInput,
        "execute-persisted",
        HUMANOID_AGENT_IDS.executor
      );
      expect(replayed).toEqual(execution);
      expect(resumed.snapshot()).toEqual(beforeReplay);
      await expect(resumed.invoke(
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
});
