import { describe, expect, it } from "vitest";
import {
  HUMANOID_END_EFFECTORS,
  ScenarioSchema,
  type JsonValue
} from "../../domain/schema.js";
import { humanoidEndEffectorPosition } from "../../world/humanoid/end-effectors.js";
import { HumanoidWorld } from "../../world/humanoid/world.js";
import { MAX_CHECKPOINT_ACTION_RECEIPTS } from "./embodied-memory.js";
import { HumanoidActionRuntime } from "./runtime.js";
import type { ScenarioBlockRemovalTransaction } from "../../domain/scenario-block-removal.js";
import { BlockRemovalAuthorityError } from "./block-removal-authority.js";

const scenario = ScenarioSchema.parse({
  title: "Humanoid action field",
  seed: 19,
  bounds: { width: 10, depth: 10 },
  visibility_radius: 6,
  robot: { x: 1.5, z: 1.5, yaw: 0 },
  obstacles: [],
  objects: [],
  zones: [],
  default_goal: {
    summary: "探索开放区域",
    predicates: [{
      type: "robot_at",
      target: { x: 1.5, y: 0, z: 2.1 },
      tolerance: 0.25
    }]
  }
});

describe("HumanoidActionRuntime", () => {
  it("exposes only authority-prepared block-removal transactions", async () => {
    const lightweight = lightweightObservationWorld();
    const prepared: Array<{
      solidId: string;
      executionTransactionId: string;
    }> = [];
    const transaction = {
      version: 1,
      transaction_id: "remove-1",
      solid_id: "block-a",
      block_id: "block-a"
    } as ScenarioBlockRemovalTransaction;
    const runtime = new HumanoidActionRuntime(lightweight.world, {
      prepareBlockRemoval: (input) => {
        prepared.push(input);
        return transaction;
      }
    });

    const accepted = await runtime.invoke(
      "remove_world_block",
      { solid_id: "block-a", execution_transaction_id: "execute-1" },
      "remove-1",
      "executor-agent"
    );
    expect(accepted).toMatchObject({
      accepted: true,
      code: "world_block_removal_authorized",
      frameCount: 0,
      detail: { removal_transaction: transaction }
    });
    expect(prepared).toMatchObject([{
      solidId: "block-a",
      executionTransactionId: "execute-1"
    }]);

    const rejectedRuntime = new HumanoidActionRuntime(lightweight.world, {
      prepareBlockRemoval: () => {
        throw new BlockRemovalAuthorityError(
          "block_removal_contact_force_insufficient",
          "contact was too weak"
        );
      }
    });
    await expect(rejectedRuntime.invoke(
      "remove_world_block",
      { solid_id: "block-a", execution_transaction_id: "execute-1" },
      "remove-2",
      "executor-agent"
    )).resolves.toMatchObject({
      accepted: false,
      code: "block_removal_contact_force_insufficient",
      detail: { reason: "contact was too weak" }
    });
  });

  it("bounds completed transaction history while preserving recent idempotency", async () => {
    const lightweight = lightweightObservationWorld();
    const runtime = new HumanoidActionRuntime(lightweight.world);
    const transactionCount = 30_000;

    for (let index = 0; index < transactionCount; index += 1) {
      await runtime.invoke(
        "observe_humanoid",
        {},
        `bounded-observation-${index}`,
        "perception-agent"
      );
    }

    const retained = Array.from({ length: transactionCount }, (_, index) => (
      runtime.receipt(`bounded-observation-${index}`)
    )).filter((receipt) => receipt !== undefined);
    expect(retained).toHaveLength(MAX_CHECKPOINT_ACTION_RECEIPTS);
    expect(runtime.receipt("bounded-observation-0")).toBeUndefined();
    expect(runtime.receipt(`bounded-observation-${transactionCount - 1}`)).toBeDefined();

    const callsBeforeRecentRetry = lightweight.observationCalls();
    await runtime.invoke(
      "observe_humanoid",
      {},
      `bounded-observation-${transactionCount - 1}`,
      "perception-agent"
    );
    expect(lightweight.observationCalls()).toBe(callsBeforeRecentRetry);

    await runtime.invoke(
      "observe_humanoid",
      {},
      "bounded-observation-0",
      "perception-agent"
    );
    expect(lightweight.observationCalls()).toBe(callsBeforeRecentRetry + 1);
  }, 20_000);

  it("retries only the durable commit after execution and fences later actions", async () => {
    const lightweight = lightweightObservationWorld();
    let commitAttempts = 0;
    const runtime = new HumanoidActionRuntime(lightweight.world, {
      receiptSink: () => {
        commitAttempts += 1;
        if (commitAttempts === 1) throw new Error("checkpoint unavailable");
      }
    });

    await expect(runtime.invoke(
      "observe_humanoid",
      {},
      "uncertain-commit",
      "perception-agent"
    )).rejects.toThrow("checkpoint unavailable");
    expect(lightweight.observationCalls()).toBe(1);

    await expect(runtime.invoke(
      "observe_humanoid",
      {},
      "later-action",
      "perception-agent"
    )).rejects.toThrow("retry transaction uncertain-commit");
    expect(lightweight.observationCalls()).toBe(1);

    const recovered = await runtime.invoke(
      "observe_humanoid",
      {},
      "uncertain-commit",
      "perception-agent"
    );
    expect(recovered.transactionId).toBe("uncertain-commit");
    expect(commitAttempts).toBe(2);
    expect(lightweight.observationCalls()).toBe(1);

    await runtime.invoke(
      "observe_humanoid",
      {},
      "later-action",
      "perception-agent"
    );
    expect(lightweight.observationCalls()).toBe(2);
  });

  it("keeps an accepted current plan outside the recent receipt window", async () => {
    const world = await HumanoidWorld.create(scenario);
    const runtime = new HumanoidActionRuntime(world);
    try {
      const input = motionPlan("protected-current-motion", runtime.snapshot(), 0.01);
      const plan = await runtime.invoke(
        "plan_whole_body_motion",
        input,
        "protected-current-plan",
        "motion-agent"
      );
      expect(plan.accepted).toBe(true);

      for (let index = 0; index < MAX_CHECKPOINT_ACTION_RECEIPTS + 8; index += 1) {
        await runtime.invoke(
          "observe_humanoid",
          {},
          `later-observation-${index}`,
          "perception-agent"
        );
      }

      expect(runtime.receipt(plan.transactionId)).toEqual(plan);
      expect(await runtime.invoke(
        "plan_whole_body_motion",
        input,
        plan.transactionId,
        "motion-agent"
      )).toEqual(plan);
      const execution = await runtime.invoke(
        "execute_whole_body_motion",
        { planning_transaction_id: plan.transactionId },
        "protected-current-execution",
        "executor-agent"
      );
      expect(execution).toMatchObject({
        accepted: true,
        code: "motion_completed"
      });
    } finally {
      await world.dispose();
    }
  }, 30_000);

  it("emits authoritative, idempotent receipts for whole-body and navigation actions", async () => {
    const world = await HumanoidWorld.create(scenario);
    const frames: number[] = [];
    const receipts: string[] = [];
    const runtime = new HumanoidActionRuntime(world, {
      frameSink: (snapshot) => {
        frames.push(snapshot.frame);
      },
      receiptSink: (receipt) => {
        receipts.push(receipt.transactionId);
      }
    });
    try {
      const observation = await runtime.invoke(
        "observe_humanoid",
        {},
        "observe-1",
        "perception-agent"
      );
      expect(observation.accepted).toBe(true);
      expect(observation.frameCount).toBe(0);
      const observationDetail = record(observation.detail);
      expect(Object.keys(record(observationDetail.joints))).toHaveLength(29);
      expect(record(observationDetail.key_links).head_link).toBeDefined();
      expect(record(observationDetail.key_links).pelvis).toBeDefined();
      expect(record(observationDetail.end_effectors)).toEqual(Object.fromEntries(
        HUMANOID_END_EFFECTORS.map((endEffector) => [endEffector, {
          world_position: humanoidEndEffectorPosition(
            world.snapshot().robot,
            endEffector,
            "world"
          ),
          pelvis_relative_position: humanoidEndEffectorPosition(
            world.snapshot().robot,
            endEffector,
            "pelvis"
          )
        }])
      ));
      expect(record(observationDetail.sensor).id).toBe("head_sensor");
      expect(observationDetail.object_tokens).toEqual([]);
      expect(runtime.receipt("observe-1")).toEqual(observation);

      const ungroundedExecution = await runtime.invoke(
        "execute_whole_body_motion",
        { planning_transaction_id: "missing-plan" },
        "execute-missing",
        "executor-agent"
      );
      expect(ungroundedExecution).toMatchObject({
        accepted: false,
        code: "planning_receipt_missing",
        frameCount: 0
      });
      const mismatchedExecution = await runtime.invoke(
        "execute_humanoid_navigation",
        { planning_transaction_id: "observe-1" },
        "execute-mismatch",
        "executor-agent"
      );
      expect(mismatchedExecution).toMatchObject({
        accepted: false,
        code: "planning_receipt_action_mismatch",
        frameCount: 0
      });

      const firstPlan = motionPlan("motion-a", runtime.snapshot(), 0.01);
      const [planned, repeated] = await Promise.all([
        runtime.invoke(
          "plan_whole_body_motion",
          firstPlan,
          "plan-a",
          "motion-agent"
        ),
        runtime.invoke(
          "plan_whole_body_motion",
          { ...firstPlan, keyframes: firstPlan.keyframes.map((frame) => ({ ...frame })) },
          "plan-a",
          "motion-agent"
        )
      ]);
      expect(planned).toEqual(repeated);
      expect(planned.accepted).toBe(true);
      expect(planned.channels).toContain("left_arm");
      expect(record(record(planned.detail).motion)).toMatchObject({
        protocol: "humanoid-motion-v1",
        generator: "task_space_constraints",
        frame_count: 25
      });
      expect(receipts.filter((id) => id === "plan-a")).toHaveLength(1);
      await expect(runtime.invoke(
        "observe_humanoid",
        {},
        "plan-a",
        "motion-agent"
      )).rejects.toThrow("transaction conflict");

      const secondPlan = await runtime.invoke(
        "plan_whole_body_motion",
        motionPlan("motion-b", runtime.snapshot(), 0.005),
        "plan-b",
        "motion-agent"
      );
      expect(secondPlan.accepted).toBe(true);

      const executed = await runtime.invoke(
        "execute_whole_body_motion",
        { planning_transaction_id: "plan-a" },
        "execute-a",
        "executor-agent"
      );
      expect(executed.accepted).toBe(true);
      expect(executed.code).toBe("motion_completed");
      expect(executed.frameCount).toBeGreaterThan(0);
      expect(executed.worldAfterRevision).toBeGreaterThan(executed.worldBeforeRevision);
      expect(frames).toHaveLength(executed.frameCount);
      expect(world.checkpoint().reference.jointTrackingWeights?.every((value) => (
        value === 0
      ))).toBe(true);

      const stale = await runtime.invoke(
        "execute_whole_body_motion",
        { planning_transaction_id: "plan-b" },
        "execute-b",
        "executor-agent"
      );
      expect(stale.accepted).toBe(false);
      expect(stale.code).toBe("plan_stale");
      expect(stale.frameCount).toBe(0);

      const candidateBefore = runtime.snapshot();
      const candidateTarget = {
        ...candidateBefore.robot.rootPosition,
        z: candidateBefore.robot.rootPosition.z + 0.04
      };
      const candidatePlan = await runtime.invoke(
        "plan_whole_body_motion_candidates",
        {
          objective: "比较模型提出的全身姿态候选并执行可行者",
          termination: {
            option_id: "runtime-forward-option",
            predicates: [{
              type: "root_near_point",
              body: null,
              end_effector: null,
              frame: null,
              object_id: null,
              solid_id: null,
              hand_surface: null,
              hand: null,
              grasp_contract_sha256: null,
              zone_id: null,
              target: candidateTarget,
              tolerance_m: 0.03,
              target_orientation: null,
              orientation_tolerance_rad: null,
              minimum_normal_force: null,
              expected: null
            }],
            stable_steps: 2,
            phases: null
          },
          candidates: [
            {
              id: "runtime-noop-candidate",
              intent: "不实现当前前进目标",
              duration_seconds: 0.8,
              keyframes: [{ at_seconds: 0 }, { at_seconds: 0.8 }]
            },
            {
              id: "runtime-grounded-candidate",
              intent: "保持支撑并连续前进",
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
        "candidate-plan",
        "motion-agent"
      );
      expect(candidatePlan, JSON.stringify(candidatePlan.detail)).toMatchObject({
        accepted: true,
        code: "whole_body_candidates_validated"
      });
      expect(record(candidatePlan.detail)).toMatchObject({
        plan_id: "runtime-grounded-candidate",
        selected_candidate_id: "runtime-grounded-candidate",
        selected_rank: 2,
        candidate_count: 2,
        selection: "model_rank_then_physics"
      });
      const candidateExecution = await runtime.invoke(
        "execute_whole_body_motion",
        { planning_transaction_id: candidatePlan.transactionId },
        "candidate-execution",
        "executor-agent"
      );
      expect(candidateExecution.accepted).toBe(true);
      expect(candidateExecution.code).toBe("motion_option_succeeded");
      expect(record(candidateExecution.detail)).toMatchObject({
        planning_action: "plan_whole_body_motion_candidates",
        candidate_count: 2,
        selected_rank: 2,
        selected_candidate_id: "runtime-grounded-candidate"
      });

      const beforeNavigation = runtime.snapshot();
      const route = await runtime.invoke(
        "plan_humanoid_navigation",
        {
          target: {
            x: beforeNavigation.robot.rootPosition.x,
            y: 0,
            z: beforeNavigation.robot.rootPosition.z + 0.55
          }
        },
        "route-1",
        "navigation-agent"
      );
      expect(route.accepted, JSON.stringify(route.detail)).toBe(true);
      const routeDetail = record(route.detail);
      expect(typeof routeDetail.plan_id).toBe("string");

      const frameCountBeforeRoute = frames.length;
      const navigated = await runtime.invoke(
        "execute_humanoid_navigation",
        { planning_transaction_id: "route-1" },
        "navigate-1",
        "executor-agent"
      );
      expect(navigated.accepted).toBe(true);
      expect(navigated.code).toBe("navigation_completed");
      expect(navigated.frameCount).toBeGreaterThan(0);
      expect(frames.length - frameCountBeforeRoute).toBe(navigated.frameCount);
      expect(runtime.snapshot().robot.fallen).toBe(false);
      expect(runtime.snapshot().robot.balance.support).not.toBe("none");
    } finally {
      await world.dispose();
    }
  }, 60_000);
});

function motionPlan(
  id: string,
  snapshot: ReturnType<HumanoidActionRuntime["snapshot"]>,
  lift: number
): {
  id: string;
  intent: string;
  duration_seconds: number;
  keyframes: Array<{
    at_seconds: number;
    left_hand?: {
      position: { x: number; y: number; z: number };
      frame: "world";
      tolerance_m: number;
    };
  }>;
} {
  const wrist = snapshot.robot.links.left_wrist_yaw_link.position;
  return {
    id,
    intent: "保持平衡并连续调整左臂",
    duration_seconds: 0.5,
    keyframes: [
      { at_seconds: 0 },
      {
        at_seconds: 0.5,
        left_hand: {
          position: { ...wrist, y: wrist.y + lift },
          frame: "world",
          tolerance_m: 0.045
        }
      }
    ]
  };
}

function record(value: JsonValue | undefined): Record<string, JsonValue> {
  if (!value || Array.isArray(value) || typeof value !== "object") {
    throw new Error("Expected a JSON object");
  }
  return value;
}

function lightweightObservationWorld(): {
  world: HumanoidWorld;
  observationCalls: () => number;
} {
  let observationCalls = 0;
  const world = {
    snapshot: () => ({ frame: 0, worldRevision: 0 }),
    consumablePlanIds: () => [],
    observe: () => {
      observationCalls += 1;
      return {
        frame: 0,
        worldRevision: 0,
        sensor: {
          position: { x: 0, y: 1, z: 0 },
          rotation: { x: 0, y: 0, z: 0, w: 1 },
          maximumRange: 1,
          horizontalFieldOfView: 1,
          verticalFieldOfView: 1
        },
        robot: {
          controller: {},
          rootPosition: { x: 0, y: 0, z: 0 },
          rootRotation: { x: 0, y: 0, z: 0, w: 1 },
          fallen: false,
          balance: {},
          feet: {},
          joints: {},
          links: {},
          contacts: []
        },
        objectTokens: [],
        solidTokens: [],
        grasp: {
          contractSha256: "fc1e2d113bb5e5f5f8a75f0faa3efc8bd97ecc18eb41463da09d26bb52cfc193",
          assessments: []
        },
        navigation: {}
      };
    }
  } as unknown as HumanoidWorld;
  return { world, observationCalls: () => observationCalls };
}
