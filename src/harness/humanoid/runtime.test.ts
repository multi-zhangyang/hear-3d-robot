import { describe, expect, it } from "vitest";
import {
  HUMANOID_END_EFFECTORS,
  ScenarioSchema,
  type JsonValue
} from "../../domain/schema.js";
import { yawFromQuaternion } from "../../world/geometry.js";
import { humanoidEndEffectorPosition } from "../../world/humanoid/end-effectors.js";
import { HumanoidWorld } from "../../world/humanoid/world.js";
import { MAX_CHECKPOINT_ACTION_RECEIPTS } from "./embodied-memory.js";
import {
  HumanoidActionRuntime,
  humanoidActionFingerprint,
  type HumanoidActionReceipt
} from "./runtime.js";
import type { ScenarioBlockRemovalTransaction } from "../../domain/scenario-block-removal.js";
import { applyScenarioChunkDeltaMutation } from "../../domain/scenario-chunk-delta.js";
import { createScenarioChunkDeltaState } from "../../domain/scenario-chunk-delta-schema.js";
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
  it("enforces observe, Skill binding, and referenced low-level planning in production mode", async () => {
    const lightweight = lightweightObservationWorld(undefined, {
      interaction: {
        object_world_model: { frame: 0, world_revision: 0, objects: [] },
        skill_catalog: {
          contract_sha256: "a".repeat(64),
          frame: 0,
          world_revision: 0,
          entries: []
        },
        carrying: { bindings: [] },
        manipulable_objects: []
      }
    });
    const runtime = new HumanoidActionRuntime(lightweight.world, {
      requireSkillBinding: true
    });
    const observation = await runtime.invoke(
      "observe_humanoid",
      {},
      "skill-observation",
      "humanoid-motion-reference"
    );
    expect(observation.detail).toMatchObject({
      control_authority: {
        physics_backend: "mujoco",
        learned_policy: {
          protocol: "humanoid-learned-policy-v1",
          capabilities: ["balance", "locomotion", "joint_reference_tracking"]
        },
        reference_control: null,
        motion_generator: { implementation: "test_task_space_generator" }
      }
    });
    expect(runtime.isActionAvailable(
      "begin_humanoid_skill",
      "humanoid-motion-reference"
    )).toBe(false);
    expect(runtime.isActionAvailable(
      "submit_humanoid_skill_plan",
      "humanoid-motion-reference"
    )).toBe(true);
    expect(runtime.isActionAvailable(
      "plan_humanoid_skill",
      "humanoid-motion-reference"
    )).toBe(false);
    const candidate = {
      skill_transaction_id: null,
      objective: "restore support",
      termination: {
        mode: "all" as const,
        option_id: "stabilize",
        predicates: [{
          type: "balance_stable" as const,
          minimum_support_margin_m: 0.02
        }],
        stable_steps: 1
      },
      candidates: [{
        id: "stabilize-current",
        intent: "restore a balanced support posture",
        duration_seconds: 0.5,
        contacts: [],
        keyframes: [
          { at_seconds: 0, channels: [] },
          { at_seconds: 0.5, channels: [] }
        ]
      }]
    };
    await expect(runtime.invoke(
      "plan_whole_body_motion_candidates",
      candidate,
      "unbound-skill-plan",
      "humanoid-motion-reference"
    )).resolves.toMatchObject({ accepted: false, code: "active_skill_required" });

    const skillPlan = await runtime.invoke(
      "submit_humanoid_skill_plan",
      {
        objective: "restore physical support",
        strategies: [{
          strategy_id: "support-recovery",
          rationale: "recover support before further motion",
          nodes: [{
            node_id: "stabilize-now",
            invocation: {
              skill: "stabilize",
              minimum_support_margin_m: 0.02
            },
            depends_on_node_ids: []
          }]
        }],
        selected_strategy_id: "support-recovery"
      },
      "stabilize-plan",
      "humanoid-motion-reference"
    );
    expect(skillPlan).toMatchObject({
      accepted: true,
      code: "humanoid_skill_plan_registered"
    });
    expect(runtime.isActionAvailable(
      "submit_humanoid_skill_plan",
      "humanoid-motion-reference"
    )).toBe(false);
    expect(runtime.isActionAvailable(
      "begin_humanoid_skill",
      "humanoid-motion-reference"
    )).toBe(true);
    expect(runtime.isActionAvailable(
      "plan_humanoid_skill",
      "humanoid-motion-reference"
    )).toBe(false);
    const skill = await runtime.invoke(
      "begin_humanoid_skill",
      {
        skill_plan_transaction_id: "stabilize-plan",
        skill_node_id: "stabilize-now",
        invocation: {
          skill: "stabilize",
          minimum_support_margin_m: 0.02
        },
        phase: "recover_support"
      },
      "stabilize-skill",
      "humanoid-motion-reference"
    );
    expect(skill).toMatchObject({ accepted: true, code: "humanoid_skill_bound" });
    expect(runtime.isActionAvailable(
      "submit_humanoid_skill_plan",
      "humanoid-motion-reference"
    )).toBe(false);
    expect(runtime.isActionAvailable(
      "begin_humanoid_skill",
      "humanoid-motion-reference"
    )).toBe(false);
    expect(runtime.isActionAvailable(
      "plan_humanoid_skill",
      "humanoid-motion-reference"
    )).toBe(true);
    expect(runtime.planningToolState("humanoid-motion-reference")).toMatchObject({
      active_skill: {
        transaction_id: "stabilize-skill",
        skill_plan_transaction_id: "stabilize-plan",
        skill_node_id: "stabilize-now",
        phase: "recover_support",
        phase_authority: "whole_body",
        planning_action: "plan_humanoid_skill"
      }
    });
    await expect(runtime.invoke(
      "plan_humanoid_skill",
      { skill_transaction_id: "stabilize-skill" },
      "bound-skill-plan",
      "humanoid-motion-reference"
    )).resolves.toMatchObject({
      code: "autonomous_skill_motion_rejected",
      detail: {
        skill_binding: {
          transaction_id: "stabilize-skill",
          invocation: { skill: "stabilize" }
        }
      }
    });
    expect(lightweight.candidatePlanningCalls()).toBe(1);
    expect(runtime.isActionAvailable(
      "submit_humanoid_skill_plan",
      "humanoid-motion-reference"
    )).toBe(true);
    expect(runtime.isActionAvailable(
      "begin_humanoid_skill",
      "humanoid-motion-reference"
    )).toBe(false);

    await expect(runtime.invoke(
      "submit_humanoid_skill_plan",
      {
        objective: "replace the rejected local Skill strategy",
        strategies: [{
          strategy_id: "replacement-support-recovery",
          rationale: "the previous low-level physical plan was rejected",
          nodes: [{
            node_id: "replacement-stabilize",
            invocation: {
              skill: "stabilize",
              minimum_support_margin_m: 0.03
            },
            depends_on_node_ids: []
          }]
        }],
        selected_strategy_id: "replacement-support-recovery"
      },
      "replacement-stabilize-plan",
      "humanoid-motion-reference"
    )).resolves.toMatchObject({
      accepted: true,
      code: "humanoid_skill_plan_registered"
    });
    expect(runtime.isActionAvailable(
      "submit_humanoid_skill_plan",
      "humanoid-motion-reference"
    )).toBe(false);
    expect(runtime.isActionAvailable(
      "begin_humanoid_skill",
      "humanoid-motion-reference"
    )).toBe(true);
    expect(runtime.planningToolState("humanoid-motion-reference")).toMatchObject({
      active_skill: null,
      skill_plan: {
        transaction_id: "replacement-stabilize-plan",
        selected_strategy_id: "replacement-support-recovery"
      }
    });
  });

  it("exposes an independent reference-control route without merging trained capabilities", async () => {
    const lightweight = lightweightObservationWorld(
      undefined,
      {},
      0,
      {},
      true
    );
    const runtime = new HumanoidActionRuntime(lightweight.world);
    const observation = await runtime.invoke(
      "observe_humanoid",
      {},
      "routed-observation",
      "humanoid-motion-reference"
    );
    expect(observation.detail).toMatchObject({
      control_authority: {
        learned_policy: { capabilities: ["balance", "locomotion"] },
        reference_control: {
          protocol: "humanoid-controller-capability-routing-v1",
          strategy: "declared_capabilities",
          mode: "reference_control",
          implementation: "yahmp_onnx"
        }
      }
    });
  });

  it("restores Recovery Policy and still requires a model-selected recovery Skill", async () => {
    const lightweight = lightweightObservationWorld(undefined, {
      interaction: {
        object_world_model: { frame: 0, world_revision: 0, objects: [] },
        skill_catalog: {
          contract_sha256: "a".repeat(64),
          frame: 0,
          world_revision: 0,
          entries: []
        },
        carrying: { bindings: [] },
        manipulable_objects: []
      }
    });
    const recoveryPolicy = {
      protocol: "humanoid-recovery-policy-v1" as const,
      source_execution_transaction_id: "failed-grasp-execution",
      source_planning_transaction_id: "failed-grasp-plan",
      source_skill_transaction_id: "failed-grasp-skill",
      source_skill: "grasp" as const,
      source_phase: "close_hand_under_contact",
      physical_failure_code: "motion_goal_unmet",
      failure_reason: "grasp_unstable" as const,
      world_revision: 0,
      candidate_skills: ["regrasp", "reach"] as const,
      requires_model_selection: true as const,
      automatic_actuation: false as const
    };
    const runtime = new HumanoidActionRuntime(lightweight.world, {
      requireSkillBinding: true,
      state: {
        version: 1,
        latest_physical_execution_revision: 0,
        skill_plans: [],
        active_skill_plan_transactions: {},
        active_skills: [],
        planning_skill_bindings: [],
        recovery_policies: [{
          agent_id: "humanoid-motion-reference",
          policy: recoveryPolicy
        }]
      }
    });

    expect(runtime.planningToolState("humanoid-motion-reference")).toMatchObject({
      recovery_policy: recoveryPolicy
    });
    await runtime.invoke(
      "observe_humanoid",
      {},
      "recovery-observation",
      "humanoid-motion-reference"
    );
    const proposal = {
      objective: "select the next recovery Skill",
      strategies: [{
        strategy_id: "unsupported-recovery",
        rationale: "exercise restored Recovery Policy authority",
        nodes: [{
          node_id: "stabilize-instead",
          invocation: { skill: "stabilize" as const, minimum_support_margin_m: 0 },
          depends_on_node_ids: []
        }]
      }],
      selected_strategy_id: "unsupported-recovery"
    };
    await runtime.invoke(
      "submit_humanoid_skill_plan",
      proposal,
      "recovery-skill-plan",
      "humanoid-motion-reference"
    );
    await expect(runtime.invoke(
      "begin_humanoid_skill",
      {
        skill_plan_transaction_id: "recovery-skill-plan",
        skill_node_id: "stabilize-instead",
        invocation: proposal.strategies[0]!.nodes[0]!.invocation,
        phase: "recover_support"
      },
      "unsupported-recovery-binding",
      "humanoid-motion-reference"
    )).resolves.toMatchObject({
      accepted: false,
      code: "recovery_skill_selection_required",
      detail: {
        automatic_actuation: false,
        recovery_policy: recoveryPolicy,
        selected_skill: "stabilize"
      }
    });
  });

  it("retires stale Recovery Policy after observing a changed world", async () => {
    const lightweight = lightweightObservationWorld(undefined, {
      interaction: {
        object_world_model: { frame: 0, world_revision: 1, objects: [] },
        skill_catalog: {
          contract_sha256: "a".repeat(64),
          frame: 0,
          world_revision: 1,
          entries: []
        },
        carrying: { bindings: [] },
        manipulable_objects: []
      }
    }, 1);
    const runtime = new HumanoidActionRuntime(lightweight.world, {
      requireSkillBinding: true,
      state: {
        version: 1,
        latest_physical_execution_revision: 0,
        skill_plans: [],
        active_skill_plan_transactions: {},
        active_skills: [],
        planning_skill_bindings: [],
        recovery_policies: [{
          agent_id: "humanoid-motion-reference",
          policy: {
            protocol: "humanoid-recovery-policy-v1",
            source_execution_transaction_id: "old-execution",
            source_planning_transaction_id: "old-plan",
            source_skill_transaction_id: "old-skill",
            source_skill: "grasp",
            source_phase: "close_hand_under_contact",
            physical_failure_code: "motion_goal_unmet",
            failure_reason: "grasp_unstable",
            world_revision: 0,
            candidate_skills: ["regrasp", "reach"],
            requires_model_selection: true,
            automatic_actuation: false
          }
        }]
      }
    });

    expect(runtime.planningToolState("humanoid-motion-reference"))
      .toMatchObject({ recovery_policy: { world_revision: 0 } });
    await runtime.invoke(
      "observe_humanoid",
      {},
      "changed-world-recovery-observation",
      "humanoid-motion-reference"
    );
    expect(runtime.planningToolState("humanoid-motion-reference"))
      .toMatchObject({ recovery_policy: null });
  });

  it("renews Skill authority across stationary physics but rejects material world changes", async () => {
    const world = await HumanoidWorld.create(scenario);
    try {
      const runtime = new HumanoidActionRuntime(world, {
        requireSkillBinding: true
      });
      await runtime.invoke(
        "observe_humanoid",
        {},
        "lease-observation",
        "humanoid-motion-reference"
      );
      const invocation = {
        skill: "stabilize" as const,
        minimum_support_margin_m: 0
      };
      const proposal = {
        objective: "maintain stable physical support",
        strategies: [{
          strategy_id: "stationary-lease",
          rationale: "continue from an unchanged physical observation",
          nodes: [{
            node_id: "stabilize-now",
            invocation,
            depends_on_node_ids: []
          }]
        }],
        selected_strategy_id: "stationary-lease"
      };
      await runtime.invoke(
        "submit_humanoid_skill_plan",
        proposal,
        "lease-skill-plan",
        "humanoid-motion-reference"
      );
      const binding = await runtime.invoke(
        "begin_humanoid_skill",
        {
          skill_plan_transaction_id: "lease-skill-plan",
          skill_node_id: "stabilize-now",
          invocation,
          phase: "recover_support"
        },
        "lease-skill-binding",
        "humanoid-motion-reference"
      );
      expect(binding.accepted).toBe(true);

      const stationary = await world.advanceStationary();
      expect(stationary?.worldRevision).toBeGreaterThan(0);
      const renewed = await runtime.invoke(
        "plan_humanoid_skill",
        { skill_transaction_id: binding.transactionId },
        "stationary-lease-plan",
        "humanoid-motion-reference"
      );
      expect(renewed.code).not.toBe("skill_observation_changed");
      expect(renewed.detail).toMatchObject({
        skill_binding: {
          transaction_id: binding.transactionId,
          observed_world_revision: stationary!.worldRevision
        }
      });

      const chunks = applyScenarioChunkDeltaMutation(
        scenario,
        createScenarioChunkDeltaState(scenario),
        {
          type: "create_block",
          block: {
            id: "lease-material-change",
            center: { x: 1.5, y: 1.3, z: 2.5 },
            size: { x: 0.4, y: 0.4, z: 0.4 },
            material: "stone",
            properties: {}
          }
        }
      );
      await world.synchronizeScenarioChunks(scenario, chunks);
      expect(world.observe().solidTokens.map(({ id }) => id)).toContain(
        "lease-material-change"
      );
      await expect(runtime.invoke(
        "plan_humanoid_skill",
        { skill_transaction_id: binding.transactionId },
        "changed-world-plan",
        "humanoid-motion-reference"
      )).resolves.toMatchObject({
        accepted: false,
        code: "skill_observation_changed",
        detail: { automatic_actuation: false }
      });
    } finally {
      await world.dispose();
    }
  }, 30_000);

  it("requires the Motion Agent to own a fresh observation before planning", async () => {
    const lightweight = lightweightObservationWorld();
    const runtime = new HumanoidActionRuntime(lightweight.world);
    const input = {
      target: { x: 0.15, y: 0, z: 0.3 },
      arrival_heading: null
    };

    expect(runtime.isActionAvailable(
      "plan_humanoid_navigation",
      "humanoid-motion-reference"
    )).toBe(false);
    expect(runtime.isActionAvailable(
      "observe_humanoid",
      "humanoid-motion-reference"
    )).toBe(true);

    await expect(runtime.invoke(
      "plan_humanoid_navigation",
      input,
      "stale-motion-plan",
      "humanoid-motion-reference"
    )).resolves.toMatchObject({
      accepted: false,
      code: "fresh_motion_observation_required",
      detail: {
        observed_world_revision: null,
        latest_physical_execution_revision: 0,
        automatic_actuation: false
      }
    });
    await expect(runtime.invoke(
      "observe_humanoid",
      {},
      "motion-observation",
      "humanoid-motion-reference"
    )).resolves.toMatchObject({ accepted: true, code: "humanoid_observed" });
    expect(runtime.isActionAvailable(
      "plan_humanoid_navigation",
      "humanoid-motion-reference"
    )).toBe(true);
    expect(runtime.isActionAvailable(
      "observe_humanoid",
      "humanoid-motion-reference"
    )).toBe(false);
    await expect(runtime.invoke(
      "plan_humanoid_navigation",
      input,
      "fresh-motion-plan",
      "humanoid-motion-reference"
    )).resolves.not.toMatchObject({ code: "fresh_motion_observation_required" });
    expect(runtime.isActionAvailable(
      "observe_humanoid",
      "humanoid-motion-reference"
    )).toBe(true);
  });

  it("requires new reachability evidence after restoring a Motion observation", async () => {
    const lightweight = lightweightObservationWorld();
    const initial = new HumanoidActionRuntime(lightweight.world);
    const observation = await initial.invoke(
      "observe_humanoid",
      {},
      "pre-restart-motion-observation",
      "humanoid-motion-reference"
    );
    const restored = new HumanoidActionRuntime(lightweight.world, {
      receipts: { [observation.transactionId]: observation }
    });

    expect(restored.isActionAvailable(
      "plan_humanoid_navigation",
      "humanoid-motion-reference"
    )).toBe(false);
    await expect(restored.invoke(
      "plan_humanoid_navigation",
      { target: { x: 0.15, y: 0, z: 0.3 }, arrival_heading: null },
      "post-restart-plan",
      "humanoid-motion-reference"
    )).resolves.toMatchObject({
      accepted: false,
      code: "fresh_motion_observation_required"
    });
  });

  it("blocks a repeated physical input without hiding new planning strategies", async () => {
    const lightweight = lightweightObservationWorld();
    const runtime = new HumanoidActionRuntime(lightweight.world);
    const input = {
      target: { x: 0.15, y: 0, z: 0.3 },
      arrival_heading: null
    };

    await runtime.invoke(
      "observe_humanoid",
      {},
      "loop-observation",
      "humanoid-motion-reference"
    );
    const first = await runtime.invoke(
      "plan_humanoid_navigation",
      input,
      "loop-plan-1",
      "humanoid-motion-reference"
    );
    const second = await runtime.invoke(
      "plan_humanoid_navigation",
      {
        ...input,
        target: { x: 0.1504, y: 0.004, z: 0.2996 }
      },
      "loop-plan-2",
      "humanoid-motion-reference"
    );

    expect(first).toMatchObject({ accepted: false, code: "humanoid_route_rejected" });
    expect(second).toMatchObject({ accepted: false, code: "humanoid_route_rejected" });
    expect(runtime.isActionAvailable(
      "plan_humanoid_navigation",
      "humanoid-motion-reference"
    )).toBe(true);
    expect(runtime.isActionAvailable(
      "plan_whole_body_motion_candidates",
      "humanoid-motion-reference"
    )).toBe(true);
    await expect(runtime.invoke(
      "plan_humanoid_navigation",
      {
        ...input,
        target: { x: 0.18, y: 0, z: 0.3 }
      },
      "loop-physically-distinct-navigation",
      "humanoid-motion-reference"
    )).resolves.toMatchObject({
      accepted: false,
      code: "humanoid_route_rejected"
    });
    await expect(runtime.invoke(
      "plan_humanoid_navigation",
      input,
      "loop-plan-3",
      "humanoid-motion-reference"
    )).resolves.toMatchObject({
      accepted: false,
      code: "repeated_planning_failure",
      detail: {
        repeated_action: "plan_humanoid_navigation",
        repeated_failure_count: 2,
        previous_code: "humanoid_route_rejected",
        automatic_actuation: false
      }
    });
    expect(runtime.isActionAvailable(
      "plan_humanoid_navigation",
      "humanoid-motion-reference"
    )).toBe(false);
    expect(runtime.planningToolState("humanoid-motion-reference")).toMatchObject({
      planning_actions: expect.arrayContaining([{
        action: "plan_humanoid_navigation",
        available: false
      }]),
      cooldown: {
        action: "plan_humanoid_navigation",
        code: "repeated_planning_failure",
        repeated_failure_count: 3,
        previous_code: "humanoid_route_rejected"
      }
    });

    const alternativeInput = {
      objective: "try a physically distinct whole-body strategy",
      termination: {
        mode: "all" as const,
        option_id: "break-navigation-loop",
        predicates: [{
          type: "root_near_point" as const,
          target: { x: 0.05, y: 0, z: 0.05 },
          tolerance_m: 0.1
        }],
        stable_steps: 1
      },
      candidates: [{
        id: "break-navigation-loop-candidate",
        intent: "hold the current physical state",
        duration_seconds: 0.5,
        contacts: [],
        keyframes: [
          { at_seconds: 0, channels: [] },
          { at_seconds: 0.5, channels: [] }
        ]
      }]
    };
    await runtime.invoke(
      "plan_whole_body_motion_candidates",
      alternativeInput,
      "loop-alternative-plan-1",
      "humanoid-motion-reference"
    );
    await runtime.invoke(
      "plan_whole_body_motion_candidates",
      alternativeInput,
      "loop-alternative-plan-2",
      "humanoid-motion-reference"
    );
    await expect(runtime.invoke(
      "plan_whole_body_motion_candidates",
      {
        ...alternativeInput,
        objective: "rename the same strategy",
        termination: {
          ...alternativeInput.termination,
          option_id: "renamed-break-navigation-loop"
        },
        candidates: alternativeInput.candidates.map((candidate) => ({
          ...candidate,
          id: "renamed-break-navigation-loop-candidate",
          intent: "rename without changing the physical motion"
        }))
      },
      "loop-alternative-plan-3",
      "humanoid-motion-reference"
    )).resolves.toMatchObject({
      accepted: false,
      code: "repeated_planning_failure",
      detail: {
        repeated_action: "plan_whole_body_motion_candidates",
        repeated_failure_count: 2,
        previous_code: "whole_body_candidates_rejected",
        automatic_actuation: false
      }
    });
    expect(lightweight.candidatePlanningCalls()).toBe(2);
    expect(runtime.isActionAvailable(
      "plan_humanoid_navigation",
      "humanoid-motion-reference"
    )).toBe(true);

    await expect(runtime.invoke(
      "plan_whole_body_motion_candidates",
      {
        ...alternativeInput,
        termination: {
          ...alternativeInput.termination,
          predicates: [{
            ...alternativeInput.termination.predicates[0],
            target: { x: 0.08, y: 0, z: 0.05 }
          }]
        }
      },
      "loop-physically-distinct-plan",
      "humanoid-motion-reference"
    )).resolves.toMatchObject({
      accepted: false,
      code: "whole_body_candidates_rejected"
    });
    expect(lightweight.candidatePlanningCalls()).toBe(3);
  });

  it("rejects a stale near-object stand position when live IK requires a sampled base", async () => {
    const objectCenter = { x: 0.2, y: 0.67, z: 0.8 };
    const placement = {
      objectId: "workpiece",
      handSurface: "left_hand_middle_1_link" as const,
      rootWorldTarget: { x: 0.08, y: 0.76, z: 0.46 },
      rootTranslationWorld: { x: 0.08, y: 0, z: 0.12 },
      rootYawRadians: 0.18,
      wristWorldTarget: { x: 0.16, y: 0.78, z: 0.73 },
      ikResidualMeters: 0.004
    };
    const lightweight = lightweightObservationWorld(undefined, {
      objectTokens: [visibleWorkpieceToken(objectCenter)],
      manipulationReachability: [{
        objectId: "workpiece",
        handSurface: "left_hand_middle_1_link",
        wristWorldTarget: placement.wristWorldTarget,
        ikReferenceReachable: false,
        ikResidualMeters: 0.2
      }],
      manipulationBasePlacements: [placement]
    });
    const runtime = new HumanoidActionRuntime(lightweight.world);

    await runtime.invoke(
      "observe_humanoid",
      {},
      "reachability-observation",
      "humanoid-motion-reference"
    );
    await expect(runtime.invoke(
      "plan_whole_body_motion_candidates",
      {
        objective: "contact the observed workpiece",
        termination: {
          mode: "all",
          option_id: "contact-workpiece",
          predicates: [{
            type: "hand_contact_object",
            hand_surface: placement.handSurface,
            object_id: "workpiece",
            minimum_normal_force: 1
          }],
          stable_steps: 1
        },
        candidates: [{
          id: "unreachable-contact",
          intent: "reach the workpiece from the current root pose",
          duration_seconds: 1,
          contacts: [{
            type: "hand_object",
            hand_surface: placement.handSurface,
            object_id: "workpiece",
            required: true
          }],
          keyframes: [
            { at_seconds: 0, channels: [] },
            {
              at_seconds: 1,
              channels: [{
                type: "end_effector_position",
                end_effector: "left_wrist",
                frame: "world",
                position: placement.wristWorldTarget,
                tolerance_m: 0.02
              }]
            }
          ]
        }]
      },
      "unreachable-endpoint-plan",
      "humanoid-motion-reference"
    )).resolves.toMatchObject({
      accepted: false,
      code: "manipulation_base_placement_required",
      detail: {
        automatic_actuation: false,
        requested_plan_ids: ["unreachable-contact"],
        reachable_base_placements: [{
          object_id: "workpiece",
          root_world_target: placement.rootWorldTarget
        }]
      }
    });
    await expect(runtime.invoke(
      "plan_whole_body_motion_candidates",
      {
        objective: "approach the workpiece while authorizing exact hand contact",
        termination: {
          mode: "all",
          option_id: "mobile-contact-workpiece",
          predicates: [{
            type: "hand_contact_object",
            hand_surface: placement.handSurface,
            object_id: "workpiece",
            minimum_normal_force: 1
          }],
          stable_steps: 1
        },
        candidates: [{
          id: "contact-guided-mobile-approach",
          intent: "translate the root into the observed contact region",
          duration_seconds: 0.5,
          contacts: [{
            type: "hand_object",
            hand_surface: placement.handSurface,
            object_id: "workpiece",
            required: true
          }],
          keyframes: [{
            at_seconds: 0,
            channels: [{
              type: "root_velocity",
              forward_mps: 0.2,
              lateral_mps: 0
            }]
          }, {
            at_seconds: 0.5,
            channels: [{
              type: "root_velocity",
              forward_mps: 0.2,
              lateral_mps: 0
            }]
          }]
        }]
      },
      "mobile-contact-plan",
      "humanoid-motion-reference"
    )).resolves.toMatchObject({
      accepted: false,
      code: "whole_body_candidates_rejected"
    });
    expect(lightweight.candidatePlanningCalls()).toBe(1);

    await expect(runtime.invoke(
      "plan_whole_body_motion_candidates",
      {
        objective: "approach with an unrelated hand surface",
        termination: {
          mode: "all",
          option_id: "unrelated-mobile-contact-workpiece",
          predicates: [{
            type: "hand_contact_object",
            hand_surface: "left_hand_index_1_link",
            object_id: "workpiece",
            minimum_normal_force: 1
          }],
          stable_steps: 1
        },
        candidates: [{
          id: "unrelated-contact-mobile-approach",
          intent: "attempt to bypass the sampled contact surface",
          duration_seconds: 0.5,
          contacts: [{
            type: "hand_object",
            hand_surface: "left_hand_index_1_link",
            object_id: "workpiece",
            required: true
          }],
          keyframes: [{
            at_seconds: 0,
            channels: [{
              type: "root_velocity",
              forward_mps: 0.2,
              lateral_mps: 0
            }]
          }, {
            at_seconds: 0.5,
            channels: [{
              type: "root_velocity",
              forward_mps: 0.2,
              lateral_mps: 0
            }]
          }]
        }]
      },
      "unrelated-mobile-contact-plan",
      "humanoid-motion-reference"
    )).resolves.toMatchObject({
      accepted: false,
      code: "manipulation_base_placement_required",
      detail: {
        requested_plan_ids: ["unrelated-contact-mobile-approach"]
      }
    });
    expect(lightweight.candidatePlanningCalls()).toBe(1);
    await expect(runtime.invoke(
      "plan_humanoid_navigation",
      {
        target: { x: 0.1, y: 0, z: 0.35 },
        arrival_heading: {
          type: "face_point",
          target: objectCenter,
          tolerance_radians: 0.25
        }
      },
      "stale-object-approach",
      "humanoid-motion-reference"
    )).resolves.toMatchObject({
      accepted: false,
      code: "manipulation_base_placement_required",
      detail: {
        automatic_actuation: false,
        observed_world_revision: 0,
        reachable_base_placements: [{
          object_id: "workpiece",
          root_world_target: placement.rootWorldTarget,
          root_yaw_radians: placement.rootYawRadians,
          ik_residual_m: placement.ikResidualMeters,
          navigation_validation_required: true
        }]
      }
    });

    await expect(runtime.invoke(
      "plan_humanoid_navigation",
      {
        target: placement.rootWorldTarget,
        arrival_heading: {
          type: "yaw",
          yaw_radians: placement.rootYawRadians,
          tolerance_radians: 0.08
        }
      },
      "sampled-object-approach",
      "humanoid-motion-reference"
    )).resolves.not.toMatchObject({
      code: "manipulation_base_placement_required"
    });
  });

  it("preserves a rejected navigation preview endpoint as structured evidence", async () => {
    const lightweight = lightweightObservationWorld({
      accepted: false,
      planId: "",
      createdRevision: 0,
      validatedStateSha256: "a".repeat(64),
      expiresRevision: 100,
      intentSha256: "b".repeat(64),
      target: { x: 0.15, y: 0, z: 0.3 },
      chunkTarget: { x: 0.15, y: 0, z: 0.3 },
      requestedArrivalHeading: null,
      arrivalHeading: null,
      waypoints: [{ x: 0.15, y: 0, z: 0.3 }],
      distance: 0.12,
      remainingDistance: 0,
      partialEndpoint: { x: 0.08, y: 0.01, z: 0.27 },
      previewFrames: 611,
      previewTravelledDistance: 0.041,
      carry: { binding_set_sha256: "c".repeat(64), bindings: [] },
      reason: "navigation_timeout"
    });
    const runtime = new HumanoidActionRuntime(lightweight.world);

    const receipt = await runtime.invoke(
      "plan_humanoid_navigation",
      { target: { x: 0.15, y: 0, z: 0.3 }, arrival_heading: null },
      "partial-route",
      "motion-agent"
    );

    expect(receipt).toMatchObject({
      accepted: false,
      code: "humanoid_route_rejected",
      frameCount: 0,
      detail: {
        partial_endpoint: { x: 0.08, y: 0.01, z: 0.27 },
        preview_frames: 611,
        preview_travelled_m: 0.041,
        reason: "navigation_timeout"
      }
    });
  });

  it("turns Skill navigation hand collisions into a bounded transit-clearance state", () => {
    const transactionId = "skill-navigation-hand-collision";
    const input = { skill_transaction_id: "explore-skill" };
    const receipt: HumanoidActionReceipt = {
      transactionId,
      agentId: "humanoid-motion-reference",
      action: "plan_humanoid_skill",
      input,
      fingerprint: humanoidActionFingerprint(
        "plan_humanoid_skill",
        "humanoid-motion-reference",
        input
      ),
      accepted: false,
      code: "autonomous_skill_route_rejected",
      worldBeforeRevision: 42,
      worldAfterRevision: 42,
      frameCount: 0,
      channels: ["locomotion"],
      detail: {
        autonomous_plan_kind: "navigation",
        failure_class: "path_or_physical_preview_infeasible",
        attempts: [{
          target: { x: 4, y: 0, z: 5 },
          score: 3.5,
          accepted: false,
          reason: "environment_contact:left_hand_index_1_link:stone-column; preview_frames=353"
        }]
      },
      committedAt: "2026-08-08T00:00:00.000Z"
    };
    const world = {
      snapshot: () => ({
        frame: 10,
        worldRevision: 42,
        robot: {
          links: {
            left_wrist_yaw_link: {
              position: { x: 1, y: 0.9, z: 2 },
              rotation: { x: 0, y: 0, z: 0, w: 1 }
            },
            left_ankle_roll_link: {
              position: { x: 0.9, y: 0.05, z: 2 },
              rotation: { x: 0, y: 0, z: 0, w: 1 }
            },
            right_ankle_roll_link: {
              position: { x: 1.1, y: 0.05, z: 2 },
              rotation: { x: 0, y: 0, z: 0, w: 1 }
            }
          },
          objects: {
            "stone-column": {
              id: "stone-column",
              position: { x: 1.2, y: 0.8, z: 2.1 }
            }
          }
        }
      }),
      consumablePlanIds: () => []
    } as unknown as HumanoidWorld;

    const runtime = new HumanoidActionRuntime(world, {
      receipts: { [transactionId]: receipt }
    });

    expect(runtime.planningToolState("humanoid-motion-reference")).toMatchObject({
      transit_clearance: {
        status: "required",
        blocked_action: "plan_humanoid_skill",
        source_transaction_id: transactionId,
        observed_world_revision: 42,
        collision_hand_surface: "left_hand_index_1_link",
        required_end_effector: "left_wrist",
        collision_target_id: "stone-column",
        current_wrist_world: { x: 1, y: 0.9, z: 2 },
        fixed_foot_world_targets: {
          left: { x: 0.9, y: 0.05, z: 2 },
          right: { x: 1.1, y: 0.05, z: 2 }
        },
        collision_target_world: { x: 1.2, y: 0.8, z: 2.1 },
        automatic_actuation: false
      }
    });
  });

  it("opens only the model-selected whole-body recovery path after a Skill route collision", async () => {
    const collisionReason =
      "environment_contact:left_hand_index_1_link:stone-column; preview_frames=353";
    const lightweight = lightweightObservationWorld({
      accepted: false,
      planId: "",
      createdRevision: 0,
      validatedStateSha256: "a".repeat(64),
      expiresRevision: 100,
      intentSha256: "b".repeat(64),
      target: { x: 4, y: 0, z: 5 },
      chunkTarget: { x: 4, y: 0, z: 5 },
      requestedArrivalHeading: null,
      arrivalHeading: null,
      waypoints: [],
      distance: 5,
      remainingDistance: 5,
      carry: { binding_set_sha256: "c".repeat(64), bindings: [] },
      reason: collisionReason
    }, {
      spatialBelief: {
        protocol: "humanoid-spatial-belief-v1",
        resolution_m: 0.5,
        observed_cell_count: 20,
        free_cell_count: 18,
        occupied_cell_count: 2,
        visited_cell_count: 4,
        total_cell_count: 100,
        coverage_ratio: 0.2,
        frontiers: [{
          id: "frontier:8:10",
          target: { x: 4, y: 0, z: 5 },
          expected_information_gain: 12,
          travel_distance_m: 5,
          revisit_penalty: 0,
          score: 6
        }]
      },
      interaction: {
        object_world_model: { frame: 0, world_revision: 0, objects: [] },
        skill_catalog: { contract_sha256: "d".repeat(64) },
        carrying: { bindings: [] },
        manipulable_objects: []
      }
    }, 0, {
      robot: {
        links: {
          left_wrist_yaw_link: {
            position: { x: 1, y: 0.9, z: 2 },
            rotation: { x: 0, y: 0, z: 0, w: 1 }
          },
          left_ankle_roll_link: {
            position: { x: 0.9, y: 0.05, z: 2 },
            rotation: { x: 0, y: 0, z: 0, w: 1 }
          },
          right_ankle_roll_link: {
            position: { x: 1.1, y: 0.05, z: 2 },
            rotation: { x: 0, y: 0, z: 0, w: 1 }
          }
        },
        objects: {
          "stone-column": {
            id: "stone-column",
            position: { x: 1.2, y: 0.8, z: 2.1 }
          }
        }
      }
    });
    const runtime = new HumanoidActionRuntime(lightweight.world, {
      requireSkillBinding: true
    });
    await runtime.invoke(
      "observe_humanoid",
      {},
      "clearance-observation",
      "humanoid-motion-reference"
    );
    const invocation = {
      skill: "explore" as const,
      frontier_id: "frontier:8:10",
      strategy: "information_gain" as const,
      maximum_travel_m: 6
    };
    await runtime.invoke(
      "submit_humanoid_skill_plan",
      {
        objective: "inspect the selected frontier",
        strategies: [{
          strategy_id: "frontier-route",
          rationale: "the frontier has current information gain",
          nodes: [{
            node_id: "reach-frontier",
            invocation,
            depends_on_node_ids: []
          }]
        }],
        selected_strategy_id: "frontier-route"
      },
      "clearance-skill-plan",
      "humanoid-motion-reference"
    );
    await runtime.invoke(
      "begin_humanoid_skill",
      {
        skill_plan_transaction_id: "clearance-skill-plan",
        skill_node_id: "reach-frontier",
        invocation,
        phase: "route_to_frontier"
      },
      "clearance-skill",
      "humanoid-motion-reference"
    );
    await expect(runtime.invoke(
      "plan_humanoid_skill",
      { skill_transaction_id: "clearance-skill" },
      "clearance-route-plan",
      "humanoid-motion-reference"
    )).resolves.toMatchObject({
      accepted: false,
      code: "autonomous_skill_route_rejected",
      detail: {
        attempts: [{ accepted: false, reason: collisionReason }]
      }
    });

    expect(runtime.isActionAvailable(
      "plan_humanoid_skill",
      "humanoid-motion-reference"
    )).toBe(false);
    expect(runtime.isActionAvailable(
      "plan_whole_body_motion_candidates",
      "humanoid-motion-reference"
    )).toBe(true);
    expect(runtime.planningToolState("humanoid-motion-reference")).toMatchObject({
      planning_actions: expect.arrayContaining([
        { action: "plan_humanoid_skill", available: false },
        { action: "plan_whole_body_motion_candidates", available: true }
      ]),
      transit_clearance: {
        status: "required",
        blocked_action: "plan_humanoid_skill",
        required_end_effector: "left_wrist"
      }
    });

    await expect(runtime.invoke(
      "plan_whole_body_motion_candidates",
      {
        skill_transaction_id: "clearance-skill",
        objective: "move the colliding wrist clear while keeping the base fixed",
        termination: {
          mode: "all",
          option_id: "left-wrist-clear",
          predicates: [{
            type: "end_effector_near_point",
            end_effector: "left_wrist",
            frame: "world",
            target: { x: 0.8, y: 1.05, z: 1.8 },
            tolerance_m: 0.04
          }],
          stable_steps: 2
        },
        candidates: [{
          id: "lift-left-wrist-away",
          intent: "clear the left hand from the obstacle",
          duration_seconds: 0.5,
          contacts: [],
          keyframes: [
            {
              at_seconds: 0,
              channels: [{
                type: "end_effector_position",
                end_effector: "left_ankle",
                frame: "world",
                position: { x: 0.9, y: 0.05, z: 2 },
                tolerance_m: 0.04
              }, {
                type: "end_effector_position",
                end_effector: "right_ankle",
                frame: "world",
                position: { x: 1.1, y: 0.05, z: 2 },
                tolerance_m: 0.04
              }]
            },
            {
              at_seconds: 0.5,
              channels: [{
                type: "end_effector_position",
                end_effector: "left_ankle",
                frame: "world",
                position: { x: 0.9, y: 0.05, z: 2 },
                tolerance_m: 0.04
              }, {
                type: "end_effector_position",
                end_effector: "right_ankle",
                frame: "world",
                position: { x: 1.1, y: 0.05, z: 2 },
                tolerance_m: 0.04
              }, {
                type: "end_effector_position",
                end_effector: "left_wrist",
                frame: "world",
                position: { x: 0.8, y: 1.05, z: 1.8 },
                tolerance_m: 0.04
              }]
            }
          ]
        }]
      },
      "clearance-posture-plan",
      "humanoid-motion-reference"
    )).resolves.toMatchObject({
      accepted: false,
      code: "whole_body_candidates_rejected",
      detail: { recovery_kind: "navigation_transit_clearance" }
    });
    expect(lightweight.candidatePlanningCalls()).toBe(1);
  });

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
      expect(observationDetail.joints).toBeUndefined();
      expect(JSON.stringify(observation.detail)).not.toContain("requestedNewtonMeters");
      expect(record(observationDetail.key_links).head_link).toBeDefined();
      expect(record(observationDetail.key_links).pelvis).toBeDefined();
      const observedRoot = record(observationDetail.root);
      const observedHeading = record(observedRoot.heading);
      const rootYaw = yawFromQuaternion(world.snapshot().robot.rootRotation);
      expect(observedHeading).toEqual({
        yaw_radians: rootYaw,
        forward_world: {
          x: Math.sin(rootYaw),
          y: 0,
          z: Math.cos(rootYaw)
        },
        left_world: {
          x: Math.cos(rootYaw),
          y: 0,
          z: -Math.sin(rootYaw)
        }
      });
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
      const authorityObservation = world.observe();
      expect(observationDetail.hand_coordination).toEqual(
        authorityObservation.handCoordination
      );
      expect(observationDetail.hand_surfaces).toBeNull();
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
      const referenceWeights = world.checkpoint().reference.jointTrackingWeights ?? [];
      expect(referenceWeights.every((value) => value === 0)).toBe(true);

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
            mode: "all",
            option_id: "runtime-forward-option",
            predicates: [{
              type: "root_near_point",
              target: candidateTarget,
              tolerance_m: 0.03
            }],
            stable_steps: 2
          },
          candidates: [
            {
              id: "runtime-noop-candidate",
              intent: "不实现当前前进目标",
              duration_seconds: 0.8,
              contacts: [],
              keyframes: [
                { at_seconds: 0, channels: [] },
                { at_seconds: 0.8, channels: [] }
              ]
            },
            {
              id: "runtime-grounded-candidate",
              intent: "保持支撑并连续前进",
              duration_seconds: 0.8,
              contacts: [],
              keyframes: [
                {
                  at_seconds: 0,
                  channels: [{
                    type: "root_velocity",
                    forward_mps: 0.2,
                    lateral_mps: 0
                  }]
                },
                {
                  at_seconds: 0.8,
                  channels: [{
                    type: "root_velocity",
                    forward_mps: 0.2,
                    lateral_mps: 0
                  }]
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
          },
          arrival_heading: null
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

function lightweightObservationWorld(
  navigationReceipt?: Awaited<ReturnType<HumanoidWorld["planNavigation"]>>,
  observationOverride: Record<string, unknown> = {},
  worldRevision = 0,
  snapshotOverride: Record<string, unknown> = {},
  capabilityRouting = false
): {
  world: HumanoidWorld;
  observationCalls: () => number;
  candidatePlanningCalls: () => number;
} {
  let observationCalls = 0;
  let candidatePlanningCalls = 0;
  const worldShape = {
    snapshot: () => ({
      frame: 0,
      worldRevision,
      ...structuredClone(snapshotOverride)
    }),
    consumablePlanIds: () => [],
    planNavigation: async () => structuredClone(navigationReceipt ?? {
      accepted: false,
      planId: "",
      createdRevision: 0,
      validatedStateSha256: "a".repeat(64),
      expiresRevision: 100,
      intentSha256: "b".repeat(64),
      target: { x: 0.15, y: 0, z: 0.3 },
      chunkTarget: { x: 0.15, y: 0, z: 0.3 },
      requestedArrivalHeading: null,
      arrivalHeading: null,
      waypoints: [{ x: 0.15, y: 0, z: 0.3 }],
      distance: 0.12,
      remainingDistance: 0,
      carry: { binding_set_sha256: "c".repeat(64), bindings: [] },
      reason: "navigation_timeout"
    }),
    planWholeBodyMotionCandidates: async () => {
      candidatePlanningCalls += 1;
      return {
        accepted: false,
        planId: "",
        selectedCandidateId: null,
        selectedRank: null,
        createdRevision: 0,
        validatedStateSha256: "a".repeat(64),
        expiresRevision: 100,
        intentSha256: null,
        channels: [],
        motion: null,
        option: null,
        selection: "model_rank_then_physics" as const,
        candidates: []
      };
    },
    observe: () => {
      observationCalls += 1;
      return {
        frame: 0,
        worldRevision,
        sensor: {
          position: { x: 0, y: 1, z: 0 },
          rotation: { x: 0, y: 0, z: 0, w: 1 },
          maximumRange: 1,
          horizontalFieldOfView: 1,
          verticalFieldOfView: 1
        },
        robot: {
          controller: {
            protocol: "humanoid-controller-v1",
            implementation: "test_learned_controller",
            actuation: "joint_position_pd",
            controlStepSeconds: 0.02,
            physicsStepSeconds: 0.005,
            learnedPolicy: {
              protocol: "humanoid-learned-policy-v1",
              runtime: "onnx",
              observationSpace: {
                protocol: "test-observation-v1",
                size: 12
              },
              actionSpace: {
                protocol: "test-action-v1",
                size: 4
              },
              capabilities: [
                "balance",
                "locomotion",
                ...(capabilityRouting ? [] : ["joint_reference_tracking"])
              ]
            },
            ...(capabilityRouting
              ? {
                  capabilityRouting: {
                    protocol: "humanoid-controller-capability-routing-v1",
                    strategy: "declared_capabilities",
                    fallback: {
                      mode: "reference_control",
                      implementation: "yahmp_onnx"
                    }
                  }
                }
              : {})
          },
          rootPosition: { x: 0, y: 0, z: 0 },
          rootRotation: { x: 0, y: 0, z: 0, w: 1 },
          fallen: false,
          balance: {},
          feet: {},
          joints: {},
          links: Object.fromEntries([
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link"
          ].map((name) => [name, {
            position: { x: 0, y: 0, z: 0 },
            rotation: { x: 0, y: 0, z: 0, w: 1 }
          }])),
          contacts: [],
          nonFootEnvironmentContacts: []
        },
        handCoordination: {
          left: {
            thumb_opposition: 0,
            thumb_curl: 0,
            index_curl: 0,
            middle_curl: 0
          },
          right: {
            thumb_opposition: 0,
            thumb_curl: 0,
            index_curl: 0,
            middle_curl: 0
          }
        },
        handSurfaces: [],
        manipulationReachability: [],
        manipulationBasePlacements: [],
        objectTokens: [],
        solidTokens: [],
        grasp: {
          contractSha256: "fc1e2d113bb5e5f5f8a75f0faa3efc8bd97ecc18eb41463da09d26bb52cfc193",
          assessments: []
        },
        interaction: {},
        navigation: {},
        motionGenerator: {
          protocol: "humanoid-motion-generator-v1",
          implementation: "test_task_space_generator"
        },
        ...structuredClone(observationOverride)
      };
    }
  };
  const world = {
    ...worldShape,
    observeManipulationReachability: async () => worldShape.observe()
  } as unknown as HumanoidWorld;
  return {
    world,
    observationCalls: () => observationCalls,
    candidatePlanningCalls: () => candidatePlanningCalls
  };
}

function visibleWorkpieceToken(position: { x: number; y: number; z: number }) {
  return {
    id: "workpiece",
    role: "manipulable" as const,
    kind: "workpiece",
    color: "#d18a45",
    size: { x: 0.03, y: 0.22, z: 0.03 },
    portable: true,
    status: "visible" as const,
    state: "active" as const,
    authority: "mujoco_exact" as const,
    exact: true,
    observable: true,
    pose: {
      position,
      rotation: { x: 0, y: 0, z: 0, w: 1 }
    },
    observedFrame: 0,
    observedWorldRevision: 0,
    position,
    rotation: { x: 0, y: 0, z: 0, w: 1 },
    linearVelocity: { x: 0, y: 0, z: 0 },
    angularVelocity: { x: 0, y: 0, z: 0 },
    firstSeenRevision: 0,
    lastSeenRevision: 0,
    lastSeenFrame: 0,
    observationCount: 1,
    ageRevisions: 0,
    relation: {
      distanceToRobot: Math.hypot(position.x, position.z),
      bearingRadians: Math.atan2(position.x, position.z),
      verticalOffset: position.y,
      distanceToLeftWrist: 0.5,
      distanceToRightWrist: 0.6
    },
    currentContacts: []
  };
}
