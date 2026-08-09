import {
  MemorySession,
  RunContext,
  Usage,
  type Model,
  type ModelRequest
} from "@openai/agents";
import { describe, expect, it } from "vitest";
import type { ProviderConfig } from "../../config/load.js";
import type { HumanoidActionReceipt } from "./runtime.js";
import {
  HUMANOID_AGENT_TOOL_CONTRACTS,
  createHumanoidAgentHierarchy,
  goalManagerInvocationInput,
  motionInvocationInput
} from "./agents.js";

const provider: ProviderConfig = {
  protocol: "openai_compatible",
  baseUrl: "https://example.test/v1",
  model: "test-model",
  apiKey: "test-key",
  temperature: 0.4,
  maxOutputTokens: 2048,
  contextWindowTokens: 32_768,
  compactTriggerTokens: 8192,
  compactRecentModelTurns: 4,
  compactMaxOutputTokens: 2048
};

describe("humanoid agent hierarchy", () => {
  it("renders an exact executor authority envelope", () => {
    const rendered = HUMANOID_AGENT_TOOL_CONTRACTS.executor.inputBuilder({
      params: {
        objective: "执行已接受导航",
        execution: {
          kind: "execute_plan",
          planning_action: "plan_humanoid_navigation",
          planning_transaction_id: "planning-call-41"
        }
      }
    });

    expect(rendered).toContain("CURRENT EXECUTION AUTHORITY");
    expect(rendered).toContain('"planning_action":"plan_humanoid_navigation"');
    expect(rendered).toContain('"planning_transaction_id":"planning-call-41"');
    expect(rendered).toContain("不能改写为内部 plan_id");
  });

  it("places exact live Goal identifiers in every Goal Manager invocation", () => {
    const evidenceRef = `goal-world:19:19:${"a".repeat(64)}`;
    const candidateId = `goal-candidate:${"b".repeat(64)}`;

    const rendered = goalManagerInvocationInput({
      run_mode: "mission",
      mission_goal: {
        summary: "进入庭院信标区",
        predicates: [{
          type: "robot_in_zone",
          zone_id: "courtyard_beacon",
          tolerance: 0.2
        }]
      },
      goal_dag: {
        status: "awaiting_model_selection",
        candidates: {
          [candidateId]: {
            status: "proposed",
            proposal_id: "mission-navigation",
            candidate_sequence: 7,
            goal: {
              summary: "进入庭院信标区",
              predicates: [{
                type: "robot_in_zone",
                zone_id: "courtyard_beacon",
                tolerance: 0.2
              }]
            },
            mission_link: "直接推进长期任务",
            dependency_candidate_ids: []
          }
        },
        current_epoch_id: null
      },
      goal_context: {
        evidence_ref: evidenceRef,
        observation: {
          zone_ids: ["courtyard_beacon"],
          visible_object_ids: ["courtyard_crate"],
          objects: [
            { id: "courtyard_crate", portable: true },
            { id: "fixed-bench", portable: false }
          ],
          solids: [
            { id: "stone_column", kind: "block" },
            { id: "wall", kind: "wall" }
          ]
        }
      },
      interaction: {
        carrying: {
          phase: "carrying",
          bindings: [{ object_id: "courtyard_crate", hand: "left" }],
          continuation_verified: true
        }
      },
      recent_receipts: [{
        transaction_id: "planning-call-42",
        action: "plan_whole_body_motion_candidates",
        accepted: false,
        code: "whole_body_candidates_rejected",
        world_after_revision: 21,
        frame_count: 0,
        detail: {
          reachable_base_placements: [{
            object_id: "courtyard_crate",
            root_world_target: { x: 2.1, y: 0.75, z: 2.4 }
          }]
        }
      }]
    });

    expect(rendered).toContain(`"current_goal_evidence_ref":"${evidenceRef}"`);
    expect(rendered).toContain('"run_mode":"mission"');
    expect(rendered).toContain('"mission_goal":{"summary":"进入庭院信标区"');
    expect(rendered).toContain(`"existing_goal_candidate_ids":["${candidateId}"]`);
    expect(rendered).toContain(
      `"candidate_sequence":7,"proposal_id":"mission-navigation","candidate_id":"${candidateId}"`
    );
    expect(rendered).toContain(
      '"status":"proposed","goal":{"summary":"进入庭院信标区"'
    );
    expect(rendered).toContain(
      '"mission_link":"直接推进长期任务","dependency_candidate_ids":[]'
    );
    expect(rendered).toContain(
      '"visible_object_ids":["courtyard_crate"],"portable_object_ids":["courtyard_crate"]'
    );
    expect(rendered).toContain('"removable_block_ids":["stone_column"]');
    expect(rendered).toContain('"predicate_types":["robot_at","robot_in_zone"');
    expect(rendered).toContain(
      '"carrying":{"phase":"carrying","object_ids":["courtyard_crate"],"continuation_verified":true}'
    );
    expect(rendered).toContain(
      '"recent_action_evidence":[{"evidence_ref":"action:planning-call-42","transaction_id":"planning-call-42"'
    );
    expect(rendered).toContain('"world_after_revision":21,"frame_count":0');
    expect(rendered).toContain(
      '"detail":{"reachable_base_placements":[{"object_id":"courtyard_crate"'
    );
    expect(rendered).toContain("候选提交和选择会由 Harness 绑定本次证据");
  });

  it("gives Motion only active Goal authority instead of Coordinator parameters", () => {
    const rendered = motionInvocationInput({
      run_mode: "mission",
      coordinator_phase: "plan",
      active_cycle: { cycle_id: "cycle-7" },
      active_goal: {
        summary: "抓取可见工件",
        predicates: [{ type: "object_grasped", object_id: "workpiece", hand: "left" }]
      },
      goal_dag: { current_epoch_id: "goal-epoch-7" },
      planning_tool_state: {
        planning_actions: [{
          action: "plan_humanoid_navigation",
          available: false
        }],
        cooldown: {
          action: "plan_humanoid_navigation",
          code: "repeated_planning_failure"
        }
      },
      robot: { root_position: { x: 9, y: 8, z: 7 } }
    });

    expect(rendered).toContain("CURRENT MOTION DELEGATION");
    expect(rendered).toContain('"current_goal_epoch_id":"goal-epoch-7"');
    expect(rendered).toContain('"active_goal":{"summary":"抓取可见工件"');
    expect(rendered).toContain(
      '"planning_tool_state":{"planning_actions":[{"action":"plan_humanoid_navigation","available":false}]'
    );
    expect(rendered).not.toContain('"root_position"');
  });

  it("owns one Model facade and one Session per concrete hierarchy node", async () => {
    const modelOwners: string[] = [];
    const sessionOwners: string[] = [];
    const models: Model[] = [];
    const recallRequests: unknown[] = [];
    const goalRecallRequests: unknown[] = [];
    const invokedActions: string[] = [];
    let cycleCompletion = {
      status: "not_ready" as "ready" | "not_ready",
      evidence_transaction_ids: [] as string[],
      execution_transaction_id: null as string | null,
      observed_after_execution: false,
      reason: "no execution" as string | null
    };
    let coordinatorPhase = "observe_or_plan" as
      | "goal_selection"
      | "goal_transition"
      | "complete_satisfied_goal"
      | "observe_or_plan"
      | "plan"
      | "replan_or_retire"
      | "execute_plan"
      | "post_execution"
      | "complete_cycle";
    let executorDelegationAvailable = false;
    let goalRetirementDelegationAvailable = false;
    let goalTransitionCompletionAvailable = false;
    let sentryDelegationAvailable = true;
    const execution = receipt({
      transactionId: "execute-accepted",
      action: "execute_whole_body_motion",
      agentId: "humanoid-executor",
      worldBeforeRevision: 8,
      worldAfterRevision: 20,
      frameCount: 12,
      channels: ["left_arm"]
    });
    const runtime = {
      invoke: async (action: string) => {
        invokedActions.push(action);
        return { ...execution, action } as HumanoidActionReceipt;
      },
      recallGoalHistory: async (request: unknown) => {
        goalRecallRequests.push(request);
        return {
          historical_only: true,
          candidates: [{ candidate_id: `goal-candidate:${"a".repeat(64)}` }]
        };
      },
      recallEmbodiedHistory: async (request: unknown) => {
        recallRequests.push(request);
        return {
          historical_only: false,
          current_world_revision: 20,
          episodes: [{ source_ref: "episode:7", sequence: 7 }]
        };
      },
      validateCycleEvidence: (transactionIds: readonly string[]) => {
        expect(transactionIds).toEqual([execution.transactionId]);
        return structuredClone(execution);
      },
      cycleCompletionReadiness: () => structuredClone(cycleCompletion),
      coordinatorPhase: () => coordinatorPhase,
      executorDelegationAvailable: () => executorDelegationAvailable,
      goalRetirementDelegationAvailable: () => goalRetirementDelegationAvailable,
      goalTransitionCompletionAvailable: () => goalTransitionCompletionAvailable,
      sentryDelegationAvailable: () => sentryDelegationAvailable,
      validateGoalTransition: () => ({ status: "superseded" }),
      validateSatisfiedGoal: () => ({
        epoch_id: "goal-epoch:test",
        cycle_id: "autonomous-cycle:test",
        physical_execution_required: false
      })
    } as never;
    const hierarchy = createHumanoidAgentHierarchy({
      provider,
      runtime,
      createModel: (agentId) => {
        modelOwners.push(agentId);
        const model = agentId === "humanoid-executor"
          ? functionCallModel(
            "execute_humanoid_skill",
            JSON.stringify({ planning_transaction_id: "planning-call-41" })
          )
          : modelStub();
        models.push(model);
        return model;
      },
      createSession: (agentId) => {
        sessionOwners.push(agentId);
        return new MemorySession({ sessionId: agentId });
      },
      callModelInputFilter: ({ modelData }) => modelData
    });

    expect(modelOwners).toEqual([
      "humanoid-goal-manager",
      "humanoid-sentry",
      "humanoid-motion-reference",
      "humanoid-executor",
      "humanoid-coordinator"
    ]);
    expect(sessionOwners).toEqual([
      "humanoid-goal-manager",
      "humanoid-sentry",
      "humanoid-motion-reference",
      "humanoid-executor",
      "humanoid-coordinator"
    ]);
    expect(new Set(models).size).toBe(5);
    expect(hierarchy.goalManager.model).not.toBe(hierarchy.coordinator.model);
    expect(hierarchy.coordinator.model).not.toBe(hierarchy.sentry.model);
    expect(hierarchy.sentry.model).not.toBe(hierarchy.motion.model);
    expect(hierarchy.motion.model).not.toBe(hierarchy.executor.model);
    expect(hierarchy.session("humanoid-sentry")).toBeDefined();
    expect(hierarchy.session("humanoid-sentry")).not.toBe(
      hierarchy.session("humanoid-motion-reference")
    );
    expect(hierarchy.coordinatorSession).toBe(
      hierarchy.session("humanoid-coordinator")
    );
    expect(hierarchy.goalManager.tools.map((entry) => entry.name)).toEqual([
      "recall_goal_history",
      "submit_goal_candidates",
      "select_goal_candidate",
      "retire_goal_epoch"
    ]);

    expect(hierarchy.coordinator.tools.map((entry) => entry.name)).toEqual([
      "recall_embodied_history",
      "delegate_goal_manager",
      "delegate_humanoid_sentry",
      "delegate_motion_reference",
      "delegate_physics_executor",
      "complete_autonomous_cycle",
      "complete_goal_transition",
      "complete_satisfied_goal"
    ]);
    expect(hierarchy.coordinator.tools.map((entry) => entry.name)).not.toContain(
      "execute_whole_body_motion"
    );
    expect(hierarchy.sentry.tools.map((entry) => entry.name)).toEqual([
      "observe_humanoid"
    ]);
    expect(hierarchy.motion.tools.map((entry) => entry.name)).toEqual([
      "observe_humanoid",
      "recall_embodied_history",
      "submit_humanoid_skill_plan",
      "begin_humanoid_skill",
      "plan_humanoid_skill",
      "plan_whole_body_motion_candidates"
    ]);
    expect(hierarchy.executor.tools.map((entry) => entry.name)).toEqual([
      "execute_humanoid_skill",
      "execute_whole_body_motion",
      "remove_world_block"
    ]);
    expect(hierarchy.sentry.tools.map((entry) => entry.name)).not.toContain(
      "recall_embodied_history"
    );
    expect(hierarchy.executor.tools.map((entry) => entry.name)).not.toContain(
      "recall_embodied_history"
    );

    const coordinatorTool = (name: string) => {
      const selected = hierarchy.coordinator.tools.find((entry) => entry.name === name);
      if (!selected || selected.type !== "function") {
        throw new Error(`Coordinator tool is missing: ${name}`);
      }
      return selected;
    };
    const visibleCoordinatorTools = async () => (
      await hierarchy.coordinator.getAllTools(
        new RunContext({ runId: "stable-coordinator-tools" })
      )
    ).map((entry) => entry.name);
    expect(await visibleCoordinatorTools()).toEqual([
      "recall_embodied_history",
      "delegate_humanoid_sentry",
      "delegate_motion_reference"
    ]);
    coordinatorPhase = "plan";
    sentryDelegationAvailable = false;
    expect(await visibleCoordinatorTools()).toEqual([
      "recall_embodied_history",
      "delegate_motion_reference"
    ]);
    coordinatorPhase = "observe_or_plan";
    sentryDelegationAvailable = true;
    const smuggledMotionParameters = await coordinatorTool(
      "delegate_motion_reference"
    ).invoke(
      new RunContext({ runId: "smuggled-motion-parameters" }),
      JSON.stringify({ objective: "Move to x=4.2 and close the left palm." }),
      {
        toolCall: {
          type: "function_call",
          callId: "smuggled-motion-parameters",
          name: "delegate_motion_reference",
          arguments: JSON.stringify({
            objective: "Move to x=4.2 and close the left palm."
          }),
          status: "completed"
        }
      }
    );
    expect(JSON.parse(String(smuggledMotionParameters))).toMatchObject({
      accepted: false,
      code: "invalid_tool_input",
      tool: "delegate_motion_reference",
      automatic_actuation: false
    });
    const unavailableExecutorDelegation = await coordinatorTool(
      "delegate_physics_executor"
    ).invoke(
      new RunContext({ runId: "unavailable-executor-delegation" }),
      JSON.stringify({
        objective: "尝试消费尚不存在的规划",
        execution: {
          kind: "execute_plan",
          planning_action: "plan_humanoid_skill",
          planning_transaction_id: "missing-plan"
        }
      }),
      {
        toolCall: {
          type: "function_call",
          callId: "unavailable-executor-delegation",
          name: "delegate_physics_executor",
          arguments: "{}",
          status: "completed"
        }
      }
    );
    expect(JSON.parse(String(unavailableExecutorDelegation))).toMatchObject({
      accepted: false,
      code: "coordinator_phase_rejected",
      tool: "delegate_physics_executor",
      automatic_actuation: false
    });
    coordinatorPhase = "replan_or_retire";
    goalRetirementDelegationAvailable = false;
    expect(await visibleCoordinatorTools()).not.toContain("delegate_goal_manager");
    const recoverableGoalRetirement = await coordinatorTool(
      "delegate_goal_manager"
    ).invoke(
      new RunContext({ runId: "recoverable-goal-retirement" }),
      "{}",
      {
        toolCall: {
          type: "function_call",
          callId: "recoverable-goal-retirement",
          name: "delegate_goal_manager",
          arguments: "{}",
          status: "completed"
        }
      }
    );
    expect(JSON.parse(String(recoverableGoalRetirement))).toMatchObject({
      accepted: false,
      code: "coordinator_phase_rejected",
      tool: "delegate_goal_manager",
      coordinator_phase: "replan_or_retire",
      automatic_actuation: false
    });
    goalRetirementDelegationAvailable = true;
    expect(await visibleCoordinatorTools()).toContain("delegate_goal_manager");
    coordinatorPhase = "goal_transition";
    expect(await visibleCoordinatorTools()).toEqual(["delegate_goal_manager"]);
    coordinatorPhase = "goal_selection";
    goalRetirementDelegationAvailable = false;
    expect(await visibleCoordinatorTools()).toEqual(["delegate_goal_manager"]);
    goalTransitionCompletionAvailable = true;
    expect(await visibleCoordinatorTools()).toEqual(["complete_goal_transition"]);
    goalTransitionCompletionAvailable = false;
    coordinatorPhase = "complete_satisfied_goal";
    goalRetirementDelegationAvailable = false;
    expect(await visibleCoordinatorTools()).toEqual(["complete_satisfied_goal"]);
    const satisfiedGoalCompletion = await coordinatorTool(
      "complete_satisfied_goal"
    ).invoke(
      new RunContext({ runId: "satisfied-goal-completion" }),
      JSON.stringify({ summary: "当前实时物理状态已经满足目标" })
    );
    expect(JSON.parse(String(satisfiedGoalCompletion))).toMatchObject({
      status: "satisfied_goal_completed",
      verification: {
        physical_execution_required: false
      }
    });
    coordinatorPhase = "observe_or_plan";
    goalRetirementDelegationAvailable = false;
    expect(await visibleCoordinatorTools()).not.toContain("delegate_goal_manager");
    executorDelegationAvailable = true;
    const invalidExecutorDelegation = await coordinatorTool(
      "delegate_physics_executor"
    ).invoke(
      new RunContext({ runId: "invalid-executor-delegation" }),
      JSON.stringify({
        objective: "缺少规划授权",
        execution: {
          kind: "execute_plan",
          planning_action: "plan_humanoid_skill",
          planning_transaction_id: null
        }
      }),
      {
        toolCall: {
          type: "function_call",
          callId: "invalid-executor-delegation",
          name: "delegate_physics_executor",
          arguments: "{}",
          status: "completed"
        }
      }
    );
    expect(JSON.parse(String(invalidExecutorDelegation))).toMatchObject({
      accepted: false,
      code: "invalid_tool_input",
      tool: "delegate_physics_executor",
      validation_issues: [expect.objectContaining({
        path: "execution.planning_transaction_id"
      })]
    });
    coordinatorPhase = "execute_plan";
    const validExecutorDelegation = await coordinatorTool(
      "delegate_physics_executor"
    ).invoke(
      new RunContext({ runId: "valid-executor-delegation" }),
      JSON.stringify({
        objective: "执行已接受导航",
        execution: {
          kind: "execute_plan",
          planning_action: "plan_humanoid_navigation",
          planning_transaction_id: "planning-call-41"
        }
      }),
      {
        toolCall: {
          type: "function_call",
          callId: "valid-executor-delegation",
          name: "delegate_physics_executor",
          arguments: "{}",
          status: "completed"
        }
      }
    );
    expect(String(validExecutorDelegation)).toContain('"accepted":true');
    expect(invokedActions).toContain("execute_humanoid_skill");
    cycleCompletion = {
      status: "ready",
      evidence_transaction_ids: [execution.transactionId],
      execution_transaction_id: execution.transactionId,
      observed_after_execution: false,
      reason: null
    };
    coordinatorPhase = "post_execution";
    expect(await visibleCoordinatorTools()).toEqual([
      "delegate_humanoid_sentry",
      "delegate_physics_executor"
    ]);
    const prematureCompletionInput = JSON.stringify({
      summary: "不能跳过执行后感知",
      evidence_transaction_ids: [execution.transactionId]
    });
    expect(JSON.parse(String(await coordinatorTool(
      "complete_autonomous_cycle"
    ).invoke(
      new RunContext({ runId: "premature-cycle-completion" }),
      prematureCompletionInput
    )))).toMatchObject({
      accepted: false,
      code: "coordinator_phase_rejected",
      tool: "complete_autonomous_cycle",
      coordinator_phase: "post_execution",
      automatic_actuation: false
    });
    cycleCompletion.observed_after_execution = true;
    expect(JSON.parse(String(await coordinatorTool(
      "complete_autonomous_cycle"
    ).invoke(
      new RunContext({ runId: "inconsistent-cycle-completion-phase" }),
      prematureCompletionInput
    )))).toMatchObject({
      accepted: false,
      code: "coordinator_phase_rejected",
      coordinator_phase: "post_execution"
    });
    coordinatorPhase = "complete_cycle";
    executorDelegationAvailable = false;
    expect(await visibleCoordinatorTools()).toEqual([
      "complete_autonomous_cycle"
    ]);
    expect(hierarchy.motion.instructions).toEqual(expect.stringContaining(
      "submit_humanoid_skill_plan 提交短程 Skill DAG"
    ));
    expect(hierarchy.motion.instructions).toEqual(expect.stringContaining(
      "只调用 plan_humanoid_skill"
    ));
    expect(hierarchy.motion.instructions).toEqual(expect.stringContaining(
      "不能提交关节角、关键帧或低层路线绕过它"
    ));
    expect(hierarchy.motion.instructions).toEqual(expect.stringContaining(
      "break_block 只能选择当前 solid_tokens 中 kind=block 的实体"
    ));
    expect(hierarchy.motion.instructions).toEqual(expect.stringContaining(
      "不得使用固定巡逻点、预设动作序列、随机电机噪声"
    ));
    expect(hierarchy.motion.instructions).toEqual(expect.stringContaining(
      "不能借用其他 Agent 或历史记忆充当当前传感事实"
    ));
    expect(hierarchy.goalManager.instructions).toEqual(expect.stringContaining(
      "不得改 tolerance、删减谓词或拼接额外条件"
    ));
    expect(hierarchy.goalManager.instructions).toEqual(expect.stringContaining(
      "一个 active Goal 可以跨越多次观察、规划、抓取、导航和执行周期"
    ));
    expect(hierarchy.executor.instructions).toEqual(expect.stringContaining(
      "plan_humanoid_skill 必须调用 execute_humanoid_skill"
    ));

    const motionBehavior = hierarchy.motion.toolUseBehavior;
    if (typeof motionBehavior !== "function") {
      throw new Error("Motion Agent must validate action receipts");
    }
    const motionTool = (name: string) => {
      const selected = hierarchy.motion.tools.find((entry) => entry.name === name);
      if (!selected) throw new Error(`Motion tool is missing: ${name}`);
      return selected;
    };
    const submitSkillPlanTool = motionTool("submit_humanoid_skill_plan");
    const rejectedMotionStep = await motionBehavior(
      new RunContext({ runId: "rejected-motion-step" }),
      [{
        type: "function_output",
        tool: submitSkillPlanTool,
        output: JSON.stringify({
          transactionId: "rejected-motion-step",
          action: "submit_humanoid_skill_plan",
          accepted: false
        })
      }] as never
    );
    expect(rejectedMotionStep).toMatchObject({
      isFinalOutput: true,
      finalOutput: expect.stringContaining("rejected-motion-step")
    });
    for (const [index, action] of [
      "observe_humanoid",
      "submit_humanoid_skill_plan",
      "begin_humanoid_skill"
    ].entries()) {
      const acceptedMotionStep = await motionBehavior(
        new RunContext({ runId: `accepted-motion-step-${index}` }),
        [{
          type: "function_output",
          tool: motionTool(action),
          output: JSON.stringify({
            transactionId: `accepted-motion-step-${index}`,
            action,
            accepted: true
          })
        }] as never
      );
      expect(acceptedMotionStep).toEqual({
        isFinalOutput: false,
        isInterrupted: undefined
      });
    }
    const acceptedMotionPlan = await motionBehavior(
      new RunContext({ runId: "accepted-motion-plan" }),
      [{
        type: "function_output",
        tool: motionTool("plan_humanoid_skill"),
        output: JSON.stringify({
          transactionId: "accepted-motion-plan",
          action: "plan_humanoid_skill",
          accepted: true
        })
      }] as never
    );
    expect(acceptedMotionPlan).toMatchObject({
      isFinalOutput: true,
      finalOutput: expect.stringContaining("accepted-motion-plan")
    });

    const coordinatorBehavior = hierarchy.coordinator.toolUseBehavior;
    if (typeof coordinatorBehavior !== "function") {
      throw new Error("Coordinator must validate terminal tool output");
    }
    const rejectedTerminal = await coordinatorBehavior(
      new RunContext({ runId: "rejected-terminal" }),
      [{
        type: "function_output",
        tool: { name: "complete_goal_transition" },
        output: "An error occurred while running the tool"
      }] as never
    );
    expect(rejectedTerminal).toEqual({
      isFinalOutput: false,
      isInterrupted: undefined
    });
    const acceptedTerminal = await coordinatorBehavior(
      new RunContext({ runId: "accepted-terminal" }),
      [{
        type: "function_output",
        tool: { name: "complete_goal_transition" },
        output: JSON.stringify({ status: "goal_transition_completed" })
      }] as never
    );
    expect(acceptedTerminal).toMatchObject({
      isFinalOutput: true,
      finalOutput: JSON.stringify({ status: "goal_transition_completed" })
    });
    const acceptedSatisfiedGoal = await coordinatorBehavior(
      new RunContext({ runId: "accepted-satisfied-goal" }),
      [{
        type: "function_output",
        tool: { name: "complete_satisfied_goal" },
        output: JSON.stringify({ status: "satisfied_goal_completed" })
      }] as never
    );
    expect(acceptedSatisfiedGoal).toMatchObject({
      isFinalOutput: true,
      finalOutput: JSON.stringify({ status: "satisfied_goal_completed" })
    });

    coordinatorPhase = "observe_or_plan";
    const recall = hierarchy.coordinator.tools.find(
      (entry) => entry.name === "recall_embodied_history"
    );
    if (!recall || recall.type !== "function") {
      throw new Error("Embodied history recall tool is missing");
    }
    const recalled = await recall.invoke(
      new RunContext({ runId: "humanoid-recall-test" }),
      JSON.stringify({ source_refs: ["episode:7"], limit: 1 })
    );
    expect(JSON.parse(String(recalled))).toMatchObject({
      historical_only: true,
      episodes: [{ source_ref: "episode:7", sequence: 7 }]
    });
    expect(recallRequests).toEqual([{ source_refs: ["episode:7"], limit: 1 }]);

    const goalRecall = hierarchy.goalManager.tools.find(
      (entry) => entry.name === "recall_goal_history"
    );
    if (!goalRecall || goalRecall.type !== "function") {
      throw new Error("Goal history recall tool is missing");
    }
    const candidateId = `goal-candidate:${"a".repeat(64)}`;
    const recalledGoal = await goalRecall.invoke(
      new RunContext({ runId: "goal-recall-test" }),
      JSON.stringify({ candidate_ids: [candidateId], limit: 1 })
    );
    expect(JSON.parse(String(recalledGoal))).toMatchObject({
      historical_only: true,
      candidates: [{ candidate_id: candidateId }]
    });
    expect(goalRecallRequests).toEqual([{
      candidate_ids: [candidateId],
      limit: 1
    }]);

    const complete = hierarchy.coordinator.tools.find(
      (entry) => entry.name === "complete_autonomous_cycle"
    );
    if (!complete || complete.type !== "function") {
      throw new Error("Cycle completion tool is missing");
    }
    coordinatorPhase = "complete_cycle";
    const output = await complete.invoke(
      new RunContext({ runId: "humanoid-hierarchy-test" }),
      JSON.stringify({
        summary: "完成一次真实全身动作",
        evidence_transaction_ids: [execution.transactionId],
        next_intent: "观察动作后的平衡变化"
      })
    );
    expect(JSON.parse(String(output))).toMatchObject({
      status: "cycle_completed",
      world_revision: 20,
      executed_action: "execute_whole_body_motion"
    });
  });

  it("rejects a shared Model facade instead of mixing agent ownership", () => {
    const shared = modelStub();
    expect(() => createHumanoidAgentHierarchy({
      provider,
      runtime: {
        invoke: async () => { throw new Error("outside test"); },
        recallEmbodiedHistory: async () => { throw new Error("outside test"); },
        validateCycleEvidence: () => { throw new Error("outside test"); }
      } as never,
      createModel: () => shared,
      createSession: (agentId) => new MemorySession({ sessionId: agentId }),
      callModelInputFilter: ({ modelData }) => modelData
    })).toThrow("cannot share one Model facade");
  });

  it("keeps Sentry available during replanning and applies the nested input filter", async () => {
    const filteredAgents: string[] = [];
    const hierarchy = createHumanoidAgentHierarchy({
      provider,
      runtime: {
        contextAnchor: () => ({}),
        invoke: async (name: string, input: unknown, transactionId: string, agentId: string) => (
          receipt({
            transactionId,
            agentId,
            action: name as "observe_humanoid",
            input: input as never,
            code: "humanoid_observed"
          })
        ),
        recallEmbodiedHistory: async () => ({ historical_only: true }),
        validateCycleEvidence: () => { throw new Error("outside test"); },
        cycleCompletionReadiness: () => ({
          status: "not_ready",
          evidence_transaction_ids: [],
          execution_transaction_id: null,
          observed_after_execution: false,
          reason: "no execution"
        }),
        coordinatorPhase: () => "replan_or_retire"
      } as never,
      createModel: (agentId) => agentId === "humanoid-sentry"
        ? functionCallModel("observe_humanoid")
        : modelStub(),
      createSession: (agentId) => new MemorySession({ sessionId: agentId }),
      callModelInputFilter: ({ modelData, agent }) => {
        filteredAgents.push(agent.name);
        return modelData;
      }
    });
    const delegate = hierarchy.coordinator.tools.find(
      (entry) => entry.name === "delegate_humanoid_sentry"
    );
    if (!delegate || delegate.type !== "function") {
      throw new Error("Sentry delegation tool is missing");
    }

    const output = await delegate.invoke(
      new RunContext({ runId: "nested-filter" }),
      JSON.stringify({}),
      {
        toolCall: {
          type: "function_call",
          callId: "delegate-sentry-filter",
          name: "delegate_humanoid_sentry",
          arguments: JSON.stringify({}),
          status: "completed"
        }
      }
    );
    expect(JSON.parse(String(output))).toMatchObject({
      agentId: "humanoid-sentry",
      action: "observe_humanoid",
      accepted: true
    });
    expect(filteredAgents).toEqual(["人形感知哨兵"]);
  });

  it("continues a prose-only Motion result inside its own Session", async () => {
    let worldRevision = 11;
    const modelRequests: ModelRequest[] = [];
    const followUps: Array<{ agentId: string; attempt: number; reason: string }> = [];
    const invokedActions: string[] = [];
    const sessions = new Map<string, MemorySession>();
    let sessionItemsAtFollowUp: unknown[] = [];
    let motionResponse = 0;
    const hierarchy = createHumanoidAgentHierarchy({
      provider,
      runtime: delegatedMotionRuntime({
        contextAnchor: () => ({
          coordinator_phase: "plan",
          active_cycle: { cycle_id: `cycle-${worldRevision}` },
          planning_tool_state: { world_revision: worldRevision }
        }),
        invoke: async (action, _actionInput, transactionId, agentId) => {
          invokedActions.push(action);
          if (action === "observe_humanoid") worldRevision = 12;
          return receipt({
            transactionId,
            agentId,
            action: action as "observe_humanoid" | "plan_humanoid_skill",
            code: action === "observe_humanoid"
              ? "humanoid_observed"
              : "humanoid_skill_planned"
          });
        }
      }),
      createModel: (agentId) => agentId === "humanoid-motion-reference"
        ? {
            getResponse: async (request) => {
              modelRequests.push(request);
              motionResponse += 1;
              if (motionResponse === 1) return functionCallResponse("observe_humanoid");
              if (motionResponse === 2) return textResponse("现在提交正式规划。");
              return functionCallResponse(
                "plan_humanoid_skill",
                JSON.stringify({ skill_transaction_id: "skill-binding-12" })
              );
            },
            getStreamedResponse: () => {
              throw new Error("Streaming is outside this test");
            }
          } as Model
        : modelStub(),
      createSession: (agentId) => {
        const session = new MemorySession({ sessionId: agentId });
        sessions.set(agentId, session);
        return session;
      },
      callModelInputFilter: ({ modelData }) => modelData,
      onDelegatedDecisionFollowUp: async (event) => {
        followUps.push(event);
        sessionItemsAtFollowUp = await sessions.get(event.agentId)?.getItems() ?? [];
      }
    });
    const delegate = hierarchy.coordinator.tools.find(
      (entry) => entry.name === "delegate_motion_reference"
    );
    if (!delegate || delegate.type !== "function") {
      throw new Error("Motion delegation tool is missing");
    }

    const output = await delegate.invoke(
      new RunContext({ runId: "motion-same-session-continuation" }),
      "{}",
      {
        toolCall: {
          type: "function_call",
          callId: "delegate-motion-continuation",
          name: "delegate_motion_reference",
          arguments: "{}",
          status: "completed"
        }
      }
    );

    expect(JSON.parse(String(output))).toMatchObject({
      agentId: "humanoid-motion-reference",
      action: "plan_humanoid_skill",
      accepted: true
    });
    expect(invokedActions).toEqual(["observe_humanoid", "plan_humanoid_skill"]);
    expect(followUps).toEqual([expect.objectContaining({
      agentId: "humanoid-motion-reference",
      attempt: 1
    })]);
    expect(modelRequests).toHaveLength(3);
    expect(JSON.stringify(modelRequests[2]?.input)).toContain(
      "SPECIALIST DECISION RECOVERY V1"
    );
    expect(JSON.stringify(modelRequests[2]?.input)).toContain("world_revision\\\":12");
    expect(JSON.stringify(sessionItemsAtFollowUp)).toContain("现在提交正式规划");
    expect(hierarchy.session("humanoid-motion-reference")).toBe(
      sessions.get("humanoid-motion-reference")
    );
  });

  it("lets cancellation stop delegated continuation without another model call", async () => {
    const controller = new AbortController();
    let modelCalls = 0;
    let actionCalls = 0;
    const hierarchy = createHumanoidAgentHierarchy({
      provider,
      runtime: delegatedMotionRuntime({
        invoke: async () => {
          actionCalls += 1;
          throw new Error("No action should be synthesized from prose");
        }
      }),
      createModel: (agentId) => agentId === "humanoid-motion-reference"
        ? {
            getResponse: async () => {
              modelCalls += 1;
              return textResponse("我会规划动作，但这一轮不调用工具。");
            },
            getStreamedResponse: () => {
              throw new Error("Streaming is outside this test");
            }
          } as Model
        : modelStub(),
      createSession: (agentId) => new MemorySession({ sessionId: agentId }),
      callModelInputFilter: ({ modelData }) => modelData,
      onDelegatedDecisionFollowUp: () => controller.abort(
        new Error("operator paused delegated continuation")
      )
    });
    const delegate = hierarchy.coordinator.tools.find(
      (entry) => entry.name === "delegate_motion_reference"
    );
    if (!delegate || delegate.type !== "function") {
      throw new Error("Motion delegation tool is missing");
    }

    await expect(delegate.invoke(
      new RunContext({ runId: "cancel-motion-continuation" }),
      "{}",
      {
        signal: controller.signal,
        toolCall: {
          type: "function_call",
          callId: "delegate-motion-cancel",
          name: "delegate_motion_reference",
          arguments: "{}",
          status: "completed"
        }
      }
    )).rejects.toThrow("operator paused delegated continuation");
    expect(modelCalls).toBe(1);
    expect(actionCalls).toBe(0);
  });

  it("does not classify a delegated transport failure as decision recovery", async () => {
    const followUps: unknown[] = [];
    const modelError = new TypeError("fetch failed");
    const hierarchy = createHumanoidAgentHierarchy({
      provider,
      runtime: delegatedMotionRuntime(),
      createModel: (agentId) => agentId === "humanoid-motion-reference"
        ? {
            getResponse: async () => { throw modelError; },
            getStreamedResponse: () => {
              throw new Error("Streaming is outside this test");
            }
          } as Model
        : modelStub(),
      createSession: (agentId) => new MemorySession({ sessionId: agentId }),
      callModelInputFilter: ({ modelData }) => modelData,
      onDelegatedDecisionFollowUp: (event) => {
        followUps.push(event);
      }
    });
    const delegate = hierarchy.coordinator.tools.find(
      (entry) => entry.name === "delegate_motion_reference"
    );
    if (!delegate || delegate.type !== "function") {
      throw new Error("Motion delegation tool is missing");
    }

    await expect(delegate.invoke(
      new RunContext({ runId: "motion-transport-failure" }),
      "{}",
      {
        toolCall: {
          type: "function_call",
          callId: "delegate-motion-transport",
          name: "delegate_motion_reference",
          arguments: "{}",
          status: "completed"
        }
      }
    )).rejects.toBe(modelError);
    expect(followUps).toEqual([]);
  });

  it("applies each resolved profile to only its owning Agent", () => {
    const { maxOutputTokens: _maxOutputTokens, ...unboundedProvider } = provider;
    const configured: ProviderConfig = {
      ...provider,
      agentModels: {
        goal_manager: { ...unboundedProvider, model: "goal-manager", temperature: 0.15 },
        coordinator: { ...provider, model: "coordinator", temperature: 0.1 },
        sentry: { ...provider, model: "sentry", temperature: 0.2 },
        motion: { ...provider, model: "motion", temperature: 0.3 },
        executor: {
          ...provider,
          model: "executor",
          temperature: 0.4,
          reasoningEffort: "high"
        },
        compactor: { ...provider, model: "compactor", temperature: 0.5 }
      }
    };
    const owners = new Map<string, string>();
    const hierarchy = createHumanoidAgentHierarchy({
      provider: configured,
      runtime: {
        invoke: async () => { throw new Error("outside test"); },
        recallEmbodiedHistory: async () => { throw new Error("outside test"); },
        validateCycleEvidence: () => { throw new Error("outside test"); }
      } as never,
      createModel: (agentId, selected) => {
        owners.set(agentId, selected.model);
        return modelStub();
      },
      createSession: (agentId) => new MemorySession({ sessionId: agentId }),
      callModelInputFilter: ({ modelData }) => modelData
    });

    expect(Object.fromEntries(owners)).toEqual({
      "humanoid-goal-manager": "goal-manager",
      "humanoid-sentry": "sentry",
      "humanoid-motion-reference": "motion",
      "humanoid-executor": "executor",
      "humanoid-coordinator": "coordinator"
    });
    expect(hierarchy.coordinator.modelSettings.temperature).toBe(0.1);
    expect(hierarchy.goalManager.modelSettings.temperature).toBe(0.15);
    expect(hierarchy.goalManager.modelSettings).not.toHaveProperty("maxTokens");
    expect(hierarchy.sentry.modelSettings.temperature).toBe(0.2);
    expect(hierarchy.motion.modelSettings.temperature).toBe(0.3);
    expect(hierarchy.executor.modelSettings.temperature).toBe(0.4);
    expect(hierarchy.executor.modelSettings.reasoning).toEqual({ effort: "high" });
    for (const agent of [
      hierarchy.goalManager,
      hierarchy.coordinator,
      hierarchy.sentry,
      hierarchy.motion,
      hierarchy.executor
    ]) {
      expect(agent.modelSettings.toolChoice).toBe("auto");
    }
  });
});

function modelStub(): Model {
  return {
    getResponse: async () => {
      throw new Error("Model calls are outside this construction test");
    },
    getStreamedResponse: () => {
      throw new Error("Model calls are outside this construction test");
    }
  } as unknown as Model;
}

function functionCallModel(toolName: string, args = "{}"): Model {
  return {
    getResponse: async () => ({
      responseId: `response-${toolName}`,
      output: [{
        type: "function_call",
        callId: `call-${toolName}`,
        name: toolName,
        arguments: args
      }],
      usage: new Usage({
        requests: 1,
        inputTokens: 100,
        outputTokens: 10,
        totalTokens: 110
      })
    }),
    getStreamedResponse: () => {
      throw new Error("Streaming is outside this test");
    }
  } as unknown as Model;
}

function functionCallResponse(toolName: string, args = "{}") {
  return {
    responseId: `response-${toolName}`,
    output: [{
      type: "function_call" as const,
      callId: `call-${toolName}`,
      name: toolName,
      arguments: args
    }],
    usage: new Usage({
      requests: 1,
      inputTokens: 100,
      outputTokens: 10,
      totalTokens: 110
    })
  };
}

function textResponse(text: string) {
  return {
    responseId: "response-prose",
    output: [{
      type: "message" as const,
      role: "assistant" as const,
      status: "completed" as const,
      content: [{ type: "output_text" as const, text }]
    }],
    usage: new Usage({
      requests: 1,
      inputTokens: 100,
      outputTokens: 10,
      totalTokens: 110
    })
  };
}

function delegatedMotionRuntime(overrides: Record<string, unknown> = {}) {
  return {
    contextAnchor: () => ({ world_revision: 1 }),
    invoke: async () => { throw new Error("Action invocation is outside this test"); },
    recallEmbodiedHistory: async () => ({ historical_only: true }),
    validateCycleEvidence: () => { throw new Error("Cycle completion is outside this test"); },
    cycleCompletionReadiness: () => ({
      status: "not_ready",
      evidence_transaction_ids: [],
      execution_transaction_id: null,
      observed_after_execution: false,
      reason: "no execution"
    }),
    coordinatorPhase: () => "observe_or_plan",
    executorDelegationAvailable: () => false,
    goalRetirementDelegationAvailable: () => false,
    sentryDelegationAvailable: () => true,
    validateGoalTransition: () => ({ status: "unchanged" }),
    validateSatisfiedGoal: () => ({ status: "unsatisfied" }),
    ...overrides
  } as never;
}

function receipt(
  overrides: Partial<HumanoidActionReceipt>
): HumanoidActionReceipt {
  return {
    transactionId: "transaction",
    agentId: "agent",
    action: "observe_humanoid",
    accepted: true,
    code: "accepted",
    fingerprint: "fingerprint",
    worldBeforeRevision: 0,
    worldAfterRevision: 0,
    frameCount: 0,
    channels: [],
    detail: {},
    committedAt: "2026-08-02T00:00:00.000Z",
    ...overrides
  };
}
