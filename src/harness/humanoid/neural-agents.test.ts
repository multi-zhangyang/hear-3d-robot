import { randomUUID } from "node:crypto";
import {
  MemorySession,
  RunContext,
  Runner,
  Usage,
  type Model,
  type ModelRequest,
  type ModelResponse
} from "@openai/agents";
import { describe, expect, it } from "vitest";
import type { ProviderConfig } from "../../config/load.js";
import {
  appendNeuralPredictionError,
  closeNeuralAuthorityLease,
  consumeNeuralSignals,
  createNeuralHierarchyState,
  establishNeuralSkillCommitment,
  issueNeuralRolloutCertificate,
  issueNeuralAuthorityLease,
  pendingNeuralSignals,
  publishNeuralSignal,
  transitionNeuralHarnessPhase,
  transitionNeuralSkillCommitment,
  type NeuralHierarchyState
} from "../../domain/neural-hierarchy.js";
import type { JsonValue } from "../../domain/schema.js";
import { withAgentInvocation } from "../agent-scope.js";
import {
  createHumanoidNeuralAgentHierarchy,
  type HumanoidNeuralAgentRuntime
} from "./neural-agents.js";
import {
  assertHumanoidNeuralSignalRoute,
  HUMANOID_NEURAL_AGENT_IDS,
  HUMANOID_NEURAL_NODE_BY_ID
} from "./neural-hierarchy-contract.js";

const provider: ProviderConfig = {
  protocol: "openai_compatible",
  baseUrl: "https://example.test/v1",
  model: "scripted-neural-model",
  apiKey: "test-key",
  temperature: 0,
  maxOutputTokens: 2048,
  contextWindowTokens: 32_768,
  compactTriggerTokens: 8192,
  compactRecentModelTurns: 4,
  compactMaxOutputTokens: 2048
};

describe("humanoid neural Agent hierarchy", () => {
  it("keeps malformed compatible delegation in the parent correction loop", async () => {
    const calls: string[] = [];
    const runtime = inMemoryNeuralRuntime();
    const hierarchy = createHumanoidNeuralAgentHierarchy({
      provider,
      runtime,
      createModel: (agentId) => malformedDelegationModel(agentId, calls),
      createSession: (agentId) => new MemorySession({ sessionId: agentId }),
      callModelInputFilter: ({ modelData }) => modelData
    });
    const runner = new Runner({
      tracingDisabled: true,
      modelSettings: { parallelToolCalls: false },
      toolExecution: { maxFunctionToolConcurrency: 2 }
    });

    const result = await withAgentInvocation(
      HUMANOID_NEURAL_AGENT_IDS.executive,
      () => runner.run(hierarchy.root, "value the current Goal", {
        session: hierarchy.session(HUMANOID_NEURAL_AGENT_IDS.executive),
        maxTurns: 8,
        reasoningItemIdPolicy: "omit",
        toolExecution: { maxFunctionToolConcurrency: 1 }
      }),
      false,
      randomUUID()
    );

    expect(result.finalOutput).toMatchObject({ signal_kind: "goal_context" });
    expect(calls).toEqual([
      HUMANOID_NEURAL_AGENT_IDS.executive,
      HUMANOID_NEURAL_AGENT_IDS.executive,
      HUMANOID_NEURAL_AGENT_IDS.executive,
      HUMANOID_NEURAL_AGENT_IDS.goalManager,
      HUMANOID_NEURAL_AGENT_IDS.executive
    ]);
  });

  it("runs Executive -> Goal Valuation -> Executive through a real Agent.asTool episode", async () => {
    const calls: string[] = [];
    const sessions = new Map<string, MemorySession>();
    const runtime = inMemoryNeuralRuntime();
    const hierarchy = createHumanoidNeuralAgentHierarchy({
      provider,
      runtime,
      createModel: (agentId) => scriptedHierarchyModel(agentId, calls),
      createSession: (agentId) => {
        const session = new MemorySession({ sessionId: agentId });
        sessions.set(agentId, session);
        return session;
      },
      callModelInputFilter: ({ modelData }) => modelData
    });
    const runner = new Runner({
      tracingDisabled: true,
      modelSettings: { parallelToolCalls: false },
      toolExecution: { maxFunctionToolConcurrency: 1 }
    });
    const rootEpisodeId = randomUUID();

    const result = await withAgentInvocation(
      HUMANOID_NEURAL_AGENT_IDS.executive,
      () => runner.run(hierarchy.root, "value the current Goal", {
        session: hierarchy.session(HUMANOID_NEURAL_AGENT_IDS.executive),
        maxTurns: 8,
        reasoningItemIdPolicy: "omit",
        toolExecution: { maxFunctionToolConcurrency: 1 }
      }),
      false,
      rootEpisodeId
    );

    expect(result.finalOutput).toMatchObject({
      signal_kind: "goal_context",
      summary: "Goal Valuation returned to Executive",
      payload_json: JSON.stringify({ status: "step_completed" }),
      source_signal_ids: [],
      confidence: 1
    });
    expect(calls).toEqual([
      HUMANOID_NEURAL_AGENT_IDS.executive,
      HUMANOID_NEURAL_AGENT_IDS.goalManager,
      HUMANOID_NEURAL_AGENT_IDS.executive
    ]);
    expect(runtime.neuralHarnessPhase().phase).toBe("perception");

    const signals = Object.values(runtime.neuralHierarchyState().signals);
    expect(signals).toHaveLength(2);
    expect(signals).toEqual(expect.arrayContaining([
      expect.objectContaining({
        direction: "descending",
        source_node_id: HUMANOID_NEURAL_AGENT_IDS.executive,
        target_node_id: HUMANOID_NEURAL_AGENT_IDS.goalManager,
        parent_episode_id: rootEpisodeId,
        status: "consumed"
      }),
      expect.objectContaining({
        direction: "ascending",
        source_node_id: HUMANOID_NEURAL_AGENT_IDS.goalManager,
        target_node_id: HUMANOID_NEURAL_AGENT_IDS.executive,
        parent_episode_id: rootEpisodeId,
        kind: "goal_selected",
        status: "pending"
      })
    ]));
    expect(Object.values(runtime.neuralHierarchyState().authority_leases)).toEqual([
      expect.objectContaining({
        issuing_parent_node_id: HUMANOID_NEURAL_AGENT_IDS.executive,
        target_child_node_id: HUMANOID_NEURAL_AGENT_IDS.goalManager,
        parent_episode_id: rootEpisodeId,
        status: "closed"
      })
    ]);

    const executiveHistory = JSON.stringify(
      await sessions.get(HUMANOID_NEURAL_AGENT_IDS.executive)?.getItems()
    );
    const goalHistory = JSON.stringify(
      await sessions.get(HUMANOID_NEURAL_AGENT_IDS.goalManager)?.getItems()
    );
    expect(executiveHistory).toContain("delegate_goal_valuation");
    expect(executiveHistory).not.toContain("continue_goal_epoch");
    expect(goalHistory).toContain("continue_goal_epoch");
  });

  it("gives every model node one schema-bound neural submission tool", async () => {
    const runtime = inMemoryNeuralRuntime();
    const hierarchy = createHumanoidNeuralAgentHierarchy({
      provider,
      runtime,
      createModel: () => ({
        getResponse: async () => textResponse("unused", "{}"),
        getStreamedResponse: () => {
          throw new Error("Streaming is outside this construction test");
        }
      } as Model),
      createSession: (agentId) => new MemorySession({ sessionId: agentId }),
      callModelInputFilter: ({ modelData }) => modelData
    });

    for (const agent of hierarchy.agents.values()) {
      const submissionTools = agent.tools.filter(
        (candidate) => candidate.name === "submit_neural_output"
      );
      expect(submissionTools).toHaveLength(1);
      expect(submissionTools[0]).toMatchObject({
        type: "function",
        strict: false
      });
    }

    const sceneSubmission = hierarchy.agent(
      HUMANOID_NEURAL_AGENT_IDS.sceneInterpreter
    )?.tools.find((candidate) => candidate.name === "submit_neural_output");
    if (!sceneSubmission || sceneSubmission.type !== "function") {
      throw new Error("Scene Interpreter has no neural submission tool");
    }
    const submitted = await withAgentInvocation(
      HUMANOID_NEURAL_AGENT_IDS.sceneInterpreter,
      () => sceneSubmission.invoke(
        new RunContext(),
        JSON.stringify({
          signal_kind: "scene_interpretation",
          summary: "native structured scene payload",
          payload: { route: { clear: true }, obstacle_ids: ["stone_column"] },
          source_signal_ids: [],
          confidence: 0.95
        }),
        {
          toolCall: {
            type: "function_call",
            callId: "scene-native-neural-output",
            name: "submit_neural_output",
            arguments: "{}",
            status: "completed"
          }
        }
      )
    );
    expect(JSON.parse(String(submitted))).toMatchObject({
      signal_kind: "scene_interpretation",
      payload_json: JSON.stringify({
        route: { clear: true },
        obstacle_ids: ["stone_column"]
      })
    });
  });

  it("keeps malformed neural submission JSON in the same leaf correction loop", async () => {
    const calls: string[] = [];
    const runtime = inMemoryNeuralRuntime();
    let turn = 0;
    const hierarchy = createHumanoidNeuralAgentHierarchy({
      provider,
      runtime,
      createModel: (agentId) => ({
        getResponse: async (request) => {
          calls.push(agentId);
          turn += 1;
          if (turn === 1) {
            return modelResponse("scene-malformed-submission", [{
              type: "function_call",
              callId: "call-scene-malformed-submission",
              name: "submit_neural_output",
              status: "completed",
              arguments: "{\"signal_kind\":\"scene_interpretation\",\"payload\":{\"occluded\"}"
            }]);
          }
          expect(functionResultTexts(request)).toContain(
            "An error occurred while parsing tool arguments. Please try again with valid JSON."
          );
          return functionCallResponse(
            "scene-corrected-submission",
            "submit_neural_output",
            {
              signal_kind: "scene_interpretation",
              summary: "corrected scene interpretation",
              payload: { route: "clear" },
              source_signal_ids: [],
              confidence: 0.9
            }
          );
        },
        getStreamedResponse: () => {
          throw new Error("Streaming is outside this correction-loop test");
        }
      } as Model),
      createSession: (agentId) => new MemorySession({ sessionId: agentId }),
      callModelInputFilter: ({ modelData }) => modelData
    });
    const scene = hierarchy.agent(HUMANOID_NEURAL_AGENT_IDS.sceneInterpreter);
    if (!scene) throw new Error("Scene Interpreter was not registered");
    const runner = new Runner({
      tracingDisabled: true,
      modelSettings: { parallelToolCalls: false },
      toolExecution: { maxFunctionToolConcurrency: 1 }
    });

    const result = await withAgentInvocation(
      HUMANOID_NEURAL_AGENT_IDS.sceneInterpreter,
      () => runner.run(scene, "interpret the bounded current scene", {
        session: hierarchy.session(HUMANOID_NEURAL_AGENT_IDS.sceneInterpreter),
        maxTurns: 4,
        reasoningItemIdPolicy: "omit",
        toolExecution: { maxFunctionToolConcurrency: 1 }
      }),
      false,
      randomUUID()
    );

    expect(result.finalOutput).toMatchObject({
      signal_kind: "scene_interpretation",
      summary: "corrected scene interpretation",
      payload_json: JSON.stringify({ route: "clear" })
    });
    expect(calls).toEqual([
      HUMANOID_NEURAL_AGENT_IDS.sceneInterpreter,
      HUMANOID_NEURAL_AGENT_IDS.sceneInterpreter
    ]);
  });

  it("keeps authority turns required while read-only DeepSeek specialists think", () => {
    const runtime = inMemoryNeuralRuntime();
    const deepSeekProvider: ProviderConfig = {
      ...provider,
      model: "deepseek-v4-flash",
      reasoningEffort: "high",
      toolChoice: "auto"
    };
    const hierarchy = createHumanoidNeuralAgentHierarchy({
      provider: deepSeekProvider,
      runtime,
      createModel: () => ({
        getResponse: async () => textResponse("unused", "{}"),
        getStreamedResponse: () => {
          throw new Error("Streaming is outside this construction test");
        }
      } as Model),
      createSession: (agentId) => new MemorySession({ sessionId: agentId }),
      callModelInputFilter: ({ modelData }) => modelData
    });

    const thinkingSpecialists = new Set([
      HUMANOID_NEURAL_AGENT_IDS.sceneInterpreter,
      HUMANOID_NEURAL_AGENT_IDS.memoryRetriever,
      HUMANOID_NEURAL_AGENT_IDS.affordance,
      HUMANOID_NEURAL_AGENT_IDS.risk,
      HUMANOID_NEURAL_AGENT_IDS.predictive,
      HUMANOID_NEURAL_AGENT_IDS.recovery
    ]);
    for (const [agentId, agent] of hierarchy.agents) {
      const thinking = thinkingSpecialists.has(agentId) ? "enabled" : "disabled";
      expect(agent.modelSettings.toolChoice).toBe(
        thinkingSpecialists.has(agentId) ? "auto" : "required"
      );
      expect(agent.modelSettings.providerData).toMatchObject({
        thinking: { type: thinking },
        providerOptions: {
          "configured-openai-compatible": {
            thinking: { type: thinking }
          }
        }
      });
    }
  });

  it("derives one legal active-Goal memory query from a high-level retrieval intent", async () => {
    let recalledWith: Record<string, unknown> | undefined;
    const calls: string[] = [];
    const runtime = inMemoryNeuralRuntime({
      contextAnchor: {
        active_goal: {
          summary: "Enter the courtyard beacon",
          predicates: [{
            type: "robot_in_zone",
            zone_id: "courtyard_beacon",
            tolerance: 0.2
          }]
        }
      },
      onRecall: (request) => { recalledWith = request; },
      recallResult: {
        historical_only: true,
        current_world_revision: 12,
        ordered_source_refs: ["action:navigation-1"],
        experiences: [{
          source_ref: "action:navigation-1",
          outcome: "succeeded",
          predicate_types: ["robot_in_zone"],
          zone_ids: ["courtyard_beacon"]
        }],
        actions: [{
          source_ref: "action:navigation-1",
          transactionId: "navigation-1",
          action: "execute_humanoid_navigation",
          accepted: true,
          code: "navigation_completed",
          worldBeforeRevision: 1,
          worldAfterRevision: 12,
          frameCount: 11
        }],
        missing_source_refs: [],
        next_before_sequence: null,
        next_before_experience_sequence: null
      }
    });
    const hierarchy = createHumanoidNeuralAgentHierarchy({
      provider,
      runtime,
      createModel: (agentId) => ({
        getResponse: async () => {
          calls.push(agentId);
          return functionCallResponse(
            "retrieve-active-goal-memory",
            "retrieve_relevant_embodied_memory",
            {
              retrieval_mode: "active_goal",
              outcome_scope: "successful"
            }
          );
        },
        getStreamedResponse: () => {
          throw new Error("Streaming is outside this memory boundary test");
        }
      } as Model),
      createSession: (agentId) => new MemorySession({ sessionId: agentId }),
      callModelInputFilter: ({ modelData }) => modelData
    });
    const memory = hierarchy.agent(HUMANOID_NEURAL_AGENT_IDS.memoryRetriever);
    if (!memory) throw new Error("Memory Retriever was not registered");
    const runner = new Runner({
      tracingDisabled: true,
      modelSettings: { parallelToolCalls: false },
      toolExecution: { maxFunctionToolConcurrency: 1 }
    });

    const result = await withAgentInvocation(
      HUMANOID_NEURAL_AGENT_IDS.memoryRetriever,
      () => runner.run(memory, "retrieve relevant memory", {
        session: hierarchy.session(HUMANOID_NEURAL_AGENT_IDS.memoryRetriever),
        maxTurns: 4,
        reasoningItemIdPolicy: "omit",
        toolExecution: { maxFunctionToolConcurrency: 1 }
      }),
      false,
      randomUUID()
    );

    expect(result.finalOutput).toMatchObject({
      signal_kind: "memory_retrieval",
      source_signal_ids: [],
      confidence: 1
    });
    expect(calls).toEqual([HUMANOID_NEURAL_AGENT_IDS.memoryRetriever]);
    expect(recalledWith).toEqual({
      outcomes: ["succeeded"],
      predicate_types: ["robot_in_zone"],
      zone_ids: ["courtyard_beacon"],
      limit: 16
    });
    expect(recalledWith).not.toHaveProperty("source_refs");
  });

  it("forks Scene and Memory only inside their Perception Manager and joins both returns", async () => {
    const calls: string[] = [];
    const sessions = new Map<string, MemorySession>();
    const fork = twoPartyBarrier();
    const runtime = inMemoryNeuralRuntime({
      contextAnchor: {
        agent_id: "invocation-scoped-child",
        neural_hierarchy: {
          directed_signals: [{ duplicate: "must_not_be_injected_twice" }]
        }
      }
    });
    await runtime.transitionNeuralHarnessPhase({
      phase: "perception",
      enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
      reason: "perception_fork_join_smoke",
      goalEpochId: "goal-epoch-perception-smoke",
      commitmentId: null
    });
    const hierarchy = createHumanoidNeuralAgentHierarchy({
      provider,
      runtime,
      createModel: (agentId) => perceptionHierarchyModel(agentId, calls, fork),
      createSession: (agentId) => {
        const session = new MemorySession({ sessionId: agentId });
        sessions.set(agentId, session);
        return session;
      },
      callModelInputFilter: ({ modelData }) => modelData
    });
    const actionSelection = hierarchy.agent(
      HUMANOID_NEURAL_AGENT_IDS.actionSelection
    );
    const delegate = actionSelection?.tools.find(
      (candidate) => candidate.name === "delegate_perception_manager"
    );
    if (!delegate || delegate.type !== "function") {
      throw new Error("Action Selection has no owned Perception Manager tool");
    }
    const actionSelectionEpisodeId = randomUUID();
    const output = await withAgentInvocation(
      HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      () => delegate.invoke(
        new RunContext(),
        JSON.stringify({
          signal_kind: "goal_context",
          source_signal_ids: [],
          ttl_revisions: 64,
          priority: 80
        }),
        {
          toolCall: {
            type: "function_call",
            callId: "action-selection-perception-smoke",
            name: "delegate_perception_manager",
            arguments: "{}",
            status: "completed"
          }
        }
      ),
      false,
      actionSelectionEpisodeId
    );

    expect(JSON.parse(String(output))).toMatchObject({
      signal_kind: "perceptual_belief",
      summary: "Perception Manager joined current scene and relevant memory",
      confidence: 0.95
    });
    expect(calls).toEqual([
      HUMANOID_NEURAL_AGENT_IDS.perceptionManager,
      HUMANOID_NEURAL_AGENT_IDS.perceptionManager,
      expect.any(String),
      expect.any(String),
      HUMANOID_NEURAL_AGENT_IDS.perceptionManager
    ]);
    expect(new Set(calls.slice(2, 4))).toEqual(new Set([
      HUMANOID_NEURAL_AGENT_IDS.sceneInterpreter,
      HUMANOID_NEURAL_AGENT_IDS.memoryRetriever
    ]));
    expect(fork.arrivals()).toEqual(new Set([
      HUMANOID_NEURAL_AGENT_IDS.sceneInterpreter,
      HUMANOID_NEURAL_AGENT_IDS.memoryRetriever
    ]));
    expect(runtime.neuralHarnessPhase().phase).toBe("skill_proposal");

    const state = runtime.neuralHierarchyState();
    const perceptionRequest = Object.values(state.signals).find((signal) => (
      signal.direction === "descending"
        && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.actionSelection
        && signal.target_node_id === HUMANOID_NEURAL_AGENT_IDS.perceptionManager
    ));
    expect(perceptionRequest).toBeDefined();
    const joinedSignals = Object.values(state.signals).filter((signal) => (
      signal.direction === "ascending"
        && signal.target_node_id === HUMANOID_NEURAL_AGENT_IDS.perceptionManager
        && [
          "sensory_evidence",
          "scene_interpretation",
          "memory_retrieval"
        ].includes(signal.kind)
    ));
    expect(joinedSignals).toHaveLength(3);
    expect(new Set(joinedSignals.map((signal) => signal.parent_episode_id))).toEqual(
      new Set([perceptionRequest!.invocation_id])
    );
    expect(joinedSignals.every((signal) => signal.status === "consumed")).toBe(true);
    const perceptionReturn = Object.values(state.signals).find((signal) => (
      signal.kind === "perceptual_belief"
        && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.perceptionManager
        && signal.target_node_id === HUMANOID_NEURAL_AGENT_IDS.actionSelection
    ));
    expect(perceptionReturn?.causal_parent_ids).toEqual(expect.arrayContaining(
      joinedSignals.map((signal) => signal.signal_id)
    ));
    expect(Object.values(state.authority_leases).every(
      (lease) => lease.status === "closed"
    )).toBe(true);

    const sceneItems = await sessions.get(
      HUMANOID_NEURAL_AGENT_IDS.sceneInterpreter
    )?.getItems() ?? [];
    const sceneHistory = JSON.stringify(sceneItems);
    const memoryHistory = JSON.stringify(
      await sessions.get(HUMANOID_NEURAL_AGENT_IDS.memoryRetriever)?.getItems()
    );
    expect(sceneHistory).toContain("scene branch current evidence");
    expect(sceneHistory).not.toContain("memory branch historical evidence");
    expect(memoryHistory).toContain("memory branch historical evidence");
    expect(memoryHistory).not.toContain("scene branch current evidence");
    const sceneInvocation = invocationPayload(sceneItems);
    expect(sceneInvocation.directed_signals).toHaveLength(1);
    expect(sceneInvocation.anchor.neural_hierarchy).not.toHaveProperty(
      "directed_signals"
    );
    expect(sceneInvocation.anchor).not.toHaveProperty("interaction");
    expect(sceneInvocation.anchor).not.toHaveProperty("goal_context");
    expect(sceneInvocation.anchor).not.toHaveProperty("goal_dag");
    const sceneSignal = sceneInvocation.directed_signals[0] as {
      payload?: { causal_inputs?: Array<Record<string, unknown>> };
    };
    expect(sceneSignal.payload?.causal_inputs).toHaveLength(1);
    expect(sceneSignal.payload?.causal_inputs?.[0]?.payload).toMatchObject({
      action: "observe_humanoid",
      accepted: true
    });
    expect(sceneSignal.payload?.causal_inputs?.[0]?.payload).not.toHaveProperty(
      "causal_inputs"
    );
  });

  it("lets Action Selection own the Sensorimotor Affordance/Risk fork, join it, and bind one Skill commitment", async () => {
    const calls: string[] = [];
    const sessions = new Map<string, MemorySession>();
    const fork = twoPartyBarrier();
    const runtime = inMemoryNeuralRuntime();
    const goalEpochId = "goal-epoch-sensorimotor-smoke";
    await runtime.transitionNeuralHarnessPhase({
      phase: "goal_valuation",
      enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
      reason: "sensorimotor_smoke_goal_valuation",
      goalEpochId: null,
      commitmentId: null
    });
    await runtime.transitionNeuralHarnessPhase({
      phase: "perception",
      enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
      reason: "sensorimotor_smoke_goal_selected",
      goalEpochId,
      commitmentId: null
    });
    await runtime.transitionNeuralHarnessPhase({
      phase: "skill_proposal",
      enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      reason: "sensorimotor_fork_join_smoke",
      goalEpochId,
      commitmentId: null
    });
    const actionSelectionInvocationId = randomUUID();
    const perceptionInvocationId = randomUUID();
    const perceptionLease = await runtime.issueNeuralAuthorityLease({
      issuingParentNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      targetChildNodeId: HUMANOID_NEURAL_AGENT_IDS.perceptionManager,
      allowedSignalKinds: ["perceptual_belief"],
      invocationId: perceptionInvocationId,
      parentInvocationId: null,
      parentEpisodeId: actionSelectionInvocationId,
      ttlRevisions: 64
    });
    const belief = await runtime.publishNeuralSignal({
      kind: "perceptual_belief",
      pathway: "perceptual_association",
      direction: "ascending",
      sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.perceptionManager,
      targetNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      ttlRevisions: 64,
      priority: 90,
      sourceAuthorityLeaseId: perceptionLease.lease_id,
      invocationId: perceptionInvocationId,
      parentInvocationId: null,
      parentEpisodeId: perceptionLease.parent_episode_id,
      payload: {
        world_revision: 0,
        support_polygon_stable: true,
        target_reachable: true
      }
    });
    await runtime.closeNeuralAuthorityLease({
      leaseId: perceptionLease.lease_id,
      closedByNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      reason: "perception_belief_ready"
    });
    const hierarchy = createHumanoidNeuralAgentHierarchy({
      provider,
      runtime,
      createModel: (agentId) => sensorimotorHierarchyModel(
        agentId,
        calls,
        fork,
        runtime,
        belief.signal_id,
        goalEpochId
      ),
      createSession: (agentId) => {
        const session = new MemorySession({ sessionId: agentId });
        sessions.set(agentId, session);
        return session;
      },
      callModelInputFilter: ({ modelData }) => modelData
    });
    const actionSelection = hierarchy.agent(
      HUMANOID_NEURAL_AGENT_IDS.actionSelection
    );
    if (!actionSelection) throw new Error("Action Selection Agent is absent");
    const runner = new Runner({
      tracingDisabled: true,
      modelSettings: { parallelToolCalls: true },
      toolExecution: { maxFunctionToolConcurrency: 2 }
    });

    const result = await withAgentInvocation(
      HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      () => runner.run(actionSelection, "propose and commit one bounded Skill", {
        session: hierarchy.session(HUMANOID_NEURAL_AGENT_IDS.actionSelection),
        maxTurns: 10,
        reasoningItemIdPolicy: "omit",
        toolExecution: { maxFunctionToolConcurrency: 2 }
      }),
      false,
      actionSelectionInvocationId
    );

    expect(result.finalOutput).toMatchObject({
      signal_kind: "skill_commitment",
      summary: "skill_committed",
      source_signal_ids: [expect.any(String)],
      confidence: 1
    });
    expect(calls).toEqual([
      HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
      expect.any(String),
      expect.any(String),
      HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
      HUMANOID_NEURAL_AGENT_IDS.actionSelection
    ]);
    expect(new Set(calls.slice(2, 4))).toEqual(new Set([
      HUMANOID_NEURAL_AGENT_IDS.affordance,
      HUMANOID_NEURAL_AGENT_IDS.risk
    ]));
    expect(fork.arrivals()).toEqual(new Set([
      HUMANOID_NEURAL_AGENT_IDS.affordance,
      HUMANOID_NEURAL_AGENT_IDS.risk
    ]));

    const state = runtime.neuralHierarchyState();
    const sensorimotorRequest = Object.values(state.signals).find((signal) => (
      signal.direction === "descending"
        && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.actionSelection
        && signal.target_node_id === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager
        && signal.kind === "perceptual_belief"
    ));
    expect(sensorimotorRequest).toBeDefined();
    const assessments = Object.values(state.signals).filter((signal) => (
      signal.direction === "ascending"
        && signal.target_node_id === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager
        && ["affordance_hypothesis", "risk_assessment"].includes(signal.kind)
    ));
    expect(assessments).toHaveLength(2);
    expect(new Set(assessments.map((signal) => signal.parent_episode_id))).toEqual(
      new Set([sensorimotorRequest!.invocation_id])
    );
    expect(assessments.every((signal) => signal.status === "consumed")).toBe(true);
    const proposal = Object.values(state.signals).find((signal) => (
      signal.kind === "skill_proposal"
        && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager
        && signal.target_node_id === HUMANOID_NEURAL_AGENT_IDS.actionSelection
    ));
    expect(proposal?.causal_parent_ids).toEqual(expect.arrayContaining([
      sensorimotorRequest!.signal_id,
      ...assessments.map((signal) => signal.signal_id)
    ]));
    expect(state.active_skill_commitment).toMatchObject({
      goal_epoch_id: goalEpochId,
      owner_node_id: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      skill: "navigate_to_zone",
      state: "committed",
      source_signal_ids: [proposal!.signal_id]
    });
    expect(result.finalOutput?.source_signal_ids).toEqual([proposal!.signal_id]);
    expect(state.harness_phase).toMatchObject({
      phase: "commitment_authorization",
      goal_epoch_id: goalEpochId
    });
    expect(Object.values(state.authority_leases).every(
      (lease) => lease.status === "closed"
    )).toBe(true);

    const affordanceHistory = JSON.stringify(
      await sessions.get(HUMANOID_NEURAL_AGENT_IDS.affordance)?.getItems()
    );
    const riskHistory = JSON.stringify(
      await sessions.get(HUMANOID_NEURAL_AGENT_IDS.risk)?.getItems()
    );
    expect(affordanceHistory).toContain("target affords bounded walking");
    expect(affordanceHistory).not.toContain("balance risk remains bounded");
    expect(riskHistory).toContain("balance risk remains bounded");
    expect(riskHistory).not.toContain("target affords bounded walking");
  });

  it("replaces a failed Skill only through Executive -> Action Selection -> Sensorimotor -> exclusive Recovery", async () => {
    const calls: string[] = [];
    const actions: string[] = [];
    const sessions = new Map<string, MemorySession>();
    const runtime = inMemoryNeuralRuntime({ onAction: (action) => actions.push(action) });
    const goalEpochId = "goal-epoch-recovery-smoke";
    await advanceToPerception(runtime, goalEpochId);
    const initialBelief = await publishPriorPerceptualBelief(runtime, {
      world_revision: 0,
      blocked_path_visible: false
    });
    await runtime.transitionNeuralHarnessPhase({
      phase: "skill_proposal",
      enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      reason: "recovery_smoke_initial_proposal"
    });
    await runtime.transitionNeuralHarnessPhase({
      phase: "commitment_authorization",
      enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      reason: "recovery_smoke_initial_authorization"
    });
    const failed = await runtime.establishNeuralSkillCommitment({
      ownerNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      goalEpochId,
      skill: "walk_straight_into_blocked_path",
      terminationContract: { type: "robot_at", tolerance: 0.35 },
      sourceSignalIds: [initialBelief.signal_id]
    });
    await runtime.transitionNeuralHarnessPhase({
      phase: "motor_assessment",
      enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      reason: "recovery_smoke_initial_commitment"
    });
    await runtime.transitionNeuralSkillCommitment({
      commitmentId: failed.commitment_id,
      ownerNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      state: "failed",
      sourceSignalIds: []
    });
    await runtime.transitionNeuralHarnessPhase({
      phase: "recovery",
      enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
      reason: "recovery_smoke_failure_feedback"
    });
    const executiveInvocationId = randomUUID();
    const failedActionSelectionInvocationId = randomUUID();
    const failedActionSelectionLease = await runtime.issueNeuralAuthorityLease({
      issuingParentNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
      targetChildNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      allowedSignalKinds: ["skill_failed"],
      correctionScope: "pathway",
      invocationId: failedActionSelectionInvocationId,
      parentInvocationId: null,
      parentEpisodeId: executiveInvocationId,
      ttlRevisions: 64
    });
    const failureToExecutive = await runtime.publishNeuralSignal({
      kind: "skill_failed",
      pathway: "sensorimotor_selection",
      direction: "ascending",
      sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      targetNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
      ttlRevisions: 64,
      priority: 100,
      sourceAuthorityLeaseId: failedActionSelectionLease.lease_id,
      invocationId: failedActionSelectionInvocationId,
      parentInvocationId: null,
      parentEpisodeId: failedActionSelectionLease.parent_episode_id,
      payload: {
        reason: "blocked_path_destabilized_controller",
        failed_commitment_id: failed.commitment_id
      }
    });
    await runtime.closeNeuralAuthorityLease({
      leaseId: failedActionSelectionLease.lease_id,
      closedByNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
      reason: "failed_action_selection_episode_returned"
    });
    const lowerFailureInvocationId = randomUUID();
    const lowerFailureLease = await runtime.issueNeuralAuthorityLease({
      issuingParentNodeId: HUMANOID_NEURAL_AGENT_IDS.executor,
      targetChildNodeId: HUMANOID_NEURAL_AGENT_IDS.reflex,
      allowedSignalKinds: ["skill_failed"],
      correctionScope: "pathway",
      invocationId: lowerFailureInvocationId,
      parentInvocationId: randomUUID(),
      parentEpisodeId: randomUUID(),
      ttlRevisions: 64
    });
    const lowerFailure = await runtime.publishNeuralSignal({
      kind: "skill_failed",
      pathway: "ascending_feedback",
      direction: "reentrant",
      sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.reflex,
      targetNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
      ttlRevisions: 64,
      priority: 100,
      sourceAuthorityLeaseId: lowerFailureLease.lease_id,
      invocationId: lowerFailureInvocationId,
      parentInvocationId: lowerFailureLease.parent_invocation_id,
      parentEpisodeId: lowerFailureLease.parent_episode_id,
      payload: { reason: "same_bounded_failure_domain" }
    });
    await runtime.closeNeuralAuthorityLease({
      leaseId: lowerFailureLease.lease_id,
      closedByNodeId: HUMANOID_NEURAL_AGENT_IDS.executor,
      reason: "lower_failure_feedback_returned"
    });

    const hierarchy = createHumanoidNeuralAgentHierarchy({
      provider,
      runtime,
      createModel: (agentId) => recoveryHierarchyModel(
        agentId,
        calls,
        runtime,
        goalEpochId,
        failureToExecutive.signal_id
      ),
      createSession: (agentId) => {
        const session = new MemorySession({ sessionId: agentId });
        sessions.set(agentId, session);
        return session;
      },
      callModelInputFilter: ({ modelData }) => modelData
    });
    const runner = new Runner({
      tracingDisabled: true,
      modelSettings: { parallelToolCalls: false },
      toolExecution: { maxFunctionToolConcurrency: 1 }
    });

    const result = await withAgentInvocation(
      HUMANOID_NEURAL_AGENT_IDS.executive,
      () => runner.run(hierarchy.root, "recover the failed Skill through the owned hierarchy", {
        session: hierarchy.session(HUMANOID_NEURAL_AGENT_IDS.executive),
        maxTurns: 12,
        reasoningItemIdPolicy: "omit",
        toolExecution: { maxFunctionToolConcurrency: 1 }
      }),
      false,
      executiveInvocationId
    );

    expect(result.finalOutput).toMatchObject({
      signal_kind: "skill_commitment",
      summary: "skill_committed",
      confidence: 1
    });
    expect(actions).toEqual([]);
    expect(calls).toEqual(expect.arrayContaining([
      HUMANOID_NEURAL_AGENT_IDS.executive,
      HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
      HUMANOID_NEURAL_AGENT_IDS.recovery
    ]));
    const state = runtime.neuralHierarchyState();
    const edges = Object.values(state.signals).map((signal) => (
      `${signal.source_node_id}->${signal.target_node_id}:${signal.kind}`
    ));
    expect(edges).toEqual(expect.arrayContaining([
      `${HUMANOID_NEURAL_AGENT_IDS.executive}->${HUMANOID_NEURAL_AGENT_IDS.actionSelection}:skill_failed`,
      `${HUMANOID_NEURAL_AGENT_IDS.actionSelection}->${HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager}:skill_failed`,
      `${HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager}->${HUMANOID_NEURAL_AGENT_IDS.recovery}:skill_failed`,
      `${HUMANOID_NEURAL_AGENT_IDS.recovery}->${HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager}:skill_proposal`,
      `${HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager}->${HUMANOID_NEURAL_AGENT_IDS.actionSelection}:skill_proposal`,
      `${HUMANOID_NEURAL_AGENT_IDS.actionSelection}->${HUMANOID_NEURAL_AGENT_IDS.executive}:skill_commitment`
    ]));
    const recoveryLease = Object.values(state.authority_leases).find((lease) => (
      lease.issuing_parent_node_id === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager
        && lease.target_child_node_id === HUMANOID_NEURAL_AGENT_IDS.recovery
    ));
    expect(recoveryLease).toMatchObject({
      exclusive: true,
      status: "closed",
      correction_scope: "pathway"
    });
    const routedRecoveryInput = Object.values(state.signals).find((signal) => (
      signal.kind === "skill_failed"
        && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.actionSelection
        && signal.target_node_id === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager
    ));
    expect(routedRecoveryInput?.status).toBe("consumed");
    expect(signalHasAncestor(
      state,
      routedRecoveryInput!,
      failureToExecutive.signal_id
    )).toBe(true);
    expect(state.signals[lowerFailure.signal_id]?.status).toBe("consumed");
    expect(state.active_skill_commitment).toMatchObject({
      goal_epoch_id: goalEpochId,
      owner_node_id: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      skill: "retreat",
      state: "committed"
    });
    expect(state.active_skill_commitment?.commitment_id).not.toBe(failed.commitment_id);
    expect(state.harness_phase.phase).toBe("motor_assessment");
    expect(Object.values(state.authority_leases).every(
      (lease) => lease.status === "closed"
    )).toBe(true);
    expect(new Set(sessions.values()).size).toBe(sessions.size);
  });

  it("returns Recovery escalation through every structural parent before Goal Valuation", async () => {
    const calls: string[] = [];
    const runtime = inMemoryNeuralRuntime();
    const goalEpochId = "goal-epoch-recovery-escalation";
    await advanceToPerception(runtime, goalEpochId);
    await runtime.transitionNeuralHarnessPhase({
      phase: "recovery",
      enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
      reason: "recovery_escalation_smoke"
    });
    const executiveInvocationId = randomUUID();
    const failedActionSelectionInvocationId = randomUUID();
    const failedActionSelectionLease = await runtime.issueNeuralAuthorityLease({
      issuingParentNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
      targetChildNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      allowedSignalKinds: ["skill_failed"],
      correctionScope: "pathway",
      invocationId: failedActionSelectionInvocationId,
      parentInvocationId: null,
      parentEpisodeId: executiveInvocationId,
      ttlRevisions: 64
    });
    const failureToExecutive = await runtime.publishNeuralSignal({
      kind: "skill_failed",
      pathway: "sensorimotor_selection",
      direction: "ascending",
      sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      targetNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
      ttlRevisions: 64,
      priority: 100,
      sourceAuthorityLeaseId: failedActionSelectionLease.lease_id,
      invocationId: failedActionSelectionInvocationId,
      parentInvocationId: null,
      parentEpisodeId: failedActionSelectionLease.parent_episode_id,
      payload: { reason: "failure_outside_pathway_recovery_scope" }
    });
    await runtime.closeNeuralAuthorityLease({
      leaseId: failedActionSelectionLease.lease_id,
      closedByNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
      reason: "failed_action_selection_episode_returned"
    });

    const hierarchy = createHumanoidNeuralAgentHierarchy({
      provider,
      runtime,
      createModel: (agentId) => recoveryEscalationHierarchyModel(
        agentId,
        calls,
        runtime,
        failureToExecutive.signal_id
      ),
      createSession: (agentId) => new MemorySession({ sessionId: agentId }),
      callModelInputFilter: ({ modelData }) => modelData
    });
    const runner = new Runner({
      tracingDisabled: true,
      modelSettings: { parallelToolCalls: false },
      toolExecution: { maxFunctionToolConcurrency: 1 }
    });
    const result = await withAgentInvocation(
      HUMANOID_NEURAL_AGENT_IDS.executive,
      () => runner.run(hierarchy.root, "escalate bounded Recovery through the owned hierarchy", {
        session: hierarchy.session(HUMANOID_NEURAL_AGENT_IDS.executive),
        maxTurns: 12,
        reasoningItemIdPolicy: "omit",
        toolExecution: { maxFunctionToolConcurrency: 1 }
      }),
      false,
      executiveInvocationId
    );

    expect(result.finalOutput).toMatchObject({ signal_kind: "escalation" });
    const state = runtime.neuralHierarchyState();
    expect(state.harness_phase.phase).toBe("goal_valuation");
    expect(calls).toEqual(expect.arrayContaining([
      HUMANOID_NEURAL_AGENT_IDS.executive,
      HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
      HUMANOID_NEURAL_AGENT_IDS.recovery
    ]));
    const escalationEdges = Object.values(state.signals)
      .filter((signal) => signal.kind === "escalation")
      .map((signal) => `${signal.source_node_id}->${signal.target_node_id}`);
    expect(escalationEdges).toEqual(expect.arrayContaining([
      `${HUMANOID_NEURAL_AGENT_IDS.recovery}->${HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager}`,
      `${HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager}->${HUMANOID_NEURAL_AGENT_IDS.actionSelection}`,
      `${HUMANOID_NEURAL_AGENT_IDS.actionSelection}->${HUMANOID_NEURAL_AGENT_IDS.executive}`
    ]));
    expect(Object.values(state.authority_leases).every(
      (lease) => lease.status === "closed"
    )).toBe(true);
  });

  it("certifies one committed Skill through Premotor, Motor Intent, MuJoCo rollout, Predictive, and the sole Serial Executor", async () => {
    const calls: string[] = [];
    const actions: string[] = [];
    const runtime = inMemoryNeuralRuntime({
      neuralExecutionAvailable: true,
      onAction: (action) => actions.push(action)
    });
    const goalEpochId = "goal-epoch-certified-motion-smoke";
    await advanceToPerception(runtime, goalEpochId);
    await runtime.transitionNeuralHarnessPhase({
      phase: "skill_proposal",
      enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      reason: "certified_motion_belief_ready",
      goalEpochId,
      commitmentId: null
    });
    const belief = await publishPriorPerceptualBelief(runtime, {
      target_reachable: true,
      support_polygon_stable: true
    });
    const actionSelectionInvocationId = randomUUID();
    const proposalInvocationId = randomUUID();
    const proposalLease = await runtime.issueNeuralAuthorityLease({
      issuingParentNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      targetChildNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
      allowedSignalKinds: ["skill_proposal"],
      invocationId: proposalInvocationId,
      parentInvocationId: null,
      parentEpisodeId: actionSelectionInvocationId,
      ttlRevisions: 64
    });
    const proposal = await runtime.publishNeuralSignal({
      kind: "skill_proposal",
      pathway: "sensorimotor_selection",
      direction: "ascending",
      sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
      targetNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      ttlRevisions: 64,
      priority: 90,
      causalParentIds: [belief.signal_id],
      sourceAuthorityLeaseId: proposalLease.lease_id,
      invocationId: proposalInvocationId,
      parentInvocationId: null,
      parentEpisodeId: proposalLease.parent_episode_id,
      payload: boundedNavigateProposal("zone-certified-motion")
    });
    await runtime.closeNeuralAuthorityLease({
      leaseId: proposalLease.lease_id,
      closedByNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      reason: "skill_proposal_seeded"
    });
    await runtime.transitionNeuralHarnessPhase({
      phase: "commitment_authorization",
      enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      reason: "skill_proposal_ready",
      goalEpochId,
      commitmentId: null
    });
    const commitment = await runtime.establishNeuralSkillCommitment({
      ownerNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      goalEpochId,
      skill: "navigate_to_zone",
      terminationContract: { type: "robot_at", tolerance: 0.35 },
      sourceSignalIds: [proposal.signal_id]
    });
    await runtime.transitionNeuralHarnessPhase({
      phase: "motor_assessment",
      enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      reason: "skill_commitment_authorized",
      goalEpochId,
      commitmentId: commitment.commitment_id
    });
    const hierarchy = createHumanoidNeuralAgentHierarchy({
      provider,
      runtime,
      createModel: (agentId) => certifiedMotionHierarchyModel(
        agentId,
        calls,
        runtime
      ),
      createSession: (agentId) => new MemorySession({ sessionId: agentId }),
      callModelInputFilter: ({ modelData }) => modelData
    });
    const actionSelection = hierarchy.agent(
      HUMANOID_NEURAL_AGENT_IDS.actionSelection
    );
    if (!actionSelection) throw new Error("Action Selection Agent is absent");
    const runner = new Runner({
      tracingDisabled: true,
      modelSettings: { parallelToolCalls: false },
      toolExecution: { maxFunctionToolConcurrency: 1 }
    });

    const authorization = await withAgentInvocation(
      HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      () => runner.run(actionSelection, "certify the committed Skill", {
        session: hierarchy.session(HUMANOID_NEURAL_AGENT_IDS.actionSelection),
        maxTurns: 16,
        reasoningItemIdPolicy: "omit",
        toolExecution: { maxFunctionToolConcurrency: 2 }
      }),
      false,
      actionSelectionInvocationId
    );
    expect(authorization.finalOutput).toMatchObject({
      signal_kind: "skill_commitment",
      summary: "skill_executing",
      source_signal_ids: [expect.any(String)]
    });
    const certified = runtime.neuralHierarchyState();
    expect(certified.active_skill_commitment?.state).toBe("executing");
    expect(calls).not.toContain(HUMANOID_NEURAL_AGENT_IDS.affordance);
    expect(calls).not.toContain(HUMANOID_NEURAL_AGENT_IDS.risk);
    expect(Object.values(certified.signals)).toContainEqual(expect.objectContaining({
      kind: "skill_proposal",
      direction: "descending",
      source_node_id: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
      target_node_id: HUMANOID_NEURAL_AGENT_IDS.premotor,
      causal_parent_ids: [proposal.signal_id],
      payload: proposal.payload
    }));
    expect(Object.values(certified.rollout_certificates)).toEqual([
      expect.objectContaining({
        status: "active",
        commitment_id: commitment.commitment_id,
        planning_action: "plan_humanoid_navigation",
        issued_by_node_id: HUMANOID_NEURAL_AGENT_IDS.predictive
      })
    ]);
    const reentrantRollout = Object.values(certified.signals).find((signal) => (
      signal.direction === "reentrant"
        && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.rolloutGate
        && signal.target_node_id === HUMANOID_NEURAL_AGENT_IDS.predictive
    ));
    const predictiveAcceptance = Object.values(certified.signals).find((signal) => (
      signal.kind === "forward_prediction"
        && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.predictive
    ));
    expect(predictiveAcceptance?.causal_parent_ids).toContain(
      reentrantRollout?.signal_id
    );
    expect(actions).toEqual(["plan_humanoid_navigation"]);

    await runtime.transitionNeuralHarnessPhase({
      phase: "execution",
      enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      reason: "predictive_certificate_authorized",
      goalEpochId,
      commitmentId: commitment.commitment_id
    });
    const sensorimotorTool = actionSelection.tools.find(
      (candidate) => candidate.name === "delegate_sensorimotor_manager"
    );
    if (!sensorimotorTool || sensorimotorTool.type !== "function") {
      throw new Error("Action Selection has no owned Sensorimotor Manager tool");
    }
    const executionOutput = await withAgentInvocation(
      HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      () => sensorimotorTool.invoke(
        new RunContext(),
        JSON.stringify({
          signal_kind: "skill_commitment",
          source_signal_ids: authorization.finalOutput!.source_signal_ids,
          ttl_revisions: 64,
          priority: 100
        }),
        {
          toolCall: {
            type: "function_call",
            callId: "action-selection-certified-execution",
            name: "delegate_sensorimotor_manager",
            arguments: "{}",
            status: "completed"
          }
        }
      ),
      false,
      randomUUID()
    );
    expect(JSON.parse(String(executionOutput))).toMatchObject({
      signal_kind: "skill_completed",
      source_signal_ids: [expect.any(String)]
    });
    expect(actions).toEqual([
      "plan_humanoid_navigation",
      "execute_humanoid_navigation"
    ]);
    const executed = runtime.neuralHierarchyState();
    expect(executed.harness_phase.phase).toBe("feedback");
    const executorIntent = Object.values(executed.signals).find((signal) => (
      signal.kind === "motor_intent"
        && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.executor
        && signal.target_node_id === HUMANOID_NEURAL_AGENT_IDS.reflex
    ));
    const bodyIntent = Object.values(executed.signals).find((signal) => (
      signal.kind === "motor_intent"
        && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.reflex
        && signal.target_node_id === HUMANOID_NEURAL_AGENT_IDS.body
    ));
    const bodySensation = Object.values(executed.signals).find((signal) => (
      signal.kind === "sensory_evidence"
        && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.body
        && signal.target_node_id === HUMANOID_NEURAL_AGENT_IDS.reflex
    ));
    const reflexReceipt = Object.values(executed.signals).find((signal) => (
      signal.kind === "execution_receipt"
        && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.reflex
        && signal.target_node_id === HUMANOID_NEURAL_AGENT_IDS.executor
    ));
    expect(executorIntent).toBeDefined();
    expect(bodyIntent?.causal_parent_ids).toEqual([executorIntent!.signal_id]);
    expect(bodySensation?.causal_parent_ids).toEqual([bodyIntent!.signal_id]);
    expect(reflexReceipt?.causal_parent_ids).toEqual([bodySensation!.signal_id]);
    const physicalSignals = Object.values(executed.signals).filter((signal) => (
      signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.executor
        && signal.target_node_id === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager
    ));
    expect(physicalSignals.map((signal) => signal.kind)).toEqual(expect.arrayContaining([
      "execution_receipt",
      "skill_completed"
    ]));
    expect(physicalSignals.find(
      (signal) => signal.kind === "execution_receipt"
    )?.causal_parent_ids).toEqual([reflexReceipt!.signal_id]);
    expect(Object.values(executed.authority_leases).every(
      (lease) => lease.status === "closed"
    )).toBe(true);

    const completionToActionSelection = Object.values(executed.signals).find((signal) => (
      signal.kind === "skill_completed"
        && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager
        && signal.target_node_id === HUMANOID_NEURAL_AGENT_IDS.actionSelection
    ));
    if (!completionToActionSelection) {
      throw new Error("Sensorimotor completion did not reach Action Selection");
    }
    const completeCommitment = actionSelection.tools.find(
      (candidate) => candidate.name === "complete_skill_commitment"
    );
    if (!completeCommitment || completeCommitment.type !== "function") {
      throw new Error("Action Selection has no commitment completion tool");
    }
    const completionOutput = await withAgentInvocation(
      HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      () => completeCommitment.invoke(
        new RunContext(),
        JSON.stringify({
          commitment_id: commitment.commitment_id,
          source_signal_ids: [completionToActionSelection.signal_id],
          reason: "Serial Executor satisfied the committed navigation Skill"
        }),
        {
          toolCall: {
            type: "function_call",
            callId: "action-selection-completes-commitment",
            name: "complete_skill_commitment",
            arguments: "{}",
            status: "completed"
          }
        }
      ),
      false,
      randomUUID()
    );
    expect(JSON.parse(String(completionOutput))).toMatchObject({
      status: "skill_completed",
      source_signal_ids: [completionToActionSelection.signal_id]
    });
    expect(runtime.neuralHierarchyState().active_skill_commitment?.state).toBe(
      "completed"
    );
    expect(runtime.neuralHarnessPhase()).toMatchObject({
      phase: "feedback",
      // The completion mutation cannot clear phase authority before the
      // Action Selection result is routed to Executive. The parent-child
      // wrapper clears this binding after publishing that direct return.
      commitment_id: commitment.commitment_id
    });

    const feedbackInvocationId = randomUUID();
    const feedbackLease = await runtime.issueNeuralAuthorityLease({
      issuingParentNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
      targetChildNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      allowedSignalKinds: ["skill_completed"],
      invocationId: feedbackInvocationId,
      parentInvocationId: null,
      parentEpisodeId: randomUUID(),
      ttlRevisions: 64
    });
    const routedCompletion = await runtime.publishNeuralSignal({
      kind: "skill_completed",
      pathway: "executive_control",
      direction: "descending",
      sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
      targetNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      ttlRevisions: 64,
      priority: 100,
      causalParentIds: [completionToActionSelection.signal_id],
      authorityLeaseId: feedbackLease.lease_id,
      invocationId: feedbackInvocationId,
      parentInvocationId: null,
      parentEpisodeId: feedbackLease.parent_episode_id,
      payload: { status: "skill_completed" }
    });

    const perceptionFork = twoPartyBarrier();
    const feedbackHierarchy = createHumanoidNeuralAgentHierarchy({
      provider,
      runtime,
      createModel: (agentId) => agentId === HUMANOID_NEURAL_AGENT_IDS.actionSelection
        ? postExecutionDelegatingActionSelectionModel(
            calls,
            routedCompletion.signal_id
          )
        : perceptionHierarchyModel(agentId, calls, perceptionFork),
      createSession: (agentId) => new MemorySession({
        sessionId: `feedback:${agentId}`
      }),
      callModelInputFilter: ({ modelData }) => modelData
    });
    const feedbackActionSelection = feedbackHierarchy.agent(
      HUMANOID_NEURAL_AGENT_IDS.actionSelection
    );
    if (!feedbackActionSelection) {
      throw new Error("Action Selection Agent is absent during feedback");
    }
    const feedbackRunner = new Runner({
      tracingDisabled: true,
      modelSettings: { parallelToolCalls: true },
      toolExecution: { maxFunctionToolConcurrency: 2 }
    });
    const actionSelectionCallsBeforeFeedback = calls.filter(
      (agentId) => agentId === HUMANOID_NEURAL_AGENT_IDS.actionSelection
    ).length;
    const feedbackResult = await withAgentInvocation(
      HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      () => feedbackRunner.run(
        feedbackActionSelection,
        "Observe after execution and return the bounded perceptual belief",
        {
          session: feedbackHierarchy.session(
            HUMANOID_NEURAL_AGENT_IDS.actionSelection
          ),
          maxTurns: 8,
          reasoningItemIdPolicy: "omit",
          toolExecution: { maxFunctionToolConcurrency: 2 }
        }
      ),
      false,
      feedbackInvocationId
    );
    expect(calls.filter(
      (agentId) => agentId === HUMANOID_NEURAL_AGENT_IDS.actionSelection
    )).toHaveLength(actionSelectionCallsBeforeFeedback + 1);
    const parsedBelief = feedbackResult.finalOutput as {
      signal_kind: string;
      source_signal_ids: string[];
    };
    expect(parsedBelief).toMatchObject({
      signal_kind: "perceptual_belief",
      source_signal_ids: [expect.any(String)]
    });
    const feedbackState = runtime.neuralHierarchyState();
    const postExecutionBelief = feedbackState.signals[
      parsedBelief.source_signal_ids[0]!
    ];
    expect(postExecutionBelief).toBeDefined();
    expect(signalHasAncestor(
      feedbackState,
      postExecutionBelief!,
      completionToActionSelection.signal_id
    )).toBe(true);
    expect(runtime.neuralHarnessPhase().phase).toBe("cycle_completion");
    const actionSelectionBelief = Object.values(
      runtime.neuralHierarchyState().signals
    ).find((signal) => (
      signal.kind === "perceptual_belief"
        && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.perceptionManager
        && signal.target_node_id === HUMANOID_NEURAL_AGENT_IDS.actionSelection
        && signalHasAncestor(
          runtime.neuralHierarchyState(),
          signal,
          completionToActionSelection.signal_id
        )
    ));
    if (!actionSelectionBelief) {
      throw new Error("Post-execution belief did not reach Action Selection");
    }
    const rootFeedbackHierarchy = createHumanoidNeuralAgentHierarchy({
      provider,
      runtime,
      createModel: (agentId) => agentId === HUMANOID_NEURAL_AGENT_IDS.actionSelection
        ? postExecutionActionSelectionModel(
            calls,
            runtime,
            completionToActionSelection.signal_id
          )
        : perceptionHierarchyModel(agentId, calls, perceptionFork),
      createSession: (agentId) => new MemorySession({
        sessionId: `cycle-completion:${agentId}`
      }),
      callModelInputFilter: ({ modelData }) => modelData
    });
    const feedbackRoot = rootFeedbackHierarchy.root;
    const actionSelectionTool = feedbackRoot.tools.find(
      (candidate) => candidate.name === "delegate_action_selection"
    );
    if (!actionSelectionTool || actionSelectionTool.type !== "function") {
      throw new Error("Executive has no owned Action Selection tool");
    }
    const rootBeliefOutput = await withAgentInvocation(
      HUMANOID_NEURAL_AGENT_IDS.executive,
      () => actionSelectionTool.invoke(
        new RunContext(),
        JSON.stringify({
          signal_kind: "skill_completed",
          source_signal_ids: [actionSelectionBelief.signal_id],
          ttl_revisions: 64,
          priority: 100
        }),
        {
          toolCall: {
            type: "function_call",
            callId: "executive-collects-post-execution-belief",
            name: "delegate_action_selection",
            arguments: "{}",
            status: "completed"
          }
        }
      ),
      false,
      randomUUID()
    );
    const parsedRootBelief = JSON.parse(String(rootBeliefOutput)) as {
      signal_kind: string;
      source_signal_ids: string[];
    };
    expect(parsedRootBelief).toMatchObject({
      signal_kind: "perceptual_belief",
      source_signal_ids: [expect.any(String)]
    });
    const executiveBelief = runtime.neuralHierarchyState().signals[
      parsedRootBelief.source_signal_ids[0]!
    ];
    expect(executiveBelief?.target_node_id).toBe(
      HUMANOID_NEURAL_AGENT_IDS.executive
    );
    expect(signalHasAncestor(
      runtime.neuralHierarchyState(),
      executiveBelief!,
      completionToActionSelection.signal_id
    )).toBe(true);
  });

  it("turns Premotor planning escalation into commitment release and exclusive Recovery replacement", async () => {
    const calls: string[] = [];
    const runtime = inMemoryNeuralRuntime({
      rejectAction: "plan_humanoid_navigation"
    });
    const goalEpochId = "goal-epoch-premotor-recovery-smoke";
    await advanceToPerception(runtime, goalEpochId);
    await runtime.transitionNeuralHarnessPhase({
      phase: "skill_proposal",
      enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      reason: "premotor_recovery_belief_ready",
      goalEpochId,
      commitmentId: null
    });
    const belief = await publishPriorPerceptualBelief(runtime, {
      target_reachable: true,
      obstacle_requires_alternate_route: false
    });
    const actionSelectionInvocationId = randomUUID();
    const proposalInvocationId = randomUUID();
    const proposalLease = await runtime.issueNeuralAuthorityLease({
      issuingParentNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      targetChildNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
      allowedSignalKinds: ["skill_proposal"],
      invocationId: proposalInvocationId,
      parentInvocationId: null,
      parentEpisodeId: actionSelectionInvocationId,
      ttlRevisions: 64
    });
    const proposal = await runtime.publishNeuralSignal({
      kind: "skill_proposal",
      pathway: "sensorimotor_selection",
      direction: "ascending",
      sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
      targetNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      ttlRevisions: 64,
      priority: 90,
      causalParentIds: [belief.signal_id],
      sourceAuthorityLeaseId: proposalLease.lease_id,
      invocationId: proposalInvocationId,
      parentInvocationId: null,
      parentEpisodeId: proposalLease.parent_episode_id,
      payload: boundedNavigateProposal("zone-direct-route")
    });
    await runtime.closeNeuralAuthorityLease({
      leaseId: proposalLease.lease_id,
      closedByNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      reason: "premotor_recovery_proposal_seeded"
    });
    await runtime.transitionNeuralHarnessPhase({
      phase: "commitment_authorization",
      enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      reason: "premotor_recovery_proposal_ready",
      goalEpochId,
      commitmentId: null
    });
    const original = await runtime.establishNeuralSkillCommitment({
      ownerNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      goalEpochId,
      skill: "navigate_to_zone",
      terminationContract: { type: "robot_at", tolerance: 0.35 },
      sourceSignalIds: [proposal.signal_id]
    });
    await runtime.transitionNeuralHarnessPhase({
      phase: "motor_assessment",
      enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      reason: "premotor_recovery_commitment_authorized",
      goalEpochId,
      commitmentId: original.commitment_id
    });
    const hierarchy = createHumanoidNeuralAgentHierarchy({
      provider,
      runtime,
      createModel: (agentId) => premotorRecoveryHierarchyModel(
        agentId,
        calls,
        runtime,
        goalEpochId
      ),
      createSession: (agentId) => new MemorySession({ sessionId: agentId }),
      callModelInputFilter: ({ modelData }) => modelData
    });
    const actionSelection = hierarchy.agent(
      HUMANOID_NEURAL_AGENT_IDS.actionSelection
    );
    if (!actionSelection) throw new Error("Action Selection Agent is absent");
    const runner = new Runner({
      tracingDisabled: true,
      modelSettings: { parallelToolCalls: false },
      toolExecution: { maxFunctionToolConcurrency: 1 }
    });

    const result = await withAgentInvocation(
      HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      () => runner.run(actionSelection, "recover a rejected committed route", {
        session: hierarchy.session(HUMANOID_NEURAL_AGENT_IDS.actionSelection),
        maxTurns: 20,
        reasoningItemIdPolicy: "omit",
        toolExecution: { maxFunctionToolConcurrency: 1 }
      }),
      false,
      actionSelectionInvocationId
    );

    expect(result.finalOutput).toMatchObject({
      signal_kind: "skill_commitment",
      summary: "skill_committed"
    });
    const state = runtime.neuralHierarchyState();
    expect(state.active_skill_commitment).toMatchObject({
      goal_epoch_id: goalEpochId,
      skill: "retreat",
      state: "committed"
    });
    expect(state.active_skill_commitment?.commitment_id).not.toBe(
      original.commitment_id
    );
    expect(calls).toEqual(expect.arrayContaining([
      HUMANOID_NEURAL_AGENT_IDS.premotor,
      HUMANOID_NEURAL_AGENT_IDS.motorIntent,
      HUMANOID_NEURAL_AGENT_IDS.recovery
    ]));
    const edges = Object.values(state.signals).map((signal) => (
      `${signal.source_node_id}->${signal.target_node_id}:${signal.kind}`
    ));
    expect(edges).toEqual(expect.arrayContaining([
      `${HUMANOID_NEURAL_AGENT_IDS.premotor}->${HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager}:escalation`,
      `${HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager}->${HUMANOID_NEURAL_AGENT_IDS.actionSelection}:escalation`,
      `${HUMANOID_NEURAL_AGENT_IDS.actionSelection}->${HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager}:escalation`,
      `${HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager}->${HUMANOID_NEURAL_AGENT_IDS.recovery}:escalation`,
      `${HUMANOID_NEURAL_AGENT_IDS.recovery}->${HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager}:skill_proposal`
    ]));
    });
  });

  it("keeps Manager tools discoverable from the durable authority lease", async () => {
    const runtime = inMemoryNeuralRuntime();
    const goalEpochId = "goal-epoch-durable-tool-discovery";
    await runtime.transitionNeuralHarnessPhase({
      phase: "goal_valuation",
      enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
      reason: "durable_tool_discovery_goal_valuation",
      goalEpochId: null,
      commitmentId: null
    });
    await runtime.transitionNeuralHarnessPhase({
      phase: "perception",
      enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
      reason: "durable_tool_discovery_goal_selected",
      goalEpochId,
      commitmentId: null
    });
    await runtime.transitionNeuralHarnessPhase({
      phase: "skill_proposal",
      enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      reason: "durable_tool_discovery",
      goalEpochId,
      commitmentId: null
    });
    const invocationId = randomUUID();
    const parentEpisodeId = randomUUID();
    const lease = await runtime.issueNeuralAuthorityLease({
      issuingParentNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      targetChildNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
      allowedSignalKinds: ["perceptual_belief"],
      invocationId,
      parentInvocationId: parentEpisodeId,
      parentEpisodeId,
      ttlRevisions: 64
    });
    await runtime.publishNeuralSignal({
      kind: "perceptual_belief",
      pathway: "sensorimotor_selection",
      direction: "descending",
      sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
      targetNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
      ttlRevisions: 64,
      priority: 90,
      authorityLeaseId: lease.lease_id,
      invocationId,
      parentInvocationId: lease.parent_invocation_id,
      payload: { target: "assembly_rod" }
    });
    const hierarchy = createHumanoidNeuralAgentHierarchy({
      provider,
      runtime,
      createModel: (agentId) => scriptedHierarchyModel(agentId, []),
      createSession: (agentId) => new MemorySession({ sessionId: agentId }),
      callModelInputFilter: ({ modelData }) => modelData
    });
    const manager = hierarchy.agent(HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager)!;
    const enabled = await Promise.all(manager.tools
      .filter((entry) => entry.name === "delegate_affordance_assessment"
        || entry.name === "delegate_risk_interoception")
      .map((entry) => entry.type === "function"
        ? entry.isEnabled({} as never, manager)
        : false));

    expect(enabled).toEqual([true, true]);
  });

function scriptedHierarchyModel(agentId: string, calls: string[]): Model {
  return {
    getResponse: async (request) => {
      calls.push(agentId);
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.executive) {
        return hasFunctionResult(request)
          ? textResponse("executive-return", JSON.stringify({
              signal_kind: "goal_context",
              summary: "Goal Valuation returned to Executive",
              payload_json: JSON.stringify({ status: "step_completed" }),
              source_signal_ids: [],
              confidence: 1
            }))
          : functionCallResponse(
              "executive-delegates-goal",
              "delegate_goal_valuation",
              {
                signal_kind: "goal_context",
                source_signal_ids: [],
                ttl_revisions: 64,
                priority: 80
              }
            );
      }
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.goalManager) {
        return functionCallResponse(
          "goal-continues-epoch",
          "continue_goal_epoch",
          {
            reason: "The current Goal remains physically relevant"
          }
        );
      }
      throw new Error(`Unexpected model activation outside the owned path: ${agentId}`);
    },
    getStreamedResponse: () => {
      throw new Error("Streaming is outside this hierarchy smoke test");
    }
  } as Model;
}

function malformedDelegationModel(agentId: string, calls: string[]): Model {
  let executiveTurns = 0;
  return {
    getResponse: async (request) => {
      calls.push(agentId);
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.executive) {
        executiveTurns += 1;
        if (executiveTurns >= 4) {
          return textResponse("executive-return-after-correction", JSON.stringify({
            signal_kind: "goal_context",
            summary: "corrected child delegation completed",
            payload_json: JSON.stringify({ status: "step_completed" }),
            source_signal_ids: [],
            confidence: 1
          }));
        }
        if (executiveTurns === 2) {
          return functionCallResponse(
            "executive-invents-source-signal",
            "delegate_goal_valuation",
            {
              signal_kind: "goal_context",
              source_signal_ids: ["00000000-0000-0000-0000-000000000000"],
              ttl_revisions: 64,
              priority: 80
            }
          );
        }
        if (executiveTurns === 3) {
          return functionCallResponse(
            "executive-corrects-delegation",
            "delegate_goal_valuation",
            {
              signal_kind: "goal_context",
              source_signal_ids: [],
              ttl_revisions: 64,
              priority: 80
            }
          );
        }
        return modelResponse("executive-malformed-delegation", [{
          type: "function_call",
          callId: "malformed-delegation-call",
          name: "delegate_goal_valuation",
          status: "completed",
          arguments: "{"
        }]);
      }
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.goalManager) {
        return functionCallResponse(
          "goal-continues-after-correction",
          "continue_goal_epoch",
          {
            reason: "The current Goal remains physically relevant"
          }
        );
      }
      throw new Error(`Unexpected model activation outside corrected path: ${agentId}`);
    },
    getStreamedResponse: () => {
      throw new Error("Streaming is outside this hierarchy smoke test");
    }
  } as Model;
}

function perceptionHierarchyModel(
  agentId: string,
  calls: string[],
  fork: ReturnType<typeof twoPartyBarrier>
): Model {
  return {
    getResponse: async (request) => {
      calls.push(agentId);
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.perceptionManager) {
        const results = functionResultRecords(request);
        if (!results.some((record) => "receipt" in record)) {
          return functionCallResponse("capture-current-sensation", "capture_sensor_fusion", {});
        }
        const signalIds = results.flatMap((record) => (
          stringSignalIds(record.source_signal_ids)
        ));
        if (signalIds.length === 1) {
          return modelResponse("perception-read-only-fork", [
            functionCall("scene-fork", "delegate_scene_interpretation", {
              signal_kind: "sensory_evidence",
              source_signal_ids: [signalIds[0]!],
              ttl_revisions: 64,
              priority: 70
            }),
            functionCall("memory-fork", "delegate_relevant_memory", {
              signal_kind: "sensory_evidence",
              source_signal_ids: [signalIds[0]!],
              ttl_revisions: 64,
              priority: 70
            })
          ]);
        }
        if (signalIds.length !== 3) {
          throw new Error(`Perception Manager expected three joined signals, got ${signalIds.length}`);
        }
        return textResponse("perception-joined", JSON.stringify({
          signal_kind: "perceptual_belief",
          summary: "Perception Manager joined current scene and relevant memory",
          payload_json: JSON.stringify({
            world_revision: 0,
            belief: "current authoritative observation plus bounded history"
          }),
          source_signal_ids: signalIds,
          confidence: 0.95
        }));
      }
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.sceneInterpreter) {
        await fork.arrive(agentId);
        return textResponse("scene-interpreted", JSON.stringify({
          signal_kind: "scene_interpretation",
          summary: "scene branch current evidence",
          payload_json: JSON.stringify({ clear_path: true }),
          source_signal_ids: [],
          confidence: 0.96
        }));
      }
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.memoryRetriever) {
        await fork.arrive(agentId);
        return textResponse("memory-retrieved", JSON.stringify({
          signal_kind: "memory_retrieval",
          summary: "memory branch historical evidence",
          payload_json: JSON.stringify({ relevant_prior_count: 0 }),
          source_signal_ids: [],
          confidence: 0.9
        }));
      }
      throw new Error(`Unexpected model activation outside perception subtree: ${agentId}`);
    },
    getStreamedResponse: () => {
      throw new Error("Streaming is outside this perception fork/join smoke test");
    }
  } as Model;
}

function postExecutionActionSelectionModel(
  calls: string[],
  runtime: HumanoidNeuralAgentRuntime,
  completionSignalId: string
): Model {
  return {
    getResponse: async () => {
      calls.push(HUMANOID_NEURAL_AGENT_IDS.actionSelection);
      const state = runtime.neuralHierarchyState();
      const belief = runtime.pendingNeuralSignals({
        targetNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
        kinds: ["perceptual_belief"]
      }).find((signal) => signalHasAncestor(
        state,
        signal,
        completionSignalId
      ));
      if (!belief) throw new Error("Action Selection has no post-execution belief");
      return textResponse("action-selection-forwards-post-execution-belief", JSON.stringify({
        signal_kind: "perceptual_belief",
        summary: "Action Selection forwards the bounded post-execution belief",
        payload_json: JSON.stringify(belief.payload),
        source_signal_ids: [belief.signal_id],
        confidence: 1
      }));
    },
    getStreamedResponse: () => {
      throw new Error("Streaming is outside this post-execution smoke test");
    }
  } as Model;
}

function postExecutionDelegatingActionSelectionModel(
  calls: string[],
  routedCompletionSignalId: string
): Model {
  return {
    getResponse: async (request) => {
      calls.push(HUMANOID_NEURAL_AGENT_IDS.actionSelection);
      if (hasFunctionResult(request)) {
        throw new Error(
          "Action Selection must terminate on the typed post-execution Perception return"
        );
      }
      return functionCallResponse(
        "action-selection-delegates-post-execution-perception",
        "delegate_perception_manager",
        {
          signal_kind: "skill_completed",
          source_signal_ids: [routedCompletionSignalId],
          ttl_revisions: 64,
          priority: 100
        }
      );
    },
    getStreamedResponse: () => {
      throw new Error("Streaming is outside this post-execution smoke test");
    }
  } as Model;
}

function sensorimotorHierarchyModel(
  agentId: string,
  calls: string[],
  fork: ReturnType<typeof twoPartyBarrier>,
  runtime: HumanoidNeuralAgentRuntime,
  beliefSignalId: string,
  goalEpochId: string
): Model {
  return {
    getResponse: async (request) => {
      calls.push(agentId);
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.actionSelection) {
        const results = functionResultRecords(request);
        const proposal = results.find((record) => (
          record.signal_kind === "skill_proposal"
            && stringSignalIds(record.source_signal_ids).length === 1
        ));
        return proposal
          ? functionCallResponse(
              "bind-sensorimotor-proposal",
              "establish_skill_commitment",
              {
                goal_epoch_id: goalEpochId,
                skill: "navigate_to_zone",
                termination_contract_json: JSON.stringify({
                  type: "robot_at",
                  tolerance: 0.35
                }),
                source_signal_ids: stringSignalIds(proposal.source_signal_ids)
              }
            )
          : functionCallResponse(
              "action-selection-delegates-sensorimotor",
              "delegate_sensorimotor_manager",
              {
                signal_kind: "perceptual_belief",
                source_signal_ids: [beliefSignalId],
                ttl_revisions: 64,
                priority: 80
              }
            );
      }
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager) {
        const results = functionResultRecords(request);
        const assessmentIds = results.flatMap((record) => (
          stringSignalIds(record.source_signal_ids)
        ));
        if (assessmentIds.length === 0) {
          return modelResponse("sensorimotor-read-only-fork", [
            functionCall("affordance-fork", "delegate_affordance_assessment", {
              signal_kind: "perceptual_belief",
              source_signal_ids: [],
              ttl_revisions: 64,
              priority: 70
            }),
            functionCall("risk-fork", "delegate_risk_interoception", {
              signal_kind: "perceptual_belief",
              source_signal_ids: [],
              ttl_revisions: 64,
              priority: 70
            })
          ]);
        }
        if (assessmentIds.length !== 2) {
          throw new Error(
            `Sensorimotor Manager expected two assessments, got ${assessmentIds.length}`
          );
        }
        const managerBelief = runtime.pendingNeuralSignals({
          targetNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
          kinds: ["perceptual_belief"]
        })[0];
        if (!managerBelief) throw new Error("Sensorimotor Manager has no current belief");
        return textResponse("sensorimotor-joined", JSON.stringify({
          signal_kind: "skill_proposal",
          summary: "Sensorimotor Manager joined affordance and risk into one bounded Skill",
          payload_json: JSON.stringify(
            boundedNavigateProposal("zone-sensorimotor-smoke")
          ),
          source_signal_ids: [managerBelief.signal_id, ...assessmentIds],
          confidence: 0.93
        }));
      }
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.affordance) {
        await fork.arrive(agentId);
        return textResponse("affordance-assessed", JSON.stringify({
          signal_kind: "affordance_hypothesis",
          summary: "target affords bounded walking",
          payload_json: JSON.stringify({ reachable: true, skill: "walk" }),
          source_signal_ids: [],
          confidence: 0.95
        }));
      }
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.risk) {
        await fork.arrive(agentId);
        return textResponse("risk-assessed", JSON.stringify({
          signal_kind: "risk_assessment",
          summary: "balance risk remains bounded",
          payload_json: JSON.stringify({ inhibited: false, maximum_risk: 0.18 }),
          source_signal_ids: [],
          confidence: 0.94
        }));
      }
      throw new Error(`Unexpected model activation outside sensorimotor subtree: ${agentId}`);
    },
    getStreamedResponse: () => {
      throw new Error("Streaming is outside this sensorimotor fork/join smoke test");
    }
  } as Model;
}

function certifiedMotionHierarchyModel(
  agentId: string,
  calls: string[],
  runtime: HumanoidNeuralAgentRuntime
): Model {
  return {
    getResponse: async (request) => {
      calls.push(agentId);
      const results = functionResultRecords(request);
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.actionSelection) {
        const rollout = results.find((record) => (
          record.signal_kind === "rollout_result"
            && stringSignalIds(record.source_signal_ids).length === 1
        ));
        if (rollout) {
          const commitment = runtime.neuralHierarchyState().active_skill_commitment;
          if (!commitment) throw new Error("Action Selection has no active commitment");
          return functionCallResponse(
            "authorize-certified-skill",
            "authorize_skill_execution",
            {
              commitment_id: commitment.commitment_id,
              reason: "Predictive accepted the exact MuJoCo rollout"
            }
          );
        }
        const commitment = runtime.neuralHierarchyState().active_skill_commitment;
        if (!commitment) throw new Error("Action Selection has no active commitment");
        return functionCallResponse(
          "action-selection-runs-committed-sensorimotor",
          "delegate_sensorimotor_manager",
          {
            signal_kind: "skill_commitment",
            source_signal_ids: commitment.source_signal_ids,
            ttl_revisions: 64,
            priority: 95
          }
        );
      }
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager) {
        const execution = results.find((record) => (
          record.action === "execute_humanoid_navigation"
            && Array.isArray(record.source_signal_ids)
        ));
        if (execution) {
          return textResponse("sensorimotor-execution-completed", JSON.stringify({
            signal_kind: "skill_completed",
            summary: "Serial Executor completed the certified navigation Skill",
            payload_json: JSON.stringify(withoutSignalIds(execution)),
            source_signal_ids: execution.source_signal_ids,
            confidence: 1
          }));
        }
        if (runtime.neuralHarnessPhase().phase === "execution") {
          const directCommitment = runtime.pendingNeuralSignals({
            targetNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
            kinds: ["skill_commitment"]
          }).find((signal) => signal.direction === "descending"
            && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.actionSelection
            && (signal.payload as { state?: string }).state === "executing");
          if (!directCommitment) {
            throw new Error("Sensorimotor has no direct executing commitment signal");
          }
          return functionCallResponse(
            "serial-executor-certified-navigation",
            "execute_certified_motor_intent",
            {
              objective: "Execute the one Predictive-certified navigation rollout",
              source_signal_ids: [directCommitment.signal_id]
            }
          );
        }
        const premotor = results.find((record) => (
          record.signal_kind === "rollout_result"
            && stringSignalIds(record.source_signal_ids).length === 1
        ));
        const predictive = results.find((record) => (
          record.signal_kind === "forward_prediction"
            && stringSignalIds(record.source_signal_ids).length === 1
        ));
        if (predictive && premotor) {
          return textResponse("sensorimotor-certified-rollout", JSON.stringify({
            signal_kind: "rollout_result",
            summary: "Sensorimotor admitted one Predictive-certified rollout",
            payload_json: JSON.stringify(premotor.payload),
            source_signal_ids: [
              ...stringSignalIds(premotor.source_signal_ids),
              ...stringSignalIds(predictive.source_signal_ids)
            ],
            confidence: 1
          }));
        }
        if (premotor) {
          const reentrant = runtime.pendingNeuralSignals({
            targetNodeId: HUMANOID_NEURAL_AGENT_IDS.predictive,
            kinds: ["rollout_result"]
          }).find((signal) => signal.direction === "reentrant");
          if (!reentrant) throw new Error("Harness has no reentrant rollout");
          return functionCallResponse(
            "sensorimotor-delegates-predictive",
            "delegate_predictive_critic",
            {
              signal_kind: "rollout_result",
              // Compatible models can select a known nested rollout id instead
              // of the direct Premotor edge. The Harness must canonicalize this
              // delegation to its unique current direct child result.
              source_signal_ids: [reentrant.signal_id],
              ttl_revisions: 64,
              priority: 95
            }
          );
        }
        const commitmentSignal = runtime.pendingNeuralSignals({
          targetNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
          kinds: ["skill_commitment"]
        }).find((signal) => signal.direction === "descending");
        if (!commitmentSignal) {
          throw new Error("Sensorimotor has no accepted commitment signal");
        }
        return functionCallResponse(
          "sensorimotor-delegates-premotor",
          "delegate_premotor_composition",
          {
            signal_kind: "skill_commitment",
            source_signal_ids: [commitmentSignal.signal_id],
            ttl_revisions: 64,
            priority: 90
          }
        );
      }
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.premotor) {
        return functionCallResponse(
          "premotor-delegates-motor-intent",
          "delegate_motor_intent",
          {
            signal_kind: "skill_proposal",
            source_signal_ids: [],
            ttl_revisions: 64,
            priority: 90
          }
        );
      }
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.motorIntent) {
        return functionCallResponse(
          "motor-intent-navigation-rollout",
          "plan_humanoid_navigation",
          {
            skill_transaction_id: null,
            target: { x: 2, y: 0, z: 3 },
            arrival_heading: null
          }
        );
      }
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.predictive) {
        const rollout = runtime.pendingNeuralSignals({
          targetNodeId: HUMANOID_NEURAL_AGENT_IDS.predictive,
          kinds: ["rollout_result"]
        })[0];
        if (!rollout) throw new Error("Predictive has no current rollout");
        return textResponse("predictive-accepts-rollout", JSON.stringify({
          signal_kind: "forward_prediction",
          summary: "MuJoCo rollout satisfies the committed Skill and risk bounds",
          payload_json: JSON.stringify({ accepted: true, maximum_risk: 0.12 }),
          source_signal_ids: [rollout.signal_id],
          confidence: 0.98,
          accepted: true
        }));
      }
      throw new Error(`Unexpected model activation in certified motion path: ${agentId}`);
    },
    getStreamedResponse: () => {
      throw new Error("Streaming is outside this certified motion smoke test");
    }
  } as Model;
}

function premotorRecoveryHierarchyModel(
  agentId: string,
  calls: string[],
  runtime: HumanoidNeuralAgentRuntime,
  goalEpochId: string
): Model {
  return {
    getResponse: async (request) => {
      calls.push(agentId);
      const results = functionResultRecords(request);
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.actionSelection) {
        const replacement = results.find((record) => (
          record.signal_kind === "skill_proposal"
            && stringSignalIds(record.source_signal_ids).length === 1
        ));
        if (replacement) {
          return functionCallResponse(
            "action-selection-binds-premotor-recovery",
            "establish_skill_commitment",
            {
              goal_epoch_id: goalEpochId,
              skill: "retreat",
              termination_contract_json: JSON.stringify({
                type: "robot_at",
                tolerance: 0.35,
                route: "alternate"
              }),
              source_signal_ids: stringSignalIds(replacement.source_signal_ids)
            }
          );
        }
        const escalation = results.find((record) => (
          record.signal_kind === "escalation"
            && stringSignalIds(record.source_signal_ids).length === 1
        ));
        const released = results.some((record) => record.status === "skill_released");
        const commitment = runtime.neuralHierarchyState().active_skill_commitment;
        if (!commitment) throw new Error("Action Selection has no active commitment");
        if (released) {
          if (!escalation || commitment.state !== "released") {
            throw new Error("Action Selection release did not preserve its recovery demand");
          }
          return functionCallResponse(
            "action-selection-routes-premotor-recovery",
            "delegate_sensorimotor_manager",
            {
              signal_kind: "escalation",
              source_signal_ids: stringSignalIds(escalation.source_signal_ids),
              ttl_revisions: 64,
              priority: 100
            }
          );
        }
        if (escalation) {
          return functionCallResponse(
            "action-selection-releases-rejected-route",
            "release_skill_commitment",
            {
              commitment_id: commitment.commitment_id,
              reason: "Premotor could not compile the committed route safely",
              source_signal_ids: stringSignalIds(escalation.source_signal_ids)
            }
          );
        }
        return functionCallResponse(
          "action-selection-delegates-rejected-route",
          "delegate_sensorimotor_manager",
          {
            signal_kind: "skill_commitment",
            source_signal_ids: commitment.source_signal_ids,
            ttl_revisions: 64,
            priority: 95
          }
        );
      }
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager) {
        if (runtime.neuralHarnessPhase().phase === "recovery") {
          return functionCallResponse(
            "sensorimotor-opens-premotor-recovery",
            "run_recovery_lease_episode",
            {}
          );
        }
        const commitmentSignal = runtime.pendingNeuralSignals({
          targetNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
          kinds: ["skill_commitment"]
        }).find((signal) => signal.direction === "descending");
        if (!commitmentSignal) {
          throw new Error("Sensorimotor has no direct committed route");
        }
        return functionCallResponse(
          "sensorimotor-delegates-rejected-premotor",
          "delegate_premotor_composition",
          {
            signal_kind: "skill_commitment",
            source_signal_ids: [commitmentSignal.signal_id],
            ttl_revisions: 64,
            priority: 95
          }
        );
      }
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.premotor) {
        return functionCallResponse(
          "premotor-delegates-rejected-motor-intent",
          "delegate_motor_intent",
          {
            signal_kind: "skill_proposal",
            source_signal_ids: [],
            ttl_revisions: 64,
            priority: 95
          }
        );
      }
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.motorIntent) {
        return functionCallResponse(
          "motor-intent-rejected-navigation",
          "plan_humanoid_navigation",
          {
            skill_transaction_id: null,
            target: { x: 2, y: 0, z: 3 },
            arrival_heading: null
          }
        );
      }
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.recovery) {
        const escalation = runtime.pendingNeuralSignals({
          targetNodeId: HUMANOID_NEURAL_AGENT_IDS.recovery,
          kinds: ["escalation"]
        }).find((signal) => signal.direction === "descending");
        if (!escalation) throw new Error("Recovery received no Premotor escalation");
        return functionCallResponse(
          "recovery-replaces-rejected-route",
          "submit_neural_output",
          {
            signal_kind: "skill_proposal",
            summary: "Recovery selected an alternate route around the obstacle",
            payload: boundedRetreatProposal(),
            source_signal_ids: [escalation.signal_id],
            confidence: 0.96
          }
        );
      }
      throw new Error(`Unexpected model activation in Premotor recovery path: ${agentId}`);
    },
    getStreamedResponse: () => {
      throw new Error("Streaming is outside this Premotor recovery smoke test");
    }
  } as Model;
}

function recoveryHierarchyModel(
  agentId: string,
  calls: string[],
  runtime: HumanoidNeuralAgentRuntime,
  goalEpochId: string,
  failureSignalId: string
): Model {
  return {
    getResponse: async (request) => {
      calls.push(agentId);
      const results = functionResultRecords(request);
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.executive) {
        return functionCallResponse(
          "executive-delegates-failure-recovery",
          "delegate_action_selection",
          {
            signal_kind: "skill_failed",
            source_signal_ids: [failureSignalId],
            ttl_revisions: 64,
            priority: 100
          }
        );
      }
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.actionSelection) {
        const proposal = results.find((record) => record.signal_kind === "skill_proposal");
        if (proposal) {
          return functionCallResponse(
            "action-selection-binds-recovery-proposal",
            "establish_skill_commitment",
            {
              goal_epoch_id: goalEpochId,
              skill: "retreat",
              termination_contract_json: JSON.stringify({
                type: "rejoin_original_goal_path",
                preserve_goal_epoch: true
              }),
              source_signal_ids: stringSignalIds(proposal.source_signal_ids)
            }
          );
        }
        const failure = runtime.pendingNeuralSignals({
          targetNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
          kinds: ["skill_failed"]
        }).find((signal) => signal.direction === "descending"
          && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.executive);
        if (!failure) {
          throw new Error("Action Selection received no direct failure signal");
        }
        return functionCallResponse(
          "action-selection-delegates-recovery",
          "delegate_sensorimotor_manager",
          {
            signal_kind: "skill_failed",
            source_signal_ids: [failure.signal_id],
            ttl_revisions: 64,
            priority: 100
          }
        );
      }
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager) {
        return functionCallResponse(
          "sensorimotor-opens-exclusive-recovery",
          "run_recovery_lease_episode",
          {}
        );
      }
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.recovery) {
        const failure = runtime.pendingNeuralSignals({
          targetNodeId: HUMANOID_NEURAL_AGENT_IDS.recovery,
          kinds: ["skill_failed"]
        }).find((signal) => signal.direction === "descending"
          && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager);
        if (!failure) {
          throw new Error("Recovery received no direct failure signal");
        }
        return functionCallResponse(
          "recovery-submits-replacement-skill",
          "submit_neural_output",
          {
            signal_kind: "skill_proposal",
            summary: "Exclusive Recovery proposed a bounded sidestep",
            payload: boundedRetreatProposal(),
            source_signal_ids: [failure.signal_id],
            confidence: 0.93
          }
        );
      }
      throw new Error(`Unexpected model activation in recovery path: ${agentId}`);
    },
    getStreamedResponse: () => {
      throw new Error("Streaming is outside this recovery hierarchy smoke test");
    }
  } as Model;
}

function recoveryEscalationHierarchyModel(
  agentId: string,
  calls: string[],
  runtime: HumanoidNeuralAgentRuntime,
  failureSignalId: string
): Model {
  return {
    getResponse: async () => {
      calls.push(agentId);
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.executive) {
        return functionCallResponse(
          "executive-delegates-recovery-escalation",
          "delegate_action_selection",
          {
            signal_kind: "skill_failed",
            source_signal_ids: [failureSignalId],
            ttl_revisions: 64,
            priority: 100
          }
        );
      }
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.actionSelection) {
        const failure = runtime.pendingNeuralSignals({
          targetNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
          kinds: ["skill_failed"]
        }).find((signal) => signal.direction === "descending"
          && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.executive);
        if (!failure) throw new Error("Action Selection received no escalation failure");
        return functionCallResponse(
          "action-selection-delegates-recovery-escalation",
          "delegate_sensorimotor_manager",
          {
            signal_kind: "skill_failed",
            source_signal_ids: [failure.signal_id],
            ttl_revisions: 64,
            priority: 100
          }
        );
      }
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager) {
        return functionCallResponse(
          "sensorimotor-opens-recovery-escalation",
          "run_recovery_lease_episode",
          {}
        );
      }
      if (agentId === HUMANOID_NEURAL_AGENT_IDS.recovery) {
        const failure = runtime.pendingNeuralSignals({
          targetNodeId: HUMANOID_NEURAL_AGENT_IDS.recovery,
          kinds: ["skill_failed"]
        }).find((signal) => signal.direction === "descending"
          && signal.source_node_id === HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager);
        if (!failure) throw new Error("Recovery received no escalation failure");
        return functionCallResponse(
          "recovery-escalates-outside-pathway",
          "submit_neural_output",
          {
            signal_kind: "escalation",
            summary: "No pathway-local replacement preserves the active Goal safely",
            payload: {
              requested_scope: "supervisory",
              physical_write_authority: false
            },
            source_signal_ids: [failure.signal_id],
            confidence: 0.99
          }
        );
      }
      throw new Error(`Unexpected model activation in recovery escalation path: ${agentId}`);
    },
    getStreamedResponse: () => {
      throw new Error("Streaming is outside this Recovery escalation smoke test");
    }
  } as Model;
}

function boundedNavigateProposal(zoneId: string): JsonValue {
  return {
    proposed_skill: {
      skill: "navigate_to_zone",
      phase: "enter_zone",
      params: { zone_id: zoneId },
      rationale: "The current reachable route advances the active Goal"
    }
  };
}

function boundedRetreatProposal(): JsonValue {
  return {
    proposed_skill: {
      skill: "retreat",
      phase: "route",
      params: {
        target: { x: 1, y: 0, z: 1 },
        minimum_obstacle_clearance_m: 0.5
      },
      rationale: "A bounded retreat restores a collision-free recovery stance"
    }
  };
}

function withoutSignalIds(record: Record<string, unknown>): Record<string, unknown> {
  const { source_signal_ids: _sourceSignalIds, ...payload } = record;
  return payload;
}

function stringSignalIds(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function signalHasAncestor(
  state: NeuralHierarchyState,
  signal: NeuralHierarchyState["signals"][string],
  ancestorSignalId: string
): boolean {
  const visited = new Set<string>();
  const pending = [...signal.causal_parent_ids];
  while (pending.length > 0) {
    const signalId = pending.pop()!;
    if (signalId === ancestorSignalId) return true;
    if (visited.has(signalId)) continue;
    visited.add(signalId);
    const parent = state.signals[signalId];
    if (parent) pending.push(...parent.causal_parent_ids);
  }
  return false;
}

function hasFunctionResult(request: ModelRequest): boolean {
  return request.input.some((item) => (
    typeof item === "object"
      && item !== null
      && "type" in item
      && item.type === "function_call_result"
  ));
}

function functionCallResponse(
  responseId: string,
  name: string,
  args: Record<string, JsonValue>
): ModelResponse {
  return modelResponse(responseId, [functionCall(
    responseId,
    name,
    args
  )]);
}

function functionCall(
  callId: string,
  name: string,
  args: Record<string, JsonValue>
) {
  return {
    type: "function_call",
    callId: `call-${callId}`,
    name,
    arguments: JSON.stringify(args)
  } as const;
}

function textResponse(responseId: string, text: string): ModelResponse {
  return modelResponse(responseId, [{
    type: "message",
    role: "assistant",
    status: "completed",
    content: [{ type: "output_text", text }]
  }]);
}

function modelResponse(responseId: string, output: unknown[]): ModelResponse {
  return {
    responseId,
    output,
    usage: new Usage({
      requests: 1,
      inputTokens: 1,
      outputTokens: 1,
      totalTokens: 2
    })
  } as ModelResponse;
}

function functionResultRecords(request: ModelRequest): Array<Record<string, unknown>> {
  const records: Array<Record<string, unknown>> = [];
  for (const item of request.input) {
    if (typeof item !== "object" || item === null
      || !("type" in item) || item.type !== "function_call_result"
      || !("output" in item)) continue;
    const output = item.output;
    const text = typeof output === "string"
      ? output
      : output !== null && typeof output === "object"
        && "type" in output && output.type === "text"
        && "text" in output && typeof output.text === "string"
        ? output.text
        : undefined;
    if (text !== undefined) {
      try {
        const parsed = JSON.parse(text) as unknown;
        if (parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)) {
          records.push(parsed as Record<string, unknown>);
        }
      } catch {
        // A non-JSON tool output is not a neural result and is ignored here.
      }
    }
  }
  return records;
}

function functionResultTexts(request: ModelRequest): string[] {
  return request.input.flatMap((item) => {
    if (typeof item !== "object" || item === null
      || !("type" in item) || item.type !== "function_call_result"
      || !("output" in item)) return [];
    const output = item.output;
    if (typeof output === "string") return [output];
    return output !== null && typeof output === "object"
      && "type" in output && output.type === "text"
      && "text" in output && typeof output.text === "string"
      ? [output.text]
      : [];
  });
}

function invocationPayload(items: readonly unknown[]): {
  anchor: { neural_hierarchy: Record<string, unknown> };
  directed_signals: unknown[];
} {
  for (const item of items) {
    if (typeof item !== "object" || item === null
      || !("type" in item) || item.type !== "message"
      || !("role" in item) || item.role !== "user"
      || !("content" in item) || typeof item.content !== "string") continue;
    const jsonStart = item.content.indexOf("{");
    if (jsonStart < 0) continue;
    const parsed = JSON.parse(item.content.slice(jsonStart)) as {
      anchor?: { neural_hierarchy?: Record<string, unknown> };
      directed_signals?: unknown[];
    };
    if (parsed.anchor?.neural_hierarchy
      && Array.isArray(parsed.directed_signals)) {
      return {
        anchor: { neural_hierarchy: parsed.anchor.neural_hierarchy },
        directed_signals: parsed.directed_signals
      };
    }
  }
  throw new Error("Child Session has no bounded invocation payload");
}

function twoPartyBarrier() {
  const entered = new Set<string>();
  let release!: () => void;
  let reject!: (error: Error) => void;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const ready = new Promise<void>((resolve, fail) => {
    release = resolve;
    reject = fail;
  });
  return {
    arrivals: () => new Set(entered),
    arrive: async (agentId: string) => {
      entered.add(agentId);
      if (entered.size === 2) {
        if (timer) clearTimeout(timer);
        release();
      } else if (!timer) {
        timer = setTimeout(() => reject(new Error(
          "Perception siblings did not enter the parent-owned parallel fork"
        )), 2_000);
      }
      await ready;
    }
  };
}

function inMemoryNeuralRuntime(options: {
  neuralExecutionAvailable?: boolean;
  autonomyReadiness?: string;
  onAction?: (action: string) => void;
  rejectAction?: string;
  contextAnchor?: JsonValue;
  onRecall?: (request: Record<string, unknown>) => void;
  recallResult?: JsonValue;
} = {}): HumanoidNeuralAgentRuntime {
  let state = createNeuralHierarchyState();
  const runtime = {
    rootAgentId: HUMANOID_NEURAL_AGENT_IDS.executive,
    currentWorldRevision: () => 0,
    neuralExecutionAvailable: () => options.neuralExecutionAvailable ?? false,
    cycleCompletionReadiness: () => ({
      status: "not_ready" as const,
      evidence_transaction_ids: [],
      observed_after_execution: false
    }),
    autonomyReadiness: () => options.autonomyReadiness ?? "observe_or_plan",
    validateCycleEvidence: () => { throw new Error("No cycle execution in Goal smoke"); },
    validateSatisfiedGoal: () => { throw new Error("No satisfied Goal in Goal smoke"); },
    neuralHierarchyState: () => structuredClone(state),
    neuralHarnessPhase: () => structuredClone(state.harness_phase),
    neuralSkillCommitmentOutcome: () => ({
      status: "completed" as const,
      detail: { authority: "test" }
    }),
    neuralNodeEnabled: ({ phases }: { phases: readonly string[] }) => (
      phases.includes(state.harness_phase.phase)
    ),
    contextAnchor: (agentId: string) => options.contextAnchor ?? ({
      agent_id: agentId,
      hierarchy_epoch_id: state.epoch_id,
      world_revision: 0
    }),
    pendingNeuralSignals: (input: {
      targetNodeId?: string;
      kinds?: Parameters<typeof pendingNeuralSignals>[0]["kinds"];
      invocationId?: string;
    } = {}) => pendingNeuralSignals({
      state,
      worldRevision: 0,
      ...(input.targetNodeId ? { targetNodeId: input.targetNodeId } : {}),
      ...(input.kinds ? { kinds: input.kinds } : {})
    }).filter((signal) => input.invocationId === undefined
      || signal.invocation_id === input.invocationId),
    publishNeuralSignal: async (input: Parameters<
      HumanoidNeuralAgentRuntime["publishNeuralSignal"]
    >[0]) => {
      const route = assertHumanoidNeuralSignalRoute({
        sourceNodeId: input.sourceNodeId,
        targetNodeId: input.targetNodeId,
        direction: input.direction,
        kind: input.kind
      });
      const leaseId = input.direction === "descending"
        ? input.authorityLeaseId
        : input.sourceAuthorityLeaseId;
      const lease = leaseId ? state.authority_leases[leaseId] : undefined;
      if (!lease) throw new Error("Test signal has no authority lease");
      const published = publishNeuralSignal(state, {
        ...input,
        sourceLayer: route.source.layer,
        targetLayer: route.target.layer,
        worldFrame: 0,
        worldRevision: 0,
        invocationId: input.invocationId ?? lease.invocation_id,
        parentInvocationId: input.parentInvocationId ?? lease.parent_invocation_id,
        parentEpisodeId: input.parentEpisodeId ?? lease.parent_episode_id
      });
      state = published.state;
      return published.signal;
    },
    consumeNeuralSignals: async (_consumerNodeId: string, signalIds: readonly string[]) => {
      state = consumeNeuralSignals(state, signalIds);
    },
    transitionNeuralHarnessPhase: async (input: Parameters<
      HumanoidNeuralAgentRuntime["transitionNeuralHarnessPhase"]
    >[0]) => {
      state = transitionNeuralHarnessPhase(state, {
        ...input,
        goalEpochId: input.goalEpochId === undefined
          ? state.harness_phase.goal_epoch_id
          : input.goalEpochId,
        commitmentId: input.commitmentId === undefined
          ? state.active_skill_commitment
            && !["completed", "failed", "released"].includes(
              state.active_skill_commitment.state
            )
              ? state.active_skill_commitment.commitment_id
              : null
          : input.commitmentId,
        worldRevision: 0
      });
      return structuredClone(state.harness_phase);
    },
    issueNeuralAuthorityLease: async (input: Parameters<
      HumanoidNeuralAgentRuntime["issueNeuralAuthorityLease"]
    >[0]) => {
      const issued = issueNeuralAuthorityLease(state, {
        ...input,
        goalEpochId: state.harness_phase.goal_epoch_id,
        commitmentId: state.harness_phase.commitment_id,
        worldRevision: 0,
        expiresWorldRevision: input.ttlRevisions ?? 64,
        expiresAt: new Date(Date.now() + (input.ttlMs ?? 120_000)).toISOString()
      });
      state = issued.state;
      return issued.lease;
    },
    closeNeuralAuthorityLease: async (input: Parameters<
      HumanoidNeuralAgentRuntime["closeNeuralAuthorityLease"]
    >[0]) => {
      state = closeNeuralAuthorityLease(state, { ...input, worldRevision: 0 });
    },
    establishNeuralSkillCommitment: async (input: Parameters<
      HumanoidNeuralAgentRuntime["establishNeuralSkillCommitment"]
    >[0]) => {
      const established = establishNeuralSkillCommitment(state, {
        ...input,
        worldRevision: 0
      });
      state = established.state;
      return established.commitment;
    },
    transitionNeuralSkillCommitment: async (input: Parameters<
      HumanoidNeuralAgentRuntime["transitionNeuralSkillCommitment"]
    >[0]) => {
      const transitioned = transitionNeuralSkillCommitment(state, {
        ...input,
        worldRevision: 0
      });
      state = transitioned.state;
      return transitioned.commitment;
    },
    recordNeuralPredictionError: async (input: Parameters<
      HumanoidNeuralAgentRuntime["recordNeuralPredictionError"]
    >[0]) => {
      const recorded = appendNeuralPredictionError(state, {
        ...input,
        worldRevision: 0
      });
      state = recorded.state;
      return recorded.error;
    },
    issueNeuralRolloutCertificate: async (input: Parameters<
      HumanoidNeuralAgentRuntime["issueNeuralRolloutCertificate"]
    >[0]) => {
      const issued = issueNeuralRolloutCertificate(state, {
        ...input,
        worldRevision: 0,
        expiresWorldRevision: input.ttlRevisions ?? 64
      });
      state = issued.state;
      return issued.certificate;
    },
    recallGoalHistory: async () => [],
    submitGoalCandidates: async () => ({ status: "goal_candidates_submitted" }),
    selectGoalCandidate: async () => ({ status: "goal_candidate_selected" }),
    retireGoalEpoch: async () => ({ status: "goal_epoch_retired" }),
    continueGoalEpoch: async () => ({ status: "goal_epoch_continued" }),
    recallEmbodiedHistory: async (request) => {
      options.onRecall?.(request);
      return options.recallResult ?? [];
    },
    invoke: async (name, rawInput, transactionId, agentId) => {
      options.onAction?.(name);
      const rejected = name === options.rejectAction;
      return {
        transactionId,
        agentId,
        action: name,
        input: rawInput as JsonValue,
        fingerprint: `test-${transactionId}`,
        accepted: !rejected,
        code: rejected
          ? "autonomous_skill_route_rejected"
          : name === "observe_humanoid"
          ? "humanoid_observed"
          : name === "plan_humanoid_navigation"
            ? "navigation_planned"
            : name === "execute_humanoid_navigation"
              ? "navigation_completed"
              : "accepted",
        worldBeforeRevision: 0,
        worldAfterRevision: 0,
        frameCount: name === "execute_humanoid_navigation" ? 48 : 0,
        channels: name === "execute_humanoid_navigation" ? ["locomotion"] : [],
        detail: rejected
          ? { reason: "validated route intersects a fixed obstacle" }
          : name === "observe_humanoid"
          ? { observation: "authoritative-current-sensation" }
          : {},
        committedAt: "2026-08-13T00:00:00.000Z"
      };
    }
  } satisfies HumanoidNeuralAgentRuntime;
  for (const id of Object.values(HUMANOID_NEURAL_AGENT_IDS)) {
    if (!HUMANOID_NEURAL_NODE_BY_ID.has(id)) throw new Error(`Missing node ${id}`);
  }
  return runtime;
}

async function advanceToPerception(
  runtime: HumanoidNeuralAgentRuntime,
  goalEpochId: string
): Promise<void> {
  await runtime.transitionNeuralHarnessPhase({
    phase: "goal_valuation",
    enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
    reason: "test_goal_valuation",
    goalEpochId: null,
    commitmentId: null
  });
  await runtime.transitionNeuralHarnessPhase({
    phase: "perception",
    enteredByNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
    reason: "test_goal_selected",
    goalEpochId,
    commitmentId: null
  });
}

async function publishPriorPerceptualBelief(
  runtime: HumanoidNeuralAgentRuntime,
  payload: JsonValue
) {
  const invocationId = randomUUID();
  const lease = await runtime.issueNeuralAuthorityLease({
    issuingParentNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
    targetChildNodeId: HUMANOID_NEURAL_AGENT_IDS.perceptionManager,
    allowedSignalKinds: ["perceptual_belief"],
    invocationId,
    parentInvocationId: null,
    parentEpisodeId: randomUUID(),
    ttlRevisions: 64
  });
  const signal = await runtime.publishNeuralSignal({
    kind: "perceptual_belief",
    pathway: "perceptual_association",
    direction: "ascending",
    sourceNodeId: HUMANOID_NEURAL_AGENT_IDS.perceptionManager,
    targetNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
    ttlRevisions: 64,
    priority: 90,
    sourceAuthorityLeaseId: lease.lease_id,
    invocationId,
    parentInvocationId: null,
    parentEpisodeId: lease.parent_episode_id,
    payload
  });
  await runtime.closeNeuralAuthorityLease({
    leaseId: lease.lease_id,
    closedByNodeId: HUMANOID_NEURAL_AGENT_IDS.actionSelection,
    reason: "test_perception_belief_ready"
  });
  return signal;
}
