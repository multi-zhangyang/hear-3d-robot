import { describe, expect, it } from "vitest";
import type { AgentManifest } from "../../domain/agent-manifest.js";
import type { AutonomousCycleRef } from "../../domain/autonomous-cycle.js";
import { assertGoalModelSource } from "../../domain/goal-epoch.js";
import {
  modelPayloadSha256,
  type ModelCallLifecycleRecord
} from "../../domain/model-call-authority.js";
import type { TaskNode } from "../../domain/schema.js";
import { HUMANOID_AGENT_IDS } from "./agents.js";
import { HumanoidModelAuthority } from "./model-authority.js";

const MANIFEST_EPOCH = "11111111-1111-4111-8111-111111111111";
const MANIFEST_SHA256 = "a".repeat(64);
const RESPONSE_SHA256 = "b".repeat(64);
const MODEL_CALL_ID = "22222222-2222-4222-8222-222222222222";
const STARTED_AT = "2026-08-04T00:00:00.000Z";
const COMPLETED_AT = "2026-08-04T00:00:01.000Z";

const cycle: AutonomousCycleRef = {
  cycle_id: "autonomous-cycle:33333333-3333-4333-8333-333333333333",
  cycle_index: 1,
  goal_epoch_id: `goal-epoch:${"c".repeat(64)}`
};

describe("HumanoidModelAuthority", () => {
  it("bounds completed authority memory while retaining referenced calls", () => {
    const secondModelCallId = "44444444-4444-4444-8444-444444444444";
    const records: ModelCallLifecycleRecord[] = [
      lifecycleStart(MODEL_CALL_ID, "2026-08-04T00:00:00.000Z"),
      lifecycleCompletion(MODEL_CALL_ID, "2026-08-04T00:00:01.000Z"),
      lifecycleStart(secondModelCallId, "2026-08-04T00:00:02.000Z"),
      lifecycleCompletion(secondModelCallId, "2026-08-04T00:00:03.000Z")
    ];
    const recentOnly = createAuthority(records);
    recentOnly.prune(new Set(), 1);
    expect(recentOnly.cycleForModelCall(MODEL_CALL_ID)).toBeUndefined();
    expect(recentOnly.cycleForModelCall(secondModelCallId)).toEqual(cycle);

    const referenced = createAuthority(records);
    referenced.prune(new Set([MODEL_CALL_ID]), 1);
    expect(referenced.cycleForModelCall(MODEL_CALL_ID)).toEqual(cycle);
    expect(referenced.cycleForModelCall(secondModelCallId)).toEqual(cycle);
  });

  it("binds an action decision to one durable model response and autonomous cycle", async () => {
    const records: ModelCallLifecycleRecord[] = [];
    const authority = createAuthority(records);
    const transactionId = "model-tool-action";
    const actionInput = { target: { x: 2, y: 0, z: 3 } };
    const argumentsSha256 = modelPayloadSha256(actionInput);

    await authority.recordStarted(
      HUMANOID_AGENT_IDS.motion,
      cycle,
      MODEL_CALL_ID,
      STARTED_AT
    );
    await authority.recordCompleted({
      modelCallId: MODEL_CALL_ID,
      agentId: HUMANOID_AGENT_IDS.motion,
      responseId: "response-1",
      responseOutputSha256: RESPONSE_SHA256,
      toolCalls: [{
        toolCallId: transactionId,
        toolName: "plan_humanoid_navigation",
        argumentsSha256
      }],
      at: COMPLETED_AT
    });

    const decision = authority.actionModelSource({
      toolAuthority: {
        tool_call_id: transactionId,
        tool_name: "plan_humanoid_navigation",
        arguments_sha256: argumentsSha256
      },
      expectedToolName: "plan_humanoid_navigation",
      actionInput,
      transactionId,
      agentId: HUMANOID_AGENT_IDS.motion
    });
    expect(decision).toEqual({
      agent_id: HUMANOID_AGENT_IDS.motion,
      agent_manifest_sha256: MANIFEST_SHA256,
      agent_manifest_epoch_id: MANIFEST_EPOCH,
      model_call_id: MODEL_CALL_ID,
      response_id: "response-1",
      response_output_sha256: RESPONSE_SHA256,
      tool_call_id: transactionId,
      tool_arguments_sha256: argumentsSha256
    });
    expect(authority.cycleForModelCall(MODEL_CALL_ID)).toEqual(cycle);
    expect(() => authority.assertDecisionCycleActive(decision, cycle)).not.toThrow();
    expect(() => authority.assertActionDecision({
      rawDecision: decision,
      expectedToolName: "plan_humanoid_navigation",
      actionInput,
      transactionId,
      agentId: HUMANOID_AGENT_IDS.motion,
      cycle
    })).not.toThrow();
    expect(records.map((record) => record.lifecycle)).toEqual([
      "started",
      "completed"
    ]);

    const recovered = createAuthority(records);
    expect(recovered.actionModelSource({
      toolAuthority: {
        tool_call_id: transactionId,
        tool_name: "plan_humanoid_navigation",
        arguments_sha256: argumentsSha256
      },
      expectedToolName: "plan_humanoid_navigation",
      actionInput,
      transactionId,
      agentId: HUMANOID_AGENT_IDS.motion
    })).toEqual(decision);
    expect(() => recovered.actionModelSource({
      toolAuthority: {
        tool_call_id: transactionId,
        tool_name: "plan_humanoid_navigation",
        arguments_sha256: argumentsSha256
      },
      expectedToolName: "plan_humanoid_navigation",
      actionInput: { target: { x: 9, y: 0, z: 9 } },
      transactionId,
      agentId: HUMANOID_AGENT_IDS.motion
    })).toThrow("arguments have no model authority");
  });

  it("does not grant authority when the terminal lifecycle record is not durable", async () => {
    const records: ModelCallLifecycleRecord[] = [];
    let failTerminalAppend = true;
    const authority = createAuthority(records, async (record) => {
      if (record.lifecycle !== "started" && failTerminalAppend) {
        failTerminalAppend = false;
        throw new Error("journal unavailable");
      }
      records.push(structuredClone(record));
    });
    const transactionId = "durable-model-tool";
    const actionInput = {};
    const argumentsSha256 = modelPayloadSha256(actionInput);
    await authority.recordStarted(
      HUMANOID_AGENT_IDS.sentry,
      cycle,
      MODEL_CALL_ID,
      STARTED_AT
    );
    const completion = {
      modelCallId: MODEL_CALL_ID,
      agentId: HUMANOID_AGENT_IDS.sentry,
      responseId: "response-durable",
      responseOutputSha256: RESPONSE_SHA256,
      toolCalls: [{
        toolCallId: transactionId,
        toolName: "observe_humanoid",
        argumentsSha256
      }],
      at: COMPLETED_AT
    };

    await expect(authority.recordCompleted(completion)).rejects.toThrow(
      "journal unavailable"
    );
    expect(() => authority.actionModelSource({
      toolAuthority: {
        tool_call_id: transactionId,
        tool_name: "observe_humanoid",
        arguments_sha256: argumentsSha256
      },
      expectedToolName: "observe_humanoid",
      actionInput,
      transactionId,
      agentId: HUMANOID_AGENT_IDS.sentry
    })).toThrow("no completed model response authority");

    await authority.recordCompleted(completion);
    expect(authority.actionModelSource({
      toolAuthority: {
        tool_call_id: transactionId,
        tool_name: "observe_humanoid",
        arguments_sha256: argumentsSha256
      },
      expectedToolName: "observe_humanoid",
      actionInput,
      transactionId,
      agentId: HUMANOID_AGENT_IDS.sentry
    }).model_call_id).toBe(MODEL_CALL_ID);
  });

  it("keeps Goal Manager authority distinct from action Agent authority", async () => {
    const records: ModelCallLifecycleRecord[] = [];
    const authority = createAuthority(records);
    const toolCallId = "goal-proposal-tool";
    const argumentsSha256 = modelPayloadSha256({ candidates: ["a", "b"] });
    await authority.recordStarted(
      HUMANOID_AGENT_IDS.goalManager,
      undefined,
      MODEL_CALL_ID,
      STARTED_AT
    );
    await authority.recordCompleted({
      modelCallId: MODEL_CALL_ID,
      agentId: HUMANOID_AGENT_IDS.goalManager,
      responseId: "goal-response",
      responseOutputSha256: RESPONSE_SHA256,
      toolCalls: [{
        toolCallId,
        toolName: "submit_goal_candidates",
        argumentsSha256
      }],
      at: COMPLETED_AT
    });

    expect(authority.goalModelSource({
      tool_call_id: toolCallId,
      tool_name: "submit_goal_candidates",
      arguments_sha256: argumentsSha256
    }, "submit_goal_candidates", { candidates: ["a", "b"] })).toMatchObject({
      agent_id: HUMANOID_AGENT_IDS.goalManager,
      model_call_id: MODEL_CALL_ID,
      tool_call_id: toolCallId
    });
    expect(() => authority.actionModelSource({
      toolAuthority: {
        tool_call_id: toolCallId,
        tool_name: "submit_goal_candidates",
        arguments_sha256: argumentsSha256
      },
      expectedToolName: "observe_humanoid",
      actionInput: {},
      transactionId: toolCallId,
      agentId: HUMANOID_AGENT_IDS.goalManager
    })).toThrow("tool authority mismatch");
  });

  it("keeps a durably verified Goal source authorized across an archived Agent epoch", async () => {
    const records: ModelCallLifecycleRecord[] = [];
    const archived = {
      ...manifest(),
      epoch_id: "44444444-4444-4444-8444-444444444444",
      identity_sha256: "d".repeat(64)
    } as AgentManifest;
    const authority = createAuthority(
      records,
      async (record) => { records.push(structuredClone(record)); },
      [archived]
    );
    const toolCallId = "archived-goal-proposal";
    const argumentsSha256 = modelPayloadSha256({ candidates: ["archived"] });
    await authority.recordStarted(
      HUMANOID_AGENT_IDS.goalManager,
      undefined,
      MODEL_CALL_ID,
      STARTED_AT
    );
    await authority.recordCompleted({
      modelCallId: MODEL_CALL_ID,
      agentId: HUMANOID_AGENT_IDS.goalManager,
      responseId: "archived-goal-response",
      responseOutputSha256: RESPONSE_SHA256,
      toolCalls: [{
        toolCallId,
        toolName: "submit_goal_candidates",
        argumentsSha256
      }],
      at: COMPLETED_AT
    });
    const source = {
      agent_id: HUMANOID_AGENT_IDS.goalManager,
      agent_manifest_sha256: archived.identity_sha256,
      agent_manifest_epoch_id: archived.epoch_id,
      model_call_id: MODEL_CALL_ID,
      response_id: "archived-goal-response",
      response_output_sha256: RESPONSE_SHA256,
      tool_call_id: toolCallId,
      tool_arguments_sha256: argumentsSha256
    };
    const harness = authority.goalHarness({
      evidence: new Map(),
      scenario: {} as never
    });

    expect(() => assertGoalModelSource(
      source,
      harness,
      "submit_goal_candidates"
    )).not.toThrow();
    expect(() => assertGoalModelSource(
      { ...source, agent_manifest_sha256: "e".repeat(64) },
      harness,
      "submit_goal_candidates"
    )).toThrowError(expect.objectContaining({ code: "unauthorized_model_source" }));
  });

  it("binds raw model arguments and normalized SDK input without weakening authority", async () => {
    const records: ModelCallLifecycleRecord[] = [];
    const authority = createAuthority(records);
    const transactionId = "normalized-model-tool";
    const rawInput = {
      target: { x: 2, y: 0, z: 3 },
      optional_sdk_field: null
    };
    const normalizedInput = { target: { x: 2, y: 0, z: 3 } };
    const rawArgumentsSha256 = modelPayloadSha256(rawInput);
    const normalizedArgumentsSha256 = modelPayloadSha256(normalizedInput);
    await authority.recordStarted(
      HUMANOID_AGENT_IDS.motion,
      cycle,
      MODEL_CALL_ID,
      STARTED_AT
    );
    await authority.recordCompleted({
      modelCallId: MODEL_CALL_ID,
      agentId: HUMANOID_AGENT_IDS.motion,
      responseId: "response-normalized",
      responseOutputSha256: RESPONSE_SHA256,
      toolCalls: [{
        toolCallId: transactionId,
        toolName: "plan_humanoid_navigation",
        argumentsSha256: rawArgumentsSha256
      }],
      at: COMPLETED_AT
    });

    const decision = authority.actionModelSource({
      toolAuthority: {
        tool_call_id: transactionId,
        tool_name: "plan_humanoid_navigation",
        arguments_sha256: rawArgumentsSha256,
        normalized_arguments_sha256: normalizedArgumentsSha256
      },
      expectedToolName: "plan_humanoid_navigation",
      actionInput: normalizedInput,
      transactionId,
      agentId: HUMANOID_AGENT_IDS.motion
    });
    expect(decision).toMatchObject({
      tool_arguments_sha256: rawArgumentsSha256,
      normalized_tool_arguments_sha256: normalizedArgumentsSha256
    });
    expect(() => authority.assertActionDecision({
      rawDecision: decision,
      expectedToolName: "plan_humanoid_navigation",
      actionInput: normalizedInput,
      transactionId,
      agentId: HUMANOID_AGENT_IDS.motion,
      cycle
    })).not.toThrow();
    expect(() => authority.assertActionDecision({
      rawDecision: decision,
      expectedToolName: "plan_humanoid_navigation",
      actionInput: { target: { x: 9, y: 0, z: 9 } },
      transactionId,
      agentId: HUMANOID_AGENT_IDS.motion,
      cycle
    })).toThrow("decision is not authoritative");
  });
});

function lifecycleStart(
  modelCallId: string,
  at: string
): ModelCallLifecycleRecord {
  return {
    version: 1,
    lifecycle: "started",
    model_call_id: modelCallId,
    agent_id: HUMANOID_AGENT_IDS.motion,
    cycle,
    at
  };
}

function lifecycleCompletion(
  modelCallId: string,
  at: string
): ModelCallLifecycleRecord {
  return {
    version: 1,
    lifecycle: "completed",
    model_call_id: modelCallId,
    agent_id: HUMANOID_AGENT_IDS.motion,
    response_id: `response-${modelCallId}`,
    response_output_sha256: RESPONSE_SHA256,
    tool_calls: [],
    cycle,
    at
  };
}

function createAuthority(
  records: ModelCallLifecycleRecord[],
  appendRecord: (record: ModelCallLifecycleRecord) => Promise<void> = async (record) => {
    records.push(structuredClone(record));
  },
  archivedManifests: readonly AgentManifest[] = []
): HumanoidModelAuthority {
  return HumanoidModelAuthority.restore({
    manifest: manifest(),
    archivedManifests,
    nodes: hierarchyNodes(),
    records,
    appendRecord
  });
}

function manifest(): AgentManifest {
  return {
    epoch_id: MANIFEST_EPOCH,
    identity_sha256: MANIFEST_SHA256,
    agents: {
      goal_manager: { agent_id: HUMANOID_AGENT_IDS.goalManager },
      coordinator: { agent_id: HUMANOID_AGENT_IDS.coordinator },
      sentry: { agent_id: HUMANOID_AGENT_IDS.sentry },
      motion: { agent_id: HUMANOID_AGENT_IDS.motion },
      executor: { agent_id: HUMANOID_AGENT_IDS.executor }
    }
  } as AgentManifest;
}

function hierarchyNodes(): Record<string, TaskNode> {
  return Object.fromEntries([
    HUMANOID_AGENT_IDS.goalManager,
    HUMANOID_AGENT_IDS.coordinator,
    HUMANOID_AGENT_IDS.sentry,
    HUMANOID_AGENT_IDS.motion,
    HUMANOID_AGENT_IDS.executor
  ].map((agentId) => [agentId, { id: agentId } as TaskNode]));
}
