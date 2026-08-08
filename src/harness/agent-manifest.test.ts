import {
  MemorySession,
  type Model
} from "@openai/agents";
import { createHash } from "node:crypto";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  loadRuntimeCatalog,
  type ProviderConfig
} from "../config/load.js";
import { AgentManifestSchema } from "../domain/agent-manifest.js";
import { RunStore } from "../persistence/run-store.js";
import {
  AgentManifestIncompatibleError,
  assertAgentManifestCompatible,
  createHumanoidAgentManifest
} from "./agent-manifest.js";
import {
  createHumanoidAgentHierarchy,
  HUMANOID_AGENT_TOOL_CONTRACTS
} from "./humanoid/agents.js";

const provider: ProviderConfig = {
  protocol: "openai_compatible",
  baseUrl: "https://models.example.test/v1",
  model: "default-model",
  apiKey: "credential-that-must-not-be-persisted",
  requestTimeoutMs: 90_000,
  temperature: 0.2,
  maxOutputTokens: 2048,
  contextWindowTokens: 32_768,
  compactTriggerTokens: 8192,
  compactRecentModelTurns: 4,
  compactMaxOutputTokens: 1024,
  agentModels: {
    goal_manager: profile("goal-manager-model", 0.15),
    coordinator: profile("coordinator-model", 0.1),
    sentry: profile("sentry-model", 0.2),
    motion: profile("motion-model", 0.3),
    executor: profile("executor-model", 0.4),
    compactor: profile("compactor-model", 0.5)
  }
};

describe("agent manifest", () => {
  it("persists the complete runtime identity without persisting credentials", () => {
    const manifest = createManifest(provider, "11111111-1111-4111-8111-111111111111");

    expect(Object.keys(manifest.agents)).toEqual([
      "goal_manager",
      "coordinator",
      "sentry",
      "motion",
      "executor",
      "compactor"
    ]);
    expect(manifest.agents.motion).toMatchObject({
      agent_id: "humanoid-motion-reference",
      agent_name: "全身运动参考智能体",
      role: "motion",
      protocol: "openai_compatible",
      model: "motion-model",
      endpoint_sha256: expect.stringMatching(/^[a-f0-9]{64}$/),
      sdk_model_settings: {
        temperature: 0.3,
        maxTokens: 2048,
        parallelToolCalls: false,
        toolChoice: "auto"
      },
      reset_tool_choice: false,
      tool_use_behavior: {
        kind: "harness_callback",
        contract_id: "accepted_humanoid_action_receipt_v1",
        terminal_tool_names: [
          "plan_humanoid_skill",
          "plan_whole_body_motion_candidates"
        ]
      },
      settings: {
        temperature: 0.3,
        context_window_tokens: 32768,
        compact_trigger_tokens: 8192
      }
    });
    expect(manifest.runtime_sdk_identity).toEqual({
      "@openai/agents": "test-sdk",
      "@openai/agents-extensions": "test-bridge"
    });
    expect(manifest.harness_contract_version).toBe(13);
    expect(manifest.agents.goal_manager.tool_use_behavior).toEqual({
      kind: "harness_callback",
      contract_id: "verified_harness_terminal_status_v1",
      terminal_tool_names: ["select_goal_candidate", "retire_goal_epoch"]
    });
    expect(manifest.agents.coordinator.tool_use_behavior).toEqual({
      kind: "harness_callback",
      contract_id: "verified_harness_terminal_status_v1",
      terminal_tool_names: [
        "complete_autonomous_cycle",
        "complete_goal_transition",
        "complete_satisfied_goal"
      ]
    });
    expect(manifest.agent_tool_contracts).toEqual([
      expect.objectContaining({
        tool_name: "delegate_goal_manager",
        target_role: "goal_manager",
        target_agent_id: "humanoid-goal-manager",
        input_builder_contract: "goal_manager_authority_envelope_v2"
      }),
      expect.objectContaining({
        tool_name: "delegate_humanoid_sentry",
        target_role: "sentry",
        target_agent_id: "humanoid-sentry",
        input_builder_contract: "live_authority_delegation_v1",
        input_builder_sha256: expect.stringMatching(/^[a-f0-9]{64}$/),
        run_options: {
          session_agent_id: "humanoid-sentry",
          context_source: "parent_run_context",
          max_turns: "unbounded"
        },
        resume_context_strategy: "merge"
      }),
      expect.objectContaining({
        tool_name: "delegate_motion_reference",
        target_role: "motion",
        input_builder_contract: "motion_authority_envelope_v1"
      }),
      expect.objectContaining({
        tool_name: "delegate_physics_executor",
        target_role: "executor",
        input_builder_contract: "validated_execution_task_json_v1"
      })
    ]);
    expect(manifest.agents.coordinator.instructions_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(manifest.agents.coordinator.tool_schema_sha256).toMatch(/^[a-f0-9]{64}$/);
    const persisted = JSON.stringify(manifest);
    expect(persisted).not.toContain("credential-that-must-not-be-persisted");
    expect(persisted).not.toContain("models.example.test");
    expect(persisted).not.toContain("/v1");
    expect(manifest.agents.motion).not.toHaveProperty("endpoint_origin");
  });

  it("persists only a digest when an endpoint contains credentials or query data", () => {
    const sensitiveEndpoint =
      "https://url-user:url-password@private.example.test/v1?api_key=url-secret";
    const sensitiveProvider: ProviderConfig = {
      ...provider,
      baseUrl: sensitiveEndpoint,
      agentModels: {
        goal_manager: { ...provider.agentModels!.goal_manager, baseUrl: sensitiveEndpoint },
        coordinator: { ...provider.agentModels!.coordinator, baseUrl: sensitiveEndpoint },
        sentry: { ...provider.agentModels!.sentry, baseUrl: sensitiveEndpoint },
        motion: { ...provider.agentModels!.motion, baseUrl: sensitiveEndpoint },
        executor: { ...provider.agentModels!.executor, baseUrl: sensitiveEndpoint },
        compactor: { ...provider.agentModels!.compactor, baseUrl: sensitiveEndpoint }
      }
    };
    const manifest = createManifest(
      sensitiveProvider,
      "11111111-1111-4111-8111-111111111111"
    );
    const persisted = JSON.stringify(manifest);

    expect(manifest.agents.coordinator.endpoint_sha256).toBe(
      createHash("sha256").update(new URL(sensitiveEndpoint).href).digest("hex")
    );
    for (const sensitivePart of [
      "url-user",
      "url-password",
      "private.example.test",
      "/v1",
      "api_key",
      "url-secret"
    ]) {
      expect(persisted).not.toContain(sensitivePart);
    }
  });

  it("rejects legacy endpoint-bearing identities instead of accepting leaked metadata", () => {
    const legacy = structuredClone(
      createManifest(provider, "11111111-1111-4111-8111-111111111111")
    ) as unknown as Record<string, unknown>;
    const agents = legacy.agents as Record<string, Record<string, unknown>>;
    for (const identity of Object.values(agents)) {
      identity.endpoint_origin = "https://legacy.example.test";
    }

    expect(AgentManifestSchema.safeParse(legacy).success).toBe(false);
  });

  it("hashes provider-specific model settings instead of persisting their contents", () => {
    const hierarchy = hierarchyFixture(provider);
    const manifest = createHumanoidAgentManifest({
      hierarchy: {
        ...hierarchy,
        coordinator: hierarchy.coordinator.clone({
          modelSettings: {
            ...hierarchy.coordinator.modelSettings,
            providerData: { authorization: "provider-runtime-secret" }
          }
        })
      },
      provider,
      epochId: "11111111-1111-4111-8111-111111111111",
      runtimeSdkIdentity: { "@openai/agents": "test-sdk" }
    });

    expect(manifest.agents.coordinator.sdk_model_settings.providerData).toEqual({
      redacted_sha256: expect.stringMatching(/^[a-f0-9]{64}$/)
    });
    expect(JSON.stringify(manifest)).not.toContain("provider-runtime-secret");
  });

  it("allows a new process to reuse only a compatible persisted epoch", () => {
    const persisted = createManifest(provider, "11111111-1111-4111-8111-111111111111");
    const sameIdentity = createManifest(provider, "22222222-2222-4222-8222-222222222222");

    expect(() => assertAgentManifestCompatible(persisted, sameIdentity)).not.toThrow();

    const changedProvider: ProviderConfig = {
      ...provider,
      agentModels: {
        ...provider.agentModels!,
        motion: { ...provider.agentModels!.motion, model: "different-motion-model" }
      }
    };
    const changed = createManifest(
      changedProvider,
      "11111111-1111-4111-8111-111111111111"
    );
    expect(() => assertAgentManifestCompatible(persisted, changed)).toThrowError(
      expect.objectContaining({
        code: "agent_manifest_incompatible",
        changedFields: ["agents.motion.model"]
      })
    );
    expect(() => assertAgentManifestCompatible(persisted, changed))
      .toThrowError(AgentManifestIncompatibleError);
  });

  it("rejects instruction, tool schema, SDK, and integrity changes explicitly", () => {
    const hierarchy = hierarchyFixture(provider);
    const base = createHumanoidAgentManifest({
      hierarchy,
      provider,
      epochId: "11111111-1111-4111-8111-111111111111",
      runtimeSdkIdentity: { "@openai/agents": "test-sdk" }
    });
    const changedInstructions = createHumanoidAgentManifest({
      hierarchy: {
        ...hierarchy,
        coordinator: hierarchy.coordinator.clone({ instructions: "changed instructions" })
      },
      provider,
      epochId: base.epoch_id,
      runtimeSdkIdentity: base.runtime_sdk_identity
    });
    expect(() => assertAgentManifestCompatible(base, changedInstructions))
      .toThrow("agents.coordinator.instructions_sha256");

    expect(() => createHumanoidAgentManifest({
      hierarchy: {
        ...hierarchy,
        coordinator: hierarchy.coordinator.clone({
          tools: hierarchy.coordinator.tools.slice(0, -1)
        })
      },
      provider,
      epochId: base.epoch_id,
      runtimeSdkIdentity: base.runtime_sdk_identity
    })).toThrow("missing terminal tool complete_satisfied_goal");

    const changedSdk = createHumanoidAgentManifest({
      hierarchy,
      provider,
      epochId: base.epoch_id,
      runtimeSdkIdentity: { "@openai/agents": "next-sdk" }
    });
    expect(() => assertAgentManifestCompatible(base, changedSdk))
      .toThrow("runtime_sdk_identity");

    const tampered = structuredClone(base);
    tampered.agents.executor.model = "tampered-without-rehash";
    expect(() => assertAgentManifestCompatible(tampered, base))
      .toThrow("persisted agent manifest identity hash is invalid");
  });

  it("rejects Agent names and SDK execution settings that changed across recovery", () => {
    const hierarchy = hierarchyFixture(provider);
    const base = createHumanoidAgentManifest({
      hierarchy,
      provider,
      epochId: "11111111-1111-4111-8111-111111111111",
      runtimeSdkIdentity: { "@openai/agents": "test-sdk" }
    });

    const renamed = createHumanoidAgentManifest({
      hierarchy: {
        ...hierarchy,
        coordinator: hierarchy.coordinator.clone({ name: "新的协调智能体名称" })
      },
      provider,
      epochId: base.epoch_id,
      runtimeSdkIdentity: base.runtime_sdk_identity
    });
    expect(() => assertAgentManifestCompatible(base, renamed)).toThrowError(
      expect.objectContaining({
        changedFields: ["agents.coordinator.agent_name"]
      })
    );

    const settingsChanged = createHumanoidAgentManifest({
      hierarchy: {
        ...hierarchy,
        motion: hierarchy.motion.clone({
          modelSettings: {
            ...hierarchy.motion.modelSettings,
            parallelToolCalls: true,
            toolChoice: "auto"
          }
        })
      },
      provider,
      epochId: base.epoch_id,
      runtimeSdkIdentity: base.runtime_sdk_identity
    });
    expect(() => assertAgentManifestCompatible(base, settingsChanged)).toThrowError(
      expect.objectContaining({
        changedFields: ["agents.motion.sdk_model_settings"]
      })
    );

    const behaviorChanged = createHumanoidAgentManifest({
      hierarchy: {
        ...hierarchy,
        executor: hierarchy.executor.clone({
          resetToolChoice: true,
          toolUseBehavior: "run_llm_again"
        })
      },
      provider,
      epochId: base.epoch_id,
      runtimeSdkIdentity: base.runtime_sdk_identity
    });
    expect(() => assertAgentManifestCompatible(base, behaviorChanged)).toThrowError(
      expect.objectContaining({
        changedFields: [
          "agents.executor.reset_tool_choice",
          "agents.executor.tool_use_behavior"
        ]
      })
    );
  });

  it("binds the explicit Agent-as-tool and Harness contracts into the identity hash", () => {
    const base = createManifest(provider, "11111111-1111-4111-8111-111111111111");
    const tamperedToolContract = structuredClone(base);
    tamperedToolContract.agent_tool_contracts[0]!.input_builder_contract =
      "validated_execution_task_json_v1";
    expect(() => assertAgentManifestCompatible(tamperedToolContract, base))
      .toThrow("persisted agent manifest identity hash is invalid");

    const tamperedHarnessVersion = structuredClone(base);
    tamperedHarnessVersion.harness_contract_version += 1;
    expect(() => assertAgentManifestCompatible(tamperedHarnessVersion, base))
      .toThrow("persisted agent manifest identity hash is invalid");
  });

  it("uses stable declared Agent-as-tool contracts across runtime compilation", () => {
    const base = createManifest(provider, "11111111-1111-4111-8111-111111111111");
    const sentryContract = HUMANOID_AGENT_TOOL_CONTRACTS.sentry as unknown as {
      inputBuilder: () => string;
      inputBuilderContract: "live_authority_delegation_v1"
        | "validated_execution_task_json_v1";
      resumeContextStrategy: "merge" | "replace" | "preferSerialized";
      runOptions: {
        sessionAgentId: string;
      };
    };
    const originalInputBuilder = sentryContract.inputBuilder;
    const originalInputBuilderContract = sentryContract.inputBuilderContract;
    const originalResumeStrategy = sentryContract.resumeContextStrategy;
    const originalSessionAgentId = sentryContract.runOptions.sessionAgentId;
    try {
      sentryContract.inputBuilder = () => "changed input builder implementation";
      const recompiledBuilder = createManifest(
        provider,
        "22222222-2222-4222-8222-222222222222"
      );
      expect(recompiledBuilder.agent_tool_contracts).toEqual(base.agent_tool_contracts);
      expect(() => assertAgentManifestCompatible(base, recompiledBuilder)).not.toThrow();

      sentryContract.inputBuilderContract = "validated_execution_task_json_v1";
      const changedContract = createManifest(
        provider,
        "33333333-3333-4333-8333-333333333333"
      );
      expect(() => assertAgentManifestCompatible(base, changedContract)).toThrowError(
        expect.objectContaining({ changedFields: ["agent_tool_contracts"] })
      );

      sentryContract.inputBuilder = originalInputBuilder;
      sentryContract.inputBuilderContract = originalInputBuilderContract;
      sentryContract.resumeContextStrategy = "replace";
      const changedResumeStrategy = createManifest(
        provider,
        "44444444-4444-4444-8444-444444444444"
      );
      expect(() => assertAgentManifestCompatible(base, changedResumeStrategy)).toThrowError(
        expect.objectContaining({ changedFields: ["agent_tool_contracts"] })
      );

      sentryContract.resumeContextStrategy = originalResumeStrategy;
      sentryContract.runOptions.sessionAgentId = "humanoid-motion-reference";
      expect(() => createManifest(
        provider,
        "55555555-5555-4555-8555-555555555555"
      )).toThrow("Session owner does not match its target role");
    } finally {
      sentryContract.inputBuilder = originalInputBuilder;
      sentryContract.inputBuilderContract = originalInputBuilderContract;
      sentryContract.resumeContextStrategy = originalResumeStrategy;
      sentryContract.runOptions.sessionAgentId = originalSessionAgentId;
    }
  });

  it("uses a dedicated compactor model identity instead of the coordinator profile", () => {
    const manifest = createManifest(provider, "11111111-1111-4111-8111-111111111111");
    expect(manifest.agents.compactor.model).toBe("compactor-model");
    expect(manifest.agents.compactor.agent_id).toBe("humanoid-context-compactor");
    expect(manifest.agents.compactor.instructions_sha256)
      .not.toBe(manifest.agents.coordinator.instructions_sha256);
  });

  it("records Node, AI SDK, and only the protocol adapters actually in use", () => {
    const manifest = createHumanoidAgentManifest({
      hierarchy: hierarchyFixture(provider),
      provider,
      epochId: "11111111-1111-4111-8111-111111111111"
    });
    expect(manifest.runtime_sdk_identity).toMatchObject({
      node: expect.stringMatching(/^\d+\.\d+\.\d+/),
      "@openai/agents": expect.stringMatching(/^\d+\.\d+\.\d+/),
      "@openai/agents-extensions": expect.stringMatching(/^\d+\.\d+\.\d+/),
      "@ai-sdk/openai-compatible": expect.stringMatching(/^\d+\.\d+\.\d+/),
      ai: expect.stringMatching(/^\d+\.\d+\.\d+/)
    });
    expect(manifest.runtime_sdk_identity).not.toHaveProperty("@ai-sdk/openai");
    expect(manifest.runtime_sdk_identity).not.toHaveProperty("@ai-sdk/anthropic");
  });

  it("uses code-point canonical ordering and keeps a fixed identity hash", () => {
    const first = createHumanoidAgentManifest({
      hierarchy: hierarchyFixture(provider),
      provider,
      epochId: "11111111-1111-4111-8111-111111111111",
      createdAt: "2026-01-01T00:00:00.000Z",
      runtimeSdkIdentity: {
        "\u{10000}": "astral",
        "\uE000": "private-use",
        A: "ascii"
      }
    });
    const second = createHumanoidAgentManifest({
      hierarchy: hierarchyFixture(provider),
      provider,
      epochId: "22222222-2222-4222-8222-222222222222",
      createdAt: "2027-01-01T00:00:00.000Z",
      runtimeSdkIdentity: {
        A: "ascii",
        "\uE000": "private-use",
        "\u{10000}": "astral"
      }
    });
    expect(first.identity_sha256).toBe(second.identity_sha256);
    expect(first.identity_sha256).toBe(
      "f2d640a9597bf272ffb6d14191f09cd34be37475b8ec0798d578950372306038"
    );
  });

  it("durably stores the epoch and refuses a run with no identity manifest", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-agent-manifest-"));
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("humanoid_courtyard", 3);
    const store = await RunStore.create(runsDir, {
      mission: "Verify the agent identity epoch",
      scenarioId: "humanoid_courtyard",
      scenario,
      goal: scenario.default_goal
    });
    await expect(store.readAgentManifest()).rejects.toThrow(
      "refusing to reuse unverified model or Session state"
    );

    const manifest = createManifest(provider, "11111111-1111-4111-8111-111111111111");
    await store.writeAgentManifest(manifest);
    const reopened = await RunStore.open(store.runDir);
    expect(await reopened.readAgentManifest()).toEqual(manifest);
  });
});

function createManifest(config: ProviderConfig, epochId: string) {
  return createHumanoidAgentManifest({
    hierarchy: hierarchyFixture(config),
    provider: config,
    epochId,
    runtimeSdkIdentity: {
      "@openai/agents": "test-sdk",
      "@openai/agents-extensions": "test-bridge"
    }
  });
}

function hierarchyFixture(config: ProviderConfig) {
  return createHumanoidAgentHierarchy({
    provider: config,
    runtime: {
      invoke: async () => { throw new Error("outside construction test"); },
      recallEmbodiedHistory: async () => { throw new Error("outside construction test"); },
      validateCycleEvidence: () => { throw new Error("outside construction test"); }
    } as never,
    createModel: () => modelStub(),
    createSession: (agentId) => new MemorySession({ sessionId: agentId }),
    callModelInputFilter: ({ modelData }) => modelData
  });
}

function profile(model: string, temperature: number) {
  return {
    protocol: "openai_compatible" as const,
    baseUrl: "https://models.example.test/v1",
    model,
    apiKey: "credential-that-must-not-be-persisted",
    requestTimeoutMs: 90_000,
    temperature,
    maxOutputTokens: 2048,
    contextWindowTokens: 32_768,
    compactTriggerTokens: 8192,
    compactRecentModelTurns: 4,
    compactMaxOutputTokens: 1024
  };
}

function modelStub(): Model {
  return {
    getResponse: async () => { throw new Error("outside construction test"); },
    getStreamedResponse: () => { throw new Error("outside construction test"); }
  } as unknown as Model;
}
