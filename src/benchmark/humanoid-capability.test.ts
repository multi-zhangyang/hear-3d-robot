import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  HUMANOID_END_EFFECTORS,
  type JsonValue
} from "../domain/schema.js";
import {
  advancePhysicalTrajectorySha256,
  physicalTrajectoryFrameSha256
} from "../domain/physical-trajectory.js";
import {
  createHumanoidCapabilityBenchmarkReport,
  measureHumanoidCapabilityRun
} from "./humanoid-capability.js";
import { loadHumanoidCapabilityBenchmark } from
  "./humanoid-capability-files.js";

const temporaryDirectories: string[] = [];
const at = "2026-08-10T10:00:00.000Z";

afterEach(async () => {
  const { rm } = await import("node:fs/promises");
  await Promise.all(temporaryDirectories.splice(0).map((path) => (
    rm(path, { recursive: true, force: true })
  )));
});

describe("humanoid capability benchmark", () => {
  it("measures learned control, planning outcomes, safety and task predicates", () => {
    const input = runInput("run-success", true);
    const execution = input.actions[1] as Record<string, unknown>;
    const detail = execution.detail as Record<string, unknown>;
    const result = detail.result as Record<string, unknown>;
    result.controller_routing = controllerRoutingEvidence();
    const measured = measureHumanoidCapabilityRun(input);

    expect(measured).toMatchObject({
      run_id: "run-success",
      mission_success: true,
      final_fallen: false,
      duration_ms: 12_000,
      model: {
        calls: 6,
        usage_available: true,
        requests: 6,
        total_tokens: 1_200
      },
      actions: {
        planning_total: 1,
        planning_accepted: 1,
        execution_total: 1,
        execution_succeeded: 1,
        execution_frame_count: 1,
        online_replans: 1,
        online_replans_accepted: 1,
        achieved_predicate_counts: { grasp_verified: 1 }
      },
      motion: {
        trajectory_count: 1,
        complete_trajectory_count: 1,
        observed_frame_count: 1,
        root_planar_path_length_m: 0.25,
        controller_usage: {
          available: true,
          complete_from_admission: true,
          observed_frame_count: 1,
          learned_policy_frame_ratio: 1,
          reference_control_frame_ratio: 0
        },
        routing: {
          decision_count: 1,
          admitted_count: 1,
          rejected_count: 0,
          memory_bridge_attempt_count: 1,
          memory_bridge_completed_count: 1,
          memory_bridge_completion_rate: 1,
          transition_attempts: 1,
          transition_successes: 1
        }
      },
      safety: {
        evidence_execution_count: 1,
        minimum_support_margin_m: 0.03,
        maximum_foot_slip_mps: 0.12,
        minimum_joint_limit_margin_rad: 0.2,
        maximum_requested_effort_utilization: 0.8,
        saturated_execution_count: 0,
        peak_contact_normal_force_n: 80
      }
    });
  });

  it("groups policies without hiding missing controller authority evidence", () => {
    const learned = measureHumanoidCapabilityRun(runInput("run-success", true));
    const legacyInput = runInput("run-failed", false);
    const execution = legacyInput.actions[1] as Record<string, unknown>;
    const detail = execution.detail as Record<string, unknown>;
    const trajectory = detail.physical_trajectory as Record<string, unknown>;
    delete trajectory.controller_usage;
    const legacy = measureHumanoidCapabilityRun(legacyInput);

    const report = createHumanoidCapabilityBenchmarkReport(
      [legacy, learned],
      "2026-08-10T10:05:00.000Z"
    );

    expect(report.run_count).toBe(2);
    expect(report.summary).toMatchObject({
      run_count: 2,
      succeeded_run_count: 1,
      success_rate: 0.5,
      fallen_run_count: 1,
      fall_rate: 0.5,
      controller_usage_complete: false,
      learned_policy_frame_ratio: null
    });
    expect(report.groups).toHaveLength(1);
    expect(report.runs.map((run) => run.run_id)).toEqual([
      "run-failed",
      "run-success"
    ]);
  });

  it("loads one or more durable run directories without mutating them", async () => {
    const root = await mkdtemp(join(tmpdir(), "hear-capability-benchmark-"));
    temporaryDirectories.push(root);
    const input = runInput("run-success", true);
    const runDir = join(root, "run-success");
    await mkdir(runDir);
    await Promise.all([
      writeFile(join(runDir, "run.json"), JSON.stringify(input.definition)),
      writeFile(join(runDir, "checkpoint.json"), JSON.stringify(input.checkpoint)),
      writeFile(join(runDir, "actions.jsonl"), input.actions.map(
        (action) => JSON.stringify(action)
      ).join("\n") + "\n")
    ]);

    const report = await loadHumanoidCapabilityBenchmark(
      root,
      "2026-08-10T10:06:00.000Z"
    );

    expect(report.runs).toHaveLength(1);
    expect(report.runs[0]?.actions.achieved_predicate_counts).toEqual({
      grasp_verified: 1
    });
  });

  it("recovers exact legacy model usage from the provider journal", async () => {
    const root = await mkdtemp(join(tmpdir(), "hear-capability-legacy-"));
    temporaryDirectories.push(root);
    const input = runInput("run-legacy", true);
    const checkpoint = input.checkpoint as Record<string, unknown>;
    delete checkpoint.model_usage;
    await Promise.all([
      writeFile(join(root, "run.json"), JSON.stringify(input.definition)),
      writeFile(join(root, "checkpoint.json"), JSON.stringify(checkpoint)),
      writeFile(join(root, "actions.jsonl"), ""),
      writeFile(join(root, "provider.jsonl"), [
        JSON.stringify({
          agent_id: "humanoid-coordinator",
          usage: {
            inputTokens: 80,
            outputTokens: 20,
            totalTokens: 100,
            inputTokensDetails: { cached_tokens: 12 },
            outputTokensDetails: { reasoning_tokens: 4 }
          },
          at
        }),
        JSON.stringify({
          agent_id: "humanoid-coordinator",
          usage: {
            requests: 2,
            inputTokens: 150,
            outputTokens: 50,
            totalTokens: 200
          },
          at: "2026-08-10T10:00:01.000Z"
        })
      ].join("\n") + "\n")
    ]);

    const report = await loadHumanoidCapabilityBenchmark(root);

    expect(report.runs[0]?.model).toMatchObject({
      usage_available: true,
      requests: 3,
      input_tokens: 230,
      output_tokens: 70,
      total_tokens: 300,
      cached_input_tokens: 12,
      reasoning_tokens: 4,
      by_agent_requests: { "humanoid-coordinator": 3 }
    });
  });

  it("marks absent model usage as unavailable instead of inventing zero tokens", () => {
    const input = runInput("run-without-usage", true);
    delete (input.checkpoint as Record<string, unknown>).model_usage;

    const measured = measureHumanoidCapabilityRun(input);

    expect(measured.model).toMatchObject({
      usage_available: false,
      requests: null,
      total_tokens: null
    });
    expect(createHumanoidCapabilityBenchmarkReport([measured]).summary).toMatchObject({
      model_usage_complete: false,
      total_model_tokens: null
    });
  });
});

function runInput(runId: string, success: boolean): {
  definition: unknown;
  checkpoint: unknown;
  actions: unknown[];
} {
  const trajectory = physicalTrajectory();
  const definition = {
    version: 1,
    run_id: runId,
    scenario_id: "humanoid_workyard",
    run_mode: "mission",
    created_at: at
  };
  const checkpoint = {
    runtime: "humanoid_g1",
    run_id: runId,
    scenario_id: "humanoid_workyard",
    status: success ? "succeeded" : "failed",
    world: {
      frame: 10,
      worldRevision: 10,
      robot: {
        fallen: !success,
        controller: {
          implementation: "workyard-policy-v1",
          learnedPolicy: {
            capabilities: ["balance", "contact_rich_manipulation"]
          }
        },
        controllerExecution: {
          mode: "learned_policy",
          activeImplementation: "workyard-policy-v1"
        }
      }
    },
    total_model_calls: 6,
    model_usage: {
      version: 1,
      total: usage(6, 1_200),
      by_agent: {
        "humanoid-goal-manager": usage(2, 400),
        "humanoid-coordinator": usage(4, 800)
      },
      updated_at: "2026-08-10T10:00:12.000Z"
    },
    checker: { success },
    created_at: at,
    updated_at: "2026-08-10T10:00:12.000Z"
  };
  const planningId = `${runId}-plan`;
  const plan = action({
    transactionId: planningId,
    action: "plan_whole_body_motion_candidates",
    accepted: true,
    code: "whole_body_candidates_validated",
    detail: {
      termination: {
        predicates: [{ type: "grasp_verified", object_id: "assembly_rod" }]
      }
    }
  });
  const execution = action({
    transactionId: `${runId}-execute`,
    action: "execute_whole_body_motion",
    accepted: success,
    code: success ? "motion_option_succeeded" : "motion_failed",
    input: { planning_transaction_id: planningId },
    frameCount: 1,
    detail: {
      physical_trajectory: trajectory,
      result: {
        physical_safety: safetyEvidence(),
        online_replans: [{ accepted: true }]
      }
    }
  });
  return { definition, checkpoint, actions: [plan, execution] };
}

function action(input: {
  transactionId: string;
  action: string;
  accepted: boolean;
  code: string;
  input?: JsonValue;
  frameCount?: number;
  detail: JsonValue;
}): unknown {
  return {
    transactionId: input.transactionId,
    agentId: input.action.startsWith("plan_")
      ? "humanoid-motion-reference"
      : "humanoid-executor",
    action: input.action,
    input: input.input ?? {},
    fingerprint: input.transactionId,
    accepted: input.accepted,
    code: input.code,
    worldBeforeRevision: 0,
    worldAfterRevision: input.frameCount ?? 0,
    frameCount: input.frameCount ?? 0,
    channels: ["left_arm"],
    detail: input.detail,
    committedAt: at
  };
}

function physicalTrajectory(): unknown {
  const endEffectors = Object.fromEntries(HUMANOID_END_EFFECTORS.map(
    (name) => [name, { x: 0, y: 1, z: 0 }]
  ));
  const identity = {
    frame: 1,
    world_revision: 1,
    root_position: { x: 0.25, y: 0.75, z: 0 },
    root_rotation: { x: 0, y: 0, z: 0, w: 1 },
    joint_positions: Array.from({ length: 43 }, () => 0),
    end_effectors: endEffectors,
    contacts: [],
    objects: [],
    support: "double" as const,
    fallen: false,
    controller_execution: {
      mode: "learned_policy" as const,
      active_implementation: "workyard-policy-v1",
      transition: null
    }
  };
  const frameSha256 = physicalTrajectoryFrameSha256(identity);
  return {
    version: 1,
    complete_from_admission: true,
    start_frame: 1,
    end_frame: 1,
    start_world_revision: 1,
    end_world_revision: 1,
    observed_frame_count: 1,
    sample_stride: 1,
    joint_names: Array.from({ length: 43 }, (_, index) => `joint-${index}`),
    samples: [{ ...identity, frame_sha256: frameSha256 }],
    root_path_length_m: 0.25,
    root_planar_path_length_m: 0.25,
    joint_total_variation_rad: 0,
    end_effector_path_length_m: Object.fromEntries(HUMANOID_END_EFFECTORS.map(
      (name) => [name, 0]
    )),
    object_path_length_m: {},
    contact_transition_count: 0,
    controller_usage: {
      protocol: "humanoid-controller-usage-v1",
      complete_from_admission: true,
      observed_frame_count: 1,
      mode_frame_counts: {
        learned_policy: 1,
        reference_control: 0,
        hybrid_control: 0
      },
      implementation_frame_counts: { "workyard-policy-v1": 1 },
      transition_frame_count: 0
    },
    trajectory_sha256: advancePhysicalTrajectorySha256(null, frameSha256)
  };
}

function safetyEvidence(): unknown {
  const frame = { frame: 1, simulated_time_seconds: 0.02 };
  return {
    protocol: "humanoid-physical-safety-evidence-v1",
    frame_count: 1,
    first_frame: 1,
    last_frame: 1,
    first_simulated_time_seconds: 0.02,
    last_simulated_time_seconds: 0.02,
    minimum_signed_support_margin: { ...frame, signed_margin_m: 0.03 },
    maximum_foot_tangential_speed: {
      ...frame,
      foot: "left",
      tangential_speed_mps: 0.12
    },
    minimum_joint_limit_margin: {
      ...frame,
      joint: "left_hip_pitch_joint",
      margin_rad: 0.2
    },
    maximum_joint_velocity: {
      ...frame,
      joint: "left_hip_pitch_joint",
      absolute_velocity_rad_s: 2.5
    },
    maximum_actuator_effort_utilization: {
      ...frame,
      joint: "left_hip_pitch_joint",
      requested_newton_meters: 8,
      applied_newton_meters: 8,
      requested_utilization: 0.8,
      applied_utilization: 0.8,
      saturated: false
    },
    peak_contact_normal_force: {
      ...frame,
      contact: {
        contact_index: 0,
        normal_force_n: 80,
        position: { x: 0, y: 0, z: 0 },
        first_body: "left_ankle_roll_link",
        second_body: null,
        first_object: null,
        second_object: null
      }
    },
    peak_total_normal_force: { ...frame, total_normal_force_n: 600 },
    peak_total_normal_force_rise_rate: {
      ...frame,
      previous_frame: 0,
      rise_rate_nps: 40
    }
  };
}

function controllerRoutingEvidence(): unknown {
  const posterior = {
    outcomes: 9,
    successes: 9,
    failures: 0,
    posteriorMean: 10 / 11,
    lowerBound: 0.7,
    upperBound: 0.99,
    recentSuccessRate: 1,
    transitionAttempts: 1,
    transitionSuccesses: 1
  };
  return {
    execution: {
      callId: "benchmark-memory-bridge",
      route: "primary",
      assessment: {
        protocol: "humanoid-policy-admission-v1",
        implementation: "workyard-policy-v1",
        skillFamily: "navigation",
        admitted: true,
        reason: "memory_bridge_completed",
        coldStart: false,
        entryStateOodScore: 1,
        commandOodScore: 0,
        posterior,
        successfulEntryPrototype: Array.from({ length: 29 }, () => 0)
      },
      attribution: {
        primarySteps: 20,
        fallbackSteps: 0,
        upperBodyOverlaySteps: 0,
        memoryBridgeSteps: 5
      },
      memoryBridge: {
        protocol: "humanoid-policy-memory-bridge-v1",
        phase: "completed",
        trigger: "entry_state_ood",
        completedSteps: 5,
        maximumSteps: 200,
        stableSteps: 5,
        requiredStableSteps: 5,
        progress: 1,
        entryStateOodScore: 1,
        jointPrototypeRmsError: 0.05,
        maximumJointVelocity: 0.1
      }
    },
    capability_evidence: [{
      implementation: "workyard-policy-v1",
      skillFamily: "navigation",
      posterior
    }]
  };
}

function usage(requests: number, totalTokens: number): unknown {
  return {
    requests,
    reported_requests: requests,
    input_tokens: Math.floor(totalTokens * 0.75),
    output_tokens: Math.ceil(totalTokens * 0.25),
    total_tokens: totalTokens,
    cached_input_tokens: 0,
    reasoning_tokens: 0
  };
}
