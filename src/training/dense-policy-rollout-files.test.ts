import { appendFile, mkdir, mkdtemp, readFile, rm } from "node:fs/promises";
import { resolve } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { modelPayloadSha256 } from "../domain/model-call-authority.js";
import {
  ActiveHumanoidSkillBindingSchema,
  humanoidEmbodiedSkillIdentity
} from "../harness/humanoid/skill-binding.js";
import {
  HumanoidEmbodiedSkillCallSchema,
  type HumanoidEmbodiedSkillIdentity
} from "../world/humanoid/embodied-skill-call.js";
import { neutralHumanoidReference } from "../world/humanoid/reference.js";
import { HumanoidSimulation } from "../world/humanoid/simulation.js";
import {
  densePolicyRolloutPath,
  DensePolicyRolloutWriter,
  loadDensePolicyRolloutReference
} from "./dense-policy-rollout-files.js";

const roots: string[] = [];

afterEach(async () => {
  await Promise.all(roots.splice(0).map((path) => (
    rm(path, { recursive: true, force: true })
  )));
});

describe("dense policy rollout files", () => {
  it("syncs a semantic Skill hash chain and resumes after a torn tail", async () => {
    const parent = resolve("artifacts", "test-tmp");
    await mkdir(parent, { recursive: true });
    const root = await mkdtemp(resolve(parent, "dense-policy-"));
    roots.push(root);
    const runId = "20260811T120000Z_workyard_12345678";
    const identity = semanticSkillIdentity();
    const simulation = await HumanoidSimulation.create();
    try {
      const writer = new DensePolicyRolloutWriter({ rootDir: root, runId });
      await simulation.step(neutralHumanoidReference(), {
        taskCommand: taskCommand(identity, 0),
        policyFrameSink: writer.recordFrame
      });
      await simulation.step(neutralHumanoidReference(), {
        taskCommand: taskCommand(identity, 1),
        policyFrameSink: writer.recordFrame
      });
      await writer.flush();
      const path = densePolicyRolloutPath(root, runId, identity.callId);
      await appendFile(path, "{\"protocol\":\"torn", "utf8");

      const resumed = new DensePolicyRolloutWriter({ rootDir: root, runId });
      await simulation.step(neutralHumanoidReference(), {
        taskCommand: taskCommand(identity, 2),
        policyFrameSink: resumed.recordFrame
      });
      await resumed.flush();

      const reference = await loadDensePolicyRolloutReference({
        rootDir: root,
        runId,
        callId: identity.callId,
        execution: {
          frameCount: 3,
          worldBeforeRevision: 12,
          worldAfterRevision: 15
        }
      });
      expect(reference).toMatchObject({
        available: true,
        frame_count: 3,
        first_call_step_index: 0,
        last_call_step_index: 2,
        complete_from_window_start: true,
        complete_through_execution_end: true,
        missing_call_step_count: 0,
        missing_world_frame_count: 0,
        teacher_frame_count: 0,
        paired_teacher_frame_count: 0
      });
      const records = (await readFile(path, "utf8")).trim().split("\n")
        .map((line) => JSON.parse(line) as Record<string, unknown>);
      expect(records.map(({ local_frame_index }) => local_frame_index))
        .toEqual([0, 1, 2]);
      expect(records[2]!.previous_frame_sha256)
        .toBe(records[1]!.frame_sha256);
    } finally {
      await simulation.dispose();
    }
  });
});

function semanticSkillIdentity(): HumanoidEmbodiedSkillIdentity {
  const invocation = {
    skill: "stabilize" as const,
    minimum_support_margin_m: 0.03
  };
  const binding = ActiveHumanoidSkillBindingSchema.parse({
    protocol: "humanoid-active-skill-v1",
    transaction_id: "binding-dense-policy",
    agent_id: "humanoid-motion-reference",
    skill_plan_transaction_id: "skill-plan-dense-policy",
    skill_node_id: "stabilize-dense-policy",
    invocation,
    invocation_sha256: modelPayloadSha256(invocation),
    phase: "recover_support",
    phase_authority: "whole_body",
    planning_action: "plan_humanoid_skill",
    observed_frame: 12,
    observed_world_revision: 12,
    skill_catalog_sha256: "a".repeat(64),
    target_position: null,
    target_solid: null,
    target_articulation: null,
    eligible_interaction_points: [],
    eligible_interaction_point_ids: [],
    learned_policy_required_capabilities: ["balance"],
    learned_policy_missing_capabilities: [],
    control_mode: "learned_policy"
  });
  return humanoidEmbodiedSkillIdentity(binding);
}

function taskCommand(identity: HumanoidEmbodiedSkillIdentity, step: number) {
  const reference = neutralHumanoidReference();
  return HumanoidEmbodiedSkillCallSchema.parse({
    protocol: "humanoid-embodied-skill-call-v2",
    identity,
    authority: {
      source: "agent_harness",
      worldFrame: 12 + step,
      worldRevision: 12 + step
    },
    window: {
      mode: "autonomous_closed_loop",
      replanPolicy: "event_driven",
      controlStepSeconds: 0.02,
      maximumSteps: 3,
      stepIndex: step,
      remainingSteps: 3 - step
    },
    requestedCapabilities: ["balance"],
    command: {
      baseTwist: {
        forwardMetersPerSecond: 0,
        lateralMetersPerSecond: 0,
        yawRadiansPerSecond: 0
      },
      rootHeightMeters: reference.rootHeight,
      leftWristPositionPelvis: null,
      rightWristPositionPelvis: null,
      endEffectors: [],
      grasps: []
    },
    contract: null,
    safety: {
      authorizedContacts: [],
      stopOnFall: true,
      stopOnUnauthorizedContact: true,
      stopOnContractViolation: true
    },
    feedback: {
      mode: "event_driven",
      progressDelta: 0.1,
      events: ["progress", "succeeded", "failed"]
    }
  });
}
