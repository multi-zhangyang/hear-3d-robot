import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  createMjlabG1VelocityController
} from "./mjlab-g1-velocity-controller.js";
import {
  neutralHumanoidReference,
  targetReference
} from "../world/humanoid/reference.js";
import { HumanoidSimulation } from "../world/humanoid/simulation.js";
import { HUMANOID_JOINT_NAMES } from "../world/humanoid/model.js";
import { HumanoidWorldSnapshotSchema } from "../world/humanoid/snapshot-schema.js";
import type {
  HumanoidControllerModuleAsset,
  HumanoidControllerModuleContext
} from "../world/humanoid/controller-module.js";
import type {
  HumanoidPolicyState
} from "../world/humanoid/whole-body-controller.js";

const POLICY_PATH = fileURLToPath(new URL(
  "../../assets/humanoid/controllers/mjlab-g1-velocity/g1_velocity.onnx",
  import.meta.url
));
const REPORT_PATH = fileURLToPath(new URL(
  "../../assets/humanoid/controllers/mjlab-g1-velocity/training-report.json",
  import.meta.url
));

describe("mjlab G1 velocity controller", () => {
  it("runs the shipped trained policy and restores its action history", async () => {
    const context = await controllerContext();
    const first = await createMjlabG1VelocityController(context);
    const second = await createMjlabG1VelocityController(context);
    try {
      expect(first.descriptor.learnedPolicy).toMatchObject({
        runtime: "onnxruntime-web/wasm",
        observationSpace: { size: 99 },
        actionSpace: { size: 29 },
        capabilities: ["balance", "locomotion"]
      });
      const state = policyState();
      const reference = neutralHumanoidReference();
      first.reset(state, reference);
      const initial = await first.infer(state, reference);
      expect(initial.positions).toHaveLength(HUMANOID_JOINT_NAMES.length);
      expect([
        ...initial.positions,
        ...initial.stiffness,
        ...initial.damping
      ].every(Number.isFinite)).toBe(true);

      const tracked = targetReference(reference, {
        joints: {
          left_elbow_joint: 0.8,
          right_shoulder_pitch_joint: -0.4
        }
      });
      second.reset(state, tracked);
      const trackedCommand = await second.infer(state, tracked);
      expect([...trackedCommand.positions]).toEqual([...initial.positions]);
      expect([...trackedCommand.stiffness]).toEqual([...initial.stiffness]);
      expect([...trackedCommand.damping]).toEqual([...initial.damping]);

      const checkpoint = first.captureState();
      second.restoreState(checkpoint);
      const [continued, restored] = await Promise.all([
        first.infer(state, reference),
        second.infer(state, reference)
      ]);
      expect([...restored.positions]).toEqual([...continued.positions]);
      expect(second.captureState()).toEqual(first.captureState());
    } finally {
      await Promise.all([first.dispose(), second.dispose()]);
    }
  });

  it("rejects a policy whose bytes do not match the training report", async () => {
    const context = await controllerContext();
    const policy = context.assets.find(({ id }) => id === "policy")!;
    const tampered = Uint8Array.from(policy.bytes);
    tampered[tampered.length - 1] ^= 1;
    const assets = context.assets.map((asset) => asset.id === "policy"
      ? {
          ...asset,
          bytes: tampered,
          sha256: sha256(tampered)
        }
      : asset);
    await expect(createMjlabG1VelocityController({
      ...context,
      assets
    })).rejects.toThrow(/does not match its training report/);
  });

  it("drives stable physical locomotion and restores its policy state", async () => {
    const context = await controllerContext();
    const controllerFactory = () => createMjlabG1VelocityController(context);
    const first = await HumanoidSimulation.create({ controllerFactory });
    const second = await HumanoidSimulation.create({ controllerFactory });
    try {
      const start = first.snapshot();
      expect(start.controller).toMatchObject({
        implementation: "mjlab_g1_velocity_onnx",
        learnedPolicy: { capabilities: ["balance", "locomotion"] },
        capabilityRouting: {
          strategy: "declared_capabilities",
          fallback: {
            mode: "reference_control",
            implementation: "yahmp_onnx"
          }
        }
      });
      expect(HumanoidWorldSnapshotSchema.safeParse({
        frame: 0,
        worldRevision: 0,
        robot: start,
        grasp: {
          contractSha256:
            "fc1e2d113bb5e5f5f8a75f0faa3efc8bd97ecc18eb41463da09d26bb52cfc193",
          assessments: []
        },
        navigation: {
          planId: null,
          status: "idle",
          target: null,
          waypoints: [],
          waypointIndex: null
        }
      }).success).toBe(true);
      const reference = targetReference(neutralHumanoidReference(), {
        rootVelocity: [0.5, 0]
      });
      let snapshot = start;
      for (let step = 0; step < 150; step += 1) {
        snapshot = await first.step(reference);
      }
      expect(snapshot.fallen).toBe(false);
      expect(snapshot.balance.upright).toBeGreaterThan(0.95);
      expect(Math.hypot(
        snapshot.rootPosition.x - start.rootPosition.x,
        snapshot.rootPosition.z - start.rootPosition.z
      )).toBeGreaterThan(0.5);
      expect(first.captureState().controller.payload).toMatchObject({
        protocol: "humanoid-controller-capability-routing-state-v1",
        active: "primary"
      });

      const taskReference = targetReference(reference, {
        joints: { right_shoulder_pitch_joint: 0.35 }
      });
      snapshot = await first.step(taskReference);
      expect(snapshot.fallen).toBe(false);
      expect(first.captureState().controller.payload).toMatchObject({
        protocol: "humanoid-controller-capability-routing-state-v1",
        active: "fallback"
      });

      const checkpoint = first.captureState();
      second.restoreState(checkpoint);
      expect(second.captureState().controller).toEqual(checkpoint.controller);
      expect(second.snapshot().rootPosition).toEqual(snapshot.rootPosition);
      expect((await second.step(reference)).fallen).toBe(false);
    } finally {
      await Promise.all([first.dispose(), second.dispose()]);
    }
  }, 30_000);
});

async function controllerContext(): Promise<HumanoidControllerModuleContext> {
  const assets = await Promise.all([
    asset("policy", POLICY_PATH),
    asset("training_report", REPORT_PATH)
  ]);
  return {
    protocol: "hear-humanoid-controller-module-v1",
    sourceSha256: "a".repeat(64),
    assets
  };
}

async function asset(
  id: string,
  path: string
): Promise<HumanoidControllerModuleAsset> {
  const bytes = await readFile(path);
  return { id, bytes, sha256: sha256(bytes) };
}

function sha256(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function policyState(): HumanoidPolicyState {
  return {
    jointPositions: Float64Array.from([
      -0.312, 0, 0, 0.669, -0.363, 0,
      -0.312, 0, 0, 0.669, -0.363, 0,
      0, 0, 0,
      0.2, 0.2, 0, 0.6, 0, 0, 0,
      0.2, -0.2, 0, 0.6, 0, 0, 0
    ]),
    jointVelocities: new Float64Array(HUMANOID_JOINT_NAMES.length),
    rootQuaternion: [1, 0, 0, 0],
    rootAngularVelocity: [0, 0, 0],
    environment: {
      protocol: "humanoid-policy-environment-v1",
      authority: "mujoco_state",
      rootVelocityFrame: "pelvis_imu",
      rootLinearVelocity: [0, 0, 0],
      rootAngularVelocity: [0, 0, 0],
      endEffectors: {},
      hands: {},
      contacts: [],
      objects: []
    }
  };
}
