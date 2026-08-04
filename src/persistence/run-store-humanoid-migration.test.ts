import { cp, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { LegacyHumanoidRunCheckpointSchema } from "../domain/humanoid-run.js";
import { HUMANOID_JOINT_NAMES } from "../world/humanoid/model.js";
import { G1_HAND_JOINT_NAMES } from "../world/humanoid/morphology.js";
import type { MutationFence } from "./mutation-fence.js";
import { RunStore } from "./run-store.js";

const FIXTURE = resolve(
  process.cwd(),
  "tests/fixtures/runs/20260802T204346Z_humanoid_courtyard_8071d876"
);

describe("RunStore legacy humanoid recovery", () => {
  it("atomically normalizes a real V4 checkpoint before every read surface", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-g1-migration-"));
    const runDir = join(runsDir, basename(FIXTURE));
    await cp(FIXTURE, runDir, { recursive: true });
    const checkpointPath = join(runDir, "checkpoint.json");
    const raw = JSON.parse(await readFile(checkpointPath, "utf8")) as {
      world_checkpoint: {
        worldRevision: number;
        reference: unknown;
        motions: unknown[];
      };
    };
    raw.world_checkpoint.motions = [{
      plan: {
        id: "legacy-motion-must-retire",
        intent: "旧形态执行中的动作不得恢复",
        duration_seconds: 0.02,
        keyframes: [{ at_seconds: 0 }, { at_seconds: 0.02 }]
      },
      artifact: {
        version: 1,
        protocol: "humanoid-motion-v1",
        generator: "task_space_reference_v1",
        controlStepSeconds: 0.02,
        durationSeconds: 0.02,
        frames: [{
          atSeconds: 0.02,
          reference: raw.world_checkpoint.reference
        }]
      },
      createdRevision: raw.world_checkpoint.worldRevision
    }];
    await writeFile(checkpointPath, `${JSON.stringify(raw, null, 2)}\n`, "utf8");
    const legacy = LegacyHumanoidRunCheckpointSchema.parse(raw);
    const fence = new RecordingMutationFence();

    try {
      const store = await RunStore.open(runDir, { mutationFence: fence });
      const persisted = JSON.parse(
        await readFile(join(runDir, "checkpoint.json"), "utf8")
      ) as Record<string, unknown>;
      expect(persisted.version).toBe(6);
      expect(fence.calls).toBeGreaterThan(0);

      const checkpoint = await store.readCheckpoint();
      const humanoid = await store.readHumanoidCheckpoint();
      const details = await store.readDetailsSnapshot({
        actions: 2,
        provider: 2,
        framework: 2
      });
      expect(humanoid).toEqual(checkpoint);
      expect(details.checkpoint).toEqual(checkpoint);
      expect(checkpoint).toMatchObject({
        version: 6,
        world: {
          robot: {
            morphology: {
              id: "unitree_g1_43dof_with_hands",
              bodyJointCount: 29,
              handJointCount: 14,
              totalJointCount: 43
            }
          }
        }
      });

      const sourceState = legacy.world_checkpoint.simulation;
      const state = checkpoint.world_checkpoint.simulation;
      expect(state.time).toBe(sourceState.time);
      expect(state.positions).toHaveLength(sourceState.positions.length + 14);
      expect(state.velocities).toHaveLength(sourceState.velocities.length + 14);
      expect(state.controls).toHaveLength(43);
      expect(state.accelerationWarmstart)
        .toHaveLength(sourceState.accelerationWarmstart.length + 14);
      expect(state.requestedActuatorTorques).toBeUndefined();
      expect(state.positions.slice(0, 7)).toEqual(sourceState.positions.slice(0, 7));
      expect(state.velocities.slice(0, 6)).toEqual(sourceState.velocities.slice(0, 6));
      expect(checkpoint.world_checkpoint.motions).toEqual([]);
      expect(checkpoint.world_checkpoint.routes).toEqual([]);
      expect(checkpoint.world_checkpoint.navigation).toEqual({
        planId: null,
        status: "idle",
        target: null,
        waypoints: [],
        waypointIndex: null
      });
      expect(checkpoint.world.navigation).toEqual(
        checkpoint.world_checkpoint.navigation
      );
      expect(checkpoint.world_checkpoint.routeSequence).toBe(
        legacy.world_checkpoint.routeSequence + 1
      );
      expect(checkpoint.world_checkpoint.planRegistryEpoch).toBe(
        legacy.world_checkpoint.planRegistryEpoch + 1
      );
      expect(checkpoint.world_checkpoint.physicalSafety).toBeUndefined();
      expect(checkpoint.world.physicalSafety).toBeUndefined();

      for (const name of HUMANOID_JOINT_NAMES) {
        expect(checkpoint.world.robot.joints[name].position).toBeCloseTo(
          legacy.world.robot.joints[name].position,
          12
        );
        expect(checkpoint.world.robot.joints[name].velocity).toBeCloseTo(
          legacy.world.robot.joints[name].velocity,
          12
        );
      }
      for (const name of G1_HAND_JOINT_NAMES) {
        const hand = checkpoint.world.robot.hands.joints[name];
        expect(hand.position).toBe(hand.target);
        expect(hand.position).toBe(0);
        expect(hand.velocity).toBe(0);
      }
      for (const [id, object] of Object.entries(legacy.world.robot.objects)) {
        const migrated = checkpoint.world.robot.objects[id]!;
        expect(migrated.id).toBe(id);
        expect(vectorDistance(migrated.position, object.position)).toBeLessThan(1e-8);
        expect(vectorDistance(migrated.linearVelocity, object.linearVelocity))
          .toBeLessThan(1e-8);
        expect(vectorDistance(migrated.angularVelocity, object.angularVelocity))
          .toBeLessThan(1e-8);
        expect(Math.hypot(
          migrated.rotation.x - object.rotation.x,
          migrated.rotation.y - object.rotation.y,
          migrated.rotation.z - object.rotation.z,
          migrated.rotation.w - object.rotation.w
        )).toBeLessThan(1e-8);
      }
      const reopened = await RunStore.open(runDir);
      expect(await reopened.readCheckpoint()).toEqual(checkpoint);
    } finally {
      await rm(runsDir, { recursive: true, force: true });
    }
  }, 30_000);

  it("rebuilds a pre-grasp V6 registry from its restored physical frame", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-g1-v6-grasp-migration-"));
    const runDir = join(runsDir, basename(FIXTURE));
    await cp(FIXTURE, runDir, { recursive: true });
    try {
      const first = await RunStore.open(runDir);
      const current = await first.readCheckpoint();
      if (current.runtime !== "humanoid_g1") {
        throw new Error("Expected humanoid checkpoint fixture");
      }
      const preGrasp = structuredClone(current);
      delete (preGrasp.world as Partial<typeof preGrasp.world>).grasp;
      delete (preGrasp.world_checkpoint as Partial<
        typeof preGrasp.world_checkpoint
      >).graspRegistry;
      await writeFile(
        join(runDir, "checkpoint.json"),
        `${JSON.stringify(preGrasp, null, 2)}\n`,
        "utf8"
      );

      const reopened = await RunStore.open(runDir);
      const migrated = await reopened.readHumanoidCheckpoint();
      expect(migrated.version).toBe(6);
      expect(migrated.goal_dag).toEqual(current.goal_dag);
      expect(migrated.action_commit_outbox).toEqual(current.action_commit_outbox);
      expect(migrated.action_execution_ledger).toEqual(
        current.action_execution_ledger
      );
      expect(migrated.committed_actions).toEqual(current.committed_actions);
      expect(migrated.world_checkpoint.motions).toEqual(
        current.world_checkpoint.motions
      );
      expect(migrated.world_checkpoint.routes).toEqual(
        current.world_checkpoint.routes
      );
      expect(migrated.world_checkpoint.graspRegistry.last_frame).toBe(
        migrated.world.frame
      );
      expect(migrated.world.grasp.contractSha256).toBe(
        migrated.world_checkpoint.graspRegistry.contract_sha256
      );
      expect(migrated.world.grasp.assessments).toEqual(
        migrated.world_checkpoint.graspRegistry.last_assessments
      );
      expect(migrated.world.grasp.assessments.every((assessment) => (
        assessment.frame === migrated.world.frame
      ))).toBe(true);
    } finally {
      await rm(runsDir, { recursive: true, force: true });
    }
  }, 30_000);

  it("refuses to discard V6 grasp evidence when its tracker is missing", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-g1-v6-grasp-split-"));
    const runDir = join(runsDir, basename(FIXTURE));
    await cp(FIXTURE, runDir, { recursive: true });
    try {
      const first = await RunStore.open(runDir);
      const current = await first.readCheckpoint();
      if (current.runtime !== "humanoid_g1") {
        throw new Error("Expected humanoid checkpoint fixture");
      }
      const split = structuredClone(current);
      delete (split.world_checkpoint as Partial<
        typeof split.world_checkpoint
      >).graspRegistry;
      await writeFile(
        join(runDir, "checkpoint.json"),
        `${JSON.stringify(split, null, 2)}\n`,
        "utf8"
      );

      await expect(RunStore.open(runDir)).rejects.toThrow(
        "display evidence without its authoritative registry"
      );
    } finally {
      await rm(runsDir, { recursive: true, force: true });
    }
  }, 30_000);
});

class RecordingMutationFence implements MutationFence {
  calls = 0;

  async runMutation<T>(operation: () => Promise<T>): Promise<T> {
    this.calls += 1;
    return operation();
  }
}

function vectorDistance(
  left: { x: number; y: number; z: number },
  right: { x: number; y: number; z: number }
): number {
  return Math.hypot(left.x - right.x, left.y - right.y, left.z - right.z);
}
