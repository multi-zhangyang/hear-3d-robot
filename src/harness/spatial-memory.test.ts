import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { loadRuntimeCatalog } from "../config/load.js";
import type { AgentSpec, Goal } from "../domain/schema.js";
import { RunStore } from "../persistence/run-store.js";
import { RapierWorld } from "../world/rapier-world.js";
import { capabilityCatalog } from "./agents.js";
import { receiptEvidenceRequirement } from "./evidence-contract.js";
import { HierarchyProjection } from "./hierarchy-projection.js";
import { createCheckpoint, HarnessRuntimeContext } from "./runtime-context.js";

function output(value: string): Record<string, unknown> {
  return JSON.parse(value) as Record<string, unknown>;
}

describe("durable spatial memory", () => {
  it("indexes authoritative receipts and recalls them after process-style resume", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-memory-"));
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("open_navigation", 17);
    const goal: Goal = {
      summary: "Remember the robot's observed pose.",
      predicates: [{
        type: "robot_at",
        target: { x: scenario.robot.x, y: 0, z: scenario.robot.z },
        tolerance: 0.25
      }]
    };
    const store = await RunStore.create(runsDir, {
      mission: "Observe and remember the current robot pose",
      scenarioId: "open_navigation",
      scenario,
      goal
    });
    let world = await RapierWorld.create(scenario);
    let resumedWorld: RapierWorld | undefined;
    try {
      const available = capabilityCatalog();
      const hierarchy = HierarchyProjection.create("Observe and remember", available);
      const checkpoint = createCheckpoint({
        store,
        hierarchy,
        capabilityCatalog: available,
        world
      });
      await store.writeCheckpoint(checkpoint);
      const runtime = new HarnessRuntimeContext({ store, goal, world, hierarchy, checkpoint });
      await runtime.start();
      const leaf: AgentSpec = {
        name: "Spatial observer",
        objective: "Observe the robot pose and recover it from durable spatial memory",
        success_criteria: ["A source-backed robot pose is recalled"],
        evidence_requirements: [receiptEvidenceRequirement(
          0,
          "recall_spatial_memory",
          { kind: "world" }
        )],
        capabilities: ["read_proprioception", "recall_spatial_memory"],
        may_delegate: false,
        references: []
      };
      const active = await runtime.beginDelegation(null, leaf, "delegate_spatial_observer");
      const observed = output(await runtime.invokeTool(
        "read_proprioception",
        {},
        "sdk_observe_pose"
      ));
      expect(observed).toMatchObject({ accepted: true, code: "proprioception" });

      const firstRecall = output(await runtime.invokeTool(
        "recall_spatial_memory",
        { kind: "robot_pose", near: { x: 1, y: 0.38, z: 1 }, radius: 2 },
        "sdk_recall_pose"
      ));
      expect(firstRecall).toMatchObject({ accepted: true, code: "spatial_memory_recalled" });
      const firstDetail = firstRecall.detail as Record<string, unknown>;
      const records = firstDetail.records as Array<Record<string, unknown>>;
      expect(records).toHaveLength(1);
      expect(records[0]).toMatchObject({
        world_id: "open_navigation:17",
        kind: "robot_pose",
        source_transaction_id: `${active.node.id}:sdk_observe_pose`,
        source_agent_id: active.node.id,
        source_action: "read_proprioception",
        world_revision: 0,
        position: { x: 1, z: 1 }
      });

      const persisted = await store.readCheckpoint();
      expect(persisted.spatial_memory).toHaveLength(1);
      expect(persisted.spatial_memory[0]?.source_transaction_id).toBe(
        `${active.node.id}:sdk_observe_pose`
      );

      world.dispose();
      resumedWorld = await RapierWorld.create(scenario, persisted.world);
      const resumedHierarchy = new HierarchyProjection(
        persisted.nodes,
        persisted.root_id,
        persisted.active_agent_id
      );
      const resumed = new HarnessRuntimeContext({
        store,
        goal,
        world: resumedWorld,
        hierarchy: resumedHierarchy,
        checkpoint: persisted
      });
      const recalledAfterResume = output(await resumed.invokeTool(
        "recall_spatial_memory",
        { kind: "robot_pose", limit: 4 },
        "sdk_recall_after_resume"
      ));
      expect(recalledAfterResume).toMatchObject({
        accepted: true,
        code: "spatial_memory_recalled",
        detail: { matched: 1, current_world_revision: 0 }
      });
      expect((recalledAfterResume.detail as Record<string, unknown>).records).toEqual(records);
    } finally {
      resumedWorld?.dispose();
      // `world` was disposed before resume; Rapier tolerates no second free.
      if (!resumedWorld) world.dispose();
      await rm(runsDir, { recursive: true, force: true });
    }
  }, 20_000);
});
