import { describe, expect, it } from "vitest";
import type { Goal, SpatialMemoryRecord } from "../domain/schema.js";
import {
  goalMemoryContextRecords,
  goalRelevantSpatialMemory
} from "./goal-memory.js";
import { VOXEL_AFFORDANCE_CONTRACT_VERSION } from "../world/voxel-affordance.js";

const goal: Goal = {
  summary: "Place a voxel and move an object into a zone.",
  predicates: [
    {
      type: "voxel_at",
      coordinate: { column: 7, level: 2, row: 9 },
      material: "grass"
    },
    {
      type: "object_in_zone",
      object_id: "crate",
      zone_id: "build_zone",
      expected: true,
      tolerance: 0.2
    }
  ]
};

describe("goal-relevant spatial memory", () => {
  it("retrieves only goal entities and coordinates with newest evidence first", () => {
    const records = [
      memory("unrelated", 9, { entityId: "other" }),
      memory("zone", 3, { entityId: "build_zone" }),
      memory("voxel", 7, { coordinate: { column: 7, level: 2, row: 9 } }),
      memory("crate", 5, { entityId: "crate" })
    ];

    expect(goalRelevantSpatialMemory(goal, records)).toEqual([
      records[2],
      records[3],
      records[1]
    ]);
  });

  it("returns cloned, bounded records", () => {
    const record = memory("voxel", 7, {
      coordinate: { column: 7, level: 2, row: 9 }
    });
    const selected = goalRelevantSpatialMemory(goal, [record], 1);
    expect(selected).toEqual([record]);
    expect(selected[0]).not.toBe(record);
    expect(goalRelevantSpatialMemory(goal, [record], 0)).toEqual([]);
  });

  it("removes stale executable affordances from automatic model context", () => {
    const record = memory("voxel", 7, {
      coordinate: { column: 7, level: 2, row: 9 }
    });
    record.data = {
      reachable_standoff_poses: [{ target: { x: 1, y: 0, z: 2 } }],
      placement_interaction_points: [{ interaction_point: { x: 3, y: 4, z: 5 } }]
    };
    record.voxel_revision = 2;

    const projected = goalMemoryContextRecords([record], {
      worldRevision: 9,
      voxelRevision: 3
    });

    expect(projected).toEqual([
      expect.objectContaining({
        id: "voxel",
        stale: true,
        must_reobserve_before_actuation: true,
        omitted_dynamic_data: true
      })
    ]);
    expect(JSON.stringify(projected)).not.toContain("reachable_standoff_poses");
    expect(JSON.stringify(projected)).not.toContain("interaction_point");
  });

  it("keeps current-revision interaction evidence available to the coordinator", () => {
    const record = memory("voxel", 7, {
      coordinate: { column: 7, level: 2, row: 9 }
    });
    record.data = {
      affordance_contract_version: VOXEL_AFFORDANCE_CONTRACT_VERSION,
      material: null,
      placement_interaction_points: [{
        interaction_point: { x: 3, y: 4, z: 5 },
        gripper_distance: 0.8,
        recommended: true
      }]
    };
    record.voxel_revision = 2;

    const projected = goalMemoryContextRecords([record], {
      worldRevision: 7,
      voxelRevision: 2
    });

    expect(projected).toEqual([
      expect.objectContaining({
        id: "voxel",
        stale: false,
        must_reobserve_before_actuation: false,
        omitted_dynamic_data: false,
        current_data: record.data
      })
    ]);
  });

  it("invalidates same-revision voxel geometry from an older affordance contract", () => {
    const record = memory("voxel", 7, {
      coordinate: { column: 7, level: 2, row: 9 }
    });
    record.source_action = "inspect_voxel";
    record.data = {
      material: null,
      placement_interaction_points: [{
        interaction_point: { x: 3, y: 4, z: 5 },
        recommended: true
      }]
    };
    record.voxel_revision = 2;

    const projected = goalMemoryContextRecords([record], {
      worldRevision: 7,
      voxelRevision: 2
    });

    expect(projected).toEqual([
      expect.objectContaining({
        stale: true,
        stale_reason: "voxel_affordance_contract_changed",
        must_reobserve_before_actuation: true,
        omitted_dynamic_data: true
      })
    ]);
    expect(JSON.stringify(projected)).not.toContain("interaction_point");
  });
});

function memory(
  id: string,
  revision: number,
  key: {
    coordinate?: { column: number; level: number; row: number };
    entityId?: string;
  }
): SpatialMemoryRecord {
  return {
    id,
    world_id: "world:1",
    kind: key.coordinate ? "voxel" : "entity",
    label: id,
    summary: `${id} observation`,
    position: null,
    coordinate: key.coordinate ?? null,
    entity_id: key.entityId ?? null,
    data: { id },
    observed_frame: revision * 10,
    world_revision: revision,
    voxel_revision: key.coordinate ? revision : null,
    source_transaction_id: `tx:${id}`,
    source_agent_id: "observer",
    source_action: "inspect",
    recorded_at: new Date(Date.UTC(2026, 0, revision + 1)).toISOString()
  };
}
