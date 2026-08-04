import { describe, expect, it } from "vitest";
import {
  ActiveAutonomousCycleSchema,
  autonomousCycleRef,
  createActiveAutonomousCycle,
  embodiedMemoryIdForCycle,
  sameAutonomousCycle
} from "./autonomous-cycle.js";

describe("autonomous cycle identity", () => {
  it("creates one Goal-bound identity reused by actions and embodied memory", () => {
    const cycle = createActiveAutonomousCycle({
      cycleIndex: 7,
      goalEpochId: `goal-epoch:${"a".repeat(64)}`,
      worldFrame: 120,
      worldRevision: 80,
      cycleUuid: "00000000-0000-4000-8000-000000000007",
      startedAt: "2026-08-03T00:00:00.000Z"
    });
    const ref = autonomousCycleRef(cycle);

    expect(cycle).toMatchObject({
      cycle_id: "autonomous-cycle:00000000-0000-4000-8000-000000000007",
      cycle_index: 7,
      started_world_frame: 120,
      started_world_revision: 80
    });
    expect(embodiedMemoryIdForCycle(ref)).toBe(
      "embodied-memory:00000000-0000-4000-8000-000000000007"
    );
    expect(sameAutonomousCycle(ref, autonomousCycleRef(cycle))).toBe(true);
    expect(sameAutonomousCycle(ref, { ...ref, cycle_index: 8 })).toBe(false);
  });

  it("rejects malformed or incomplete cycle identities", () => {
    expect(() => createActiveAutonomousCycle({
      cycleIndex: 0,
      goalEpochId: `goal-epoch:${"a".repeat(64)}`,
      worldFrame: 0,
      worldRevision: 0
    })).toThrow();
    expect(ActiveAutonomousCycleSchema.safeParse({
      cycle_id: "autonomous-cycle:invented",
      cycle_index: 1,
      goal_epoch_id: `goal-epoch:${"a".repeat(64)}`,
      started_world_frame: 0,
      started_world_revision: 0,
      started_at: "2026-08-03T00:00:00.000Z"
    }).success).toBe(false);
  });
});
