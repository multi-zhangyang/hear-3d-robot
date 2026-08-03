import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { goalSha256 } from "./goal-identity.js";
import {
  HumanoidRunCheckpointSchema,
  HumanoidRunCheckpointV4Schema,
  HumanoidRunCheckpointV5Schema
} from "./humanoid-run.js";

const V4_FIXTURE = resolve(
  process.cwd(),
  "tests/fixtures/runs/20260802T204346Z_humanoid_courtyard_8071d876/checkpoint.json"
);

describe("humanoid checkpoint migration", () => {
  it("migrates v4 checkpoints to aligned v5 goal progress without inventing stability", async () => {
    const raw = JSON.parse(await readFile(V4_FIXTURE, "utf8")) as unknown;
    const legacy = HumanoidRunCheckpointV4Schema.parse(raw);
    const migrated = HumanoidRunCheckpointSchema.parse(raw);

    expect(migrated).toMatchObject({
      version: 5,
      runtime: "humanoid_g1",
      goal_progress: {
        version: 1,
        goal_sha256: goalSha256(legacy.goal),
        predicate_count: legacy.goal.predicates.length,
        last_world_frame: legacy.world.frame,
        last_world_revision: legacy.world.worldRevision
      }
    });
    expect(migrated.goal_progress.predicate_streaks).toEqual(
      legacy.goal.predicates.map(() => 0)
    );
    expect(HumanoidRunCheckpointV5Schema.parse(migrated)).toEqual(migrated);
  });

  it("rejects v5 progress with a changed goal, count, frame or world checkpoint", async () => {
    const raw = JSON.parse(await readFile(V4_FIXTURE, "utf8")) as unknown;
    const migrated = HumanoidRunCheckpointSchema.parse(raw);

    const wrongHash = structuredClone(migrated);
    wrongHash.goal_progress.goal_sha256 = "0".repeat(64);
    expect(HumanoidRunCheckpointV5Schema.safeParse(wrongHash).success).toBe(false);

    const wrongCount = structuredClone(migrated);
    wrongCount.goal_progress.predicate_count += 1;
    wrongCount.goal_progress.predicate_streaks.push(0);
    expect(HumanoidRunCheckpointV5Schema.safeParse(wrongCount).success).toBe(false);

    const staleProgress = structuredClone(migrated);
    staleProgress.goal_progress.last_world_frame -= 1;
    expect(HumanoidRunCheckpointV5Schema.safeParse(staleProgress).success).toBe(false);

    const inventedInstantaneousProgress = structuredClone(migrated);
    inventedInstantaneousProgress.goal_progress.predicate_streaks[0] = 1;
    expect(HumanoidRunCheckpointV5Schema.safeParse(
      inventedInstantaneousProgress
    ).success).toBe(false);

    const splitWorld = structuredClone(migrated);
    splitWorld.world_checkpoint.worldRevision -= 1;
    expect(HumanoidRunCheckpointV5Schema.safeParse(splitWorld).success).toBe(false);
  });

  it("fails closed instead of migrating a misaligned v4 physical state", async () => {
    const raw = HumanoidRunCheckpointV4Schema.parse(
      JSON.parse(await readFile(V4_FIXTURE, "utf8"))
    );
    raw.world.frame += 1;
    expect(() => HumanoidRunCheckpointSchema.parse(raw)).toThrow(
      "Humanoid world snapshot and physical checkpoint are not aligned"
    );
  });
});
