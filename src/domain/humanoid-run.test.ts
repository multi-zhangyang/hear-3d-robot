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
  it("parses historical V4/V5 snapshots without inventing 43DoF hands", async () => {
    const raw = JSON.parse(await readFile(V4_FIXTURE, "utf8")) as unknown;
    const legacy = HumanoidRunCheckpointV4Schema.parse(raw);
    expect(legacy.world.robot).not.toHaveProperty("morphology");
    expect(legacy.world.robot).not.toHaveProperty("hands");

    const legacyV5 = v5Checkpoint(legacy);
    expect(legacyV5.world.robot).not.toHaveProperty("morphology");
    expect(legacyV5.world.robot).not.toHaveProperty("hands");
    expect(() => HumanoidRunCheckpointSchema.parse(raw)).toThrow(
      "Legacy 29DoF humanoid checkpoint requires physical migration"
    );
    expect(() => HumanoidRunCheckpointSchema.parse(legacyV5)).toThrow(
      "Legacy 29DoF humanoid checkpoint requires physical migration"
    );
  });

  it("rejects v5 progress with a changed goal, count, frame or world checkpoint", async () => {
    const legacy = HumanoidRunCheckpointV4Schema.parse(
      JSON.parse(await readFile(V4_FIXTURE, "utf8"))
    );
    const migrated = v5Checkpoint(legacy);

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

function v5Checkpoint(
  legacy: ReturnType<typeof HumanoidRunCheckpointV4Schema.parse>
) {
  const { version: _version, ...source } = legacy;
  return HumanoidRunCheckpointV5Schema.parse({
    version: 5,
    ...source,
    goal_progress: {
      version: 1,
      goal_sha256: goalSha256(legacy.goal),
      predicate_count: legacy.goal.predicates.length,
      last_world_frame: legacy.world.frame,
      last_world_revision: legacy.world.worldRevision,
      predicate_streaks: legacy.goal.predicates.map(() => 0)
    }
  });
}
