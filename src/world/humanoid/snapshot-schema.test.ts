import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import {
  HumanoidWorldSnapshotSchema,
  LegacyHumanoidWorldSnapshotSchema
} from "./snapshot-schema.js";

describe("humanoid snapshot morphology versions", () => {
  it("parses recorded 29DoF V4 worlds only through the explicit legacy schema", async () => {
    const fixture = JSON.parse(await readFile(new URL(
      "../../../tests/fixtures/runs/20260802T204346Z_humanoid_courtyard_8071d876/checkpoint.json",
      import.meta.url
    ), "utf8")) as { world: unknown };

    expect(HumanoidWorldSnapshotSchema.safeParse(fixture.world).success).toBe(false);
    const legacy = LegacyHumanoidWorldSnapshotSchema.parse(fixture.world);
    expect("morphology" in legacy.robot).toBe(false);
    expect("hands" in legacy.robot).toBe(false);
    expect(legacy.robot.contacts.every((contact) => (
      contact.firstHandLink === null && contact.secondHandLink === null
    ))).toBe(true);
  });
});
