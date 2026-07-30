import { describe, expect, it } from "vitest";
import { WorldCommandScheduler } from "./world-command-scheduler.js";

describe("WorldCommandScheduler", () => {
  it("coalesces independent body channels into one physical step", async () => {
    const batches: string[][] = [];
    const scheduler = new WorldCommandScheduler(async (ids) => {
      batches.push(ids);
    });
    scheduler.begin({
      id: "base",
      agentId: "agent_a",
      agentName: "Base worker",
      skill: "drive_base",
      channels: ["base"]
    }, 0);
    scheduler.begin({
      id: "head",
      agentId: "agent_b",
      agentName: "Head worker",
      skill: "set_head_target",
      channels: ["head"]
    }, 0);

    await Promise.all([
      scheduler.advance("base", "driving"),
      scheduler.advance("head", "tracking")
    ]);
    expect(batches).toEqual([["base", "head"]]);
    expect(scheduler.snapshot()).toMatchObject([
      { id: "base", phase: "driving" },
      { id: "head", phase: "tracking" }
    ]);
  });

  it("rejects conflicting channels and restores legacy command snapshots", () => {
    const scheduler = new WorldCommandScheduler(async () => undefined);
    scheduler.begin({
      id: "base_a",
      agentId: "agent_a",
      agentName: "A",
      skill: "drive_base",
      channels: ["base"]
    }, 0);
    expect(() => scheduler.begin({
      id: "base_b",
      agentId: "agent_b",
      agentName: "B",
      skill: "drive_base",
      channels: ["base"]
    }, 0)).toThrow("Body channels are already active");

    const command = scheduler.focused();
    if (!command) throw new Error("expected active command");
    scheduler.restore([command], null, 4);
    expect(scheduler.ids()).toEqual(["base_a"]);
    expect(scheduler.complete("base_a", "done", true, 12, 4)).toMatchObject({ id: "base_a" });
    expect(scheduler.last).toMatchObject({ result_code: "done", accepted: true });
  });
});
