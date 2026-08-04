import { describe, expect, it } from "vitest";
import { humanoidRunShouldFinish } from "./run-mode.js";

describe("humanoid run mode", () => {
  it("ends a finite mission only after the exact mission Goal completes", () => {
    expect(humanoidRunShouldFinish({
      mode: "mission",
      activeGoalCompleted: true,
      missionGoalCompleted: true
    })).toBe(true);
    expect(humanoidRunShouldFinish({
      mode: "mission",
      activeGoalCompleted: true,
      missionGoalCompleted: false
    })).toBe(false);
  });

  it("keeps continuous autonomy alive after a completed Goal", () => {
    expect(humanoidRunShouldFinish({
      mode: "continuous",
      activeGoalCompleted: true,
      missionGoalCompleted: true
    })).toBe(false);
  });
});
