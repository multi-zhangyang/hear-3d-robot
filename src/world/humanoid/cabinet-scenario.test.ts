import { describe, expect, it } from "vitest";
import { loadRuntimeCatalog } from "../../config/load.js";
import { checkHumanoidGoal } from "../../runtime/humanoid-checker.js";
import { humanoidEnvironment } from "./environment.js";
import { neutralHumanoidReference } from "./reference.js";
import { HumanoidSimulation } from "./simulation.js";
import type { HumanoidWorldSnapshot } from "./world.js";

describe("authored cabinet task", () => {
  it("loads the integrated articulated cabinet and workcell", async () => {
    const catalog = await loadRuntimeCatalog();
    const scenario = catalog.materialize("humanoid_cabinet", 9);
    const simulation = await HumanoidSimulation.create(humanoidEnvironment(scenario));
    try {
      const robot = simulation.snapshot();
      expect(Object.keys(robot.objects).sort()).toEqual([
        "cabinet-door",
        "cabinet-frame",
        "cabinet-workpiece",
        "destination-bin",
        "material-cart",
        "side-workbench",
        "tool-slot",
        "workbench-button",
        "workbench-drawer",
        "workbench-switch",
        "workbench-tool",
        "workbench-valve"
      ]);
      expect(robot.objects["cabinet-door"]?.articulation).toMatchObject({
        type: "hinge",
        position: 0
      });
      expect(robot.objects["cabinet-workpiece"]?.articulation).toBeUndefined();
      expect(robot.objects["workbench-drawer"]?.articulation).toMatchObject({
        type: "slide",
        position: 0
      });
      expect(robot.objects["workbench-button"]?.articulation).toMatchObject({
        type: "slide",
        position: 0
      });
      expect(robot.objects["workbench-valve"]?.articulation).toMatchObject({
        type: "hinge",
        position: 0
      });

      const checker = checkHumanoidGoal(
        scenario.default_goal,
        scenario,
        {
          frame: 0,
          worldRevision: 0,
          robot,
          grasp: {
            contractSha256: "a".repeat(64),
            assessments: []
          },
          navigation: {
            planId: null,
            status: "idle",
            target: null,
            waypoints: [],
            waypointIndex: null
          }
        } as unknown as HumanoidWorldSnapshot
      );
      expect(checker.success).toBe(false);
      expect(checker.checks).toEqual(expect.arrayContaining([
        expect.objectContaining({ name: "1:object_inside", passed: false }),
        expect.objectContaining({ name: "2:articulation_state", passed: true })
      ]));

      const neutral = neutralHumanoidReference();
      let settled = robot;
      for (let index = 0; index < 150; index += 1) {
        settled = await simulation.step(neutral);
      }
      expect(settled.objects["cabinet-door"]?.articulation).toMatchObject({
        position: 0,
        velocity: 0
      });
      expect(settled.contacts.filter(({ firstObject, secondObject }) => (
        firstObject === "cabinet-door" || secondObject === "cabinet-door"
      ))).toEqual([]);
    } finally {
      await simulation.dispose();
    }
  }, 30_000);
});
