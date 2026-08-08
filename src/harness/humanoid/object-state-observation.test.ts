import { describe, expect, it } from "vitest";
import { ScenarioSchema } from "../../domain/schema.js";
import { HumanoidWorld } from "../../world/humanoid/world.js";
import { HumanoidActionRuntime } from "./runtime.js";

const scenario = ScenarioSchema.parse({
  title: "Role object observation",
  seed: 43,
  bounds: { width: 8, depth: 8 },
  visibility_radius: 5,
  robot: { x: 0, z: 0.36, yaw: 0 },
  obstacles: [],
  objects: [
    {
      id: "stand",
      kind: "stand",
      color: "#76877c",
      position: { x: 0.2, y: 0.555, z: 0.8 },
      size: { x: 0.12, y: 0.01, z: 0.12 },
      portable: false
    },
    {
      id: "crate",
      kind: "workpiece",
      color: "#8b6b45",
      position: { x: 0.2, y: 0.67, z: 0.8 },
      size: { x: 0.03, y: 0.22, z: 0.03 },
      portable: true
    }
  ],
  zones: [],
  default_goal: {
    summary: "Observe the crate",
    predicates: [{
      type: "robot_at",
      target: { x: 0, y: 0, z: 0 },
      tolerance: 0.2
    }]
  }
});

describe("humanoid role object observation", () => {
  it("publishes authority and observation identity to the model harness", async () => {
    const world = await HumanoidWorld.create(scenario);
    const runtime = new HumanoidActionRuntime(world);
    try {
      const before = world.snapshot();
      const receipt = await runtime.invoke(
        "observe_humanoid",
        {},
        "observe-role-object",
        "humanoid-motion-reference"
      );
      expect(receipt.accepted).toBe(true);
      const detail = record(receipt.detail);
      const tokens = detail.object_tokens;
      if (!Array.isArray(tokens)) throw new Error("Expected object token array");
      expect(tokens).toHaveLength(1);
      expect(tokens[0]).toMatchObject({
        id: "crate",
        role: "manipulable",
        status: "visible",
        state: "active",
        authority: "mujoco_exact",
        exact: true,
        observable: true,
        observed_frame: detail.frame,
        observed_world_revision: receipt.worldAfterRevision,
        pose: {
          position: expect.any(Object),
          rotation: expect.any(Object)
        }
      });
      expect(detail.grasp).toMatchObject({
        contractSha256: expect.stringMatching(/^[a-f0-9]{64}$/),
        assessments: [
          expect.objectContaining({
            frame: detail.frame,
            object_id: "crate",
            hand: "left"
          }),
          expect.objectContaining({
            frame: detail.frame,
            object_id: "crate",
            hand: "right"
          })
        ]
      });
      const handSurfaces = record(detail.hand_surfaces);
      const leftHand = record(handSurfaces.left);
      expect(leftHand.contact_surfaces).toEqual(expect.arrayContaining([
        expect.objectContaining({
          hand_surface: "left_hand_palm_link",
          world_position: expect.any(Object),
          surface_from_wrist_world: expect.any(Object)
        })
      ]));
      expect(record(detail.manipulation_geometry).objects).toEqual([
        expect.objectContaining({
          object_id: "crate",
          hands: expect.objectContaining({
            left: expect.objectContaining({
              current_wrist_world: expect.any(Object),
              interaction_alignments: expect.arrayContaining([
                expect.objectContaining({
                  interaction_point_id: expect.any(String),
                  hand_surface: expect.any(String),
                  wrist_world_target: expect.any(Object),
                  delta_from_current_wrist_world: expect.any(Object),
                  distance_from_current_wrist_m: expect.any(Number),
                  ik_reference_reachable: expect.any(Boolean)
                })
              ])
            })
          })
        })
      ]);
      const geometries = record(detail.manipulation_geometry).objects;
      if (!Array.isArray(geometries)) {
        throw new Error("Expected manipulation geometry objects");
      }
      const hands = record(geometries[0]).hands;
      const leftAlignments = record(record(hands).left)
        .interaction_alignments;
      if (!Array.isArray(leftAlignments)) {
        throw new Error("Expected left-hand reachability alignments");
      }
      expect(leftAlignments.length).toBeGreaterThan(0);
      expect(new Set(leftAlignments.map((alignment) => (
        record(alignment).interaction_point_id
      ))).size).toBe(leftAlignments.length);
      expect(leftAlignments).toEqual(expect.arrayContaining([
        expect.objectContaining({
          interaction_point_id: expect.any(String),
          hand_surface: expect.stringMatching(/^left_/),
          wrist_world_target: expect.any(Object),
          wrist_world_orientation: expect.anything(),
          ik_reference_reachable: expect.any(Boolean),
          ik_residual_m: expect.any(Number)
        })
      ]));
      expect(record(detail.manipulation_geometry).objects).toEqual([
        expect.objectContaining({
          reachable_base_placements: expect.arrayContaining([
            expect.objectContaining({
              hand_surface: expect.any(String),
              root_world_target: expect.any(Object),
              root_translation_world: expect.any(Object),
              root_yaw_radians: expect.any(Number),
              navigation_validation_required: true
            })
          ])
        })
      ]);
      const after = world.snapshot();
      expect(after.frame).toBe(before.frame);
      expect(after.worldRevision).toBe(before.worldRevision);
      expect(after.robot.rootPosition).toEqual(before.robot.rootPosition);
      expect(after.robot.contacts).toEqual(before.robot.contacts);
    } finally {
      await world.dispose();
    }
  }, 30_000);
});

function record(value: unknown): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Expected object observation detail");
  }
  return value as Record<string, unknown>;
}
