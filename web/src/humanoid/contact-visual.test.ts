import { describe, expect, it } from "vitest";
import type { HumanoidWorldSnapshot } from "../types";
import { humanoidContactVisuals } from "./contact-visual";

describe("humanoid contact visuals", () => {
  it("prioritises true hand and non-foot contacts without inferring a grasp", () => {
    const snapshot = contactSnapshot();
    snapshot.robot.contacts = [
      contact("left_ankle_roll_link", null, 180, { x: 1, y: 0, z: 0 }),
      {
        ...contact("left_hand_palm_link", "crate", 8, { x: 2, y: 1, z: 0 }),
        firstHandLink: "left_hand_palm_link"
      },
      contact("torso_link", "wall", 12, { x: 3, y: 1, z: 0 }),
      {
        ...contact("left_wrist_yaw_link", null, 9, { x: 4, y: 1, z: 0 }),
        firstSolid: "block-a"
      }
    ];

    expect(humanoidContactVisuals(snapshot, 3)).toEqual([
      expect.objectContaining({ kind: "solid", normalForce: 9, position: { x: 4, y: 1, z: 0 } }),
      expect.objectContaining({ kind: "hand", normalForce: 8, position: { x: 2, y: 1, z: 0 } }),
      expect.objectContaining({ kind: "body", normalForce: 12, position: { x: 3, y: 1, z: 0 } })
    ]);
  });

  it("uses measured legacy foot points only when raw contacts are unavailable", () => {
    const snapshot = contactSnapshot();
    snapshot.robot.feet.left.points = [{ x: 0.1, y: 0, z: 0.2 }];

    expect(humanoidContactVisuals(snapshot)).toEqual([{
      kind: "foot",
      position: { x: 0.1, y: 0, z: 0.2 },
      normalForce: 0,
      scale: 0.72
    }]);
  });

  it("rejects an invalid visual budget", () => {
    expect(() => humanoidContactVisuals(contactSnapshot(), -1)).toThrow(/limit/i);
  });
});

function contact(
  firstBody: string,
  secondBody: string | null,
  normalForce: number,
  position: { x: number; y: number; z: number }
): HumanoidWorldSnapshot["robot"]["contacts"][number] {
  return {
    position,
    normal: { x: 0, y: 1, z: 0 },
    normalForce,
    firstBody,
    secondBody,
    firstObject: null,
    secondObject: null
  };
}

function contactSnapshot(): HumanoidWorldSnapshot {
  return {
    robot: {
      contacts: [],
      feet: {
        left: { points: [] },
        right: { points: [] }
      }
    }
  } as unknown as HumanoidWorldSnapshot;
}
