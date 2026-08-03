import { describe, expect, it } from "vitest";
import type { HumanoidWorldSnapshot } from "../types";
import {
  HUMANOID_BODY_CHANNELS,
  movingHumanoidChannels
} from "./HumanoidMissionWorkspace";

describe("人形身体通道", () => {
  it("按服务端契约展示六个通道", () => {
    expect(HUMANOID_BODY_CHANNELS).toEqual([
      "locomotion",
      "left_leg",
      "right_leg",
      "torso",
      "left_arm",
      "right_arm"
    ]);
  });

  it("分别识别根运动、双腿、躯干和双臂的实时活动", () => {
    const frame = {
      robot: {
        links: {
          pelvis: {
            linearVelocity: { x: 0.06, y: 0, z: 0 },
            angularVelocity: { x: 0, y: 0, z: 0 }
          }
        },
        joints: {
          left_ankle_pitch_joint: { velocity: 0.1 },
          right_knee_joint: { velocity: -0.11 },
          waist_yaw_joint: { velocity: 0.12 },
          left_elbow_joint: { velocity: 0.13 },
          right_shoulder_pitch_joint: { velocity: -0.14 }
        }
      }
    } as unknown as HumanoidWorldSnapshot;

    expect(movingHumanoidChannels(frame)).toEqual(HUMANOID_BODY_CHANNELS);
  });
});
