import { describe, expect, it } from "vitest";
import {
  humanoidMotionArtifactSha256,
  humanoidMotionArtifactSummary,
  serializeHumanoidReference,
  type HumanoidMotionArtifact
} from "./motion-artifact.js";
import { neutralHumanoidReference } from "./reference.js";

describe("humanoid motion artifact identity", () => {
  it("produces a stable content hash and changes it when a frame changes", () => {
    const artifact = motionArtifact();
    const repeated = structuredClone(artifact);
    const changed = structuredClone(artifact);
    changed.frames[0]!.reference.rootYawVelocity = 0.25;

    const hash = humanoidMotionArtifactSha256(artifact);
    expect(hash).toMatch(/^[a-f0-9]{64}$/);
    expect(humanoidMotionArtifactSha256(repeated)).toBe(hash);
    expect(humanoidMotionArtifactSha256(changed)).not.toBe(hash);
    expect(humanoidMotionArtifactSummary(artifact).sha256).toBe(hash);
  });
});

function motionArtifact(): HumanoidMotionArtifact {
  return {
    version: 1,
    protocol: "humanoid-motion-v1",
    generator: "test-generator",
    controlStepSeconds: 0.02,
    durationSeconds: 0.04,
    frames: [
      {
        atSeconds: 0.02,
        reference: serializeHumanoidReference(neutralHumanoidReference())
      },
      {
        atSeconds: 0.04,
        reference: serializeHumanoidReference(neutralHumanoidReference())
      }
    ]
  };
}
