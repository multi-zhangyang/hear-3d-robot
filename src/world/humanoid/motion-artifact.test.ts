import { describe, expect, it } from "vitest";
import { createHash } from "node:crypto";
import {
  HumanoidReferenceStateSchema,
  hydrateHumanoidReference,
  humanoidMotionArtifactSha256,
  humanoidMotionArtifactSummary,
  serializeHumanoidReference,
  type HumanoidMotionArtifact
} from "./motion-artifact.js";
import { HUMANOID_JOINT_NAMES } from "./model.js";
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

  it("serializes tracking weights and restores them exactly", () => {
    const reference = neutralHumanoidReference();
    reference.jointTrackingWeights[3] = 0.4;
    reference.jointTrackingWeights[15] = 1;

    const serialized = serializeHumanoidReference(reference);
    const hydrated = hydrateHumanoidReference(serialized);

    expect(serialized.jointTrackingWeights).toEqual([...reference.jointTrackingWeights]);
    expect([...hydrated.jointTrackingWeights]).toEqual([...reference.jointTrackingWeights]);
  });

  it("explicitly migrates legacy references to autonomous residual authority", () => {
    const legacy = serializeHumanoidReference(neutralHumanoidReference());
    delete legacy.jointTrackingWeights;
    const encoded = JSON.stringify(legacy);

    const parsed = HumanoidReferenceStateSchema.parse(legacy);
    const hydrated = hydrateHumanoidReference(parsed);

    expect(JSON.stringify(parsed)).toBe(encoded);
    expect([...hydrated.jointTrackingWeights]).toEqual(
      Array.from({ length: HUMANOID_JOINT_NAMES.length }, () => 0)
    );
  });

  it("preserves legacy artifact hashes while rejecting invalid new weights", () => {
    const legacy = motionArtifact();
    for (const frame of legacy.frames) delete frame.reference.jointTrackingWeights;
    const expectedLegacyHash = createHash("sha256")
      .update(JSON.stringify(legacy))
      .digest("hex");
    expect(humanoidMotionArtifactSha256(legacy)).toBe(expectedLegacyHash);

    const invalid = serializeHumanoidReference(neutralHumanoidReference());
    invalid.jointTrackingWeights![0] = 1.01;
    expect(() => HumanoidReferenceStateSchema.parse(invalid)).toThrow();
    invalid.jointTrackingWeights = new Array(HUMANOID_JOINT_NAMES.length - 1).fill(0);
    expect(() => HumanoidReferenceStateSchema.parse(invalid)).toThrow();
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
