import { describe, expect, it } from "vitest";
import { createHash } from "node:crypto";
import {
  HumanoidMotionArtifactSchema,
  HumanoidReferenceStateSchema,
  hydrateHumanoidReference,
  humanoidMotionArtifactSha256,
  humanoidMotionArtifactSummary,
  serializeHumanoidReference,
  type HumanoidMotionArtifact
} from "./motion-artifact.js";
import { createG1HandArtifactCommand } from "./hand-coordination.js";
import { HUMANOID_JOINT_NAMES } from "./model.js";
import { neutralHumanoidReference } from "./reference.js";
import { HUMANOID_TASK_SPACE_SERVO_DESCRIPTOR } from "./task-space-servo.js";

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

  it("binds every version 2 frame to its complete resolved hand command", () => {
    const coordination = {
      left: {
        thumb_opposition: 0.2,
        thumb_curl: 0.4,
        index_curl: 0.6,
        middle_curl: 0.8
      },
      right: {
        thumb_opposition: 0.1,
        thumb_curl: 0.3,
        index_curl: 0.5,
        middle_curl: 0.7
      }
    };
    const artifact = HumanoidMotionArtifactSchema.parse({
      version: 2,
      protocol: "humanoid-motion-v2",
      generator: "test-generator",
      controlStepSeconds: 0.02,
      durationSeconds: 0.02,
      frames: [{
        atSeconds: 0.02,
        reference: serializeHumanoidReference(neutralHumanoidReference()),
        handCommand: createG1HandArtifactCommand(coordination)
      }]
    });
    const changed = structuredClone(artifact);
    if (changed.version !== 2) throw new Error("Expected a version 2 artifact");
    changed.frames[0]!.handCommand = createG1HandArtifactCommand({
      ...coordination,
      left: { ...coordination.left, index_curl: 0.7 }
    });
    expect(humanoidMotionArtifactSha256(changed)).not.toBe(
      humanoidMotionArtifactSha256(artifact)
    );
    const reordered = structuredClone(artifact);
    if (reordered.version !== 2) throw new Error("Expected version 2");
    reordered.frames[0]!.handCommand.jointTargets = Object.fromEntries(
      Object.entries(reordered.frames[0]!.handCommand.jointTargets).reverse()
    ) as typeof reordered.frames[0]["handCommand"]["jointTargets"];
    expect(humanoidMotionArtifactSha256(reordered)).toBe(
      humanoidMotionArtifactSha256(artifact)
    );

    const coordinationTamper = structuredClone(artifact);
    if (coordinationTamper.version !== 2) throw new Error("Expected version 2");
    coordinationTamper.frames[0]!.handCommand.coordination.left.index_curl = 0.1;
    expect(() => HumanoidMotionArtifactSchema.parse(coordinationTamper))
      .toThrow(/does not match coordination input/);

    const targetTamper = structuredClone(artifact);
    if (targetTamper.version !== 2) throw new Error("Expected version 2");
    targetTamper.frames[0]!.handCommand.jointTargets.left_hand_index_0_joint += 0.01;
    expect(() => HumanoidMotionArtifactSchema.parse(targetTamper))
      .toThrow(/does not match coordination input/);

    const missing = structuredClone(artifact) as unknown as {
      frames: Array<Record<string, unknown>>;
    };
    delete missing.frames[0]!.handCommand;
    expect(() => HumanoidMotionArtifactSchema.parse(missing)).toThrow();
  });

  it("binds closed-loop task-space targets and their authority into the artifact hash", () => {
    const target = {
      body: "right_wrist_yaw_link" as const,
      frame: "world" as const,
      position: { x: 0.2, y: 1.1, z: 0.3 },
      tolerance: 0.04
    };
    const artifact = HumanoidMotionArtifactSchema.parse({
      version: 1,
      protocol: "humanoid-motion-v1",
      generator: "task_space_constraints",
      controlStepSeconds: 0.02,
      durationSeconds: 0.02,
      taskSpaceServo: HUMANOID_TASK_SPACE_SERVO_DESCRIPTOR,
      frames: [{
        atSeconds: 0.02,
        reference: serializeHumanoidReference(neutralHumanoidReference()),
        taskSpaceTargets: [target]
      }]
    });
    const changed = structuredClone(artifact);
    changed.frames[0]!.taskSpaceTargets![0]!.position.x += 0.01;
    expect(humanoidMotionArtifactSha256(changed)).not.toBe(
      humanoidMotionArtifactSha256(artifact)
    );

    const missingAuthority = structuredClone(artifact) as unknown as {
      taskSpaceServo?: unknown;
    };
    delete missingAuthority.taskSpaceServo;
    expect(() => HumanoidMotionArtifactSchema.parse(missingAuthority))
      .toThrow(/authority descriptor/);

    const emptyAuthority = structuredClone(artifact) as unknown as {
      frames: Array<Record<string, unknown>>;
    };
    delete emptyAuthority.frames[0]!.taskSpaceTargets;
    expect(() => HumanoidMotionArtifactSchema.parse(emptyAuthority))
      .toThrow(/requires executable frame targets/);
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
