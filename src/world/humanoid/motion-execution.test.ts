import { describe, expect, it } from "vitest";
import {
  humanoidHandContactKey,
  type HumanoidMotionPlan
} from "./motion-plan.js";
import { HumanoidMotionExecution } from "./motion-execution.js";
import { HumanoidWorldCheckpointSchema } from "./checkpoint.js";
import {
  HumanoidMotionArtifactSchema,
  serializeHumanoidReference
} from "./motion-artifact.js";
import { neutralHumanoidReference } from "./reference.js";
import { humanoidMotionIntentSha256 } from "./plan-lifecycle.js";
import {
  HumanoidSimulation,
  type HumanoidSimulationSnapshot
} from "./simulation.js";
import type { StoredHumanoidMotionPlan } from "./world-plan-state.js";
import { HumanoidGraspRegistry } from "./grasp-registry.js";

describe("humanoid hand-contact execution authority", () => {
  it("rejects an unlisted hand surface and persists exact failure evidence", async () => {
    const fixture = await handContactFixture();
    try {
      const stored = storedMotion([]);
      const execution = createExecution(stored);

      await execution.step(fixture.simulation);

      expect(execution.result().failures).toEqual([expect.objectContaining({
        code: "environment_contact",
        handSurfaces: ["left_hand_palm_link"],
        contacts: [{
          handSurface: "left_hand_palm_link",
          objectId: "crate",
          solidId: null,
          normalForce: 9
        }]
      })]);
      expect(stored.progress.failure).toEqual({
        code: "environment_contact",
        atSeconds: 0.02,
        handSurfaces: ["left_hand_palm_link"],
        contacts: [{
          handSurface: "left_hand_palm_link",
          objectId: "crate",
          solidId: null,
          normalForce: 9
        }]
      });
    } finally {
      await fixture.dispose();
    }
  }, 30_000);

  it("records only the exact authorized surface as required contact evidence", async () => {
    const fixture = await handContactFixture();
    try {
      const stored = storedMotion([{
        hand_surface: "left_hand_palm_link",
        object_id: "crate",
        required: true
      }]);
      const execution = createExecution(stored);

      await execution.step(fixture.simulation);

      expect(execution.result().failures).toEqual([]);
      expect(stored.progress.satisfiedContactKeys).toEqual([
        humanoidHandContactKey("left_hand_palm_link", "crate")
      ]);
      expect(stored.progress.failure).toBeNull();
    } finally {
      await fixture.dispose();
    }
  }, 30_000);

  it("restores only namespaced contact keys and exact contact evidence", () => {
    const stored = storedMotion([{
      hand_surface: "left_hand_palm_link",
      object_id: "crate",
      required: true
    }]);
    stored.progress.nextFrameIndex = 1;
    stored.progress.satisfiedContactKeys = [
      humanoidHandContactKey("left_hand_palm_link", "crate")
    ];
    const checkpoint = checkpointWith(stored);

    expect(HumanoidWorldCheckpointSchema.safeParse(checkpoint).success).toBe(true);

    const wrongGraspFrame = structuredClone(checkpoint);
    wrongGraspFrame.graspRegistry.last_frame = 0;
    expect(HumanoidWorldCheckpointSchema.safeParse(wrongGraspFrame).success)
      .toBe(false);

    const legacyBodyKey = structuredClone(checkpoint);
    legacyBodyKey.motions[0]!.progress.satisfiedContactKeys = [
      "left_wrist_yaw_link\u0000crate"
    ];
    expect(HumanoidWorldCheckpointSchema.safeParse(legacyBodyKey).success).toBe(false);

    const ambiguousEvidence = structuredClone(checkpoint);
    ambiguousEvidence.motions[0]!.progress.failure = {
      code: "environment_contact",
      atSeconds: 0.02,
      bodies: ["left_wrist_yaw_link"],
      handSurfaces: ["left_hand_palm_link"],
      contacts: [{
        body: "left_wrist_yaw_link",
        handSurface: "left_hand_palm_link",
        objectId: "crate",
        normalForce: 9
      }]
    };
    expect(HumanoidWorldCheckpointSchema.safeParse(ambiguousEvidence).success)
      .toBe(false);
  });
});

function createExecution(
  stored: StoredHumanoidMotionPlan
): HumanoidMotionExecution {
  return new HumanoidMotionExecution({
    stored,
    reference: neutralHumanoidReference(),
    detectorInput: () => {
      throw new Error("A non-option motion cannot request option evidence");
    }
  });
}

function storedMotion(
  contactConstraints: NonNullable<HumanoidMotionPlan["contact_constraints"]>
): StoredHumanoidMotionPlan {
  const reference = neutralHumanoidReference();
  const artifact = HumanoidMotionArtifactSchema.parse({
    version: 1,
    protocol: "humanoid-motion-v1",
    generator: "hand-contact-execution-test",
    controlStepSeconds: 0.02,
    durationSeconds: 0.02,
    frames: [{
      atSeconds: 0.02,
      reference: serializeHumanoidReference(reference)
    }]
  });
  const plan: HumanoidMotionPlan = {
    id: "hand-contact-execution",
    intent: "验证精确手部接触授权",
    duration_seconds: 0.02,
    contact_constraints: contactConstraints,
    keyframes: [{ at_seconds: 0 }, { at_seconds: 0.02 }]
  };
  return {
    plan,
    artifact,
    rollout: null,
    createdRevision: 0,
    validatedRevision: 0,
    validatedStateSha256: "a".repeat(64),
    expiresRevision: 10,
    intentSha256: humanoidMotionIntentSha256(plan),
    revalidationCount: 0,
    terminal: null,
    option: null,
    progress: {
      nextFrameIndex: 0,
      satisfiedContactKeys: [],
      driftStreak: 0,
      lastDrift: null,
      failure: null
    }
  };
}

function checkpointWith(stored: StoredHumanoidMotionPlan): unknown {
  const graspRegistry = new HumanoidGraspRegistry({ portableObjectIds: [] });
  graspRegistry.observe(1, { objects: {} } as HumanoidSimulationSnapshot);
  return {
    version: 1,
    frame: 1,
    worldRevision: 1,
    routeSequence: 0,
    simulation: {
      time: 0.02,
      positions: [],
      velocities: [],
      controls: [],
      activations: [],
      accelerationWarmstart: [],
      controller: {
        protocol: "humanoid-controller-state-v1",
        version: 1,
        implementation: "checkpoint-test",
        payload: {}
      }
    },
    reference: serializeHumanoidReference(neutralHumanoidReference()),
    motions: [stored],
    routes: [],
    navigation: {
      planId: null,
      status: "idle",
      target: null,
      waypoints: [],
      waypointIndex: null
    },
    graspRegistry: graspRegistry.checkpoint()
  };
}

async function handContactFixture(): Promise<{
  simulation: HumanoidSimulation;
  dispose(): Promise<void>;
}> {
  const authority = await HumanoidSimulation.create({
    objects: [{
      id: "crate",
      center: { x: 4, y: 0.2, z: 4 },
      size: { x: 0.3, y: 0.3, z: 0.3 },
      mass: 0.2
    }]
  });
  const snapshot = structuredClone(authority.snapshot());
  snapshot.contacts.push({
    position: { x: 0.25, y: 0.8, z: 0.2 },
    normal: { x: 0, y: 0, z: 1 },
    normalForce: 9,
    firstBody: "left_wrist_yaw_link",
    secondBody: null,
    firstObject: null,
    secondObject: "crate",
    firstHandLink: "left_hand_palm_link",
    secondHandLink: null
  });
  snapshot.contactCount = snapshot.contacts.length;
  const simulation = {
    async step(): Promise<HumanoidSimulationSnapshot> {
      return structuredClone(snapshot);
    }
  } as unknown as HumanoidSimulation;
  return {
    simulation,
    dispose: () => authority.dispose()
  };
}
