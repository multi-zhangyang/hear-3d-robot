import { describe, expect, it } from "vitest";
import {
  DEFAULT_HUMANOID_PLAN_INTENT_LEASE_SECONDS,
  humanoidMotionIntentSha256,
  humanoidNavigationIntentSha256,
  humanoidPlanExpiryRevision,
  humanoidPlanIntentIsActive
} from "./plan-lifecycle.js";

describe("humanoid plan intent lifecycle", () => {
  it("covers planning handoff across independent model turns", () => {
    expect(DEFAULT_HUMANOID_PLAN_INTENT_LEASE_SECONDS).toBeGreaterThanOrEqual(600);
  });

  it("keeps the exact model motion intent identity stable", () => {
    const plan = {
      id: "reach-left",
      intent: "左腕靠近当前可见目标",
      duration_seconds: 0.4,
      keyframes: [
        { at_seconds: 0 },
        { at_seconds: 0.4, torso_yaw: 0.1 }
      ]
    };
    const identity = humanoidMotionIntentSha256(plan);
    expect(identity).toMatch(/^[a-f0-9]{64}$/);
    expect(humanoidMotionIntentSha256(structuredClone(plan))).toBe(identity);
    expect(humanoidMotionIntentSha256({ ...plan, intent: "不同意图" })).not.toBe(identity);
  });

  it("normalizes negative zero in navigation identities", () => {
    expect(humanoidNavigationIntentSha256({ x: -0, y: 0, z: 2 }))
      .toBe(humanoidNavigationIntentSha256({ x: 0, y: 0, z: 2 }));
  });

  it("binds a model-selected arrival heading into navigation intent identity", () => {
    const target = { x: 1, y: 0, z: 2 };
    const facing = {
      type: "face_point" as const,
      target: { x: 2, y: 0.7, z: 2 },
      tolerance_radians: 0.1
    };

    expect(humanoidNavigationIntentSha256(target, facing)).not.toBe(
      humanoidNavigationIntentSha256(target)
    );
    expect(humanoidNavigationIntentSha256(target, structuredClone(facing))).toBe(
      humanoidNavigationIntentSha256(target, facing)
    );
  });

  it("derives an inclusive revision lease from the controller cadence", () => {
    const expiresRevision = humanoidPlanExpiryRevision({
      createdRevision: 40,
      controlStepSeconds: 0.02,
      leaseSeconds: 1
    });
    expect(expiresRevision).toBe(90);
    expect(humanoidPlanIntentIsActive(90, expiresRevision)).toBe(true);
    expect(humanoidPlanIntentIsActive(91, expiresRevision)).toBe(false);
  });
});
