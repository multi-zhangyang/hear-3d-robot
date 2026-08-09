import { describe, expect, it } from "vitest";
import type { HumanoidLearnedPolicyCapability } from "../../domain/humanoid-policy.js";
import type { JsonValue } from "../../domain/schema.js";
import {
  neutralHumanoidReference,
  targetReference
} from "./reference.js";
import {
  CapabilityRoutingHumanoidController,
  humanoidControllerNeedsReferenceFallback
} from "./capability-routing-controller.js";
import { HUMANOID_JOINT_NAMES } from "./model.js";
import type {
  HumanoidControllerDescriptor,
  HumanoidControllerInferenceOptions,
  HumanoidControllerState,
  HumanoidJointPositionCommand,
  HumanoidPolicyState,
  HumanoidWholeBodyController
} from "./whole-body-controller.js";

describe("humanoid controller capability routing", () => {
  it("keeps learned locomotion active while overlaying upper-body reference tracking", async () => {
    const primary = controller("trained-velocity", ["balance", "locomotion"], 1);
    const fallback = controller(
      "reference-tracking",
      ["balance", "locomotion", "joint_reference_tracking"],
      2
    );
    const routed = new CapabilityRoutingHumanoidController(primary, fallback);
    const state = policyState();
    const navigation = neutralHumanoidReference();
    const armTask = targetReference(navigation, {
      joints: { right_shoulder_pitch_joint: 0.45 }
    });
    const shoulderIndex = HUMANOID_JOINT_NAMES.indexOf(
      "right_shoulder_pitch_joint"
    );

    expect(routed.descriptor).toMatchObject({
      implementation: "trained-velocity",
      learnedPolicy: { capabilities: ["balance", "locomotion"] },
      capabilityRouting: {
        protocol: "humanoid-controller-capability-routing-v1",
        strategy: "declared_capabilities",
        fallback: {
          mode: "reference_control",
          implementation: "reference-tracking"
        }
      }
    });

    routed.reset(state, navigation);
    expect((await routed.infer(state, navigation)).positions[0]).toBe(1);
    routed.advanceHistory(state, navigation);
    expect(primary.calls).toMatchObject({ reset: 1, infer: 1, advance: 1 });
    expect(fallback.calls).toMatchObject({ reset: 1, infer: 0, advance: 0 });

    const armCommand = await routed.infer(state, armTask);
    expect(armCommand.positions[0]).toBe(1);
    expect(armCommand.positions[shoulderIndex]).toBe(2);
    routed.advanceHistory(state, armTask);
    expect(primary.calls).toMatchObject({ reset: 1, infer: 2, advance: 2 });
    expect(fallback.calls).toMatchObject({ reset: 2, infer: 1, advance: 1 });
    expect(routed.executionState()).toMatchObject({
      mode: "hybrid_control",
      activeImplementation: "trained-velocity+reference-tracking"
    });

    expect((await routed.infer(state, navigation)).positions[0]).toBe(1);
    expect(primary.calls.reset).toBe(1);
  });

  it("uses full fallback when tracking includes the waist or legs", async () => {
    const primary = controller("trained-velocity", ["balance", "locomotion"], 1);
    const fallback = controller(
      "reference-tracking",
      ["balance", "locomotion", "joint_reference_tracking"],
      2
    );
    const routed = new CapabilityRoutingHumanoidController(primary, fallback);
    const state = policyState();
    const navigation = neutralHumanoidReference();

    for (const joint of ["waist_yaw_joint", "left_knee_joint"] as const) {
      const task = targetReference(navigation, { joints: { [joint]: 0.2 } });
      routed.reset(state, navigation);
      expect((await routed.infer(state, task)).positions[0]).toBe(2);
      expect(routed.executionState()).toMatchObject({
        mode: "reference_control",
        activeImplementation: "reference-tracking"
      });
    }
  });

  it("blends only tracked upper-body joints by their reference authority", async () => {
    const routed = new CapabilityRoutingHumanoidController(
      controller("trained-velocity", ["balance", "locomotion"], 1),
      controller(
        "reference-tracking",
        ["balance", "locomotion", "joint_reference_tracking"],
        3
      )
    );
    const state = policyState();
    const reference = neutralHumanoidReference();
    const elbowIndex = HUMANOID_JOINT_NAMES.indexOf("left_elbow_joint");
    reference.jointTrackingWeights[elbowIndex] = 0.25;

    routed.reset(state, reference);
    const command = await routed.infer(state, reference);
    expect(command.positions[0]).toBe(1);
    expect(command.positions[elbowIndex]).toBe(1.5);
    expect(command.stiffness[elbowIndex]).toBe(1.5);
    expect(command.damping[elbowIndex]).toBe(1.5);
  });

  it("routes a missing task capability without inventing it on the trained policy", async () => {
    const primary = controller("trained-velocity", ["balance", "locomotion"], 1);
    const fallback = controller(
      "reference-tracking",
      ["balance", "locomotion", "joint_reference_tracking"],
      2
    );
    const routed = new CapabilityRoutingHumanoidController(primary, fallback);
    const state = policyState();
    const reference = targetReference(neutralHumanoidReference(), {
      joints: { left_wrist_pitch_joint: 0.25 }
    });
    const options = taskOptions(["locomotion", "contact_rich_manipulation"]);

    routed.reset(state, reference);
    expect((await routed.infer(state, reference, options)).positions[0]).toBe(2);
    routed.advanceHistory(state, reference, options);
    expect(primary.calls.infer).toBe(0);
    expect(fallback.calls.infer).toBe(1);
    expect(routed.descriptor.learnedPolicy?.capabilities).toEqual([
      "balance",
      "locomotion"
    ]);
  });

  it("captures both branches, restores the active route, and accepts legacy primary state", async () => {
    const state = policyState();
    const navigation = neutralHumanoidReference();
    const task = targetReference(navigation, {
      joints: { left_elbow_joint: 0.7 }
    });
    const firstPrimary = controller("trained-velocity", ["balance", "locomotion"], 1);
    const firstFallback = controller(
      "reference-tracking",
      ["balance", "locomotion", "joint_reference_tracking"],
      2
    );
    const first = new CapabilityRoutingHumanoidController(
      firstPrimary,
      firstFallback
    );
    first.reset(state, navigation);
    await first.infer(state, task);
    first.advanceHistory(state, task);
    const checkpoint = first.captureState();

    expect(checkpoint).toMatchObject({
      implementation: "trained-velocity",
      payload: {
        protocol: "humanoid-controller-capability-routing-state-v3",
        active: "upper_body_overlay",
        primary: { implementation: "trained-velocity" },
        fallback: { implementation: "reference-tracking" },
        last_command: { kind: "joint_position_pd" },
        handoff: null
      }
    });

    const restoredPrimary = controller("trained-velocity", ["balance", "locomotion"], 1);
    const restoredFallback = controller(
      "reference-tracking",
      ["balance", "locomotion", "joint_reference_tracking"],
      2
    );
    const restored = new CapabilityRoutingHumanoidController(
      restoredPrimary,
      restoredFallback
    );
    restored.restoreState(checkpoint);
    expect(restoredPrimary.calls.restore).toBe(1);
    expect(restoredFallback.calls.restore).toBe(1);
    await restored.infer(state, task);
    expect(restoredFallback.calls.reset).toBe(0);

    const v2Payload = checkpoint.payload as Record<string, JsonValue>;
    const v1Primary = controller("trained-velocity", ["balance", "locomotion"], 1);
    const v1Fallback = controller(
      "reference-tracking",
      ["balance", "locomotion", "joint_reference_tracking"],
      2
    );
    const restoredV1 = new CapabilityRoutingHumanoidController(
      v1Primary,
      v1Fallback
    );
    restoredV1.restoreState({
      ...checkpoint,
      payload: {
        protocol: "humanoid-controller-capability-routing-state-v1",
        active: "fallback",
        primary: v2Payload.primary!,
        fallback: v2Payload.fallback!
      }
    });
    expect(restoredV1.executionState()).toMatchObject({
      mode: "reference_control",
      activeImplementation: "reference-tracking",
      transition: null
    });

    const restoredV2 = new CapabilityRoutingHumanoidController(
      controller("trained-velocity", ["balance", "locomotion"], 1),
      controller(
        "reference-tracking",
        ["balance", "locomotion", "joint_reference_tracking"],
        2
      )
    );
    restoredV2.restoreState({
      ...checkpoint,
      payload: {
        protocol: "humanoid-controller-capability-routing-state-v2",
        active: "fallback",
        primary: v2Payload.primary!,
        fallback: v2Payload.fallback!,
        last_command: v2Payload.last_command!,
        handoff: null
      }
    });
    expect(restoredV2.executionState()).toMatchObject({
      mode: "reference_control",
      activeImplementation: "reference-tracking",
      transition: null
    });

    const legacyPrimary = controller("trained-velocity", ["balance", "locomotion"], 1);
    const legacyFallback = controller(
      "reference-tracking",
      ["balance", "locomotion", "joint_reference_tracking"],
      2
    );
    const legacy = new CapabilityRoutingHumanoidController(
      legacyPrimary,
      legacyFallback
    );
    legacy.restoreState(firstPrimary.captureState());
    await legacy.infer(state, navigation);
    expect(legacyPrimary.calls.restore).toBe(1);
    expect(legacyPrimary.calls.reset).toBe(0);
    expect(legacyFallback.calls.restore).toBe(0);
  });

  it("hands control over continuously and resumes an in-flight handoff after restore", async () => {
    const primary = controller(
      "trained-velocity",
      ["balance", "locomotion"],
      1,
      0.02,
      0.1
    );
    const fallback = controller(
      "reference-tracking",
      ["balance", "locomotion", "joint_reference_tracking"],
      2,
      0.02,
      0.1
    );
    const routed = new CapabilityRoutingHumanoidController(primary, fallback);
    const state = policyState();
    const navigation = neutralHumanoidReference();
    const task = targetReference(navigation, {
      joints: { right_elbow_joint: 0.6 }
    });
    const elbowIndex = HUMANOID_JOINT_NAMES.indexOf("right_elbow_joint");

    routed.reset(state, navigation);
    const navigationCommand = await routed.infer(state, navigation);
    expect(navigationCommand.positions[0]).toBe(1);
    expect(routed.executionState()).toEqual({
      protocol: "humanoid-controller-execution-v1",
      mode: "learned_policy",
      activeImplementation: "trained-velocity",
      transition: null
    });

    const firstTaskCommand = await routed.infer(state, task);
    expect(firstTaskCommand.positions[0]).toBe(1);
    expect(firstTaskCommand.positions[elbowIndex]).toBeCloseTo(1.104, 12);
    expect(firstTaskCommand.stiffness[elbowIndex]).toBeCloseTo(1.104, 12);
    expect(firstTaskCommand.damping[elbowIndex]).toBeCloseTo(1.104, 12);
    expect(routed.executionState()).toMatchObject({
      mode: "hybrid_control",
      activeImplementation: "trained-velocity+reference-tracking",
      transition: {
        fromImplementation: "trained-velocity",
        toImplementation: "trained-velocity+reference-tracking",
        progress: 0.2,
        durationSeconds: 0.1
      }
    });
    const secondTaskCommand = await routed.infer(state, task);
    expect(secondTaskCommand.positions[elbowIndex]).toBeCloseTo(1.352, 12);
    const checkpoint = routed.captureState();

    const malformed = structuredClone(checkpoint);
    const malformedPayload = malformed.payload as Record<string, JsonValue>;
    const malformedHandoff = malformedPayload.handoff as Record<string, JsonValue>;
    malformedHandoff.completed_steps = malformedHandoff.total_steps!;
    const rejectsMalformed = new CapabilityRoutingHumanoidController(
      controller("trained-velocity", ["balance", "locomotion"], 1, 0.02, 0.1),
      controller(
        "reference-tracking",
        ["balance", "locomotion", "joint_reference_tracking"],
        2,
        0.02,
        0.1
      )
    );
    expect(() => rejectsMalformed.restoreState(malformed)).toThrow(
      "Invalid humanoid capability-routing handoff state"
    );

    const restored = new CapabilityRoutingHumanoidController(
      controller("trained-velocity", ["balance", "locomotion"], 1, 0.02, 0.1),
      controller(
        "reference-tracking",
        ["balance", "locomotion", "joint_reference_tracking"],
        2,
        0.02,
        0.1
      )
    );
    restored.restoreState(checkpoint);
    expect(restored.executionState().transition?.progress).toBe(0.4);
    const [continued, resumed] = await Promise.all([
      routed.infer(state, task),
      restored.infer(state, task)
    ]);
    expect([...resumed.positions]).toEqual([...continued.positions]);
    expect(continued.positions[elbowIndex]).toBeCloseTo(1.648, 12);

    await routed.infer(state, task);
    const completed = await routed.infer(state, task);
    expect(completed.positions[0]).toBe(1);
    expect(completed.positions[elbowIndex]).toBe(2);
    expect(routed.executionState().transition).toBeNull();

    const returned = await routed.infer(state, navigation);
    expect(returned.positions[0]).toBe(1);
    expect(returned.positions[elbowIndex]).toBeCloseTo(1.896, 12);
    expect(routed.executionState()).toMatchObject({
      mode: "learned_policy",
      activeImplementation: "trained-velocity",
      transition: { progress: 0.2 }
    });
  });

  it("rejects controller pairs that cannot switch at one physical control boundary", () => {
    const primary = controller("trained-velocity", ["balance", "locomotion"], 1);
    expect(() => new CapabilityRoutingHumanoidController(
      primary,
      controller("wrong-timing", ["joint_reference_tracking"], 2, 0.04)
    )).toThrow("identical timing and actuation");
    expect(() => new CapabilityRoutingHumanoidController(
      primary,
      controller("no-reference-tracking", ["balance", "locomotion"], 2)
    )).toThrow("must support balance, locomotion, and joint tracking");
    expect(() => new CapabilityRoutingHumanoidController(
      primary,
      controller("no-locomotion", ["balance", "joint_reference_tracking"], 2)
    )).toThrow("must support balance, locomotion, and joint tracking");
  });

  it("does not wrap controllers without a learned policy or controllers already routed", () => {
    const reference = controller(
      "reference-only",
      ["balance", "locomotion", "joint_reference_tracking"],
      1
    );
    expect(humanoidControllerNeedsReferenceFallback(reference.descriptor)).toBe(false);
    const primary = controller("trained-velocity", ["balance", "locomotion"], 1);
    expect(humanoidControllerNeedsReferenceFallback(primary.descriptor)).toBe(true);
    const trackingOnly = controller(
      "trained-tracking",
      ["joint_reference_tracking"],
      1
    );
    expect(humanoidControllerNeedsReferenceFallback(trackingOnly.descriptor)).toBe(true);
    const routed = new CapabilityRoutingHumanoidController(primary, reference);
    expect(humanoidControllerNeedsReferenceFallback(routed.descriptor)).toBe(false);

    const descriptorWithoutPolicy: HumanoidControllerDescriptor = {
      protocol: "humanoid-controller-v1",
      implementation: "classical-control",
      actuation: "joint_position_pd",
      controlStepSeconds: 0.02,
      physicsStepSeconds: 0.005
    };
    expect(humanoidControllerNeedsReferenceFallback(descriptorWithoutPolicy)).toBe(false);
  });
});

interface TestController extends HumanoidWholeBodyController {
  calls: { reset: number; infer: number; advance: number; restore: number };
}

function controller(
  implementation: string,
  capabilities: HumanoidLearnedPolicyCapability[],
  commandValue: number,
  controlStepSeconds = 0.02,
  commandResponseHorizonSeconds?: number
): TestController {
  let revision = 0;
  const calls = { reset: 0, infer: 0, advance: 0, restore: 0 };
  const descriptor: HumanoidControllerDescriptor = {
    protocol: "humanoid-controller-v1",
    implementation,
    actuation: "joint_position_pd",
    controlStepSeconds,
    physicsStepSeconds: 0.005,
    ...(commandResponseHorizonSeconds === undefined
      ? {}
      : { commandResponseHorizonSeconds }),
    learnedPolicy: {
      protocol: "humanoid-learned-policy-v1",
      runtime: "test",
      observationSpace: { protocol: "test-observation-v1", size: 1 },
      actionSpace: { protocol: "test-action-v1", size: HUMANOID_JOINT_NAMES.length },
      capabilities
    }
  };
  return {
    descriptor,
    calls,
    reset() {
      calls.reset += 1;
    },
    async infer(): Promise<HumanoidJointPositionCommand> {
      calls.infer += 1;
      revision += 1;
      return {
        kind: "joint_position_pd",
        positions: new Float64Array(HUMANOID_JOINT_NAMES.length).fill(commandValue),
        stiffness: new Float64Array(HUMANOID_JOINT_NAMES.length).fill(commandValue),
        damping: new Float64Array(HUMANOID_JOINT_NAMES.length).fill(commandValue)
      };
    },
    advanceHistory() {
      calls.advance += 1;
    },
    captureState(): HumanoidControllerState {
      return {
        protocol: "humanoid-controller-state-v1",
        version: 1,
        implementation,
        payload: { revision }
      };
    },
    restoreState(value: HumanoidControllerState) {
      if (value.implementation !== implementation) throw new Error("wrong state");
      calls.restore += 1;
      revision = Number((value.payload as { revision: number }).revision);
    },
    async dispose() {}
  };
}

function policyState(): HumanoidPolicyState {
  return {
    jointPositions: new Float64Array(HUMANOID_JOINT_NAMES.length),
    jointVelocities: new Float64Array(HUMANOID_JOINT_NAMES.length),
    rootQuaternion: [1, 0, 0, 0],
    rootAngularVelocity: [0, 0, 0]
  };
}

function taskOptions(
  requestedCapabilities: HumanoidLearnedPolicyCapability[]
): HumanoidControllerInferenceOptions {
  return {
    taskCommand: {
      protocol: "humanoid-controller-task-v1",
      taskId: "capability-routing-test",
      source: "motion_option",
      requestedCapabilities,
      goal: null,
      endEffectors: [],
      grasps: []
    }
  };
}
