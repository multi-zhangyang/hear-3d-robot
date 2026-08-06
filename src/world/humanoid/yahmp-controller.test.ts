import { describe, expect, it, vi } from "vitest";
import {
  HUMANOID_JOINT_NAMES,
  YAHMP_POLICY
} from "./model.js";
import { neutralHumanoidReference } from "./reference.js";
import type { HumanoidPolicyState } from "./whole-body-controller.js";
import { YahmpController } from "./yahmp-controller.js";

describe("YAHMP joint tracking authority", () => {
  it("attenuates policy residuals using each reference tracking weight", async () => {
    const action = Float32Array.from(
      { length: HUMANOID_JOINT_NAMES.length },
      (_, index) => (index + 1) / 100
    );
    const session = fakeSession(action);
    const controller = createController(session);
    const reference = neutralHumanoidReference();
    reference.jointTrackingWeights[0] = 0;
    reference.jointTrackingWeights[1] = 0.25;
    reference.jointTrackingWeights[2] = 1;

    const command = await controller.infer(policyState(reference.jointPositions), reference);

    for (const index of [0, 1, 2]) {
      expect(command.positions[index]).toBeCloseTo(
        reference.jointPositions[index]!
          + action[index]! * YAHMP_POLICY.actionScale[index]!
            * (1 - reference.jointTrackingWeights[index]!),
        12
      );
    }
    expect(command.positions[2]).toBe(reference.jointPositions[2]);
    expect(command.stiffness[0]).toBe(YAHMP_POLICY.stiffness[0]);
    expect(command.stiffness[1]).toBeGreaterThan(YAHMP_POLICY.stiffness[1]);
    expect(command.stiffness[1]).toBeLessThan(command.stiffness[2]!);
    expect(command.stiffness[2]).toBe(100);
    expect(command.damping[0]).toBe(YAHMP_POLICY.damping[0]);
    expect(command.damping[2]).toBeGreaterThan(YAHMP_POLICY.damping[2]);
    expect(session.run).toHaveBeenCalledOnce();
    expect(session.output.dispose).toHaveBeenCalledOnce();
    await controller.dispose();
  });

  it("fails closed before inference when tracking weights are invalid", async () => {
    const session = fakeSession(new Float32Array(HUMANOID_JOINT_NAMES.length));
    const controller = createController(session);
    const reference = neutralHumanoidReference();
    reference.jointTrackingWeights[0] = Number.NaN;

    await expect(
      controller.infer(policyState(reference.jointPositions), reference)
    ).rejects.toThrow("tracking weights");
    expect(session.run).not.toHaveBeenCalled();
    await controller.dispose();
  });

  it("separates a tracked joint from the locomotion policy command on request", async () => {
    const session = fakeSession(new Float32Array(HUMANOID_JOINT_NAMES.length));
    const controller = createController(session);
    const reference = neutralHumanoidReference();
    const trackedIndex = 15;
    reference.jointTrackingWeights[trackedIndex] = 1;
    const jointPositions = reference.jointPositions.slice();
    jointPositions[trackedIndex] = reference.jointPositions[trackedIndex]! + 0.4;
    const jointVelocities = new Float64Array(HUMANOID_JOINT_NAMES.length);
    jointVelocities[trackedIndex] = 0.3;
    const state = { ...policyState(jointPositions), jointVelocities };

    await controller.infer(state, reference, {
      trackedJointPolicyCommand: "measured"
    });
    await controller.infer(state, reference, {
      trackedJointPolicyCommand: "neutral"
    });

    expect(session.observations[0]![trackedIndex]).toBeCloseTo(
      state.jointPositions[trackedIndex]!,
      6
    );
    expect(session.observations[0]![HUMANOID_JOINT_NAMES.length + trackedIndex])
      .toBeCloseTo(state.jointVelocities[trackedIndex]!, 6);
    expect(session.observations[1]![trackedIndex]).toBeCloseTo(
      YAHMP_POLICY.defaultJointPositions[trackedIndex]!,
      6
    );
    expect(session.observations[1]![HUMANOID_JOINT_NAMES.length + trackedIndex])
      .toBe(0);
    await controller.dispose();
  });
});

function policyState(jointPositions: Float64Array): HumanoidPolicyState {
  return {
    jointPositions: jointPositions.slice(),
    jointVelocities: new Float64Array(HUMANOID_JOINT_NAMES.length),
    rootQuaternion: [1, 0, 0, 0],
    rootAngularVelocity: [0, 0, 0]
  };
}

function fakeSession(action: Float32Array) {
  const output = { data: action, dispose: vi.fn() };
  const observations: Float32Array[] = [];
  return {
    run: vi.fn(async (feeds: { obs: { data: ArrayLike<number> } }) => {
      observations.push(Float32Array.from(feeds.obs.data));
      return { actions: output };
    }),
    release: vi.fn(async () => undefined),
    output,
    observations
  };
}

function createController(session: ReturnType<typeof fakeSession>): YahmpController {
  const ConstructibleController = YahmpController as unknown as new (
    session: ReturnType<typeof fakeSession>
  ) => YahmpController;
  return new ConstructibleController(session);
}
