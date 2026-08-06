import { describe, expect, it } from "vitest";
import type { JsonValue } from "../../domain/schema.js";
import { HUMANOID_JOINT_INDEX, HUMANOID_JOINT_NAMES } from "./model.js";
import {
  inverseQuaternion,
  multiplyQuaternion,
  normalizeQuaternion,
  rotateVector,
  subtract
} from "../geometry.js";
import {
  interpolateReference,
  neutralHumanoidReference,
  targetReference
} from "./reference.js";
import { HumanoidSimulation } from "./simulation.js";
import {
  humanoidEndEffectorJointIndexes,
  humanoidEndEffectorTrackingJointIndexes
} from "./task-space-targets.js";
import type {
  HumanoidControllerDescriptor,
  HumanoidControllerState,
  HumanoidJointPositionCommand,
  HumanoidPolicyState,
  HumanoidWholeBodyController
} from "./whole-body-controller.js";

describe("HumanoidSimulation", () => {
  it("releases native contact handles across long-running snapshots", async () => {
    const simulation = await HumanoidSimulation.create();
    try {
      const neutral = neutralHumanoidReference();
      for (let index = 0; index < 100; index += 1) {
        await simulation.step(neutral);
      }
      for (let index = 0; index < 100; index += 1) simulation.snapshot();
      const baselineExternalBytes = process.memoryUsage().external;
      for (let index = 0; index < 2_000; index += 1) simulation.snapshot();
      const growthBytes = process.memoryUsage().external - baselineExternalBytes;

      expect(growthBytes).toBeLessThan(8 * 1_024 * 1_024);
    } finally {
      await simulation.dispose();
    }
  }, 30_000);

  it("holds physical balance and tracks a generated whole-body arm reference", async () => {
    const simulation = await HumanoidSimulation.create();
    try {
      const neutral = neutralHumanoidReference();
      for (let index = 0; index < 100; index += 1) await simulation.step(neutral);
      const standing = simulation.snapshot();
      expect(standing.controller).toEqual({
        protocol: "humanoid-controller-v1",
        implementation: "yahmp_onnx",
        actuation: "joint_position_pd",
        controlStepSeconds: 0.02,
        physicsStepSeconds: 0.005,
        commandResponseHorizonSeconds: 0.2,
        minimumEffectivePlanarSpeedMetersPerSecond: 0.15
      });
      expect(simulation.captureState().controller).toMatchObject({
        version: 1,
        implementation: "yahmp_onnx"
      });
      expect(standing.fallen).toBe(false);
      expect(standing.rootPosition.y).toBeGreaterThan(0.68);
      expect(standing.contactCount).toBeGreaterThan(0);
      expect(standing.balance.support).not.toBe("none");
      expect(standing.balance.centerOfMass.y).toBeGreaterThan(0.5);
      expect(standing.feet.left.normalForce + standing.feet.right.normalForce).toBeGreaterThan(0);
      expect(standing.nonFootEnvironmentContacts).toEqual([]);
      expect(Object.keys(standing.links)).toContain("left_ankle_roll_link");
      expect(Object.keys(standing.links)).toContain("right_wrist_yaw_link");
      expect(Object.keys(standing.links)).toContain("head_link");

      const leftFootInPelvis = rotateVector(
        inverseQuaternion(standing.links.pelvis.rotation),
        subtract(
          standing.links.left_ankle_roll_link.position,
          standing.links.pelvis.position
        )
      );
      const footSolution = simulation.solveEndEffectorTargets(neutral, [{
        body: "left_ankle_roll_link",
        position: {
          ...leftFootInPelvis,
          y: leftFootInPelvis.y + 0.08,
          z: leftFootInPelvis.z + 0.03
        },
        frame: "pelvis",
        tolerance: 0.025
      }]);
      expect(footSolution.residuals).toEqual([
        expect.objectContaining({
          body: "left_ankle_roll_link",
          error: expect.any(Number)
        })
      ]);
      expect(footSolution.residuals[0]!.error).toBeLessThanOrEqual(0.025);
      expect(footSolution.reference.jointPositions[3]).toBeGreaterThan(
        neutral.jointPositions[3]! + 0.2
      );
      expect(Array.from(footSolution.reference.jointPositions.slice(6))).toEqual(
        Array.from(neutral.jointPositions.slice(6))
      );
      const leftLegIndexes = new Set(
        humanoidEndEffectorJointIndexes("left_ankle_roll_link")
      );
      const leftLegTrackingIndexes = new Set(
        humanoidEndEffectorTrackingJointIndexes("left_ankle_roll_link")
      );
      expect(leftLegIndexes.size).toBe(5);
      expect(leftLegTrackingIndexes).toEqual(leftLegIndexes);
      for (let index = 0; index < HUMANOID_JOINT_NAMES.length; index += 1) {
        expect(footSolution.reference.jointTrackingWeights[index]).toBe(
          leftLegTrackingIndexes.has(index) ? 1 : 0
        );
      }
      expect(footSolution.reference.jointTrackingWeights[
        HUMANOID_JOINT_INDEX.get("left_ankle_roll_joint")!
      ]).toBe(0);

      const raised = targetReference(neutral, {
        joints: {
          left_shoulder_pitch_joint: -1.1,
          left_shoulder_roll_joint: 0.55,
          left_elbow_joint: 1.25
        }
      });
      for (let index = 0; index < 150; index += 1) {
        await simulation.step(interpolateReference(neutral, raised, index / 149, 3));
      }
      const moved = simulation.snapshot();
      const shoulder = HUMANOID_JOINT_INDEX.get("left_shoulder_pitch_joint");
      expect(shoulder).toBeDefined();
      expect(moved.joints.left_shoulder_pitch_joint.position).toBeLessThan(-0.65);
      expect(moved.fallen).toBe(false);
      expect(moved.rootPosition.y).toBeGreaterThan(0.64);
    } finally {
      await simulation.dispose();
    }
  }, 30_000);

  it("simulates arbitrary movable world objects in the same contact world", async () => {
    const simulation = await HumanoidSimulation.create({
      objects: [{
        id: "movable-object",
        center: { x: 1.5, y: 1.2, z: 0 },
        size: { x: 0.4, y: 0.4, z: 0.4 },
        mass: 0.35
      }]
    });
    try {
      const neutral = neutralHumanoidReference();
      const initial = simulation.snapshot().objects["movable-object"];
      expect(initial).toBeDefined();
      for (let index = 0; index < 180; index += 1) await simulation.step(neutral);
      const settled = simulation.snapshot();
      expect(settled.objects["movable-object"]!.position.y).toBeCloseTo(0.2, 1);
      expect(settled.objects["movable-object"]!.position.y).toBeLessThan(initial!.position.y);
      expect(settled.fallen).toBe(false);
    } finally {
      await simulation.dispose();
    }
  }, 30_000);

  it("solves and tracks a complete end-effector pose with analytic Jacobians", async () => {
    const simulation = await HumanoidSimulation.create();
    try {
      const neutral = neutralHumanoidReference();
      const wrist = simulation.snapshot().links.left_wrist_yaw_link;
      const localYaw = normalizeQuaternion({
        x: 0,
        y: 0,
        z: Math.sin(0.18),
        w: Math.cos(0.18)
      });
      const targetOrientation = normalizeQuaternion(multiplyQuaternion(
        wrist.rotation,
        localYaw
      ));
      const solution = simulation.solveEndEffectorTargets(neutral, [{
        body: "left_wrist_yaw_link",
        position: wrist.position,
        frame: "world",
        tolerance: 0.003,
        orientation: targetOrientation,
        orientationTolerance: 0.01
      }]);

      expect(solution.residuals[0]).toMatchObject({
        body: "left_wrist_yaw_link",
        orientationTarget: targetOrientation
      });
      expect(solution.residuals[0]!.error).toBeLessThanOrEqual(0.003);
      expect(solution.residuals[0]!.orientationError).toBeLessThanOrEqual(0.01);
      expect(solution.reference.jointPositions[
        HUMANOID_JOINT_INDEX.get("left_wrist_yaw_joint")!
      ]).toBeGreaterThan(0.05);
      expect(solution.reference.jointTrackingWeights[
        HUMANOID_JOINT_INDEX.get("left_wrist_yaw_joint")!
      ]).toBe(1);

      const initial = simulation.snapshot();
      const anklePosition = rotateVector(
        inverseQuaternion(initial.links.pelvis.rotation),
        subtract(
          initial.links.left_ankle_roll_link.position,
          initial.links.pelvis.position
        )
      );
      const ankleOrientation = normalizeQuaternion(multiplyQuaternion(
        inverseQuaternion(initial.links.pelvis.rotation),
        initial.links.left_ankle_roll_link.rotation
      ));
      const ankleTargetOrientation = normalizeQuaternion(multiplyQuaternion(
        ankleOrientation,
        { x: 0, y: 0, z: Math.sin(0.05), w: Math.cos(0.05) }
      ));
      const ankleSolution = simulation.solveEndEffectorTargets(neutral, [{
        body: "left_ankle_roll_link",
        position: anklePosition,
        frame: "pelvis",
        tolerance: 0.002,
        orientation: ankleTargetOrientation,
        orientationTolerance: 0.005
      }]);
      expect(ankleSolution.residuals[0]!.error).toBeLessThanOrEqual(0.002);
      expect(ankleSolution.residuals[0]!.orientationError).toBeLessThanOrEqual(0.005);
      expect(ankleSolution.reference.jointPositions[
        HUMANOID_JOINT_INDEX.get("left_ankle_roll_joint")!
      ]).toBeGreaterThan(0.05);
      expect(ankleSolution.reference.jointTrackingWeights[
        HUMANOID_JOINT_INDEX.get("left_ankle_roll_joint")!
      ]).toBe(1);

      expect(() => simulation.solveEndEffectorTargets(neutral, [{
        body: "left_wrist_yaw_link",
        position: wrist.position,
        frame: "world",
        tolerance: 0.01,
        orientation: targetOrientation
      }])).toThrow(/provided together/);
    } finally {
      await simulation.dispose();
    }
  }, 30_000);

  it("reports only dynamic objects inside the physical head sensor cone with clear sight", async () => {
    const simulation = await HumanoidSimulation.create({
      solids: [{
        id: "manipulation-support",
        center: { x: 0.17, y: 0.3375, z: 0.28 },
        size: { x: 0.057, y: 0.675, z: 0.057 }
      }, {
        id: "occlusion-wall",
        center: { x: -0.6, y: 1.05, z: 1.2 },
        size: { x: 0.5, y: 0.8, z: 0.1 }
      }],
      objects: [
        {
          id: "manipulation-zone",
          center: { x: 0.17, y: 0.7, z: 0.28 },
          size: { x: 0.05, y: 0.05, z: 0.05 },
          mass: 0.25
        },
        {
          id: "near",
          center: { x: 0, y: 1.05, z: 1.2 },
          size: { x: 0.3, y: 0.3, z: 0.3 },
          mass: 0.2
        },
        {
          id: "occluded",
          center: { x: -1.2, y: 1.05, z: 2.4 },
          size: { x: 0.3, y: 0.3, z: 0.3 },
          mass: 0.2
        },
        {
          id: "behind",
          center: { x: 0, y: 1.05, z: -1.2 },
          size: { x: 0.3, y: 0.3, z: 0.3 },
          mass: 0.2
        }
      ]
    });
    try {
      const sensed = simulation.senseObjects(5);
      expect(Object.keys(sensed.objects)).toEqual(["manipulation-zone", "near"]);
      expect(sensed.sensor.position.z).toBeGreaterThan(
        simulation.snapshot().links.head_link.position.z
      );
      expect(sensed.sensor.position.y).toBeGreaterThan(
        simulation.snapshot().links.head_link.position.y + 0.4
      );
      expect(rotateVector(sensed.sensor.rotation, { x: 0, y: 0, z: 1 }).y)
        .toBeLessThan(-0.35);
      expect(sensed.sensor.maximumRange).toBe(5);
    } finally {
      await simulation.dispose();
    }
  }, 30_000);

  it("uses MuJoCo rays for solid occlusion and reports exact tactile solid identity", async () => {
    const simulation = await HumanoidSimulation.create({
      solids: [{
        id: "near-solid",
        center: { x: 0, y: 1.05, z: 1.2 },
        size: { x: 0.4, y: 0.4, z: 0.2 }
      }, {
        id: "far-solid",
        center: { x: 0, y: 0.85, z: 2.4 },
        size: { x: 0.4, y: 0.4, z: 0.2 }
      }, {
        id: "hand-probe",
        center: { x: 0.24, y: 0.68, z: 0.11 },
        size: { x: 0.08, y: 0.08, z: 0.08 }
      }]
    });
    try {
      expect(simulation.solidIds()).toEqual([
        "far-solid",
        "hand-probe",
        "near-solid"
      ]);
      expect(Object.keys(simulation.senseSolids(5).solids)).toEqual([
        "near-solid"
      ]);

      for (let index = 0; index < 2; index += 1) {
        await simulation.step(neutralHumanoidReference());
      }
      const contact = simulation.snapshot().contacts.find((candidate) => (
        (candidate.firstSolid === "hand-probe"
          && candidate.secondHandLink === "left_hand_palm_link")
        || (candidate.secondSolid === "hand-probe"
          && candidate.firstHandLink === "left_hand_palm_link")
      ));
      expect(contact).toMatchObject({
        normalForce: expect.any(Number)
      });
      expect(contact!.normalForce).toBeGreaterThan(0);
    } finally {
      await simulation.dispose();
    }
  }, 30_000);

  it("accepts a provider-neutral controller and round-trips its private state", async () => {
    const controller = new ContractController();
    const simulation = await HumanoidSimulation.create({
      controllerFactory: async () => controller
    });
    try {
      expect(simulation.snapshot().controller).toEqual(controller.descriptor);
      expect(simulation.snapshot().joints.left_hip_pitch_joint.effort).toBeUndefined();
      expect(simulation.captureState().requestedActuatorTorques).toBeUndefined();
      await simulation.step(neutralHumanoidReference());
      const captured = simulation.captureState();
      expect(captured.controller).toEqual({
        protocol: "humanoid-controller-state-v1",
        version: 1,
        implementation: "contract_test_pd",
        payload: { inference_count: 1 }
      });
      simulation.restoreState(captured);
      expect(controller.inferenceCount).toBe(1);

      controller.commandMode = "saturated";
      await simulation.step(neutralHumanoidReference());
      const saturated = simulation.snapshot().joints.left_hip_pitch_joint.effort;
      expect(saturated).toMatchObject({
        saturated: true,
        requestedUtilization: expect.any(Number),
        appliedUtilization: 1
      });
      expect(saturated!.requestedUtilization).toBeGreaterThan(1);
      const saturatedState = simulation.captureState();
      controller.commandMode = "valid";
      await simulation.step(neutralHumanoidReference());
      simulation.restoreState(saturatedState);
      expect(simulation.snapshot().joints.left_hip_pitch_joint.effort).toEqual(saturated);
    } finally {
      await simulation.dispose();
    }
    expect(controller.disposed).toBe(true);
  });

  it("rejects an invalid controller contract and releases the controller", async () => {
    const controller = new ContractController({
      controlStepSeconds: 0.019,
      physicsStepSeconds: 0.005
    });
    await expect(HumanoidSimulation.create({
      controllerFactory: async () => controller
    })).rejects.toThrow("invalid timing or actuation contract");
    expect(controller.disposed).toBe(true);
  });

  it("rejects malformed, non-finite, and negative controller commands", async () => {
    const controller = new ContractController();
    const simulation = await HumanoidSimulation.create({
      controllerFactory: async () => controller
    });
    try {
      controller.commandMode = "wrong_dimension";
      await expect(simulation.step(neutralHumanoidReference())).rejects.toThrow(
        "invalid actuation command"
      );

      controller.commandMode = "non_finite";
      await expect(simulation.step(neutralHumanoidReference())).rejects.toThrow(
        "non-finite or negative PD parameters"
      );

      controller.commandMode = "negative_gain";
      await expect(simulation.step(neutralHumanoidReference())).rejects.toThrow(
        "non-finite or negative PD parameters"
      );
    } finally {
      await simulation.dispose();
    }
  });

  it("rejects malformed task-space tracking weights before controller inference", async () => {
    const controller = new ContractController();
    const simulation = await HumanoidSimulation.create({
      controllerFactory: async () => controller
    });
    try {
      const neutral = neutralHumanoidReference();
      await expect(simulation.step({
        ...neutral,
        jointTrackingWeights: new Float64Array(HUMANOID_JOINT_NAMES.length - 1)
      })).rejects.toThrow("invalid joint count");

      const nonFinite = neutral.jointTrackingWeights.slice();
      nonFinite[0] = Number.NaN;
      await expect(simulation.step({
        ...neutral,
        jointTrackingWeights: nonFinite
      })).rejects.toThrow("invalid left_hip_pitch_joint tracking weight");

      const outOfRange = neutral.jointTrackingWeights.slice();
      outOfRange[0] = 1.01;
      await expect(simulation.step({
        ...neutral,
        jointTrackingWeights: outOfRange
      })).rejects.toThrow("invalid left_hip_pitch_joint tracking weight");
      expect(controller.inferenceCount).toBe(0);
    } finally {
      await simulation.dispose();
    }
  });
});

class ContractController implements HumanoidWholeBodyController {
  readonly descriptor: HumanoidControllerDescriptor;
  inferenceCount = 0;
  disposed = false;
  commandMode: "valid" | "wrong_dimension" | "non_finite" | "negative_gain"
    | "saturated" = "valid";

  constructor(timing: Partial<Pick<
    HumanoidControllerDescriptor,
    "controlStepSeconds" | "physicsStepSeconds"
  >> = {}) {
    this.descriptor = {
      protocol: "humanoid-controller-v1",
      implementation: "contract_test_pd",
      actuation: "joint_position_pd",
      controlStepSeconds: timing.controlStepSeconds ?? 0.02,
      physicsStepSeconds: timing.physicsStepSeconds ?? 0.005
    };
  }

  reset(_state: HumanoidPolicyState, _reference: ReturnType<typeof neutralHumanoidReference>): void {
    this.inferenceCount = 0;
  }

  async infer(
    _state: HumanoidPolicyState,
    reference: ReturnType<typeof neutralHumanoidReference>
  ): Promise<HumanoidJointPositionCommand> {
    this.inferenceCount += 1;
    const length = this.commandMode === "wrong_dimension"
      ? HUMANOID_JOINT_NAMES.length - 1
      : HUMANOID_JOINT_NAMES.length;
    const command: HumanoidJointPositionCommand = {
      kind: "joint_position_pd",
      positions: Float64Array.from({ length }, (_, index) => (
        reference.jointPositions[index] ?? 0
      )),
      stiffness: new Float64Array(length),
      damping: new Float64Array(length)
    };
    if (this.commandMode === "non_finite") command.positions[0] = Number.NaN;
    if (this.commandMode === "negative_gain") command.stiffness[0] = -1;
    if (this.commandMode === "saturated") {
      command.positions[0] = command.positions[0]! + 1;
      command.stiffness[0] = 1000;
    }
    return command;
  }

  advanceHistory(
    _state: HumanoidPolicyState,
    _reference: ReturnType<typeof neutralHumanoidReference>
  ): void {}

  captureState(): HumanoidControllerState {
    return {
      protocol: "humanoid-controller-state-v1",
      version: 1,
      implementation: this.descriptor.implementation,
      payload: { inference_count: this.inferenceCount }
    };
  }

  restoreState(state: HumanoidControllerState): void {
    if (state.protocol !== "humanoid-controller-state-v1"
      || state.version !== 1
      || state.implementation !== this.descriptor.implementation
      || !isRecord(state.payload)
      || !Number.isSafeInteger(state.payload.inference_count)) {
      throw new Error("Invalid contract test controller state");
    }
    this.inferenceCount = state.payload.inference_count as number;
  }

  async dispose(): Promise<void> {
    this.disposed = true;
  }
}

function isRecord(value: JsonValue): value is Record<string, JsonValue> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
