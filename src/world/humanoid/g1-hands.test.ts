import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import {
  G1_HAND_JOINT_NAMES,
  G1_HAND_CONTACT_SURFACE_NAMES,
  G1_HAND_LINK_NAMES,
  G1_MORPHOLOGY
} from "./morphology.js";
import { HUMANOID_JOINT_NAMES, YAHMP_POLICY } from "./model.js";
import {
  inverseQuaternion,
  multiplyQuaternion,
  quaternionAngularDistance
} from "../geometry.js";
import { neutralHumanoidReference, type HumanoidReference } from "./reference.js";
import { HumanoidSimulation } from "./simulation.js";
import { g1HandObjectContacts } from "./hand-contact-evidence.js";
import { HumanoidSimulationStateCheckpointSchema } from "./checkpoint.js";
import { HumanoidWorldSnapshotSchema } from "./snapshot-schema.js";
import { G1HandActuator, guardClosingTarget } from "./hand-actuator.js";
import {
  g1HandContactGeomName
} from "./hand-collision-geometry.js";
import {
  humanoidModelPath,
  loadHumanoidMujoco
} from "./mujoco-runtime.js";
import { resolveMujocoActuatedJoints } from "./mujoco-joints.js";
import type {
  HumanoidControllerState,
  HumanoidJointPositionCommand,
  HumanoidPolicyState,
  HumanoidWholeBodyController
} from "./whole-body-controller.js";

describe("Unitree G1 43DoF hands", () => {
  it("bounds stored closing energy without slowing release for either hand", () => {
    expect(guardClosingTarget(-0.10, -1.20, 0.25)).toBeCloseTo(-0.35);
    expect(guardClosingTarget(0.10, 1.20, 0.25)).toBeCloseTo(0.35);
    expect(guardClosingTarget(-0.60, -0.20, 0.25)).toBeCloseTo(-0.20);
    expect(guardClosingTarget(0.60, 0.20, 0.25)).toBeCloseTo(0.20);
    expect(guardClosingTarget(-0.60, 0, 0.25)).toBe(0);
    expect(guardClosingTarget(0.60, 0, 0.25)).toBe(0);
    expect(() => guardClosingTarget(0, 1, 0)).toThrow(/must be positive/);
  });

  it("maps interleaved body and hand actuators by joint identity without overlap", async () => {
    const runtime = await loadHumanoidMujoco();
    const model = runtime.MjModel.from_xml_path(humanoidModelPath());
    const data = new runtime.MjData(model);
    try {
      const body = resolveMujocoActuatedJoints(runtime, model, HUMANOID_JOINT_NAMES);
      const hands = new G1HandActuator(runtime, model, data).bindings();
      const bodyActuators = new Set(body.map((binding) => binding.actuatorId));
      const handActuators = new Set(hands.map((binding) => binding.actuatorId));
      expect(body).toHaveLength(29);
      expect(hands).toHaveLength(14);
      expect(new Set([...bodyActuators, ...handActuators]).size).toBe(43);
      expect([...handActuators].some((id) => id < Math.max(...bodyActuators))).toBe(true);
      expect([...handActuators].every((id) => !bodyActuators.has(id))).toBe(true);
      expect(model).toMatchObject({ nq: 50, nv: 49, nu: 43, njnt: 44, nbody: 45 });
    } finally {
      data.delete();
      model.delete();
    }
  });

  it("loads the pinned upstream model and keeps YAHMP's body contract at 29 joints", async () => {
    const source = await readFile(new URL(
      "../../../assets/humanoid/g1/g1_with_hands.xml",
      import.meta.url
    ));
    expect(createHash("sha256").update(source).digest("hex")).toBe(
      "6d80af81ab8153f18158b190ab52fbfb880418b295c3bcb1827040896246057a"
    );
    const controller = new RecordingBodyController();
    const simulation = await HumanoidSimulation.create({
      controllerFactory: async () => controller
    });
    try {
      const initial = simulation.snapshot();
      expect(initial.morphology).toEqual(G1_MORPHOLOGY);
      expect(Object.keys(initial.joints)).toEqual(HUMANOID_JOINT_NAMES);
      expect(Object.keys(initial.hands.joints)).toEqual(G1_HAND_JOINT_NAMES);
      expect(Object.keys(initial.hands.links)).toEqual(G1_HAND_LINK_NAMES);
      const handSurfaces = simulation.handSurfaceObservations(initial);
      expect(handSurfaces.map((surface) => surface.handSurface)).toEqual(
        G1_HAND_CONTACT_SURFACE_NAMES
      );
      const leftPalm = handSurfaces.find((surface) => (
        surface.handSurface === "left_hand_palm_link"
      ));
      expect(leftPalm).toMatchObject({
        hand: "left",
        wristWorldPosition: initial.links.left_wrist_yaw_link.position
      });
      expect(Math.hypot(
        leftPalm!.surfaceFromWristWorld.x,
        leftPalm!.surfaceFromWristWorld.y,
        leftPalm!.surfaceFromWristWorld.z
      )).toBeGreaterThan(0.04);
      const runtime = await loadHumanoidMujoco();
      const model = runtime.MjModel.from_xml_path(humanoidModelPath());
      try {
        const geometryId = runtime.mj_name2id(
          model,
          runtime.mjtObj.mjOBJ_GEOM.value,
          g1HandContactGeomName("left_hand_palm_link")
        );
        const offset = geometryId * 4;
        const expectedLocalRotation = {
          x: model.geom_quat[offset + 2]!,
          y: model.geom_quat[offset + 3]!,
          z: model.geom_quat[offset + 1]!,
          w: model.geom_quat[offset]!
        };
        const observedLocalRotation = multiplyQuaternion(
          inverseQuaternion(initial.links.left_wrist_yaw_link.rotation),
          leftPalm!.worldRotation
        );
        expect(quaternionAngularDistance(
          observedLocalRotation,
          expectedLocalRotation
        )).toBeCloseTo(0, 8);
        expect(quaternionAngularDistance(
          leftPalm!.worldRotation,
          initial.links.left_wrist_yaw_link.rotation
        )).toBeGreaterThan(1);
      } finally {
        model.delete();
      }
      expect(initial.hands.controller).toEqual({
        protocol: "g1-hand-controller-v1",
        implementation: "mujoco_continuous_position_pd",
        actuation: "joint_position_pd",
        jointCount: 14
      });
      expect(Object.values(initial.hands.joints).every((joint) => (
        Math.abs(joint.stiffnessNewtonMetersPerRadian - 2.5) < 1e-9
          && Math.abs(joint.dampingNewtonMeterSecondsPerRadian - 0.3) < 1e-9
      ))).toBe(true);
      expect(controller.policyJointCounts).toEqual([29]);
      await simulation.step(neutralHumanoidReference());
      expect(controller.policyJointCounts).toEqual([29, 29, 29]);
      expect(simulation.captureState()).toMatchObject({
        positions: { length: 50 },
        velocities: { length: 49 },
        controls: { length: 43 }
      });
    } finally {
      await simulation.dispose();
    }
  });

  it("tracks arbitrary continuous joint targets and restores targets with physical state", async () => {
    const simulation = await HumanoidSimulation.create({
      controllerFactory: async () => new RecordingBodyController()
    });
    try {
      const baseline = simulation.snapshot();
      const baselinePosition = baseline.hands.joints.left_hand_index_0_joint.position;
      const commanded = simulation.setHandJointTargets({
        left_hand_index_0_joint: -0.63,
        left_hand_middle_1_joint: -0.81,
        right_hand_thumb_2_joint: -0.72
      });
      expect(commanded.hands.joints.left_hand_index_0_joint).toMatchObject({
        position: baselinePosition,
        target: -0.63
      });

      await simulation.step(neutralHumanoidReference());
      const firstStep = simulation.snapshot();
      expect(firstStep.hands.joints.left_hand_index_0_joint.position)
        .toBeLessThan(baselinePosition);
      expect(firstStep.hands.joints.left_hand_index_0_joint.position).not.toBe(-0.63);
      for (let index = 0; index < 52; index += 1) {
        await simulation.step(neutralHumanoidReference());
      }
      const captured = simulation.captureState();
      const checkpoint = HumanoidSimulationStateCheckpointSchema.parse({
        time: captured.time,
        positions: [...captured.positions],
        velocities: [...captured.velocities],
        controls: [...captured.controls],
        activations: [...captured.activations],
        accelerationWarmstart: [...captured.accelerationWarmstart],
        ...(captured.requestedActuatorTorques
          ? { requestedActuatorTorques: [...captured.requestedActuatorTorques] }
          : {}),
        controller: captured.controller
      });
      const capturedHand = simulation.snapshot().hands.joints.left_hand_index_0_joint;
      // The contact-policy hand is intentionally compliant (kp=2.5, kv=0.3),
      // so checkpointing must not assume the near-instantaneous kp=500
      // response that the hand actuators inherited before deployment parity.
      expect(Math.abs(capturedHand.position - capturedHand.target))
        .toBeLessThan(Math.abs(baselinePosition - capturedHand.target) / 2);

      simulation.setHandJointTargets({ left_hand_index_0_joint: -0.05 });
      for (let index = 0; index < 4; index += 1) {
        await simulation.step(neutralHumanoidReference());
      }
      expect(simulation.snapshot().hands.joints.left_hand_index_0_joint.target).toBe(-0.05);
      simulation.restoreState({
        time: checkpoint.time,
        positions: Float64Array.from(checkpoint.positions),
        velocities: Float64Array.from(checkpoint.velocities),
        controls: Float64Array.from(checkpoint.controls),
        activations: Float64Array.from(checkpoint.activations),
        accelerationWarmstart: Float64Array.from(checkpoint.accelerationWarmstart),
        ...(checkpoint.requestedActuatorTorques
          ? {
              requestedActuatorTorques: Float64Array.from(
                checkpoint.requestedActuatorTorques
              )
            }
          : {}),
        controller: checkpoint.controller
      });
      expect(simulation.snapshot().hands.joints.left_hand_index_0_joint).toMatchObject({
        position: capturedHand.position,
        velocity: capturedHand.velocity,
        target: -0.63
      });

      simulation.restoreState({
        ...simulation.captureState(),
        requestedActuatorTorques: new Float64Array(0)
      });
      expect(simulation.captureState().requestedActuatorTorques).toBeUndefined();
      const beforeMalformedEvidence = simulation.captureState();
      expect(() => simulation.restoreState({
        ...beforeMalformedEvidence,
        requestedActuatorTorques: new Float64Array(28)
      })).toThrow(/all 29 body joints/);
      expect(simulation.captureState()).toEqual(beforeMalformedEvidence);

      const beforeRejected = simulation.snapshot().hands.joints.left_hand_index_0_joint.target;
      expect(() => simulation.setHandJointTargets({
        left_hand_index_0_joint: -0.2,
        right_hand_index_0_joint: -0.1
      })).toThrow(/exceeds right_hand_index_0_joint limits/);
      expect(simulation.snapshot().hands.joints.left_hand_index_0_joint.target)
        .toBe(beforeRejected);
    } finally {
      await simulation.dispose();
    }
  });

  it("reports hand-object contact from MuJoCo instead of inferring a grasp", async () => {
    const probe = await HumanoidSimulation.create({
      controllerFactory: async () => new RecordingBodyController()
    });
    const contactCenter = probe.snapshot().hands.links.left_hand_index_1_link.position;
    await probe.dispose();

    const simulation = await HumanoidSimulation.create({
      controllerFactory: async () => new RecordingBodyController(),
      objects: [{
        id: "contact-probe",
        center: contactCenter,
        size: { x: 0.025, y: 0.025, z: 0.025 },
        mass: 0.08
      }]
    });
    try {
      await simulation.step(neutralHumanoidReference());
      const contacts = g1HandObjectContacts(
        simulation.snapshot().contacts,
        "contact-probe"
      );
      expect(contacts.length).toBeGreaterThan(0);
      expect(contacts).toEqual(expect.arrayContaining([
        expect.objectContaining({
          handLink: expect.stringMatching(/^left_hand_/),
          kind: "object",
          objectId: "contact-probe",
          normalForce: expect.any(Number)
        })
      ]));
      expect(contacts.every((contact) => contact.normalForce >= 0)).toBe(true);
    } finally {
      await simulation.dispose();
    }
  });

  it("distinguishes the palm geom from another collider on the same wrist body", async () => {
    const centers = await leftPalmAndWristColliderCenters();
    const simulation = await HumanoidSimulation.create({
      controllerFactory: async () => new RecordingBodyController(),
      objects: [
        {
          id: "palm-probe",
          center: centers.palm,
          size: { x: 0.012, y: 0.012, z: 0.012 },
          mass: 0.01
        },
        {
          id: "wrist-probe",
          center: centers.wrist,
          size: { x: 0.012, y: 0.012, z: 0.012 },
          mass: 0.01
        }
      ]
    });
    try {
      const snapshot = simulation.snapshot();
      const parsed = HumanoidWorldSnapshotSchema.parse({
        frame: 0,
        worldRevision: 0,
        robot: snapshot,
        grasp: {
          contractSha256: "fc1e2d113bb5e5f5f8a75f0faa3efc8bd97ecc18eb41463da09d26bb52cfc193",
          assessments: []
        },
        navigation: {
          planId: null,
          status: "idle",
          target: null,
          waypoints: [],
          waypointIndex: null
        }
      });
      expect(parsed.robot.contacts.some((contact) => (
        contact.firstHandLink === "left_hand_palm_link"
          || contact.secondHandLink === "left_hand_palm_link"
      ))).toBe(true);
      expect(g1HandObjectContacts(snapshot.contacts, "palm-probe")).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            handLink: "left_hand_palm_link",
            hand: "left",
            objectId: "palm-probe"
          })
        ])
      );
      const wristContacts = snapshot.contacts.filter((contact) => (
        (contact.firstObject === "wrist-probe"
            && contact.secondBody === "left_wrist_yaw_link")
          || (contact.secondObject === "wrist-probe"
            && contact.firstBody === "left_wrist_yaw_link")
      ));
      expect(wristContacts.length).toBeGreaterThan(0);
      expect(wristContacts.every((contact) => (
        contact.firstHandLink === null && contact.secondHandLink === null
      ))).toBe(true);
      expect(g1HandObjectContacts(snapshot.contacts, "wrist-probe")).toEqual([]);
    } finally {
      await simulation.dispose();
    }
  });
}, 30_000);

async function leftPalmAndWristColliderCenters(): Promise<{
  palm: { x: number; y: number; z: number };
  wrist: { x: number; y: number; z: number };
}> {
  const runtime = await loadHumanoidMujoco();
  const model = runtime.MjModel.from_xml_path(humanoidModelPath());
  const data = new runtime.MjData(model);
  try {
    data.qpos.fill(0);
    data.qpos[2] = 0.793;
    data.qpos[3] = 1;
    resolveMujocoActuatedJoints(runtime, model, HUMANOID_JOINT_NAMES)
      .forEach((binding, index) => {
        data.qpos[binding.positionAddress] = YAHMP_POLICY.defaultJointPositions[index]!;
      });
    runtime.mj_forward(model, data);
    const palmGeometryId = runtime.mj_name2id(
      model,
      runtime.mjtObj.mjOBJ_GEOM.value,
      g1HandContactGeomName("left_hand_palm_link")
    );
    const wristBodyId = runtime.mj_name2id(
      model,
      runtime.mjtObj.mjOBJ_BODY.value,
      "left_wrist_yaw_link"
    );
    const wristGeometryId = Array.from({ length: model.ngeom }, (_, index) => index)
      .find((index) => (
        model.geom_bodyid[index] === wristBodyId && index !== palmGeometryId
      ));
    if (palmGeometryId < 0 || wristBodyId < 0 || wristGeometryId === undefined) {
      throw new Error("G1 palm/wrist collision provenance is incomplete");
    }
    return {
      palm: geometryCenter(data.geom_xpos, palmGeometryId),
      wrist: geometryCenter(data.geom_xpos, wristGeometryId)
    };
  } finally {
    data.delete();
    model.delete();
  }
}

function geometryCenter(
  positions: ArrayLike<number>,
  geometryId: number
): { x: number; y: number; z: number } {
  const offset = geometryId * 3;
  return {
    x: positions[offset + 1]!,
    y: positions[offset + 2]!,
    z: positions[offset]!
  };
}

class RecordingBodyController implements HumanoidWholeBodyController {
  readonly descriptor = {
    protocol: "humanoid-controller-v1",
    implementation: "g1_body_mapping_test",
    actuation: "joint_position_pd",
    controlStepSeconds: 0.02,
    physicsStepSeconds: 0.005
  } as const;

  readonly policyJointCounts: number[] = [];

  reset(state: HumanoidPolicyState, _reference: HumanoidReference): void {
    this.policyJointCounts.push(state.jointPositions.length);
  }

  async infer(
    state: HumanoidPolicyState,
    reference: HumanoidReference
  ): Promise<HumanoidJointPositionCommand> {
    this.policyJointCounts.push(state.jointPositions.length);
    return {
      kind: "joint_position_pd",
      positions: Float64Array.from(reference.jointPositions),
      stiffness: Float64Array.from(YAHMP_POLICY.stiffness),
      damping: Float64Array.from(YAHMP_POLICY.damping)
    };
  }

  advanceHistory(state: HumanoidPolicyState, _reference: HumanoidReference): void {
    this.policyJointCounts.push(state.jointPositions.length);
  }

  captureState(): HumanoidControllerState {
    return {
      protocol: "humanoid-controller-state-v1",
      version: 1,
      implementation: this.descriptor.implementation,
      payload: { policy_joint_count: HUMANOID_JOINT_NAMES.length }
    };
  }

  restoreState(state: HumanoidControllerState): void {
    if (state.implementation !== this.descriptor.implementation) {
      throw new Error("G1 body mapping test controller state mismatch");
    }
  }

  async dispose(): Promise<void> {}
}
