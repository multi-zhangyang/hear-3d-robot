import * as ort from "onnxruntime-web";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import {
  HUMANOID_JOINT_INDEX,
  HUMANOID_JOINT_NAMES,
  type HumanoidJointName,
  YAHMP_POLICY
} from "./model.js";
import {
  assertHumanoidReference,
  type HumanoidReference
} from "./reference.js";
import type {
  HumanoidControllerDescriptor,
  HumanoidControllerInferenceOptions,
  HumanoidControllerState,
  HumanoidJointPositionCommand,
  HumanoidPolicyState,
  HumanoidWholeBodyController
} from "./whole-body-controller.js";

const POLICY_PATH = fileURLToPath(
  new URL("../../../assets/humanoid/controllers/g1_yahmp.onnx", import.meta.url)
);

const EXPLICIT_TRACKING_STIFFNESS: Readonly<Record<
  "leg" | "ankle" | "waist" | "arm" | "wrist",
  number
>> = Object.freeze({
  leg: 100,
  ankle: 70,
  waist: 65,
  arm: 80,
  wrist: 40
});

const YahmpControllerStatePayloadSchema = z.object({
  previous_action: z.array(z.number().finite()).length(HUMANOID_JOINT_NAMES.length),
  history: z.array(
    z.array(z.number().finite()).length(
      YAHMP_POLICY.observationSize / (YAHMP_POLICY.historyLength + 1)
    )
  ).length(YAHMP_POLICY.historyLength)
}).strict();

export class YahmpController implements HumanoidWholeBodyController {
  readonly descriptor: HumanoidControllerDescriptor = {
    protocol: "humanoid-controller-v1",
    implementation: "yahmp_onnx",
    actuation: "joint_position_pd",
    controlStepSeconds: YAHMP_POLICY.controlDt,
    physicsStepSeconds: YAHMP_POLICY.physicsDt,
    commandResponseHorizonSeconds: 0.2,
    minimumEffectivePlanarSpeedMetersPerSecond: 0.15
  };
  readonly #session: ort.InferenceSession;
  #previousAction = new Float32Array(HUMANOID_JOINT_NAMES.length);
  #history: Float32Array[] = [];

  static async create(): Promise<YahmpController> {
    ort.env.wasm.numThreads = 1;
    const session = await ort.InferenceSession.create(await readFile(POLICY_PATH), {
      executionProviders: ["wasm"]
    });
    return new YahmpController(session);
  }

  private constructor(session: ort.InferenceSession) {
    this.#session = session;
  }

  reset(
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options: HumanoidControllerInferenceOptions = {}
  ): void {
    this.#previousAction.fill(0);
    const current = this.#observationBlock(state, reference, options);
    this.#history = Array.from(
      { length: YAHMP_POLICY.historyLength },
      () => current.slice()
    );
  }

  async infer(
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options: HumanoidControllerInferenceOptions = {}
  ): Promise<HumanoidJointPositionCommand> {
    if (this.#history.length === 0) this.reset(state, reference, options);
    const current = this.#observationBlock(state, reference, options);
    const observation = new Float32Array(YAHMP_POLICY.observationSize);
    observation.set(current, 0);
    let offset = current.length;
    for (const historical of this.#history) {
      observation.set(historical, offset);
      offset += historical.length;
    }
    const input = new ort.Tensor("float32", observation, [1, observation.length]);
    let result: Awaited<ReturnType<ort.InferenceSession["run"]>> | undefined;
    let action: Float32Array;
    try {
      result = await this.#session.run({ obs: input });
      const output = result.actions?.data;
      if (!(output instanceof Float32Array)
        || output.length !== HUMANOID_JOINT_NAMES.length) {
        throw new Error("YAHMP returned an invalid whole-body action tensor");
      }
      action = output.slice();
    } finally {
      for (const tensor of Object.values(result ?? {})) tensor.dispose();
      input.dispose();
    }
    this.#previousAction = action.slice();
    const stiffness = Float64Array.from(YAHMP_POLICY.stiffness, (value, index) => {
      const joint = HUMANOID_JOINT_NAMES[index];
      if (!joint) throw new Error(`Missing humanoid joint stiffness identity: ${index}`);
      const target = Math.max(value, explicitTrackingStiffness(joint));
      return value + (target - value) * reference.jointTrackingWeights[index]!;
    });
    return {
      kind: "joint_position_pd",
      positions: Float64Array.from(action, (value, index) => (
        reference.jointPositions[index]!
        + value * YAHMP_POLICY.actionScale[index]!
          * (1 - reference.jointTrackingWeights[index]!)
      )),
      stiffness,
      damping: Float64Array.from(YAHMP_POLICY.damping, (value, index) => (
        value * Math.sqrt(stiffness[index]! / YAHMP_POLICY.stiffness[index]!)
      ))
    };
  }

  advanceHistory(
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options: HumanoidControllerInferenceOptions = {}
  ): void {
    if (this.#history.length === 0) {
      this.reset(state, reference, options);
      return;
    }
    this.#history.shift();
    this.#history.push(this.#observationBlock(state, reference, options));
  }

  captureState(): HumanoidControllerState {
    return {
      protocol: "humanoid-controller-state-v1",
      version: 1,
      implementation: "yahmp_onnx",
      payload: {
        previous_action: [...this.#previousAction],
        history: this.#history.map((entry) => [...entry])
      }
    };
  }

  restoreState(state: HumanoidControllerState): void {
    if (state.protocol !== "humanoid-controller-state-v1"
      || state.version !== 1
      || state.implementation !== "yahmp_onnx") {
      throw new Error("Invalid YAHMP controller state");
    }
    const payload = YahmpControllerStatePayloadSchema.safeParse(state.payload);
    if (!payload.success) throw new Error("Invalid YAHMP controller state");
    this.#previousAction = Float32Array.from(payload.data.previous_action);
    this.#history = payload.data.history.map((entry) => Float32Array.from(entry));
  }

  async dispose(): Promise<void> {
    await this.#session.release();
  }

  #observationBlock(
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    options: HumanoidControllerInferenceOptions
  ): Float32Array {
    assertHumanoidReference(reference);
    if (state.jointPositions.length !== HUMANOID_JOINT_NAMES.length
      || state.jointVelocities.length !== HUMANOID_JOINT_NAMES.length) {
      throw new Error("Humanoid policy state has an invalid joint count");
    }
    const projectedGravity = inverseRotate(state.rootQuaternion, [0, 0, -1]);
    const neutralTrackedCommand = options.trackedJointPolicyCommand === "neutral";
    const policyReferencePositions = Array.from(
      reference.jointPositions,
      (value, index) => mix(
        value,
        neutralTrackedCommand
          ? YAHMP_POLICY.defaultJointPositions[index]!
          : state.jointPositions[index]!,
        reference.jointTrackingWeights[index]!
      )
    );
    const policyReferenceVelocities = Array.from(
      reference.jointVelocities,
      (value, index) => mix(
        value,
        neutralTrackedCommand ? 0 : state.jointVelocities[index]!,
        reference.jointTrackingWeights[index]!
      )
    );
    return Float32Array.from([
      ...policyReferencePositions,
      ...policyReferenceVelocities,
      reference.rootVelocity[0],
      reference.rootVelocity[1],
      reference.rootYawVelocity,
      reference.rootHeight,
      reference.rootRoll,
      reference.rootPitch,
      ...state.rootAngularVelocity,
      ...projectedGravity,
      ...Array.from(state.jointPositions, (value, index) => (
        value - YAHMP_POLICY.defaultJointPositions[index]!
      )),
      ...Array.from(state.jointVelocities),
      ...this.#previousAction
    ]);
  }
}

function mix(start: number, end: number, amount: number): number {
  return start + (end - start) * amount;
}

function explicitTrackingStiffness(joint: HumanoidJointName): number {
  const index = HUMANOID_JOINT_INDEX.get(joint);
  if (index === undefined) throw new Error(`Unknown humanoid tracking joint: ${joint}`);
  if (joint.includes("ankle")) return EXPLICIT_TRACKING_STIFFNESS.ankle;
  if (joint.startsWith("left_hip_")
    || joint.startsWith("right_hip_")
    || joint.includes("knee")) {
    return EXPLICIT_TRACKING_STIFFNESS.leg;
  }
  if (joint.startsWith("waist_")) return EXPLICIT_TRACKING_STIFFNESS.waist;
  if (joint.includes("wrist")) return EXPLICIT_TRACKING_STIFFNESS.wrist;
  return EXPLICIT_TRACKING_STIFFNESS.arm;
}

function inverseRotate(
  quaternion: HumanoidPolicyState["rootQuaternion"],
  vector: readonly [number, number, number]
): [number, number, number] {
  const [w, x, y, z] = quaternion;
  const tx = 2 * (y * vector[2] - z * vector[1]);
  const ty = 2 * (z * vector[0] - x * vector[2]);
  const tz = 2 * (x * vector[1] - y * vector[0]);
  return [
    vector[0] - w * tx + (y * tz - z * ty),
    vector[1] - w * ty + (z * tx - x * tz),
    vector[2] - w * tz + (x * ty - y * tx)
  ];
}
