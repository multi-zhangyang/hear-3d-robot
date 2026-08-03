import * as ort from "onnxruntime-web";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import {
  HUMANOID_JOINT_NAMES,
  YAHMP_POLICY
} from "./model.js";
import type { HumanoidReference } from "./reference.js";
import type {
  HumanoidControllerDescriptor,
  HumanoidControllerState,
  HumanoidJointPositionCommand,
  HumanoidPolicyState,
  HumanoidWholeBodyController
} from "./whole-body-controller.js";

const POLICY_PATH = fileURLToPath(
  new URL("../../../assets/humanoid/controllers/g1_yahmp.onnx", import.meta.url)
);

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
    physicsStepSeconds: YAHMP_POLICY.physicsDt
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

  reset(state: HumanoidPolicyState, reference: HumanoidReference): void {
    this.#previousAction.fill(0);
    const current = this.#observationBlock(state, reference);
    this.#history = Array.from(
      { length: YAHMP_POLICY.historyLength },
      () => current.slice()
    );
  }

  async infer(
    state: HumanoidPolicyState,
    reference: HumanoidReference
  ): Promise<HumanoidJointPositionCommand> {
    if (this.#history.length === 0) this.reset(state, reference);
    const current = this.#observationBlock(state, reference);
    const observation = new Float32Array(YAHMP_POLICY.observationSize);
    observation.set(current, 0);
    let offset = current.length;
    for (const historical of this.#history) {
      observation.set(historical, offset);
      offset += historical.length;
    }
    const result = await this.#session.run({
      obs: new ort.Tensor("float32", observation, [1, observation.length])
    });
    const action = result.actions?.data;
    if (!(action instanceof Float32Array) || action.length !== HUMANOID_JOINT_NAMES.length) {
      throw new Error("YAHMP returned an invalid whole-body action tensor");
    }
    this.#previousAction = action.slice();
    return {
      kind: "joint_position_pd",
      positions: Float64Array.from(action, (value, index) => (
        reference.jointPositions[index]! + value * YAHMP_POLICY.actionScale[index]!
      )),
      stiffness: Float64Array.from(YAHMP_POLICY.stiffness),
      damping: Float64Array.from(YAHMP_POLICY.damping)
    };
  }

  advanceHistory(state: HumanoidPolicyState, reference: HumanoidReference): void {
    if (this.#history.length === 0) {
      this.reset(state, reference);
      return;
    }
    this.#history.shift();
    this.#history.push(this.#observationBlock(state, reference));
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
    reference: HumanoidReference
  ): Float32Array {
    if (state.jointPositions.length !== HUMANOID_JOINT_NAMES.length
      || state.jointVelocities.length !== HUMANOID_JOINT_NAMES.length) {
      throw new Error("Humanoid policy state has an invalid joint count");
    }
    const projectedGravity = inverseRotate(state.rootQuaternion, [0, 0, -1]);
    return Float32Array.from([
      ...reference.jointPositions,
      ...reference.jointVelocities,
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
