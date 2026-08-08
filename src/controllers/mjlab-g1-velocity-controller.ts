import { createHash } from "node:crypto";
import * as ort from "onnxruntime-web";
import { z } from "zod";
import {
  HUMANOID_JOINT_NAMES,
  type HumanoidJointName
} from "../world/humanoid/model.js";
import { assertHumanoidReference } from "../world/humanoid/reference.js";
import type {
  HumanoidReference
} from "../world/humanoid/reference.js";
import type {
  HumanoidControllerModuleAsset,
  HumanoidControllerModuleContext
} from "../world/humanoid/controller-module.js";
import type {
  HumanoidControllerDescriptor,
  HumanoidControllerInferenceOptions,
  HumanoidControllerState,
  HumanoidJointPositionCommand,
  HumanoidPolicyState,
  HumanoidWholeBodyController
} from "../world/humanoid/whole-body-controller.js";

const POLICY_ASSET_ID = "policy";
const REPORT_ASSET_ID = "training_report";
const OBSERVATION_SIZE = 99;
const CONTROL_STEP_SECONDS = 0.02;
const PHYSICS_STEP_SECONDS = 0.005;
const TASK_TRACKING_VELOCITY_LEAD_SECONDS = 0.08;

const TrainingArtifactSchema = z.object({
  file: z.string().min(1),
  bytes: z.number().int().positive(),
  sha256: z.string().regex(/^[a-f0-9]{64}$/)
}).strict();

const TrainingReportSchema = z.object({
  version: z.literal(1),
  created_at: z.iso.datetime({ offset: true }),
  framework: z.literal("mjlab"),
  framework_version: z.string().min(1),
  task: z.literal("Mjlab-Velocity-Flat-Unitree-G1"),
  seed: z.number().int().nonnegative(),
  training: z.object({
    iterations_requested: z.number().int().positive(),
    checkpoint_iteration: z.number().int().nonnegative(),
    environment_count: z.number().int().positive(),
    accelerator: z.string().min(1),
    cuda_version: z.string().min(1)
  }).strict(),
  evaluation: z.object({
    environment_count: z.number().int().positive(),
    physics_steps: z.number().int().positive(),
    mean_episode_reward_sum: z.number().finite(),
    mean_planar_displacement_m: z.number().finite().nonnegative(),
    maximum_planar_displacement_m: z.number().finite().nonnegative(),
    minimum_root_height_m: z.number().finite(),
    termination_count: z.number().int().nonnegative(),
    maximum_absolute_action: z.number().finite().nonnegative()
  }).strict(),
  checkpoint: TrainingArtifactSchema,
  onnx: TrainingArtifactSchema.extend({
    inputs: z.tuple([z.literal("obs")]),
    outputs: z.tuple([z.literal("actions")]),
    metadata: z.object({
      run_path: z.string().min(1),
      joint_names: z.string().min(1),
      joint_stiffness: z.string().min(1),
      joint_damping: z.string().min(1),
      default_joint_pos: z.string().min(1),
      command_names: z.literal("twist"),
      observation_names: z.literal(
        "base_lin_vel,base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command"
      ),
      observation_terms_scale: z.string().min(1),
      observation_terms_flatten_history_dim: z.string().min(1),
      observation_terms_history_length: z.string().min(1),
      observation_terms_clip: z.string().min(1),
      action_scale: z.string().min(1)
    }).strict()
  }).strict()
}).strict();

const ControllerStatePayloadSchema = z.object({
  policy_sha256: z.string().regex(/^[a-f0-9]{64}$/),
  previous_action: z.array(z.number().finite()).length(HUMANOID_JOINT_NAMES.length)
}).strict();

interface MjlabG1VelocityPolicy {
  readonly policySha256: string;
  readonly defaultJointPositions: readonly number[];
  readonly actionScale: readonly number[];
  readonly stiffness: readonly number[];
  readonly damping: readonly number[];
}

export async function createMjlabG1VelocityController(
  context: HumanoidControllerModuleContext
): Promise<MjlabG1VelocityController> {
  const policyAsset = requiredAsset(context, POLICY_ASSET_ID);
  const reportAsset = requiredAsset(context, REPORT_ASSET_ID);
  const policy = parseTrainingBundle(policyAsset, reportAsset);
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(policyAsset.bytes, {
    executionProviders: ["wasm"]
  });
  try {
    assertSessionContract(session);
    return new MjlabG1VelocityController(session, policy);
  } catch (error) {
    await session.release();
    throw error;
  }
}

export class MjlabG1VelocityController implements HumanoidWholeBodyController {
  readonly descriptor: HumanoidControllerDescriptor = {
    protocol: "humanoid-controller-v1",
    implementation: "mjlab_g1_velocity_onnx",
    actuation: "joint_position_pd",
    controlStepSeconds: CONTROL_STEP_SECONDS,
    physicsStepSeconds: PHYSICS_STEP_SECONDS,
    commandResponseHorizonSeconds: 0.2,
    minimumEffectivePlanarSpeedMetersPerSecond: 0.15,
    learnedPolicy: {
      protocol: "humanoid-learned-policy-v1",
      runtime: "onnxruntime-web/wasm",
      observationSpace: {
        protocol: "mjlab-g1-velocity-observation-v1",
        size: OBSERVATION_SIZE
      },
      actionSpace: {
        protocol: "mjlab-g1-joint-position-residual-v1",
        size: HUMANOID_JOINT_NAMES.length
      },
      observationFeatures: [
        "proprioception",
        "root_kinematics"
      ],
      capabilities: ["balance", "locomotion"]
    }
  };
  readonly #session: ort.InferenceSession;
  readonly #policy: MjlabG1VelocityPolicy;
  #previousAction = new Float32Array(HUMANOID_JOINT_NAMES.length);

  constructor(session: ort.InferenceSession, policy: MjlabG1VelocityPolicy) {
    this.#session = session;
    this.#policy = policy;
  }

  reset(state: HumanoidPolicyState, reference: HumanoidReference): void {
    assertPolicyInput(state, reference);
    this.#previousAction.fill(0);
  }

  async infer(
    state: HumanoidPolicyState,
    reference: HumanoidReference,
    _options: HumanoidControllerInferenceOptions = {}
  ): Promise<HumanoidJointPositionCommand> {
    assertPolicyInput(state, reference);
    const observation = this.#observation(state, reference);
    const input = new ort.Tensor("float32", observation, [1, OBSERVATION_SIZE]);
    let result: Awaited<ReturnType<ort.InferenceSession["run"]>> | undefined;
    try {
      result = await this.#session.run({ obs: input });
      const output = result.actions;
      if (!output
        || output.type !== "float32"
        || output.dims.length !== 2
        || output.dims[0] !== 1
        || output.dims[1] !== HUMANOID_JOINT_NAMES.length
        || !(output.data instanceof Float32Array)
        || output.data.length !== HUMANOID_JOINT_NAMES.length
        || !output.data.every(Number.isFinite)) {
        throw new Error("mjlab G1 policy returned an invalid action tensor");
      }
      this.#previousAction = output.data.slice();
    } finally {
      for (const tensor of Object.values(result ?? {})) tensor.dispose();
      input.dispose();
    }
    return this.#command(reference);
  }

  advanceHistory(): void {}

  captureState(): HumanoidControllerState {
    return {
      protocol: "humanoid-controller-state-v1",
      version: 1,
      implementation: this.descriptor.implementation,
      payload: {
        policy_sha256: this.#policy.policySha256,
        previous_action: [...this.#previousAction]
      }
    };
  }

  restoreState(state: HumanoidControllerState): void {
    if (state.protocol !== "humanoid-controller-state-v1"
      || state.version !== 1
      || state.implementation !== this.descriptor.implementation) {
      throw new Error("Invalid mjlab G1 controller state");
    }
    const payload = ControllerStatePayloadSchema.safeParse(state.payload);
    if (!payload.success
      || payload.data.policy_sha256 !== this.#policy.policySha256) {
      throw new Error("Invalid mjlab G1 controller state");
    }
    this.#previousAction = Float32Array.from(payload.data.previous_action);
  }

  async dispose(): Promise<void> {
    await this.#session.release();
  }

  #observation(
    state: HumanoidPolicyState,
    reference: HumanoidReference
  ): Float32Array {
    const environment = state.environment!;
    const projectedGravity = inverseRotate(state.rootQuaternion, [0, 0, -1]);
    const observation = Float32Array.from([
      ...environment.rootLinearVelocity,
      ...environment.rootAngularVelocity,
      ...projectedGravity,
      ...Array.from(state.jointPositions, (value, index) => (
        value - this.#policy.defaultJointPositions[index]!
      )),
      ...Array.from(state.jointVelocities),
      ...this.#previousAction,
      reference.rootVelocity[0],
      reference.rootVelocity[1],
      reference.rootYawVelocity
    ]);
    if (observation.length !== OBSERVATION_SIZE
      || !observation.every(Number.isFinite)) {
      throw new Error("mjlab G1 policy observation is invalid");
    }
    return observation;
  }

  #command(reference: HumanoidReference): HumanoidJointPositionCommand {
    const positions = Float64Array.from(this.#previousAction, (action, index) => {
      const defaultPosition = this.#policy.defaultJointPositions[index]!;
      const weight = reference.jointTrackingWeights[index]!;
      const residualAuthority = taskTrackingResidualAuthority(
        HUMANOID_JOINT_NAMES[index]!
      );
      return defaultPosition
        + action * this.#policy.actionScale[index]!
          * (1 - weight * (1 - residualAuthority))
        + weight * (
          reference.jointPositions[index]! - defaultPosition
          + reference.jointVelocities[index]!
            * TASK_TRACKING_VELOCITY_LEAD_SECONDS
        );
    });
    const stiffness = Float64Array.from(this.#policy.stiffness, (value, index) => {
      const joint = HUMANOID_JOINT_NAMES[index]!;
      const target = Math.max(value, taskTrackingStiffness(joint));
      return value + (target - value) * reference.jointTrackingWeights[index]!;
    });
    const damping = Float64Array.from(this.#policy.damping, (value, index) => (
      value * Math.sqrt(stiffness[index]! / this.#policy.stiffness[index]!)
    ));
    if (![...positions, ...stiffness, ...damping].every(Number.isFinite)) {
      throw new Error("mjlab G1 policy produced a non-finite joint command");
    }
    return { kind: "joint_position_pd", positions, stiffness, damping };
  }
}

function parseTrainingBundle(
  policyAsset: HumanoidControllerModuleAsset,
  reportAsset: HumanoidControllerModuleAsset
): MjlabG1VelocityPolicy {
  let reportJson: unknown;
  try {
    reportJson = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(
      reportAsset.bytes
    ));
  } catch (error) {
    throw new Error("mjlab G1 training report is not valid UTF-8 JSON", {
      cause: error
    });
  }
  const parsed = TrainingReportSchema.safeParse(reportJson);
  if (!parsed.success) {
    throw new Error("mjlab G1 training report does not match version 1");
  }
  const report = parsed.data;
  if (report.onnx.file !== "g1_velocity.onnx"
    || report.onnx.bytes !== policyAsset.bytes.byteLength
    || report.onnx.sha256 !== policyAsset.sha256
    || sha256(policyAsset.bytes) !== policyAsset.sha256) {
    throw new Error("mjlab G1 ONNX does not match its training report");
  }
  const metadata = report.onnx.metadata;
  const jointNames = metadata.joint_names.split(",");
  if (!arrayEqual(jointNames, HUMANOID_JOINT_NAMES)) {
    throw new Error("mjlab G1 policy joint order does not match the HEAR morphology");
  }
  assertMetadataVector(
    metadata.observation_terms_scale,
    "observation scale",
    7,
    (value) => value === 1
  );
  assertMetadataVector(
    metadata.observation_terms_flatten_history_dim,
    "observation flatten history dimension",
    7,
    (value) => value === 1
  );
  assertMetadataVector(
    metadata.observation_terms_history_length,
    "observation history length",
    7,
    (value) => value === 0
  );
  if (metadata.observation_terms_clip.split(",").length !== 7
    || metadata.observation_terms_clip.split(",").some((value) => value !== "-inf;inf")) {
    throw new Error("mjlab G1 policy observation clipping metadata is unsupported");
  }
  return Object.freeze({
    policySha256: policyAsset.sha256,
    defaultJointPositions: metadataVector(
      metadata.default_joint_pos,
      "default joint position",
      HUMANOID_JOINT_NAMES.length
    ),
    actionScale: metadataVector(
      metadata.action_scale,
      "action scale",
      HUMANOID_JOINT_NAMES.length,
      (value) => value > 0
    ),
    stiffness: metadataVector(
      metadata.joint_stiffness,
      "joint stiffness",
      HUMANOID_JOINT_NAMES.length,
      (value) => value > 0
    ),
    damping: metadataVector(
      metadata.joint_damping,
      "joint damping",
      HUMANOID_JOINT_NAMES.length,
      (value) => value > 0
    )
  });
}

function requiredAsset(
  context: HumanoidControllerModuleContext,
  id: string
): HumanoidControllerModuleAsset {
  if (context.protocol !== "hear-humanoid-controller-module-v1"
    || !/^[a-f0-9]{64}$/.test(context.sourceSha256)) {
    throw new Error("Invalid humanoid controller module context");
  }
  const asset = context.assets.find((candidate) => candidate.id === id);
  if (!asset) throw new Error(`mjlab G1 controller asset is missing: ${id}`);
  return asset;
}

function assertSessionContract(session: ort.InferenceSession): void {
  if (!arrayEqual(session.inputNames, ["obs"])
    || !arrayEqual(session.outputNames, ["actions"])
    || !tensorMetadataMatches(session.inputMetadata[0], "obs", [1, OBSERVATION_SIZE])
    || !tensorMetadataMatches(
      session.outputMetadata[0],
      "actions",
      [1, HUMANOID_JOINT_NAMES.length]
    )) {
    throw new Error("mjlab G1 ONNX input or output contract is incompatible");
  }
}

function tensorMetadataMatches(
  metadata: ort.InferenceSession.ValueMetadata | undefined,
  name: string,
  shape: readonly number[]
): boolean {
  return metadata?.name === name
    && metadata.isTensor
    && metadata.type === "float32"
    && arrayEqual(metadata.shape, shape);
}

function assertPolicyInput(
  state: HumanoidPolicyState,
  reference: HumanoidReference
): void {
  assertHumanoidReference(reference);
  if (state.jointPositions.length !== HUMANOID_JOINT_NAMES.length
    || state.jointVelocities.length !== HUMANOID_JOINT_NAMES.length
    || ![
      ...Array.from(state.jointPositions),
      ...Array.from(state.jointVelocities)
    ].every(Number.isFinite)
    || !state.rootQuaternion.every(Number.isFinite)
    || !state.rootAngularVelocity.every(Number.isFinite)
    || state.environment?.protocol !== "humanoid-policy-environment-v1"
    || state.environment.authority !== "mujoco_state"
    || state.environment.rootVelocityFrame !== "pelvis_imu"
    || !state.environment.rootLinearVelocity.every(Number.isFinite)
    || !state.environment.rootAngularVelocity.every(Number.isFinite)) {
    throw new Error("mjlab G1 policy state is invalid");
  }
}

function metadataVector(
  source: string,
  label: string,
  length: number,
  predicate: (value: number) => boolean = () => true
): readonly number[] {
  const values = source.split(",").map(Number);
  if (values.length !== length
    || values.some((value) => !Number.isFinite(value) || !predicate(value))) {
    throw new Error(`mjlab G1 policy ${label} metadata is invalid`);
  }
  return Object.freeze(values);
}

function assertMetadataVector(
  source: string,
  label: string,
  length: number,
  predicate: (value: number) => boolean
): void {
  metadataVector(source, label, length, predicate);
}

function sha256(value: Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}

function arrayEqual(
  left: ArrayLike<string | number>,
  right: ArrayLike<string | number>
): boolean {
  return left.length === right.length
    && Array.from(left).every((value, index) => value === right[index]);
}

function taskTrackingStiffness(joint: HumanoidJointName): number {
  if (joint.includes("ankle")) return 55;
  if (joint.startsWith("left_hip_")
    || joint.startsWith("right_hip_")
    || joint.includes("knee")) return 80;
  if (joint.startsWith("waist_")) return 45;
  if (joint.includes("wrist")) return 40;
  return 80;
}

function taskTrackingResidualAuthority(joint: HumanoidJointName): number {
  if (joint.startsWith("left_hip_")
    || joint.startsWith("right_hip_")
    || joint.includes("knee")
    || joint.includes("ankle")) return 0.65;
  if (joint.startsWith("waist_")) return 0.35;
  return 0;
}

function inverseRotate(
  [w, x, y, z]: readonly [number, number, number, number],
  [vx, vy, vz]: readonly [number, number, number]
): [number, number, number] {
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  return [
    vx - w * tx + (y * tz - z * ty),
    vy - w * ty + (z * tx - x * tz),
    vz - w * tz + (x * ty - y * tx)
  ];
}
