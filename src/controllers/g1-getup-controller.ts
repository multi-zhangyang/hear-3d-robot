import { createHash } from "node:crypto";
import * as ort from "onnxruntime-web";
import { z } from "zod";
import {
  HUMANOID_JOINT_NAMES
} from "../world/humanoid/model.js";
import type {
  HumanoidControllerModuleAsset
} from "../world/humanoid/controller-module.js";
import type { HumanoidReference } from "../world/humanoid/reference.js";
import type {
  HumanoidControllerDescriptor,
  HumanoidControllerInferenceOptions,
  HumanoidControllerInferenceTrace,
  HumanoidControllerState,
  HumanoidJointPositionCommand,
  HumanoidPolicyState,
  HumanoidWholeBodyController
} from "../world/humanoid/whole-body-controller.js";
import {
  G1_GETUP_OBSERVATION_PROTOCOL,
  G1_GETUP_OBSERVATION_SIZE,
  encodeG1GetupObservation
} from "./g1-getup-observation.js";

export const G1_GETUP_POLICY_ASSET_ID = "getup_policy";
export const G1_GETUP_REPORT_ASSET_ID = "getup_policy_report";
export const G1_GETUP_ACTION_PROTOCOL = "hear-g1-getup-joint-target-v1";
const ACTION_SIZE = 29;
const CONTROL_STEP_SECONDS = 0.02;
const PHYSICS_STEP_SECONDS = 0.005;

const Sha256Schema = z.string().regex(/^[a-f0-9]{64}$/);
const ArtifactSchema = z.object({
  file: z.literal("g1_getup.onnx"),
  bytes: z.number().int().positive(),
  sha256: Sha256Schema
}).strict();
const JointVectorSchema = z.array(z.number().finite()).length(ACTION_SIZE);

const G1GetupDeploymentReportSchema = z.object({
  protocol: z.literal("hear-g1-getup-policy-deployment-v1"),
  created_at: z.iso.datetime({ offset: true }),
  framework: z.object({
    name: z.literal("mjlab"),
    version: z.string().trim().min(1),
    task_id: z.literal("Hear-G1-Getup-v1")
  }).strict(),
  source: z.object({
    checkpoint: z.object({
      file: z.string().trim().min(1),
      bytes: z.number().int().positive(),
      sha256: Sha256Schema
    }).strict(),
    training_contract_sha256: Sha256Schema,
    environment_sha256: Sha256Schema,
    seed: z.number().int().nonnegative(),
    iterations: z.number().int().positive(),
    environment_count: z.number().int().positive()
  }).strict(),
  policy: z.object({
    onnx: ArtifactSchema,
    runtime: z.literal("onnxruntime-web/wasm"),
    input: z.literal("obs"),
    input_protocol: z.literal(G1_GETUP_OBSERVATION_PROTOCOL),
    input_size: z.literal(G1_GETUP_OBSERVATION_SIZE),
    output: z.literal("actions"),
    output_protocol: z.literal(G1_GETUP_ACTION_PROTOCOL),
    output_size: z.literal(ACTION_SIZE),
    inputs: z.tuple([z.literal("obs")]),
    outputs: z.tuple([z.literal("actions")]),
    joint_names: z.array(z.string().trim().min(1)).length(ACTION_SIZE),
    default_joint_positions: JointVectorSchema,
    joint_lower_limits: JointVectorSchema,
    joint_upper_limits: JointVectorSchema,
    stiffness: z.array(z.number().finite().positive()).length(ACTION_SIZE),
    damping: z.array(z.number().finite().positive()).length(ACTION_SIZE),
    action_mapping: z.literal("neutral_piecewise_soft_joint_limits"),
    actor_inputs: z.literal("proprioception_only_no_reference_phase")
  }).strict(),
  training: z.object({
    iterations: z.number().int().positive(),
    environment_count: z.number().int().positive(),
    steps_per_environment_per_iteration: z.number().int().positive(),
    environment_steps: z.number().int().positive(),
    accelerator: z.string().trim().min(1),
    cuda_version: z.string().trim().min(1)
  }).strict(),
  evaluation: z.object({
    episode_count: z.number().int().min(500),
    overall_success_rate: z.number().finite().min(0.80).max(1),
    prone_success_rate: z.number().finite().min(0.75).max(1),
    supine_success_rate: z.number().finite().min(0.75).max(1),
    side_success_rate: z.number().finite().min(0.75).max(1),
    median_recovery_seconds: z.number().finite().positive().max(12),
    p95_recovery_seconds: z.number().finite().positive().max(15),
    stable_exit_rate: z.number().finite().min(0.75).max(1),
    non_finite_action_count: z.literal(0),
    reset_category_counts: z.record(
      z.string().trim().min(1), z.number().int().nonnegative()
    ),
    maximum_absolute_action: z.number().finite().min(0).max(1.000001),
    deployment_accepted: z.literal(true)
  }).strict()
}).strict().superRefine((report, context) => {
  if (JSON.stringify(report.policy.joint_names)
    !== JSON.stringify(HUMANOID_JOINT_NAMES)) {
    context.addIssue({
      code: "custom",
      path: ["policy", "joint_names"],
      message: "Get-up policy joint order differs from the HEAR G1 morphology"
    });
  }
  for (let index = 0; index < ACTION_SIZE; index += 1) {
    const lower = report.policy.joint_lower_limits[index]!;
    const neutral = report.policy.default_joint_positions[index]!;
    const upper = report.policy.joint_upper_limits[index]!;
    if (!(lower < neutral && neutral < upper)) {
      context.addIssue({
        code: "custom",
        path: ["policy", "joint_lower_limits", index],
        message: "Get-up joint limits must contain the neutral pose"
      });
    }
  }
});

const ControllerStateSchema = z.object({
  protocol: z.literal("hear-g1-getup-controller-state-v1"),
  policy_sha256: Sha256Schema,
  previous_action: z.array(z.number().finite().min(-1).max(1)).length(ACTION_SIZE)
}).strict();

export interface G1GetupPolicyParameters {
  readonly policySha256: string;
  readonly defaultJointPositions: readonly number[];
  readonly jointLowerLimits: readonly number[];
  readonly jointUpperLimits: readonly number[];
  readonly stiffness: readonly number[];
  readonly damping: readonly number[];
}

export async function createG1GetupController(input: {
  assets: readonly HumanoidControllerModuleAsset[];
}): Promise<G1GetupController> {
  const policyAsset = requiredAsset(input.assets, G1_GETUP_POLICY_ASSET_ID);
  const reportAsset = requiredAsset(input.assets, G1_GETUP_REPORT_ASSET_ID);
  const policy = parseG1GetupDeploymentBundle(policyAsset, reportAsset);
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(policyAsset.bytes, {
    executionProviders: ["wasm"]
  });
  try {
    assertSessionContract(session);
    return new G1GetupController(session, policy);
  } catch (error) {
    await session.release();
    throw error;
  }
}

export class G1GetupController implements HumanoidWholeBodyController {
  readonly descriptor: HumanoidControllerDescriptor = {
    protocol: "humanoid-controller-v1",
    implementation: "g1_proprioceptive_getup_onnx",
    actuation: "joint_position_pd",
    controlStepSeconds: CONTROL_STEP_SECONDS,
    physicsStepSeconds: PHYSICS_STEP_SECONDS,
    learnedPolicy: {
      protocol: "humanoid-learned-policy-v1",
      runtime: "onnxruntime-web/wasm",
      observationSpace: {
        protocol: G1_GETUP_OBSERVATION_PROTOCOL,
        size: G1_GETUP_OBSERVATION_SIZE
      },
      actionSpace: {
        protocol: G1_GETUP_ACTION_PROTOCOL,
        size: ACTION_SIZE
      },
      observationFeatures: [
        "proprioception",
        "command_history",
        "root_kinematics",
        "contact_state"
      ],
      capabilities: [
        "balance",
        "whole_body_recovery",
        "joint_reference_tracking"
      ]
    }
  };
  readonly #session: ort.InferenceSession;
  readonly #policy: G1GetupPolicyParameters;
  #previousAction = new Float32Array(ACTION_SIZE);
  #lastInferenceTrace: HumanoidControllerInferenceTrace | null = null;

  constructor(
    session: ort.InferenceSession,
    policy: G1GetupPolicyParameters
  ) {
    this.#session = session;
    this.#policy = policy;
  }

  reset(state: HumanoidPolicyState, _reference: HumanoidReference): void {
    encodeG1GetupObservation(state, this.#previousAction, this.#policy);
    this.#previousAction.fill(0);
    this.#lastInferenceTrace = null;
  }

  async infer(
    state: HumanoidPolicyState,
    _reference: HumanoidReference,
    _options: HumanoidControllerInferenceOptions = {}
  ): Promise<HumanoidJointPositionCommand> {
    const observation = encodeG1GetupObservation(
      state,
      this.#previousAction,
      this.#policy
    );
    const input = new ort.Tensor(
      "float32",
      observation,
      [1, G1_GETUP_OBSERVATION_SIZE]
    );
    let result: Awaited<ReturnType<ort.InferenceSession["run"]>> | undefined;
    try {
      result = await this.#session.run({ obs: input });
      const output = result.actions;
      if (!output
        || output.type !== "float32"
        || output.dims.length !== 2
        || output.dims[0] !== 1
        || output.dims[1] !== ACTION_SIZE
        || !(output.data instanceof Float32Array)
        || output.data.length !== ACTION_SIZE
        || !output.data.every((value) => (
          Number.isFinite(value) && value >= -1.000001 && value <= 1.000001
        ))) {
        throw new Error("G1 get-up ONNX returned an invalid action tensor");
      }
      this.#previousAction = Float32Array.from(output.data, clampUnit);
    } finally {
      for (const tensor of Object.values(result ?? {})) tensor.dispose();
      input.dispose();
    }
    this.#lastInferenceTrace = {
      protocol: "humanoid-controller-inference-trace-v1",
      implementation: this.descriptor.implementation,
      route: "direct",
      components: [{
        protocol: "humanoid-controller-tensor-trace-v1",
        role: "direct",
        implementation: this.descriptor.implementation,
        observation: {
          protocol: G1_GETUP_OBSERVATION_PROTOCOL,
          values: [...observation]
        },
        action: {
          protocol: G1_GETUP_ACTION_PROTOCOL,
          values: [...this.#previousAction]
        }
      }]
    };
    return this.#command();
  }

  inferenceTrace(): HumanoidControllerInferenceTrace | null {
    return this.#lastInferenceTrace
      ? structuredClone(this.#lastInferenceTrace)
      : null;
  }

  advanceHistory(): void {}

  captureState(): HumanoidControllerState {
    return {
      protocol: "humanoid-controller-state-v1",
      version: 1,
      implementation: this.descriptor.implementation,
      payload: {
        protocol: "hear-g1-getup-controller-state-v1",
        policy_sha256: this.#policy.policySha256,
        previous_action: [...this.#previousAction]
      }
    };
  }

  restoreState(state: HumanoidControllerState): void {
    if (state.protocol !== "humanoid-controller-state-v1"
      || state.version !== 1
      || state.implementation !== this.descriptor.implementation) {
      throw new Error("G1 get-up controller state is incompatible");
    }
    const parsed = ControllerStateSchema.safeParse(state.payload);
    if (!parsed.success
      || parsed.data.policy_sha256 !== this.#policy.policySha256) {
      throw new Error("G1 get-up controller state is incompatible");
    }
    this.#previousAction = Float32Array.from(parsed.data.previous_action);
    this.#lastInferenceTrace = null;
  }

  async dispose(): Promise<void> {
    await this.#session.release();
  }

  #command(): HumanoidJointPositionCommand {
    const positions = Float64Array.from(this.#previousAction, (action, index) => {
      const neutral = this.#policy.defaultJointPositions[index]!;
      return action >= 0
        ? neutral + action * (this.#policy.jointUpperLimits[index]! - neutral)
        : neutral + action * (neutral - this.#policy.jointLowerLimits[index]!);
    });
    return {
      kind: "joint_position_pd",
      positions,
      stiffness: Float64Array.from(this.#policy.stiffness),
      damping: Float64Array.from(this.#policy.damping)
    };
  }
}

export function parseG1GetupDeploymentBundle(
  policyAsset: HumanoidControllerModuleAsset,
  reportAsset: HumanoidControllerModuleAsset
): G1GetupPolicyParameters {
  let reportJson: unknown;
  try {
    reportJson = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(
      reportAsset.bytes
    ));
  } catch (error) {
    throw new Error("G1 get-up report is not valid UTF-8 JSON", { cause: error });
  }
  const parsed = G1GetupDeploymentReportSchema.safeParse(reportJson);
  if (!parsed.success) {
    throw new Error("G1 get-up report does not match deployment version 1", {
      cause: parsed.error
    });
  }
  const report = parsed.data;
  if (report.policy.onnx.bytes !== policyAsset.bytes.byteLength
    || report.policy.onnx.sha256 !== policyAsset.sha256
    || sha256(policyAsset.bytes) !== policyAsset.sha256) {
    throw new Error("G1 get-up ONNX does not match its deployment report");
  }
  return Object.freeze({
    policySha256: policyAsset.sha256,
    defaultJointPositions: Object.freeze([...report.policy.default_joint_positions]),
    jointLowerLimits: Object.freeze([...report.policy.joint_lower_limits]),
    jointUpperLimits: Object.freeze([...report.policy.joint_upper_limits]),
    stiffness: Object.freeze([...report.policy.stiffness]),
    damping: Object.freeze([...report.policy.damping])
  });
}

function requiredAsset(
  assets: readonly HumanoidControllerModuleAsset[],
  id: string
): HumanoidControllerModuleAsset {
  const asset = assets.find((candidate) => candidate.id === id);
  if (!asset) throw new Error(`G1 get-up controller asset is missing: ${id}`);
  return asset;
}

function assertSessionContract(session: ort.InferenceSession): void {
  const input = session.inputMetadata[0];
  const output = session.outputMetadata[0];
  if (session.inputNames.length !== 1 || session.inputNames[0] !== "obs"
    || session.outputNames.length !== 1 || session.outputNames[0] !== "actions"
    || !input?.isTensor || input.type !== "float32"
    || input.shape.length !== 2
    || input.shape[1] !== G1_GETUP_OBSERVATION_SIZE
    || !output?.isTensor || output.type !== "float32"
    || output.shape.length !== 2 || output.shape[1] !== ACTION_SIZE) {
    throw new Error("G1 get-up ONNX input or output contract is incompatible");
  }
}

function clampUnit(value: number): number {
  return Math.max(-1, Math.min(1, value));
}

function sha256(value: Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}
