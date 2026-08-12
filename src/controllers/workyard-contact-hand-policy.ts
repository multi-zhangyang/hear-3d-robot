import { createHash } from "node:crypto";
import * as ort from "onnxruntime-web";
import { z } from "zod";
import type { JsonValue } from "../domain/schema.js";
import type {
  HumanoidControllerModuleAsset
} from "../world/humanoid/controller-module.js";
import type {
  HumanoidHandSynergyPolicy,
  HumanoidHandSynergyPolicyDescriptor,
  HumanoidHandSynergyPolicyInput,
  HumanoidHandSynergyPolicyOutput
} from "../world/humanoid/hand-synergy-overlay-controller.js";
import {
  WORKYARD_CONTACT_OBSERVATION_PROTOCOL,
  WORKYARD_CONTACT_OBSERVATION_SIZE,
  encodeWorkyardContactObservation
} from "./workyard-contact-observation.js";

const ACTION_SIZE = 8;
const POLICY_ASSET_ID = "contact_policy";
const REPORT_ASSET_ID = "contact_policy_report";
const PLANT_ASSET_ID = "contact_plant";

const Sha256Schema = z.string().regex(/^[a-f0-9]{64}$/);
const ArtifactSchema = z.object({
  file: z.string().trim().min(1),
  bytes: z.number().int().positive(),
  sha256: Sha256Schema
}).strict();

const ContactPolicyReportSchema = z.object({
  protocol: z.literal("hear-frozen-contact-policy-export-v1"),
  created_at: z.iso.datetime({ offset: true }),
  source: z.object({
    training_report_file: z.string().trim().min(1),
    training_report_sha256: Sha256Schema,
    training_contract_sha256: Sha256Schema,
    training_environment_sha256: Sha256Schema,
    checkpoint_file: z.string().trim().min(1),
    checkpoint_bytes: z.number().int().positive(),
    checkpoint_sha256: Sha256Schema,
    selected_source: z.enum(["dagger", "ppo"]),
    formal_gate_protocol: z.literal(
      "hear-workyard-contact-independent-500-gate-v1"
    ),
    formal_gate_passed: z.literal(true),
    held_out_episode_count: z.literal(500),
    held_out_success_rate: z.number().finite().min(0.75).max(1),
    maximum_active_hand_force_n: z.number().finite().min(0).max(30)
  }).strict(),
  plant: z.object({
    protocol: z.literal("hear-workyard-contact-deployment-plant-v1"),
    g1_xml: ArtifactSchema.extend({
      file: z.literal("g1_with_hands.xml")
    }),
    hand_joint_count: z.literal(14),
    hand_position_kp: z.literal(2.5),
    hand_velocity_damping: z.literal(0.3),
    workyard_rod: z.object({
      shape: z.literal("cylinder"),
      radius_m: z.literal(0.03),
      half_height_m: z.literal(0.11),
      mass_kg: z.literal(0.35),
      friction: z.tuple([z.literal(0.8), z.literal(0.012), z.literal(0.002)])
    }).strict()
  }).strict(),
  policy: z.object({
    torchscript: ArtifactSchema.extend({
      runtime: z.literal("torchscript")
    }),
    onnx: ArtifactSchema.extend({
      runtime: z.literal("onnxruntime"),
      opset: z.literal(17),
      input: z.literal("observation"),
      output: z.literal("synergy_action")
    }),
    input_protocol: z.literal(WORKYARD_CONTACT_OBSERVATION_PROTOCOL),
    input_size: z.literal(WORKYARD_CONTACT_OBSERVATION_SIZE),
    output_protocol: z.literal("hear-active-hand-synergy-action-v1"),
    output_size: z.literal(ACTION_SIZE),
    distribution: z.literal("beta_bounded_minus_one_one"),
    deterministic_statistic: z.literal("mean"),
    batch_dynamic: z.literal(true),
    normalizer_epsilon: z.literal(0.01),
    parameter_count: z.number().int().positive(),
    gradient_parameter_count: z.literal(0)
  }).strict(),
  harness: z.object({
    terminal_pose_hold: z.literal("closure_latch_measured_active_arm"),
    coordination_step: z.literal(0.0075),
    maximum_closing_joint_lead_rad: z.literal(0.25),
    force_release_threshold_n: z.number().finite().positive().max(30),
    emergency_force_release_threshold_n: z.number().finite().positive().max(30)
  }).strict(),
  validation: z.object({
    probe_batch_size: z.number().int().positive(),
    maximum_torchscript_error: z.number().finite().min(0).max(1e-6),
    maximum_onnx_error: z.number().finite().min(0).max(1e-5),
    maximum_absolute_action: z.number().finite().min(0).max(1.000001),
    finite: z.literal(true),
    onnx_checker_full: z.literal(true)
  }).strict()
}).strict();

export async function createWorkyardContactHandPolicy(input: {
  assets: readonly HumanoidControllerModuleAsset[];
}): Promise<WorkyardContactHandPolicy> {
  const policyAsset = requiredAsset(input.assets, POLICY_ASSET_ID);
  const reportAsset = requiredAsset(input.assets, REPORT_ASSET_ID);
  const plantAsset = requiredAsset(input.assets, PLANT_ASSET_ID);
  const report = parseReport(policyAsset, reportAsset, plantAsset);
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(policyAsset.bytes, {
    executionProviders: ["wasm"]
  });
  try {
    assertSessionContract(session);
    return new WorkyardContactHandPolicy({
      session,
      policySha256: policyAsset.sha256,
      report
    });
  } catch (error) {
    await session.release();
    throw error;
  }
}

export class WorkyardContactHandPolicy implements HumanoidHandSynergyPolicy {
  readonly descriptor: HumanoidHandSynergyPolicyDescriptor = {
    protocol: "humanoid-hand-synergy-policy-v1",
    implementation: "workyard_contact_8d_onnx",
    runtime: "onnxruntime-web/wasm",
    observation: {
      protocol: WORKYARD_CONTACT_OBSERVATION_PROTOCOL,
      size: WORKYARD_CONTACT_OBSERVATION_SIZE
    },
    action: {
      protocol: "hear-active-hand-synergy-action-v1",
      size: ACTION_SIZE,
      coordinationStep: 0.0075,
      maximumClosingJointLeadRadians: 0.25
    }
  };
  readonly #session: ort.InferenceSession;
  readonly #policySha256: string;
  readonly #forceReleaseThresholdN: number;
  readonly #emergencyForceReleaseThresholdN: number;

  constructor(input: {
    session: ort.InferenceSession;
    policySha256: string;
    report: z.infer<typeof ContactPolicyReportSchema>;
  }) {
    this.#session = input.session;
    this.#policySha256 = input.policySha256;
    this.#forceReleaseThresholdN = input.report.harness.force_release_threshold_n;
    this.#emergencyForceReleaseThresholdN = (
      input.report.harness.emergency_force_release_threshold_n
    );
    if (this.#emergencyForceReleaseThresholdN <= this.#forceReleaseThresholdN) {
      throw new Error("Workyard emergency force reflex must exceed directional reflex");
    }
    if (input.report.policy.onnx.sha256 !== input.policySha256) {
      throw new Error("Workyard contact report changed after policy validation");
    }
  }

  reset(): void {}

  async infer(
    input: HumanoidHandSynergyPolicyInput
  ): Promise<HumanoidHandSynergyPolicyOutput> {
    assertWorkyardRodPlant(input);
    const observation = encodeWorkyardContactObservation(input);
    const tensor = new ort.Tensor(
      "float32",
      observation,
      [1, WORKYARD_CONTACT_OBSERVATION_SIZE]
    );
    let result: Awaited<ReturnType<ort.InferenceSession["run"]>> | undefined;
    let action: Float32Array;
    try {
      result = await this.#session.run({ observation: tensor });
      const output = result.synergy_action;
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
        throw new Error("Workyard contact policy returned an invalid action tensor");
      }
      action = Float32Array.from(output.data, (value) => (
        Math.max(-1, Math.min(1, value))
      ));
      applyContactForceReflex(
        action,
        input,
        this.#forceReleaseThresholdN,
        this.#emergencyForceReleaseThresholdN
      );
    } finally {
      for (const output of Object.values(result ?? {})) output.dispose();
      tensor.dispose();
    }
    return { observation, action };
  }

  captureState(): JsonValue {
    return {
      protocol: "workyard-contact-hand-policy-state-v1",
      policy_sha256: this.#policySha256
    };
  }

  restoreState(state: JsonValue): void {
    if (!isRecord(state)
      || state.protocol !== "workyard-contact-hand-policy-state-v1"
      || state.policy_sha256 !== this.#policySha256) {
      throw new Error("Workyard contact hand policy state is incompatible");
    }
  }

  async dispose(): Promise<void> {
    await this.#session.release();
  }
}

function assertWorkyardRodPlant(input: HumanoidHandSynergyPolicyInput): void {
  const object = input.state.environment?.objects.find(
    ({ id }) => id === input.authority.objectId
  );
  if (!object
    || object.shape !== "cylinder"
    || object.size?.x !== 0.06
    || object.size.y !== 0.22
    || object.size.z !== 0.06
    || object.massKg !== 0.35
    || object.friction?.sliding !== 0.8
    || object.friction.torsional !== 0.012
    || object.friction.rolling !== 0.002) {
    throw new Error(
      "Workyard contact authority targets an object outside the qualified plant"
    );
  }
}

function applyContactForceReflex(
  action: Float32Array,
  input: HumanoidHandSynergyPolicyInput,
  thresholdN: number,
  emergencyThresholdN: number
): void {
  const environment = input.state.environment;
  const activeHand = input.authority.activeHand;
  const objectId = input.authority.objectId;
  if (!environment) {
    throw new Error("Workyard force reflex requires MuJoCo contact state");
  }
  let thumbContact = false;
  let opposingContact = false;
  const force = environment.contacts.reduce((total, contact) => {
    const handLink = contact.firstHandLink ?? contact.secondHandLink;
    const matchesObject = contact.firstObject === objectId
      || contact.secondObject === objectId;
    if (!matchesObject || !handLink?.startsWith(`${activeHand}_`)) return total;
    thumbContact ||= handLink.includes("_thumb_");
    opposingContact ||= handLink.includes("_index_")
      || handLink.includes("_middle_");
    return total + contact.normalForce;
  }, 0);
  if (force < thresholdN) return;
  const offset = activeHand === "left" ? 0 : 4;
  // Match the analytic teacher's directional low-level reflex.  Unload links
  // that are already carrying force, but do not take motion authority away
  // from the missing side of a one-sided grasp.  Once both sides touch (or the
  // contact cannot be classified), all curl channels release together.
  action[offset] = 0;
  if (force >= emergencyThresholdN) {
    action.fill(-1, offset + 1, offset + 4);
  } else if (thumbContact && !opposingContact) {
    action[offset + 1] = -1;
  } else if (opposingContact && !thumbContact) {
    action.fill(-1, offset + 2, offset + 4);
  } else {
    action.fill(-1, offset + 1, offset + 4);
  }
}

function parseReport(
  policyAsset: HumanoidControllerModuleAsset,
  reportAsset: HumanoidControllerModuleAsset,
  plantAsset: HumanoidControllerModuleAsset
): z.infer<typeof ContactPolicyReportSchema> {
  let value: unknown;
  try {
    value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(
      reportAsset.bytes
    ));
  } catch (error) {
    throw new Error("Workyard contact policy report is not valid UTF-8 JSON", {
      cause: error
    });
  }
  const parsed = ContactPolicyReportSchema.safeParse(value);
  if (!parsed.success) {
    throw new Error("Workyard contact policy report is not formally qualified");
  }
  const onnx = parsed.data.policy.onnx;
  if (onnx.file !== "workyard_contact.onnx"
    || onnx.bytes !== policyAsset.bytes.byteLength
    || onnx.sha256 !== policyAsset.sha256
    || sha256(policyAsset.bytes) !== policyAsset.sha256) {
    throw new Error("Workyard contact ONNX does not match its export report");
  }
  if (parsed.data.plant.g1_xml.bytes !== plantAsset.bytes.byteLength
    || parsed.data.plant.g1_xml.sha256 !== plantAsset.sha256
    || sha256(plantAsset.bytes) !== plantAsset.sha256) {
    throw new Error("Workyard contact policy plant does not match the runtime G1 XML");
  }
  return parsed.data;
}

function requiredAsset(
  assets: readonly HumanoidControllerModuleAsset[],
  id: string
): HumanoidControllerModuleAsset {
  const asset = assets.find((candidate) => candidate.id === id);
  if (!asset) throw new Error(`Workyard contact controller asset is missing: ${id}`);
  return asset;
}

function assertSessionContract(session: ort.InferenceSession): void {
  const input = session.inputMetadata[0];
  const output = session.outputMetadata[0];
  if (!arrayEqual(session.inputNames, ["observation"])
    || !arrayEqual(session.outputNames, ["synergy_action"])
    || !dynamicTensorMetadataMatches(
      input,
      "observation",
      WORKYARD_CONTACT_OBSERVATION_SIZE
    )
    || !dynamicTensorMetadataMatches(output, "synergy_action", ACTION_SIZE)) {
    throw new Error("Workyard contact ONNX input or output contract is incompatible");
  }
}

function dynamicTensorMetadataMatches(
  metadata: ort.InferenceSession.ValueMetadata | undefined,
  name: string,
  width: number
): boolean {
  if (!metadata || !metadata.isTensor) return false;
  const batch = metadata.shape[0];
  return metadata.name === name
    && metadata.type === "float32"
    && metadata.shape.length === 2
    && (batch === 1 || batch === -1 || typeof batch === "string")
    && metadata.shape[1] === width;
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

function isRecord(value: JsonValue): value is Record<string, JsonValue> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
