import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type {
  HumanoidHandSynergyPolicyInput
} from "../world/humanoid/hand-synergy-overlay-controller.js";
import type {
  HumanoidControllerInferenceOptions,
  HumanoidPolicyState
} from "../world/humanoid/whole-body-controller.js";
import {
  WORKYARD_CONTACT_OBSERVATION_SIZE,
  WORKYARD_REACH_OBSERVATION_PROTOCOL,
  WORKYARD_REACH_OBSERVATION_SIZE,
  encodeWorkyardContactObservation,
  encodeWorkyardReachObservation
} from "../controllers/workyard-contact-observation.js";

const FIXTURE_PROTOCOL = "hear-workyard-observation-parity-fixture-v1";
const MAXIMUM_ERROR = 2e-5;
const FEATURE_GROUPS = [
  ["body_position", 0, 29],
  ["body_velocity", 29, 58],
  ["previous_reach_action", 58, 72],
  ["hand_position", 72, 86],
  ["hand_velocity", 86, 100],
  ["dynamic_com", 100, 104],
  ["previous_locomotion_action", 104, 133],
  ["root_kinematics", 133, 142],
  ["end_effector_pose", 142, 170],
  ["foot_contact", 170, 176],
  ["hand_contact", 176, 182],
  ["object_pose_twist", 182, 195],
  ["target_zone", 195, 200],
  ["capability_progress_hand", 200, 208],
  ["task_command", 208, 231],
  ["hand_history", 231, 247]
] as const;

interface Fixture {
  protocol: string;
  episode_seed: number;
  active_hand: "left" | "right";
  control_step: number;
  milestone: string;
  input: {
    state: unknown;
    options: unknown;
    previousTeacherAction: number[];
    previousReachAction: number[];
    coordination: unknown;
    previousAuthorizedAction: number[];
  };
  expected: {
    reachObservation: number[];
    contactObservation: number[];
  };
}

function main(): void {
  const options = parseOptions(process.argv.slice(2));
  const report = jsonObject(options.report);
  const evaluation = objectValue(report.evaluation, "evaluation");
  const fixturesValue = evaluation.observation_parity_fixtures;
  if (!Array.isArray(fixturesValue) || fixturesValue.length < 4) {
    throw new Error("Workyard report has fewer than four parity fixtures");
  }
  const metadata = bodyMetadata(options.bodyReport);
  const results = fixturesValue.map((value) => validateFixture(
    value as unknown as Fixture,
    metadata
  ));
  const maximumError = Math.max(...results.map(({ maximum_error }) => maximum_error));
  const sides = new Set(results.map(({ active_hand }) => active_hand));
  const milestones = new Set(results.map(({ milestone }) => milestone));
  const payload = {
    protocol: "hear-workyard-observation-parity-report-v1",
    fixture_count: results.length,
    active_hands: [...sides].sort(),
    milestones: [...milestones].sort(),
    maximum_absolute_error: maximumError,
    threshold: MAXIMUM_ERROR,
    passed: maximumError <= MAXIMUM_ERROR
      && sides.has("left") && sides.has("right"),
    fixtures: results
  };
  process.stdout.write(`${JSON.stringify(payload, null, 2)}\n`);
  if (!payload.passed) process.exitCode = 1;
}

function validateFixture(
  fixture: Fixture,
  metadata: ReturnType<typeof bodyMetadata>
) {
  if (fixture.protocol !== FIXTURE_PROTOCOL
    || !Number.isSafeInteger(fixture.episode_seed)
    || !Number.isSafeInteger(fixture.control_step)) {
    throw new Error("Workyard parity fixture identity is invalid");
  }
  const expectedReach = finiteVector(
    fixture.expected.reachObservation,
    WORKYARD_REACH_OBSERVATION_SIZE,
    "expected reach observation"
  );
  const expectedContact = finiteVector(
    fixture.expected.contactObservation,
    WORKYARD_CONTACT_OBSERVATION_SIZE,
    "expected contact observation"
  );
  const reach = encodeWorkyardReachObservation({
    state: fixture.input.state as HumanoidPolicyState,
    options: fixture.input.options as HumanoidControllerInferenceOptions,
    previousTeacherAction: fixture.input.previousTeacherAction,
    previousReachAction: fixture.input.previousReachAction
  }, metadata);
  const contact = encodeWorkyardContactObservation({
    bodyInference: {
      protocol: "humanoid-controller-inference-trace-v1",
      implementation: "parity_fixture",
      route: "upper_body_overlay",
      components: [{
        protocol: "humanoid-controller-tensor-trace-v1",
        role: "fallback",
        implementation: "workyard_frozen_reach_onnx",
        observation: {
          protocol: WORKYARD_REACH_OBSERVATION_PROTOCOL,
          values: [...reach]
        },
        action: {
          protocol: "bounded-upper-body-residual-mean",
          values: new Array(14).fill(0)
        }
      }]
    },
    coordination: fixture.input.coordination,
    previousAuthorizedAction: Float64Array.from(
      fixture.input.previousAuthorizedAction
    )
  } as HumanoidHandSynergyPolicyInput);
  const reachErrors = absoluteErrors(reach, expectedReach);
  const contactErrors = absoluteErrors(contact, expectedContact);
  const combined = [...reachErrors, ...contactErrors.slice(
    WORKYARD_REACH_OBSERVATION_SIZE
  )];
  const maximumError = Math.max(...combined);
  const maximumIndex = combined.indexOf(maximumError);
  return {
    episode_seed: fixture.episode_seed,
    active_hand: fixture.active_hand,
    control_step: fixture.control_step,
    milestone: fixture.milestone,
    maximum_error: maximumError,
    maximum_error_index: maximumIndex,
    maximum_error_group: featureGroup(maximumIndex),
    groups: Object.fromEntries(FEATURE_GROUPS.map(([name, start, end]) => [
      name,
      Math.max(...combined.slice(start, end))
    ]))
  };
}

function bodyMetadata(path: string) {
  const report = jsonObject(path);
  const onnx = objectValue(report.onnx, "body ONNX report");
  const metadata = objectValue(onnx.metadata, "body ONNX metadata");
  return {
    bodyDefaultJointPositions: metadataVector(
      metadata.default_joint_pos,
      "default joint position"
    ),
    bodyActionScale: metadataVector(metadata.action_scale, "action scale"),
    targetZoneId: "assembly_bay"
  };
}

function metadataVector(value: unknown, label: string): number[] {
  if (typeof value !== "string") {
    throw new Error(`Body policy ${label} metadata is missing`);
  }
  return finiteVector(value.split(",").map(Number), 29, label);
}

function absoluteErrors(
  actual: ArrayLike<number>,
  expected: readonly number[]
): number[] {
  if (actual.length !== expected.length) {
    throw new Error("Parity tensor size changed");
  }
  return Array.from(actual, (value, index) => Math.abs(value - expected[index]!));
}

function featureGroup(index: number): string {
  return FEATURE_GROUPS.find(([, start, end]) => (
    index >= start && index < end
  ))?.[0] ?? "unknown";
}

function finiteVector(
  value: unknown,
  size: number,
  label: string
): number[] {
  if (!Array.isArray(value)
    || value.length !== size
    || value.some((entry) => typeof entry !== "number" || !Number.isFinite(entry))) {
    throw new Error(`Workyard parity ${label} is invalid`);
  }
  return value as number[];
}

function jsonObject(path: string): Record<string, unknown> {
  const value: unknown = JSON.parse(readFileSync(resolve(path), "utf8"));
  return objectValue(value, path);
}

function objectValue(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`Expected a JSON object for ${label}`);
  }
  return value as Record<string, unknown>;
}

function parseOptions(args: string[]): { report: string; bodyReport: string } {
  const values = new Map<string, string>();
  for (let index = 0; index < args.length; index += 2) {
    const name = args[index];
    const value = args[index + 1];
    if (!name || !value || !["--report", "--body-report"].includes(name)) {
      throw new Error(`Unknown or incomplete parity option: ${name ?? ""}`);
    }
    values.set(name, value);
  }
  const report = values.get("--report");
  const bodyReport = values.get("--body-report")
    ?? "artifacts/training/g1-residual-teacher/training-report.json";
  if (!report) throw new Error("--report is required");
  return { report, bodyReport };
}

main();
