import { createHash } from "node:crypto";
import {
  copyFile,
  mkdir,
  readFile,
  writeFile
} from "node:fs/promises";
import { basename, resolve } from "node:path";

import { loadRuntimeCatalog } from "../config/load.js";
import {
  createMjlabG1VelocityController,
  parseMjlabG1VelocityTrainingBundle
} from "../controllers/mjlab-g1-velocity-controller.js";
import { createWorkyardReachController } from
  "../controllers/workyard-reach-controller.js";
import type { Quaternion, Scenario, Vec3 } from "../domain/schema.js";
import type {
  HumanoidControllerModuleAsset,
  HumanoidControllerModuleContext
} from "../world/humanoid/controller-module.js";
import {
  HumanoidEmbodiedSkillCallSchema,
  legacyHumanoidEmbodiedSkillIdentity
} from "../world/humanoid/embodied-skill-call.js";
import { humanoidEnvironment } from "../world/humanoid/environment.js";
import { solveG1PregraspPose } from "../world/humanoid/pregrasp-pose.js";
import { captureHumanoidPhysicalSafetyFrame } from
  "../world/humanoid/physical-safety.js";
import { neutralHumanoidReference } from "../world/humanoid/reference.js";
import { HumanoidSimulation } from "../world/humanoid/simulation.js";

const TARGET_PROTOCOL = "typescript-pregrasp-geometry-top-wrist-target-v1";
const CANDIDATE_PROTOCOL = "hear-whole-body-reach-policy-candidate-v3";
const DEPLOYMENT_PROTOCOL = "hear-whole-body-reach-policy-deployment-v3";
const QUALIFICATION_PROTOCOL = "hear-typescript-mujoco-reach-deployment-gate-v1";
const REACH_OBSERVATION_PROTOCOL =
  "hear-workyard-whole-body-reach-observation-v5";
const REACH_OBSERVATION_SIZE = 246;
const REACH_ACTION_PROTOCOL = "bounded-whole-body-reach-mean";
const REACH_ACTION_SIZE = 29;
const CONTROL_STEP_SECONDS = 0.02;
const SETTLE_STEPS = 100;
const MAXIMUM_REACH_STEPS = 600;
const REQUIRED_STABLE_STEPS = 5;
const TARGET_TOLERANCE_METERS = 0.06;
const MINIMUM_SUPPORT_MARGIN_METERS = -0.04;
const MAXIMUM_FOOT_PLANAR_DISPLACEMENT_METERS = 0.08;
const MAXIMUM_FOOT_SLIP_SPEED_METERS_PER_SECOND = 0.20;
const MAXIMUM_DOUBLE_SUPPORT_LOSS_RATE = 0.10;
const MAXIMUM_NO_FOOT_CONTACT_RATE = 0.01;

interface QualificationCase {
  id: string;
  hand: "left" | "right";
  rootForwardOffsetMeters: number;
  rootLateralOffsetMeters: number;
  rootYawRadians: number;
}

interface CaseResult {
  id: string;
  hand: "left" | "right";
  spawn: { position: Vec3; yaw: number };
  target: { position: Vec3 };
  initialWristErrorMeters: number;
  terminalWristErrorMeters: number;
  minimumWristErrorMeters: number;
  completedSteps: number;
  stableSteps: number;
  fallen: boolean;
  unauthorizedCollisionCount: number;
  terminalAssistanceSteps: number;
  minimumSupportMarginMeters: number;
  maximumFootPlanarDisplacementMeters: number;
  maximumFootSlipSpeedMetersPerSecond: number;
  doubleSupportLossRate: number;
  noFootContactRate: number;
  success: boolean;
}

const args = parseArgs(process.argv.slice(2));
await main();

async function main(): Promise<void> {
  const candidateRoot = resolve(args.candidateRoot);
  const outputRoot = resolve(args.outputRoot);
  const candidateJitPath = resolve(candidateRoot, "workyard_reach.jit.pt");
  const candidateOnnxPath = resolve(candidateRoot, "workyard_reach.onnx");
  const candidateReportPath = resolve(candidateRoot, "reach-policy-report.json");
  const [candidateJit, candidateOnnx, candidateReportBytes] = await Promise.all([
    readFile(candidateJitPath),
    readFile(candidateOnnxPath),
    readFile(candidateReportPath)
  ]);
  const candidateReport = JSON.parse(candidateReportBytes.toString("utf8"));
  assertCandidate(candidateReport, candidateJit, candidateOnnx);
  await mkdir(outputRoot, { recursive: false });

  const catalog = await loadRuntimeCatalog();
  const scenario = catalog.materialize("humanoid_workyard", 0);
  const rod = scenario.objects.find(({ id }) => id === "assembly_rod");
  if (!rod) throw new Error("Workyard deployment target is missing");
  const assets = await loadAssets(candidateOnnx, candidateReportBytes);
  const results: CaseResult[] = [];
  for (const testCase of qualificationCases()) {
    results.push(await runCase({
      testCase,
      scenario,
      rod,
      assets
    }));
  }

  const successCount = results.filter(({ success }) => success).length;
  const successRate = successCount / results.length;
  const initialMaximum = Math.max(...results.map(
    ({ initialWristErrorMeters }) => initialWristErrorMeters
  ));
  const terminalMaximum = Math.max(...results.map(
    ({ terminalWristErrorMeters }) => terminalWristErrorMeters
  ));
  const fallCount = results.filter(({ fallen }) => fallen).length;
  const unauthorizedCollisionCount = results.reduce(
    (total, result) => total + result.unauthorizedCollisionCount,
    0
  );
  const terminalAssistanceStepCount = results.reduce(
    (total, result) => total + result.terminalAssistanceSteps,
    0
  );
  const minimumSupportMargin = Math.min(...results.map(
    ({ minimumSupportMarginMeters }) => minimumSupportMarginMeters
  ));
  const maximumFootPlanarDisplacement = Math.max(...results.map(
    ({ maximumFootPlanarDisplacementMeters }) => maximumFootPlanarDisplacementMeters
  ));
  const maximumFootSlipSpeed = Math.max(...results.map(
    ({ maximumFootSlipSpeedMetersPerSecond }) => maximumFootSlipSpeedMetersPerSecond
  ));
  const maximumDoubleSupportLossRate = Math.max(...results.map(
    ({ doubleSupportLossRate }) => doubleSupportLossRate
  ));
  const maximumNoFootContactRate = Math.max(...results.map(
    ({ noFootContactRate }) => noFootContactRate
  ));
  const accepted = results.length >= 12
    && successRate >= 0.9
    && initialMaximum >= 0.35
    && terminalMaximum <= TARGET_TOLERANCE_METERS
    && fallCount === 0
    && unauthorizedCollisionCount === 0
    && terminalAssistanceStepCount === 0
    && minimumSupportMargin >= MINIMUM_SUPPORT_MARGIN_METERS
    && maximumFootPlanarDisplacement <= MAXIMUM_FOOT_PLANAR_DISPLACEMENT_METERS
    && maximumFootSlipSpeed <= MAXIMUM_FOOT_SLIP_SPEED_METERS_PER_SECOND
    && maximumDoubleSupportLossRate <= MAXIMUM_DOUBLE_SUPPORT_LOSS_RATE
    && maximumNoFootContactRate <= MAXIMUM_NO_FOOT_CONTACT_RATE;
  const qualification = {
    protocol: QUALIFICATION_PROTOCOL,
    created_at: new Date().toISOString(),
    accepted,
    runtime: "typescript-mujoco-onnxruntime-web",
    scenario_id: "humanoid_workyard",
    target_protocol: TARGET_PROTOCOL,
    controller_mode: "learned_policy_only",
    policy: {
      file: "workyard_reach.onnx",
      bytes: candidateOnnx.byteLength,
      sha256: sha256(candidateOnnx)
    },
    thresholds: {
      case_count_minimum: 12,
      success_rate_minimum: 0.9,
      initial_wrist_error_maximum_minimum_m: 0.35,
      terminal_wrist_error_maximum_m: TARGET_TOLERANCE_METERS,
      fall_count_maximum: 0,
      unauthorized_collision_count_maximum: 0,
      terminal_assistance_step_count_maximum: 0,
      minimum_support_margin_m: MINIMUM_SUPPORT_MARGIN_METERS,
      maximum_foot_planar_displacement_m: MAXIMUM_FOOT_PLANAR_DISPLACEMENT_METERS,
      maximum_foot_slip_speed_m_s: MAXIMUM_FOOT_SLIP_SPEED_METERS_PER_SECOND,
      double_support_loss_rate_maximum: MAXIMUM_DOUBLE_SUPPORT_LOSS_RATE,
      no_foot_contact_rate_maximum: MAXIMUM_NO_FOOT_CONTACT_RATE
    },
    summary: {
      case_count: results.length,
      success_count: successCount,
      success_rate: successRate,
      initial_wrist_error_maximum_m: initialMaximum,
      terminal_wrist_error_maximum_m: terminalMaximum,
      fall_count: fallCount,
      unauthorized_collision_count: unauthorizedCollisionCount,
      terminal_assistance_step_count: terminalAssistanceStepCount,
      minimum_support_margin_m: minimumSupportMargin,
      maximum_foot_planar_displacement_m: maximumFootPlanarDisplacement,
      maximum_foot_slip_speed_m_s: maximumFootSlipSpeed,
      double_support_loss_rate_maximum: maximumDoubleSupportLossRate,
      no_foot_contact_rate_maximum: maximumNoFootContactRate
    },
    cases: results
  };
  const qualificationPath = resolve(
    outputRoot, "deployment-qualification-report.json"
  );
  const qualificationBytes = Buffer.from(
    `${JSON.stringify(qualification, null, 2)}\n`, "utf8"
  );
  await writeFile(qualificationPath, qualificationBytes);
  if (!accepted) {
    throw new Error(
      `Reach candidate failed TypeScript MuJoCo deployment qualification: ${qualificationPath}`
    );
  }

  const deploymentReport = {
    ...candidateReport,
    protocol: DEPLOYMENT_PROTOCOL,
    deployment: {
      protocol: QUALIFICATION_PROTOCOL,
      accepted: true,
      runtime: "typescript-mujoco-onnxruntime-web",
      scenario_id: "humanoid_workyard",
      target_protocol: TARGET_PROTOCOL,
      controller_mode: "learned_policy_only",
      case_count: results.length,
      success_rate: successRate,
      initial_wrist_error_maximum_m: initialMaximum,
      terminal_wrist_error_maximum_m: terminalMaximum,
      fall_count: fallCount,
      unauthorized_collision_count: unauthorizedCollisionCount,
      terminal_assistance_step_count: terminalAssistanceStepCount,
      minimum_support_margin_m: minimumSupportMargin,
      maximum_foot_planar_displacement_m: maximumFootPlanarDisplacement,
      maximum_foot_slip_speed_m_s: maximumFootSlipSpeed,
      double_support_loss_rate_maximum: maximumDoubleSupportLossRate,
      no_foot_contact_rate_maximum: maximumNoFootContactRate,
      report_file: basename(qualificationPath),
      report_sha256: sha256(qualificationBytes)
    }
  };
  await Promise.all([
    copyFile(candidateJitPath, resolve(outputRoot, "workyard_reach.jit.pt")),
    copyFile(candidateOnnxPath, resolve(outputRoot, "workyard_reach.onnx")),
    writeFile(
      resolve(outputRoot, "reach-policy-report.json"),
      `${JSON.stringify(deploymentReport, null, 2)}\n`,
      "utf8"
    )
  ]);
  process.stdout.write(`${JSON.stringify(qualification.summary, null, 2)}\n`);
}

async function runCase(input: {
  testCase: QualificationCase;
  scenario: Scenario;
  rod: Scenario["objects"][number];
  assets: {
    bodyPolicy: HumanoidControllerModuleAsset;
    bodyReport: HumanoidControllerModuleAsset;
    reachPolicy: HumanoidControllerModuleAsset;
    reachReport: HumanoidControllerModuleAsset;
  };
}): Promise<CaseResult> {
  const { testCase, scenario, rod, assets } = input;
  const side = testCase.hand === "right" ? 1 : -1;
  const spawn = {
    position: {
      x: rod.position.x + side * testCase.rootLateralOffsetMeters,
      y: 0,
      z: rod.position.z - testCase.rootForwardOffsetMeters
    },
    yaw: testCase.rootYawRadians
  };
  const environment = humanoidEnvironment(scenario);
  const simulation = await HumanoidSimulation.create({
    ...environment,
    spawn,
    controllerFactory: () => createCandidateController(assets)
  });
  const reference = neutralHumanoidReference();
  try {
    for (let step = 0; step < SETTLE_STEPS; step += 1) {
      await simulation.step(reference, {
        taskCommand: stationKeepingCommand(`settle:${testCase.id}:${step}`)
      });
    }
    const settled = simulation.snapshot();
    const wristBody = `${testCase.hand}_wrist_yaw_link` as const;
    const wrist = settled.links[wristBody];
    const observedRod = settled.objects.assembly_rod;
    if (!observedRod) throw new Error("Assembly rod disappeared during settling");
    const pregrasp = solveG1PregraspPose({
      hand: testCase.hand,
      wristRotation: wrist.rotation,
      handSurfaces: simulation.handSurfaceObservations(settled),
      interactionPoint: {
        x: observedRod.position.x,
        y: observedRod.position.y + rod.size.y / 2,
        z: observedRod.position.z
      },
      approachDirection: { x: 0, y: -1, z: 0 }
    });
    const initialError = distance(wrist.position, pregrasp.position);
    let minimumError = initialError;
    let terminalError = initialError;
    let stableSteps = 0;
    let completedSteps = 0;
    let fallen = settled.fallen;
    let unauthorizedCollisionCount = settled.nonFootEnvironmentContacts.length;
    let terminalAssistanceSteps = 0;
    const initialFootPositions = {
      left: settled.links.left_ankle_roll_link.position,
      right: settled.links.right_ankle_roll_link.position
    };
    let minimumSupportMargin = Number.POSITIVE_INFINITY;
    let maximumFootPlanarDisplacement = 0;
    let maximumFootSlipSpeed = 0;
    let doubleSupportLossCount = 0;
    let noFootContactCount = 0;
    for (let step = 0; step < MAXIMUM_REACH_STEPS; step += 1) {
      const snapshot = await simulation.step(reference, {
        taskCommand: reachCommand({
          id: testCase.id,
          hand: testCase.hand,
          step,
          targetPosition: pregrasp.position,
          targetRotation: pregrasp.rotation
        })
      });
      completedSteps = step + 1;
      terminalError = distance(
        snapshot.links[wristBody].position,
        pregrasp.position
      );
      minimumError = Math.min(minimumError, terminalError);
      fallen ||= snapshot.fallen;
      unauthorizedCollisionCount += snapshot.nonFootEnvironmentContacts.length;
      if (snapshot.controllerExecution?.mode !== "learned_policy") {
        terminalAssistanceSteps += 1;
      }
      const safety = captureHumanoidPhysicalSafetyFrame(step, snapshot);
      minimumSupportMargin = Math.min(
        minimumSupportMargin,
        safety.support.signed_margin_m ?? Number.NEGATIVE_INFINITY
      );
      maximumFootPlanarDisplacement = Math.max(
        maximumFootPlanarDisplacement,
        planarDistance(
          snapshot.links.left_ankle_roll_link.position,
          initialFootPositions.left
        ),
        planarDistance(
          snapshot.links.right_ankle_roll_link.position,
          initialFootPositions.right
        )
      );
      maximumFootSlipSpeed = Math.max(
        maximumFootSlipSpeed,
        safety.foot_slip.maximum?.tangential_speed_mps ?? 0
      );
      if (snapshot.balance.support !== "double") doubleSupportLossCount += 1;
      if (snapshot.balance.support === "none") noFootContactCount += 1;
      stableSteps = terminalError <= TARGET_TOLERANCE_METERS
        ? stableSteps + 1
        : 0;
      if (stableSteps >= REQUIRED_STABLE_STEPS || fallen
        || unauthorizedCollisionCount > 0) break;
    }
    return {
      id: testCase.id,
      hand: testCase.hand,
      spawn: { position: spawn.position, yaw: spawn.yaw },
      target: { position: pregrasp.position },
      initialWristErrorMeters: initialError,
      terminalWristErrorMeters: terminalError,
      minimumWristErrorMeters: minimumError,
      completedSteps,
      stableSteps,
      fallen,
      unauthorizedCollisionCount,
      terminalAssistanceSteps,
      minimumSupportMarginMeters: minimumSupportMargin,
      maximumFootPlanarDisplacementMeters: maximumFootPlanarDisplacement,
      maximumFootSlipSpeedMetersPerSecond: maximumFootSlipSpeed,
      doubleSupportLossRate: completedSteps === 0
        ? 1
        : doubleSupportLossCount / completedSteps,
      noFootContactRate: completedSteps === 0 ? 1 : noFootContactCount / completedSteps,
      success: stableSteps >= REQUIRED_STABLE_STEPS
        && !fallen
        && unauthorizedCollisionCount === 0
        && minimumSupportMargin >= MINIMUM_SUPPORT_MARGIN_METERS
        && maximumFootPlanarDisplacement <= MAXIMUM_FOOT_PLANAR_DISPLACEMENT_METERS
        && maximumFootSlipSpeed <= MAXIMUM_FOOT_SLIP_SPEED_METERS_PER_SECOND
        && doubleSupportLossCount / Math.max(completedSteps, 1)
          <= MAXIMUM_DOUBLE_SUPPORT_LOSS_RATE
        && noFootContactCount / Math.max(completedSteps, 1)
          <= MAXIMUM_NO_FOOT_CONTACT_RATE
    };
  } finally {
    await simulation.dispose();
  }
}

async function createCandidateController(assets: Awaited<ReturnType<
  typeof loadAssets
>>) {
  const bodyContext: HumanoidControllerModuleContext = {
    protocol: "hear-humanoid-controller-module-v1",
    sourceSha256: sha256(Buffer.concat([
      assets.bodyPolicy.bytes,
      assets.bodyReport.bytes
    ])),
    assets: [assets.bodyPolicy, assets.bodyReport]
  };
  const bodyPolicy = parseMjlabG1VelocityTrainingBundle(
    assets.bodyPolicy,
    assets.bodyReport
  );
  const body = await createMjlabG1VelocityController(bodyContext);
  try {
    return await createWorkyardReachController({
      assets: [assets.reachPolicy, assets.reachReport],
      body,
      bodyPolicy,
      targetZoneId: "assembly_bay",
      qualificationCandidate: true,
      terminalTaskSpaceReflex: "disabled"
    });
  } catch (error) {
    await body.dispose();
    throw error;
  }
}

async function loadAssets(
  reachOnnx: Uint8Array,
  reachReport: Uint8Array
) {
  const bodyRoot = resolve(
    "assets/humanoid/controllers/mjlab-g1-velocity"
  );
  const [bodyOnnx, bodyReport] = await Promise.all([
    readFile(resolve(bodyRoot, "g1_velocity.onnx")),
    readFile(resolve(bodyRoot, "training-report.json"))
  ]);
  return {
    bodyPolicy: asset("policy", bodyOnnx),
    bodyReport: asset("training_report", bodyReport),
    reachPolicy: asset("reach_policy", reachOnnx),
    reachReport: asset("reach_policy_report", reachReport)
  };
}

function stationKeepingCommand(callId: string) {
  return HumanoidEmbodiedSkillCallSchema.parse({
    protocol: "humanoid-embodied-skill-call-v2",
    identity: legacyHumanoidEmbodiedSkillIdentity({
      callId,
      runtimeKind: "station_keeping",
      phase: "settle",
      observedFrame: 0,
      observedWorldRevision: 0
    }),
    authority: {
      source: "deterministic_runtime",
      worldFrame: 0,
      worldRevision: 0
    },
    window: {
      mode: "autonomous_closed_loop",
      replanPolicy: "event_driven",
      controlStepSeconds: CONTROL_STEP_SECONDS,
      maximumSteps: 1,
      stepIndex: 0,
      remainingSteps: 1
    },
    requestedCapabilities: ["balance"],
    command: {
      baseTwist: {
        forwardMetersPerSecond: 0,
        lateralMetersPerSecond: 0,
        yawRadiansPerSecond: 0
      },
      rootHeightMeters: neutralHumanoidReference().rootHeight,
      leftWristPositionPelvis: null,
      rightWristPositionPelvis: null,
      endEffectors: [],
      grasps: []
    },
    contract: null,
    safety: {
      authorizedContacts: [],
      stopOnFall: true,
      stopOnUnauthorizedContact: true,
      stopOnContractViolation: true
    },
    feedback: {
      mode: "event_driven",
      progressDelta: 0.1,
      events: ["progress", "failed"]
    }
  });
}

function reachCommand(input: {
  id: string;
  hand: "left" | "right";
  step: number;
  targetPosition: Vec3;
  targetRotation: Quaternion;
}) {
  const maximumSteps = MAXIMUM_REACH_STEPS;
  const wristBody = `${input.hand}_wrist_yaw_link`;
  return HumanoidEmbodiedSkillCallSchema.parse({
    protocol: "humanoid-embodied-skill-call-v2",
    identity: legacyHumanoidEmbodiedSkillIdentity({
      callId: `qualification:${input.id}`,
      runtimeKind: "legacy_motion",
      phase: "reach",
      observedFrame: 0,
      observedWorldRevision: 0
    }),
    authority: {
      source: "deterministic_runtime",
      worldFrame: input.step,
      worldRevision: input.step
    },
    window: {
      mode: "autonomous_closed_loop",
      replanPolicy: "event_driven",
      controlStepSeconds: CONTROL_STEP_SECONDS,
      maximumSteps,
      stepIndex: input.step,
      remainingSteps: maximumSteps - input.step
    },
    requestedCapabilities: ["balance", "joint_reference_tracking"],
    command: {
      baseTwist: {
        forwardMetersPerSecond: 0,
        lateralMetersPerSecond: 0,
        yawRadiansPerSecond: 0
      },
      rootHeightMeters: neutralHumanoidReference().rootHeight,
      leftWristPositionPelvis: null,
      rightWristPositionPelvis: null,
      endEffectors: [{
        body: wristBody,
        frame: "world",
        position: input.targetPosition,
        tolerance: TARGET_TOLERANCE_METERS,
        orientation: input.targetRotation,
        orientationTolerance: 0.35
      }],
      grasps: [{
        objectId: "assembly_rod",
        hand: input.hand,
        minimumNormalForceN: 2,
        minimumDistinctContactSurfaces: 2
      }]
    },
    contract: null,
    safety: {
      authorizedContacts: [],
      stopOnFall: true,
      stopOnUnauthorizedContact: true,
      stopOnContractViolation: true
    },
    feedback: {
      mode: "event_driven",
      progressDelta: 0.05,
      events: ["progress", "succeeded", "failed"]
    }
  });
}

function qualificationCases(): QualificationCase[] {
  return (["left", "right"] as const).flatMap((hand) => {
    const side = hand === "right" ? 1 : -1;
    return [
      [0.34, 0.06, 0.45],
      [0.34, 0.12, 0.75],
      [0.40, 0.06, 0.60],
      [0.40, 0.12, 0.90],
      [0.46, 0.06, 0.75],
      [0.46, 0.12, 1.05]
    ].map(([forward, lateral, yaw], index) => ({
      id: `${hand}-${index + 1}`,
      hand,
      rootForwardOffsetMeters: forward!,
      rootLateralOffsetMeters: lateral!,
      rootYawRadians: side * yaw!
    }));
  });
}

function assertCandidate(
  report: any,
  jit: Uint8Array,
  onnx: Uint8Array
): void {
  if (report?.protocol !== CANDIDATE_PROTOCOL
    || report?.source?.deployment_distribution_covered !== true
    || report?.source?.training_deployment_accepted !== false
    || report?.source?.initial_wrist_error_maximum_m < 0.35
    || report?.source?.target_protocol !== TARGET_PROTOCOL
    || report?.policy?.file !== "workyard_reach.jit.pt"
    || report?.policy?.bytes !== jit.byteLength
    || report?.policy?.sha256 !== sha256(jit)
    || report?.policy?.input !== REACH_OBSERVATION_PROTOCOL
    || report?.policy?.input_size !== REACH_OBSERVATION_SIZE
    || report?.policy?.output !== REACH_ACTION_PROTOCOL
    || report?.policy?.output_size !== REACH_ACTION_SIZE
    || report?.onnx?.file !== "workyard_reach.onnx"
    || report?.onnx?.bytes !== onnx.byteLength
    || report?.onnx?.sha256 !== sha256(onnx)
    || report?.onnx?.input_protocol !== REACH_OBSERVATION_PROTOCOL
    || report?.onnx?.input_size !== REACH_OBSERVATION_SIZE
    || report?.onnx?.output_protocol !== REACH_ACTION_PROTOCOL
    || report?.onnx?.output_size !== REACH_ACTION_SIZE) {
    throw new Error("Reach candidate is not eligible for deployment qualification");
  }
}

function asset(id: string, bytes: Uint8Array): HumanoidControllerModuleAsset {
  return { id, bytes: bytes.slice(), sha256: sha256(bytes) };
}

function distance(left: Vec3, right: Vec3): number {
  return Math.hypot(left.x - right.x, left.y - right.y, left.z - right.z);
}

function planarDistance(left: Vec3, right: Vec3): number {
  return Math.hypot(left.x - right.x, left.z - right.z);
}

function sha256(value: Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}

function parseArgs(values: string[]): {
  candidateRoot: string;
  outputRoot: string;
} {
  const parsed = {
    candidateRoot: "artifacts/training/workyard-reach-candidate-v3",
    outputRoot: "artifacts/training/workyard-reach-deployment-v3"
  };
  for (let index = 0; index < values.length; index += 2) {
    const name = values[index];
    const value = values[index + 1];
    if (!value) throw new Error(`Incomplete deployment option: ${name ?? ""}`);
    if (name === "--candidate-root") parsed.candidateRoot = value;
    else if (name === "--output") parsed.outputRoot = value;
    else throw new Error(`Unknown deployment option: ${name ?? ""}`);
  }
  return parsed;
}
