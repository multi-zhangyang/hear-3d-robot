import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

import { loadConfiguredHumanoidControllerSource } from
  "../world/humanoid/controller-module.js";
import { modelPayloadSha256 } from "../domain/model-call-authority.js";
import {
  HUMANOID_SKILL_CONTRACTS,
  HUMANOID_SKILL_IDS
} from "../domain/humanoid-skill.js";
import {
  HUMANOID_RECOVERY_CONTROL_STEP_SECONDS,
  HUMANOID_RECOVERY_HANDOFF_STEPS,
  HUMANOID_RECOVERY_MAXIMUM_STEPS,
  HUMANOID_RECOVERY_STABLE_STEPS
} from "../domain/humanoid-policy.js";
import {
  HumanoidEmbodiedSkillIdentitySchema,
  type HumanoidEmbodiedSkillIdentity
} from "../world/humanoid/embodied-skill-call.js";
import { controllerTaskCommand } from
  "../world/humanoid/motion-frame-application.js";
import { neutralHumanoidReference } from "../world/humanoid/reference.js";
import {
  HumanoidRecoveryExecutionContractSchema,
  type HumanoidRecoveryExecutionContract
} from "../world/humanoid/recovery-execution-contract.js";
import {
  humanoidRecoverySafetyFailure,
  humanoidRecoveryStandingSatisfied
} from "../world/humanoid/recovery-execution.js";
import {
  HumanoidSimulation,
  type HumanoidSimulationSnapshot,
  type HumanoidSimulationState
} from "../world/humanoid/simulation.js";
import type {
  HumanoidWholeBodyControllerFactory
} from "../world/humanoid/whole-body-controller.js";
import {
  G1_GETUP_POLICY_DIRECTORY_ENV,
  G1_GETUP_QUALIFICATION_MODE_ENV
} from "../controllers/g1-recovery-module.js";

const QUALIFICATION_PROTOCOL =
  "hear-typescript-mujoco-g1-getup-deployment-gate-v1";
const BODY_JOINT_COUNT = 29;
const FREE_JOINT_POSITION_COUNT = 7;
const GROUND_JOINT_POSE = new Map<number, number>([
  [0, -0.45],
  [3, 1.05],
  [6, -0.45],
  [9, 1.05],
  [15, 0.75],
  [18, 1.20],
  [22, 0.75],
  [25, 1.20]
]);

interface QualificationPose {
  id: "prone" | "supine" | "left_side" | "right_side";
  rootHeightMeters: number;
  rootQuaternionWxyz: readonly [number, number, number, number];
}

interface QualificationResult {
  pose: QualificationPose["id"];
  initial_fallen: boolean;
  recovery_expert_steps: number;
  stable_steps: number;
  handoff_steps: number;
  completed_steps: number;
  maximum_peak_contact_normal_force_n: number;
  maximum_total_contact_normal_force_n: number;
  maximum_total_contact_force_rise_rate_n_s: number;
  maximum_trajectory_joint_speed_rad_s: number;
  minimum_trajectory_joint_limit_margin_rad: number;
  safety_accepted: boolean;
  terminal_root_height_m: number;
  terminal_upright: number;
  terminal_support_margin_m: number | null;
  terminal_both_feet_contact: boolean;
  terminal_maximum_joint_speed_rad_s: number;
  terminal_controller_implementation: string;
  recovered: boolean;
}

const args = parseArgs(process.argv.slice(2));
await main();

async function main(): Promise<void> {
  const output = resolve(args.output);
  const policyDirectory = resolve(
    args.policyDirectory ?? "assets/humanoid/controllers/g1-getup"
  );
  if (args.policyDirectory) {
    process.env[G1_GETUP_POLICY_DIRECTORY_ENV] = policyDirectory;
    process.env[G1_GETUP_QUALIFICATION_MODE_ENV] = "1";
  }
  const [policyBytes, deploymentReportBytes] = await Promise.all([
    readFile(resolve(policyDirectory, "g1_getup.onnx")),
    readFile(resolve(policyDirectory, "getup-policy-report.json"))
  ]);
  const controllerSource = await loadConfiguredHumanoidControllerSource();
  const results: QualificationResult[] = [];
  for (const pose of qualificationPoses()) {
    results.push(await runPose(controllerSource.controllerFactory, pose));
  }
  const accepted = results.length === qualificationPoses().length
    && results.every(({ recovered }) => recovered);
  const report = {
    protocol: QUALIFICATION_PROTOCOL,
    created_at: new Date().toISOString(),
    accepted,
    runtime: "typescript-mujoco-onnxruntime-web",
    controller_source_sha256: controllerSource.sourceSha256,
    policy_sha256: sha256(policyBytes),
    deployment_report_sha256: sha256(deploymentReportBytes),
    controller: {
      normal: "yahmp_onnx",
      recovery: "g1_proprioceptive_getup_onnx",
      gate: "g1_recovery_expert_gate"
    },
    contract: {
      control_step_seconds: HUMANOID_RECOVERY_CONTROL_STEP_SECONDS,
      maximum_steps: HUMANOID_RECOVERY_MAXIMUM_STEPS,
      required_stable_steps: HUMANOID_RECOVERY_STABLE_STEPS,
      required_handoff_steps: HUMANOID_RECOVERY_HANDOFF_STEPS,
      required_poses: qualificationPoses().map(({ id }) => id)
    },
    summary: {
      pose_count: results.length,
      recovered_count: results.filter(({ recovered }) => recovered).length,
      maximum_completion_steps: Math.max(
        ...results.map(({ completed_steps }) => completed_steps)
      )
    },
    poses: results
  };
  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  process.stdout.write(`${JSON.stringify(report.summary, null, 2)}\n`);
  if (!accepted) {
    throw new Error(
      `G1 get-up policy failed the TypeScript MuJoCo deployment gate: ${output}`
    );
  }
}

async function runPose(
  controllerFactory: HumanoidWholeBodyControllerFactory,
  pose: QualificationPose
): Promise<QualificationResult> {
  const simulation = await HumanoidSimulation.create({ controllerFactory });
  const reference = neutralHumanoidReference();
  try {
    const state = fallenState(simulation.captureState(), pose);
    simulation.restoreState(state);
    const initial = simulation.snapshot();
    const initialFallen = initial.fallen;
    if (!initialFallen) {
      throw new Error(`G1 deployment pose did not produce a fall: ${pose.id}`);
    }
    const identity = recoveryIdentity(pose.id);
    const contract = recoveryContract(initial);
    let terminal = initial;
    let recoveryExpertSteps = 0;
    let stableSteps = 0;
    let handoffSteps = 0;
    let handoffStarted = false;
    let completedSteps = 0;
    let maximumPeakContactForce = 0;
    let maximumTotalContactForce = 0;
    let maximumContactForceRiseRate = 0;
    let maximumTrajectoryJointSpeed = 0;
    let minimumTrajectoryJointMargin = Number.POSITIVE_INFINITY;
    let previousTotalContactForce: number | null = null;
    let safetyAccepted = true;
    for (let step = 0; step < HUMANOID_RECOVERY_MAXIMUM_STEPS; step += 1) {
      terminal = await simulation.step(reference, {
        trackedJointPolicyCommand: "measured",
        taskCommand: controllerTaskCommand({
          taskId: identity.callId,
          taskGoal: contract,
          skillIdentity: identity,
          authority: {
            worldFrame: step,
            worldRevision: step
          },
          controlWindow: {
            maximumSteps: HUMANOID_RECOVERY_MAXIMUM_STEPS,
            stepIndex: step
          },
          authorizedContacts: [],
          recoverySafety: true,
          controlStepSeconds: HUMANOID_RECOVERY_CONTROL_STEP_SECONDS,
          reference,
          taskSpaceTargets: [],
          carryTaskSpaceTargets: [],
          graspTargets: []
        })
      });
      completedSteps = step + 1;
      const peakContactForce = Math.max(
        0,
        ...terminal.contacts.map(({ normalForce }) => normalForce)
      );
      const totalContactForce = terminal.contacts.reduce(
        (sum, { normalForce }) => sum + normalForce,
        0
      );
      const contactForceRiseRate = previousTotalContactForce === null
        ? 0
        : (totalContactForce - previousTotalContactForce)
          / HUMANOID_RECOVERY_CONTROL_STEP_SECONDS;
      const trajectoryJointSpeed = Math.max(
        ...Object.values(terminal.joints).map(
          ({ velocity }) => Math.abs(velocity)
        )
      );
      const trajectoryJointMargin = Math.min(
        ...Object.values(terminal.joints).map((joint) => Math.min(
          joint.position - joint.minimum,
          joint.maximum - joint.position
        ))
      );
      maximumPeakContactForce = Math.max(
        maximumPeakContactForce,
        peakContactForce
      );
      maximumTotalContactForce = Math.max(
        maximumTotalContactForce,
        totalContactForce
      );
      maximumContactForceRiseRate = Math.max(
        maximumContactForceRiseRate,
        contactForceRiseRate
      );
      maximumTrajectoryJointSpeed = Math.max(
        maximumTrajectoryJointSpeed,
        trajectoryJointSpeed
      );
      minimumTrajectoryJointMargin = Math.min(
        minimumTrajectoryJointMargin,
        trajectoryJointMargin
      );
      if (humanoidRecoverySafetyFailure(
        terminal,
        contract,
        step + 1,
        previousTotalContactForce,
        HUMANOID_RECOVERY_CONTROL_STEP_SECONDS
      )) {
        safetyAccepted = false;
      }
      previousTotalContactForce = totalContactForce;
      stableSteps = humanoidRecoveryStandingSatisfied(terminal, contract)
        ? stableSteps + 1
        : 0;
      const execution = terminal.controllerExecution;
      if (execution?.activeImplementation === "g1_proprioceptive_getup_onnx") {
        recoveryExpertSteps += 1;
      }
      if (execution?.transition?.fromImplementation
        === "g1_proprioceptive_getup_onnx") {
        handoffStarted = true;
        handoffSteps += 1;
      } else if (handoffStarted
        && execution?.activeImplementation !== "g1_proprioceptive_getup_onnx"
        && execution?.transition === null) {
        // The gate clears its transition on the final blended command.  Count
        // that command as the last handoff step, exactly as production
        // recovery execution does.
        handoffSteps += 1;
      }
      if (productionRecoveryCompleted(
        terminal,
        contract,
        recoveryExpertSteps,
        stableSteps,
        handoffSteps
      )) {
        break;
      }
    }
    const maximumJointSpeed = Math.max(
      ...Object.values(terminal.joints).map(({ velocity }) => Math.abs(velocity))
    );
    const recovered = initialFallen
      && safetyAccepted
      && productionRecoveryCompleted(
        terminal,
        contract,
        recoveryExpertSteps,
        stableSteps,
        handoffSteps
      );
    return {
      pose: pose.id,
      initial_fallen: initialFallen,
      recovery_expert_steps: recoveryExpertSteps,
      stable_steps: stableSteps,
      handoff_steps: handoffSteps,
      completed_steps: completedSteps,
      maximum_peak_contact_normal_force_n: maximumPeakContactForce,
      maximum_total_contact_normal_force_n: maximumTotalContactForce,
      maximum_total_contact_force_rise_rate_n_s: maximumContactForceRiseRate,
      maximum_trajectory_joint_speed_rad_s: maximumTrajectoryJointSpeed,
      minimum_trajectory_joint_limit_margin_rad: minimumTrajectoryJointMargin,
      safety_accepted: safetyAccepted,
      terminal_root_height_m: terminal.rootPosition.y,
      terminal_upright: terminal.balance.upright,
      terminal_support_margin_m: terminal.balance.supportMargin,
      terminal_both_feet_contact:
        terminal.feet.left.touching && terminal.feet.right.touching,
      terminal_maximum_joint_speed_rad_s: maximumJointSpeed,
      terminal_controller_implementation:
        terminal.controllerExecution?.activeImplementation ?? "unavailable",
      recovered
    };
  } finally {
    await simulation.dispose();
  }
}

function fallenState(
  source: HumanoidSimulationState,
  pose: QualificationPose
): HumanoidSimulationState {
  const state = structuredClone(source);
  if (state.positions.length < FREE_JOINT_POSITION_COUNT + BODY_JOINT_COUNT) {
    throw new Error("G1 deployment plant has an incompatible position vector");
  }
  state.time = 0;
  state.positions[2] = pose.rootHeightMeters;
  for (let index = 0; index < 4; index += 1) {
    state.positions[3 + index] = pose.rootQuaternionWxyz[index]!;
  }
  for (const [jointIndex, position] of GROUND_JOINT_POSE) {
    state.positions[FREE_JOINT_POSITION_COUNT + jointIndex] = position;
  }
  state.velocities.fill(0);
  state.controls.fill(0);
  state.activations.fill(0);
  state.accelerationWarmstart.fill(0);
  state.requestedActuatorTorques?.fill(0);
  return state;
}

function recoveryIdentity(pose: QualificationPose["id"]): HumanoidEmbodiedSkillIdentity {
  const invocation = {
    skill: "stabilize" as const,
    minimum_support_margin_m: 0
  };
  return HumanoidEmbodiedSkillIdentitySchema.parse({
    protocol: "humanoid-embodied-skill-identity-v1",
    callId: `g1-getup-deployment:${pose}`,
    runtimeKind: "semantic_skill",
    agentId: "humanoid-action-selection-gate",
    bindingTransactionId: `g1-getup-deployment-binding:${pose}`,
    skillPlanTransactionId: `g1-getup-deployment-plan:${pose}`,
    skillNodeId: `g1-getup-deployment-node:${pose}`,
    skillId: "stabilize",
    phase: "recover_support",
    invocation,
    invocationSha256: modelPayloadSha256(invocation),
    skillCatalogSha256: modelPayloadSha256(
      HUMANOID_SKILL_IDS.map((id) => HUMANOID_SKILL_CONTRACTS[id])
    ),
    observedFrame: 0,
    observedWorldRevision: 0
  });
}

function recoveryContract(
  snapshot: HumanoidSimulationSnapshot
): HumanoidRecoveryExecutionContract {
  const at = new Date().toISOString();
  return HumanoidRecoveryExecutionContractSchema.parse({
    protocol: "humanoid-embodied-recovery-contract-v1",
    safetyInterrupt: {
      interrupt_id: randomUUID(),
      kind: "stationary_fall",
      source_node_id: "humanoid-mujoco-body",
      relay_node_id: "humanoid-controller-reflex",
      target_node_id: "humanoid-action-selection-gate",
      world_frame: 0,
      world_revision: 0,
      commitment_id: null,
      status: "acknowledged",
      detail: {
        source: "typescript_mujoco_deployment_qualification",
        fallen: snapshot.fallen
      },
      created_at: at,
      acknowledged_at: at,
      acknowledged_by_node_id: "humanoid-action-selection-gate",
      resolved_at: null,
      resolved_world_revision: null
    },
    minimumSupportMarginMeters: 0,
    stableSteps: HUMANOID_RECOVERY_STABLE_STEPS,
    handoffSteps: HUMANOID_RECOVERY_HANDOFF_STEPS,
    maximumSteps: HUMANOID_RECOVERY_MAXIMUM_STEPS,
    authorizedContacts: [],
    standing: {
      minimumRootHeightMeters: 0.7,
      minimumUpright: 0.9,
      maximumRootLinearSpeedMetersPerSecond: 0.35,
      maximumRootAngularSpeedRadiansPerSecond: 0.5,
      maximumJointSpeedRadiansPerSecond: 1.5,
      requireBothFeetContact: true
    },
    safetyLimits: {
      maximumPeakContactNormalForceN: 2500,
      maximumTotalContactNormalForceN: 4000,
      maximumTotalContactForceRiseRateNPerSecond: 100000,
      maximumJointSpeedRadiansPerSecond: 40,
      minimumJointLimitMarginRadians: -0.1
    }
  });
}

function productionRecoveryCompleted(
  snapshot: HumanoidSimulationSnapshot,
  contract: HumanoidRecoveryExecutionContract,
  recoveryExpertSteps: number,
  stableSteps: number,
  handoffSteps: number
): boolean {
  const execution = snapshot.controllerExecution;
  return recoveryExpertSteps > 0
    && stableSteps >= HUMANOID_RECOVERY_STABLE_STEPS
    && handoffSteps >= HUMANOID_RECOVERY_HANDOFF_STEPS
    && execution?.activeImplementation === "yahmp_onnx"
    && execution.transition === null
    && !snapshot.fallen
    && humanoidRecoveryStandingSatisfied(snapshot, contract);
}

function qualificationPoses(): readonly QualificationPose[] {
  const halfSqrt = Math.SQRT1_2;
  return [
    {
      id: "prone",
      rootHeightMeters: 0.245,
      rootQuaternionWxyz: [halfSqrt, 0, halfSqrt, 0]
    },
    {
      id: "supine",
      rootHeightMeters: 0.245,
      rootQuaternionWxyz: [halfSqrt, 0, -halfSqrt, 0]
    },
    {
      id: "left_side",
      rootHeightMeters: 0.27,
      rootQuaternionWxyz: [halfSqrt, halfSqrt, 0, 0]
    },
    {
      id: "right_side",
      rootHeightMeters: 0.27,
      rootQuaternionWxyz: [halfSqrt, -halfSqrt, 0, 0]
    }
  ];
}

function parseArgs(values: string[]): {
  output: string;
  policyDirectory?: string;
} {
  const parsed: {
    output: string;
    policyDirectory?: string;
  } = {
    output: "artifacts/training/g1-getup/runtime-deployment-report.json"
  };
  for (let index = 0; index < values.length; index += 2) {
    const name = values[index];
    const value = values[index + 1];
    if (!value) throw new Error(`Incomplete G1 deployment option: ${name ?? ""}`);
    if (name === "--output") parsed.output = value;
    else if (name === "--policy-directory") parsed.policyDirectory = value;
    else throw new Error(`Unknown G1 deployment option: ${name ?? ""}`);
  }
  return parsed;
}

function sha256(value: Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}
