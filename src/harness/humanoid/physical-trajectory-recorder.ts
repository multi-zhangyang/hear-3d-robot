import {
  HUMANOID_END_EFFECTORS,
  type HumanoidEndEffector,
  type Vec3
} from "../../domain/schema.js";
import {
  PhysicalTrajectoryFrameSchema,
  PhysicalTrajectorySummarySchema,
  advancePhysicalTrajectorySha256,
  physicalTrajectoryFrameSha256,
  type PhysicalTrajectoryFrame,
  type PhysicalTrajectorySummary
} from "../../domain/physical-trajectory.js";
import { humanoidEndEffectorPosition } from "../../world/humanoid/end-effectors.js";
import { HUMANOID_JOINT_NAMES } from "../../world/humanoid/model.js";
import { G1_HAND_JOINT_NAMES } from "../../world/humanoid/morphology.js";
import type { HumanoidWorldSnapshot } from "../../world/humanoid/world.js";

const MAX_TRAJECTORY_SAMPLES = 64;
const JOINT_NAMES = [...HUMANOID_JOINT_NAMES, ...G1_HAND_JOINT_NAMES]
  .sort(compareCodePoints);

export function createPhysicalTrajectory(
  world: HumanoidWorldSnapshot,
  completeFromAdmission = true
): PhysicalTrajectorySummary {
  const frame = captureFrame(world);
  return PhysicalTrajectorySummarySchema.parse({
    version: 1,
    complete_from_admission: completeFromAdmission,
    start_frame: frame.frame,
    end_frame: frame.frame,
    start_world_revision: frame.world_revision,
    end_world_revision: frame.world_revision,
    observed_frame_count: 1,
    sample_stride: 1,
    joint_names: JOINT_NAMES,
    samples: [frame],
    root_path_length_m: 0,
    root_planar_path_length_m: 0,
    joint_total_variation_rad: 0,
    end_effector_path_length_m: endEffectorNumbers(0),
    object_path_length_m: Object.fromEntries(frame.objects.map(({ id }) => [id, 0])),
    contact_transition_count: 0,
    controller_usage: controllerUsage(frame, true),
    trajectory_sha256: advancePhysicalTrajectorySha256(null, frame.frame_sha256)
  });
}

export function advancePhysicalTrajectory(
  persisted: PhysicalTrajectorySummary,
  world: HumanoidWorldSnapshot
): PhysicalTrajectorySummary {
  return advancePhysicalTrajectoryFrame(
    PhysicalTrajectorySummarySchema.parse(persisted),
    world,
    true
  );
}

/**
 * Controller-frequency variant for a summary already admitted by the runtime.
 * Durable checkpoints still pass through the full schema; reparsing the same
 * 64-sample summary twice on every 50 Hz tick adds no new authority.
 */
export function advanceTrustedPhysicalTrajectory(
  persisted: PhysicalTrajectorySummary,
  world: HumanoidWorldSnapshot
): PhysicalTrajectorySummary {
  return advancePhysicalTrajectoryFrame(persisted, world, false);
}

function advancePhysicalTrajectoryFrame(
  summary: PhysicalTrajectorySummary,
  world: HumanoidWorldSnapshot,
  validate: boolean
): PhysicalTrajectorySummary {
  const frame = captureFrame(world, validate);
  const previous = summary.samples.at(-1)!;
  if (frame.frame === summary.end_frame
    && frame.world_revision === summary.end_world_revision) {
    return summary;
  }
  if (frame.frame !== summary.end_frame + 1
    || frame.world_revision !== summary.end_world_revision + 1) {
    throw new Error(
      `Physical trajectory is not contiguous: ${summary.end_frame}/${summary.end_world_revision}`
      + ` -> ${frame.frame}/${frame.world_revision}`
    );
  }

  const objectDistance = { ...summary.object_path_length_m };
  const previousObjects = new Map(previous.objects.map((object) => [object.id, object.position]));
  for (const object of frame.objects) {
    objectDistance[object.id] = rounded(
      (objectDistance[object.id] ?? 0)
        + distance(previousObjects.get(object.id), object.position)
    );
  }
  let stride = summary.sample_stride;
  let samples = [...summary.samples];
  const terminal = samples.at(-1)!;
  if (terminal.frame !== summary.start_frame
    && (terminal.frame - summary.start_frame) % stride !== 0) {
    samples.pop();
  }
  samples.push(frame);
  while (samples.length > MAX_TRAJECTORY_SAMPLES) {
    stride *= 2;
    samples = samples.filter((sample, index) => (
      index === 0
        || index === samples.length - 1
        || (sample.frame - summary.start_frame) % stride === 0
    ));
  }

  const next = {
    ...summary,
    end_frame: frame.frame,
    end_world_revision: frame.world_revision,
    observed_frame_count: summary.observed_frame_count + 1,
    sample_stride: stride,
    samples,
    root_path_length_m: rounded(
      summary.root_path_length_m + distance(previous.root_position, frame.root_position)
    ),
    root_planar_path_length_m: rounded(
      summary.root_planar_path_length_m
        + Math.hypot(
          frame.root_position.x - previous.root_position.x,
          frame.root_position.z - previous.root_position.z
        )
    ),
    joint_total_variation_rad: rounded(
      summary.joint_total_variation_rad
        + frame.joint_positions.reduce((total, position, index) => (
          total + Math.abs(position - previous.joint_positions[index]!)
        ), 0)
    ),
    end_effector_path_length_m: Object.fromEntries(HUMANOID_END_EFFECTORS.map((name) => [
      name,
      rounded(
        summary.end_effector_path_length_m[name]
          + distance(previous.end_effectors[name], frame.end_effectors[name])
      )
    ])),
    object_path_length_m: objectDistance,
    contact_transition_count: summary.contact_transition_count
      + (sameContactState(previous, frame) ? 0 : 1),
    controller_usage: advanceControllerUsage(summary.controller_usage, frame),
    trajectory_sha256: advancePhysicalTrajectorySha256(
      summary.trajectory_sha256,
      frame.frame_sha256
    )
  } as PhysicalTrajectorySummary;
  return validate ? PhysicalTrajectorySummarySchema.parse(next) : next;
}

function captureFrame(
  world: HumanoidWorldSnapshot,
  validate = true
): PhysicalTrajectoryFrame {
  const robot = world.robot;
  const identity = {
    frame: world.frame,
    world_revision: world.worldRevision,
    root_position: vector(robot.rootPosition),
    root_rotation: {
      x: rounded(robot.rootRotation.x),
      y: rounded(robot.rootRotation.y),
      z: rounded(robot.rootRotation.z),
      w: rounded(robot.rootRotation.w)
    },
    joint_positions: JOINT_NAMES.map((name) => rounded(
      name in robot.joints
        ? robot.joints[name as keyof typeof robot.joints]!.position
        : robot.hands.joints[name as keyof typeof robot.hands.joints]!.position
    )),
    end_effectors: Object.fromEntries(HUMANOID_END_EFFECTORS.map((name) => {
      const position = humanoidEndEffectorPosition(robot, name, "world");
      if (!position) throw new Error(`Missing humanoid end-effector state: ${name}`);
      return [name, vector(position)];
    })) as Record<HumanoidEndEffector, Vec3>,
    contacts: aggregateContacts(robot.contacts),
    objects: Object.values(robot.objects).map((object) => ({
      id: object.id,
      position: vector(object.position)
    })).sort((left, right) => compareCodePoints(left.id, right.id)),
    support: robot.balance.support,
    fallen: robot.fallen,
    controller_execution: controllerExecution(world)
  };
  const frame = {
    ...identity,
    frame_sha256: physicalTrajectoryFrameSha256(identity)
  } as PhysicalTrajectoryFrame;
  return validate ? PhysicalTrajectoryFrameSchema.parse(frame) : frame;
}

function controllerExecution(
  world: HumanoidWorldSnapshot
): NonNullable<PhysicalTrajectoryFrame["controller_execution"]> {
  const execution = world.robot.controllerExecution ?? {
    protocol: "humanoid-controller-execution-v1" as const,
    mode: world.robot.controller.learnedPolicy
      ? "learned_policy" as const
      : "reference_control" as const,
    activeImplementation: world.robot.controller.implementation,
    transition: null
  };
  return {
    mode: execution.mode,
    active_implementation: execution.activeImplementation,
    transition: execution.transition
      ? {
          from_implementation: execution.transition.fromImplementation,
          to_implementation: execution.transition.toImplementation,
          progress: rounded(execution.transition.progress),
          duration_seconds: rounded(execution.transition.durationSeconds)
        }
      : null,
    ...(execution.routing?.assessment
      ? {
          routing: {
            call_id: execution.routing.callId,
            route: execution.routing.route,
            implementation: execution.routing.assessment.implementation,
            skill_family: execution.routing.assessment.skillFamily,
            admitted: execution.routing.assessment.admitted,
            reason: execution.routing.assessment.reason,
            cold_start: execution.routing.assessment.coldStart,
            entry_state_ood_score:
              nullableRounded(execution.routing.assessment.entryStateOodScore),
            command_ood_score:
              nullableRounded(execution.routing.assessment.commandOodScore),
            attribution: {
              primary_steps: execution.routing.attribution.primarySteps,
              fallback_steps: execution.routing.attribution.fallbackSteps,
              upper_body_overlay_steps:
                execution.routing.attribution.upperBodyOverlaySteps,
              memory_bridge_steps:
                execution.routing.attribution.memoryBridgeSteps
            },
            memory_bridge: execution.routing.memoryBridge
              ? {
                  protocol: execution.routing.memoryBridge.protocol,
                  phase: execution.routing.memoryBridge.phase,
                  trigger: execution.routing.memoryBridge.trigger,
                  completed_steps:
                    execution.routing.memoryBridge.completedSteps,
                  maximum_steps: execution.routing.memoryBridge.maximumSteps,
                  stable_steps: execution.routing.memoryBridge.stableSteps,
                  required_stable_steps:
                    execution.routing.memoryBridge.requiredStableSteps,
                  progress: rounded(execution.routing.memoryBridge.progress),
                  entry_state_ood_score: rounded(
                    execution.routing.memoryBridge.entryStateOodScore
                  ),
                  joint_prototype_rms_error: rounded(
                    execution.routing.memoryBridge.jointPrototypeRmsError
                  ),
                  maximum_joint_velocity: rounded(
                    execution.routing.memoryBridge.maximumJointVelocity
                  )
                }
              : null,
            posterior: {
              outcomes: execution.routing.assessment.posterior.outcomes,
              successes: execution.routing.assessment.posterior.successes,
              failures: execution.routing.assessment.posterior.failures,
              mean: rounded(
                execution.routing.assessment.posterior.posteriorMean
              ),
              lower_bound: rounded(
                execution.routing.assessment.posterior.lowerBound
              ),
              upper_bound: rounded(
                execution.routing.assessment.posterior.upperBound
              ),
              recent_success_rate: nullableRounded(
                execution.routing.assessment.posterior.recentSuccessRate
              ),
              transition_attempts:
                execution.routing.assessment.posterior.transitionAttempts,
              transition_successes:
                execution.routing.assessment.posterior.transitionSuccesses
            }
          }
        }
      : {})
  };
}

function controllerUsage(
  frame: PhysicalTrajectoryFrame,
  completeFromAdmission: boolean
): NonNullable<PhysicalTrajectorySummary["controller_usage"]> {
  const execution = frame.controller_execution;
  if (!execution) {
    throw new Error("Cannot capture controller usage without execution authority");
  }
  return {
    protocol: "humanoid-controller-usage-v1",
    complete_from_admission: completeFromAdmission,
    observed_frame_count: 1,
    mode_frame_counts: {
      learned_policy: execution.mode === "learned_policy" ? 1 : 0,
      reference_control: execution.mode === "reference_control" ? 1 : 0,
      hybrid_control: execution.mode === "hybrid_control" ? 1 : 0
    },
    implementation_frame_counts: {
      [execution.active_implementation]: 1
    },
    transition_frame_count: execution.transition ? 1 : 0,
    ...(execution.routing
      ? { routing: initialRoutingUsage(execution.routing) }
      : {})
  };
}

function advanceControllerUsage(
  current: PhysicalTrajectorySummary["controller_usage"],
  frame: PhysicalTrajectoryFrame
): PhysicalTrajectorySummary["controller_usage"] {
  const execution = frame.controller_execution;
  if (!execution) return current;
  if (!current) return controllerUsage(frame, false);
  const routing = frame.controller_execution?.routing;
  const nextRouting = !routing
    ? current.routing
    : !current.routing
      ? initialRoutingUsage(routing)
      : current.routing.last_call_id === routing.call_id
        ? advanceCurrentRoutingUsage(current.routing, routing)
        : advanceRoutingUsage(current.routing, routing);
  return {
    ...current,
    observed_frame_count: current.observed_frame_count + 1,
    mode_frame_counts: {
      ...current.mode_frame_counts,
      [execution.mode]: current.mode_frame_counts[execution.mode] + 1
    },
    implementation_frame_counts: {
      ...current.implementation_frame_counts,
      [execution.active_implementation]:
        (current.implementation_frame_counts[execution.active_implementation] ?? 0) + 1
    },
    transition_frame_count: current.transition_frame_count
      + (execution.transition ? 1 : 0),
    ...(nextRouting ? { routing: nextRouting } : {})
  };
}

function initialRoutingUsage(
  routing: NonNullable<
    NonNullable<PhysicalTrajectoryFrame["controller_execution"]>["routing"]
  >
): NonNullable<
  NonNullable<PhysicalTrajectorySummary["controller_usage"]>["routing"]
> {
  return {
    last_call_id: routing.call_id,
    decision_count: 1,
    admitted_count: routing.admitted ? 1 : 0,
    rejected_count: routing.admitted ? 0 : 1,
    cold_start_count: routing.cold_start ? 1 : 0,
    rejection_reason_counts: {
      insufficient_success_posterior:
        routing.reason === "insufficient_success_posterior" ? 1 : 0,
      entry_state_ood: routing.reason === "entry_state_ood" ? 1 : 0,
      command_ood: routing.reason === "command_ood" ? 1 : 0,
      memory_bridge_timeout:
        routing.reason === "memory_bridge_timeout" ? 1 : 0
    },
    memory_bridge_attempt_count: routing.memory_bridge ? 1 : 0,
    memory_bridge_completed_count:
      routing.memory_bridge?.phase === "completed" ? 1 : 0,
    memory_bridge_timeout_count:
      routing.memory_bridge?.phase === "timed_out" ? 1 : 0,
    memory_bridge_aborted_count:
      routing.memory_bridge?.phase === "aborted" ? 1 : 0,
    last_memory_bridge_phase: routing.memory_bridge?.phase ?? null
  };
}

function advanceRoutingUsage(
  current: NonNullable<
    NonNullable<PhysicalTrajectorySummary["controller_usage"]>["routing"]
  >,
  routing: NonNullable<
    NonNullable<PhysicalTrajectoryFrame["controller_execution"]>["routing"]
  >
): typeof current {
  const added = initialRoutingUsage(routing);
  return {
    last_call_id: routing.call_id,
    decision_count: current.decision_count + 1,
    admitted_count: current.admitted_count + added.admitted_count,
    rejected_count: current.rejected_count + added.rejected_count,
    cold_start_count: current.cold_start_count + added.cold_start_count,
    rejection_reason_counts: {
      insufficient_success_posterior:
        current.rejection_reason_counts.insufficient_success_posterior
          + added.rejection_reason_counts.insufficient_success_posterior,
      entry_state_ood: current.rejection_reason_counts.entry_state_ood
        + added.rejection_reason_counts.entry_state_ood,
      command_ood: current.rejection_reason_counts.command_ood
        + added.rejection_reason_counts.command_ood,
      memory_bridge_timeout:
        current.rejection_reason_counts.memory_bridge_timeout
          + added.rejection_reason_counts.memory_bridge_timeout
    },
    memory_bridge_attempt_count: current.memory_bridge_attempt_count
      + added.memory_bridge_attempt_count,
    memory_bridge_completed_count: current.memory_bridge_completed_count
      + added.memory_bridge_completed_count,
    memory_bridge_timeout_count: current.memory_bridge_timeout_count
      + added.memory_bridge_timeout_count,
    memory_bridge_aborted_count: current.memory_bridge_aborted_count
      + added.memory_bridge_aborted_count,
    last_memory_bridge_phase: added.last_memory_bridge_phase
  };
}

function advanceCurrentRoutingUsage(
  current: NonNullable<
    NonNullable<PhysicalTrajectorySummary["controller_usage"]>["routing"]
  >,
  routing: NonNullable<
    NonNullable<PhysicalTrajectoryFrame["controller_execution"]>["routing"]
  >
): typeof current {
  const phase = routing.memory_bridge?.phase ?? null;
  if (phase === current.last_memory_bridge_phase) return current;
  return {
    ...current,
    memory_bridge_completed_count: current.memory_bridge_completed_count
      + (phase === "completed" ? 1 : 0),
    memory_bridge_timeout_count: current.memory_bridge_timeout_count
      + (phase === "timed_out" ? 1 : 0),
    memory_bridge_aborted_count: current.memory_bridge_aborted_count
      + (phase === "aborted" ? 1 : 0),
    last_memory_bridge_phase: phase
  };
}

function aggregateContacts(
  contacts: HumanoidWorldSnapshot["robot"]["contacts"]
): Array<{ key: string; normal_force_n: number }> {
  const forceByKey = new Map<string, number>();
  for (const contact of contacts) {
    const key = contactKey(contact);
    forceByKey.set(key, rounded((forceByKey.get(key) ?? 0) + contact.normalForce));
  }
  return [...forceByKey].map(([key, normalForce]) => ({
    key,
    normal_force_n: normalForce
  })).sort((left, right) => compareCodePoints(left.key, right.key));
}

function contactKey(contact: HumanoidWorldSnapshot["robot"]["contacts"][number]): string {
  const first = contactParty(contact, "first");
  const second = contactParty(contact, "second");
  return [first, second].sort(compareCodePoints).join("<->");
}

function contactParty(
  contact: HumanoidWorldSnapshot["robot"]["contacts"][number],
  side: "first" | "second"
): string {
  const title = side === "first" ? "first" : "second";
  const values = [
    contact[`${title}Body`],
    contact[`${title}Object`],
    contact[`${title}Solid`],
    contact[`${title}HandLink`]
  ].filter((value): value is string => typeof value === "string");
  return values.length > 0 ? values.sort(compareCodePoints).join("+") : "environment";
}

function sameContactState(
  left: PhysicalTrajectoryFrame,
  right: PhysicalTrajectoryFrame
): boolean {
  return JSON.stringify(left.contacts.map(({ key }) => key))
    === JSON.stringify(right.contacts.map(({ key }) => key));
}

function endEffectorNumbers(value: number): Record<HumanoidEndEffector, number> {
  return Object.fromEntries(
    HUMANOID_END_EFFECTORS.map((name) => [name, value])
  ) as Record<HumanoidEndEffector, number>;
}

function vector(value: Vec3): Vec3 {
  return { x: rounded(value.x), y: rounded(value.y), z: rounded(value.z) };
}

function distance(left: Vec3 | undefined, right: Vec3): number {
  return left
    ? Math.hypot(right.x - left.x, right.y - left.y, right.z - left.z)
    : 0;
}

function rounded(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}

function nullableRounded(value: number | null): number | null {
  return value === null ? null : rounded(value);
}

function compareCodePoints(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}
