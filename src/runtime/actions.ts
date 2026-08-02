import { z } from "zod";
import {
  QuaternionSchema,
  SpatialMemoryKindSchema,
  Vec3Schema,
  VoxelCoordinateSchema,
  VoxelMaterialSchema,
  type BodyChannel,
  type JsonValue
} from "../domain/schema.js";
import {
  RapierWorld,
  type ArmJointTargets,
  type CommandResult,
  type SourceCommand
} from "../world/rapier-world.js";
import { ROBOT_SPEC } from "../world/robot-model.js";

const EmptyInput = z.object({}).strict();
const EntityInput = z.object({ entity_id: z.string().trim().min(1) }).strict();
const JointTargetsSchema = z.object({
  shoulder: z.number().finite().optional(),
  elbow: z.number().finite().optional(),
  wrist: z.number().finite().optional()
}).strict();
const MotionOptionsSchema = z.object({
  max_velocity: z.number().finite().positive().optional(),
  max_duration_seconds: z.number().finite().positive().optional(),
  tolerance: z.number().finite().positive().optional()
}).strict();

export const ToolInputs = {
  read_proprioception: EmptyInput,
  sense_scene: EmptyInput,
  survey_terrain: z.object({
    radius_cells: z.number().int().min(1).max(24).optional()
  }).strict(),
  scan_voxels: z.object({
    // The world clamps both values before scanning. Accepting a larger model
    // request here avoids an SDK-only validation failure that has no safety
    // value and never reaches the receipt/denial recovery path.
    radius: z.number().finite().positive().optional(),
    limit: z.number().int().min(1).optional()
  }).strict(),
  inspect_voxel: z.object({ coordinate: VoxelCoordinateSchema }).strict(),
  recall_spatial_memory: z.object({
    kind: SpatialMemoryKindSchema.optional(),
    near: Vec3Schema.optional(),
    radius: z.number().finite().positive().max(768).optional(),
    coordinate: VoxelCoordinateSchema.optional(),
    entity_id: z.string().trim().min(1).optional(),
    text: z.string().trim().min(1).max(160).optional(),
    limit: z.number().int().min(1).max(48).optional()
  }).strict().refine(
    (input) => input.radius === undefined || input.near !== undefined,
    "radius requires near"
  ),
  inspect_entity: EntityInput,
  query_contacts: EmptyInput,
  plan_base_path: z.object({
    target: Vec3Schema,
    face_point: Vec3Schema.optional()
  }).strict(),
  plan_arm_retraction: z.object({
    target: Vec3Schema,
    face_point: Vec3Schema.optional()
  }).strict(),
  plan_joint_targets: z.object({
    targets: JointTargetsSchema
  }).strict(),
  solve_end_effector_position: z.object({
    position: Vec3Schema,
    seed: JointTargetsSchema.optional()
  }).strict(),
  solve_end_effector_pose: z.object({
    position: Vec3Schema,
    orientation: QuaternionSchema.optional(),
    seed: JointTargetsSchema.optional()
  }).strict(),
  inspect_command: EmptyInput
} as const;

export type ToolName = keyof typeof ToolInputs;

const SkillInputs = {
  navigate_frontier: z.object({
    survey_transaction_id: z.string().trim().min(1),
    survey_world_revision: z.number().int().nonnegative(),
    choice_id: z.string().trim().min(1),
    target: Vec3Schema,
    face_point: Vec3Schema,
    options: MotionOptionsSchema.optional()
  }).strict(),
  execute_base_plan: z.object({
    plan_id: z.string().trim().min(1),
    options: MotionOptionsSchema.optional()
  }).strict(),
  execute_joint_plan: z.object({
    plan_id: z.string().trim().min(1),
    options: MotionOptionsSchema.optional()
  }).strict(),
  drive_base: z.object({
    linear_meters_per_second: z.number().finite(),
    angular_radians_per_second: z.number().finite(),
    duration_seconds: z.number().finite().positive()
  }).strict(),
  set_head_target: z.object({
    yaw: z.number().finite(),
    pitch: z.number().finite(),
    options: MotionOptionsSchema.optional()
  }).strict(),
  set_joint_targets: z.object({
    targets: JointTargetsSchema,
    options: MotionOptionsSchema.optional()
  }).strict(),
  set_gripper_target: z.object({
    aperture: z.number().finite().nonnegative(),
    max_force: z.number().finite().positive().optional(),
    options: MotionOptionsSchema.optional()
  }).strict(),
  break_voxel: z.object({ coordinate: VoxelCoordinateSchema }).strict(),
  place_voxel: z.object({
    coordinate: VoxelCoordinateSchema,
    material: VoxelMaterialSchema
  }).strict()
} as const;

export const AgentSkillInputs = {
  ...SkillInputs,
  navigate_frontier: z.object({
    survey_transaction_id: z.string().trim().min(1),
    choice_id: z.string().trim().min(1),
    options: MotionOptionsSchema.optional()
  }).strict(),
  execute_base_plan: z.object({
    planning_transaction_id: z.string().trim().min(1),
    options: MotionOptionsSchema.optional()
  }).strict(),
  execute_joint_plan: z.object({
    planning_transaction_id: z.string().trim().min(1),
    options: MotionOptionsSchema.optional()
  }).strict()
} as const;

export type SkillName = keyof typeof SkillInputs;

export const ToolDescriptions: Record<ToolName, string> = {
  read_proprioception: "Read the current base pose, authoritative joint positions, limits and targets, link transforms, wheel odometry, gripper state, contacts, and attachment. The reading only changes when a body command changes the world, so calling it repeatedly without acting in between returns the identical snapshot and is refused as a repeated action.",
  sense_scene: "Read entities currently visible to the articulated sensor and the known static world geometry. In a world larger than the sensor range this reports only what is in view right now, which early in a run is often nothing; use survey_terrain to decide where to go looking.",
  survey_terrain: "Read the voxel terrain around the base as a local height map of everything the robot has actually seen so far, plus a motion-entropy-ordered set of reachable frontier choices. Every choice contains a choice_id, target, face_point, travel distance, turn amount and information gain ready for model selection. For exploration, pass the survey transaction and selected choice_id to navigate_frontier; use plan_base_path for other observed world-space targets. The ordering varies independently between new runs, but the harness never selects or executes a choice. Rows are text, one character per cell: '.' walkable floor, '1'-'9' a solid column that many blocks high, '?' not yet seen, '#' outside the world. Optional radius_cells sets how far the local map extends, defaulting to 12.",
  scan_voxels: "Scan source-backed exposed voxel faces inside the articulated head sensor's real range and field of view. Each result contains an exact integer coordinate, material, chunk, face interaction points, adjacent placement coordinates and current gripper reach. Use these coordinates rather than inventing a block location. The backend returns only chunks actually loaded in physics and safely caps the effective radius to sensor range and results to 48, even when a larger positive request is supplied.",
  inspect_voxel: "Read the current material, world-space center, chunk, interaction geometry, support state, voxel revision, and Recast-validated reachable_standoff_poses for one exact coordinate. Each standoff already pairs a base target with the voxel face_point so the planar arm can be aligned before IK. For an occupied cell, use an exposed_faces interaction_point before breaking. For an empty supported cell, use a placement_interaction_points interaction_point before placing; integer column/level/row values are voxel indexes, never world-space arm positions. It does not edit the world. A coordinate should come from scan_voxels, the structured goal, or an accepted voxel receipt.",
  recall_spatial_memory: "Query durable structured spatial memory built only from accepted action receipts and authoritative world snapshots. Filter by kind, exact voxel coordinate, entity id, text, or a world-space near/radius region. Every result includes the world and voxel revision plus its source transaction and agent, so stale observations remain identifiable rather than being presented as current sensor truth. Use this before revisiting an area or entity, then re-observe before acting when the remembered revision is old.",
  inspect_entity: "Read current source-backed state for one visible entity by exact identifier. The result includes reachable_standoff_poses: base poses the navmesh actually accepts around that entity, each already paired with the face_point to pass to plan_base_path. Prefer one of those over a guessed approach position. For a portable object it also includes grasp_pose: the exact gripper-center position to pass to solve_end_effector_position, with the aperture to open to before descending and the aperture that holds the object. Use those measured values rather than deriving a grasp point from the object's size.",
  query_contacts: "Read the current Rapier contact pairs for the robot links and gripper fingers.",
  plan_base_path: "Build a dynamically obstacle-aware navmesh path for the full articulated rig from the current pose to a world-space base target. Optional face_point is another world-space point the base must face after arrival, not a direction vector. The ground plane uses x/z; y is vertical and cannot create a lateral detour. Portable obstacles are expanded by the mobile-base footprint, so a manipulation target must be a clear standoff position rather than an object center. An accepted result reports both the requested target and the collision-safe resolved target after Recast projection and Rapier sweep validation. A base_path_unavailable denial reports nearest_reachable_alternatives: retry with one of those targets rather than nudging the rejected one. A base_path_collision reports collision_segments from the current articulated posture; when several targets name the same non-base link, that body channel must be reconfigured before more base planning, and a leaf lacking that capability must report_blocked.",
  plan_arm_retraction: "For one model-selected base target and optional face_point, search current joint limits for arm postures that both have a continuous Rapier collision-free trajectory from the current joints and make the entire requested Recast base route pass an articulated swept-volume check. It returns a bounded ranked choice set with exact shoulder, elbow, and wrist targets. It never moves the robot, never selects a candidate, and never stores a base plan: the model must choose one option, call set_joint_targets, then call plan_base_path again at the new world revision. Use this after base_path_collision repeatedly names an arm, wrist, or finger link rather than guessing joint angles.",
  plan_joint_targets: "Plan, but do not execute, a collision-checked trajectory to model-selected shoulder, elbow, and wrist targets in robot joint space. Execute its accepted receipt with execute_joint_plan. Because this plan owns a relative joint posture rather than a fixed world-space gripper point, it is the arm planner to use when an independent base worker will move concurrently. Fixed world-space manipulation still requires solve_end_effector_position or solve_end_effector_pose and must not run beside base motion.",
  solve_end_effector_position: "Plan a collision-checked arm trajectory to one observed world-space gripper position while deliberately leaving wrist orientation unconstrained. Use this for voxel interaction_points, ordinary object grasp_pose positions, lifts, and other tasks whose criterion is a point rather than a rotation. The schema cannot add an orientation constraint. Execute the accepted planning receipt with execute_joint_plan; this tool never moves the arm itself.",
  solve_end_effector_pose: "Plan a collision-checked arm trajectory to a world-space gripper position and an optional explicit quaternion orientation. Use this only when the mission physically requires a particular rotation, such as aligning a keyed affordance. For voxel faces, ordinary grasps, lifts, and any point-only reach, use solve_end_effector_position instead so an unnecessary identity quaternion cannot overconstrain the three-joint planar arm. The requested point must come from current observation, and this planner never actuates.",
  inspect_command: "Read the active or most recently completed body command and its source, phase, focus, and physical result."
};

export const SkillDescriptions: Record<SkillName, string> = {
  navigate_frontier: "Physically navigate to one model-selected reachable frontier from a current accepted survey_terrain receipt. Pass that receipt's exact transaction_id and one exact choice_id returned inside it. The harness validates ownership, world revision and candidate identity, then atomically plans and executes only that chosen target through Recast and Rapier. It never selects a choice, substitutes another frontier, retries automatically, or moves without this explicit model call. After any body command, survey again before the next frontier movement.",
  execute_base_plan: "Execute one unconsumed base-footprint path plan from an accepted plan_base_path receipt granted to this agent. Pass that receipt's exact transaction_id as planning_transaction_id. The plan_id inside the receipt's result is the world's internal handle and is never accepted here. Every step also collision-checks the current articulated links and attached payload; a named-link collision requires a model-chosen joint reconfiguration or different target before replanning. A plan stays valid across any number of observations and expires only when a body command changes the world; plan_already_consumed and stale_plan_revision both mean plan_base_path must be called again from the current pose.",
  execute_joint_plan: "Execute one unconsumed arm plan from an accepted plan_joint_targets, solve_end_effector_position, or solve_end_effector_pose receipt granted to this agent. Pass that receipt's exact transaction_id as planning_transaction_id. The plan_id inside the receipt's result is the world's internal handle and is never accepted here. It moves only the arm joints returned by that plan. A joint-target plan may execute beside an independently leased moving base; a fixed world-space IK plan may not. A plan stays valid across observations and expires when a body command changes the world before execution begins.",
  drive_base: "Drive the mobile base with explicit linear and angular velocity for a bounded duration. Positive linear velocity moves along the current forward heading; negative linear velocity reverses without first turning, which permits a model-selected retreat from a close manipulation pose.",
  set_head_target: "Move the sensor head to explicit yaw and pitch targets within joint limits.",
  set_joint_targets: "Move any supplied shoulder, elbow, and wrist joints to model-selected targets within their reported limits, including reconfiguring the arm's swept envelope before base motion.",
  set_gripper_target: "Move the two-sided gripper to an explicit aperture with an optional force limit. Center the gripper on the current object pose before closing: one-sided contact can physically push an object but cannot attach it. Open wider than the object before descending onto it, then close to just under its narrowest width; closing to the gripper minimum crushes it and is denied with gripper_force_limit, which reports the width that would have held. Closing creates an attachment only after both Rapier fingers contact the same portable object; opening removes that physical constraint. An attached object still rests on the ground until the arm lifts it, and a payload touching the ground blocks every base plan, so raise the end effector after grasping and before driving.",
  break_voxel: "Remove one visible existing voxel only when the physical gripper is within the reported interaction distance. The accepted command mutates the authoritative 3D voxel store, adds the recovered material to inventory, rebuilds the affected Rapier chunk collider and rebuilds navigation before returning. Choose the coordinate from scan_voxels and move the arm to its interaction_point first.",
  place_voxel: "Place one inventory voxel at an empty, supported placement_coordinate reported by scan_voxels or the structured goal, with the gripper physically in reach and the requested volume clear of robot links, payloads and objects. Inspect the empty target and move the gripper to one reported placement_interaction_points interaction_point before calling this tool; never use the integer voxel coordinate or the future block center as an arm position. Acceptance mutates the voxel store and updates physics, navigation, checkpoint and live chunk rendering; no local builder substitutes another location."
};

const SkillChannels: Record<SkillName, BodyChannel[]> = {
  navigate_frontier: ["base"],
  execute_base_plan: ["base"],
  execute_joint_plan: ["arm"],
  drive_base: ["base"],
  set_head_target: ["head"],
  set_joint_targets: ["arm"],
  set_gripper_target: ["gripper"],
  break_voxel: ["arm", "gripper"],
  place_voxel: ["arm", "gripper"]
};

export function requiredChannels(name: SkillName, _rawInput: unknown): BodyChannel[] {
  return [...SkillChannels[name]];
}

export async function executeTool(
  world: RapierWorld,
  name: string,
  rawInput: unknown,
  extensions?: {
    recallSpatialMemory: (
      input: z.infer<typeof ToolInputs.recall_spatial_memory>
    ) => CommandResult;
  }
): Promise<CommandResult> {
  if (!isToolName(name)) return rejected("unknown_tool", { name });
  if (name === "read_proprioception") {
    ToolInputs.read_proprioception.parse(rawInput);
    const robot = world.snapshot().robot;
    return accepted("proprioception", {
      ...robot,
      kinematic_model: {
        base: {
          center_height: ROBOT_SPEC.base.centerY,
          half_extents: ROBOT_SPEC.base.halfExtents,
          corner_radius: ROBOT_SPEC.base.cornerRadius,
          footprint_radius: ROBOT_SPEC.base.footprintRadius,
          maximum_linear_velocity: ROBOT_SPEC.base.maximumLinearVelocity,
          maximum_angular_velocity: ROBOT_SPEC.base.maximumAngularVelocity,
          differential_drive: {
            wheel_radius: ROBOT_SPEC.wheels.radius,
            track_width: ROBOT_SPEC.wheels.trackWidth
          }
        },
        arm_motion_plane: "base_forward_vertical_plane",
        base_forward: {
          x: Math.sin(robot.yaw),
          y: 0,
          z: Math.cos(robot.yaw)
        },
        shoulder_height: ROBOT_SPEC.arm.shoulderHeight,
        shoulder_forward_offset: ROBOT_SPEC.arm.shoulderForwardOffset,
        upper_arm_length: ROBOT_SPEC.arm.upperLength,
        forearm_length: ROBOT_SPEC.arm.forearmLength,
        wrist_length: ROBOT_SPEC.arm.wristLength,
        gripper: {
          aperture_minimum: ROBOT_SPEC.joints.gripper_aperture.minimum,
          aperture_maximum: ROBOT_SPEC.joints.gripper_aperture.maximum,
          finger_half_extents: ROBOT_SPEC.gripper.fingerHalfExtents,
          attachment_requires_same_object_bilateral_contact: true
        }
      }
    } as unknown as JsonValue);
  }
  if (name === "sense_scene") {
    ToolInputs.sense_scene.parse(rawInput);
    return accepted("scene_observation", world.observe());
  }
  if (name === "survey_terrain") {
    const input = ToolInputs.survey_terrain.parse(rawInput);
    return world.surveyTerrain(input.radius_cells ?? 12);
  }
  if (name === "scan_voxels") {
    const input = ToolInputs.scan_voxels.parse(rawInput);
    return world.scanVoxels(input.radius ?? 8, input.limit ?? 24);
  }
  if (name === "inspect_voxel") {
    const input = ToolInputs.inspect_voxel.parse(rawInput);
    return world.inspectVoxel(input.coordinate);
  }
  if (name === "recall_spatial_memory") {
    const input = ToolInputs.recall_spatial_memory.parse(rawInput);
    return extensions?.recallSpatialMemory(input)
      ?? rejected("spatial_memory_context_unavailable", {
        recovery: "Spatial memory is owned by a mission harness, not a standalone physics world."
      });
  }
  if (name === "inspect_entity") {
    const input = ToolInputs.inspect_entity.parse(rawInput);
    return world.inspectEntity(input.entity_id);
  }
  if (name === "query_contacts") {
    ToolInputs.query_contacts.parse(rawInput);
    return world.queryContacts();
  }
  if (name === "plan_base_path") {
    const input = ToolInputs.plan_base_path.parse(rawInput);
    return world.planBasePath(input.target, input.face_point);
  }
  if (name === "plan_arm_retraction") {
    const input = ToolInputs.plan_arm_retraction.parse(rawInput);
    return world.planArmRetraction(input.target, input.face_point);
  }
  if (name === "plan_joint_targets") {
    const input = ToolInputs.plan_joint_targets.parse(rawInput);
    return world.planJointTargets(jointTargets(input.targets));
  }
  if (name === "solve_end_effector_position") {
    const input = ToolInputs.solve_end_effector_position.parse(rawInput);
    return world.solveEndEffectorPosition(
      input.position,
      input.seed !== undefined ? jointTargets(input.seed) : undefined
    );
  }
  if (name === "solve_end_effector_pose") {
    const input = ToolInputs.solve_end_effector_pose.parse(rawInput);
    return world.solveEndEffector({
      position: input.position,
      ...(input.orientation !== undefined ? { orientation: input.orientation } : {}),
      ...(input.seed !== undefined ? { seed: jointTargets(input.seed) } : {})
    });
  }
  ToolInputs.inspect_command.parse(rawInput);
  return world.inspectCommand();
}

export async function executeSkill(
  world: RapierWorld,
  command: SourceCommand,
  name: string,
  rawInput: unknown
): Promise<CommandResult> {
  if (!isSkillName(name)) return rejected("unknown_skill", { name });
  if (name === "navigate_frontier") {
    const input = SkillInputs.navigate_frontier.parse(rawInput);
    const source = {
      survey_transaction_id: input.survey_transaction_id,
      choice_id: input.choice_id,
      selected_target: input.target,
      selected_face_point: input.face_point
    };
    const currentWorldRevision = world.snapshot().world_revision;
    if (currentWorldRevision !== input.survey_world_revision) {
      return rejected("stale_survey_revision", {
        ...source,
        surveyed_world_revision: input.survey_world_revision,
        current_world_revision: currentWorldRevision,
        recovery: "The world changed before execution began. Call survey_terrain again and make a fresh model choice from the new frontier set."
      });
    }
    const planned = world.planBasePath(input.target, input.face_point);
    const planningDetail = objectDetail(planned.detail);
    if (!planned.accepted) {
      return {
        ...planned,
        detail: { ...source, phase: "planning", ...planningDetail }
      };
    }
    const planId = planningDetail.plan_id;
    if (typeof planId !== "string" || planId.length === 0) {
      return rejected("frontier_plan_missing_id", {
        ...source,
        phase: "planning",
        planning_result: planned.detail
      });
    }
    const executed = await world.executeBasePlan(
      command,
      planId,
      motionOptions(input.options)
    );
    return {
      ...executed,
      detail: {
        ...source,
        ...objectDetail(executed.detail),
        requested_target: planningDetail.requested_target ?? input.target,
        resolved_target: planningDetail.resolved_target ?? input.target,
        face: planningDetail.face ?? input.face_point,
        planning_distance: planningDetail.distance ?? null
      }
    };
  }
  if (name === "execute_base_plan") {
    const input = SkillInputs.execute_base_plan.parse(rawInput);
    return world.executeBasePlan(command, input.plan_id, motionOptions(input.options));
  }
  if (name === "drive_base") {
    const input = SkillInputs.drive_base.parse(rawInput);
    return world.driveBase(
      command,
      input.linear_meters_per_second,
      input.angular_radians_per_second,
      input.duration_seconds
    );
  }
  if (name === "execute_joint_plan") {
    const input = SkillInputs.execute_joint_plan.parse(rawInput);
    return world.executeJointPlan(command, input.plan_id, motionOptions(input.options));
  }
  if (name === "set_head_target") {
    const input = SkillInputs.set_head_target.parse(rawInput);
    return world.setHeadTarget(command, input.yaw, input.pitch, motionOptions(input.options));
  }
  if (name === "set_joint_targets") {
    const input = SkillInputs.set_joint_targets.parse(rawInput);
    return world.executeJointTargets(command, jointTargets(input.targets), motionOptions(input.options));
  }
  if (name === "set_gripper_target") {
    const input = SkillInputs.set_gripper_target.parse(rawInput);
    return world.setGripperTarget(
      command,
      input.aperture,
      input.max_force,
      motionOptions(input.options)
    );
  }
  if (name === "break_voxel") {
    const input = SkillInputs.break_voxel.parse(rawInput);
    return world.breakVoxel(command, input.coordinate);
  }
  if (name === "place_voxel") {
    const input = SkillInputs.place_voxel.parse(rawInput);
    return world.placeVoxel(command, input.coordinate, input.material);
  }
  return assertUnreachable(name);
}

function isToolName(name: string): name is ToolName {
  return Object.hasOwn(ToolInputs, name);
}

export function isSkillName(name: string): name is SkillName {
  return Object.hasOwn(SkillInputs, name);
}

function motionOptions(input: z.infer<typeof MotionOptionsSchema> | undefined) {
  if (!input) return undefined;
  return {
    ...(input.max_velocity !== undefined ? { maxVelocity: input.max_velocity } : {}),
    ...(input.max_duration_seconds !== undefined
      ? { maxDurationSeconds: input.max_duration_seconds }
      : {}),
    ...(input.tolerance !== undefined ? { tolerance: input.tolerance } : {})
  };
}

function jointTargets(input: z.infer<typeof JointTargetsSchema>): Partial<ArmJointTargets> {
  return {
    ...(input.shoulder !== undefined ? { shoulder: input.shoulder } : {}),
    ...(input.elbow !== undefined ? { elbow: input.elbow } : {}),
    ...(input.wrist !== undefined ? { wrist: input.wrist } : {})
  };
}

function objectDetail(value: JsonValue): Record<string, JsonValue> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value
    : { value };
}

function assertUnreachable(value: never): never {
  throw new Error(`Unhandled skill: ${String(value)}`);
}

function accepted(code: string, detail: CommandResult["detail"]): CommandResult {
  return { accepted: true, code, detail };
}

function rejected(code: string, detail: CommandResult["detail"]): CommandResult {
  return { accepted: false, code, detail };
}
