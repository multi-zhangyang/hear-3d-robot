import { z } from "zod";
import { Vec3Schema } from "./schema.js";
import type {
  HumanoidLearnedPolicyCapability
} from "./humanoid-policy.js";

export const HUMANOID_SKILL_IDS = [
  "navigate_to_zone",
  "explore",
  "break_block",
  "approach",
  "reach",
  "grasp",
  "lift",
  "carry",
  "carry_to_zone",
  "place",
  "push",
  "pull",
  "press",
  "open",
  "close",
  "turn",
  "regrasp",
  "bimanual_support",
  "bimanual_carry",
  "stabilize",
  "retreat"
] as const;

export const HumanoidSkillIdSchema = z.enum(HUMANOID_SKILL_IDS);
export type HumanoidSkillId = z.infer<typeof HumanoidSkillIdSchema>;

const HandSchema = z.enum(["left", "right"]);
const HandsSchema = z.enum(["left", "right", "both"]);
const ObjectIdSchema = z.string().trim().min(1);
const InteractionPointIdSchema = z.string().trim().min(1);
const DirectionSchema = Vec3Schema.refine(
  (value) => Math.abs(Math.hypot(value.x, value.y, value.z) - 1) <= 1e-3,
  "direction must be normalized"
);

const ObjectPointParameters = {
  object_id: ObjectIdSchema,
  interaction_point_id: InteractionPointIdSchema.nullable()
} as const;

const PlacementDestinationSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("semantic_zone"),
    zone_id: ObjectIdSchema,
    tolerance_m: z.number().finite().nonnegative().max(1)
  }).strict(),
  z.object({
    type: z.literal("support_surface"),
    object_id: ObjectIdSchema,
    local_target: Vec3Schema.nullable()
  }).strict(),
  z.object({
    type: z.literal("container"),
    object_id: ObjectIdSchema,
    local_target: Vec3Schema.nullable()
  }).strict(),
  z.object({
    type: z.literal("slot"),
    object_id: ObjectIdSchema,
    interaction_point_id: InteractionPointIdSchema,
    insertion_depth_m: z.number().finite().positive()
  }).strict(),
  z.object({
    type: z.literal("world_pose"),
    position: Vec3Schema,
    position_tolerance_m: z.number().finite().positive()
  }).strict()
]);

export const HumanoidSkillInvocationSchema = z.discriminatedUnion("skill", [
  z.object({
    skill: z.literal("navigate_to_zone"),
    zone_id: ObjectIdSchema
  }).strict(),
  z.object({
    skill: z.literal("explore"),
    frontier_id: z.string().regex(/^frontier:\d+:\d+$/),
    strategy: z.enum(["information_gain", "balanced", "coverage"]),
    maximum_travel_m: z.number().finite().positive().max(50)
  }).strict(),
  z.object({
    skill: z.literal("break_block"),
    solid_id: ObjectIdSchema,
    hand: HandSchema,
    strategy: z.enum(["strike", "press"]),
    approach_clearance_m: z.number().finite().min(0.2).max(1.2)
  }).strict(),
  z.object({
    skill: z.literal("approach"),
    ...ObjectPointParameters,
    hand: HandSchema.nullable().optional(),
    standoff_m: z.number().finite().min(0.15).max(1.5)
  }).strict(),
  z.object({
    skill: z.literal("reach"),
    ...ObjectPointParameters,
    hand: HandSchema,
    tolerance_m: z.number().finite().positive().max(0.25)
  }).strict(),
  z.object({
    skill: z.literal("grasp"),
    ...ObjectPointParameters,
    hand: HandSchema
  }).strict(),
  z.object({
    skill: z.literal("lift"),
    object_id: ObjectIdSchema,
    hand: HandSchema,
    clearance_m: z.number().finite().positive().max(1)
  }).strict(),
  z.object({
    skill: z.literal("carry"),
    object_id: ObjectIdSchema,
    hands: HandsSchema,
    target: Vec3Schema,
    tolerance_m: z.number().finite().positive().max(1)
  }).strict(),
  z.object({
    skill: z.literal("carry_to_zone"),
    object_id: ObjectIdSchema,
    hands: HandsSchema,
    zone_id: ObjectIdSchema,
    tolerance_m: z.number().finite().positive().max(1)
  }).strict(),
  z.object({
    skill: z.literal("place"),
    object_id: ObjectIdSchema,
    hands: HandsSchema,
    destination: PlacementDestinationSchema,
    release_after_settled: z.boolean()
  }).strict(),
  z.object({
    skill: z.literal("push"),
    ...ObjectPointParameters,
    hand: HandSchema,
    direction_world: DirectionSchema,
    distance_m: z.number().finite().positive().max(2)
  }).strict(),
  z.object({
    skill: z.literal("pull"),
    ...ObjectPointParameters,
    hand: HandSchema,
    direction_world: DirectionSchema,
    distance_m: z.number().finite().positive().max(2)
  }).strict(),
  z.object({
    skill: z.literal("press"),
    ...ObjectPointParameters,
    hand: HandSchema,
    travel_m: z.number().finite().positive().max(0.3)
  }).strict(),
  z.object({
    skill: z.literal("open"),
    ...ObjectPointParameters,
    joint_id: ObjectIdSchema,
    hand: HandSchema,
    minimum_open_fraction: z.number().finite().min(0.5).max(1)
  }).strict(),
  z.object({
    skill: z.literal("close"),
    ...ObjectPointParameters,
    joint_id: ObjectIdSchema,
    hand: HandSchema,
    maximum_open_fraction: z.number().finite().min(0).max(0.5)
  }).strict(),
  z.object({
    skill: z.literal("turn"),
    ...ObjectPointParameters,
    joint_id: ObjectIdSchema,
    hand: HandSchema,
    direction: z.enum(["increasing", "decreasing"]),
    rotation_radians: z.number().finite().positive().max(Math.PI * 2)
  }).strict(),
  z.object({
    skill: z.literal("regrasp"),
    object_id: ObjectIdSchema,
    interaction_point_id: InteractionPointIdSchema,
    from_hand: HandSchema,
    to_hand: HandSchema,
    excluded_interaction_point_ids: z.array(InteractionPointIdSchema)
  }).strict().superRefine((value, context) => {
    if (value.from_hand === value.to_hand) {
      context.addIssue({
        code: "custom",
        path: ["to_hand"],
        message: "regrasp must transfer support to the other hand"
      });
    }
    if (value.excluded_interaction_point_ids.includes(value.interaction_point_id)) {
      context.addIssue({
        code: "custom",
        path: ["interaction_point_id"],
        message: "regrasp target cannot be one of the excluded grasp points"
      });
    }
  }),
  z.object({
    skill: z.literal("bimanual_support"),
    object_id: ObjectIdSchema,
    left_interaction_point_id: InteractionPointIdSchema,
    right_interaction_point_id: InteractionPointIdSchema
  }).strict().refine(
    (value) => value.left_interaction_point_id !== value.right_interaction_point_id,
    { path: ["right_interaction_point_id"], message: "hands need distinct support points" }
  ),
  z.object({
    skill: z.literal("bimanual_carry"),
    object_id: ObjectIdSchema,
    target: Vec3Schema,
    tolerance_m: z.number().finite().positive().max(1)
  }).strict(),
  z.object({
    skill: z.literal("stabilize"),
    minimum_support_margin_m: z.number().finite().nonnegative().max(0.3)
  }).strict(),
  z.object({
    skill: z.literal("retreat"),
    target: Vec3Schema,
    minimum_obstacle_clearance_m: z.number().finite().positive().max(2)
  }).strict()
]);

export type HumanoidSkillInvocation = z.infer<typeof HumanoidSkillInvocationSchema>;

export function humanoidSkillPhaseLearnedPolicyCapabilities(
  invocation: HumanoidSkillInvocation,
  phase: string
): HumanoidLearnedPolicyCapability[] {
  const process = HUMANOID_SKILL_CONTRACTS[invocation.skill].process.find(
    (entry) => entry.phase === phase
  );
  if (!process) return [];
  const capabilities = [...process.learned_policy_capabilities];
  if ((invocation.skill === "carry" || invocation.skill === "carry_to_zone"
      || invocation.skill === "place")
    && invocation.hands === "both") {
    capabilities.push("bimanual_manipulation");
  }
  return [...new Set(capabilities)];
}

export const BeginHumanoidSkillSchema = z.object({
  skill_plan_transaction_id: z.string().trim().min(1).nullable().default(null),
  skill_node_id: z.string().trim().min(1).nullable().default(null),
  invocation: HumanoidSkillInvocationSchema,
  phase: z.string().trim().min(1)
}).strict();

export type BeginHumanoidSkill = z.infer<typeof BeginHumanoidSkillSchema>;

export const HUMANOID_SKILL_FAILURE_CODES = [
  "precondition_failed",
  "target_unobserved",
  "affordance_missing",
  "interaction_point_missing",
  "unreachable",
  "path_blocked",
  "contact_missing",
  "grasp_unstable",
  "object_slipped",
  "articulation_stalled",
  "placement_misaligned",
  "unexpected_world_change",
  "collision_risk",
  "balance_lost"
] as const;

export type HumanoidSkillFailureCode = typeof HUMANOID_SKILL_FAILURE_CODES[number];

export interface HumanoidSkillContract {
  id: HumanoidSkillId;
  parameters: string[];
  required_affordances: string[];
  preconditions: string[];
  prerequisite_skill_groups: HumanoidSkillId[][];
  process: Array<{
    phase: string;
    authority: "sensor" | "navigation" | "whole_body" | "grasp" | "checker";
    learned_policy_capabilities: HumanoidLearnedPolicyCapability[];
  }>;
  success_conditions: string[];
  failure_reasons: HumanoidSkillFailureCode[];
  recovery_entry: HumanoidSkillId[];
}

const COMMON_FAILURES: HumanoidSkillFailureCode[] = [
  "target_unobserved",
  "unexpected_world_change",
  "collision_risk",
  "balance_lost"
];

export const HUMANOID_SKILL_CONTRACTS: Readonly<Record<
  HumanoidSkillId,
  HumanoidSkillContract
>> = Object.freeze({
  navigate_to_zone: contract("navigate_to_zone", ["zone_id"], [],
    ["zone exists in the current observation", "route into the zone is physically reachable"],
    [["observe", "sensor"], ["enter_zone", "navigation"], ["verify_zone_membership", "checker"]],
    ["robot enters the selected zone while remaining upright"],
    ["path_blocked", "unreachable"], ["navigate_to_zone", "explore", "retreat"]),
  explore: contract("explore", ["frontier_id", "strategy", "maximum_travel_m"], [],
    ["frontier exists in the current spatial belief", "route is physically reachable"],
    [["observe", "sensor"], ["route_to_frontier", "navigation"], ["survey", "sensor"], ["verify_information_gain", "checker"]],
    ["robot reaches the selected frontier and the spatial belief gains observations"],
    ["path_blocked", "unreachable"], ["explore", "retreat"]),
  break_block: contract("break_block", ["solid_id", "hand", "strategy", "approach_clearance_m"], [],
    ["block is currently visible", "selected hand can reach an exposed block face"],
    [["observe", "sensor"], ["approach", "navigation"], ["contact", "whole_body"], ["verify_contact", "checker"]],
    ["selected hand sustains removal-authorizing contact with the selected block"],
    ["path_blocked", "contact_missing", "unreachable"], ["approach", "stabilize", "retreat"]),
  approach: contract("approach", ["object_id", "interaction_point_id", "hand", "standoff_m"], [],
    ["target observable", "reachable base placement exists"],
    [["observe", "sensor"], ["route", "navigation"], ["verify_standoff", "checker"]],
    ["base reaches a collision-free manipulation stance"],
    ["path_blocked", "unreachable"], ["retreat", "approach"]),
  reach: contract("reach", ["object_id", "interaction_point_id", "hand", "tolerance_m"], [],
    ["target observable", "interaction point compatible with hand", "robot balanced"],
    [["observe", "sensor"], ["solve_whole_body_reach", "whole_body"], ["verify_pose", "checker"]],
    ["selected hand reaches the live interaction point"],
    ["interaction_point_missing", "unreachable"], ["approach", "stabilize", "reach"]),
  grasp: contract("grasp", ["object_id", "interaction_point_id", "hand"], ["graspable"],
    ["reach satisfied", "selected hand free", "two contact surfaces available"],
    [["observe", "sensor"], ["close_hand_under_contact", "grasp"], ["verify_grasp", "checker"]],
    ["opposed contact, stable relative pose and lift evidence verified"],
    ["contact_missing", "grasp_unstable", "object_slipped"], ["regrasp", "reach"]),
  lift: contract("lift", ["object_id", "hand", "clearance_m"], ["graspable"],
    ["verified grasp exists", "robot balanced"],
    [["observe", "sensor"], ["lift_whole_body", "whole_body"], ["verify_clearance", "checker"]],
    ["object clears its previous support while grasp remains verified"],
    ["object_slipped", "unreachable"], ["regrasp", "stabilize"]),
  carry: contract("carry", ["object_id", "hands", "target", "tolerance_m"], ["movable"],
    ["verified carried-object binding exists", "route available"],
    [["observe", "sensor"], ["carry_route", "navigation"], ["verify_binding", "checker"]],
    ["robot reaches target and carried-object continuation stays verified"],
    ["path_blocked", "object_slipped"], ["regrasp", "bimanual_support", "retreat"]),
  carry_to_zone: contract("carry_to_zone", ["object_id", "hands", "zone_id", "tolerance_m"], ["movable"],
    ["verified carried-object binding exists", "semantic destination zone is observable", "route available"],
    [["observe", "sensor"], ["carry_route", "navigation"], ["verify_binding", "checker"]],
    ["carried object reaches the semantic zone while continuation stays verified"],
    ["path_blocked", "object_slipped"], ["regrasp", "bimanual_support", "retreat"]),
  place: contract("place", ["object_id", "hands", "destination", "release_after_settled"], ["movable"],
    ["verified carried-object binding exists", "destination object, slot, pose or semantic zone is observable"],
    [["observe", "sensor"], ["align_destination", "whole_body"], ["lower", "whole_body"], ["settle_and_release", "grasp"], ["verify_relation", "checker"]],
    ["destination relation satisfied; object settles after release or remains physically grasped when release is not requested"],
    ["placement_misaligned", "object_slipped"], ["regrasp", "place"]),
  push: contract("push", ["object_id", "interaction_point_id", "hand", "direction_world", "distance_m"], ["pushable"],
    ["push point observable", "force direction feasible"],
    [["observe", "sensor"], ["establish_contact", "whole_body"], ["apply_force", "whole_body"], ["verify_displacement", "checker"]],
    ["object or articulation moves in the requested direction"],
    ["contact_missing", "articulation_stalled"], ["approach", "pull", "stabilize"]),
  pull: contract("pull", ["object_id", "interaction_point_id", "hand", "direction_world", "distance_m"], ["pullable"],
    ["pull point observable", "contact or grasp established"],
    [["observe", "sensor"], ["establish_contact", "grasp"], ["apply_force", "whole_body"], ["verify_displacement", "checker"]],
    ["object or articulation moves in the requested direction"],
    ["contact_missing", "articulation_stalled"], ["regrasp", "approach", "push"]),
  press: contract("press", ["object_id", "interaction_point_id", "hand", "travel_m"], ["pressable"],
    ["press point observable", "hand path reachable"],
    [["observe", "sensor"], ["press_stroke", "whole_body"], ["verify_joint", "checker"]],
    ["press articulation reaches its activation range"],
    ["contact_missing", "articulation_stalled"], ["approach", "reach"]),
  open: contract("open", ["object_id", "interaction_point_id", "joint_id", "hand", "minimum_open_fraction"], ["openable"],
    ["joint and handle observable", "joint not already open"],
    [["observe", "sensor"], ["reach_handle", "whole_body"], ["establish_grasp", "grasp"], ["actuate_joint", "whole_body"], ["verify_open", "checker"]],
    ["joint open fraction reaches requested threshold"],
    ["articulation_stalled", "contact_missing"], ["regrasp", "approach", "stabilize"]),
  close: contract("close", ["object_id", "interaction_point_id", "joint_id", "hand", "maximum_open_fraction"], ["closeable"],
    ["joint and interaction point observable", "joint not already closed"],
    [["observe", "sensor"], ["reach_interaction", "whole_body"], ["establish_grasp", "grasp"], ["actuate_joint", "whole_body"], ["verify_closed", "checker"]],
    ["joint open fraction falls below requested threshold"],
    ["articulation_stalled", "contact_missing"], ["approach", "stabilize"]),
  turn: contract("turn", ["object_id", "interaction_point_id", "joint_id", "hand", "direction", "rotation_radians"], ["rotatable"],
    ["joint and turn point observable", "requested rotation is inside the joint range"],
    [["observe", "sensor"], ["reach_turn_point", "whole_body"], ["establish_grasp", "grasp"], ["actuate_joint", "whole_body"], ["verify_rotation", "checker"]],
    ["joint moves through the requested physical rotation"],
    ["articulation_stalled", "contact_missing"], ["regrasp", "approach", "stabilize"]),
  regrasp: contract("regrasp", ["object_id", "interaction_point_id", "from_hand", "to_hand", "excluded_interaction_point_ids"], ["graspable"],
    ["object observable or physically carried", "source hand has verified support", "selected alternative grasp point is not excluded", "destination hand is available"],
    [["observe", "sensor"], ["select_alternative_point", "checker"], ["support_object", "whole_body"], ["transfer_grasp", "grasp"], ["verify_grasp", "checker"]],
    ["new hand has a verified grasp at a different point"],
    ["interaction_point_missing", "grasp_unstable"], ["bimanual_support", "place", "retreat"]),
  bimanual_support: contract("bimanual_support", ["object_id", "left_interaction_point_id", "right_interaction_point_id"], ["graspable"],
    ["two distinct compatible grasp points exist", "both hands reachable"],
    [["observe", "sensor"], ["establish_two_hand_contact", "whole_body"], ["verify_bimanual_support", "checker"]],
    ["both hands maintain distinct verified support contacts"],
    ["unreachable", "grasp_unstable"], ["regrasp", "place", "stabilize"]),
  bimanual_carry: contract("bimanual_carry", ["object_id", "target", "tolerance_m"], ["movable"],
    ["bimanual support verified", "route available"],
    [["observe", "sensor"], ["carry_route", "navigation"], ["verify_bimanual_support", "checker"]],
    ["target reached with both support contacts maintained"],
    ["path_blocked", "object_slipped"], ["bimanual_support", "retreat", "place"]),
  stabilize: contract("stabilize", ["minimum_support_margin_m"], [],
    ["physical state observable"],
    [["observe", "sensor"], ["recover_support", "whole_body"], ["verify_balance", "checker"]],
    ["upright posture and requested support margin restored"],
    ["balance_lost", "collision_risk"], ["retreat", "stabilize"]),
  retreat: contract("retreat", ["target", "minimum_obstacle_clearance_m"], [],
    ["safe route available"],
    [["observe", "sensor"], ["route", "navigation"], ["verify_clearance", "checker"]],
    ["robot reaches a stable collision-free recovery stance"],
    ["path_blocked", "balance_lost"], ["stabilize", "retreat"])
});

function contract(
  id: HumanoidSkillId,
  parameters: string[],
  requiredAffordances: string[],
  preconditions: string[],
  process: Array<[string, HumanoidSkillContract["process"][number]["authority"]]>,
  successConditions: string[],
  failures: HumanoidSkillFailureCode[],
  recovery: HumanoidSkillId[]
): HumanoidSkillContract {
  return {
    id,
    parameters,
    required_affordances: requiredAffordances,
    preconditions,
    prerequisite_skill_groups: prerequisiteSkillGroups(id),
    process: process.map(([phase, authority]) => ({
      phase,
      authority,
      learned_policy_capabilities: learnedPolicyCapabilities(id, phase, authority)
    })),
    success_conditions: successConditions,
    failure_reasons: [...new Set([...COMMON_FAILURES, ...failures])],
    recovery_entry: recovery
  };
}

function prerequisiteSkillGroups(skill: HumanoidSkillId): HumanoidSkillId[][] {
  if (skill === "reach") return [["approach"]];
  if (skill === "grasp") return [["reach"], ["approach"]];
  if (skill === "lift") return [["grasp", "regrasp", "bimanual_support"]];
  if (skill === "carry" || skill === "carry_to_zone"
    || skill === "bimanual_carry") {
    return [["lift", "grasp", "regrasp", "bimanual_support"]];
  }
  if (skill === "place") {
    return [["carry", "carry_to_zone", "bimanual_carry", "lift", "grasp", "regrasp"]];
  }
  if (["push", "pull", "press", "open", "close", "turn"].includes(skill)) {
    return [["reach"], ["approach"]];
  }
  return [];
}

function learnedPolicyCapabilities(
  skill: HumanoidSkillId,
  phase: string,
  authority: HumanoidSkillContract["process"][number]["authority"]
): HumanoidLearnedPolicyCapability[] {
  if (authority === "sensor" || authority === "checker") return [];
  if (authority === "navigation") {
    return skill === "carry" || skill === "carry_to_zone"
      || skill === "bimanual_carry"
      ? ["locomotion", "contact_rich_manipulation"]
      : ["locomotion"];
  }
  if (skill === "stabilize") return ["balance"];
  if (skill === "regrasp" || skill === "bimanual_support"
    || skill === "bimanual_carry") {
    return ["joint_reference_tracking", "contact_rich_manipulation", "bimanual_manipulation"];
  }
  if (authority === "grasp" || [
    "break_block",
    "lift",
    "place",
    "push",
    "pull",
    "press",
    "open",
    "close",
    "turn"
  ].includes(skill)) {
    return ["joint_reference_tracking", "contact_rich_manipulation"];
  }
  if (phase === "solve_whole_body_reach") {
    return ["balance", "joint_reference_tracking"];
  }
  return ["joint_reference_tracking"];
}
