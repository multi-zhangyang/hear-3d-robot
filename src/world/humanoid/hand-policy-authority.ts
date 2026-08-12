import { z } from "zod";
import type { Vec3 } from "../../domain/schema.js";
import {
  add,
  rotateVector,
  subtract
} from "../geometry.js";
import type { HumanoidEmbodiedSkillCall } from "./embodied-skill-call.js";
import type { HumanoidSimulationSnapshot } from "./simulation.js";

const HAND_POLICY_PLANAR_CLOSURE_TOLERANCE_METERS = 0.035;
const HAND_POLICY_VERTICAL_CLOSURE_TOLERANCE_METERS = 0.040;
const HAND_POLICY_CONTACT_SUPPORTED_PLANAR_TOLERANCE_METERS = 0.045;
const HAND_POLICY_CONTACT_SUPPORTED_RIGHT_PLANAR_TOLERANCE_METERS = 0.060;
const HAND_POLICY_CONTACT_SUPPORTED_VERTICAL_TOLERANCE_METERS = 0.045;
const HAND_POLICY_CONTACT_SUPPORTED_FORCE_NEWTONS = 2;

export const HumanoidHandPolicyAuthorityStateSchema = z.object({
  protocol: z.literal("humanoid-hand-policy-authority-state-v1"),
  callId: z.string().trim().min(1),
  activeHand: z.enum(["left", "right"]),
  objectId: z.string().trim().min(1),
  closureGeometryLatched: z.literal(true),
  authorizedSurfaceCount: z.number().int().positive()
}).strict();

export type HumanoidHandPolicyAuthorityState = z.infer<
  typeof HumanoidHandPolicyAuthorityStateSchema
>;

type HumanoidHandPolicyAuthorityReason =
  | "task_command_missing"
  | "contact_capability_missing"
  | "single_active_grasp_required"
  | "contact_surface_authority_missing"
  | "active_wrist_target_missing"
  | "closure_geometry_pending"
  | "closure_geometry_latched";

export interface HumanoidHandPolicyAuthorityAssessment {
  protocol: "humanoid-hand-policy-authority-assessment-v1";
  granted: boolean;
  reason: HumanoidHandPolicyAuthorityReason;
  state: HumanoidHandPolicyAuthorityState | null;
  geometry: {
    planarErrorMeters: number | null;
    verticalErrorMeters: number | null;
    planarToleranceMeters: number;
    verticalToleranceMeters: number;
  };
}

/**
 * Grants the learned hand actor only the contact pocket owned by a typed Skill.
 *
 * The latch is deterministic physical runtime state.  It is deliberately not
 * an actor observation: the policy can propose finger motion, but cannot grant
 * itself contact authority, choose another hand, or widen the set of objects
 * and hand surfaces it may touch.
 */
export function assessHumanoidHandPolicyAuthority(input: {
  previous: HumanoidHandPolicyAuthorityState | null;
  taskCommand?: HumanoidEmbodiedSkillCall | undefined;
  snapshot?: HumanoidSimulationSnapshot | undefined;
}): HumanoidHandPolicyAuthorityAssessment {
  const task = input.taskCommand;
  if (!task) return denied("task_command_missing");
  if (!task.requestedCapabilities.includes("contact_rich_manipulation")) {
    return denied("contact_capability_missing");
  }
  if (task.command.grasps.length !== 1) {
    return denied("single_active_grasp_required");
  }
  const grasp = task.command.grasps[0]!;
  const authorizedSurfaces = new Set(task.safety.authorizedContacts.flatMap(
    (constraint) => (
      "hand_surface" in constraint
        && "object_id" in constraint
        && constraint.hand_surface.startsWith(`${grasp.hand}_`)
        && constraint.object_id === grasp.objectId
        ? [constraint.hand_surface]
        : []
    )
  ));
  if (authorizedSurfaces.size < grasp.minimumDistinctContactSurfaces) {
    return denied("contact_surface_authority_missing");
  }

  const previous = input.previous
    && input.previous.callId === task.identity.callId
    && input.previous.activeHand === grasp.hand
    && input.previous.objectId === grasp.objectId
    ? HumanoidHandPolicyAuthorityStateSchema.parse(input.previous)
    : null;
  if (previous) {
    return granted(previous, null, null);
  }

  const wristBody = grasp.hand === "left"
    ? "left_wrist_yaw_link"
    : "right_wrist_yaw_link";
  const targets = task.command.endEffectors.filter((target) => (
    target.body === wristBody
  ));
  if (targets.length !== 1 || !input.snapshot) {
    return denied("active_wrist_target_missing");
  }
  const targetWorld = endEffectorTargetWorld(
    targets[0]!,
    input.snapshot
  );
  if (!targetWorld) return denied("active_wrist_target_missing");
  const wrist = input.snapshot.links[wristBody].position;
  const error = subtract(wrist, targetWorld);
  // HEAR's application frame is Y-up (MuJoCo X/Y/Z maps to app Z/X/Y).
  // Closure authority therefore measures the ground plane in app X/Z and
  // height on app Y.  Treating app Z as vertical can latch a hand that is at
  // the wrong height while rejecting a valid forward approach.
  const planarErrorMeters = Math.hypot(error.x, error.z);
  const verticalErrorMeters = Math.abs(error.y);
  const geometryReady = (
    planarErrorMeters <= HAND_POLICY_PLANAR_CLOSURE_TOLERANCE_METERS
    && verticalErrorMeters <= HAND_POLICY_VERTICAL_CLOSURE_TOLERANCE_METERS
  );
  const authorizedContactForceNewtons = input.snapshot.contacts.reduce(
    (total, contact) => {
      const handSurface = contact.firstHandLink ?? contact.secondHandLink;
      const matchesObject = contact.firstObject === grasp.objectId
        || contact.secondObject === grasp.objectId;
      return matchesObject && handSurface && authorizedSurfaces.has(handSurface)
        ? total + contact.normalForce
        : total;
    },
    0
  );
  // Contact becomes the terminal physical constraint once the aligned open
  // hand has gently met the authorized object.  This lets the arm hold its
  // measured pose before a 20 ms position-control update can turn first touch
  // into a high-force impulse.  The wider band is still fixed Harness state:
  // neither the learned hand actor nor an Agent can alter it.
  const contactSupportedReady = (
    authorizedContactForceNewtons >= HAND_POLICY_CONTACT_SUPPORTED_FORCE_NEWTONS
    && planarErrorMeters <= (grasp.hand === "right"
      ? HAND_POLICY_CONTACT_SUPPORTED_RIGHT_PLANAR_TOLERANCE_METERS
      : HAND_POLICY_CONTACT_SUPPORTED_PLANAR_TOLERANCE_METERS)
    && verticalErrorMeters <= HAND_POLICY_CONTACT_SUPPORTED_VERTICAL_TOLERANCE_METERS
  );
  if (!geometryReady && !contactSupportedReady) {
    return denied(
      "closure_geometry_pending",
      planarErrorMeters,
      verticalErrorMeters
    );
  }
  const state = HumanoidHandPolicyAuthorityStateSchema.parse({
    protocol: "humanoid-hand-policy-authority-state-v1",
    callId: task.identity.callId,
    activeHand: grasp.hand,
    objectId: grasp.objectId,
    closureGeometryLatched: true,
    authorizedSurfaceCount: authorizedSurfaces.size
  });
  return granted(state, planarErrorMeters, verticalErrorMeters);
}

function endEffectorTargetWorld(
  target: HumanoidEmbodiedSkillCall["command"]["endEffectors"][number],
  snapshot: HumanoidSimulationSnapshot
): Vec3 | null {
  if (target.frame === "world") return { ...target.position };
  if (target.frame === "pelvis") {
    return add(
      snapshot.rootPosition,
      rotateVector(snapshot.rootRotation, target.position)
    );
  }
  const torso = snapshot.links.torso_link;
  if (!torso) return null;
  return add(torso.position, rotateVector(torso.rotation, target.position));
}

function denied(
  reason: Exclude<HumanoidHandPolicyAuthorityReason, "closure_geometry_latched">,
  planarErrorMeters: number | null = null,
  verticalErrorMeters: number | null = null
): HumanoidHandPolicyAuthorityAssessment {
  return {
    protocol: "humanoid-hand-policy-authority-assessment-v1",
    granted: false,
    reason,
    state: null,
    geometry: geometry(planarErrorMeters, verticalErrorMeters)
  };
}

function granted(
  state: HumanoidHandPolicyAuthorityState,
  planarErrorMeters: number | null,
  verticalErrorMeters: number | null
): HumanoidHandPolicyAuthorityAssessment {
  return {
    protocol: "humanoid-hand-policy-authority-assessment-v1",
    granted: true,
    reason: "closure_geometry_latched",
    state,
    geometry: geometry(planarErrorMeters, verticalErrorMeters)
  };
}

function geometry(
  planarErrorMeters: number | null,
  verticalErrorMeters: number | null
): HumanoidHandPolicyAuthorityAssessment["geometry"] {
  return {
    planarErrorMeters,
    verticalErrorMeters,
    planarToleranceMeters: HAND_POLICY_PLANAR_CLOSURE_TOLERANCE_METERS,
    verticalToleranceMeters: HAND_POLICY_VERTICAL_CLOSURE_TOLERANCE_METERS
  };
}
