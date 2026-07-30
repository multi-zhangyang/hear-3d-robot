/**
 * Affordances: state an object can change into, which no motion command names
 * directly. A container unlocks because its key was brought into the right
 * socket at the right angle — the agent never issues "unlock", it arranges the
 * geometry and the world draws the conclusion.
 *
 * That is why this is evaluated from the scene rather than commanded: keeping
 * it here means the rule is one readable predicate over poses instead of a
 * special case threaded through gripper and base motion.
 */
import type { JsonValue, Scenario, Vec3, WorldSnapshot } from "../domain/schema.js";
import type { Quaternion } from "./kinematics.js";
import {
  clamp,
  dot,
  inverseQuaternion,
  normalizeVector,
  rotateVector,
  subtract
} from "./geometry.js";

export type AffordanceEvent = WorldSnapshot["affordance_events"][number];

/** One scenario object as the affordance rules need to see it. */
interface AffordanceObject {
  id: string;
  position: Vec3;
  rotation: Quaternion;
  locked: boolean;
  keyId: string | null;
}

export interface AffordanceInput {
  affordances: NonNullable<Scenario["affordances"]>;
  /** Every object in the scene, by id. */
  objects: ReadonlyMap<string, AffordanceObject>;
  /** True when the two objects are actually touching this step. */
  inContact: (firstId: string, secondId: string) => boolean;
  frame: number;
  activeCommandId: string | null;
  /**
   * Pairs already reported as unsupported. Mutated so the same scenario defect
   * is reported once rather than on every physics step.
   */
  reportedUnsupported: Set<string>;
}

export interface AffordanceOutcome {
  /** Objects whose `locked` flag must flip to false. */
  unlocked: string[];
  events: AffordanceEvent[];
}

/**
 * Evaluates every affordance rule against the current scene. Returns what
 * changed rather than mutating, so the caller stays the single writer of
 * simulation state.
 */
export function evaluateAffordances(input: AffordanceInput): AffordanceOutcome {
  const unlocked: string[] = [];
  const events: AffordanceEvent[] = [];

  for (const affordance of input.affordances) {
    if (affordance.type !== "keyed_lock") continue;
    const container = input.objects.get(affordance.container_id);
    const key = input.objects.get(affordance.key_id);
    if (!container || !key || !container.locked || input.activeCommandId === null) continue;
    if (!input.inContact(container.id, key.id)) continue;

    // Touching is not inserting. The key must sit inside the socket volume in
    // the container's own frame *and* be aligned with the insertion axis —
    // otherwise a key merely resting against the lid would open it.
    const socket = affordance.socket;
    const keyLocal = rotateVector(
      inverseQuaternion(container.rotation),
      subtract(key.position, container.position)
    );
    const delta = subtract(keyLocal, socket.center);
    const inside = Math.abs(delta.x) <= socket.half_extents.x
      && Math.abs(delta.y) <= socket.half_extents.y
      && Math.abs(delta.z) <= socket.half_extents.z;
    const insertionAxis = normalizeVector(rotateVector(container.rotation, socket.insertion_axis));
    const keyAxis = normalizeVector(rotateVector(key.rotation, { x: 0, y: 0, z: 1 }));
    const axisAngle = Math.acos(clamp(dot(insertionAxis, keyAxis), -1, 1));
    if (!inside || axisAngle > socket.maximum_axis_angle) continue;

    unlocked.push(container.id);
    events.push({
      frame: input.frame,
      affordance_id: affordance.id,
      code: "keyed_lock_transition",
      entity_id: container.id,
      source_command_id: input.activeCommandId,
      detail: {
        from: "locked",
        to: "unlocked",
        key_id: key.id,
        contact: true,
        socket_local_position: keyLocal as unknown as JsonValue,
        insertion_axis_angle: axisAngle
      } as JsonValue
    });
  }

  // A scenario can declare `key_id` on a locked object without defining the
  // socket geometry that would let it open. Silently never unlocking would look
  // like an agent failure, so the world says plainly that the scenario is
  // underspecified — once per pair.
  for (const container of input.objects.values()) {
    if (!container.locked || !container.keyId) continue;
    if (input.affordances.some((item) => item.container_id === container.id)) continue;
    const key = input.objects.get(container.keyId);
    if (!key || !input.inContact(container.id, key.id)) continue;
    const pairKey = `${container.id}:${key.id}`;
    if (input.reportedUnsupported.has(pairKey)) continue;
    input.reportedUnsupported.add(pairKey);
    events.push({
      frame: input.frame,
      affordance_id: null,
      code: "unsupported_keyed_lock_geometry",
      entity_id: container.id,
      source_command_id: input.activeCommandId,
      detail: {
        key_id: key.id,
        reason: "Scenario must define a keyed_lock socket region and insertion axis"
      } as JsonValue
    });
  }

  return { unlocked, events };
}
