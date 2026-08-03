import type {
  HumanoidEndEffector,
  Quaternion,
  Vec3
} from "../../domain/schema.js";
import {
  inverseQuaternion,
  rotateVector,
  subtract
} from "../geometry.js";
import type { HumanoidBodyName } from "./model.js";
import type { HumanoidEndEffectorBody } from "./task-space-targets.js";

const HUMANOID_END_EFFECTOR_BODIES: Readonly<Record<
  HumanoidEndEffector,
  HumanoidEndEffectorBody
>> = {
  left_wrist: "left_wrist_yaw_link",
  right_wrist: "right_wrist_yaw_link",
  left_ankle: "left_ankle_roll_link",
  right_ankle: "right_ankle_roll_link"
};

interface EndEffectorLinkSnapshot {
  position: Vec3;
  rotation?: Quaternion;
}

interface EndEffectorRobotSnapshot {
  links: Readonly<Partial<Record<HumanoidBodyName, EndEffectorLinkSnapshot>>>;
}

export function humanoidEndEffectorBody(
  endEffector: HumanoidEndEffector
): HumanoidEndEffectorBody {
  return HUMANOID_END_EFFECTOR_BODIES[endEffector];
}

export function humanoidEndEffectorPosition(
  snapshot: EndEffectorRobotSnapshot,
  endEffector: HumanoidEndEffector,
  frame: "world" | "pelvis"
): Vec3 | null {
  const link = snapshot.links[humanoidEndEffectorBody(endEffector)];
  if (!link) return null;
  if (frame === "world") return { ...link.position };

  const pelvis = snapshot.links.pelvis;
  if (!pelvis?.rotation) return null;
  return rotateVector(
    inverseQuaternion(pelvis.rotation),
    subtract(link.position, pelvis.position)
  );
}
