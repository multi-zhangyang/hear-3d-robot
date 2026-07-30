import RAPIER from "@dimforge/rapier3d-compat";
import type { Vec3, WorldSnapshot } from "../domain/schema.js";
import { quaternionDistance, type Quaternion } from "./kinematics.js";
import {
  inverseQuaternion,
  multiplyQuaternion,
  quaternion,
  rotateVector,
  subtract,
  vector,
  vectorLength
} from "./geometry.js";
import { ROBOT_SPEC } from "./robot-model.js";

type ImpulseJoint = InstanceType<typeof RAPIER.ImpulseJoint>;
type RigidBody = InstanceType<typeof RAPIER.RigidBody>;
type World = InstanceType<typeof RAPIER.World>;

const ZERO_VECTOR = { x: 0, y: 0, z: 0 } as const;
const IDENTITY_ROTATION = { x: 0, y: 0, z: 0, w: 1 } as const;

interface AttachmentState {
  objectId: string;
  constraintId: string;
  sourceCommandId: string;
  objectBody: RigidBody;
  joint: ImpulseJoint;
  anchorPosition: Vec3;
  anchorRotation: Quaternion;
  driftFrames: number;
}

export interface AttachmentSlip {
  objectId: string;
  sourceCommandId: string;
  positionDrift: number;
  rotationDrift: number;
  detectionFrames: number;
}

type AttachmentSnapshot = NonNullable<WorldSnapshot["robot"]["attachment"]>;

/** Rapier constraint and measured-slip lifecycle for one gripper payload. */
export class GraspAttachment {
  readonly #world: World;
  readonly #anchor: RigidBody;
  #state: AttachmentState | null = null;

  constructor(world: World, anchor: RigidBody) {
    this.#world = world;
    this.#anchor = anchor;
  }

  get objectId(): string | null {
    return this.#state?.objectId ?? null;
  }

  get constraintId(): string | null {
    return this.#state?.constraintId ?? null;
  }

  get sourceCommandId(): string | null {
    return this.#state?.sourceCommandId ?? null;
  }

  get anchorPosition(): Vec3 | null {
    return this.#state ? { ...this.#state.anchorPosition } : null;
  }

  get anchorRotation(): Quaternion | null {
    return this.#state ? { ...this.#state.anchorRotation } : null;
  }

  get attached(): boolean {
    return this.#state !== null;
  }

  snapshot(): AttachmentSnapshot | null {
    return this.#state
      ? {
          object_id: this.#state.objectId,
          constraint_id: this.#state.constraintId,
          source_command_id: this.#state.sourceCommandId
        }
      : null;
  }

  install(objectId: string, objectBody: RigidBody, constraintId: string, sourceCommandId: string): void {
    if (this.#state) throw new Error("Cannot install a second gripper attachment");
    const anchorPosition = vector(this.#anchor.translation());
    const anchorRotation = quaternion(this.#anchor.rotation());
    const objectPosition = vector(objectBody.translation());
    const objectRotation = quaternion(objectBody.rotation());
    const localPosition = rotateVector(
      inverseQuaternion(anchorRotation),
      subtract(objectPosition, anchorPosition)
    );
    const localRotation = multiplyQuaternion(inverseQuaternion(anchorRotation), objectRotation);
    const joint = this.#world.createImpulseJoint(
      RAPIER.JointData.fixed(localPosition, localRotation, ZERO_VECTOR, IDENTITY_ROTATION),
      this.#anchor,
      objectBody,
      true
    );
    this.#state = {
      objectId,
      constraintId,
      sourceCommandId,
      objectBody,
      joint,
      anchorPosition: localPosition,
      anchorRotation: localRotation,
      driftFrames: 0
    };
  }

  remove(): boolean {
    if (!this.#state) return false;
    this.#world.removeImpulseJoint(this.#state.joint, true);
    this.#state = null;
    return true;
  }

  validate(bilateralContactObjectId: string | null = null): AttachmentSlip | null {
    const state = this.#state;
    if (!state) return null;
    const anchorPosition = vector(this.#anchor.translation());
    const anchorRotation = quaternion(this.#anchor.rotation());
    const offset = rotateVector(anchorRotation, state.anchorPosition);
    const expectedPosition = {
      x: anchorPosition.x + offset.x,
      y: anchorPosition.y + offset.y,
      z: anchorPosition.z + offset.z
    };
    const expectedRotation = multiplyQuaternion(anchorRotation, state.anchorRotation);
    const positionDrift = vectorLength(subtract(vector(state.objectBody.translation()), expectedPosition));
    const rotationDrift = quaternionDistance(quaternion(state.objectBody.rotation()), expectedRotation);
    // Contact and constraint solvers share a finite iteration budget. While
    // both fingers still physically contact this exact object, a transient
    // rotational residual is solver lag rather than evidence that the grasp
    // has opened. Once bilateral support is gone, the bounded drift window
    // below remains the authority for releasing a genuinely slipping payload.
    if (bilateralContactObjectId === state.objectId
      || (positionDrift <= ROBOT_SPEC.gripper.maximumAttachmentPositionDrift
        && rotationDrift <= ROBOT_SPEC.gripper.maximumAttachmentRotationDrift)) {
      state.driftFrames = 0;
      return null;
    }
    state.driftFrames += 1;
    if (state.driftFrames < ROBOT_SPEC.gripper.slipDetectionFrames) return null;
    const slip: AttachmentSlip = {
      objectId: state.objectId,
      sourceCommandId: state.sourceCommandId,
      positionDrift,
      rotationDrift,
      detectionFrames: ROBOT_SPEC.gripper.slipDetectionFrames
    };
    this.remove();
    return slip;
  }
}
