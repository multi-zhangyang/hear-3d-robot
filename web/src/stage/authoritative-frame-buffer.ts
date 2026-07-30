import type {
  Quaternion,
  RobotLinkState,
  Vec3,
  WorldSnapshot
} from "../types";

const DEFAULT_CAPACITY = 96;
const MAX_VISUAL_LAG_SECONDS = 0.25;
const CATCH_UP_DELAY_SECONDS = 0.05;
const TIME_EPSILON = 1e-6;

type FrameListener = () => void;

/**
 * A bounded, monotonic queue of authoritative physics snapshots.
 *
 * The queue never predicts a future pose. While a run is live, its playhead
 * advances only between snapshots that the server has already delivered. When
 * a run stops, sampling returns the newest snapshot itself so the final visual
 * pose is byte-for-byte authoritative rather than an interpolated approximation.
 */
export class AuthoritativeFrameBuffer {
  readonly #capacity: number;
  readonly #listeners = new Set<FrameListener>();
  #frames: WorldSnapshot[] = [];
  #playheadSeconds: number | null = null;
  #lastWallTimeMs: number | null = null;
  #version = 0;

  constructor(initial?: WorldSnapshot, capacity = DEFAULT_CAPACITY) {
    if (!Number.isSafeInteger(capacity) || capacity < 2) {
      throw new Error("Frame buffer capacity must be an integer of at least two");
    }
    this.#capacity = capacity;
    if (initial) this.reset(initial);
  }

  get latest(): WorldSnapshot | null {
    return this.#frames.at(-1) ?? null;
  }

  get version(): number {
    return this.#version;
  }

  get pending(): boolean {
    const latest = this.latest;
    return latest !== null
      && this.#playheadSeconds !== null
      && latest.simulated_time - this.#playheadSeconds > TIME_EPSILON;
  }

  /** A defensive copy intended for diagnostics and deterministic tests. */
  snapshots(): readonly WorldSnapshot[] {
    return [...this.#frames];
  }

  reset(frame: WorldSnapshot): void {
    assertFrame(frame);
    this.#frames = [frame];
    this.#playheadSeconds = frame.simulated_time;
    this.#lastWallTimeMs = null;
    this.#version += 1;
    this.#notify();
  }

  /**
   * Adds a possibly unordered SSE batch. Older frames are ignored, duplicate
   * frame numbers replace their queued snapshot, and simulated time can never
   * move backwards.
   */
  push(incoming: readonly WorldSnapshot[]): number {
    const candidates = incoming
      .filter(validFrame)
      .sort((left, right) => left.frame - right.frame || left.simulated_time - right.simulated_time);
    if (candidates.length === 0) return 0;

    if (this.#frames.length === 0) {
      const [first, ...rest] = candidates;
      this.#frames = [first!];
      this.#playheadSeconds = first!.simulated_time;
      this.#lastWallTimeMs = null;
      let accepted = 1;
      for (const frame of rest) accepted += this.#append(frame);
      this.#trimToCapacity();
      this.#version += 1;
      this.#notify();
      return accepted;
    }

    const previousLatest = this.latest!;
    const wasAtTail = this.#playheadSeconds === null
      || previousLatest.simulated_time - this.#playheadSeconds <= TIME_EPSILON;
    let accepted = 0;
    for (const frame of candidates) accepted += this.#append(frame);
    if (accepted === 0) return 0;

    this.#trimToCapacity();
    if (wasAtTail && this.latest!.simulated_time > previousLatest.simulated_time) {
      // Do not count wall time spent waiting on a model or the network as
      // simulation time. The first animation frame anchors the new live batch.
      this.#lastWallTimeMs = null;
    }
    this.#version += 1;
    this.#notify();
    return accepted;
  }

  /** Samples a visual pose without ever advancing beyond received simulation time. */
  sample(wallTimeMs: number, live: boolean): WorldSnapshot | null {
    const latest = this.latest;
    if (!latest) return null;
    if (!Number.isFinite(wallTimeMs)) throw new Error("Frame sample time must be finite");

    if (!live) {
      this.#playheadSeconds = latest.simulated_time;
      this.#lastWallTimeMs = wallTimeMs;
      this.#frames = [latest];
      return latest;
    }

    this.#playheadSeconds ??= this.#frames[0]!.simulated_time;
    if (this.#lastWallTimeMs === null) {
      this.#lastWallTimeMs = wallTimeMs;
    } else {
      const elapsed = Math.max(0, (wallTimeMs - this.#lastWallTimeMs) / 1000);
      this.#lastWallTimeMs = wallTimeMs;
      this.#playheadSeconds = Math.min(latest.simulated_time, this.#playheadSeconds + elapsed);
    }

    const lag = latest.simulated_time - this.#playheadSeconds;
    if (lag > MAX_VISUAL_LAG_SECONDS) {
      this.#playheadSeconds = Math.max(
        this.#frames[0]!.simulated_time,
        latest.simulated_time - CATCH_UP_DELAY_SECONDS
      );
    }

    return this.#sampleAt(this.#playheadSeconds);
  }

  subscribe(listener: FrameListener): () => void {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  }

  #append(frame: WorldSnapshot): number {
    const latest = this.latest!;
    if (frame.frame < latest.frame || frame.simulated_time + TIME_EPSILON < latest.simulated_time) {
      return 0;
    }
    if (frame.frame === latest.frame) {
      // The terminal command snapshot can share a physics-frame number with the
      // last moving snapshot while clearing its active command. Replace rather
      // than duplicate it so stopping still lands on the exact terminal state.
      this.#frames[this.#frames.length - 1] = frame;
      return 1;
    }
    this.#frames.push(frame);
    return 1;
  }

  #sampleAt(time: number): WorldSnapshot {
    const first = this.#frames[0]!;
    if (time <= first.simulated_time + TIME_EPSILON) return first;
    const latest = this.latest!;
    if (time >= latest.simulated_time - TIME_EPSILON) {
      this.#frames = [latest];
      return latest;
    }

    let rightIndex = 1;
    while (rightIndex < this.#frames.length
      && this.#frames[rightIndex]!.simulated_time < time) rightIndex += 1;
    const leftIndex = Math.max(0, rightIndex - 1);
    const left = this.#frames[leftIndex]!;
    const right = this.#frames[rightIndex] ?? latest;
    const duration = right.simulated_time - left.simulated_time;
    const alpha = duration <= TIME_EPSILON ? 1 : clamp01((time - left.simulated_time) / duration);
    if (leftIndex > 0) this.#frames.splice(0, leftIndex);
    return interpolateWorldFrame(left, right, alpha);
  }

  #trimToCapacity(): void {
    if (this.#frames.length <= this.#capacity) return;
    this.#frames.splice(0, this.#frames.length - this.#capacity);
    this.#playheadSeconds = Math.max(
      this.#playheadSeconds ?? this.#frames[0]!.simulated_time,
      this.#frames[0]!.simulated_time
    );
  }

  #notify(): void {
    for (const listener of [...this.#listeners]) listener();
  }
}

export function interpolateWorldFrame(
  left: WorldSnapshot,
  right: WorldSnapshot,
  alpha: number
): WorldSnapshot {
  const fraction = clamp01(alpha);
  if (fraction <= 0) return left;
  if (fraction >= 1) return right;

  const leftObjects = new Map(left.objects.map((object) => [object.id, object]));
  const links = Object.fromEntries(Object.entries(right.robot.links).map(([id, link]) => {
    const previous = left.robot.links[id];
    return [id, previous ? interpolateLink(previous, link, fraction) : link];
  }));
  const jointStatus = Object.fromEntries(Object.entries(right.robot.joint_status).map(([id, joint]) => {
    const previous = left.robot.joint_status[id];
    return [id, previous ? {
      ...joint,
      position: mix(previous.position, joint.position, fraction),
      velocity: mix(previous.velocity, joint.velocity, fraction)
    } : joint];
  }));

  return {
    ...right,
    frame: mix(left.frame, right.frame, fraction),
    simulated_time: mix(left.simulated_time, right.simulated_time, fraction),
    world_revision: fraction < 1 ? left.world_revision : right.world_revision,
    robot: {
      ...right.robot,
      position: interpolateVec3(left.robot.position, right.robot.position, fraction),
      yaw: interpolateAngle(left.robot.yaw, right.robot.yaw, fraction),
      joints: {
        head_yaw: mix(left.robot.joints.head_yaw, right.robot.joints.head_yaw, fraction),
        head_pitch: mix(left.robot.joints.head_pitch, right.robot.joints.head_pitch, fraction),
        shoulder: mix(left.robot.joints.shoulder, right.robot.joints.shoulder, fraction),
        elbow: mix(left.robot.joints.elbow, right.robot.joints.elbow, fraction),
        wrist: mix(left.robot.joints.wrist, right.robot.joints.wrist, fraction),
        gripper_aperture: mix(
          left.robot.joints.gripper_aperture,
          right.robot.joints.gripper_aperture,
          fraction
        )
      },
      contacts: fraction < 0.5 ? left.robot.contacts : right.robot.contacts,
      attachment: fraction < 0.5 ? left.robot.attachment : right.robot.attachment,
      odometry: {
        left_wheel: {
          position: mix(
            left.robot.odometry.left_wheel.position,
            right.robot.odometry.left_wheel.position,
            fraction
          ),
          velocity: mix(
            left.robot.odometry.left_wheel.velocity,
            right.robot.odometry.left_wheel.velocity,
            fraction
          )
        },
        right_wheel: {
          position: mix(
            left.robot.odometry.right_wheel.position,
            right.robot.odometry.right_wheel.position,
            fraction
          ),
          velocity: mix(
            left.robot.odometry.right_wheel.velocity,
            right.robot.odometry.right_wheel.velocity,
            fraction
          )
        }
      },
      links,
      joint_status: jointStatus,
      gripper: {
        ...right.robot.gripper,
        aperture: mix(left.robot.gripper.aperture, right.robot.gripper.aperture, fraction),
        left_contact_force: mix(
          left.robot.gripper.left_contact_force,
          right.robot.gripper.left_contact_force,
          fraction
        ),
        right_contact_force: mix(
          left.robot.gripper.right_contact_force,
          right.robot.gripper.right_contact_force,
          fraction
        )
      }
    },
    objects: right.objects.map((object) => {
      const previous = leftObjects.get(object.id);
      return previous ? {
        ...object,
        position: interpolateVec3(previous.position, object.position, fraction),
        rotation: interpolateQuaternion(previous.rotation, object.rotation, fraction),
        linear_velocity: interpolateVec3(previous.linear_velocity, object.linear_velocity, fraction),
        angular_velocity: interpolateVec3(previous.angular_velocity, object.angular_velocity, fraction)
      } : object;
    })
  };
}

function interpolateLink(left: RobotLinkState, right: RobotLinkState, alpha: number): RobotLinkState {
  return {
    position: interpolateVec3(left.position, right.position, alpha),
    rotation: interpolateQuaternion(left.rotation, right.rotation, alpha),
    linear_velocity: interpolateVec3(left.linear_velocity, right.linear_velocity, alpha),
    angular_velocity: interpolateVec3(left.angular_velocity, right.angular_velocity, alpha)
  };
}

function interpolateVec3(left: Vec3, right: Vec3, alpha: number): Vec3 {
  return {
    x: mix(left.x, right.x, alpha),
    y: mix(left.y, right.y, alpha),
    z: mix(left.z, right.z, alpha)
  };
}

function interpolateQuaternion(left: Quaternion, right: Quaternion, alpha: number): Quaternion {
  let target = right;
  let dot = left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
  if (dot < 0) {
    dot = -dot;
    target = { x: -right.x, y: -right.y, z: -right.z, w: -right.w };
  }
  if (dot > 0.9995) {
    return normalizeQuaternion({
      x: mix(left.x, target.x, alpha),
      y: mix(left.y, target.y, alpha),
      z: mix(left.z, target.z, alpha),
      w: mix(left.w, target.w, alpha)
    });
  }
  const theta = Math.acos(Math.min(1, Math.max(-1, dot)));
  const sine = Math.sin(theta);
  const leftWeight = Math.sin((1 - alpha) * theta) / sine;
  const rightWeight = Math.sin(alpha * theta) / sine;
  return normalizeQuaternion({
    x: left.x * leftWeight + target.x * rightWeight,
    y: left.y * leftWeight + target.y * rightWeight,
    z: left.z * leftWeight + target.z * rightWeight,
    w: left.w * leftWeight + target.w * rightWeight
  });
}

function normalizeQuaternion(value: Quaternion): Quaternion {
  const length = Math.hypot(value.x, value.y, value.z, value.w) || 1;
  return {
    x: value.x / length,
    y: value.y / length,
    z: value.z / length,
    w: value.w / length
  };
}

function interpolateAngle(left: number, right: number, alpha: number): number {
  const delta = ((right - left + Math.PI) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2) - Math.PI;
  return left + delta * alpha;
}

function mix(left: number, right: number, alpha: number): number {
  return left + (right - left) * alpha;
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function validFrame(frame: WorldSnapshot): boolean {
  return Number.isSafeInteger(frame.frame)
    && frame.frame >= 0
    && Number.isFinite(frame.simulated_time)
    && frame.simulated_time >= 0;
}

function assertFrame(frame: WorldSnapshot): void {
  if (!validFrame(frame)) throw new Error("Invalid authoritative world frame");
}
