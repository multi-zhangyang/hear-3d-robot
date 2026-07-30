export interface ArmPose {
  shoulder: number;
  elbow: number;
  wrist: number;
}

export interface JointBound {
  minimum: number;
  maximum: number;
}

export interface ArmTrajectory {
  waypoints: ArmPose[];
  direct: boolean;
  expanded_states: number;
  checked_states: number;
}

export interface ArmTrajectoryFailure {
  code: "arm_trajectory_unavailable";
  expanded_states: number;
  checked_states: number;
}

const JOINTS = ["shoulder", "elbow", "wrist"] as const;
const GRID_STEP = 0.16;
const EDGE_STEP = 0.055;
const MAX_EXPANDED_STATES = 2400;

/**
 * Collision-aware joint-space planning for the articulated arm.
 *
 * A direct interpolated edge is preferred. When it is blocked, A* searches the
 * three-dimensional joint space and then removes every waypoint that is not
 * required by collision geometry. The caller owns collision authority through
 * `isPoseValid`; this module never guesses which contacts are acceptable.
 */
export function planArmTrajectory(input: {
  start: ArmPose;
  target: ArmPose;
  bounds: Record<keyof ArmPose, JointBound>;
  isPoseValid: (pose: ArmPose) => boolean;
}): ArmTrajectory | ArmTrajectoryFailure {
  let checkedStates = 0;
  const validity = new Map<string, boolean>();
  const valid = (pose: ArmPose): boolean => {
    const key = poseKey(pose);
    const cached = validity.get(key);
    if (cached !== undefined) return cached;
    checkedStates += 1;
    const accepted = insideBounds(pose, input.bounds) && input.isPoseValid(pose);
    validity.set(key, accepted);
    return accepted;
  };
  const edgeClear = (from: ArmPose, to: ArmPose): boolean => {
    const distance = maximumDelta(from, to);
    const steps = Math.max(1, Math.ceil(distance / EDGE_STEP));
    for (let step = 1; step <= steps; step += 1) {
      if (!valid(interpolate(from, to, step / steps))) return false;
    }
    return true;
  };

  if (!valid(input.target)) {
    return { code: "arm_trajectory_unavailable", expanded_states: 0, checked_states: checkedStates };
  }

  if (edgeClear(input.start, input.target)) {
    return { waypoints: [clonePose(input.target)], direct: true, expanded_states: 0, checked_states: checkedStates };
  }

  const startKey = poseKey(input.start);
  const frontier = new MinHeap();
  const poses = new Map([[startKey, clonePose(input.start)]]);
  const cameFrom = new Map<string, string>();
  const costs = new Map([[startKey, 0]]);
  frontier.push({ key: startKey, priority: distance(input.start, input.target) });
  let expanded = 0;
  let goalKey: string | null = null;

  while (frontier.size > 0 && expanded < MAX_EXPANDED_STATES) {
    const currentEntry = frontier.pop()!;
    const current = poses.get(currentEntry.key);
    if (!current) continue;
    const currentCost = costs.get(currentEntry.key);
    if (currentCost === undefined) continue;
    expanded += 1;

    if (edgeClear(current, input.target)) {
      const exactKey = poseKey(input.target);
      poses.set(exactKey, clonePose(input.target));
      cameFrom.set(exactKey, currentEntry.key);
      goalKey = exactKey;
      break;
    }

    for (const next of neighbours(current, input.bounds)) {
      if (!edgeClear(current, next)) continue;
      const nextKey = poseKey(next);
      const nextCost = currentCost + distance(current, next);
      if (nextCost >= (costs.get(nextKey) ?? Number.POSITIVE_INFINITY)) continue;
      poses.set(nextKey, next);
      costs.set(nextKey, nextCost);
      cameFrom.set(nextKey, currentEntry.key);
      frontier.push({ key: nextKey, priority: nextCost + distance(next, input.target) });
    }
  }

  if (!goalKey) {
    return {
      code: "arm_trajectory_unavailable",
      expanded_states: expanded,
      checked_states: checkedStates
    };
  }
  const route = reconstruct(goalKey, startKey, cameFrom, poses);
  return {
    waypoints: simplify(input.start, route, edgeClear),
    direct: false,
    expanded_states: expanded,
    checked_states: checkedStates
  };
}

function neighbours(
  pose: ArmPose,
  bounds: Record<keyof ArmPose, JointBound>
): ArmPose[] {
  const result: ArmPose[] = [];
  for (const joint of JOINTS) {
    for (const direction of [-1, 1]) {
      const value = clamp(pose[joint] + direction * GRID_STEP, bounds[joint]);
      if (Math.abs(value - pose[joint]) < 1e-8) continue;
      result.push({ ...pose, [joint]: value });
    }
  }
  return result;
}

function reconstruct(
  goalKey: string,
  startKey: string,
  cameFrom: Map<string, string>,
  poses: Map<string, ArmPose>
): ArmPose[] {
  const reversed: ArmPose[] = [];
  let key = goalKey;
  while (key !== startKey) {
    const pose = poses.get(key);
    const parent = cameFrom.get(key);
    if (!pose || !parent) throw new Error("Arm trajectory search produced a broken route");
    reversed.push(clonePose(pose));
    key = parent;
  }
  return reversed.reverse();
}

function simplify(
  start: ArmPose,
  route: ArmPose[],
  edgeClear: (from: ArmPose, to: ArmPose) => boolean
): ArmPose[] {
  const simplified: ArmPose[] = [];
  let anchor = start;
  let index = 0;
  while (index < route.length) {
    let furthest = index;
    for (let candidate = route.length - 1; candidate > index; candidate -= 1) {
      if (edgeClear(anchor, route[candidate]!)) {
        furthest = candidate;
        break;
      }
    }
    const waypoint = route[furthest]!;
    simplified.push(clonePose(waypoint));
    anchor = waypoint;
    index = furthest + 1;
  }
  return simplified;
}

function insideBounds(pose: ArmPose, bounds: Record<keyof ArmPose, JointBound>): boolean {
  return JOINTS.every((joint) =>
    Number.isFinite(pose[joint])
      && pose[joint] >= bounds[joint].minimum
      && pose[joint] <= bounds[joint].maximum
  );
}

function interpolate(from: ArmPose, to: ArmPose, amount: number): ArmPose {
  return {
    shoulder: from.shoulder + (to.shoulder - from.shoulder) * amount,
    elbow: from.elbow + (to.elbow - from.elbow) * amount,
    wrist: from.wrist + (to.wrist - from.wrist) * amount
  };
}

function maximumDelta(left: ArmPose, right: ArmPose): number {
  return Math.max(...JOINTS.map((joint) => Math.abs(left[joint] - right[joint])));
}

function distance(left: ArmPose, right: ArmPose): number {
  return Math.hypot(...JOINTS.map((joint) => left[joint] - right[joint]));
}

function poseKey(pose: ArmPose): string {
  return JOINTS.map((joint) => pose[joint].toFixed(5)).join(":");
}

function clonePose(pose: ArmPose): ArmPose {
  return { shoulder: pose.shoulder, elbow: pose.elbow, wrist: pose.wrist };
}

function clamp(value: number, bound: JointBound): number {
  return Math.min(bound.maximum, Math.max(bound.minimum, value));
}

interface HeapEntry {
  key: string;
  priority: number;
}

class MinHeap {
  readonly #items: HeapEntry[] = [];

  get size(): number {
    return this.#items.length;
  }

  push(entry: HeapEntry): void {
    this.#items.push(entry);
    let index = this.#items.length - 1;
    while (index > 0) {
      const parent = Math.floor((index - 1) / 2);
      if (this.#items[parent]!.priority <= entry.priority) break;
      this.#items[index] = this.#items[parent]!;
      index = parent;
    }
    this.#items[index] = entry;
  }

  pop(): HeapEntry | undefined {
    const first = this.#items[0];
    const last = this.#items.pop();
    if (!first || !last || this.#items.length === 0) return first;
    let index = 0;
    while (true) {
      const left = index * 2 + 1;
      const right = left + 1;
      if (left >= this.#items.length) break;
      const child = right < this.#items.length
        && this.#items[right]!.priority < this.#items[left]!.priority ? right : left;
      if (this.#items[child]!.priority >= last.priority) break;
      this.#items[index] = this.#items[child]!;
      index = child;
    }
    this.#items[index] = last;
    return first;
  }
}
