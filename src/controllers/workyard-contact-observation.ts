import type { Quaternion, Vec3 } from "../domain/schema.js";
import {
  HUMANOID_JOINT_NAMES,
  YAHMP_POLICY
} from "../world/humanoid/model.js";
import {
  G1_HAND_JOINT_LIMITS,
  G1_HAND_JOINT_NAMES
} from "../world/humanoid/morphology.js";
import type {
  HumanoidHandSynergyPolicyInput
} from "../world/humanoid/hand-synergy-overlay-controller.js";
import type {
  HumanoidControllerInferenceOptions,
  HumanoidPolicyState
} from "../world/humanoid/whole-body-controller.js";

export const WORKYARD_CONTACT_OBSERVATION_PROTOCOL =
  "hear-workyard-contact-observation-v1";
export const WORKYARD_CONTACT_OBSERVATION_SIZE = 247;
export const WORKYARD_REACH_OBSERVATION_PROTOCOL =
  "hear-workyard-residual-observation-v4";
export const WORKYARD_REACH_OBSERVATION_SIZE = 231;

type PolicyVec3 = readonly [x: number, y: number, z: number];
type PolicyQuaternion = readonly [w: number, x: number, y: number, z: number];

export interface WorkyardContactObservationMetadata {
  bodyDefaultJointPositions: readonly number[];
  bodyActionScale: readonly number[];
  targetZoneId: string;
}

export interface WorkyardReachObservationInput {
  state: HumanoidPolicyState;
  options: HumanoidControllerInferenceOptions;
  previousTeacherAction: ArrayLike<number>;
  previousReachAction: ArrayLike<number>;
}

/** Builds the exact 231D observation shared by reach and contact actors. */
export function encodeWorkyardReachObservation(
  input: WorkyardReachObservationInput,
  metadata: WorkyardContactObservationMetadata
): Float32Array {
  assertMetadata(metadata);
  const environment = input.state.environment;
  const task = input.options.taskCommand;
  if (!environment || !environment.rootPosition || !environment.feet
    || !environment.centerOfMass || !environment.centerOfMassVelocity || !task) {
    throw new Error("Workyard contact policy requires complete MuJoCo environment state");
  }
  const grasp = task.command.grasps[0];
  if (!grasp || task.command.grasps.length !== 1) {
    throw new Error("Workyard contact observation requires one authorized grasp");
  }
  const object = environment.objects.find(({ id }) => id === grasp.objectId);
  const zone = environment.zones?.find(({ id }) => id === metadata.targetZoneId);
  if (!object || !zone) {
    throw new Error("Workyard contact object or target zone is unavailable");
  }

  const rootPosition = appVectorToPolicy(environment.rootPosition);
  const rootRotation = normalizeQuaternion(input.state.rootQuaternion);
  const teacherAction = finiteVector(
    input.previousTeacherAction,
    HUMANOID_JOINT_NAMES.length,
    "previous locomotion action"
  );
  const reachAction = finiteVector(
    input.previousReachAction,
    14,
    "previous reach action"
  );

  const handPositions = G1_HAND_JOINT_NAMES.map((name) => {
    const joint = environment.hands[name];
    if (!joint) throw new Error(`Workyard hand joint is unavailable: ${name}`);
    const [minimum, maximum] = G1_HAND_JOINT_LIMITS[name];
    return clamp((joint.position - minimum) / (maximum - minimum), 0, 1);
  });
  const handVelocities = G1_HAND_JOINT_NAMES.map((name) => (
    environment.hands[name]!.velocity
  ));
  const dynamicCom = supportRelativeDynamicCom(input, rootPosition, rootRotation);
  const projectedGravity = inverseRotate(rootRotation, [0, 0, -1]);
  const endEffectorPose = [
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link"
  ].flatMap((name) => {
    const endEffector = environment.endEffectors[name];
    if (!endEffector) throw new Error(`Workyard end effector is unavailable: ${name}`);
    const position = inverseRotate(
      rootRotation,
      subtract(appVectorToPolicy(endEffector.position), rootPosition)
    );
    const rotation = multiplyQuaternion(
      conjugateQuaternion(rootRotation),
      appQuaternionToPolicy(endEffector.rotation)
    );
    return [...position, ...rotation];
  });
  const feet = [
    ["left", "left_ankle_roll_link"],
    ["right", "right_ankle_roll_link"]
  ] as const;
  const footFound = feet.map(([side]) => environment.feet![side].touching ? 1 : 0);
  const footForce = feet.map(([side]) => environment.feet![side].normalForce);
  const footSlip = feet.map(([side, name]) => {
    const velocity = environment.endEffectors[name]?.linearVelocity;
    if (!velocity || !environment.feet![side].touching) return 0;
    const policy = appVectorToPolicy(velocity);
    return Math.hypot(policy[0], policy[1]);
  });
  const handContact = ["left", "right"].flatMap((hand) => {
    const contacts = environment.contacts.filter((contact) => {
      const surface = contact.firstHandLink ?? contact.secondHandLink;
      const matchingObject = contact.firstObject === object.id
        || contact.secondObject === object.id;
      return matchingObject && surface?.startsWith(`${hand}_`);
    });
    return [
      contacts.length > 0 ? 1 : 0,
      contacts.reduce((sum, contact) => sum + contact.normalForce, 0),
      new Set(contacts.map((contact) => (
        contact.firstHandLink ?? contact.secondHandLink
      ))).size
    ];
  });
  // Training concatenates [found_left, found_right, force_left, force_right,
  // surfaces_left, surfaces_right], not per-hand triplets.
  const reorderedHandContact = [
    handContact[0]!, handContact[3]!,
    handContact[1]!, handContact[4]!,
    handContact[2]!, handContact[5]!
  ];

  const objectPosition = appVectorToPolicy(object.position);
  const objectRotation = appQuaternionToPolicy(object.rotation);
  const objectPositionPelvis = inverseRotate(
    rootRotation,
    subtract(objectPosition, rootPosition)
  );
  const objectRotationPelvis = multiplyQuaternion(
    conjugateQuaternion(rootRotation),
    objectRotation
  );
  const objectLinearVelocityPelvis = inverseRotate(
    rootRotation,
    appVectorToPolicy(object.linearVelocity)
  );
  const objectAngularVelocityPelvis = inverseRotate(
    rootRotation,
    appVectorToPolicy(object.angularVelocity)
  );
  const zonePosition = appVectorToPolicy(zone.center);
  const zoneDelta = inverseRotate(
    rootRotation,
    subtract(zonePosition, objectPosition)
  );
  const zoneDistance = Math.hypot(zoneDelta[0], zoneDelta[1]);
  const zoneInside = Math.abs(object.position.x - zone.center.x) <= zone.size.x / 2
    && Math.abs(object.position.z - zone.center.z) <= zone.size.z / 2
    && object.position.y <= zone.center.y + 0.145
    ? 1
    : 0;
  const capabilities = [
    "balance",
    "locomotion",
    "joint_reference_tracking",
    "contact_rich_manipulation",
    "bimanual_manipulation"
  ].map((capability) => task.requestedCapabilities.includes(
    capability as typeof task.requestedCapabilities[number]
  ) ? 1 : 0);
  const activeHand = grasp.hand === "left" ? [1, 0] : [0, 1];
  const wristTargets = ["left", "right"].flatMap((hand) => (
    wristTargetPelvis(input, hand)
  ));
  const wristTolerances = ["left", "right"].map((hand) => {
    const body = `${hand}_wrist_yaw_link`;
    return task.command.endEffectors.find((target) => target.body === body)
      ?.tolerance ?? 0.06;
  });
  const progress = clamp(task.window.stepIndex / task.window.maximumSteps, 0, 1);
  const reachObservation = Float32Array.from([
    ...Array.from(input.state.jointPositions, (value, index) => (
      value - metadata.bodyDefaultJointPositions[index]!
    )),
    ...Array.from(input.state.jointVelocities),
    ...reachAction,
    ...handPositions,
    ...handVelocities,
    ...dynamicCom,
    ...teacherAction,
    ...environment.rootLinearVelocity,
    ...environment.rootAngularVelocity,
    ...projectedGravity,
    ...endEffectorPose,
    ...footFound,
    ...footForce,
    ...footSlip,
    ...reorderedHandContact,
    ...objectPositionPelvis,
    ...objectRotationPelvis,
    ...objectLinearVelocityPelvis,
    ...objectAngularVelocityPelvis,
    ...zoneDelta,
    zoneDistance,
    zoneInside,
    ...capabilities,
    progress,
    ...activeHand,
    task.command.baseTwist.forwardMetersPerSecond,
    task.command.baseTwist.lateralMetersPerSecond,
    task.command.baseTwist.yawRadiansPerSecond,
    ...wristTargets,
    ...wristTolerances,
    grasp.minimumNormalForceN,
    grasp.minimumDistinctContactSurfaces,
    0.15,
    0
  ]);
  if (reachObservation.length !== WORKYARD_REACH_OBSERVATION_SIZE) {
    throw new Error(
      `Workyard frozen-reach observation drifted: ${reachObservation.length}`
    );
  }
  if (!reachObservation.every(Number.isFinite)) {
    throw new Error("Workyard frozen-reach observation is non-finite");
  }
  return reachObservation;
}

/** Appends the hand policy's two 8D recurrent state terms to frozen reach. */
export function encodeWorkyardContactObservation(
  input: HumanoidHandSynergyPolicyInput
): Float32Array {
  const component = input.bodyInference?.components.find(({ observation }) => (
    observation.protocol === WORKYARD_REACH_OBSERVATION_PROTOCOL
  ));
  if (!component
    || component.observation.values.length !== WORKYARD_REACH_OBSERVATION_SIZE
    || component.observation.values.some((value) => !Number.isFinite(value))) {
    throw new Error(
      "Workyard contact policy requires the exact frozen-reach observation trace"
    );
  }
  const coordination = input.coordination;
  // MJLab clips the public actor observation group after concatenation.  The
  // frozen reach actor consumes the raw 231D builder, while the contact actor
  // was trained on this clipped 247D tensor.
  const observation = Float32Array.from([
    ...component.observation.values,
    coordination.left.thumb_opposition,
    coordination.left.thumb_curl,
    coordination.left.index_curl,
    coordination.left.middle_curl,
    coordination.right.thumb_opposition,
    coordination.right.thumb_curl,
    coordination.right.index_curl,
    coordination.right.middle_curl,
    ...input.previousAuthorizedAction
  ], (value) => clamp(value, -20, 20));
  if (observation.length !== WORKYARD_CONTACT_OBSERVATION_SIZE
    || !observation.every(Number.isFinite)) {
    throw new Error("Workyard contact observation is invalid");
  }
  return observation;
}

function supportRelativeDynamicCom(
  input: WorkyardReachObservationInput,
  rootPosition: PolicyVec3,
  rootRotation: PolicyQuaternion
): number[] {
  const environment = input.state.environment!;
  const names = ["left_ankle_roll_link", "right_ankle_roll_link"] as const;
  const contacts = [environment.feet!.left.touching, environment.feet!.right.touching];
  const weights = contacts.some(Boolean)
    ? contacts.map((contact) => contact ? 1 / contacts.filter(Boolean).length : 0)
    : [0.5, 0.5];
  const footPositions = names.map((name) => {
    const value = environment.endEffectors[name]?.position;
    if (!value) throw new Error(`Workyard support frame is missing: ${name}`);
    return appVectorToPolicy(value);
  });
  const footVelocities = names.map((name) => {
    const value = environment.endEffectors[name]?.linearVelocity;
    if (!value) throw new Error(`Workyard support velocity is missing: ${name}`);
    return appVectorToPolicy(value);
  });
  const supportPosition = weightedVector(footPositions, weights);
  const supportVelocity = weightedVector(footVelocities, weights);
  const comPosition = appVectorToPolicy(environment.centerOfMass!);
  const relativePosition = inverseRotate(
    rootRotation,
    subtract(comPosition, supportPosition)
  );
  const supportVelocityPelvis = inverseRotate(rootRotation, supportVelocity);
  const comVelocityPelvis = inverseRotate(
    rootRotation,
    appVectorToPolicy(environment.centerOfMassVelocity!)
  );
  const relativeVelocity: PolicyVec3 = subtract(
    comVelocityPelvis,
    supportVelocityPelvis
  );
  // Keep rootPosition in the signature so axis/authority changes cannot be
  // silently removed from this physical encoder.
  if (!rootPosition.every(Number.isFinite)) {
    throw new Error("Workyard root position is non-finite");
  }
  return [
    relativePosition[0], relativePosition[1],
    relativeVelocity[0], relativeVelocity[1]
  ];
}

function wristTargetPelvis(
  input: WorkyardReachObservationInput,
  hand: string
): number[] {
  const environment = input.state.environment!;
  const task = input.options.taskCommand!;
  const body = `${hand}_wrist_yaw_link`;
  const measured = environment.endEffectors[body];
  if (!measured || !environment.rootPosition) {
    throw new Error(`Workyard wrist state is missing: ${body}`);
  }
  const rootPosition = appVectorToPolicy(environment.rootPosition);
  const rootRotation = normalizeQuaternion(input.state.rootQuaternion);
  const target = task.command.endEffectors.find((candidate) => candidate.body === body);
  if (!target) {
    return poseRelativeTo(
      rootPosition,
      rootRotation,
      appVectorToPolicy(measured.position),
      appQuaternionToPolicy(measured.rotation)
    );
  }
  if (target.frame === "pelvis") {
    const position = appVectorToPolicy(target.position);
    const rotation = target.orientation
      ? appQuaternionToPolicy(target.orientation)
      : poseRelativeTo(
          rootPosition,
          rootRotation,
          appVectorToPolicy(measured.position),
          appQuaternionToPolicy(measured.rotation)
        ).slice(3);
    return [...position, ...rotation];
  }
  let worldPosition = appVectorToPolicy(target.position);
  let worldRotation = target.orientation
    ? appQuaternionToPolicy(target.orientation)
    : appQuaternionToPolicy(measured.rotation);
  if (target.frame === "torso") {
    const torso = environment.endEffectors.torso_link;
    if (!torso) throw new Error("Workyard torso frame is unavailable");
    const torsoPosition = appVectorToPolicy(torso.position);
    const torsoRotation = appQuaternionToPolicy(torso.rotation);
    worldPosition = add(
      torsoPosition,
      rotate(torsoRotation, appVectorToPolicy(target.position))
    );
    if (target.orientation) {
      worldRotation = multiplyQuaternion(
        torsoRotation,
        appQuaternionToPolicy(target.orientation)
      );
    }
  }
  return poseRelativeTo(rootPosition, rootRotation, worldPosition, worldRotation);
}

function poseRelativeTo(
  rootPosition: PolicyVec3,
  rootRotation: PolicyQuaternion,
  worldPosition: PolicyVec3,
  worldRotation: PolicyQuaternion
): number[] {
  return [
    ...inverseRotate(rootRotation, subtract(worldPosition, rootPosition)),
    ...multiplyQuaternion(conjugateQuaternion(rootRotation), worldRotation)
  ];
}

function assertMetadata(metadata: WorkyardContactObservationMetadata): void {
  if (metadata.bodyDefaultJointPositions.length !== HUMANOID_JOINT_NAMES.length
    || metadata.bodyActionScale.length !== HUMANOID_JOINT_NAMES.length
    || [...metadata.bodyDefaultJointPositions, ...metadata.bodyActionScale]
      .some((value) => !Number.isFinite(value))
    || metadata.bodyActionScale.some((value) => value <= 0)
    || metadata.bodyDefaultJointPositions.some((value, index) => (
      Math.abs(value - YAHMP_POLICY.defaultJointPositions[index]!) > 1e-5
    ))
    || metadata.targetZoneId.trim().length === 0) {
    throw new Error("Workyard contact observation metadata is incompatible");
  }
}

function appVectorToPolicy(value: Vec3): PolicyVec3 {
  return [value.z, value.x, value.y];
}

function appQuaternionToPolicy(value: Quaternion): PolicyQuaternion {
  return normalizeQuaternion([value.w, value.z, value.x, value.y]);
}

function normalizeQuaternion(value: ArrayLike<number>): PolicyQuaternion {
  if (value.length !== 4) throw new Error("Policy quaternion must have four values");
  const magnitude = Math.hypot(value[0]!, value[1]!, value[2]!, value[3]!);
  if (!Number.isFinite(magnitude) || magnitude <= 1e-12) {
    throw new Error("Policy quaternion is invalid");
  }
  return [
    value[0]! / magnitude,
    value[1]! / magnitude,
    value[2]! / magnitude,
    value[3]! / magnitude
  ];
}

function conjugateQuaternion(
  [w, x, y, z]: PolicyQuaternion
): PolicyQuaternion {
  return [w, -x, -y, -z];
}

function multiplyQuaternion(
  [aw, ax, ay, az]: PolicyQuaternion,
  [bw, bx, by, bz]: PolicyQuaternion
): PolicyQuaternion {
  return normalizeQuaternion([
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw
  ]);
}

function inverseRotate(
  rotation: PolicyQuaternion,
  vector: PolicyVec3
): PolicyVec3 {
  return rotate(conjugateQuaternion(rotation), vector);
}

function rotate(
  [w, x, y, z]: PolicyQuaternion,
  [vx, vy, vz]: PolicyVec3
): PolicyVec3 {
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  return [
    vx + w * tx + (y * tz - z * ty),
    vy + w * ty + (z * tx - x * tz),
    vz + w * tz + (x * ty - y * tx)
  ];
}

function subtract(left: PolicyVec3, right: PolicyVec3): PolicyVec3 {
  return [left[0] - right[0], left[1] - right[1], left[2] - right[2]];
}

function add(left: PolicyVec3, right: PolicyVec3): PolicyVec3 {
  return [left[0] + right[0], left[1] + right[1], left[2] + right[2]];
}

function weightedVector(
  values: readonly PolicyVec3[],
  weights: readonly number[]
): PolicyVec3 {
  return values.reduce<PolicyVec3>((sum, value, index) => ([
    sum[0] + value[0] * weights[index]!,
    sum[1] + value[1] * weights[index]!,
    sum[2] + value[2] * weights[index]!
  ]), [0, 0, 0]);
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function finiteVector(
  input: ArrayLike<number>,
  size: number,
  label: string
): number[] {
  const values = Array.from(input);
  if (values.length !== size || values.some((value) => !Number.isFinite(value))) {
    throw new Error(`Workyard ${label} is invalid`);
  }
  return values;
}
