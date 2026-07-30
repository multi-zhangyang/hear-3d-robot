/**
 * Builds the physics scene: ground, static obstacles, the robot's rigid-body
 * chain, and the scenario's objects.
 *
 * Construction is separated from simulation because it answers a different
 * question. Everything here is decided once from the scenario and the robot
 * spec and never changes again; everything in the world facade is about what
 * happens to those bodies over time. Splitting them means the rig's shape can
 * be read in one place without the motion logic wrapped around it.
 */
import RAPIER from "@dimforge/rapier3d-compat";
import type { Scenario } from "../domain/schema.js";
import { roundCuboidDesc } from "./collision.js";
import { scale } from "./geometry.js";
import { ROBOT_SPEC } from "./robot-model.js";

type World = InstanceType<typeof RAPIER.World>;
type RigidBody = InstanceType<typeof RAPIER.RigidBody>;
type Collider = InstanceType<typeof RAPIER.Collider>;

/** One scenario object plus the bodies simulating it. */
export interface SimObject {
  config: Scenario["objects"][number];
  body: RigidBody;
  collider: Collider;
  locked: boolean;
}

export interface BuiltScene {
  world: World;
  /** The base body — the rig's root and the only one navigation moves. */
  robot: RigidBody;
  /** Every robot link by id, including those with no collider. */
  linkBodies: Map<string, RigidBody>;
  /** Colliders for the links that have one; drives contact reporting. */
  linkColliders: Map<string, Collider>;
  gripperAnchor: RigidBody;
  leftFingerCollider: Collider;
  rightFingerCollider: Collider;
  objects: Map<string, SimObject>;
}

/**
 * Creates the world and every body in it, at the scenario's initial pose.
 * The rig is left at the base position; the caller poses the chain by joint
 * angles immediately afterwards.
 */
export function buildScene(scenario: Scenario): BuiltScene {
  const world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
  world.timestep = 1 / 60;
  // A carried dynamic body is constrained to a kinematic gripper while also
  // touching both fingers. Extra velocity-solver iterations keep that coupled
  // contact/joint system converged instead of misclassifying solver lag as a
  // physical grasp slip during an ordinary arm trajectory.
  world.numSolverIterations = 8;

  buildGround(world, scenario);
  buildObstacles(world, scenario);

  const linkBodies = new Map<string, RigidBody>();
  const linkColliders = new Map<string, Collider>();
  // The whole rig is kinematic: joint angles are commanded, so poses are
  // written directly rather than reached by solving forces.
  const createLink = (
    id: string,
    collider: InstanceType<typeof RAPIER.ColliderDesc> | null
  ): RigidBody => {
    const body = world.createRigidBody(
      RAPIER.RigidBodyDesc.kinematicPositionBased()
        .setTranslation(scenario.robot.x, ROBOT_SPEC.base.centerY, scenario.robot.z)
        .setUserData({ kind: "robot", link_id: id })
    );
    if (collider) {
      linkColliders.set(id, world.createCollider(collider.setFriction(0.85), body));
    }
    linkBodies.set(id, body);
    return body;
  };

  const robot = world.createRigidBody(
    RAPIER.RigidBodyDesc.kinematicPositionBased()
      .setTranslation(scenario.robot.x, ROBOT_SPEC.base.centerY, scenario.robot.z)
      .setRotation(yawRotation(scenario.robot.yaw))
      .setUserData({ kind: "robot", link_id: "base" })
  );
  const baseCollider = world.createCollider(
    roundCuboidDesc(ROBOT_SPEC.base.halfExtents, ROBOT_SPEC.base.cornerRadius).setFriction(0.9),
    robot
  );
  linkBodies.set("base", robot);
  linkColliders.set("base", baseCollider);

  createLink("torso", roundCuboidDesc(ROBOT_SPEC.torso.halfExtents, 0.05));
  createLink("sensor_head", roundCuboidDesc(ROBOT_SPEC.sensorHead.halfExtents, 0.05));
  createLink("upper_arm", cuboidDesc(ROBOT_SPEC.arm.upperHalfExtents));
  createLink("forearm", cuboidDesc(ROBOT_SPEC.arm.forearmHalfExtents));
  createLink("wrist", cuboidDesc(ROBOT_SPEC.arm.wristHalfExtents));

  // The gripper anchor carries no collider of its own: it is the frame the
  // fingers and any grasp constraint hang from, not a solid part of the rig.
  const gripperAnchor = createLink("gripper", null);
  const leftFinger = createLink("left_finger", null);
  const rightFinger = createLink("right_finger", null);
  // Fingers are gripped against, so they get the highest friction in the scene.
  const fingerDesc = (): InstanceType<typeof RAPIER.ColliderDesc> => roundCuboidDesc(
    ROBOT_SPEC.gripper.fingerHalfExtents,
    ROBOT_SPEC.gripper.cornerRadius
  ).setFriction(1.1);
  const leftFingerCollider = world.createCollider(fingerDesc(), leftFinger);
  const rightFingerCollider = world.createCollider(fingerDesc(), rightFinger);
  linkColliders.set("left_finger", leftFingerCollider);
  linkColliders.set("right_finger", rightFingerCollider);

  return {
    world,
    robot,
    linkBodies,
    linkColliders,
    gripperAnchor,
    leftFingerCollider,
    rightFingerCollider,
    objects: buildObjects(world, scenario)
  };
}

/** A slab centred under the scene; thick enough that nothing tunnels through. */
function buildGround(world: World, scenario: Scenario): void {
  const ground = world.createRigidBody(
    RAPIER.RigidBodyDesc.fixed()
      .setTranslation(scenario.bounds.width / 2, -0.08, scenario.bounds.depth / 2)
      .setUserData({ kind: "ground" })
  );
  world.createCollider(
    RAPIER.ColliderDesc.cuboid(scenario.bounds.width / 2, 0.08, scenario.bounds.depth / 2)
      .setFriction(0.9),
    ground
  );
}

function buildObstacles(world: World, scenario: Scenario): void {
  for (const obstacle of scenario.obstacles) {
    const body = world.createRigidBody(
      RAPIER.RigidBodyDesc.fixed()
        .setTranslation(obstacle.center.x, obstacle.center.y, obstacle.center.z)
        .setUserData({ kind: "obstacle", id: obstacle.id })
    );
    world.createCollider(
      RAPIER.ColliderDesc.cuboid(
        obstacle.size.x / 2,
        obstacle.size.y / 2,
        obstacle.size.z / 2
      ).setFriction(0.8),
      body
    );
  }
}

function buildObjects(world: World, scenario: Scenario): Map<string, SimObject> {
  const objects = new Map<string, SimObject>();
  for (const object of scenario.objects) {
    // Portability is what decides whether an object can be carried at all, so
    // it decides the body type: fixed scenery versus a dynamic body the
    // gripper can constrain. Damping keeps a released object from skating.
    const descriptor = object.portable
      ? RAPIER.RigidBodyDesc.dynamic()
          .setLinearDamping(0.8)
          .setAngularDamping(0.9)
          // A released payload may fall from the top of a voxel column. Hard
          // CCD keeps the small rounded body from tunnelling through a thin
          // floor or platform during the high-speed part of that real fall.
          .setCcdEnabled(true)
      : RAPIER.RigidBodyDesc.fixed();
    const body = world.createRigidBody(
      descriptor
        .setTranslation(object.position.x, object.position.y, object.position.z)
        .setUserData({ kind: "object", id: object.id })
    );
    // Rounded corners keep contact normals stable when a finger meets an edge;
    // the radius is capped so it never rounds a small object into a ball.
    const collider = world.createCollider(
      roundCuboidDesc(
        scale(object.size, 0.5),
        Math.min(0.04, object.size.x / 8, object.size.z / 8)
      )
        .setDensity(object.portable ? 0.8 : 0)
        .setFriction(0.75),
      body
    );
    objects.set(object.id, { config: object, body, collider, locked: object.locked ?? false });
  }
  return objects;
}

function cuboidDesc(halfExtents: { x: number; y: number; z: number }):
InstanceType<typeof RAPIER.ColliderDesc> {
  return RAPIER.ColliderDesc.cuboid(halfExtents.x, halfExtents.y, halfExtents.z);
}

function yawRotation(yaw: number): { x: number; y: number; z: number; w: number } {
  return { x: 0, y: Math.sin(yaw / 2), z: 0, w: Math.cos(yaw / 2) };
}
