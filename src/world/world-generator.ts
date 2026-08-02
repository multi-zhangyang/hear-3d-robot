/**
 * Turns a template and a seed into a concrete scenario.
 *
 * A template says what kind of world to build — how large, how broken up, which
 * entities exist and what they look like. It does not say where anything is.
 * Every position, the robot's start pose and heading, and its starting arm
 * configuration are drawn from the seed, so two runs of the same template are
 * two different worlds rather than the same one replayed.
 *
 * The seed is stored in the scenario the run persists, which is what keeps that
 * variation compatible with resuming: a checkpoint rebuilds the world it was
 * taken from, not a fresh roll.
 */
import { randomBytes } from "node:crypto";
import type { Scenario, ScenarioTemplate } from "../domain/schema.js";
import { ROBOT_SPEC, type RobotJointState } from "./robot-model.js";
import { createRandom, deriveSeed, randomBetween, shuffle } from "./random.js";
import {
  cellCenter,
  cellClearance,
  generateTerrain,
  pointInCell,
  walkableCells,
  type Cell
} from "./terrain.js";

/**
 * How far apart placed entities must be, as a multiple of the terrain cell.
 * Enough that the mission always involves crossing ground rather than turning
 * on the spot, and that two objects never share an approach pose.
 */
const MINIMUM_SEPARATION_CELLS = 4;

export function materializeScenario(
  template: ScenarioTemplate,
  seed: number,
  motionSeed = deriveSeed(seed, "motion")
): Scenario {
  if (template.kind === "authored") {
    return { ...structuredClone(template.scenario), seed, motion_seed: motionSeed };
  }

  const generator = template.generate;
  const terrain = generateTerrain(generator.terrain, seed);
  const bounds = {
    width: terrain.columns * terrain.cell,
    depth: terrain.rows * terrain.cell
  };

  const random = createRandom(deriveSeed(seed, "placement"));
  const footprint = ROBOT_SPEC.base.footprintRadius;
  // Somewhere the base can stand without any part of it inside a raised column,
  // with a little slack so it is not wedged against one from the first frame.
  const open = shuffle(
    walkableCells(terrain).filter((cell) => cellClearance(terrain, cell, footprint * 1.25)),
    random
  );
  if (open.length === 0) {
    throw new Error(`Generated terrain for seed ${seed} has no room for the robot`);
  }

  const taken: Cell[] = [];
  const claim = (radius: number): Cell => {
    const cell = open.find((candidate) =>
      cellClearance(terrain, candidate, radius)
      && taken.every((other) => cellDistance(other, candidate) >= MINIMUM_SEPARATION_CELLS));
    if (!cell) {
      throw new Error(
        `Generated terrain for seed ${seed} cannot place ${taken.length + 1} entities `
        + `with ${MINIMUM_SEPARATION_CELLS} cells of separation`
      );
    }
    open.splice(open.indexOf(cell), 1);
    taken.push(cell);
    return cell;
  };

  const robotCell = claim(footprint * 1.25);
  const robotPoint = cellCenter(terrain, robotCell);

  const objects = generator.objects.map((object) => {
    // Clearance for the object itself plus the base standing beside it, so
    // every generated object has at least one legal approach pose.
    const reach = Math.max(object.size.x, object.size.z) / 2 + footprint * 1.1;
    const cell = claim(reach);
    const point = pointInCell(terrain, cell, random, Math.max(object.size.x, object.size.z) / 2);
    return {
      ...structuredClone(object),
      position: { x: point.x, y: object.size.y / 2, z: point.z }
    };
  });

  const zones = generator.zones.map((zone) => {
    const cell = claim(Math.max(zone.size.x, zone.size.z) / 2 + footprint * 1.1);
    const point = cellCenter(terrain, cell);
    return {
      ...structuredClone(zone),
      center: { x: point.x, y: 0.01, z: point.z }
    };
  });

  return {
    title: template.title,
    seed,
    motion_seed: motionSeed,
    bounds,
    terrain,
    visibility_radius: generator.visibility_radius,
    robot: {
      x: robotPoint.x,
      z: robotPoint.z,
      // A start heading, not just a start position. Facing a fixed direction
      // every run makes the first turn the same turn every run.
      yaw: randomBetween(random, -Math.PI, Math.PI),
      joints: startingJoints(seed)
    },
    obstacles: [],
    objects,
    zones,
    affordances: [],
    default_goal: structuredClone(generator.default_goal)
  };
}

/**
 * A start configuration for the arm and head, drawn from the seed inside a
 * conservative window of each joint's range.
 *
 * The window matters: a pose sampled across the full range is as likely to be a
 * self-collision as a usable stance, and the robot would begin the mission
 * already stuck. Sampling around the neutral stance instead varies where the
 * arm starts without varying whether the arm is legal.
 */
function startingJoints(seed: number): RobotJointState {
  const random = createRandom(deriveSeed(seed, "joints"));
  const around = (joint: keyof typeof ROBOT_SPEC.joints, span: number): number => {
    const limit = ROBOT_SPEC.joints[joint];
    const centre = ROBOT_SPEC.defaultJoints[joint];
    const reach = (limit.maximum - limit.minimum) * span;
    return Math.min(
      limit.maximum,
      Math.max(limit.minimum, centre + randomBetween(random, -reach, reach))
    );
  };
  return {
    head_yaw: around("head_yaw", 0.2),
    head_pitch: around("head_pitch", 0.15),
    shoulder: around("shoulder", 0.06),
    elbow: around("elbow", 0.06),
    wrist: around("wrist", 0.08),
    gripper_aperture: around("gripper_aperture", 0.12)
  };
}

/** A run seed drawn from the platform's randomness, used once and then stored. */
export function drawSeed(): number {
  return randomBytes(4).readUInt32LE(0);
}

function cellDistance(left: Cell, right: Cell): number {
  return Math.hypot(left.column - right.column, left.row - right.row);
}
