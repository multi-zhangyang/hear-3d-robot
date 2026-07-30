import type {
  CheckerResult,
  Goal,
  JsonValue,
  VoxelCoordinate,
  VoxelMaterial,
  WorldSnapshot
} from "../domain/schema.js";

export function checkGoal(
  goal: Goal,
  world: WorldSnapshot,
  voxelMaterialAt?: (coordinate: VoxelCoordinate) => VoxelMaterial | null
): CheckerResult {
  const checks: CheckerResult["checks"] = goal.predicates.map((predicate, index) => {
    const name = `${index + 1}:${predicate.type}`;
    if (predicate.type === "robot_at") {
      const distance = planarDistance(world.robot.position, predicate.target);
      return {
        name,
        passed: distance <= predicate.tolerance,
        actual: {
          distance,
          tolerance: predicate.tolerance,
          position: world.robot.position,
          target: predicate.target
        } as JsonValue
      };
    }

    if (predicate.type === "robot_in_zone") {
      const zone = world.zones.find((candidate) => candidate.id === predicate.zone_id);
      const gap = zone ? planarDistance(world.robot.position, zone.center) : null;
      const inside = zone !== undefined
        && Math.abs(world.robot.position.x - zone.center.x) <= zone.size.x / 2 + predicate.tolerance
        && Math.abs(world.robot.position.z - zone.center.z) <= zone.size.z / 2 + predicate.tolerance;
      return {
        name,
        passed: inside,
        actual: {
          zone_id: predicate.zone_id,
          robot_position: world.robot.position,
          zone_center: zone?.center ?? null,
          zone_size: zone?.size ?? null,
          distance: gap,
          inside,
          tolerance: predicate.tolerance
        } as JsonValue
      };
    }

    if (predicate.type === "terrain_explored") {
      const fraction = world.explored.total === 0
        ? 0
        : world.explored.seen / world.explored.total;
      return {
        name,
        passed: fraction >= predicate.minimum_fraction,
        actual: {
          cells_seen: world.explored.seen,
          cells_total: world.explored.total,
          fraction,
          minimum_fraction: predicate.minimum_fraction
        } as JsonValue
      };
    }

    if (predicate.type === "voxel_at") {
      const material = voxelMaterialAt
        ? voxelMaterialAt(predicate.coordinate)
        : materialFromMutation(world, predicate.coordinate);
      return {
        name,
        passed: material === predicate.material,
        actual: {
          coordinate: predicate.coordinate,
          material,
          expected_material: predicate.material,
          voxel_revision: world.voxels?.revision ?? null
        } as JsonValue
      };
    }

    const object = world.objects.find((candidate) => candidate.id === predicate.object_id);
    if (predicate.type === "object_in_zone") {
      const zone = world.zones.find((candidate) => candidate.id === predicate.zone_id);
      const inside = object !== undefined && zone !== undefined
        && objectInsideZone(object, zone, predicate.tolerance);
      return {
        name,
        passed: object !== undefined && zone !== undefined && inside === predicate.expected,
        actual: {
          object_id: predicate.object_id,
          object_position: object?.position ?? null,
          object_size: object?.size ?? null,
          zone_id: predicate.zone_id,
          zone_center: zone?.center ?? null,
          zone_size: zone?.size ?? null,
          inside,
          expected: predicate.expected,
          tolerance: predicate.tolerance
        } as JsonValue
      };
    }
    if (predicate.type === "object_at") {
      const distance = object ? distance3(object.position, predicate.target) : null;
      return {
        name,
        passed: distance !== null && distance <= predicate.tolerance,
        actual: {
          object_id: predicate.object_id,
          position: object?.position ?? null,
          target: predicate.target,
          distance,
          tolerance: predicate.tolerance
        } as JsonValue
      };
    }
    if (predicate.type === "object_property") {
      const value = object?.[predicate.property] ?? null;
      return {
        name,
        passed: value === predicate.expected,
        actual: {
          object_id: predicate.object_id,
          property: predicate.property,
          value,
          expected: predicate.expected
        } as JsonValue
      };
    }

    const attachedObjectId = world.robot.attachment?.object_id ?? null;
    const attached = attachedObjectId === predicate.object_id;
    return {
      name,
      passed: object !== undefined && attached === predicate.expected,
      actual: {
        object_id: predicate.object_id,
        attached,
        attached_object_id: attachedObjectId,
        expected: predicate.expected
      } as JsonValue
    };
  });

  return {
    success: checks.every((check) => check.passed),
    goal,
    world_frame: world.frame,
    world_revision: world.world_revision,
    checks,
    checked_at: new Date().toISOString()
  };
}

/**
 * Says which predicates are still open and what physically has to happen for
 * each one to close.
 *
 * A verdict alone is not actionable. Each unmet predicate therefore includes
 * the physical state change required to satisfy it, preventing unchanged-world
 * checks from becoming the only available next step.
 *
 * The text is derived from the predicate and the measured state, so it stays
 * true for any goal the scenario file declares rather than describing one
 * mission.
 */
export function unmetGoalRecovery(result: CheckerResult): Record<string, JsonValue> {
  const failed = result.checks.filter((check) => !check.passed);
  if (failed.length === 0) return {};
  const remedies = failed.map((check) => `${check.name}: ${remedyFor(check.name, check.actual)}`);
  return {
    unmet_predicates: failed.map((check) => check.name),
    recovery: "The mission is not complete and the world has not changed since the last check, "
      + "so checking again returns this same answer. Each unmet predicate below names the "
      + "physical change that would close it — delegate that work, then check once more. "
      + remedies.join(" ")
  };
}

function remedyFor(name: string, actual: JsonValue): string {
  const detail = (actual ?? {}) as Record<string, JsonValue>;
  if (name.endsWith("robot_in_zone")) {
    const gap = typeof detail.distance === "number" ? detail.distance : null;
    const zone = point(detail.zone_center ?? null);
    return `the robot is not standing in ${String(detail.zone_id)}`
      + (gap === null ? ", and the zone has not been seen yet" : `, ${gap.toFixed(2)}m from its center`)
      + (zone === null
        ? ". Survey the terrain for unexplored frontier cells and drive towards them until the zone comes into view."
        : ` at (${zone.x.toFixed(2)}, ${zone.z.toFixed(2)}). Plan a base path onto that point and execute it.`);
  }
  if (name.endsWith("terrain_explored")) {
    const fraction = typeof detail.fraction === "number" ? detail.fraction : 0;
    const minimum = typeof detail.minimum_fraction === "number" ? detail.minimum_fraction : 0;
    return `the robot has mapped ${(fraction * 100).toFixed(1)}% of the terrain and must reach `
      + `${(minimum * 100).toFixed(1)}%. Survey the current frontier, choose a reachable target `
      + "that faces unseen ground, drive there, and survey again.";
  }
  if (name.endsWith("voxel_at")) {
    const coordinate = detail.coordinate as Record<string, JsonValue> | undefined;
    const expected = detail.expected_material;
    const current = detail.material;
    return `voxel (${String(coordinate?.column)}, ${String(coordinate?.level)}, `
      + `${String(coordinate?.row)}) is ${JSON.stringify(current)} and must be `
      + `${JSON.stringify(expected)}. Inspect that voxel from the current world, move the gripper `
      + "to its reported interaction point, then break or place exactly that coordinate.";
  }
  if (name.endsWith("object_in_zone")) {
    const object = point(detail.object_position ?? null);
    const zone = point(detail.zone_center ?? null);
    const gap = object && zone ? Math.hypot(object.x - zone.x, object.z - zone.z) : null;
    return `${String(detail.object_id)} is not resting in ${String(detail.zone_id)}`
      + (gap === null ? "" : `, ${gap.toFixed(2)}m away from its center`)
      + `. Grasp it, drive the base to a standoff pose facing ${String(detail.zone_id)} `
      + "— inspect_entity on the zone lists reachable ones — then release it low enough "
      + "over the zone surface that it settles inside rather than bouncing out.";
  }
  if (name.endsWith("object_attached")) {
    return detail.expected === true
      ? `${String(detail.object_id)} must be held. Solve an end-effector pose at its grasp `
        + "point and close the gripper on it."
      : `${String(detail.object_id)} is still attached to the gripper. Open the gripper to `
        + "release it where it should stay.";
  }
  if (name.endsWith("object_at") || name.endsWith("robot_at")) {
    const gap = typeof detail.distance === "number" ? detail.distance : null;
    return `still ${gap === null ? "short of" : `${gap.toFixed(2)}m from`} the target position`
      + ` against a tolerance of ${String(detail.tolerance)}. Move it closer.`;
  }
  return `${String(detail.property)} on ${String(detail.object_id)} reads `
    + `${JSON.stringify(detail.value)} and must read ${JSON.stringify(detail.expected)}.`;
}

function materialFromMutation(
  world: WorldSnapshot,
  coordinate: VoxelCoordinate
): VoxelMaterial | null {
  const mutation = world.voxels?.mutations.findLast((candidate) =>
    candidate.coordinate.column === coordinate.column
      && candidate.coordinate.level === coordinate.level
      && candidate.coordinate.row === coordinate.row
  );
  return mutation?.after ?? null;
}

function point(value: JsonValue): { x: number; z: number } | null {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return null;
  const record = value as Record<string, JsonValue>;
  return typeof record.x === "number" && typeof record.z === "number"
    ? { x: record.x, z: record.z }
    : null;
}

function objectInsideZone(
  object: { position: { x: number; y: number; z: number }; size: { x: number; y: number; z: number } },
  zone: { center: { x: number; y: number; z: number }; size: { x: number; y: number; z: number } },
  tolerance: number
): boolean {
  const objectBottom = object.position.y - object.size.y / 2;
  const zoneSurface = zone.center.y + zone.size.y / 2;
  const supportTolerance = Math.max(tolerance, 0.025);
  return Math.abs(object.position.x - zone.center.x) + object.size.x / 2
      <= zone.size.x / 2 + tolerance
    && Math.abs(object.position.z - zone.center.z) + object.size.z / 2
      <= zone.size.z / 2 + tolerance
    && Math.abs(objectBottom - zoneSurface) <= supportTolerance;
}

function planarDistance(left: { x: number; z: number }, right: { x: number; z: number }): number {
  return Math.hypot(left.x - right.x, left.z - right.z);
}

function distance3(
  left: { x: number; y: number; z: number },
  right: { x: number; y: number; z: number }
): number {
  return Math.hypot(left.x - right.x, left.y - right.y, left.z - right.z);
}
