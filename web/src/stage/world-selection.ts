import type { Vec3, VoxelCoordinate, VoxelMaterial, WorldSnapshot } from "../types";
import { entityLabel, materialLabel } from "../ui-text";

export type WorldEntityType = "robot" | "object" | "obstacle" | "zone";

export type WorldSelection =
  | {
      kind: "voxel";
      coordinate: VoxelCoordinate;
      material: VoxelMaterial | "ground";
    }
  | {
      kind: "entity";
      entityType: WorldEntityType;
      id: string;
    };

export interface WorldSelectionView {
  badge: string;
  title: string;
  detail: string;
}

export function worldSelectionKey(selection: WorldSelection): string {
  return selection.kind === "voxel"
    ? `voxel:${selection.coordinate.column}:${selection.coordinate.level}:${selection.coordinate.row}`
    : `${selection.entityType}:${selection.id}`;
}

export function sameWorldSelection(
  left: WorldSelection | null,
  right: WorldSelection | null
): boolean {
  if (left === null || right === null) return left === right;
  if (worldSelectionKey(left) !== worldSelectionKey(right)) return false;
  return left.kind !== "voxel"
    || (right.kind === "voxel" && left.material === right.material);
}

export function describeWorldSelection(
  selection: WorldSelection,
  frame: WorldSnapshot
): WorldSelectionView {
  if (selection.kind === "voxel") {
    const { column, level, row } = selection.coordinate;
    return {
      badge: selection.material === "ground" ? "地形" : "体素",
      title: materialLabel(selection.material),
      detail: `[${column}, ${level}, ${row}]`
    };
  }

  if (selection.entityType === "robot") {
    return {
      badge: "机器人",
      title: "具身智能体",
      detail: formatPosition(frame.robot.position)
    };
  }

  if (selection.entityType === "object") {
    const object = frame.objects.find((candidate) => candidate.id === selection.id);
    return {
      badge: object ? objectKindLabel(object.kind) : "物体",
      title: entityLabel(selection.id),
      detail: object ? formatPosition(object.position) : "不可用"
    };
  }

  if (selection.entityType === "obstacle") {
    const obstacle = frame.obstacles.find((candidate) => candidate.id === selection.id);
    return {
      badge: "结构",
      title: entityLabel(selection.id),
      detail: obstacle ? formatPosition(obstacle.center) : "不可用"
    };
  }

  const zone = frame.zones.find((candidate) => candidate.id === selection.id);
  return {
    badge: "区域",
    title: entityLabel(selection.id),
    detail: zone ? formatPosition(zone.center) : "不可用"
  };
}

function formatPosition(position: Vec3): string {
  return `${position.x.toFixed(1)}, ${position.y.toFixed(1)}, ${position.z.toFixed(1)}`;
}

function objectKindLabel(value: string): string {
  const labels: Record<string, string> = {
    block: "方块",
    container: "容器",
    key: "钥匙",
    prop: "物体"
  };
  return labels[value] ?? "物体";
}
