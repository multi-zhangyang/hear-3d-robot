import { createHash } from "node:crypto";

const NON_PHYSICAL_PLANNING_KEYS = new Set([
  "id",
  "intent",
  "option_id"
]);

export function autonomyContentSha256(value: unknown): string {
  return createHash("sha256")
    .update(JSON.stringify(canonicalValue(value)))
    .digest("hex");
}

export function autonomyPlanningBehaviorSha256(
  action: string,
  input: unknown
): string {
  return autonomyContentSha256({
    action,
    input: normalizeAutonomyPlanningInput(input)
  });
}

export function normalizeAutonomyPlanningInput(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(normalizeAutonomyPlanningInput);
  if (!record(value)) return value;
  return Object.fromEntries(Object.entries(value)
    .filter(([key]) => !NON_PHYSICAL_PLANNING_KEYS.has(key))
    .map(([key, entry]) => [key, normalizeAutonomyPlanningInput(entry)]));
}

export function quantizeAutonomyVec3(value: unknown): {
  x: number | null;
  y: number | null;
  z: number | null;
} | null {
  if (!record(value)) return null;
  const coordinate = (axis: "x" | "y" | "z"): number | null => (
    typeof value[axis] === "number" && Number.isFinite(value[axis])
      ? Math.round(value[axis] * 1_000) / 1_000
      : null
  );
  return { x: coordinate("x"), y: coordinate("y"), z: coordinate("z") };
}

function canonicalValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalValue);
  if (!record(value)) return value;
  return Object.fromEntries(Object.entries(value)
    .filter(([, entry]) => entry !== undefined)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, entry]) => [key, canonicalValue(entry)]));
}

function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
