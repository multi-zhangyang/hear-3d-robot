import { createHash } from "node:crypto";
import { GoalSchema, type Goal } from "./schema.js";

export function goalSha256(goal: Goal): string {
  const canonical = GoalSchema.parse(goal);
  return createHash("sha256")
    .update(JSON.stringify(canonical))
    .digest("hex");
}
