import { randomUUID } from "node:crypto";
import { z } from "zod";

const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const GOAL_EPOCH_PATTERN = /^goal-epoch:[a-f0-9]{64}$/;

const AutonomousCycleIdSchema = z.string().regex(
  /^autonomous-cycle:[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
);

export const AutonomousCycleRefSchema = z.object({
  cycle_id: AutonomousCycleIdSchema,
  cycle_index: z.number().int().positive(),
  goal_epoch_id: z.string().regex(GOAL_EPOCH_PATTERN)
}).strict();

export const ActiveAutonomousCycleSchema = AutonomousCycleRefSchema.extend({
  started_world_frame: z.number().int().nonnegative(),
  started_world_revision: z.number().int().nonnegative(),
  started_at: z.string().datetime()
}).strict();

export const EmbodiedMemoryIdSchema = z.string().regex(
  /^embodied-memory:[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
);

export type AutonomousCycleRef = z.infer<typeof AutonomousCycleRefSchema>;
export type ActiveAutonomousCycle = z.infer<typeof ActiveAutonomousCycleSchema>;

export function createActiveAutonomousCycle(input: {
  cycleIndex: number;
  goalEpochId: string;
  worldFrame: number;
  worldRevision: number;
  cycleUuid?: string;
  startedAt?: string;
}): ActiveAutonomousCycle {
  const cycleUuid = input.cycleUuid ?? randomUUID();
  if (!UUID_PATTERN.test(cycleUuid)) {
    throw new Error("Autonomous cycle UUID is invalid");
  }
  return ActiveAutonomousCycleSchema.parse({
    cycle_id: `autonomous-cycle:${cycleUuid.toLowerCase()}`,
    cycle_index: input.cycleIndex,
    goal_epoch_id: input.goalEpochId,
    started_world_frame: input.worldFrame,
    started_world_revision: input.worldRevision,
    started_at: input.startedAt ?? new Date().toISOString()
  });
}

export function autonomousCycleRef(
  cycle: ActiveAutonomousCycle
): AutonomousCycleRef {
  return AutonomousCycleRefSchema.parse({
    cycle_id: cycle.cycle_id,
    cycle_index: cycle.cycle_index,
    goal_epoch_id: cycle.goal_epoch_id
  });
}

export function embodiedMemoryIdForCycle(
  cycle: AutonomousCycleRef
): string {
  const parsed = AutonomousCycleRefSchema.parse(cycle);
  return EmbodiedMemoryIdSchema.parse(
    parsed.cycle_id.replace("autonomous-cycle:", "embodied-memory:")
  );
}

export function sameAutonomousCycle(
  left: AutonomousCycleRef | null | undefined,
  right: AutonomousCycleRef | null | undefined
): boolean {
  return left !== null
    && left !== undefined
    && right !== null
    && right !== undefined
    && left.cycle_id === right.cycle_id
    && left.cycle_index === right.cycle_index
    && left.goal_epoch_id === right.goal_epoch_id;
}
