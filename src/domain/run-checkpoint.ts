import {
  HumanoidRunCheckpointSchema,
  type HumanoidRunCheckpoint
} from "./humanoid-run.js";
import type { ZodType } from "zod";

export const AnyRunCheckpointSchema: ZodType<HumanoidRunCheckpoint> =
  HumanoidRunCheckpointSchema;

export type AnyRunCheckpoint = HumanoidRunCheckpoint;
