import { z } from "zod";

export const HumanoidMotionGeneratorDescriptorSchema = z.object({
  protocol: z.literal("humanoid-motion-generator-v1"),
  implementation: z.string().trim().min(1),
  motionClass: z.enum(["constraint_solver", "generative_model"]),
  sampling: z.enum(["deterministic", "stochastic"])
}).strict();

export type HumanoidMotionGeneratorDescriptor = z.infer<
  typeof HumanoidMotionGeneratorDescriptorSchema
>;

export const TASK_SPACE_MOTION_GENERATOR_DESCRIPTOR = {
  protocol: "humanoid-motion-generator-v1",
  implementation: "task_space_constraints",
  motionClass: "constraint_solver",
  sampling: "deterministic"
} as const satisfies HumanoidMotionGeneratorDescriptor;
