import { z } from "zod";
import { Vec3Schema } from "../../domain/schema.js";
import type { HumanoidActionName } from "../../domain/humanoid-action.js";
import { HumanoidMotionPlanSchema } from "../../world/humanoid/motion-plan.js";

export type { HumanoidActionName } from "../../domain/humanoid-action.js";

export const HumanoidActionInputs = {
  observe_humanoid: z.object({}).strict(),
  plan_whole_body_motion: HumanoidMotionPlanSchema,
  execute_whole_body_motion: z.object({
    planning_transaction_id: z.string().trim().min(1)
  }).strict(),
  plan_humanoid_navigation: z.object({
    target: Vec3Schema
  }).strict(),
  execute_humanoid_navigation: z.object({
    planning_transaction_id: z.string().trim().min(1)
  }).strict()
} as const;

export const HumanoidActionDescriptions: Record<HumanoidActionName, string> = {
  observe_humanoid: "读取当前人形机器人的 29 个关节、全身 link、双脚接触、质心、平衡、头部传感器可见物体、持久 3D 物体记录和导航状态。不会暴露视野外且从未见过的对象。",
  plan_whole_body_motion: "提交根运动、躯干朝向与双手末端位置组成的任务空间关键帧，以及可选的精确身体-物体接触约束。运动后端求解连续 G1 全身参考，运行时在当前状态上完整物理预演；末端不可达、碰错物体、失衡或跌倒会拒绝该计划。",
  execute_whole_body_motion: "执行一个已接受的 plan_whole_body_motion 回执。必须传入该规划工具调用的 planning_transaction_id，harness 会验证来源、运动制品和世界版本；已加载的神经全身控制器产生关节控制，MuJoCo 处理重力、接触和平衡。",
  plan_humanoid_navigation: "为模型选择的世界坐标计算 Recast 路径，并用真实双足控制器对整条路径进行 MuJoCo 预演。不可行路线不会被替换成程序性移动。",
  execute_humanoid_navigation: "执行一个已接受的 plan_humanoid_navigation 回执。必须传入该规划工具调用的 planning_transaction_id；机器人通过双足策略闭环行走，禁止直接修改根节点位置。"
};
