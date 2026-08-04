import { z } from "zod";
import { Vec3Schema } from "../../domain/schema.js";
import type { HumanoidActionName } from "../../domain/humanoid-action.js";
import {
  HumanoidMotionPlanSchema
} from "../../world/humanoid/motion-plan.js";
import { HumanoidMotionCandidateBatchInputSchema } from "./motion-candidate-input.js";

export type { HumanoidActionName } from "../../domain/humanoid-action.js";

export const HumanoidActionInputs = {
  observe_humanoid: z.object({}).strict(),
  plan_whole_body_motion: HumanoidMotionPlanSchema,
  plan_whole_body_motion_candidates: HumanoidMotionCandidateBatchInputSchema,
  execute_whole_body_motion: z.object({
    planning_transaction_id: z.string().trim().min(1)
  }).strict(),
  plan_humanoid_navigation: z.object({
    target: Vec3Schema
  }).strict(),
  execute_humanoid_navigation: z.object({
    planning_transaction_id: z.string().trim().min(1)
  }).strict(),
  remove_world_block: z.object({
    solid_id: z.string().trim().min(1),
    execution_transaction_id: z.string().trim().min(1)
  }).strict()
} as const;

export const HumanoidActionDescriptions: Record<HumanoidActionName, string> = {
  observe_humanoid: "读取当前人形机器人的 43 自由度状态、全身 Link、双脚接触、平衡、头部传感器可见物体、持久 3D 物体记录、掌指抓取证据和导航状态。不会暴露视野外且从未见过的对象。",
  plan_whole_body_motion: "提交根运动、躯干朝向、双手腕与双脚踝末端连续位姿组成的任务空间关键帧，以及可选的精确身体-物体接触约束。运动后端求解连续 G1 全身参考，运行时在当前状态上完整物理预演；末端不可达、碰错物体、失衡或跌倒会拒绝该计划。",
  plan_whole_body_motion_candidates: "提交受限 all/any/not 物理条件与可选 precondition/during/terminal 阶段，以及 2 至 3 个由模型按偏好排序的连续全身动作候选。终止条件可验收末端位置与朝向；每个候选从同一 MuJoCo 状态完整预演，只有持续约束未被违反且终止条件稳定达成的候选才可能被选择。",
  execute_whole_body_motion: "执行一个已接受的单候选或多候选全身动作规划回执。必须传入规划工具的 planning_transaction_id；Harness 验证来源、世界版本、终止契约、不可变运动制品与预演轨迹，逐帧判断物理目标，并在真实状态持续偏离预演时提前截断。",
  plan_humanoid_navigation: "为模型选择的世界坐标计算 Recast 路径，并用真实双足控制器对整条路径进行 MuJoCo 预演。不可行路线不会被替换成程序性移动。",
  execute_humanoid_navigation: "执行一个已接受的 plan_humanoid_navigation 回执。必须传入该规划工具调用的 planning_transaction_id；机器人通过双足策略闭环行走，禁止直接修改根节点位置。",
  remove_world_block: "拆除一个已由同一执行智能体在本自主周期内稳定接触的静态方块。必须引用最近一次成功全身执行回执并逐字使用其终止接触证据中的 solid_id；Harness 固定验收连续接触帧、法向力、规划来源和原子世界事务，不能删除固定对象或未接触方块。"
};
