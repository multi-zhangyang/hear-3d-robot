import { z } from "zod";
import { Vec3Schema } from "../../domain/schema.js";
import type { HumanoidActionName } from "../../domain/humanoid-action.js";
import {
  HumanoidMotionPlanSchema
} from "../../world/humanoid/motion-plan.js";
import { HumanoidMotionCandidateBatchInputSchema } from "./motion-candidate-input.js";
import { HumanoidNavigationArrivalHeadingSchema } from "../../world/humanoid/navigation-arrival.js";
import { BeginHumanoidSkillSchema } from "../../domain/humanoid-skill.js";
import { HumanoidSkillPlanProposalSchema } from "../../domain/humanoid-skill-plan.js";

export type { HumanoidActionName } from "../../domain/humanoid-action.js";

export const HumanoidActionInputs = {
  observe_humanoid: z.object({}).strict(),
  submit_humanoid_skill_plan: HumanoidSkillPlanProposalSchema,
  begin_humanoid_skill: BeginHumanoidSkillSchema,
  plan_humanoid_skill: z.object({
    skill_transaction_id: z.string().trim().min(1)
  }).strict(),
  execute_humanoid_skill: z.object({
    planning_transaction_id: z.string().trim().min(1)
  }).strict(),
  plan_whole_body_motion: HumanoidMotionPlanSchema,
  plan_whole_body_motion_candidates: HumanoidMotionCandidateBatchInputSchema,
  execute_whole_body_motion: z.object({
    planning_transaction_id: z.string().trim().min(1)
  }).strict(),
  plan_humanoid_navigation: z.object({
    skill_transaction_id: z.string().trim().min(1).nullable().default(null),
    target: Vec3Schema.describe("机器人根节点要占据的开放地面世界坐标"),
    arrival_heading: HumanoidNavigationArrivalHeadingSchema.nullable()
      .describe("到达后必须物理满足的朝向；reachable_base_placements 给出 root_yaw_radians 时必须用 type=yaw 保留该偏航，否则可自主选择 face_point、yaw 或 null")
  }).strict().superRefine((input, context) => {
    if (input.arrival_heading?.type !== "face_point") return;
    const distance = Math.hypot(
      input.arrival_heading.target.x - input.target.x,
      input.arrival_heading.target.z - input.target.z
    );
    if (distance < 0.05) {
      context.addIssue({
        code: "custom",
        path: ["arrival_heading", "target"],
        message: "Facing point must be distinct from the root arrival target"
      });
    }
  }),
  execute_humanoid_navigation: z.object({
    planning_transaction_id: z.string().trim().min(1)
  }).strict(),
  remove_world_block: z.object({
    solid_id: z.string().trim().min(1),
    execution_transaction_id: z.string().trim().min(1)
  }).strict()
} as const;

export const HumanoidActionDescriptions: Record<HumanoidActionName, string> = {
  observe_humanoid: "读取当前人形机器人的根姿态、关键 Link、双脚接触、平衡、末端位置、手部协调状态、掌指碰撞面真实几何、必要接触、头部传感器可见物体、持久 3D 物体记录、抓取证据、携带生命周期和目标区域。不会暴露视野外且从未见过且未被当前持握的对象。",
  submit_humanoid_skill_plan: "提交一个局部 Skill DAG，可同时给出多个合法策略并由模型显式选择其中一个。Harness 验证节点、依赖和无环性并绑定当前世界版本；不会替模型补参数、选择策略或生成动作。",
  begin_humanoid_skill: "从实时对象世界模型中绑定一个统一人形 Skill 及其当前执行阶段。Harness 验证对象可见性、Affordance、交互点、手、关节、携带状态、世界版本和阶段执行权限；只建立模型选择的技能契约，不生成动作或坐标。",
  plan_humanoid_skill: "将模型选择的 Skill、对象、交互点、手和策略交给通用运动求解层。求解层自动生成并排序可达基座、闭环任务空间轨迹或信息增益 frontier 路线，逐个通过 Recast 与 MuJoCo 预演；不选择目标或任务策略。",
  execute_humanoid_skill: "执行 plan_humanoid_skill 产生且已通过真实物理预演的路线或全身轨迹。执行期间持续使用当前 MuJoCo 状态闭环修正末端和双足控制，不直接改写姿态。",
  plan_whole_body_motion: "提交根运动、躯干朝向、双手腕与双脚踝末端连续位姿组成的任务空间关键帧，以及可选的精确身体-物体接触约束。运动后端求解连续 G1 全身参考，运行时在当前状态上完整物理预演；末端不可达、碰错物体、失衡或跌倒会拒绝该计划。",
  plan_whole_body_motion_candidates: "提交一个可观测终止条件和 1 至 3 个按偏好排序的连续全身候选。谓词、接触目标和关键帧通道均使用带 type 的紧凑联合类型，只填写该类型真实需要的字段；每个候选从同一 MuJoCo 状态完整预演。",
  execute_whole_body_motion: "执行一个已接受的单候选或多候选全身动作规划回执。必须传入规划工具的 planning_transaction_id；Harness 验证来源、世界版本、终止契约、不可变运动制品与预演轨迹，逐帧判断物理目标，并在真实状态持续偏离预演时提前截断。",
  plan_humanoid_navigation: "为模型选择的根节点世界坐标计算 Recast 路径，并可要求到达后朝向指定世界点或偏航角。位置和朝向都由真实双足控制器在 MuJoCo 中预演；不可行路线不会被替换成程序性移动。",
  execute_humanoid_navigation: "执行一个已接受的 plan_humanoid_navigation 回执。必须传入该规划工具调用的 planning_transaction_id；机器人通过双足策略闭环行走，禁止直接修改根节点位置。",
  remove_world_block: "拆除一个已由同一执行智能体在本自主周期内稳定接触的静态方块。必须引用最近一次成功全身执行回执并逐字使用其终止接触证据中的 solid_id；Harness 固定验收连续接触帧、法向力、规划来源和原子世界事务，不能删除固定对象或未接触方块。"
};
