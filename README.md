# HEAR

[![CI](https://github.com/multi-zhangyang/hear-3d-robot/actions/workflows/ci.yml/badge.svg)](https://github.com/multi-zhangyang/hear-3d-robot/actions/workflows/ci.yml)

HEAR 是一个由层级智能体自主驱动的虚拟 3D 人形机器人运行时。模型负责观察、选择目标和提出任务空间动作；Harness 负责权限、计划来源、物理预演、执行回执和目标验收；Unitree G1 模型在 MuJoCo 中通过全身控制器完成实际运动。

![HEAR 人形机器人世界](docs/screenshots/mission.png)

## 核心能力

- OpenAI Agents SDK 层级编排，每个智能体拥有独立 Model、持久 Session、缓存亲和键和上下文生命周期
- 遮挡感知的持久空间信念、未知区域 frontier、对象中心世界模型、可供性目录与模型提交的 Skill DAG
- G1 29 个全身关节与 14 个手部关节、双足接触、质心、支撑面和跌倒检测
- 模型选择 Goal、Skill、对象、手、交互点和策略；通用求解层自动生成可达站位与任务空间轨迹
- 任务可约束左/右手腕或脚踝的世界或骨盆相对三维位姿，并要求位置、朝向在连续物理帧内稳定成立
- 双手掌指接触、抓取稳定、携带绑定、主动释放与支撑面稳放提供真实物理验收，不以坐标吸附替代策略能力
- 每个几何候选从同一 MuJoCo 状态独立预演，只执行首个满足物理约束与 Skill 终止条件的候选
- 全身 Option 使用受限物理条件树和前置、持续、终止三个阶段，成功必须经过连续帧稳定验收
- 预演轨迹与运动制品分别进行 SHA-256 校验；真实执行持续偏离预演时立即截断并重新规划
- 预演与真实执行分别累计支撑余量、足底滑移、关节余量、速度、接触冲击和执行器力矩证据
- 默认 mjlab G1 学习策略在 50 Hz 执行平衡和移动，YAHMP 在需要关节参考跟踪时接管对应控制步，MuJoCo 在 200 Hz 处理重力、力矩、碰撞与接触
- Recast 导航网格和分段物理预演，不通过修改根节点坐标伪造移动
- 导航遇到实时阻塞时保持语义目标不变并从当前物理状态在线重规划，安全失败返回上层恢复
- 可见方块可通过接近、稳定掌指接触、执行授权和原子世界事务完成通用拆除
- 程序化方块世界，世界种子、障碍、物体和目标区域按任务生成
- 头部视场感知与角色化 3D 对象状态，当前 MuJoCo 精确状态和历史观察严格分离
- 追加式事件日志、可寻址具身历史、物理检查点、独立会话和结构化上下文压缩
- 中文实时界面，展示 3D 世界、层级执行流、动作回执、模型活动与物理状态
- Three.js 优先使用 WebGPU，并兼容 WebGL2
- Linux、Windows 和 macOS 持续集成

## 运行链路

```text
任务目标
  └─ 自主目标管理智能体
       │ Goal
       ▼
人形自主协调智能体
  ├─ 人形感知哨兵
  ├─ 全身运动参考智能体
  └─ 人形物理执行智能体
       │
       ▼
模型选择的 Goal → 局部 Skill DAG
       │
       ▼
通用 Skill 求解器 → Recast / 任务空间 IK 候选
       │ 独立 MuJoCo 预演
       ▼
已验证路线或全身物理 Option
       │
       ▼
能力路由 → 学习式全身控制 / 参考控制 → MuJoCo 实际执行
       │ 达成 / 在线重规划 / 语义恢复
       ▼
权威世界帧与动作回执 → 目标检查 / 滚动重规划
```

目标管理、协调、感知、运动参考和物理执行是五个边界明确的智能体，而不是一个模型扮演多个名称。协调智能体不能直接修改世界；目标管理智能体负责提出并选择当前 Goal；运动参考智能体只能提交规划；执行智能体只能消费当前世界版本中已接受规划的原始回执。

一次动作需要同时满足以下条件才会改变世界：

1. 工具参数通过严格 schema 校验。
2. 规划回执属于正确智能体和当前世界版本。
3. Skill、对象、手、交互点与策略来自当前运动智能体的真实模型响应。
4. 通用求解器至少生成一个与该语义身份一致的可达任务空间候选。
5. 被选候选的完整 MuJoCo 预演没有跌倒、非法接触、持续条件违规或缺失的必需接触。
6. 运动制品、终止合约和逐帧预演轨迹与规划证书一致。
7. 真实执行逐帧满足物理 Option；目标稳定达成后立即停止，持续偏离预演或违反条件时立即交回重规划。

自主差异来自模型对实时观察、空间 frontier、对象可供性、目标和历史回执的决策，以及每次任务独立生成的世界。程序不会从预设动作表挑选行为，也不会用随机电机噪声代替自主决策。Harness 只把模型选定的语义 Skill 求解为物理候选，不会改换目标、对象、手或策略。

## 机器人与世界

默认机器人使用 Unitree G1 模型，身体控制按六个通道组织：

| 通道 | 范围 |
|---|---|
| `locomotion` | 根速度、朝向与神经双足步态 |
| `left_leg` | 左腿运动链与左脚踝末端目标 |
| `right_leg` | 右腿运动链与右脚踝末端目标 |
| `torso` | 腰部与躯干姿态 |
| `left_arm` | 左臂和左手末端 |
| `right_arm` | 右臂和右手末端 |

标准运动链只接收语义 Skill。模型决定探索 frontier、操作对象、交互点、左右手和策略，求解层再从当前腕部碰撞面、对象尺寸、关节轴、铰链锚点、持物绑定和根姿态计算站位与末端目标。多末端 SE(3) 阻尼最小二乘逆运动学生成连续腿臂参考，学习式全身控制器生成实际关节控制。接近、伸手、抓取、抬升、搬运、放置、推拉、按压、开合、换手、双手支撑、稳定、撤退和方块交互使用同一套 Skill 约束表达；当前控制器无法完成的候选会被物理预演拒绝，不会由场景专用动作或成功回执补齐。

每个预演证书与真实执行回执分别记录动态安全证据。证据来自对应的实际物理帧，包括支撑凸包余量、足底切向滑移、关节极限余量、关节速度、峰值接触力、法向力上升率，以及执行器请求与实际力矩、利用率和饱和状态；缺失的观测保持缺失，不会合成默认数值。恢复后的证据只覆盖真正执行过的轨迹前缀。

默认生成器是确定性的任务约束求解器，自主性来自上层模型选择的目标、Skill 和策略，不来自固定动作表。`humanoid-motion-generator-v1` 协议允许接入学习式生成器；生成器类型、采样方式和实现身份会进入实时世界状态与检查点，恢复时必须与原运行一致，不会静默切换后端。

### 运动策略与训练接入

层级智能体不直接输出电机值或关节动作。它负责语义目标、Skill DAG、交互对象和终止条件；低层运动由 [`HumanoidWholeBodyController`](src/world/humanoid/whole-body-controller.ts) 执行。`HumanoidWorld.create` 的 `controllerFactory` 会分别为权威 MuJoCo 世界和独立预演池创建控制器实例，场景资源重建时继续使用同一工厂。控制器状态必须支持捕获与恢复，策略的观察空间、动作空间和真实能力随世界快照公开。

不设置 `HEAR_HUMANOID_CONTROLLER_MODULE` 时，正式 CLI 和 Operator 默认加载仓库随附的 mjlab G1 学习策略，并在缺少关节参考跟踪能力的控制步切换到 YAHMP。设置该变量可以用自己的训练产物覆盖默认主策略；值可以是相对路径、绝对路径或已安装包名。模块必须导出 `createHumanoidWholeBodyController(context)`，并在每次调用时创建独立的 [`HumanoidWholeBodyController`](src/world/humanoid/whole-body-controller.ts) 实例。模块可以通过 `humanoidControllerAssets` 声明 ONNX、训练报告等本地策略文件，运行来源身份会同时覆盖入口与全部资产内容。运行定义不保存本机路径；恢复时入口或任一策略资产发生变化都会在创建世界前被拒绝。未记录来源身份的历史运行仍使用创建时的 YAHMP，不会被新的默认策略静默迁移。

YAHMP 参考控制器声明 `balance`、`locomotion` 和 `joint_reference_tracking`。任务空间 IK、接触柔顺和抓取检查器是参考生成与物理验收组件，不代表策略已经学会接触式操作或双手操作。后续可以接入强化学习、模仿学习或其他已训练策略；新控制器只有在真实支持时才应声明 `contact_rich_manipulation` 或 `bimanual_manipulation`，Harness 仍会用同一 MuJoCo 预演和执行回执验证结果。只运行参考控制器时可设置 `HEAR_HUMANOID_CONTROLLER_MODULE=hear/controllers/yahmp`。

外接学习策略缺少平衡、移动或关节参考跟踪中的任一基础能力时，模拟器会把它作为主控制器，并创建独立 YAHMP 参考控制器组成能力路由。站立、普通导航、持物导航和全身运动的每个控制步都会声明实际能力需求与任务目标；路由从当前 MuJoCo 状态选择覆盖该阶段的控制器。分支切换不会直接跳变电机目标，而是按控制器声明的响应周期连续插值关节目标、刚度和阻尼；正在进行的交接与两个控制器的内部状态共同保存到物理检查点。主策略的 `learnedPolicy.capabilities` 保持原值，参考控制不会被合并或冒充为训练能力。已经完整实现能力路由的外接控制器可以通过 `capabilityRouting` 描述自身边界，运行时不会再次包装。

控制器协议同时提供可声明的策略观察特征与逐控制步任务命令。训练策略可按自身编码消费根运动、手部状态、末端状态、MuJoCo 接触、对象与关节状态，以及当前任务空间目标和抓取约束；YAHMP 仍只读取原有本体状态与命令历史。语义 Skill 不会被展开成模型供应商或训练框架专用格式，因此本地 ONNX、远程策略服务和后续训练产物可以共用同一控制器边界。

统一 Skill 合约逐阶段声明完成学习式执行所需的策略能力，实时目录会分别公开当前环境可用性与已训练能力覆盖。未被当前策略覆盖的阶段会明确标记为参考控制回退，不能被描述为已经训练完成。控制器收到的任务命令同时包含能力要求、任务空间目标、抓取约束和可观测物理终止谓词，训练侧不需要读取 Agent 提示词或依赖模型供应商格式。

仓库随附一个由 mjlab 1.5.3 训练的 G1 速度策略及其训练报告，并将它作为新运行的默认主策略。标准控制器严格校验报告版本、ONNX SHA-256、张量形状、29 关节顺序、控制周期和策略元数据，使用 MuJoCo 骨盆 IMU 坐标下的 99 维真实观察推理。它只声明并输出训练得到的 `balance` 与 `locomotion` 动作，不在策略输出中混入未训练的任务关节控制；这类任务由上述独立能力路由执行。

仓库同时提供基于 [mjlab](https://github.com/mujocolab/mjlab) 与 RSL-RL 的 G1 速度策略训练入口。训练直接使用 mjlab 的正式 `Mjlab-Velocity-Flat-Unitree-G1` 环境和 MuJoCo Warp，不在仓库内另写一套强化学习算法。已登录 Colab CLI 后可启动 GPU 训练：

```sh
pnpm train:g1:colab -- --gpu H100 --iterations 1000 --num-envs 4096
```

命令会创建独立 Colab 会话，训练真实 PPO checkpoint，由 mjlab 导出带控制元数据的 ONNX，并在 GPU MuJoCo 环境中执行无界面策略评估。checkpoint、ONNX、环境配置、评估指标和 SHA-256 报告下载到 `artifacts/training/`，同时解包为可直接运行的策略目录。将 `HEAR_MJLAB_G1_POLICY_DIRECTORY` 指向该目录即可替换随附策略；训练、下载、解包或校验失败都会直接返回错误，不生成替代策略。结束或失败后 Colab 会话都会释放。

程序化场景生成开阔区域、方块障碍、可动物体和目标区域。头部相机以真实水平与垂直视场持续更新 0.5 米空间信念网格；可见物理几何会截断其后的地面射线，墙后区域保持未知，移动或拆除实体留下的占据只有在重新进入视野后才会清除。模型从这种未知区域边界选择探索目标。Recast 根据当前静态与动态几何生成导航路径，每段路线都先在当前物理状态副本中完整执行；执行中出现新的几何阻塞时，执行监控层保持原 Skill 目标并从真实终态重新规划，最多进行两次有界尝试。跌倒、物体滑脱或语义前提失效不会被低层重规划掩盖，而是返回模型选择恢复 Skill。

可动物体的抓取不是吸附或坐标绑定。系统从当前掌指接触面、接触力、对向接触、离开支撑面的高度、手物相对位姿稳定性和连续抬升帧建立抓取证据；只有通过证据的手物关系才能进入携带状态。持物导航逐帧验证抓取延续和未授权碰撞，放置动作必须由模型产生张手与撤手运动，并同时满足物体进入目标区域、手部脱离和非人形支撑面稳定承托。

## 长期运行

每个层级节点拥有独立的 Agents SDK Session。稳定指令和历史位于请求前缀，实时世界权限位于末尾；缓存亲和键按凭证、协议、模型和 Agent 角色保持稳定，因此完全一致的公共前缀可以跨任务复用。亲和键只影响供应商缓存路由，不承载对话内容；不同 Run 的 Session 和物理状态始终隔离。端点明确拒绝缓存扩展参数时，该节点的 Model facade 只协商一次并继续使用协议自身的自动前缀缓存。界面和日志中的缓存读取量直接来自模型服务返回的 usage，不使用本地估算。活动上下文接近配置阈值时，系统调用模型生成结构化压缩记录，并保留近期原始轮次。压缩结果必须引用真实动作回执；完整事件、模型生命周期、动作、具身经历、检查器和上下文记录继续保存在追加式日志中。Goal DAG 把同一次模型调用产生的候选视为一个决策批次：显式选择一个候选时，其余候选以未采用结果绑定同一选择证据。检查点只保留当前工作集、仍被依赖的已完成候选和近期 epoch；更早的完整决策批次会先写入哈希链式追加日志，再从检查点裁剪，进程中断后可幂等恢复。目标管理智能体可按状态、谓词、对象、方块、语义区域、世界空间范围或候选标识分页召回归档 Goal。模型生成的经历摘要不会进入权威状态块。协调与运动智能体可以按 `episode:N` 或 `action:<transactionId>` 精确召回成功、拒绝、漂移、约束违规和停滞记录；所有召回结果均标记为历史信息，不能代替当前传感。

上下文压缩本身由独立模型完成。无效输出可以在同一压缩回合内重新生成；网络中断会立即交还原业务 Agent 的标准传输恢复流程，原始历史和 Session 不会被替代摘要覆盖。只有通过 schema、来源引用和当前世界权限校验的压缩记录才会成为新基线；基线提交后的业务请求若中断，恢复只保留该基线和真实热历史，不会重新灌入已经裁剪的旧前缀。配置窗口不足属于明确的容量错误，不会无限重试。

运行时按权威世界版本、真实物理帧和动作回执检测长期无进展循环。守卫只中断并重建停滞的模型上下文，不生成默认动作，也不替模型选择行为。目标稳定进度随检查点持久化；恢复时只接受与 Goal 哈希、世界快照和 MuJoCo 检查点一致的证据。

模型传输中断时，可序列化的 Agents SDK RunState 只在 Goal、动作账本、上下文压缩和自主循环身份仍兼容时继续使用。每份 RunState 同时绑定五个独立 Session 的精确历史前缀；恢复会先核验全部前缀，再统一移除断线后未形成新状态的会话后缀。任一前缀分歧都会拒绝该 RunState，已经提交的物理动作、Goal 证据和追加式日志不会回滚。OpenAI-compatible 传输还会在请求边界清理进程中断留下的半边工具协议片段，完整工具调用与结果保持原顺序，动作事实继续由当前 Harness 权威块提供。

运行检查点包含：

- 层级节点、活动智能体、模型调用计数和 provider 返回的逐智能体 token 用量
- MuJoCo 状态、控制器历史和当前全身参考
- 世界版本、导航计划、物体记忆与任务检查结果
- 具名末端目标的逐帧稳定进度与 Goal 身份校验
- 已提交动作回执、候选筛选证据和待处理生命周期事件
- 不可变运动制品、物理预演轨迹、Option 监控状态与执行游标
- 每个智能体的独立 Session 与可恢复 SDK 状态

Operator 异常退出后，未完成任务会转为可恢复状态。恢复操作从持久化物理状态和上下文继续，不播放录制动画。尚未完成的物理动作使用原 transaction ID 和原规划制品续接，完成后才恢复上层模型循环；正常暂停会把不足周期的执行尾帧一并写入账本。旧检查点若已保存更靠后的精确 MuJoCo 状态，只在规划进度与世界版本完全一致时从该状态继续，无法重建的中间轨迹明确标记为不完整。有限任务的最终 Goal 验收、Run 成功状态和生命周期事件在同一检查点事务中提交，恢复后不会继续创建多余 Goal。

## Web 界面

![层级智能体执行流](docs/screenshots/hierarchy.png)

3D 世界始终保持在主视图。界面提供跟随、世界和头部三个观察视角，并实时显示：

- 当前活动智能体与五节点层级
- 身体通道活动状态
- 世界版本、物理时间、双脚法向力、支撑和直立度
- 当前实际执行的学习或参考控制器，以及连续交接进度
- 任务谓词与上下文占用
- 具名末端目标的实时稳定帧进度
- 规划、执行和拒绝回执
- 经过整理的模型活动与输出

| 行动历程 | 智能体输出 |
|---|---|
| ![行动历程](docs/screenshots/actions.png) | ![智能体输出](docs/screenshots/logs.png) |

![移动端界面](docs/screenshots/mobile.png)

## 环境要求

| 组件 | 要求 |
|---|---|
| 操作系统 | Linux、Windows 或 macOS |
| Node.js | 22.13.0 或更高版本 |
| 包管理器 | pnpm 11（建议通过 Corepack 启用） |
| 浏览器 | 支持 WebGPU 或 WebGL2 的现代浏览器 |
| 模型服务 | 支持流式响应和工具调用的 API |

默认 MuJoCo 和 ONNX 后端可以在 CPU 上运行，不要求 GPU。实际速度取决于处理器、所选模型服务和场景规模。

## 安装

```sh
git clone https://github.com/multi-zhangyang/hear-3d-robot.git
cd hear-3d-robot
corepack enable
pnpm install --frozen-lockfile
```

Linux 与 macOS：

```sh
cp .env.example .env
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

## 配置

HEAR 不绑定特定模型或服务商。传输协议、地址、模型和凭证均由环境变量提供。

```dotenv
AI_PROVIDER=openai_compatible
AI_BASE_URL=https://api.example.com/v1
AI_MODEL=your-model-id
AI_API_KEY=your-api-key
AI_REQUEST_TIMEOUT_MS=300000
AI_STREAM_EVENT_IDLE_TIMEOUT_MS=300000

AI_TEMPERATURE=0.2
AI_REASONING_EFFORT=
AI_TOOL_CHOICE=auto
AI_MAX_OUTPUT_TOKENS=
AI_CONTEXT_WINDOW_TOKENS=262144
AI_COMPACT_TRIGGER_TOKENS=
AI_COMPACT_RECENT_MODEL_TURNS=4
AI_COMPACT_MAX_OUTPUT_TOKENS=

HEAR_HOST=127.0.0.1
HEAR_PORT=8765
HEAR_OPERATOR_PASSWORD=
HEAR_RUNS_DIR=./runs
HEAR_HUMANOID_CONTROLLER_MODULE=
HEAR_MJLAB_G1_POLICY_DIRECTORY=
```

可用传输协议：

| `AI_PROVIDER` | 协议 |
|---|---|
| `openai_compatible` | OpenAI-compatible Chat Completions |
| `openai_responses` | OpenAI Responses API |
| `anthropic_messages` | Anthropic Messages API |

`AI_CONTEXT_WINDOW_TOKENS` 应填写模型实际上下文上限，默认值为 `262144`。`AI_REASONING_EFFORT` 可按模型能力设置为 `none`、`minimal`、`low`、`medium`、`high`、`xhigh` 或 `max`，留空则不发送该参数。`AI_TOOL_CHOICE` 支持 `required` 和 `auto`，默认值为 `auto`；`none` 会在启动前被拒绝，因为所有业务节点和压缩节点都必须产生正式工具结果。当前阶段只有一个合法工具时，Harness 使用协议原生的命名工具约束；端点明确拒绝该能力时只协商一次并回到配置模式。业务节点没有产生正式工具结果时，会保留同一个 Agent、模型门面和 Session，在原会话继续恢复；多工具阶段的恢复通过标准 `required` 约束要求模型自行选择，端点不支持时自动回到原配置，保留模型思考，不替模型选择工具、参数或动作。恢复不使用固定次数终止，而由物理与 Goal 权威状态、上下文压缩生命周期和外部取消划分边界。`AI_MAX_OUTPUT_TOKENS` 与 `AI_COMPACT_MAX_OUTPUT_TOKENS` 默认留空，运行时不会向模型请求发送输出上限；通常不建议设置，只有端点明确要求限制时才填写。压缩阈值留空时固定取各智能体实际上下文窗口的 `85%`；角色覆盖自己的窗口后也会独立重算，不继承其他模型的低阈值。

`AI_REQUEST_TIMEOUT_MS` 默认是 `300000`，表示 HTTP 建连或相邻响应数据之间允许的最长静默时间。`AI_STREAM_EVENT_IDLE_TIMEOUT_MS` 默认同为 `300000`，约束相邻 Agents SDK 模型事件之间的静默时间；只有真实模型事件会续期。两者均可按端点能力在 5 秒至 10 分钟之间调整，任务总时限、人工停止和进程恢复仍独立生效。

目标管理、协调、感知、运动、执行和压缩可以使用彼此独立的模型配置。未设置的角色变量继承同名 `AI_*` 默认值；设置时使用 `AI_<ROLE>_<SETTING>`：

| `ROLE` | 运行职责 |
|---|---|
| `GOAL_MANAGER` | 自主 Goal 候选与选择 |
| `COORDINATOR` | 自主循环协调 |
| `SENTRY` | 实时感知 |
| `MOTION` | 全身运动规划 |
| `EXECUTOR` | 物理执行 |
| `COMPACTOR` | 长期上下文压缩 |

`SETTING` 支持 `PROVIDER`、`BASE_URL`、`MODEL`、`API_KEY`、`REQUEST_TIMEOUT_MS`、`STREAM_EVENT_IDLE_TIMEOUT_MS`、`TEMPERATURE`、`REASONING_EFFORT`、`TOOL_CHOICE`、`MAX_OUTPUT_TOKENS`、`CONTEXT_WINDOW_TOKENS`、`COMPACT_TRIGGER_TOKENS`、`COMPACT_RECENT_MODEL_TURNS` 和 `COMPACT_MAX_OUTPUT_TOKENS`。例如 `AI_MOTION_MODEL` 只覆盖运动节点，`AI_COMPACTOR_CONTEXT_WINDOW_TOKENS` 只描述压缩模型的真实上下文上限。配置仍基于协议能力，不绑定服务商或模型名称。

五个业务层级节点各自持有独立 Model facade 与持久 Session；压缩器使用独立模型配置和无历史污染的有界 SDK 回合。每个 Run 会写入不含凭证和端点明文的 Agent 身份清单。恢复时会核验模型、协议、端点身份哈希、模型参数、指令、工具 Schema 和 Agents SDK 版本；不兼容配置会被明确拒绝，不会静默复用旧 Session。

## 启动

开发模式：

```sh
pnpm dev
```

打开 <http://127.0.0.1:8765>。

生产模式：

```sh
pnpm build
pnpm start
```

命令行使用同一套人形运行时：

```text
pnpm hear scenarios
pnpm hear run --scenario ID --mission TEXT --goal JSON [--mode mission|continuous] [--seed N] --confirm
pnpm hear resume --run RUN_ID --confirm
pnpm hear operator [--host HOST] [--port PORT] [--dev]
```

`mission` 在模型选择并经物理验收完成与任务约束完全一致的 Goal 后结束；`continuous` 在每个 Goal 完成后继续自主选择下一目标，直到操作者暂停。命令行默认使用 `mission`，Web Operator 默认使用 `continuous`。

## 场景

| 场景 | 规模 | 内容 |
|---|---:|---|
| `humanoid_courtyard` | 18 × 16 | 固定庭院、立柱、低台和目标信标 |
| `humanoid_workyard` | 28 × 22 | 可抓取装配件、承托台、持物通道和稳放区域 |
| `humanoid_cabinet` | 20 × 18 | 带铰链柜门、遮挡工件和目标容器的连续操作空间 |
| `humanoid_frontier` | 36 × 36 | 随机障碍与可动物体组成的方块边境 |
| `humanoid_realm` | 54 × 54 | 更大的程序化方块疆域与多个区域 |

程序化场景默认生成新的世界种子，也可以在命令行显式指定，以便继续同一个物理世界或复查某次运行。

## 数据目录

每个任务保存在 `${HEAR_RUNS_DIR}/<run-id>/`，默认位置为 `runs/<run-id>/`。

| 路径 | 内容 |
|---|---|
| `run.json` | 任务、场景、目标与可选控制器来源身份 |
| `checkpoint.json` | 层级、世界、物理和记忆检查点 |
| `agent-state.json` | 可恢复的 Agents SDK 运行状态 |
| `session.json` | 协调智能体 Session |
| `sessions/` | 专职智能体独立 Session |
| `agent-manifest.json` | Agent 配置、指令、工具与 SDK 身份清单；不含 API 凭证 |
| `actions.jsonl` | 规划和执行回执 |
| `episodes.jsonl` | 可按来源标识召回的长期具身经历 |
| `events.jsonl` | 实时与恢复事件 |
| `provider.jsonl` | 模型调用生命周期 |
| `framework.jsonl` | Agents SDK 流事件 |
| `context.jsonl` | 上下文压缩记录 |
| `goal_evidence.jsonl` | Goal 物理证据 |
| `goal_history.jsonl` | 已归档的 Goal 决策批次与 epoch 哈希链；保留选中及未采用候选，可按状态、实体、语义区域或世界空间范围召回 |
| `checker.jsonl` | 任务谓词检查结果 |

## 开发与验证

```sh
pnpm typecheck
pnpm test
pnpm check
pnpm test:browser
pnpm test:live:mission
pnpm test:live:continuous
pnpm test:live:manipulation
pnpm test:live:endurance
```

`pnpm check` 包含无用代码扫描、服务端和前端类型检查、单元与集成测试、生产构建及生产启动检查。浏览器测试在桌面与移动视口中验证 G1 网格加载、实时界面、三个相机视角、延迟加载面板和真实 WebGL/WebGPU 画布。

真实模型检查使用正常环境配置，不包含在离线 CI 中。有限任务检查验证模型决策、物理执行、Goal 状态和具身记忆的完整因果链；持续运行检查直接启动正式 `continuous` 内核，在配置的观察时间内不注入动作、不设置 Cycle 或 Goal 配额，也不把运行次数当作自主性结论；配置具备接触操作能力的控制器后，物体交互检查要求真实抓取、持物导航、主动释放和稳放全部成立；耐久检查通过真实进程中断与恢复确认同一运行可以继续。

## 项目结构

```text
src/harness/humanoid/  层级智能体、工具与动作回执
src/world/humanoid/    G1、MuJoCo、YAHMP、任务空间运动与物体记忆
src/runtime/           任务生命周期、上下文压缩与目标检查
src/model/             模型协议与 Agents SDK 模型适配
src/persistence/       检查点、追加式日志与 Session
src/server/            Operator API、SSE 与运行管理
training/              mjlab G1 GPU 训练、ONNX 导出与物理评估
web/src/humanoid/      G1 场景、相机和人形实时工作区
web/src/flow/          层级流、行动历程与输出视图
tests/browser/         桌面端和移动端浏览器验收
```

## 当前边界

HEAR 当前面向虚拟人形机器人，不直接控制实体硬件。默认学习策略覆盖行走、转向、平衡和关节参考跟踪；系统可以生成并验证躯干、手腕和脚踝末端参考，但默认策略没有声明接触式操作或双手操作已经训练完成。复杂地形跑跳、精细抓取、负载行走和高动态全身交互需要接入相应训练策略。所有成功状态均来自当前仿真世界和检查器，不以模型文字作为物理完成证明。

## 安全

- `.env`、运行数据、构建产物和浏览器测试产物不会提交到 Git。
- API 凭证只在服务端读取，不会进入浏览器启动数据。
- Operator 默认监听 `127.0.0.1`；绑定非回环地址时必须设置操作密码，并建议同时使用反向代理访问控制。
- 任务日志包含模型输出和世界状态，应按部署环境的数据策略管理。

## 许可证

本项目采用 [MIT License](LICENSE)。
