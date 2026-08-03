# HEAR

[![CI](https://github.com/multi-zhangyang/hear-3d-robot/actions/workflows/ci.yml/badge.svg)](https://github.com/multi-zhangyang/hear-3d-robot/actions/workflows/ci.yml)

HEAR 是一个由层级智能体自主驱动的虚拟 3D 人形机器人运行时。模型负责观察、选择目标和提出任务空间动作；Harness 负责权限、计划来源、物理预演、执行回执和目标验收；Unitree G1 模型在 MuJoCo 中通过全身控制器完成实际运动。

![HEAR 人形机器人世界](docs/screenshots/mission.png)

## 核心能力

- OpenAI Agents SDK 层级编排，每个智能体拥有独立 Model 实例、Session 和上下文生命周期
- G1 29 自由度人形模型、双足接触、质心、支撑面和跌倒检测
- 模型按偏好提交多个根运动、躯干朝向、双手腕和双脚踝末端目标候选，不直接编写关节角
- 每个全身候选从同一 MuJoCo 状态独立预演，只选择排序最前且物理可行的模型候选
- 全身 Option 使用受限物理条件树和前置、持续、终止三个阶段，成功必须经过连续帧稳定验收
- 预演轨迹与运动制品分别进行 SHA-256 校验；真实执行持续偏离预演时立即截断并重新规划
- YAHMP ONNX 全身策略在 50 Hz 生成控制目标，MuJoCo 在 200 Hz 处理重力、力矩、碰撞与接触
- Recast 导航网格和分段物理预演，不通过修改根节点坐标伪造移动
- 程序化方块世界，世界种子、障碍、物体和目标区域按任务生成
- 头部视场感知与角色化 3D 对象状态，当前 MuJoCo 精确状态和历史观察严格分离
- 追加式事件日志、可寻址具身历史、物理检查点、独立会话和结构化上下文压缩
- 中文实时界面，展示 3D 世界、层级执行流、动作回执、模型活动与物理状态
- Three.js 优先使用 WebGPU，并兼容 WebGL2
- Linux、Windows 和 macOS 持续集成

## 运行链路

```text
任务目标
  └─ 人形自主协调智能体
       ├─ 人形感知哨兵
       ├─ 全身运动参考智能体
       └─ 人形物理执行智能体
              │
              ▼
       连续运动意图
              │
       模型排序的全身候选
              │
       运动生成器 / 独立物理预演
              │ 首个可行候选
       已选择的全身参考 + 物理 Option
              │
       YAHMP 神经运动跟踪
              │
       MuJoCo 实际执行 + 逐帧验收
              │ 达成 / 偏离 / 违反约束
       提前终止或交回重规划
              │
       权威世界帧与动作回执
              │
       目标检查 / 滚动重规划
```

协调、感知、运动参考和物理执行是四个边界明确的智能体，而不是一个模型扮演多个名称。协调智能体不能直接修改世界；运动参考智能体只能提交规划；执行智能体只能消费当前世界版本中已接受规划的原始回执。

一次动作需要同时满足以下条件才会改变世界：

1. 工具参数通过严格 schema 校验。
2. 规划回执属于正确智能体和当前世界版本。
3. 多个候选确实由当前运动参考智能体按偏好排序提交，而不是 Harness 生成的动作表。
4. 至少一个任务空间候选能够生成合法的连续全身参考。
5. 被选候选的完整 MuJoCo 预演没有跌倒、非法接触、持续条件违规或缺失的必需接触。
6. 运动制品、终止合约和逐帧预演轨迹与规划证书一致。
7. 真实执行逐帧满足物理 Option；目标稳定达成后立即停止，持续偏离预演或违反条件时立即交回重规划。

自主差异来自模型对实时观察、目标和历史回执的采样决策、模型生成并排序的不同全身候选，以及每次任务独立生成的世界。程序不会从预设动作表挑选行为，也不会用随机电机噪声代替自主决策。Harness 只在这些模型候选中应用物理硬约束，不会悄悄创造默认动作。

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

运动工具接收任务空间关键帧。根运动使用身体局部速度，双手腕与双脚踝目标可使用世界坐标或骨盆相对坐标；多末端 DLS 逆运动学求出连续腿臂关节参考，再交给 YAHMP 跟踪。一次非导航决策提交 2 至 3 个不同候选，每个候选都从当前物理状态完整预演，并按模型给出的偏好顺序选择首个可行候选。模型可以组合行走、转身、抬腿、跨步、躯干调整和双臂动作，而不依赖固定动作名称或动作表。

默认生成器是确定性的任务约束求解器，自主性来自上层模型产生的连续目标，不来自固定动作表。`humanoid-motion-generator-v1` 协议允许接入学习式生成器；生成器类型、采样方式和实现身份会进入实时世界状态与检查点，恢复时必须与原运行一致，不会静默切换后端。

程序化场景生成开阔区域、方块障碍、可动物体和目标区域。Recast 根据当前几何生成导航路径，每段路线都先在当前物理状态的副本中完整执行；失败的预演只返回证据，不会自动换成另一条路线或程序性位移。

## 长期运行

每个层级节点拥有独立的 Agents SDK Session。活动上下文接近配置阈值时，系统调用模型生成结构化压缩记录，并保留近期原始轮次。压缩结果必须引用真实动作回执；完整事件、模型生命周期、动作、具身经历、检查器和上下文记录继续保存在追加式日志中。协调与运动智能体可以按 `episode:N` 或 `action:<transactionId>` 精确召回成功、拒绝、漂移、约束违规和停滞记录；召回结果始终标记为历史信息，不能代替当前传感。

运行检查点包含：

- 层级节点、活动智能体和模型调用计数
- MuJoCo 状态、控制器历史和当前全身参考
- 世界版本、导航计划、物体记忆与任务检查结果
- 已提交动作回执、候选筛选证据和待处理生命周期事件
- 不可变运动制品、物理预演轨迹、Option 监控状态与执行游标
- 每个智能体的独立 Session 与可恢复 SDK 状态

Operator 异常退出后，未完成任务会转为可恢复状态。恢复操作从持久化物理状态和上下文继续，不播放录制动画。

## Web 界面

![层级智能体执行流](docs/screenshots/hierarchy.png)

3D 世界始终保持在主视图。界面提供跟随、世界和头部三个观察视角，并实时显示：

- 当前活动智能体与四节点执行链
- 身体通道活动状态
- 世界版本、物理时间、双脚法向力、支撑和直立度
- 任务谓词与上下文占用
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
AI_REQUEST_TIMEOUT_MS=90000

AI_TEMPERATURE=0.2
AI_MAX_OUTPUT_TOKENS=8192
AI_CONTEXT_WINDOW_TOKENS=65536
AI_COMPACT_TRIGGER_TOKENS=18000
AI_COMPACT_RECENT_MODEL_TURNS=4
AI_COMPACT_MAX_OUTPUT_TOKENS=4096

HEAR_HOST=127.0.0.1
HEAR_PORT=8765
HEAR_OPERATOR_PASSWORD=
HEAR_RUNS_DIR=./runs
```

可用传输协议：

| `AI_PROVIDER` | 协议 |
|---|---|
| `openai_compatible` | OpenAI-compatible Chat Completions |
| `openai_responses` | OpenAI Responses API |
| `anthropic_messages` | Anthropic Messages API |

`AI_CONTEXT_WINDOW_TOKENS` 应填写模型实际上下文上限。压缩阈值必须为输出和工具回执预留足够空间。

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
pnpm hear run --scenario ID --mission TEXT --goal JSON [--seed N] --confirm
pnpm hear resume --run RUN_ID --confirm
pnpm hear operator [--host HOST] [--port PORT] [--dev]
```

## 场景

| 场景 | 规模 | 内容 |
|---|---:|---|
| `humanoid_courtyard` | 18 × 16 | 固定庭院、立柱、低台和目标信标 |
| `humanoid_frontier` | 36 × 36 | 随机障碍与可动物体组成的方块边境 |
| `humanoid_realm` | 54 × 54 | 更大的程序化方块疆域与多个区域 |

程序化场景默认生成新的世界种子，也可以在命令行显式指定，以便继续同一个物理世界或复查某次运行。

## 数据目录

每个任务保存在 `${HEAR_RUNS_DIR}/<run-id>/`，默认位置为 `runs/<run-id>/`。

| 路径 | 内容 |
|---|---|
| `run.json` | 任务、场景和目标定义 |
| `checkpoint.json` | 层级、世界、物理和记忆检查点 |
| `agent-state.json` | 可恢复的 Agents SDK 运行状态 |
| `session.json` | 协调智能体 Session |
| `sessions/` | 专职智能体独立 Session |
| `actions.jsonl` | 规划和执行回执 |
| `episodes.jsonl` | 可按来源标识召回的长期具身经历 |
| `events.jsonl` | 实时与恢复事件 |
| `provider.jsonl` | 模型调用生命周期 |
| `framework.jsonl` | Agents SDK 流事件 |
| `context.jsonl` | 上下文压缩记录 |
| `checker.jsonl` | 任务谓词检查结果 |

## 开发与验证

```sh
pnpm typecheck
pnpm test
pnpm check
pnpm test:browser
```

`pnpm check` 包含无用代码扫描、服务端和前端类型检查、单元与集成测试、生产构建及生产启动检查。浏览器测试在桌面与移动视口中验证 G1 网格加载、实时界面、三个相机视角、延迟加载面板和真实 WebGL/WebGPU 画布。

## 项目结构

```text
src/harness/humanoid/  层级智能体、工具与动作回执
src/world/humanoid/    G1、MuJoCo、YAHMP、任务空间运动与物体记忆
src/runtime/           任务生命周期、上下文压缩与目标检查
src/model/             模型协议与 Agents SDK 模型适配
src/persistence/       检查点、追加式日志与 Session
src/server/            Operator API、SSE 与运行管理
web/src/humanoid/      G1 场景、相机和人形实时工作区
web/src/flow/          层级流、行动历程与输出视图
tests/browser/         桌面端和移动端浏览器验收
```

## 当前边界

HEAR 当前面向虚拟人形机器人，不直接控制实体硬件。默认运动后端适合行走、转向、平衡、躯干和双手末端位置控制；复杂地形跑跳、精细手指抓取和高动态全身交互仍取决于更强的运动生成与跟踪策略。所有成功状态均来自当前仿真世界和检查器，不以模型文字作为物理完成证明。

## 安全

- `.env`、运行数据、构建产物和浏览器测试产物不会提交到 Git。
- API 凭证只在服务端读取，不会进入浏览器启动数据。
- Operator 默认监听 `127.0.0.1`；对外开放前应设置操作密码和反向代理访问控制。
- 任务日志包含模型输出和世界状态，应按部署环境的数据策略管理。

## 许可证

本项目采用 [MIT License](LICENSE)。
