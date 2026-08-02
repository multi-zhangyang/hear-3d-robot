# HEAR

[![CI](https://github.com/multi-zhangyang/hear-3d-robot/actions/workflows/ci.yml/badge.svg)](https://github.com/multi-zhangyang/hear-3d-robot/actions/workflows/ci.yml)

HEAR 是一个运行在虚拟 3D 体素世界中的模型驱动具身智能体系统。项目将层级智能体编排、物理仿真、导航、操作、持久记忆与实时 Web 界面整合在同一套运行时中。

![HEAR 虚拟机器人世界](docs/screenshots/mission.png)

## 主要特性

- 基于 OpenAI Agents SDK 的层级智能体编排
- 每个具体智能体拥有独立的模型上下文与持久会话
- 通过身体通道租约并行控制移动底盘、传感头、机械臂与夹爪
- 使用 Three.js WebGPURenderer，优先启用 WebGPU 并自动兼容 WebGL2
- 使用 Rapier 进行关节机器人仿真与碰撞处理
- 使用 Recast 动态生成导航网格并规划路径
- 支持位置与姿态逆运动学，以及经过碰撞检查的机械臂轨迹
- 支持分块加载的程序化体素地形、物品栏、方块破坏与放置
- 支持持久检查点、上下文压缩和带动作回执的空间记忆
- 使用类型化证据契约校验动作、结果、目标、时效与实时任务谓词
- 探索移动由模型从实时可达边界中选择，Harness 原子完成来源校验、Recast 规划与 Rapier 执行
- 根据当前世界状态校验结构化任务目标
- 支持第一人称观察、世界点选，并实时展示智能体层级、行动历程与模型输出

## 系统架构

```text
浏览器
  └─ 操作界面 · Three.js WebGPU/WebGL2 · SSE
       └─ Fastify 运行时
            ├─ OpenAI Agents SDK
            │    ├─ 任务协调智能体
            │    └─ 专项执行智能体
            ├─ 具身 Harness
            │    ├─ 能力边界
            │    ├─ 身体通道租约
            │    ├─ 计划、动作回执与证据契约
            │    └─ 任务检查器
            ├─ 虚拟世界
            │    ├─ Rapier 物理仿真
            │    ├─ Recast 导航
            │    └─ 体素分块
            └─ 持久化
                 ├─ 会话与运行状态
                 ├─ 检查点与事件日志
                 └─ 上下文与空间记忆
```

任务协调智能体负责完整任务生命周期，并通过 Agents SDK 的 `agent.asTool()` 模式调用边界明确的专项智能体。每个层级节点都有独立的 Agent 实例、模型适配器、Session、上下文预算和压缩作用域。

Harness 向智能体提供类型化的感知、规划和执行工具。只有当智能体拥有所需能力、对应身体通道可用，并且计划与当前世界版本一致时，物理指令才会被接受。专项智能体提交完成结果时，Harness 会核对真实回执的动作名、结果码、效果类型、目标和世界版本；最终任务谓词始终由检查器直接读取当前仿真状态。任务描述和模型文本只用于表达，不具备完成权威。

自主探索采用“勘察—模型选择—原子导航”协议。同一移动智能体在没有当前勘察时只能调用 `survey_terrain`；获得有效候选后只能调用 `navigate_frontier`，直到身体移动使勘察版本失效。`survey_terrain` 返回当前世界版本下由 Recast 验证可达的边界候选，模型明确选择其中一个 `choice_id` 后调用 `navigate_frontier`。阶段门只隐藏必然无效的状态转换，既不排序也不选择候选。Harness 只执行模型选中的候选，并验证勘察回执归属、候选身份和世界版本；它不会换用其他候选、自动重试或生成替代动作。一般目标导航仍使用独立的路线规划与执行回执。

模型生成的工具参数如果不是合法 JSON 或不符合声明的 schema，会作为带字段路径的拒绝结果返回 Agents SDK，由原模型重新生成完整调用。无效输入不会创建层级节点或触发 Harness 动作，运行时也不会猜测、补齐或修复动作参数。

## 机器人与世界

机器人包含四组可独立租用的控制通道：

| 通道 | 部件 |
|---|---|
| `base` | 移动底盘与轮式里程计 |
| `head` | 传感头偏航与俯仰 |
| `arm` | 肩、肘与腕关节 |
| `gripper` | 夹爪开合、接触与载荷连接 |

互不冲突的身体通道可以在同一组仿真帧中并行执行；通道已被占用时，新指令会收到明确的忙碌回执，由智能体重新规划或稍后重试。关节空间机械臂计划可以与底盘运动并行；绑定固定世界坐标末端目标的计划则要求底盘保持静止。

程序化世界覆盖 80×80 至 192×192 的体素区域。机器人附近的地形以 16×16 分块加载物理碰撞体和导航数据。方块变化会同步更新体素存储、物品栏、碰撞体、导航网格、检查点与前端场景。

世界交互使用真实场景射线检测。体素、物体、障碍、区域和机器人都可以点选，选中状态会随权威世界帧更新。第一人称视角固定在机器人传感头上；桌面端可使用 Pointer Lock 环视，移动端保留触控观察。相机交互不会绕过 Harness 控制机器人。

## 长期运行与记忆

HEAR 将短期上下文与持久状态分开管理：

- Agents SDK Session 只保存每个智能体压缩后的热上下文分支。
- 序列化运行状态用于恢复中断的智能体循环。
- 上下文压缩将较早的模型轮次整理为结构化检查点，并原样保留近期轮次；完整原始上下文仍保存在追加式日志中。
- 固定宽度偏移索引为追加式日志提供有界内存的尾部读取、分页与恢复。
- 空间记忆只索引已接受的动作回执与权威世界快照，并保留版本和来源信息。
- 物理检查点保存机器人、世界、活动指令、计划、体素变化和已探索区域。
- 临时网络、限流与网关故障会经过有界退避重试；持续不可用时任务进入可恢复的中断状态，不会丢弃已经持久化的模型分支和物理进度。

模型上下文窗口和压缩阈值均通过环境变量配置。

## 操作界面

3D 世界始终保持可见，底部工作区可以切换四种视图：

- **世界**：机器人状态、身体通道、任务目标、路径、运动遥测，以及跟随、第一人称和全局视角下的场景点选
- **智能体流**：层级结构、活动智能体、模型活动与动作回执
- **行动历程**：可筛选的感知、规划、执行和检查器事件
- **输出**：经过排版的智能体输出与任务终态

| 智能体流 | 行动历程 |
|---|---|
| ![智能体层级](docs/screenshots/hierarchy.png) | ![机器人行动历程](docs/screenshots/actions.png) |

| 输出 | 移动端 |
|---|---|
| ![智能体输出](docs/screenshots/logs.png) | ![移动端界面](docs/screenshots/mobile.png) |

## 环境要求

| 组件 | 要求 |
|---|---|
| 操作系统 | Linux、macOS 或 Windows |
| Node.js | 22.13.0 或更高版本 |
| 包管理器 | 通过 Corepack 使用 pnpm 11 |
| 浏览器 | 支持 WebGPU 或 WebGL2 的现代浏览器 |
| 模型 API | 支持流式响应与工具调用 |

GitHub Actions 会在 Linux、macOS 和 Windows 上执行构建与测试。

## 安装

```text
git clone https://github.com/multi-zhangyang/hear-3d-robot.git
cd hear-3d-robot
corepack enable
pnpm install --frozen-lockfile
```

创建本地环境配置文件。

Linux 与 macOS：

```sh
cp .env.example .env
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

## 配置

传输协议、服务地址、模型和凭证全部通过环境变量选择。

```dotenv
AI_PROVIDER=openai_compatible
AI_BASE_URL=https://api.example.com/v1
AI_MODEL=your-model-id
AI_API_KEY=your-api-key

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

支持的传输协议：

| `AI_PROVIDER` | 传输方式 |
|---|---|
| `openai_compatible` | OpenAI-compatible Chat Completions |
| `openai_responses` | OpenAI Responses API |
| `anthropic_messages` | Anthropic Messages API |

`AI_CONTEXT_WINDOW_TOKENS` 应与所用模型的可用上下文窗口一致。创建任务前，HEAR 会检查压缩阈值和输出预留空间是否能安全容纳在上下文窗口内。

## 运行

生产构建：

```text
pnpm build
pnpm start
```

开发模式需要同时启动运行时和 Web 开发服务器。

终端一：

```text
pnpm dev
```

终端二：

```text
pnpm dev:web
```

打开 <http://127.0.0.1:5173>。生产构建启动后则访问 <http://127.0.0.1:8765>，即可从 Web 界面创建任务。

命令行使用同一套运行时：

```text
pnpm hear scenarios
pnpm hear run --scenario ID --mission TEXT --goal JSON [--seed N] --confirm
pnpm hear resume --run RUN_ID [--fresh-context] --confirm
pnpm hear operator [--host HOST] [--port PORT] [--dev]
```

## 场景

| 场景 | 类型 | 内容 |
|---|---|---|
| `voxel_expanse` | 程序化 | 80×80 体素环境 |
| `voxel_highlands` | 程序化 | 96×96 高差地形 |
| `voxel_survey` | 程序化 | 80×80 探索与操作世界 |
| `voxel_realm` | 程序化 | 192×192 分块加载世界 |
| `open_navigation` | 固定场景 | 导航任务 |
| `fetch_red_block` | 固定场景 | 导航、抓取、搬运与释放 |
| `locked_container` | 固定场景 | 带钥匙约束的多步骤容器交互 |

程序化场景默认使用随机世界种子，也可以显式指定。每个新任务都会独立生成运动种子，即使复用了同一个世界种子也不例外。恢复任务时会同时恢复两个种子和物理检查点。

## 持久化

每个任务保存在 `${HEAR_RUNS_DIR}/<run-id>/`；未配置时使用 `runs/<run-id>/`：

| 文件 | 内容 |
|---|---|
| `run.json` | 任务定义与实例化场景 |
| `checkpoint.json` | 当前层级、世界、记忆和任务状态 |
| `agent-state.json` | 序列化的 Agents SDK 运行状态 |
| `session.json` | 任务协调智能体的 Session 历史 |
| `sessions/` | 各具体智能体按节点独立保存的 Session 历史 |
| `actions.jsonl` | 工具调用与物理动作回执 |
| `context.jsonl` | 上下文历史与压缩检查点 |
| `provider.jsonl` | 模型传输生命周期与用量 |
| `framework.jsonl` | Agents SDK 流事件 |
| `hierarchy.jsonl` | 智能体层级变化 |
| `checker.jsonl` | 结构化任务检查结果 |
| `events.jsonl` | Web 界面和 SSE 使用的实时运行事件 |
| `*.offsets` | 对应追加式日志的分页与尾部读取索引 |

同一运行目录由单个 Operator 进程持有租约。异常退出留下的同机租约会在确认原进程结束后回收，第二个仍存活的 Operator 无法把正在执行的任务误判为孤儿。

## 开发

```text
pnpm typecheck
pnpm test
pnpm build
pnpm check
pnpm test:browser
pnpm audit --prod
```

`pnpm check` 会执行无用代码扫描、服务端和前端类型检查、单元与集成测试以及生产构建。`pnpm test:browser` 会构建 Web 应用，并运行桌面端和移动端 Playwright 测试。

## 项目结构

```text
src/model/        模型传输与适配器
src/harness/      Agents SDK 层级、能力、租约与记忆
src/runtime/      工具、任务生命周期、恢复与检查器
src/world/        机器人、物理、导航、地形与体素系统
src/server/       Fastify API、SSE 与任务管理
src/persistence/  检查点、日志与 Session
web/src/flow/     智能体流、行动历程与输出展示
web/src/game/     世界外壳与工作区覆盖层
web/src/stage/    Three.js 场景、相机、HUD 与路径覆盖层
tests/browser/    桌面端与移动端浏览器测试
```

## 安全

- `.env`、任务运行数据、构建产物和浏览器测试产物不会提交到 Git。
- API 凭证只保留在服务端，不会进入浏览器启动数据或任务日志。
- 操作服务默认只监听 `127.0.0.1`。
- 对外开放操作服务前应设置 `HEAR_OPERATOR_PASSWORD`。
- 任务日志可能包含提示词、模型输出和世界状态，应按照部署环境的数据策略妥善管理。

## 许可证

本项目采用 [MIT License](LICENSE)。
