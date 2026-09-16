# DataFlow-MM-Agent

[English](README.md) | **简体中文**

`dataflow-mm-agent` 让多模态 Agent 能够在多种视觉环境中交互，并将每次运行
返回为结构化的 `Trajectory`。它可以用于验证 Agent 与环境之间的交互，也可以
合成以图像为依据的轨迹数据，用于评测、监督微调和强化学习。目前规范化支持的
内容类型是文本和图像；相关契约在设计上允许未来加入更多模态，而不要求所有 Env
都必须是有状态的。

Python 包：`dataflow_mm_agent` · Python `>=3.10` · Apache-2.0

<table>
  <tr>
    <td align="center" width="33%" valign="top">
      <a href="examples/showcases/01_geometry_proof.md"><img src="examples/showcases/assets/geometry_proof/trajectory.gif" height="180" alt="Agent 逐步构造并证明一道奥林匹克几何题"></a><br>
      <sub>数学推理 · 几何构图与证明</sub>
    </td>
    <td align="center" width="33%" valign="top">
      <a href="examples/showcases/02_pixel_game.md"><img src="examples/showcases/assets/pixel_game/trajectory.gif" height="180" alt="Agent 在 Pyxel 游戏中收集五颗宝石"></a><br>
      <sub>Pyxel 游戏 · 视觉规划</sub>
    </td>
    <td align="center" width="33%" valign="top">
      <a href="examples/showcases/03_pptx.md"><img src="examples/showcases/assets/pptx/trajectory.gif" height="180" alt="Agent 将参考幻灯片复刻成可编辑的 PowerPoint"></a><br>
      <sub>PPT 复刻 · 可编辑幻灯片</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%" valign="top">
      <a href="examples/showcases/05_mobile_weekly_alarms.md"><img src="examples/showcases/assets/mobile_weekly_alarms/trajectory.gif" height="180" alt="Agent 在 Android 手机中设置每周闹钟"></a><br>
      <sub>手机操作 · Android 闹钟</sub>
    </td>
    <td align="center" width="33%" valign="top">
      <a href="examples/showcases/07_blender_lighthouse.md"><img src="examples/showcases/assets/blender_lighthouse/trajectory.gif" height="180" alt="Agent 在 Blender 中搭建低多边形海岛灯塔"></a><br>
      <sub>Blender · 三维场景搭建</sub>
    </td>
    <td align="center" width="33%" valign="top">
      <a href="examples/showcases/08_vlmgym_2048.md"><img src="examples/showcases/assets/vlmgym_2048/trajectory.gif" height="180" alt="Agent 读取 2048 棋盘并合成 64 图块"></a><br>
      <sub>2048 · 看图规划与合并</sub>
    </td>
  </tr>
</table>

## 这个包可以做什么？

1. **基于图像的数学推理**——
   [观看 Agent 构图并证明一道奥林匹克几何题](examples/showcases/01_geometry_proof.md)。
2. **平面游戏中的视觉规划**——
   [查看 Pyxel Agent 如何在步数限制内收集五颗宝石](examples/showcases/02_pixel_game.md)。
3. **可编辑视觉复刻**——
   [根据三页参考图复刻一份可编辑的 PowerPoint](examples/showcases/03_pptx.md)。
4. **从文档合成流程图**——
   [将两页事故响应手册转化为可编辑的操作流程图](examples/showcases/04_diagram.md)。
5. **移动端 UI 自动化**——
   [在独立 Android 设备上设置工作日、周末与阅读闹钟](examples/showcases/05_mobile_weekly_alarms.md)。
6. **浏览器中的规划与表单操作**——
   [按时间和预算约束安排活动日程并保存方案](examples/showcases/06_playwright_studio_day.md)。
7. **三维场景搭建**——
   [在 Blender 中搭建低多边形海岛灯塔，并如实保留步数上限结果](examples/showcases/07_blender_lighthouse.md)。
8. **视觉游戏规划：2048**——
   [从 G1 VLM-Gym 画面读出数字，从开局合成 64](examples/showcases/08_vlmgym_2048.md)。
9. **视觉路径规划：连连看**——
   [在 G1 的 12×12 棋盘上用不超过两次转弯的路径连接相同图块](examples/showcases/09_vlmgym_shisensho.md)。
10. **类别级视觉匹配**——
   [清空一盘以 CIFAR-10 照片为图块的连连看](examples/showcases/10_vlmgym_shisensho_cifar10.md)。
11. **带连锁消除的三消**——
   [在 G1 的 Swap 棋盘上经历自动洗牌拿到 150 分](examples/showcases/11_vlmgym_swap.md)。
12. **为什么需要确定性 Verifier**——
   [查看一条获得 Judge 1.0 分、却没有通过精确状态验证的轨迹](examples/showcases/12_why_deterministic_verifier.md)。

示例页使用 GitHub 原生 Markdown、完整轨迹 GIF 预览、每一步的图片观察和精简
JSON，无需 JavaScript 或嵌入大量 base64 图片的 HTML。产物与运行记录见
[Showcase 索引](examples/showcases/README.md)。

## 快速开始

```bash
conda create -n dataflow-mm-agent python=3.12 pip -y
conda activate dataflow-mm-agent
python -m pip install .
```

配置好模型后端后，跑通第一次 rollout：

```python
from dataflow_mm_agent import AgentRollout, Message, RolloutConfig, Task
from dataflow_mm_agent.serving import create_model_serving_from_env

task = Task(
    task_id="draw-001",
    env_id="my_visual_env",
    messages=(Message.text("user", "Create the requested diagram."),),
)
trajectory = AgentRollout(
    serving=create_model_serving_from_env(),
    config=RolloutConfig(max_steps=32),
).run(task)
```

[**快速开始文档**](QUICKSTART.zh-CN.md) 涵盖安装、模型后端配置、多模态任务、
流程各阶段（Generate、ReplayVerify、Judge、Refine、Filter/Select）、运行期设置、
训练数据导出、离线轨迹查看器，以及开发模式安装。

## 工作方式

一个 `Task` 指定 Env 并携带模型看到的消息。`AgentRollout` 创建全新的 Env、
运行一次工具循环，把每一步动作和 observation 记录成 `Trajectory`。各算子围绕
这份记录组合：

| 阶段 | 做什么 |
| --- | --- |
| Generate | 运行工具循环并记录未评分的 trajectory，支持逐条续跑 |
| ReplayVerify | 在全新 Env 中重放已记录的动作，并执行任务自带的确定性 verifier |
| Judge | 依据真实 observation 按 rubric 评分，并指出哪些步骤出了问题 |
| Refine | 带着 verifier 结论、评审建议和被标记的步骤重新探索 |
| Filter / Select | 按声明式的质量与多样性条件保留 trajectory |
| Export | 把 trajectory 转成 ms-swift 的 `messages` JSONL 用于监督微调 |

Judge 和 ReplayVerify 回答的是不同问题：前者评估过程，后者重放动作并检查精确
状态。开放式创作任务的 replay 结果是 `not_applicable`，而不是硬凑一个 verifier。

## 轻量级 Env 设计

一个 Env 只需要提供工具目录和调用分发器：

```python
from dataflow_mm_agent import TextContent, ToolResult, ToolSpec
from dataflow_mm_agent.env import register_env


class EchoEnv:
    def tools(self):
        return (ToolSpec(
            name="echo",
            description="Echo one string.",
            operation_type="query",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        ),)

    def call(self, tool_name, args):
        if tool_name != "echo":
            return ToolResult.failure("unknown_tool", tool_name)
        return ToolResult.success((TextContent(args["text"]),))


def register():
    register_env(
        "echo",
        EchoEnv,
        description="A stateless echo service.",
        modalities=("text",),
    )
```

下面就是完整的强制接口：

```python
def tools(self) -> Sequence[ToolSpec]: ...
def call(self, tool_name: str, args: Mapping[str, Any]) -> ToolResult: ...
```

有状态 Env 可以额外实现 `start(init, workspace)` 和 `close()`，但不需要实现
task provider、Scenario、snapshot 或 verifier。Runner 会提供 `finish`；Env
不得自行注册 finish 工具。

Rollout 和 replay 使用相同的启动顺序：

```text
创建 Env -> 可选 start(init, workspace) -> tools() -> 工具循环 -> close()
```

没有 `start` 时直接跳过；实现了 `start` 时，`tools()` 只需在启动成功后可用。
运行时在第一次模型决策或重放动作之前读取一次工具目录，并在整个 episode 中
保持固定。启动失败会停止工具发现与执行，但仍会调用可用的清理 hook。模型消息
顺序不变：system prompt、task messages，然后是可选的初始 observation。

直接构造 `ToolLoop(env)` 时，应先完成可选的启动步骤；构造函数会读取
`env.tools()`，不会替调用方启动 Env。

### 接入 MCP

一个 MCP server 可以通过薄 adapter 接入：

1. 将 `list_tools()` 的结果映射成 `ToolSpec`；
2. 将 `call_tool()` 的内容和错误映射成 `ToolResult`；
3. 使用 `register_env` 注册 adapter factory。

不需要框架专属的 task/verifier bundle。无状态 MCP adapter 可以只实现
`tools()` 和 `call()`；如有需要，会话启动和清理可以使用可选的生命周期 hook。
有会话的 MCP 在 `start()` 中连接并完成握手，再由 `tools()` 通过同一会话发现
固定工具目录；`call()` 复用该会话，`close()` 负责关闭。无需预生成 catalog，
也无需另开一次发现会话；MCP SDK 和传输细节仍由 Env 包负责。

包中附带的 [`create-env` workspace skill](dataflow_mm_agent/skills/create-env/SKILL.md)
给出了 adapter 工作流和验证要求。

## 核心契约

```text
Task ──> AgentRollout ──> Trajectory
 │           │
 │           └── fresh Env selected by task.env_id
 │
 └── optional Scenario(init)

Trajectory + TaskResolver + optional VerifierResolver
                              └──> ReplayVerify ──> ReplayVerification
```

- Runner 始终接收一个 `Task`；Task 与其 trajectories 是一对多关系。
- 只有当私有初始化数据必须进入新 Env 时，才需要 `Scenario`。
- Registry 负责 Env factory 和面向求解器的元数据，不负责存储任务。
- Verification 独立解析，不会强制要求 Scenario。
- `Trajectory` 保存动作和观察，不保存 verifier 分数或私有 Scenario 数据。

## 仓库结构

```text
dataflow-mm-agent/
├── dataflow_mm_agent/
│   ├── contracts/          # Task、Env、消息、工具和 trajectory
│   ├── env/                # registry、plugin 和进程隔离 adapter
│   ├── runtime_components/ # rollout、工具循环、上下文策略和 ReplayVerify
│   ├── operators/          # Generate、Judge、Refine、Filter 和 Select
│   ├── export/             # ms-swift 训练数据导出
│   ├── prompts.py          # 内置的中英双语提示词
│   ├── serving/            # OpenAI-compatible 与 Gemini 多模态 serving
│   ├── visualization/      # 离线 trajectory HTML 导出器与查看器
│   ├── skills/create-env/  # 用于 Env 和 MCP 接入的 workspace skill
│   └── storage/            # task 与 trajectory store
├── examples/showcases/     # GitHub 原生 trajectory 演示
├── QUICKSTART.zh-CN.md
├── LICENSE
└── pyproject.toml
```

具体 Env 位于核心发行包之外，因此安装一个集成不会迫使
`dataflow-mm-agent` 同时安装所有渲染或游戏依赖。如果某个集成需要独立的解释器
或依赖边界，可以使用本包提供的进程代理。

## 进一步阅读

- [快速开始：安装、配置与运行](QUICKSTART.zh-CN.md)
- [Showcase 索引](examples/showcases/README.md)
- [离线轨迹 HTML 报告](dataflow_mm_agent/visualization/README.md)
- [创建 Env 或 MCP 适配器](dataflow_mm_agent/skills/create-env/SKILL.md)
- [Env 契约与包结构](dataflow_mm_agent/skills/create-env/references/contracts-and-layout.md)
- [任务生成](dataflow_mm_agent/skills/create-env/references/task-generation.md)
- [验证策略](dataflow_mm_agent/skills/create-env/references/validation.md)
