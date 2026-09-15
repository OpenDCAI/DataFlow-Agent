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

## 安装

### 环境要求

- Conda，可以使用 Miniconda 或 Anaconda
- 安装时可以访问网络，以便解析并下载 Python 依赖

具体视觉 Env 可能还需要浏览器、渲染、游戏或 Office 相关依赖；这些依赖属于
对应的 Env 集成，不属于本核心包。

### 推荐方式：下载 ZIP 后安装

1. 在 GitHub 仓库的 `mm-agent` 分支页面选择 **Code → Download ZIP**。
2. 解压下载的文件，并在解压后的目录中打开终端；该目录应当包含
   `pyproject.toml`。
3. 创建并激活推荐的 Conda 环境：

```bash
conda create -n dataflow-mm-agent python=3.12 pip -y
conda activate dataflow-mm-agent
```

4. 更新打包工具，然后安装解压后的包：

```bash
python -m pip install --upgrade pip
python -m pip install .
```

5. 验证安装结果：

```bash
python -c "import dataflow_mm_agent as d; print(d.__version__)"
```

该命令应当输出已安装的包版本。以后升级时，重新下载并解压新版 ZIP，激活同一个
Conda 环境，然后在新目录中运行 `python -m pip install --upgrade .` 即可。

### 配置模型后端

安装本身不需要 API key；运行 rollout 前，请配置模型接口。
`create_model_serving_from_env()` 会从当前进程的环境变量中读取配置。

使用 OpenAI-compatible 接口：

```bash
export SERVING_BACKEND=openai
export MODEL=your-model-name
export API_URL=https://your-endpoint.example/v1
export DF_API_KEY=your-api-key
```

在 Windows PowerShell 中，使用 `$env:` 设置同样的变量，例如：

```powershell
$env:SERVING_BACKEND = "openai"
$env:MODEL = "your-model-name"
$env:API_URL = "https://your-endpoint.example/v1"
$env:DF_API_KEY = "your-api-key"
```

请通过 shell 或密钥管理服务设置这些变量，不要把实际值提交到仓库。
也支持 Gemini 后端，配置方式见 [serving factory](dataflow_mm_agent/serving/serving_factory.py)。

### 开发模式安装

如果需要直接修改源码，可以使用 test extra 进行可编辑安装：

```bash
python -m pip install -e ".[test]"
```

远程 MCP adapter 可以保持得很轻，因为工具及其集成专属依赖运行在上游 MCP
server 中。

## 最小 Rollout 示例

下面假设外部 Env 包已经注册了 `my_visual_env`；请将这个占位 ID 替换成已注册的
Env ID（参见[轻量级 Env 设计](#轻量级-env-设计)）。

```python
from dataflow_mm_agent import AgentRollout, Message, RolloutConfig, Task
from dataflow_mm_agent.serving import create_model_serving_from_env

task = Task(
    task_id="draw-001",
    env_id="my_visual_env",
    messages=(Message.text("user", "Create the requested diagram."),),
)

serving = create_model_serving_from_env()
trajectory = AgentRollout(
    serving=serving,
    config=RolloutConfig(max_steps=32),
).run(task)

print(trajectory.termination_reason)
print(trajectory.steps[-1].action)
```

`Task` 可以复用：一个任务可以产生多条 trajectory。它的 `messages` 可以包含
文本和任意数量的图片。`Scenario` 是可选的私有运行时输入，并不是每个任务都
必须套用的包装层。`judge_ref` 是可选的公开评分范围与任务专用评判标准；省略
时 Judge 使用内置通用 rubric。

## 多模态任务

```python
from pathlib import Path

from dataflow_mm_agent import ImageContent, Message, Task, TextContent

reference = ImageContent.from_bytes(
    Path("reference.png").read_bytes(),
    "image/png",
    detail="original",
)
task = Task(
    task_id="reconstruct-001",
    env_id="diagram",
    messages=(Message.of(
        "user",
        (TextContent("Reconstruct this as an editable diagram."), reference),
    ),),
)
```

图片在 rollout、Refine、Judge 和 trajectory 存储的整个流程中始终是一等内容块，
不会被转换成文本占位符。

物化 JSON task store 可以用受目录约束且带 SHA-256 的 `text_ref` 保存 JSON
之外的来源文档（仅支持 UTF-8 `text/plain` 或 `text/markdown`，上限 512 KiB）。
store 会在 rollout 前把它解析为普通 `TextContent`，就像把 `image_ref` 解析成
内联 `ImageContent`；路径缺失、越界或哈希不符会直接拒绝任务，不会交给模型自行
联网补资源。

## Trajectory 数据流

DataFlow-MM-Agent 沿用 DataFlow 的可组合算子风格，同时将生成、重放和质量评估
作为彼此独立的关注点：

- **Generate** 运行共享的多模态工具循环，并记录尚未评分的 trajectory。设置
  `checkpoint_dir` 后，每完成一条就立即保存为 `<sample key>.jsonl`；重跑时跳过
  已有非基础设施错误结果的样本；`max_retries` 会重跑抛出异常或以
  `infrastructure_error` 结束的样本；`run_manifest.jsonl` 记录每次运行的数量、
  配置和模型用量汇总。serving 适配器上报 token 用量时，每条 trajectory 都会在
  `metadata.usage` 中记录。
- **ReplayVerify** 在全新的 Env 中重放已存储的动作；如果任务配置了确定性
  verifier，还会独立执行该 verifier。
- **Judge** 解析 Task 的可选 `judge_ref`（缺省时注入通用 rubric），逐项评分，
  将各项按配置范围归一化后取算术平均作为 `traj_overall`。所有环境使用统一的
  rationale + scores 输出协议；任务特有评分标准只写在 task rubric 中。Judge
  不能代替精确的状态验证。序列化后超过 16,000 字符的 rubric 会逐 criterion
  分片评判，每个分片仍注入完整 task rubric；组合 verdict 解析失败时也走同一条
  全有或全无的分片回退路径，避免用残缺标准计算均分。
- **Refine** 接收原始任务消息、视觉观察和失败诊断，并产生一条新 trajectory，
  而不是修改原 trajectory。对于有状态的视觉产物，它可以先在新 workspace 中
  重放原 trajectory 的 `finish` 前动作，恢复成品后再追加最新诊断，让模型只做
  局部续写与修复。
- **Filter 和 Select** 保留符合流程质量与多样性要求的 trajectory。
  `AgentMMTrajectorySelector` 只保留同时满足所有已传入条件的 trajectory。所有开关
  都是可选的：内置字段（`num_steps`、`num_tool_calls`、`num_tool_errors`、
  `avg_observation_len`、`is_finish`、`replay_passed`、`judge_score` 等），以及通过
  `register_selector_feature(name, fn)` 注册的自定义字段，其中 `fn(trajectory, row)`
  可以读取 trajectory 或它在 storage 中的整行。条件可以是一个值（`is_finish=True`）
  或比较运算（`num_steps={"gte": 2}`）。可选的 `sort_by`、`group_by`、
  `dedupe_threshold` 和 `max_selected` 用于排序与截断。加权打分由使用者自己注册
  字段实现，可直接组合 `operators.selector_features` 中公开的内置函数：

  ```python
  from dataflow_mm_agent.operators import AgentMMTrajectorySelector, register_selector_feature, uses_tool
  from dataflow_mm_agent.operators import selector_features as sf

  register_selector_feature("use_api_tool", uses_tool("api", successful=True))

  @register_selector_feature("quality_score")
  def quality_score(trajectory, row):
      return 0.6 * sf.replay_passed(trajectory, row) + 0.4 * min(sf.num_steps(trajectory, row) / 5, 1)

  selector = AgentMMTrajectorySelector(
      is_finish=True, use_api_tool=True, quality_score={"gte": 0.5},
      sort_by="quality_score", group_by="task_id", max_selected=3,
  )
  ```

开放式创作任务不需要虚构一个 verifier。它们的 ReplayVerify 状态为
`not_applicable`，由 Judge 评估渲染结果及其生成过程。

### 导出训练数据

Trajectory 可以转换为 [ms-swift](https://github.com/modelscope/ms-swift) 的
`messages` JSONL 用于监督微调。导出只做格式转换，不按验证或 Judge 结果筛选。

```bash
dataflow-mm-export-swift trajectories/*.jsonl -o sft/train.jsonl
# 或：python -m dataflow_mm_agent.export trajectories/*.jsonl -o sft/train.jsonl
```

每条 trajectory 导出为一行。system、user、assistant 消息保留原始记录文本
（assistant 文本即模型原始动作响应）；回应 assistant 的 observation 转为
`tool_response`；图片转为 `<image>` 标签并按顺序写入 `images`。图片默认按内容
去重写入 `sft/train_images/` 并使用绝对路径引用，也可用 `--image-mode base64`
内联。输入可以是 trajectory JSON、`TrajectoryStore` JSONL，或带 `trajectory`
列的 pipeline JSONL。最后一个 assistant 回合之后的消息会被丢弃，没有 assistant
回合的 trajectory 会被跳过。

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
│   ├── runtime_components/ # rollout、工具循环、finish 和 ReplayVerify
│   ├── operators/          # Generate、Judge、Refine、Filter 和 Select
│   ├── export/             # ms-swift 训练数据导出
│   ├── serving/            # OpenAI-compatible 与 Gemini 多模态 serving
│   ├── visualization/      # 离线 trajectory HTML 导出器与查看器
│   ├── skills/create-env/  # 用于 Env 和 MCP 接入的 workspace skill
│   └── storage/            # task 与 trajectory store
├── examples/showcases/     # GitHub 原生 trajectory 演示
├── LICENSE
└── pyproject.toml
```

具体 Env 位于核心发行包之外，因此安装一个集成不会迫使
`dataflow-mm-agent` 同时安装所有渲染或游戏依赖。如果某个集成需要独立的解释器
或依赖边界，可以使用本包提供的进程代理。

## 进一步阅读

- [离线 trajectory HTML 报告](dataflow_mm_agent/visualization/README.md)
- [创建 Env 或 MCP adapter](dataflow_mm_agent/skills/create-env/SKILL.md)
- [Env 契约与包结构](dataflow_mm_agent/skills/create-env/references/contracts-and-layout.md)
- [Task 生成](dataflow_mm_agent/skills/create-env/references/task-generation.md)
- [验证策略](dataflow_mm_agent/skills/create-env/references/validation.md)
- [Showcase 索引](examples/showcases/README.md)
