# DataFlow-MM-Agent

<p align="center">
  <a href="README.md">English</a> | <strong>简体中文</strong> | <a href="QUICKSTART.zh-CN.md">快速开始</a>
</p>

<p align="center"><img src="assets/banner.png" alt="DataFlow-MM-Agent：在任意 Env 中运行多模态 Agent，并保留可验证的轨迹" width="100%"></p>

<p align="center">
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="python"></a>
  <a href="dataflow_mm_agent/version.py"><img src="https://img.shields.io/badge/version-1.0.7-blue" alt="version"></a>
  <a href="https://github.com/OpenDCAI/DataFlow"><img src="https://img.shields.io/badge/built%20on-open--dataflow--mm-6c8cff" alt="built on"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-lightgrey" alt="license"></a>
</p>

**让多模态 Agent 在视觉环境中真正动手，并把每一次运行都还原成结构化、可重放
的 `Trajectory`。**

- **看着 Agent 在画面上操作** —— 一个任务、一个 Env、一次工具循环；每一步动作
  和渲染出的 observation 都被完整记录。
- **合成轨迹数据** —— 规模化生成，然后验证、评审、修复、筛选，再导出用于评测、
  SFT 或 RL。
- **保留下来的数据是可信的** —— 在全新 Env 中确定性重放，回答“这串动作是否真的
  能得到那个最终状态”。

Python 包：`dataflow_mm_agent`。目前规范化支持的内容类型是文本和图像；相关契约
在设计上允许未来加入更多模态，而不要求所有 Env 都必须是有状态的。

<div align="center">
<table align="center">
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
</div>

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

## 框架设计

DataFlow-MM-Agent 通过统一的数据契约连接环境交互、轨迹记录与数据处理，
让新接入的 Env 能够复用同一套 rollout 和评估组件。

- **轻量接入环境。** Env 通过 `tools()` 暴露工具、通过 `call()` 执行调用，
  按需实现 `start()` 和 `close()`。集成代码与应用依赖放在独立的 Env 包中，
  也可以使用单独的 Python 进程运行。
- **统一记录多模态轨迹。** `Trajectory` 保存模型消息、动作和观察，文本与图像
  使用明确的内容块表示。同一份记录可以用于可视化、在新 Env 中重放动作，
  以及导出训练数据。
- **分别验证结果与评估质量。** ReplayVerify 重放动作并检查任务的确定性条件；
  Judge 根据轨迹中的证据按 rubric 评分，并给出修复建议。两类结果分别保留，
  便于 pipeline 按任务特点选择适用的检查方式。
- **按需组合数据处理步骤。** 生成、搜索、验证、评审、修复、筛选和导出由独立
  算子或工具完成，可以自由组合成 pipeline。Refine 产生新的 trajectory，
  原始尝试仍保留用于对比。

## 轨迹生成与处理流程

一个 `Task` 指定 Env 并携带模型看到的消息。`AgentRollout` 创建全新的 Env、
运行一次工具循环，把每一步动作和 observation 记录成 `Trajectory`。各算子围绕
这份记录组合：

<p align="center"><img src="assets/pipeline.png" alt="任意 Env 插进 tools() + call()，任意模型插进 ModelServing，各算子围绕记录下来的 Trajectory 自由拼接" width="100%"></p>

| 阶段 | 算子 | 做什么 |
| --- | --- | --- |
| Generate | `AgentMMExploreGenerator` | 运行工具循环并记录未评分的 trajectory，支持逐条续跑 |
| Search | `AgentMMExploreTreeGenerator` | 每个节点分支出多个动作，每个子节点都从全新 Env 重放得到 |
| ReplayVerify | `AgentMMReplayVerifier` | 在全新 Env 中重放已记录的动作，并执行任务自带的确定性 verifier |
| Judge | `AgentMMTrajectoryQualityEvaluator` | 依据真实 observation 按 rubric 评分，并指出哪些步骤出了问题 |
| Refine | `AgentMMTrajectoryRefiner` | 带着 verifier 结论、评审建议和被标记的步骤重新探索 |
| Select | `AgentMMTrajectorySelector` | 按声明式条件筛选 trajectory，再排序、去重并截断数量 |
| Export | `dataflow_mm_agent.export` | 把 trajectory 转成 ms-swift 的 `messages` JSONL 用于监督微调 |

Judge 和 ReplayVerify 回答的是不同问题：前者评估过程，后者重放动作并检查精确
状态。开放式创作任务的 replay 结果是 `not_applicable`，而不是硬凑一个 verifier。

## 轻量级 Env 设计

一个 Env 就是工具目录加调用分发器，下面是全部的强制接口：

```python
def tools(self) -> Sequence[ToolSpec]: ...
def call(self, tool_name: str, args: Mapping[str, Any]) -> ToolResult: ...
```

```python
class EchoEnv:
    def tools(self):
        return (ToolSpec(
            name="echo",
            description="Echo one string.",
            operation_type="query",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}},
                          "required": ["text"], "additionalProperties": False},
        ),)

    def call(self, tool_name, args):
        if tool_name != "echo":
            return ToolResult.failure("unknown_tool", tool_name)
        return ToolResult.success((TextContent(args["text"]),))


register_env("echo", EchoEnv, description="A stateless echo service.", modalities=("text",))
```

有状态 Env 可以额外实现 `start(init, workspace)` 和 `close()`，但不需要实现
task provider、Scenario、snapshot 或 verifier，`finish` 由 runner 提供。
Rollout 和 replay 使用相同的启动顺序：

```text
创建 Env -> 可选 start(init, workspace) -> tools() -> 工具循环 -> close()
```

MCP server 通过同一套接口接入：把 `list_tools()` 映射成 `ToolSpec`，把
`call_tool()` 映射成 `ToolResult`，再注册 adapter factory。有会话的 server 在
`start()` 中连接、通过同一会话发现工具目录、并在 `close()` 中释放；MCP SDK
仍然留在 Env 包里。

包中附带的 [`create-env` workspace skill](dataflow_mm_agent/skills/create-env/SKILL.md)
完整给出了工具目录发现、生命周期规则、adapter 工作流和验证要求。

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
│   ├── operators/          # Generate、Judge、Refine 和 Select
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

## 适用范围

- 目前规范化的内容类型是文本和图像，音频和视频尚未进入契约。
- 本包提供运行时、算子和契约；具体的 Env、它们的依赖和任务由独立的 Env 包提供。
- 确定性重放要求 Env 的动作能复现同一状态；依赖无种子随机性或外部在线服务的
  Env 无法用这种方式验证。
- 进程隔离是可选项：只有当 Env 需要独立解释器或依赖边界时才需要启用。

## 进一步阅读

- [快速开始：安装、配置与运行](QUICKSTART.zh-CN.md)
- [Showcase 索引](examples/showcases/README.md)
- [离线轨迹 HTML 报告](dataflow_mm_agent/visualization/README.md)
- [创建 Env 或 MCP 适配器](dataflow_mm_agent/skills/create-env/SKILL.md)
- [Env 契约与包结构](dataflow_mm_agent/skills/create-env/references/contracts-and-layout.md)
- [任务生成](dataflow_mm_agent/skills/create-env/references/task-generation.md)
- [验证策略](dataflow_mm_agent/skills/create-env/references/validation.md)

## 许可证

Apache-2.0，详见 [LICENSE](LICENSE)。

## 致谢

- [DataFlow-MM](https://github.com/OpenDCAI/DataFlow)：本包沿用了它的算子、
  存储和 registry 约定。
- 各 Showcase 背后的 Env 包与上游项目；每个 Env 包都在自己的文档中说明来源与
  许可证。
