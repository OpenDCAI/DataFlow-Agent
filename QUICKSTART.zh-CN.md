<a name="快速开始"></a>

# 🔁 从安装到第一条轨迹

[English](QUICKSTART.md) | **简体中文** · [← README](README.zh-CN.md)

从安装和模型配置开始，跑通首条 rollout，再查看轨迹、导出为 ms-swift 格式。
先完整运行一个任务，再按需组合 pipeline 各阶段。

<a name="目录"></a>

## 📑 目录

- [环境要求](#环境要求)
- [安装](#安装)
- [配置模型后端](#配置模型后端)
- [第一次 Rollout](#第一次-rollout)
- [多模态任务](#多模态任务)
- [流程各阶段](#流程各阶段)
- [导出为 ms-swift 格式](#导出为-ms-swift-格式)
- [查看轨迹](#查看轨迹)
- [开发模式安装与测试](#开发模式安装与测试)


<a name="环境要求"></a>

## 🛠️ 环境要求

- Conda，可以使用 Miniconda 或 Anaconda
- 安装时可以访问网络，以便解析并下载 Python 依赖

具体视觉 Env 可能还需要浏览器、渲染、游戏或 Office 相关依赖；这些依赖属于
对应的 Env 集成，不属于本核心包。

<a name="安装"></a>

## 📦 安装

如果已经有 Python 3.10+ 环境和 Git，直接使用 [README 快速开始](README.zh-CN.md#快速开始)
中的两行安装命令即可。下面给出 Conda + 源码 ZIP 的完整步骤，不需要 Git。

1. 在 GitHub 仓库的 `mm-agent` 分支页面选择 **Code → Download ZIP**。
2. 解压下载的文件，并在解压后的目录中打开终端；该目录应当包含
   `pyproject.toml`。
3. 创建并激活推荐的 Conda 环境：

```bash
conda create -n dataflow-agentmm python=3.12 pip -y
conda activate dataflow-agentmm
```

4. 更新打包工具，然后安装解压后的包：

```bash
python -m pip install --upgrade pip
python -m pip install .
```

5. 验证安装结果：

```bash
python -c "import dataflow_agentmm as d; print(d.__version__)"
```

该命令应当输出已安装的包版本。以后升级时，重新下载并解压新版 ZIP，激活同一个
Conda 环境，然后在新目录中运行 `python -m pip install --upgrade .` 即可。

<a name="配置模型后端"></a>

## ⚙️ 配置模型后端

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
也支持 Gemini 后端，配置方式见 [serving factory](dataflow_agentmm/serving/serving_factory.py)。

<a name="第一次-rollout"></a>

## ▶️ 第一次 Rollout

下面假设外部 Env 包已经注册了 `my_visual_env`；请将这个占位 ID 替换成已注册的
Env ID（参见[轻量级 Env 设计](README.zh-CN.md#轻量级-env-设计)）。

```python
from dataflow_agentmm import AgentRollout, Message, RolloutConfig, Task
from dataflow_agentmm.serving import create_model_serving_from_env

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

<a name="多模态任务"></a>

## 🖼️ 多模态任务

```python
from pathlib import Path

from dataflow_agentmm import ImageContent, Message, Task, TextContent

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

<a name="流程各阶段"></a>

## 🔄 流程各阶段

DataFlow-AgentMM 沿用 DataFlow 的可组合算子风格，同时将生成、重放和质量评估
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
  全有或全无的分片回退路径，避免用残缺标准计算均分。ReplayVerify 结果只在调用方
  提供时注入。除评分外，Judge 还产出供修复阶段使用的证据：`important_steps`
  （出问题的步骤序号）和 `refine_suggestion`（该怎么改），没有问题时两者为空。
- **Refine** 接收原始任务消息、失败诊断，以及 Judge 和 ReplayVerify 产出的证据，
  然后在全新 Env 中重新探索，而不是修改原 trajectory。它的修复上下文包含确定性
  verifier 失败的检查项、评审给出的修改建议，以及 Judge 标记的每个关键步骤及其
  前后 `important_step_window` 步，这些步骤连同 observation 和图片完整呈现、不做
  截断。其余步骤的摘要保留最新的步骤并省略中间部分。
- **Select** 保留符合流程质量与多样性要求的 trajectory。
  `AgentMMTrajectorySelector` 只保留同时满足所有已传入条件的 trajectory；
  `reject_reason(trajectory, row)` 返回第一个未满足的条件，pipeline 可以据此把被
  筛掉的行送进 Refine 分支。所有开关都是可选的：内置字段（`num_steps`、
  `num_tool_calls`、`num_tool_errors`、`num_invalid_tool_calls`、
  `num_parse_errors`、`max_repeated_action`、`has_final_answer`、`is_success`、
  `avg_observation_len`、`is_finish`、`replay_passed`、`judge_score` 等），以及通过
  `register_selector_feature(name, fn)` 注册的自定义字段，其中 `fn(trajectory, row)`
  可以读取 trajectory 或它在 storage 中的整行。条件可以是一个值（`is_finish=True`）
  或比较运算（`num_steps={"gte": 2}`）。可选的 `sort_by`、`group_by`、
  `dedupe_threshold` 和 `max_selected` 用于排序与截断。加权打分由使用者自己注册
  字段实现，可直接组合 `operators.selector_features` 中公开的内置函数：

  ```python
  from dataflow_agentmm.operators import AgentMMTrajectorySelector, register_selector_feature, uses_tool
  from dataflow_agentmm.operators import selector_features as sf

  register_selector_feature("use_api_tool", uses_tool("api", successful=True))

  @register_selector_feature("quality_score")
  def quality_score(trajectory, row):
      return 0.6 * sf.replay_passed(trajectory, row) + 0.4 * min(sf.num_steps(trajectory, row) / 5, 1)

  selector = AgentMMTrajectorySelector(
      is_finish=True, use_api_tool=True, quality_score={"gte": 0.5},
      sort_by="quality_score", group_by="task_id", max_selected=3,
  )
  ```

另外有两项贯穿各阶段的 runtime 设置：

- **请求上下文**。`RolloutConfig(context_policy=ContextPolicy(...))` 决定每次请求
  携带多少已记录的历史。默认关闭，也就是每次都发送完整历史。启用后会保留 system
  prompt、任务消息、最近 `keep_last_steps` 个回合（每个回合完整保留，模型能看到
  自己当时传的参数），以及最新一张带图的 observation，其余合并成一行说明，注明
  省略了多少步。还可以设置 `max_prefix_chars` 字符预算，或通过 `summarizer` 回调
  接入模型总结。裁剪只作用于请求：trajectory 始终记录全部内容，重放、导出和评审
  都不受影响。
- **提示词语言**。内置提示词提供英文和中文两套（`dataflow_agentmm/prompts.py`），
  默认英文。给 `RolloutConfig`、`AgentMMTrajectoryQualityEvaluator` 或
  `AgentMMTrajectoryRefiner` 传 `language="zh"` 即可切换 system prompt、通用 rubric
  文案和修复模板；显式传入的 `system_prompt=` 优先级更高。任务本身用什么语言，
  仍由 Task 的 messages 决定。

开放式创作任务不需要虚构一个 verifier。它们的 ReplayVerify 状态为
`not_applicable`，由 Judge 评估渲染结果及其生成过程。

<a name="导出为-ms-swift-格式"></a>

## 📤 导出为 ms-swift 格式

Trajectory 可以转换为 [ms-swift](https://github.com/modelscope/ms-swift) 的
`messages` JSONL 格式。导出只做格式转换，不按验证或 Judge 结果筛选。

```bash
dataflow-agentmm-export-swift trajectories/*.jsonl -o exports/trajectories.jsonl
# 或：python -m dataflow_agentmm.export trajectories/*.jsonl -o exports/trajectories.jsonl
```

每条 trajectory 导出为一行。system、user、assistant 消息保留原始记录文本
（assistant 文本即模型原始动作响应）；回应 assistant 的 observation 转为
`tool_response`；图片转为 `<image>` 标签并按顺序写入 `images`。图片默认按内容
去重写入 `exports/trajectories_images/` 并使用绝对路径引用，也可用 `--image-mode base64`
内联。输入可以是 trajectory JSON、`TrajectoryStore` JSONL，或带 `trajectory`
列的 pipeline JSONL。最后一个 assistant 回合之后的消息会被丢弃，没有 assistant
回合的 trajectory 会被跳过。

<a name="查看轨迹"></a>

## 🔎 查看轨迹

把任意 trajectory、pipeline JSONL 或 rollout 目录导出成一个离线 HTML 页面，
包含每一步的动作、observation 和图片：

```bash
dataflow-agentmm-trajectory-html trajectory.json -o report.html
# 或：python -m dataflow_agentmm.visualization trajectory.json -o report.html
```

字段含义、Judge 相关列以及它刻意区分开的几类证据，见
[查看器文档](dataflow_agentmm/visualization/README.md)。

<a name="开发模式安装与测试"></a>

## 🧪 开发模式安装与测试

如果需要直接修改源码，可以使用 test extra 进行可编辑安装：

```bash
python -m pip install -e ".[test]"
```

远程 MCP adapter 可以保持得很轻，因为工具及其集成专属依赖运行在上游 MCP
server 中。
