# Trajectory HTML 可视化

把已有 trajectory 导出成可直接打开、离线分享的单文件 HTML。不绑定 Env、MCP
或模型提供商，不重跑工具、不调用模型，不需要额外前端依赖或服务器。

## 命令行

```bash
# 单条 trajectory JSON / TrajectoryStore JSONL
python -m dataflow_mm_agent.visualization trajectory.json -o report.html

# pipeline JSONL：支持多行，trajectory 字段可以是对象或 JSON 字符串
python -m dataflow_mm_agent.visualization pipeline_step3.jsonl -o pipeline.html --tasks ./tasks

# rollout 目录：读取 trajectory.json，并自动附加同目录的 task.json / summary.json
python -m dataflow_mm_agent.visualization ./run-directory -o run.html

# 自定义 pipeline 列名
python -m dataflow_mm_agent.visualization rows.jsonl -o report.html \
  --trajectory-key rollout --score-key quality --score-prefix quality_
```

安装/更新本包后，也可使用 `dataflow-mm-trajectory-html`，参数相同。
`--task task.json`、`--summary summary.json` 可显式指定补充数据；`--title` 设置标题。
默认不覆盖文件；明确使用 `--force` 才覆盖已有 HTML，仍禁止覆盖输入/sidecar 或输出符号链接。

## Python API

```python
from dataflow_mm_agent.visualization import render_trajectory_html, export_trajectory_html

# 输入 Trajectory 对象、trajectory dict、pipeline 行，或上述记录组成的 list/tuple。
html = render_trajectory_html(trajectory, title="My rollout", task=task)
html = render_trajectory_html(pipeline_rows, score_key="traj_overall", score_prefix="traj_")

# 文件导出；tasks_dir 可选，通过 JsonTaskStore 只补充对应 Task。
path = export_trajectory_html("pipeline.jsonl", "report.html", tasks_dir="tasks")
```

## 可以看到什么

- 多条记录导航、按 task/Env/episode 搜索；当前与 `trajectory_original` 原始轨迹切换。
- 轨迹首次模型响应之前记录的用户任务及补充指令；完整 system prompt（含工具目录）。
- 初始上下文与每一步响应之前的消息前缀。few-shot、startup observation 和 refine
  的补充消息都保留；“全部消息”中还能看到未绑定 step 的消息或基础设施错误。
- 每一步的 tool、args、action thought、模型响应原文、控制状态、错误码和记录耗时。
- 对应 observation 文本/JSON/图片；图片可点击放大。finish 没有 observation 时明确标注。
- final_answer、终止原因、最后一次图片 observation、metadata 和可选 summary。
- Judge overall、原始分、模型分、归一化分、rubric 与 rationale；独立 Replay verification。

没有 Judge 字段时显示“未记录”，不把缺失当作 0。真实 0 分正常显示。
NaN/Infinity 作为缺失处理。列名默认对应 AgentMMTrajectoryQualityEvaluator 的
`traj_overall`、`traj_judge_scores`、`traj_judge_model_scores`、
`traj_judge_normalized_scores`、`traj_rationale` 和 `judge_ref`。
其他 pipeline 字段仍能在行级元数据中查看。

## 必须区分的证据

**记录的上下文 ≠ provider 网络原始请求。** Trajectory 保存 canonical messages；
serving 可能改角色、限制图片数，refiner 也可能压缩上下文。现有记录未包含变换后的
完整网络 payload 时，报告仅展示记录的消息前缀，不声称逐字还原 provider 输入。

**行级 Judge 分数 ≠ 修订后分数。** Refiner 通常保留旧评分，也允许之后再次 Judge。
现有行字段没有明确 score→episode 绑定，因此有原始/修订版本时始终显示归属提示，
不随版本切换改称“当前版本分数”。`_refined=true` 也不代表质量已提升。

**finish ≠ 验证通过。** 流程终止、Judge 和 Replay verification 是独立证据。
最后一次图片可能只是缩略图或中间结果，不自动认定为最终交付文件。

**公开任务 ≠ 私有 Scenario。** 补充 Task 只导出 schema_version、task_id、env_id、
messages 和 judge_ref，不导出 scenario/init 或 verification。实际模型可见内容以
trajectory 中记录的 messages 为准。Task/summary 标识不匹配时显示提示并拒绝附加。

## 文件、安全与限制

- 输入支持 JSON 对象、JSON 数组和 JSONL；单个输入/sidecar 限 128 MiB。
- 支持当前契约的 text 和 base64 image；图片限 32 MiB，复用核心图片有效性校验。
  PNG/JPEG/GIF/WEBP/BMP 可展示，相同图片在 HTML 内去重。SVG、HTML、远程 URL
  和其他内容块不作为活跃内容执行；无效图片显示诊断。
- 不读取 `.env`，不追踪 action/observation 中的文件路径，也不请求远程图片、字体、脚本。
  自动 sidecar 发现仅限输入为 `trajectory.json` 或 rollout 目录时的两个固定文件名。
- 不使用 innerHTML/eval 渲染用户数据。内嵌 JSON 转义 script 结束标记，CSP 限制脚本
  为打包代码且禁止网络连接，文本按纯文本展示（不执行 Markdown/HTML）。
- 对无效 step/message 索引给出提示，不用负索引误读最后一条消息。未知字段尽量保留。
- **报告含完整提示词、模型输出与图片，分享前仍需检查敏感信息。** 这不是自动脱敏工具，
  导出器不会删除本来就在消息里的敏感内容。不要仅凭报告终止状态判断任务正确。

HTML/CSS/JS 随 wheel 打包。测试：`python -m pytest tests/test_trajectory_html.py -q`。
