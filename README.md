# DataFlow-Agent

DataFlow-Agent 是一个基于状态驱动（State-driven）的模块化 AI Agent 框架，提供可扩展的 `Agent / Workflow / Tool` 体系，并提供 CLI 脚手架与可视化 Gradio 页面，面向“数据流/算子编排”类任务（算子推荐、管线生成/调试、算子问答、网页采集等）。

中文 | [English](README_EN.md)

---

## 目录

- [特性](#特性)
- [安装](#安装)
- [启动 UI（Gradio）](#启动-ui-gradio)
- [CLI 使用](#cli-使用)
- [工作流（Workflows）](#工作流workflows)
- [配置与环境变量](#配置与环境变量)
- [文档（MkDocs）](#文档mkdocs)
- [开发与贡献](#开发与贡献)
- [常见问题](#常见问题)

---

## 特性

- **统一状态模型**：围绕 `MainState / DFState` 等状态对象组织多智能体执行过程。
- **Agent 插件化**：通过注册机制自动发现/加载 Agent，实现灵活组合。
- **Workflow 编排**：基于图结构编排节点（GraphBuilder），支持复杂流程与工具调用。
- **工具管理**：通过 `ToolManager` 注入 pre-tools/post-tools，统一管理工具与权限边界。
- **可视化页面**：内置 `gradio_app/` 页面用于 operator/pipeline 等常见能力。
- **CLI 脚手架**：`dfa create` 一键生成 workflow/agent/gradio page/prompt/state 等模板。

---

## 安装

### 1) 克隆仓库

```bash
git clone https://github.com/OpenDCAI/DataFlow-Agent
cd DataFlow-Agent
```

### 2) 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3) 安装依赖

开发/本地调试推荐：

```bash
pip install -r requirements-dev.txt
pip install -e .
```

生产/最小安装（仅基础依赖）：

```bash
pip install -r requirements.txt
pip install -e .
```

---

## 启动 UI（Gradio）

本仓库保留 DataFlow-Agent 的 operator/pipeline 等页面，启动方式：

```bash
python gradio_app/app.py --page_set data
```

参数说明：

- `--page_set data`：只加载数据流相关页面集合（推荐）
- `--page_set all`：加载 `gradio_app/pages/` 下的全部页面

默认端口：

- 环境变量 `GRADIO_SERVER_PORT` 或命令行 `--server_port`（默认 7860）
- 服务监听地址可用 `GRADIO_SERVER_NAME`（默认 `0.0.0.0`）

---

## CLI 使用

查看 CLI 帮助：

```bash
dfa --help
```

常用脚手架命令：

```bash
dfa create --agent_name my_agent
dfa create --wf_name my_workflow
dfa create --gradio_name my_page
dfa create --prompt_name my_prompt
dfa create --state_name my_state
```

生成文件位置（约定）：

- Workflow：`dataflow_agent/workflow/wf_<name>.py`
- Agent：`dataflow_agent/agentroles/common_agents/<name>_agent.py`
- Gradio Page：`gradio_app/pages/page_<name>.py`
- Prompt Template：`dataflow_agent/promptstemplates/resources/pt_<name>_repo.py`
- State：`dataflow_agent/states/<name>_state.py`

---

## 工作流（Workflows）

工作流位于 `dataflow_agent/workflow/`，文件名约定 `wf_*.py`。系统启动时会尝试自动导入并注册工作流；若某个工作流依赖的外部环境/包缺失，会在日志中提示并跳过导入。

查看当前成功注册的工作流：

```bash
python - <<'PY'
from dataflow_agent.workflow import list_workflows
print(sorted(list_workflows()))
PY
```

运行方式（以 `run_workflow` 为例）：

```bash
python - <<'PY'
import asyncio
from dataflow_agent.workflow import run_workflow
from dataflow_agent.state import MainState

async def main():
    state = MainState()
    out = await run_workflow("operator_qa", state)
    print(out)

asyncio.run(main())
PY
```

说明：

- 具体工作流需要匹配各自的 `State/Request` 输入；请参考对应 `wf_*.py` 文件与 `docs/`。

---

## 配置与环境变量

### LLM 相关

- `DF_API_URL`：LLM API Base URL（默认 `test`）
- `DF_API_KEY`：API Key（默认 `test`）
- `DATAFLOW_LOG_LEVEL`：日志级别（默认 `INFO`）
- `DATAFLOW_LOG_FILE`：日志文件（默认 `dataflow_agent.log`）

### 路径相关（可选）

`dataflow_agent/state.py` 会尽量通过 `dataflow.cli_funcs.paths.DataFlowPath` 获取路径；若外部包不可用，则回退到环境变量：

- `DATAFLOW_DIR`：数据目录根路径（默认使用仓库根路径）
- `DATAFLOW_STATICS_DIR`：静态目录（默认 `./statics`）

---

## 文档（MkDocs）

启动本地文档站点：

```bash
mkdocs serve
```

配置文件：`mkdocs.yml`

---

## 开发与贡献

- 代码结构约定：`dataflow_agent/` 为核心包；`gradio_app/` 为可视化页面；`docs/` 为文档。
- 建议在提交前至少运行一次：`python -m compileall -q dataflow_agent gradio_app`

---

## 常见问题

### 1) `import dataflow_agent.workflow` 后提示某些工作流被跳过

这是预期行为：工作流可能依赖特定外部组件（例如算子包、运行环境）。请根据日志提示补齐依赖后重试。

### 2) `ModuleNotFoundError: No module named 'dataflow.utils.*'`

说明当前环境中安装的 `dataflow` 包不是 DataFlow 生态期望的版本（或缺少对应子模块）。请确认安装了正确的 `open-dataflow` 及其依赖来源（内部/私有源/特定版本）。

### 3) `OSError: libmpi.so.*: cannot open shared object file`

通常由某些环境下 `torch` 触发的系统库缺失导致。可通过安装系统 MPI 运行库、或在无 MPI 环境中使用不依赖该链路的功能来规避。

---

## License

Apache-2.0，见 `LICENSE`。
