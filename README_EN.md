<div align="center">

<img src="static/new_logo_bgrm.png" alt="DataFlow-Agent Logo" width="180"/>

# DataFlow-Agent

A state-driven, modular AI agent framework with an extensible `Agent / Workflow / Tool` system. It ships with a CLI scaffold and a multi-page Gradio UI for “dataflow/operator orchestration” tasks such as operator recommendation, pipeline generation/debugging, operator QA, and web collection.

[![DataFlow](https://img.shields.io/badge/DataFlow-OpenDCAI%2FDataFlow-0F9D58?style=flat-square&logo=github&logoColor=white)](https://github.com/OpenDCAI/DataFlow)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-2F80ED?style=flat-square&logo=apache&logoColor=white)](LICENSE)
[![GitHub Repo](https://img.shields.io/badge/GitHub-OpenDCAI%2FDataFlow--Agent-24292F?style=flat-square&logo=github&logoColor=white)](https://github.com/OpenDCAI/DataFlow-Agent)
[![Stars](https://img.shields.io/github/stars/OpenDCAI/DataFlow-Agent?style=flat-square&logo=github&label=Stars&color=F2C94C)](https://github.com/OpenDCAI/DataFlow-Agent/stargazers)

[中文](README_ori.md) | English

<a href="#-quickstart">
  <img alt="Quickstart" src="https://img.shields.io/badge/🚀-Quickstart-2F80ED?style=for-the-badge" />
</a>
<a href="docs/">
  <img alt="Docs" src="https://img.shields.io/badge/📚-Docs-2D9CDB?style=for-the-badge" />
</a>
<a href="docs/contributing.md">
  <img alt="Contributing" src="https://img.shields.io/badge/🤝-Contributing-27AE60?style=for-the-badge" />
</a>

</div>

---

## Contents

- [Highlights](#highlights)
- [Feature Overview](#feature-overview)
- [Feature Details](#feature-details)
- [🚀 Quickstart](#-quickstart)
- [Launch UI (Gradio)](#launch-ui-gradio)
- [CLI](#cli)
- [Workflows](#workflows)
- [Configuration](#configuration)
- [Docs (MkDocs)](#docs-mkdocs)
- [Project Layout](#project-layout)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Community](#community)

---

## Highlights

- **Unified state model**: drive multi-agent execution around `MainState / DFState` and other state objects.
- **Pluggable agents**: discover/load agents via registration for flexible composition.
- **Workflow orchestration**: graph-based workflows (GraphBuilder) for complex flows and tool calls.
- **Tool management**: inject pre-tools/post-tools via `ToolManager` to manage tools and boundaries.
- **Multi-page UI**: `gradio_app/` pages for operator/pipeline/prompt/web collection capabilities.
- **CLI scaffolding**: `dfa create` generates workflow/agent/gradio page/prompt/state templates.

---

## Feature Overview

The Gradio “DataFlow Agent Platform” includes 6 pages (screenshots in `static/dfa/`):

- [PromptAgent Frontend](#promptagent-frontend): generate/iterate operator Prompt Templates.
- [Op Assemble Line](#op-assemble-line): pick operators and assemble a runnable pipeline.
- [Operator QA](#operator-qa): Q&A assistant for operators/tools.
- [Operator Write](#operator-write): generate custom operator code and debug in-page.
- [Pipeline Rec](#pipeline-rec): generate pipelines from tasks and refine them.
- [Web Collection](#web-collection): collect web data and transform it into structured outputs.

---

## Feature Details

### PromptAgent Frontend

Generate and refine “operator Prompt Templates” by reusing existing operators:

- Inputs: task description, operator name (`op-name`), parameter list, output format (optional)
- Outputs: reusable Prompt Templates and rewrite suggestions (ready to be saved into a prompt repo)

<div align="center">
  <img src="static/dfa/PromptAgent.png" width="92%" alt="PromptAgent Frontend"/>
</div>

---

### Op Assemble Line

Select operators from the operator library, assemble a runnable pipeline, and debug it:

- Choose operator category and operator
- Configure operator params (JSON) and append to the pipeline queue
- Run the pipeline end-to-end for validation

<div align="center">
  <img src="static/dfa/OpAssemble.png" width="92%" alt="Op Assemble Line"/>
</div>

---

### Operator QA

A Q&A assistant for operators/tools to quickly answer “how to use / what to use / what to watch out for”:

- Recommend relevant operators based on your task
- Explain operator inputs/outputs and key parameters
- Provide examples and reusable snippets

<div align="center">
  <img src="static/dfa/OpQA.png" width="92%" alt="Operator QA"/>
</div>

---

### Operator Write

Generate DataFlow operator code from requirements and close the loop with testing & debugging:

- Code generation: implement operators from goals/constraints
- Operator matching: align to existing operator conventions to land in the operator library
- Execution & debug: inspect results, debug info, and logs

<div align="center">
  <img src="static/dfa/OpWrite.png" width="92%" alt="Operator Write"/>
</div>

---

### Pipeline Rec

Generate runnable pipelines (code/JSON) from tasks and refine them in multiple rounds:

- Generate: map natural language tasks to operator sequences
- Refine: iterate on an existing pipeline
- Artifacts: pipeline code / pipeline JSON / execution logs

<div align="center">
  <img src="static/dfa/PipelineRecRefine.png" width="92%" alt="Pipeline Rec & Refine"/>
</div>

---

### Web Collection

Collect web data and convert it into structured outputs for “data production → governance/training” workflows:

- Configure target, data type, and scale
- Collect and export structured results
- View execution logs and result summary

<div align="center">
  <img src="static/dfa/WebCollection.png" width="92%" alt="Web Collection"/>
</div>

---

## 🚀 Quickstart

### 1) Clone

```bash
git clone https://github.com/OpenDCAI/DataFlow-Agent
cd DataFlow-Agent
```

### 2) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
```

### 3) Install dependencies

For local development:

```bash
pip install -r requirements-dev.txt
pip install -e .
```

For minimal/production install:

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Launch UI (Gradio)

Load the dataflow-focused page set (recommended):

```bash
python gradio_app/app.py --page_set data
```

Pages (6):

- PromptAgent Frontend / Op Assemble Line / Operator QA / Operator Write / Pipeline Rec / Web Collection

Default server:

- Port: `GRADIO_SERVER_PORT` env var or `--server_port` (default `7860`)
- Host: `GRADIO_SERVER_NAME` (default `0.0.0.0`)

---

## CLI

CLI help:

```bash
dfa --help
```

Common scaffolding commands:

```bash
dfa create --agent_name my_agent
dfa create --wf_name my_workflow
dfa create --gradio_name my_page
dfa create --prompt_name my_prompt
dfa create --state_name my_state
```

Generated file locations (convention):

- Workflow: `dataflow_agent/workflow/wf_<name>.py`
- Agent: `dataflow_agent/agentroles/common_agents/<name>_agent.py`
- Gradio Page: `gradio_app/pages/page_<name>.py`
- Prompt Template: `dataflow_agent/promptstemplates/resources/pt_<name>_repo.py`
- State: `dataflow_agent/states/<name>_state.py`

---

## Workflows

Workflows live in `dataflow_agent/workflow/` and follow the `wf_*.py` naming convention. On startup, the system tries to import and register workflows; if a workflow requires missing external deps/env, it will be skipped with a log message.

List successfully registered workflows:

```bash
python - <<'PY'
from dataflow_agent.workflow import list_workflows
print(sorted(list_workflows()))
PY
```

Run an example workflow (`run_workflow`):

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

---

## Configuration

### LLM

- `DF_API_URL`: LLM API base URL (default `test`)
- `DF_API_KEY`: API key (default `test`)
- `DATAFLOW_LOG_LEVEL`: log level (default `INFO`)
- `DATAFLOW_LOG_FILE`: log file (default `dataflow_agent.log`)

### Paths (optional)

`dataflow_agent/state.py` tries to use `dataflow.cli_funcs.paths.DataFlowPath`; if unavailable, it falls back to env vars:

- `DATAFLOW_DIR`: data directory root (defaults to repo root)
- `DATAFLOW_STATICS_DIR`: statics dir (default `./statics`)

---

## Docs (MkDocs)

Run local docs:

```bash
mkdocs serve
```

Config: `mkdocs.yml`

---

## Project Layout

```
DataFlow-Agent/
├── dataflow_agent/          # core package
├── gradio_app/              # Gradio UI
├── docs/                    # docs
├── static/                  # README images, etc.
├── script/                  # scripts
└── tests/                   # tests
```

---

## Roadmap

| Area | Status | Items |
| --- | --- | --- |
| 🔄 Easy-DataFlow (data governance pipeline) | ✅ Done | Pipeline rec / Operator write / Visual orchestration / Prompt optimization / Web collection |
| 🎨 Workflow visual editor (drag & drop) | 🚧 In progress | Drag UI / 5 agent modes / 20+ preset nodes |
| 💾 Trace export (training data export) | 🚧 In progress | JSON/JSONL / SFT / DPO |

---

## Contributing

Contributions are welcome:

- Bugs & feature requests: https://github.com/OpenDCAI/DataFlow-Agent/issues
- Discussions: https://github.com/OpenDCAI/DataFlow-Agent/discussions
- Pull requests: https://github.com/OpenDCAI/DataFlow-Agent/pulls
- Guide: `docs/contributing.md`

---

## License

Apache-2.0. See `LICENSE`.

---

## Community

- GitHub Issues: https://github.com/OpenDCAI/DataFlow-Agent/issues
- GitHub Pull Requests: https://github.com/OpenDCAI/DataFlow-Agent/pulls
- Community chat: connect with contributors and maintainers

<div align="center">
  <img src="static/team_wechat.png" alt="DataFlow-Agent WeChat Group" width="560"/>
  <br>
  <sub>Scan to join the DataFlow-Agent WeChat group</sub>
</div>
