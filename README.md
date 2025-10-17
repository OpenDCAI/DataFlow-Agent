# DataFlow-Agent 目录结构说明

下面对本仓库的核心目录 / 文件做简要中文说明，帮助新同事快速了解各模块用途及放置内容。  
（括号内为常见文件类型，仅作示例）

| 级别 | 路径 | 主要内容 | 作用 |
| ---- | ---- | -------- | ---- |
| 根 | `LICENSE` | - | 开源协议（Apache-2.0）。 |
| 根 | `README.md` | - | 项目总览与快速上手。 |
| 根 | `pyproject.toml` | - | Python 包元数据、入口脚本、依赖声明。 |
| 根 | `requirements.txt` | txt | 运行时依赖列表。 |
| 根 | `requirements-dev.txt` | txt | 开发 / 测试 / 格式化工具依赖。 |
| 根 | `docs/` | md, png | MkDocs/Sphinx 源文件，存放详细文档。 |
| 根 | `static/` | png, gif | Logo、流程图、演示 GIF 等静态资源。 |
| 根 | `gradio_app/` | py, css | Gradio Web UI（`dataflow_agent webui`）相关代码。 |
| 根 | `script/` | py, sh | 常用启动脚本、批处理脚本、Docker 等。 |
| 根 | `tests/` | py | PyTest 单元 / 集成测试。 |
| 包 | `dataflow_agent/` | 见下表 | Python 主包，所有业务代码。 |

## `dataflow_agent/` 细分目录

| 子目录 / 文件 | 说明 |
| --------------- | ---- |
| `__init__.py`、`state.py`、`utils.py` | 包初始化、全局状态管理、通用工具函数。 |
| `agentroles/` | GPT-style「角色」实现：<br>• `base_agent.py` 抽象基类<br>• `classifier.py`、`rewriter.py`… 等业务角色<br>• `resources/` 默认提示词模板（JSON）。 |
| `graghbuilder/` | 负责把推荐的 Operator / Pipeline 组装成有向图。`gragh_builder.py` 暴露核心接口。 |
| `promptstemplates/` | 通用 Prompt 引擎：<br>• `prompt_template.py` 模板基类<br>• `prompts_repo.py` 模板仓库<br>• `resources/` 内置 JSON 模板。 |
| `storage/` | 统一的落盘 / KV / 向量存储封装，`storage_service.py` 提供读写接口。 |
| `toolkits/` | 各类工具集：<br>• `basetool/` 文件、LLM 客户端等<br>• `optool/` 与 DataFlow Operator 交互<br>• `pipetool/` Pipeline 相关<br>• `dockertool/` 生成轻量化 Docker 镜像<br>• `tool_manager.py` 动态加载 & 管理。 |
| `workflow/` | 端到端工作流封装：推荐、写入、精修等；通常直接被脚本调用。 |
| `resources/taskinfo.yaml` | 任务级配置示例（例如模型、数据路径）。 |
| `tmps/` | 运行时中间产物；建议仅保留示例文件并在 `.gitignore` 中排除临时生成物。 |

## `script/` 常见脚本

| 脚本 | 用途 |
| ---- | ---- |
| `run_dfa_operator_write.py` | 调用 *Agent-Writer* 生成自定义 Operator。 |
| `run_dfa_pipeline_rec.py`、`run_pipeline_recommend.py` | 调用 *Pipeline Recommender*。 |
| `run_dfa_pipeline_refine.py` | 基于已有 Pipeline 进行增量优化。 |
| `run_web_collection.py` | Web 数据采集示例。 |
| `*.sh` | 一键运行 / Demo 用 Bash 脚本。 |

## `tests/` 说明

| 文件 | 作用 |
| ---- | ---- |
<!-- | `test_smoke.py` | 整体冒烟测试，确保包可 import、核心 CLI 可运行。 | -->
| `test_classifier.py` | 分类角色逻辑单测。 |
| `test_pipelinebuild_gragh.py` | Graph 结构正确性验证。 |
| `test_recommender.py` | Pipeline 推荐准确性 / 召回率测试。 |

> 提示：运行 `pytest -q --cov=dataflow_agent` 可查看整体测试覆盖率。

---

如需新增模块，请优先考虑是否已有对应目录，保持包结构扁平且语义清晰；同时记得补充单元测试与文档。