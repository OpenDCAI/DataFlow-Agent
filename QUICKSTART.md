# Quickstart

**English** | [简体中文](QUICKSTART.zh-CN.md) · [← README](README.md)

Everything needed to install `dataflow-mm-agent`, point it at a model, run a
first rollout, and use the bundled tools.

## Contents

- [Install](#install)
- [Configure a model backend](#configure-a-model-backend)
- [First rollout](#first-rollout)
- [Multimodal tasks](#multimodal-tasks)
- [Pipeline stages](#pipeline-stages)
- [Runtime settings](#runtime-settings)
- [Export training data](#export-training-data)
- [Read a trajectory](#read-a-trajectory)
- [Development install and tests](#development-install-and-tests)

## Install

Requirements:

- Conda, either through Miniconda or Anaconda
- Internet access during installation so that Python dependencies can be resolved

Concrete visual Envs may have additional browser, rendering, game, or office
dependencies; those belong to the Env integration rather than this core package.

1. On the repository's `mm-agent` branch page, choose **Code → Download ZIP**.
2. Extract the archive and open a terminal in the extracted directory—the one
   containing `pyproject.toml`.
3. Create and activate the recommended Conda environment:

```bash
conda create -n dataflow-mm-agent python=3.12 pip -y
conda activate dataflow-mm-agent
```

4. Upgrade the packaging tools and install the extracted package:

```bash
python -m pip install --upgrade pip
python -m pip install .
```

5. Verify the installation:

```bash
python -c "import dataflow_mm_agent as d; print(d.__version__)"
```

The command should print the installed package version. To upgrade later,
download the new ZIP, extract it, activate the same Conda environment, and run
`python -m pip install --upgrade .` from the new directory.

## Configure a model backend

Installation itself does not require an API key; configure your model endpoint
before a live rollout. `create_model_serving_from_env()` reads configuration
from the process environment.

For an OpenAI-compatible endpoint:

```bash
export SERVING_BACKEND=openai
export MODEL=your-model-name
export API_URL=https://your-endpoint.example/v1
export DF_API_KEY=your-api-key
```

On Windows PowerShell, set the same values with `$env:`, for example:

```powershell
$env:SERVING_BACKEND = "openai"
$env:MODEL = "your-model-name"
$env:API_URL = "https://your-endpoint.example/v1"
$env:DF_API_KEY = "your-api-key"
```

Set these variables through your shell or secret manager and never commit their
values. Gemini is also supported; see the
[serving factory](dataflow_mm_agent/serving/serving_factory.py) for its
configuration.

## First rollout

This example assumes an external Env pack has registered `my_visual_env`;
replace that placeholder with your registered Env ID (see
[Lightweight Env design](README.md#lightweight-env-design)).

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

`Task` is reusable: one task may produce many trajectories. Its `messages` may
contain text and any number of images. `Scenario` is optional private runtime
input, not a mandatory wrapper around every task. `judge_ref` is an optional
public score range plus task-specific criteria; when omitted, Judge uses its
generic rubric.

## Multimodal tasks

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

Images remain first-class content blocks through rollout, Refine, Judge, and
trajectory storage. They are not converted into text placeholders.

Materialized JSON task stores may keep source documents outside the JSON body
with confined, SHA-256-pinned `text_ref` blocks (`text/plain` or
`text/markdown`, UTF-8, at most 512 KiB). The store resolves them to ordinary
`TextContent` before rollout, just as `image_ref` resolves to inline
`ImageContent`; unresolved paths never reach the model.

## Pipeline stages

The operators follow DataFlow's composable-operator style while keeping
generation, replay, and quality evaluation as separate concerns.

**Generate** runs the shared multimodal tool loop and records the unscored
trajectory. With `checkpoint_dir`, each finished row is saved immediately as
`<sample key>.jsonl`; rerunning skips rows that already have a
non-infrastructure result, `max_retries` re-runs rows that raised or ended in
`infrastructure_error`, and `run_manifest.jsonl` records each run's counts,
configuration, and summed model usage. Every trajectory records its provider
token usage in `metadata.usage` when the serving adapter reports it.

**ReplayVerify** replays stored actions in a fresh Env and, when configured,
evaluates an independent deterministic verifier.

**Judge** resolves the Task's optional `judge_ref` (or injects the generic
fallback), scores every configured criterion, and computes `traj_overall` as the
arithmetic mean of range-normalized scores. Task-specific grading rules live
only in the task rubric, and Judge does not replace exact state verification.
Rubrics over 16,000 serialized characters are evaluated one criterion at a
time—with the complete task rubric still injected into every shard—and
malformed combined verdicts fall back to the same all-or-nothing shard path. A
ReplayVerify result is injected only when the caller supplies one. Besides the
scores, Judge returns repair evidence for the next stage: `important_steps` (the
step numbers that went wrong) and `refine_suggestion` (what to change), both
empty when it sees no problem.

**Refine** receives the original task messages, the failure diagnosis, and the
evidence Judge and ReplayVerify produced, then explores again in a fresh Env
rather than mutating the old trajectory. Its repair context contains the
deterministic verifier's failed checks, the reviewer's repair suggestion, and
every step Judge flagged together with `important_step_window` neighbours,
rendered in full with their observations and images. The step summary of the
rest of the attempt keeps the newest steps and elides the middle.

**Filter and Select** keep the trajectories that meet the pipeline's quality and
diversity requirements. `AgentMMTrajectorySelector` keeps a trajectory only when
every condition you pass holds. All switches are optional: built-in features
(`num_steps`, `num_tool_calls`, `num_tool_errors`, `avg_observation_len`,
`is_finish`, `replay_passed`, `judge_score`, ...) and features you register with
`register_selector_feature(name, fn)`, where `fn(trajectory, row)` may read the
trajectory or its storage row. A condition is a value (`is_finish=True`) or
comparisons (`num_steps={"gte": 2}`). Optional `sort_by`, `group_by`,
`dedupe_threshold`, and `max_selected` rank and cap the result. Weighted scoring
is a custom feature built from the public functions in
`operators.selector_features`:

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

Open-ended authoring tasks do not need a pretend verifier. Their ReplayVerify
status is `not_applicable`, while Judge evaluates the rendered result and the
process that produced it.

## Runtime settings

**Request context.** `RolloutConfig(context_policy=ContextPolicy(...))` decides
how much recorded history each live request carries. It is off by default (every
request sends the whole history). A policy keeps the system prompt, the task
messages, and the newest `keep_last_steps` turns — each turn complete, so the
model still sees the arguments it used — plus the newest image observation, and
replaces the rest with one note saying how many steps were elided. An optional
`max_prefix_chars` budget and a `summarizer` hook (for model-written summaries)
are available. Trimming applies to the request only: the trajectory always
records everything, so replay, export, and review are unaffected.

```python
from dataflow_mm_agent.runtime_components import ContextPolicy, RolloutConfig

config = RolloutConfig(max_steps=64, context_policy=ContextPolicy(keep_last_steps=3))
```

**Prompt language.** Built-in prompts ship in English and Chinese
(`dataflow_mm_agent/prompts.py`). Pass `language="zh"` to `RolloutConfig`,
`AgentMMTrajectoryQualityEvaluator`, or `AgentMMTrajectoryRefiner` to switch the
system prompt, the generic rubric text, and the repair templates; an explicit
`system_prompt=` always wins. Task language still comes from the Task messages.

## Export training data

Trajectories convert to [ms-swift](https://github.com/modelscope/ms-swift)
`messages` JSONL for supervised fine-tuning. The export is a format conversion
only: it does not filter by verification or Judge results.

```bash
dataflow-mm-export-swift trajectories/*.jsonl -o sft/train.jsonl
# or: python -m dataflow_mm_agent.export trajectories/*.jsonl -o sft/train.jsonl
```

Each trajectory becomes one row. System, user, and assistant messages keep the
recorded text (the assistant text is the model's raw action response);
observations that answer an assistant turn become `tool_response`, and images
become `<image>` tags listed in order under `images`. Images are written once to
`sft/train_images/` and referenced by absolute path, or inlined with
`--image-mode base64`. Inputs may be trajectory JSON, `TrajectoryStore` JSONL,
or pipeline JSONL with a `trajectory` column. Messages after the last assistant
turn are dropped, and a trajectory with no assistant turn is skipped.

## Read a trajectory

Export any trajectory, pipeline JSONL, or rollout directory as one offline HTML
page with every action, observation, and image:

```bash
dataflow-mm-trajectory-html trajectory.json -o report.html
# or: python -m dataflow_mm_agent.visualization trajectory.json -o report.html
```

See the [viewer documentation](dataflow_mm_agent/visualization/README.md) for
pipeline columns, Judge fields, and the evidence it deliberately separates.

## Development install and tests

If you plan to edit the source, install it in editable mode with the test extra:

```bash
python -m pip install -e ".[test]"
```

A remote MCP adapter can remain small because its tools and integration-specific
dependencies run in the upstream MCP server. The bundled
[`create-env` workspace skill](dataflow_mm_agent/skills/create-env/SKILL.md)
documents the adapter workflow and validation requirements.
