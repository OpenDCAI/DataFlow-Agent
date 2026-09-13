# DataFlow-MM-Agent

**English** | [简体中文](README.zh-CN.md)

`dataflow-mm-agent` lets multimodal agents interact with visual environments
and returns every run as a structured `Trajectory`. It can be used to validate
agent–environment interactions and to synthesize image-grounded trajectory data
for evaluation, supervised fine-tuning, and reinforcement learning. The current
canonical content types are text and image; the contracts are designed so that
additional modalities can be introduced later without making every Env stateful.

This is an extension package built on top of **DataFlow-MM**. Its
`open-dataflow-mm` dependency is declared by the package and installed
automatically by `pip`.

Python package: `dataflow_mm_agent` · Python `>=3.10` · Apache-2.0

<table>
  <tr>
    <td align="center" width="50%">
      <a href="examples/showcases/01_geometry_proof.md"><img src="examples/showcases/assets/geometry_proof/trajectory.gif" alt="Agent progressively constructing an olympiad geometry proof"></a><br>
      <sub>Constructing and proving an olympiad geometry problem</sub>
    </td>
    <td align="center" width="50%">
      <a href="examples/showcases/02_pixel_game.md"><img src="examples/showcases/assets/pixel_game/trajectory.gif" alt="Agent collecting five gems in a visual grid game"></a><br>
      <sub>Collecting five gems under a deterministic move budget</sub>
    </td>
  </tr>
</table>

## What can this package do?

1. **Image-grounded mathematical reasoning** —
   [watch an agent construct and prove an olympiad geometry problem](examples/showcases/01_geometry_proof.md).
2. **Visual planning in planar games** —
   [follow a Pyxel agent collecting five gems under a move budget](examples/showcases/02_pixel_game.md).
3. **Editable visual reconstruction** —
   [recreate a three-page reference deck as an editable PowerPoint](examples/showcases/03_pptx.md).
4. **Document-to-diagram synthesis** —
   [turn two incident-runbook pages into an editable operational flow](examples/showcases/04_diagram.md).
5. **Mobile UI automation** —
   [set weekday, weekend, and reading alarms on an isolated Android device](examples/showcases/05_mobile_weekly_alarms.md).
6. **Browser-based planning and form interaction** —
   [save an activity-day schedule under time and budget constraints](examples/showcases/06_playwright_studio_day.md).
7. **3D scene construction** —
   [build a low-poly island lighthouse in Blender, with the step-limit outcome preserved](examples/showcases/07_blender_lighthouse.md).
8. **Why a deterministic verifier is necessary** —
   [inspect a trajectory that received Judge 1.0 but failed exact state verification](examples/showcases/08_why_deterministic_verifier.md).

The showcase pages use GitHub-native Markdown, full-trajectory GIF previews,
ordinary image assets under every corresponding tool step, and compact JSON.
They do not require JavaScript or embed images as base64 inside a large HTML file. See the
[showcase index](examples/showcases/README.md) for artifacts and run metadata.

## Installation

### Requirements

- Conda, either through Miniconda or Anaconda
- Internet access during installation so that Python dependencies can be
  resolved

Concrete visual Envs may have additional browser, rendering, game, or office
dependencies; those belong to the Env integration rather than this core
package.

### Recommended: install from a downloaded ZIP

1. On the GitHub repository page, choose **Code → Download ZIP**.
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

`pip` installs `dataflow-mm-agent`, its DataFlow-MM base package
(`open-dataflow-mm`), and the other declared Python dependencies automatically.

5. Verify the installation:

```bash
python -c "import dataflow_mm_agent as d; print(d.__version__)"
```

The command should print the installed package version. To upgrade later,
download the new ZIP, extract it, activate the same Conda environment, and run
`python -m pip install --upgrade .` from the new directory.

### Configure a model backend

Installation itself does not require an API key. A live rollout does. The
`create_model_serving_from_env()` helper reads configuration from the process
environment.

For an OpenAI-compatible endpoint:

```bash
export SERVING_BACKEND=openai
export MODEL=your-model-name
export API_URL=https://your-endpoint.example/v1
export DF_API_KEY=your-api-key
```

For the Gemini API:

```bash
export SERVING_BACKEND=gemini
export MODEL=your-gemini-model
export GEMINI_API_KEY=your-api-key
```

On Windows PowerShell, set the same values with `$env:`, for example:

```powershell
$env:SERVING_BACKEND = "gemini"
$env:MODEL = "your-gemini-model"
$env:GEMINI_API_KEY = "your-api-key"
```

Set these variables through your shell or secret manager and never commit their
values. `API_URL` is optional for Gemini and defaults to Google's Generative
Language API.

### Development install

If you plan to edit the source, install it in editable mode with the test extra:

```bash
python -m pip install -e ".[test]"
```

A remote MCP adapter can remain small because its tools and integration-specific
dependencies run in the upstream MCP server.

## Minimal rollout

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

## The trajectory data flow

DataFlow-MM-Agent follows DataFlow's composable-operator style while keeping
generation, replay, and quality evaluation as separate concerns:


- **Generate** runs the shared multimodal tool loop and records the unscored
  trajectory.
- **ReplayVerify** replays stored actions in a fresh Env and, when configured,
  evaluates an independent deterministic verifier.
- **Judge** resolves the Task's optional `judge_ref` (or injects the generic
  fallback), scores every configured criterion, and computes `traj_overall` as
  the arithmetic mean of range-normalized scores. Every environment uses the
  same rationale-and-scores response; task-specific grading rules live only
  in the task rubric. Judge does not replace exact state verification. Rubrics over 16,000 serialized characters are evaluated one
  criterion at a time—with the complete task rubric still injected into every
  shard—and malformed combined verdicts fall back to the same all-or-nothing
  shard path.
- **Refine** receives the original task messages, visual observations, and
  failure diagnosis, then produces a new trajectory rather than mutating the old
  one. For stateful visual artifacts it can first replay the recorded pre-finish
  actions in a fresh workspace, append the newest diagnosis after restoration,
  and ask the model only for localized continuation edits.
- **Filter and Select** keep the trajectories that meet the pipeline's quality
  and diversity requirements.

Open-ended authoring tasks do not need a pretend verifier. Their ReplayVerify
status is `not_applicable`, while Judge evaluates the rendered result and the
process that produced it.

## Lightweight Env design

An Env needs only a tool catalog and a dispatcher:

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

That is the complete mandatory surface:

```python
def tools(self) -> Sequence[ToolSpec]: ...
def call(self, tool_name: str, args: Mapping[str, Any]) -> ToolResult: ...
```

Stateful Envs may additionally expose `start(init, workspace)` and `close()`.
They do not need to implement a task provider, Scenario, snapshot, or verifier.
The runner supplies `finish`; an Env must not register its own finish tool.

Rollout and replay use the same startup order:

```text
create Env -> optional start(init, workspace) -> tools() -> tool loop -> close()
```

If `start` is absent, it is skipped. If it is present, `tools()` only needs to
work after startup succeeds. The runtime reads the catalog once, before the
first model decision or replayed action, and keeps it fixed for that episode.
Startup failures stop discovery and execution; cleanup still runs when
available. Model messages retain their order: system prompt, task messages,
then any initial observation.

When constructing `ToolLoop(env)` directly, complete optional startup first;
the constructor reads `env.tools()` and does not start the Env itself.

### MCP adoption

An MCP server can be attached through a thin adapter:

1. map `list_tools()` results to `ToolSpec`;
2. map `call_tool()` content and errors to `ToolResult`;
3. register the adapter factory with `register_env`.

No framework-specific task/verifier bundle is required. A stateless MCP adapter
can implement only `tools()` and `call()`; session startup and cleanup can use
the optional lifecycle hooks when needed. For a session-based MCP, connect and
complete the MCP handshake in `start()`, then let `tools()` discover the fixed
catalog through that same session. Keep the session for `call()` and close it
in `close()`. No pre-generated catalog or separate discovery session is
required. MCP SDK and transport details remain in the Env package.

The bundled
[`create-env` workspace skill](dataflow_mm_agent/skills/create-env/SKILL.md)
documents the adapter workflow and validation requirements.

## Core contracts

```text
Task ──> AgentRollout ──> Trajectory
 │           │
 │           └── fresh Env selected by task.env_id
 │
 └── optional Scenario(init)

Trajectory + TaskResolver + optional VerifierResolver
                              └──> ReplayVerify ──> ReplayVerification
```

- A runner always receives a `Task`; a Task and its trajectories have a
  one-to-many relationship.
- `Scenario` exists only when private initialization data must enter a fresh Env.
- The registry owns Env factories and solver-facing metadata, not tasks.
- Verification is resolved independently and never forces a Scenario.
- `Trajectory` contains actions and observations, not a verifier score or
  private Scenario data.

## Repository layout

```text
dataflow-mm-agent/
├── dataflow_mm_agent/
│   ├── contracts/          # Task, Env, messages, tools, trajectory
│   ├── env/                # registry, plugins, process-isolated adapters
│   ├── runtime_components/ # rollout, tool loop, finish, ReplayVerify
│   ├── operators/          # Generate, Judge, Refine, Filter, Select
│   ├── serving/            # OpenAI-compatible and Gemini multimodal serving
│   ├── visualization/      # offline trajectory HTML exporter and viewer
│   ├── skills/create-env/  # workspace skill for Env and MCP adoption
│   └── storage/            # task and trajectory stores
├── examples/showcases/     # GitHub-native trajectory walkthroughs
├── LICENSE
└── pyproject.toml
```

Concrete Envs are outside the core distribution so installing one integration
does not force every rendering or game dependency into `dataflow-mm-agent`.
An integration may use the package's process proxy when it needs a dedicated
interpreter or dependency boundary.

## Further reading

- [Offline trajectory HTML reports](dataflow_mm_agent/visualization/README.md)
- [Create an Env or MCP adapter](dataflow_mm_agent/skills/create-env/SKILL.md)
- [Env contracts and package layout](dataflow_mm_agent/skills/create-env/references/contracts-and-layout.md)
- [Task generation](dataflow_mm_agent/skills/create-env/references/task-generation.md)
- [Validation strategy](dataflow_mm_agent/skills/create-env/references/validation.md)
- [Showcase index](examples/showcases/README.md)
