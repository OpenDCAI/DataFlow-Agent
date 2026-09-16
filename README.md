# DataFlow-MM-Agent

**English** | [简体中文](README.zh-CN.md)

`dataflow-mm-agent` lets multimodal agents interact with visual environments
and returns every run as a structured `Trajectory`. It can be used to validate
agent–environment interactions and to synthesize image-grounded trajectory data
for evaluation, supervised fine-tuning, and reinforcement learning. The current
canonical content types are text and image; the contracts are designed so that
additional modalities can be introduced later without making every Env stateful.

Python package: `dataflow_mm_agent` · Python `>=3.10` · Apache-2.0

<table>
  <tr>
    <td align="center" width="33%" valign="top">
      <a href="examples/showcases/01_geometry_proof.md"><img src="examples/showcases/assets/geometry_proof/trajectory.gif" height="180" alt="Agent progressively constructing an olympiad geometry proof"></a><br>
      <sub>Mathematical reasoning · geometry proofs</sub>
    </td>
    <td align="center" width="33%" valign="top">
      <a href="examples/showcases/02_pixel_game.md"><img src="examples/showcases/assets/pixel_game/trajectory.gif" height="180" alt="Agent collecting five gems in a Pyxel game"></a><br>
      <sub>Pyxel game · visual planning</sub>
    </td>
    <td align="center" width="33%" valign="top">
      <a href="examples/showcases/03_pptx.md"><img src="examples/showcases/assets/pptx/trajectory.gif" height="180" alt="Agent recreating a reference deck as an editable PowerPoint"></a><br>
      <sub>PowerPoint · editable reconstruction</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%" valign="top">
      <a href="examples/showcases/05_mobile_weekly_alarms.md"><img src="examples/showcases/assets/mobile_weekly_alarms/trajectory.gif" height="180" alt="Agent setting weekly alarms on an Android phone"></a><br>
      <sub>Mobile use · Android alarms</sub>
    </td>
    <td align="center" width="33%" valign="top">
      <a href="examples/showcases/07_blender_lighthouse.md"><img src="examples/showcases/assets/blender_lighthouse/trajectory.gif" height="180" alt="Agent building a low-poly island lighthouse in Blender"></a><br>
      <sub>Blender · 3D scene construction</sub>
    </td>
    <td align="center" width="33%" valign="top">
      <a href="examples/showcases/08_vlmgym_2048.md"><img src="examples/showcases/assets/vlmgym_2048/trajectory.gif" height="180" alt="Agent reading a 2048 board and merging tiles to reach 64"></a><br>
      <sub>2048 · planning from pixels</sub>
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
8. **Visual game planning: 2048** —
   [read tile values from G1 VLM-Gym frames and build a 64 tile from a fresh board](examples/showcases/08_vlmgym_2048.md).
9. **Visual path planning: Shisen-Sho** —
   [connect identical tiles on G1's 12x12 board with at most two turns](examples/showcases/09_vlmgym_shisensho.md).
10. **Class-level visual matching** —
   [clear a Shisen-Sho board whose tiles are CIFAR-10 photos](examples/showcases/10_vlmgym_shisensho_cifar10.md).
11. **Match-3 with cascades** —
   [reach 150 points on G1's Swap board through reshuffles](examples/showcases/11_vlmgym_swap.md).
12. **Why a deterministic verifier is necessary** —
   [inspect a trajectory that received Judge 1.0 but failed exact state verification](examples/showcases/12_why_deterministic_verifier.md).

The showcase pages use GitHub-native Markdown, full-trajectory GIF previews,
ordinary image assets under every corresponding tool step, and compact JSON.
They do not require JavaScript or embed images as base64 inside a large HTML file. See the
[showcase index](examples/showcases/README.md) for artifacts and run metadata.

## Quickstart

```bash
conda create -n dataflow-mm-agent python=3.12 pip -y
conda activate dataflow-mm-agent
python -m pip install .
```

Then point the package at a model endpoint and run a first rollout:

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

[**QUICKSTART**](QUICKSTART.md) covers installation, model backend configuration,
multimodal tasks, the pipeline stages (Generate, ReplayVerify, Judge, Refine,
Filter/Select), runtime settings, training-data export, the offline trajectory
viewer, and the development install.

## How it works

A `Task` names an Env and carries the messages the model sees. `AgentRollout`
creates a fresh Env, runs one tool loop, and records every action and
observation as a `Trajectory`. Operators then compose around that record:

| Stage | What it does |
| --- | --- |
| Generate | Runs the tool loop and records unscored trajectories; resumable per sample. |
| ReplayVerify | Replays stored actions in a fresh Env and runs the task's deterministic verifier. |
| Judge | Scores the task rubric from the real observations and reports which steps failed. |
| Refine | Re-explores with the verifier findings, the reviewer suggestion, and the flagged steps. |
| Filter / Select | Keeps trajectories meeting declarative quality and diversity conditions. |
| Export | Converts trajectories to ms-swift `messages` JSONL for supervised fine-tuning. |

Judge and ReplayVerify answer different questions: one reviews the process, the
other reproduces the actions and checks exact state. Open-ended authoring tasks
report `not_applicable` for replay rather than pretending to have a verifier.

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
│   ├── runtime_components/ # rollout, tool loop, context policy, ReplayVerify
│   ├── operators/          # Generate, Judge, Refine, Filter, Select
│   ├── export/             # ms-swift training-data exporter
│   ├── prompts.py          # built-in English and Chinese prompt text
│   ├── serving/            # OpenAI-compatible and Gemini multimodal serving
│   ├── visualization/      # offline trajectory HTML exporter and viewer
│   ├── skills/create-env/  # workspace skill for Env and MCP adoption
│   └── storage/            # task and trajectory stores
├── examples/showcases/     # GitHub-native trajectory walkthroughs
├── QUICKSTART.md
├── LICENSE
└── pyproject.toml
```

Concrete Envs are outside the core distribution so installing one integration
does not force every rendering or game dependency into `dataflow-mm-agent`.
An integration may use the package's process proxy when it needs a dedicated
interpreter or dependency boundary.

## Further reading

- [Quickstart: install, configure, and run](QUICKSTART.md)
- [Showcase index](examples/showcases/README.md)
- [Offline trajectory HTML reports](dataflow_mm_agent/visualization/README.md)
- [Create an Env or MCP adapter](dataflow_mm_agent/skills/create-env/SKILL.md)
- [Env contracts and package layout](dataflow_mm_agent/skills/create-env/references/contracts-and-layout.md)
- [Task generation](dataflow_mm_agent/skills/create-env/references/task-generation.md)
- [Validation strategy](dataflow_mm_agent/skills/create-env/references/validation.md)
