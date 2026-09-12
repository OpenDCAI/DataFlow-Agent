# Agent-MM Env contracts and layout v2

## Recommended package

```text
my-agent-mm-env/
├── pyproject.toml
├── README.md
├── src/my_agent_mm_env/
│   ├── __init__.py          # register(); optional TASKS
│   ├── environment.py       # Env and registration metadata
│   ├── backend.py           # optional external-process/domain seam
│   ├── model.py             # optional canonical state
│   └── tasks/task0001.json  # optional strict Task v2
└── tests/
```

Use installed Agent-MM contracts; never copy them into the Env package.

## Mandatory and optional surfaces

```python
class MyEnv:
    def tools(self) -> Sequence[ToolSpec]: ...
    def call(self, tool_name: str, args: Mapping[str, Any]) -> ToolResult: ...

    # Optional capabilities:
    def start(self, init: Mapping[str, Any] | None, workspace: Path) -> ToolResult | None: ...
    def close(self) -> None: ...
```

An Env can be stateful without requiring a Scenario; `start` then receives
`None`. A stateless MCP can omit `start`. Snapshots, renderers, artifact
codecs, and `verify_task` are optional domain capabilities.

Rollout and replay use `create -> optional start -> tools -> calls -> close`.
If `start` exists, `tools()` may depend on its successful completion; if it is
absent, the catalog must be available without that hook. Discovery occurs
once before the first model decision or replayed action and the catalog stays
fixed for the episode. Startup or catalog failures still invoke cleanup.
Startup failures must not proceed to discovery.

For a session-based MCP, open the connection and complete its handshake in
`start`; map `list_tools()` in `tools` and `call_tool()` in `call` using the
same session, then release it in `close`. A pre-generated catalog is not a
contract requirement. Direct `ToolLoop(env)` users must start their Env first
when needed; the loop itself only discovers tools and dispatches actions.

### Optional capability signatures

```python
# Returns a JSON-safe mapping that StatePredicateVerifier can evaluate.
def snapshot(self) -> Mapping[str, Any]: ...

# Receives the resolved verifier binding and the finished rollout Trajectory.
# Must return VerificationResult; raise for infrastructure failures instead of
# returning failed results so ReplayVerify can mark the check diverged.
def verify_task(
    self,
    binding: Mapping[str, Any],
    rollout: Trajectory,
) -> VerificationResult: ...
```

These capabilities are discovered by `getattr`; omitting either method simply
means the corresponding verification path is unavailable for that Env.

`EnvironmentSpec` contains only stable id, name, description, solver rules,
and modalities. `ToolSpec.operation_type` is `query`, `mutation`, or `unknown`.
Use `unknown` when an upstream declaration does not prove the distinction.

## Tasks and verification

```python
Task(
    task_id="task0001",
    env_id="my_env",
    messages=(Message.text("user", "..."),),
    scenario=Scenario(init={...}),
    judge_ref=JudgeReference(
        score_min=1,
        score_max=5,
        criteria=(JudgeCriterion("accuracy", "Check the result."),),
    ),
)
```

Task is required by the runner and reusable across rollouts. Scenario is
private and optional. JudgeReference is public and optional; it defines a score
range and equally weighted task-specific criteria, while an absent value uses
the generic Judge rubric. Replay verifier factories are resolved independently
by `ReplayVerifierResolver`; they receive the live replay Env and immutable
Trajectory, and may return a graded `VerificationResult`.

## Tool and observation rules

- Use one authoritative catalog and unique stable names.
- Prefer closed, bounded JSON schemas.
- Validate semantic constraints and path confinement in the adapter.
- Return `ToolResult.failure` for expected errors.
- Keep failed multi-object mutations atomic.
- Return concise text receipts and images after relevant visual mutations.
- Do not expose shell/eval/unrestricted files or hidden truth-query tools.
- Do not implement `finish`; the runtime owns it.

Use a canonical model only when it materially improves replay, artifacts, or
rendering. External applications may stay authoritative behind an audited
session adapter.

## Registration

```python
from dataflow_mm_agent.env import register_env

def register() -> None:
    register_env(
        "my_env",
        MyEnv,
        name="My Env",
        description="Solver-facing purpose and limits.",
        modalities=("text", "image"),
    )
```

Registration never includes tasks or verifiers. Applications construct a
`JsonTaskStore` or another explicit `TaskResolver` separately.

```toml
[project.entry-points."dataflow_mm_agent.environments"]
my_env = "my_agent_mm_env:register"
```

Pin concrete MCP/browser/file dependencies in the Env distribution and test
the built wheel and sdist, including task/assets when present.

## Process-isolated deployment

For production deployments, each Env runs in a dedicated worker process via
`register_isolated_environments`. The host process never imports the Env's
dependencies directly; it discovers registered Envs through a one-shot RPC and
proxies all calls.

Environment variables that configure isolation:

| Variable | Meaning | Default |
|---|---|---|
| `DATAFLOW_MM_AGENT_ENV_MODE` | `process` (isolated, default) or `inprocess` | `process` |
| `DATAFLOW_MM_AGENT_ENV_PYTHON` | Path to the Python interpreter that has the Env package installed | Auto-detected conda env |
| `DATAFLOW_MM_AGENT_ENV_NAME` | Conda env name used for auto-detection | `dataflow-mm-envs` |
| `DATAFLOW_MM_AGENT_ENV_RPC_TIMEOUT` | Seconds to wait for each worker RPC | `180` |
| `DATAFLOW_MM_AGENT_ENV_WORKER` | Internal marker set by the worker itself (`1`) | unset |

Model-routing variables (`API_URL`, `BASE_URL`, `MODEL`, `SERVING_BACKEND`,
`KIGRESS_*`) and any variable ending in a secret suffix (`*_API_KEY`,
`*_TOKEN`, `*_SECRET`, `*_PASSWORD`, `*_CREDENTIAL*`, `*_USER_KEY`) are
stripped from the worker's environment and never reach the isolated process.

```python
from dataflow_mm_agent.env.process_runtime import register_isolated_environments

register_isolated_environments(
    plugin_modules=("my_agent_mm_env",),
    project_root="/path/to/my-agent-mm-env",
)
```

The worker process must have `dataflow-mm-agent` and the Env package
installed. Keep `require_distinct=True` so every Env gets its own interpreter.
