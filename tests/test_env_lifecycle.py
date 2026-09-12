"""Regression coverage for startup before per-episode tool discovery."""

from dataclasses import replace
import json

import pytest

from dataflow_mm_agent import (
    AgentRollout, ContentLimits, EnvironmentSpec, Message, ReplayVerify,
    ReplayVerifyConfig, RolloutConfig, Scenario, ScriptedServing, Task,
    TextContent, ToolResult, ToolSpec, VerificationResult,
)
from dataflow_mm_agent.env.registry import ENVIRONMENTS, EnvironmentRegistration
from dataflow_mm_agent.env.worker import _Worker


SPEC = EnvironmentSpec(env_id="lifecycle-test", name="Echo", description="Echo text.")
TASK = Task(
    task_id="echo-task", env_id=SPEC.env_id,
    messages=(Message.text("user", "Echo hello."), Message.text("user", "Then finish.")),
)
ECHO_TOOL = ToolSpec(
    name="echo", description="Echo text.", operation_type="query",
    input_schema={
        "type": "object", "properties": {"text": {"type": "string"}},
        "required": ["text"], "additionalProperties": False,
    },
)
ECHO = json.dumps({"tool": "echo", "args": {"text": "hello"}})
FINISH = json.dumps({"tool": "finish", "args": {"answer": "done"}})


class MinimalEnv:
    """Deliberately implements only the mandatory Env methods."""

    def __init__(self):
        self.events = []
        self.last_text = None

    def tools(self):
        self.events.append("tools")
        return (ECHO_TOOL,)

    def call(self, tool_name, args):
        assert tool_name == "echo"
        self.events.append("call")
        self.last_text = args["text"]
        return ToolResult.success((TextContent(self.last_text),))


class SessionEnv(MinimalEnv):
    """Models an MCP whose tools require the episode's live connection."""

    def __init__(self, failure=None, initial_text="ready"):
        super().__init__()
        self.failure = failure
        self.initial_text = initial_text
        self.started = False

    def start(self, init, workspace):
        self.events.append("start")
        self.init = init
        self.workspace = workspace
        assert workspace.is_dir()
        self.started = True  # cleanup is required even after partial startup
        if self.failure == "raise":
            raise RuntimeError("connection failed")
        if self.failure == "failure":
            return ToolResult.failure("connect_failed", "connection failed")
        if self.failure == "final":
            return ToolResult.success(is_final=True)
        if self.failure == "invalid_type":
            return "not a ToolResult"
        if self.failure == "invalid_content":
            return ToolResult.success((TextContent("x" * 65_000),))
        if self.initial_text is None:
            return None
        return ToolResult.success((TextContent(self.initial_text),))

    def tools(self):
        assert self.started, "tools require a live session"
        tools = super().tools()
        if self.failure == "catalog":
            raise RuntimeError("discovery failed")
        if self.failure == "duplicate":
            return (*tools, *tools)
        return tools

    def call(self, tool_name, args):
        assert self.started
        return super().call(tool_name, args)

    def close(self):
        self.events.append("close")
        self.started = False


class RecordingServing(ScriptedServing):
    def __init__(self, responses):
        super().__init__(responses)
        self.options = []

    def generate_messages_with_options(self, conversations, request_options):
        self.options.extend(request_options)
        return super().generate_messages_with_options(conversations, request_options)


def runner(env, serving=None, **config):
    return AgentRollout(
        serving=serving if serving is not None else ScriptedServing([ECHO, FINISH]),
        config=RolloutConfig(max_steps=4, include_host_tools=False, **config),
        env_resolver=lambda _: env, spec_resolver=lambda _: SPEC,
    )


class Resolver:
    def __init__(self, task=TASK, with_verifier=True):
        self.task = task
        self.with_verifier = with_verifier
        self.verifier_created = False

    def resolve(self, task_id, *, env_id):
        assert (task_id, env_id) == (self.task.task_id, self.task.env_id)
        return self.task

    def verifier_factory(self, task_id, *, env_id):
        self.resolve(task_id, env_id=env_id)
        return self.build_verifier if self.with_verifier else None

    def build_verifier(self):
        self.verifier_created = True
        return self

    def verify(self, env, rollout):
        assert env.last_text == "hello"
        if isinstance(env, SessionEnv):
            assert env.started
        env.events.append("verify")
        return VerificationResult(passed=True, reward=1)


def replay(trajectory, env, resolver=None, **config):
    resolver = resolver if resolver is not None else Resolver()
    return ReplayVerify(
        task_resolver=resolver, verifier_resolver=resolver,
        env_resolver=lambda _: env, config=ReplayVerifyConfig(**config),
    ).verify(trajectory)


@pytest.mark.parametrize("scenario", [None, Scenario(init={"seed": 7})])
def test_session_discovery_follows_start_and_preserves_first_model_input(scenario):
    task = replace(TASK, scenario=scenario)
    env = SessionEnv(initial_text="ready " * 10)
    serving = RecordingServing([ECHO, FINISH])
    trajectory = runner(
        env, serving, structured_actions=True, max_observation_chars=16,
    ).run(task)
    assert trajectory.success
    assert env.events == ["start", "tools", "call", "close"]
    assert env.init == (dict(scenario.init) if scenario is not None else None)
    first = serving.requests[0]
    assert [message.role for message in first] == [
        "system", "user", "user", "observation",
    ]
    assert first[1:3] == task.messages
    assert first[-1].name == "env.start"
    assert "...[truncated" in first[-1].content[0].text
    assert '"name": "echo"' in first[0].content[0].text
    branches = serving.options[0]["response_format"]["json_schema"]["schema"]["oneOf"]
    assert {branch["properties"]["tool"]["const"] for branch in branches} == {
        "echo", "finish",
    }
    assert trajectory.steps[0].response_message_index == len(first)
    observation = trajectory.messages[trajectory.steps[0].observation_message_index]
    assert observation.name == "echo"


@pytest.mark.parametrize("env_type", [MinimalEnv, SessionEnv])
def test_fresh_replay_uses_same_lifecycle_and_supports_minimal_env(env_type):
    env = env_type()
    trajectory = runner(env).run(TASK)
    fresh = env_type()
    result = replay(trajectory, fresh)
    assert trajectory.success and result.status == "passed"
    assert result.replayed_steps == 2
    if env_type is SessionEnv:
        assert fresh.events == ["start", "tools", "call", "verify", "close"]
        assert fresh.workspace != env.workspace
    else:
        assert fresh.events == ["tools", "call", "verify"]


@pytest.mark.parametrize("entrypoint", ["rollout", "replay"])
@pytest.mark.parametrize(
    "failure,reason",
    [
        ("raise", "connection failed"),
        ("failure", "connect_failed"),
        ("final", "must not terminate"),
        ("invalid_type", "must return ToolResult"),
        ("invalid_content", "invalid_content"),
        ("catalog", "discovery failed"),
        ("duplicate", "must be unique"),
    ],
)
def test_failures_stop_before_actions_and_always_close(entrypoint, failure, reason):
    env = SessionEnv(failure=failure)
    if entrypoint == "rollout":
        serving = RecordingServing([ECHO, FINISH])
        result = runner(env, serving).run(TASK)
        assert result.termination_reason == "infrastructure_error"
        assert result.steps == ()
        assert serving.requests == []
        assert result.messages[:len(TASK.messages)] == TASK.messages
        assert reason in result.messages[-1].content[0].text
    else:
        trajectory = runner(MinimalEnv()).run(TASK)
        resolver = Resolver()
        result = replay(trajectory, env, resolver)
        assert result.status == "error"
        assert result.replayed_steps == 0
        assert not resolver.verifier_created
        assert reason in result.reason
    expected = ["start"]
    if failure in {"catalog", "duplicate"}:
        expected.append("tools")
    assert env.events == [*expected, "close"]


def test_custom_initial_content_limits_apply_before_discovery():
    env = SessionEnv(initial_text="x" * 100)
    result = runner(
        env, content_limits=ContentLimits(max_text_chars=50),
    ).run(TASK)
    assert result.termination_reason == "infrastructure_error"
    assert env.events == ["start", "close"]
    fresh = SessionEnv(initial_text="x" * 100)
    result = replay(
        runner(MinimalEnv()).run(TASK), fresh,
        content_limits=ContentLimits(max_text_chars=50),
    )
    assert result.status == "error" and "invalid_content" in result.reason
    assert fresh.events == ["start", "close"]


def test_start_may_return_none():
    env = SessionEnv(initial_text=None)
    serving = RecordingServing([ECHO, FINISH])
    result = runner(env, serving).run(TASK)
    assert result.success
    assert env.events == ["start", "tools", "call", "close"]
    assert [message.role for message in serving.requests[0]] == ["system", "user", "user"]


@pytest.mark.parametrize("prefix", [(), (ECHO,)])
def test_refine_prefix_context_and_empty_tree_prefix_keep_message_order(prefix):
    env = SessionEnv()
    if not prefix:
        result = runner(env).run_responses(TASK, ())
        assert result.steps == ()
        assert [message.role for message in result.messages] == [
            "system", "user", "user", "observation",
        ]
        assert env.events == ["start", "tools", "close"]
        return
    serving = RecordingServing([FINISH])
    correction = Message.text("user", "Finish the task.")
    result = runner(env, serving).run_with_response_prefix(
        TASK, prefix, continuation_messages=(correction,), compact_live_context=True,
    )
    assert result.success
    request = serving.requests[0]
    assert request[0].role == "system"
    assert request[1:3] == TASK.messages
    assert request[3].name == "echo"
    assert request[-1] == correction
    assert env.events == ["start", "tools", "call", "close"]


def test_no_verifier_still_skips_env_creation():
    trajectory = runner(MinimalEnv()).run(TASK)
    resolver = Resolver(with_verifier=False)
    verifier = ReplayVerify(
        task_resolver=resolver, verifier_resolver=resolver,
        env_resolver=lambda _: pytest.fail("Env must not be created"),
    )
    assert verifier.verify(trajectory).status == "not_applicable"


def test_worker_existing_rpc_surface_allows_start_before_tools(monkeypatch, tmp_path):
    monkeypatch.setattr("dataflow_mm_agent.env.worker.load_environment_plugins", lambda **_: None)
    env = SessionEnv()
    monkeypatch.setitem(
        ENVIRONMENTS, SPEC.env_id, EnvironmentRegistration(SPEC, lambda: env),
    )
    worker = _Worker(())
    try:
        worker.dispatch({"op": "describe"})
        worker.dispatch({"op": "create_env", "env_id": SPEC.env_id})
        assert env.events == []
        worker.dispatch({"op": "start", "init": None, "workspace": str(tmp_path)})
        tools, _ = worker.dispatch({"op": "tools"})
        assert tools == [ECHO_TOOL.to_dict()]
        result, _ = worker.dispatch({
            "op": "call", "tool_name": "echo", "args": {"text": "hello"},
        })
        assert result["ok"]
    finally:
        worker.dispatch({"op": "shutdown"})
    assert env.events == ["start", "tools", "call", "close"]
