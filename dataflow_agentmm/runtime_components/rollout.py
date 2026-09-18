"""Multimodal JSON-action rollouts over lightweight environments."""

from __future__ import annotations

import json
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Mapping, Sequence

from ..contracts import (
    ContentLimits,
    EnvironmentSpec,
    ImageContent,
    Message,
    Task,
    ToolResult,
    ToolSpec,
    close_env,
    start_env,
    validate_content,
)
from ..contracts.trajectory import EpisodeStep, Trajectory, utc_now
from ..env.registry import get_environment_spec, make_env
from ..prompts import LANGUAGES, ROLLOUT_SYSTEM_PROMPT, for_language
from ..serving import ModelResponseFormatError, ModelServing, track_usage
from .context import ContextPolicy
from .host import HostPolicy
from .tool_loop import ToolLoop, _safe_result


_SYSTEM_PROMPT = ROLLOUT_SYSTEM_PROMPT["en"]


EnvResolver = Callable[[str], Any]
SpecResolver = Callable[[str], EnvironmentSpec]
ResponseProvider = Callable[
    [Sequence[Message], Mapping[str, Any] | None],
    tuple[str, float] | None,
]
WorkspaceRetention = Literal["ephemeral", "full"]


@dataclass(frozen=True)
class RolloutConfig:
    max_steps: int = 64
    system_prompt: str = _SYSTEM_PROMPT
    language: str = "en"
    # Which recorded history each live request carries; None sends all of it.
    context_policy: "ContextPolicy | None" = None
    include_host_tools: bool = True
    include_tool_catalog: bool = True
    validate_tool_names: bool = True
    structured_actions: bool = False
    action_format_retries: int = 0
    max_observation_chars: int = 8000
    workspace_root: Path | None = None
    workspace_retention: WorkspaceRetention = "ephemeral"
    content_limits: ContentLimits = ContentLimits()

    def __post_init__(self) -> None:
        if self.context_policy is not None and not isinstance(self.context_policy, ContextPolicy):
            raise TypeError("context_policy must be a ContextPolicy or None")
        if self.language not in LANGUAGES:
            raise ValueError(f"language must be one of {sorted(LANGUAGES)}")
        if self.system_prompt == _SYSTEM_PROMPT and self.language != "en":
            # Only the untouched built-in default follows `language`.
            object.__setattr__(
                self, "system_prompt", for_language(ROLLOUT_SYSTEM_PROMPT, self.language)
            )
        if (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps < 1
        ):
            raise ValueError("max_steps must be a positive integer")
        if self.max_observation_chars < 1:
            raise ValueError("max_observation_chars must be positive")
        if (
            isinstance(self.action_format_retries, bool)
            or not isinstance(self.action_format_retries, int)
            or self.action_format_retries < 0
        ):
            raise ValueError("action_format_retries must be a non-negative integer")
        if self.workspace_retention not in ("ephemeral", "full"):
            raise ValueError("workspace_retention must be 'ephemeral' or 'full'")
        if self.workspace_retention == "full" and self.workspace_root is None:
            raise ValueError(
                "workspace_root is required when workspace_retention='full'"
            )


class AgentRollout:
    """Run one required Task; verification is intentionally out of scope."""

    def __init__(
        self,
        *,
        serving: ModelServing,
        config: RolloutConfig = RolloutConfig(),
        env_resolver: EnvResolver = make_env,
        spec_resolver: SpecResolver = get_environment_spec,
    ):
        self.serving = serving
        self.config = config
        self.env_resolver = env_resolver
        self.spec_resolver = spec_resolver

    parse_action = staticmethod(ToolLoop.parse_action)

    @staticmethod
    def _action_request_options(tools: Sequence[ToolSpec]) -> dict[str, Any]:
        branches = [{
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "tool": {"type": "string", "const": tool.name},
                "args": json.loads(json.dumps(tool.input_schema)),
            },
            "required": ["thought", "tool", "args"],
            "additionalProperties": False,
        } for tool in tools]
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_action",
                    "strict": True,
                    "schema": {"type": "object", "oneOf": branches},
                },
            },
        }

    def sample_responses(
        self,
        messages: Sequence[Message],
        count: int,
        *,
        request_options: Mapping[str, Any] | None = None,
    ) -> list[str]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer")
        conversations = [tuple(messages) for _ in range(count)]
        responses = (
            self.serving.generate_messages_with_options(
                conversations, [request_options for _ in range(count)]
            )
            if request_options is not None
            else self.serving.generate_messages(conversations)
        )
        if len(responses) != count:
            raise RuntimeError(
                f"serving returned {len(responses)} responses for {count} requests"
            )
        return responses

    @contextmanager
    def _episode_workspace(self, *, env_id: str, episode_id: str) -> Iterator[Path]:
        root = self.config.workspace_root
        if self.config.workspace_retention == "full":
            assert root is not None
            workspace = (Path(root).resolve() / env_id / episode_id).resolve()
            workspace.mkdir(parents=True, exist_ok=False)
            yield workspace
            return
        if root is not None:
            Path(root).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f"agentmm-{env_id}-", dir=str(root) if root else None
        ) as temporary:
            yield Path(temporary).resolve()

    def run(self, task: Task) -> Trajectory:
        def live_response(
            messages: Sequence[Message],
            request_options: Mapping[str, Any] | None,
        ) -> tuple[str, float]:
            before = time.perf_counter()
            response = self.sample_responses(
                messages, 1, request_options=request_options
            )[0]
            return response, (time.perf_counter() - before) * 1000

        return self._run_with_response_provider(
            task,
            live_response,
            exhaustion_reason="max_steps",
            step_limit=self.config.max_steps,
        )

    def run_with_response_prefix(
        self,
        task: Task,
        responses: Sequence[str],
        *,
        continuation_messages: Sequence[Message] = (),
        live_context: ContextPolicy | None = None,
    ) -> Trajectory:
        """Restore an episode through recorded actions, then continue live.

        Currently unused: the Refiner repairs from a fresh Env instead. This
        entry point is kept because restoring a recorded prefix is the right
        primitive for stateful authoring Envs, and it is planned to return as an
        Env-side repair tool rather than an operator flag.

        Prefix responses pass through the same parser, schema validation, tool
        dispatcher, and fresh Env lifecycle as ordinary model output.  No live
        model request is made until every recorded response has executed.

        ``live_context`` overrides ``RolloutConfig.context_policy`` for this
        call; either way the continuation messages and everything produced after
        them are never trimmed, and the trajectory records the full history.
        """

        if live_context is None:
            live_context = self.config.context_policy

        prefix = tuple(responses)
        if any(not isinstance(item, str) or not item for item in prefix):
            raise TypeError("response prefix must contain non-empty strings")
        continuation = tuple(continuation_messages)
        if any(not isinstance(item, Message) for item in continuation):
            raise TypeError("continuation_messages must contain Message values")
        if len(prefix) >= self.config.max_steps:
            raise ValueError("response prefix leaves no step budget for continuation")
        iterator = iter(prefix)
        live_context_start: int | None = None

        def prefixed_live_response(
            messages: Sequence[Message],
            request_options: Mapping[str, Any] | None,
        ) -> tuple[str, float]:
            try:
                return next(iterator), 0.0
            except StopIteration:
                nonlocal live_context_start
                request_messages = messages
                if live_context is not None:
                    if live_context_start is None:
                        live_context_start = len(messages) - len(continuation)
                    request_messages = live_context.project(
                        messages,
                        boundary=live_context_start,
                        task_message_count=len(task.messages),
                    )
                before = time.perf_counter()
                response = self.sample_responses(
                    request_messages, 1, request_options=request_options
                )[0]
                return response, (time.perf_counter() - before) * 1000

        trajectory = self._run_with_response_provider(
            task,
            prefixed_live_response,
            exhaustion_reason="max_steps",
            step_limit=self.config.max_steps,
            continuation_messages=continuation,
            continuation_step=len(prefix) + 1,
        )
        return Trajectory(
            episode_id=trajectory.episode_id,
            task_id=trajectory.task_id,
            env_id=trajectory.env_id,
            messages=trajectory.messages,
            steps=trajectory.steps,
            final_answer=trajectory.final_answer,
            termination_reason=trajectory.termination_reason,
            started_at=trajectory.started_at,
            completed_at=trajectory.completed_at,
            metadata={
                **dict(trajectory.metadata),
                "response_prefix_steps": len(prefix),
                "response_prefix_compact_live_context": live_context is not None,
            },
        )

    def run_responses(
        self,
        task: Task,
        responses: Sequence[str],
        *,
        exhaustion_reason: str = "responses_exhausted",
    ) -> Trajectory:
        fixed = tuple(responses)
        if any(not isinstance(item, str) for item in fixed):
            raise TypeError("responses must contain only strings")
        iterator = iter(fixed)

        def fixed_response(
            _messages: Sequence[Message],
            _request_options: Mapping[str, Any] | None,
        ) -> tuple[str, float] | None:
            try:
                return next(iterator), 0.0
            except StopIteration:
                return None

        return self._run_with_response_provider(
            task,
            fixed_response,
            exhaustion_reason=exhaustion_reason,
            step_limit=len(fixed),
        )

    def _run_with_response_provider(
        self,
        task: Task,
        response_provider: ResponseProvider,
        *,
        exhaustion_reason: str,
        step_limit: int,
        continuation_messages: Sequence[Message] = (),
        continuation_step: int | None = None,
    ) -> Trajectory:
        """Run one episode and record the model usage its requests reported."""

        with track_usage() as meter:
            trajectory = self._run_episode(
                task,
                response_provider,
                exhaustion_reason=exhaustion_reason,
                step_limit=step_limit,
                continuation_messages=continuation_messages,
                continuation_step=continuation_step,
            )
        usage = meter.to_dict()
        if not usage["requests"]:
            return trajectory
        return replace(trajectory, metadata={**trajectory.metadata, "usage": usage})

    def _run_episode(
        self,
        task: Task,
        response_provider: ResponseProvider,
        *,
        exhaustion_reason: str,
        step_limit: int,
        continuation_messages: Sequence[Message] = (),
        continuation_step: int | None = None,
    ) -> Trajectory:
        if not isinstance(task, Task):
            raise TypeError("AgentRollout.run requires a Task")
        for message in task.messages:
            validate_content(message.content, self.config.content_limits)
        started_at = utc_now()
        episode_id = f"{task.task_id}-{uuid.uuid4().hex[:12]}"

        with self._episode_workspace(env_id=task.env_id, episode_id=episode_id) as workspace:
            env = self.env_resolver(task.env_id)
            spec = self.spec_resolver(task.env_id)
            if spec.env_id != task.env_id:
                close_env(env)
                raise ValueError("environment description does not match task.env_id")
            messages: list[Message] = list(task.messages)
            steps: list[EpisodeStep] = []
            final_answer: str | None = None
            termination_reason = exhaustion_reason
            format_retries_used = 0
            format_retries_recovered = 0
            tool_names: tuple[str, ...] = ()

            def project(request_messages: Sequence[Message]) -> Sequence[Message]:
                """Trim recorded history for the request only; the trajectory keeps it."""

                policy = self.config.context_policy
                if policy is None:
                    return request_messages
                return policy.project(
                    request_messages,
                    boundary=len(request_messages),
                    task_message_count=len(task.messages),
                )

            def request_response(
                request_messages: Sequence[Message],
                request_options: Mapping[str, Any] | None,
            ) -> tuple[str, float] | None:
                """Map provider-level format failures into action-format retries.

                A Gemini ``MALFORMED_FUNCTION_CALL`` contains no assistant text,
                so it cannot reach ``parse_action`` naturally.  Represent only
                that typed provider failure as an invalid response; the normal
                parse-repair observation can then guide a retry.  Network, Env,
                authentication, and arbitrary runtime errors still propagate to
                the episode-level infrastructure-error handler.
                """

                try:
                    return response_provider(project(request_messages), request_options)
                except ModelResponseFormatError as exc:
                    return f"[model_response_format_error] {exc}", 0.0

            try:
                # Some backends can discover tools only after opening a session.
                initial = start_env(
                    env,
                    task.scenario.init if task.scenario is not None else None,
                    workspace,
                )
                if initial is not None:
                    initial = _safe_result(
                        initial,
                        content_limits=self.config.content_limits,
                        max_observation_chars=self.config.max_observation_chars,
                    )
                    if not initial.ok:
                        messages.append(ToolLoop.observation(initial, "env.start"))
                        code = initial.error.code if initial.error is not None else "unknown"
                        raise RuntimeError(f"Env.start failed with {code}")
                    if initial.is_final:
                        raise RuntimeError("Env.start must not terminate an episode")

                host_policy = HostPolicy(
                    workspace=workspace,
                    content_limits=self.config.content_limits,
                ) if self.config.include_host_tools else None
                loop = ToolLoop(
                    env,
                    host_policy=host_policy,
                    validate_tool_names=self.config.validate_tool_names,
                    content_limits=self.config.content_limits,
                    max_observation_chars=self.config.max_observation_chars,
                )
                tool_names = loop.tool_names
                request_options = (
                    self._action_request_options(loop.tools)
                    if self.config.structured_actions
                    else None
                )
                catalog = (
                    ToolLoop.catalog(loop.tools)
                    if self.config.include_tool_catalog
                    else "(tools are described by the caller)"
                )
                messages.insert(0, Message.text(
                    "system",
                    self.config.system_prompt.format(
                        tool_catalog=catalog,
                        environment_context=json.dumps(
                            spec.solver_context(), ensure_ascii=False, indent=2
                        ),
                    ),
                ))
                if initial is not None and initial.content:
                    messages.append(loop.observation(initial, "env.start"))

                for step_index in range(1, step_limit + 1):
                    if continuation_step == step_index:
                        messages.extend(continuation_messages)
                    generated = request_response(tuple(messages), request_options)
                    if generated is None:
                        break
                    response, elapsed_ms = generated
                    action = loop.parse_action(response)
                    retry_context = list(messages)
                    retries_this_step = 0
                    while (
                        action is None
                        and retries_this_step < self.config.action_format_retries
                    ):
                        retries_this_step += 1
                        format_retries_used += 1
                        retry_context.extend([
                            Message.text("assistant", response),
                            loop.observation(loop.parse_failure(), "agent.parse"),
                        ])
                        retried = request_response(
                            tuple(retry_context), request_options
                        )
                        if retried is None:
                            break
                        response, retry_elapsed = retried
                        elapsed_ms += retry_elapsed
                        action = loop.parse_action(response)
                    if retries_this_step and action is not None:
                        format_retries_recovered += 1

                    messages.append(Message.text("assistant", response))
                    response_index = len(messages) - 1
                    if action is None:
                        result = loop.parse_failure()
                        messages.append(loop.observation(result, "agent.parse"))
                        steps.append(EpisodeStep(
                            index=step_index,
                            response_message_index=response_index,
                            action=None,
                            observation_message_index=len(messages) - 1,
                            parse_error=True,
                            elapsed_ms=elapsed_ms,
                            tool_ok=False,
                            error_code=result.error.code if result.error else None,
                            retryable=result.error.retryable if result.error else None,
                        ))
                        continue

                    execution = loop.execute(action)
                    observation_index = None
                    if execution.emit_observation:
                        messages.append(loop.observation(
                            execution.result, str(action.get("tool") or "agent.action")
                        ))
                        observation_index = len(messages) - 1
                    steps.append(EpisodeStep(
                        index=step_index,
                        response_message_index=response_index,
                        action=dict(action),
                        observation_message_index=observation_index,
                        elapsed_ms=elapsed_ms,
                        tool_ok=execution.result.ok,
                        error_code=(
                            execution.result.error.code
                            if execution.result.error is not None
                            else None
                        ),
                        retryable=(
                            execution.result.error.retryable
                            if execution.result.error is not None
                            else None
                        ),
                        is_final=execution.result.is_final,
                    ))
                    if execution.terminal_kind is not None:
                        termination_reason = execution.terminal_kind
                        final_answer = execution.final_answer
                        break
            except Exception as exc:
                termination_reason = "infrastructure_error"
                messages.append(Message.text(
                    "observation",
                    f"FATAL [{type(exc).__name__}]: {exc}",
                    name="runtime",
                ))
            finally:
                try:
                    close_env(env)
                except Exception as exc:
                    if termination_reason != "infrastructure_error":
                        termination_reason = "infrastructure_error"
                        messages.append(Message.text(
                            "observation",
                            f"FATAL [close {type(exc).__name__}]: {exc}",
                            name="runtime",
                        ))

            return Trajectory(
                episode_id=episode_id,
                task_id=task.task_id,
                env_id=task.env_id,
                messages=tuple(messages),
                steps=tuple(steps),
                final_answer=final_answer,
                termination_reason=termination_reason,
                started_at=started_at,
                completed_at=utc_now(),
                metadata={
                    "tools": list(tool_names),
                    "max_steps": step_limit,
                    "workspace_retention": self.config.workspace_retention,
                    **(
                        {"workspace": str(workspace)}
                        if self.config.workspace_retention == "full"
                        else {}
                    ),
                    "task_image_count": sum(
                        isinstance(item, ImageContent)
                        for message in task.messages
                        for item in message.content
                    ),
                    "structured_actions": self.config.structured_actions,
                    "action_format_retries": self.config.action_format_retries,
                    "format_retries_used": format_retries_used,
                    "format_retries_recovered": format_retries_recovered,
                    "host_tools": self.config.include_host_tools,
                    "validate_tool_names": self.config.validate_tool_names,
                },
            )

    def run_batch(self, tasks: Sequence[Task], *, max_workers: int = 1) -> list[Trajectory]:
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        if max_workers == 1:
            return [self.run(task) for task in tasks]
        results: list[Trajectory | None] = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.run, task): index
                for index, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return [result for result in results if result is not None]


__all__ = ["AgentRollout", "RolloutConfig"]
