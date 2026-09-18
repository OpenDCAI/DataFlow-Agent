"""Shared JSON-action execution semantics for rollout and replay."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..contracts import (
    ContentLimits,
    Env,
    Message,
    TextContent,
    ToolResult,
    ToolSpec,
    validate_content,
)
from .finish import FINISH_TOOL_SPEC
from .host import HostPolicy, HostTools


def _safe_result(
    result: ToolResult,
    *,
    content_limits: ContentLimits,
    max_observation_chars: int,
) -> ToolResult:
    """Normalize observations without requiring a discovered tool catalog."""

    if not isinstance(result, ToolResult):
        return ToolResult.failure(
            "environment_error",
            f"Env returned {type(result).__name__}, expected ToolResult",
        )
    try:
        validate_content(result.content, content_limits)
        content = []
        for item in result.content:
            if (
                isinstance(item, TextContent)
                and len(item.text) > max_observation_chars
            ):
                omitted = len(item.text) - max_observation_chars
                content.append(TextContent(
                    item.text[:max_observation_chars]
                    + f"\n...[truncated {omitted} chars of {len(item.text)} total]"
                ))
            else:
                content.append(item)
        return ToolResult(
            ok=result.ok,
            content=tuple(content),
            error=result.error,
            is_final=result.is_final,
            metadata=result.metadata,
        )
    except (TypeError, ValueError) as exc:
        return ToolResult.failure("invalid_content", str(exc))


@dataclass(frozen=True)
class ActionExecution:
    result: ToolResult
    terminal_kind: str | None = None
    final_answer: str | None = None
    emit_observation: bool = True


class ToolLoop:
    """One episode's authoritative tool catalog and dispatcher.

    Callers must complete optional Env startup before constructing the loop.
    The catalog is read once here and retained for this episode.
    """

    def __init__(
        self,
        env: Env,
        *,
        host_policy: HostPolicy | None = None,
        validate_tool_names: bool = True,
        content_limits: ContentLimits = ContentLimits(),
        max_observation_chars: int = 8000,
    ):
        self.env = env
        self.host = HostTools(host_policy) if host_policy is not None else None
        self.validate_tool_names = validate_tool_names
        self.content_limits = content_limits
        self.max_observation_chars = max_observation_chars
        self.env_tools = tuple(env.tools())
        self.host_tools = tuple(self.host.tools()) if self.host is not None else ()
        self.tools = self.env_tools + self.host_tools + (FINISH_TOOL_SPEC,)
        names = [tool.name for tool in self.tools]
        if len(set(names)) != len(names):
            raise ValueError("environment, host, and runtime tool names must be unique")
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.host_names = {tool.name for tool in self.host_tools}

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(tool.name for tool in self.tools)

    @staticmethod
    def parse_action(text: str) -> dict[str, Any] | None:
        """Return the last syntactically valid JSON tool action in a response."""

        if not text:
            return None
        candidates: list[str] = []
        start: int | None = None
        depth = 0
        in_string = False
        escaped = False
        for index, character in enumerate(text):
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character == "{":
                if depth == 0:
                    start = index
                depth += 1
            elif character == "}" and depth:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start:index + 1])
                    start = None
        for candidate in reversed(candidates):
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict):
                continue
            tool = value.get("tool")
            args = value.get("args", {})
            if not isinstance(tool, str) or not isinstance(args, dict):
                continue
            value["args"] = args
            return value
        return None

    @staticmethod
    def validate_args(tool: ToolSpec, args: Mapping[str, Any]) -> str | None:
        try:
            import jsonschema

            jsonschema.validate(instance=dict(args), schema=dict(tool.input_schema))
        except ImportError:
            required = tool.input_schema.get("required") or []
            missing = [name for name in required if name not in args]
            return f"missing required arguments: {missing}" if missing else None
        except Exception as exc:
            return str(exc).splitlines()[0]
        return None

    @staticmethod
    def parse_failure() -> ToolResult:
        return ToolResult.failure(
            "invalid_action",
            "response must contain one JSON action object",
        )

    def safe_result(self, result: ToolResult) -> ToolResult:
        return _safe_result(
            result,
            content_limits=self.content_limits,
            max_observation_chars=self.max_observation_chars,
        )

    @staticmethod
    def observation(result: ToolResult, tool_name: str) -> Message:
        content = list(result.content)
        if result.error is not None:
            content.insert(0, TextContent(
                f"ERROR [{result.error.code}]: {result.error.message}"
            ))
        if not content:
            content.append(TextContent("OK" if result.ok else "ERROR"))
        return Message.of("observation", content, name=tool_name)

    def execute(self, action: Mapping[str, Any]) -> ActionExecution:
        tool_name = action.get("tool")
        args = action.get("args", {})
        if not isinstance(tool_name, str) or not isinstance(args, Mapping):
            return ActionExecution(self.parse_failure())

        if tool_name == "finish":
            validation_error = self.validate_args(FINISH_TOOL_SPEC, args)
            if validation_error:
                return ActionExecution(ToolResult.failure(
                    "invalid_arguments", validation_error
                ))
            answer = str(args["answer"])
            return ActionExecution(
                result=ToolResult.success(is_final=True),
                terminal_kind="finish",
                final_answer=answer,
                emit_observation=False,
            )

        spec = self.tool_map.get(tool_name)
        if spec is None and self.validate_tool_names:
            return ActionExecution(ToolResult.failure(
                "unknown_tool",
                f"unknown tool {tool_name!r}; available: {sorted(self.tool_map)}",
            ))
        if spec is not None:
            validation_error = self.validate_args(spec, args)
            if validation_error:
                return ActionExecution(ToolResult.failure(
                    "invalid_arguments", validation_error
                ))

        try:
            result = (
                self.host.call(tool_name, args)
                if self.host is not None and tool_name in self.host_names
                else self.env.call(tool_name, args)
            )
        except Exception as exc:
            result = ToolResult.failure(
                "environment_error",
                f"{type(exc).__name__}: {exc}",
                retryable=False,
            )
        safe = self.safe_result(result)
        return ActionExecution(
            result=safe,
            terminal_kind="environment_final" if safe.is_final else None,
        )

    @staticmethod
    def catalog(tools: Sequence[ToolSpec]) -> str:
        return json.dumps(
            [tool.to_dict() for tool in tools], ensure_ascii=False, indent=2
        )


__all__ = ["ActionExecution", "ToolLoop"]
