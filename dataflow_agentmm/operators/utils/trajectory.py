"""Canonical trajectory projections shared by DataFlow-AgentMM operators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ...contracts import ImageContent, Message, TextContent
from ...contracts.trajectory import Trajectory
from ...storage import TrajectoryStore


def as_trajectory(value: Any) -> Trajectory | None:
    """Parse a canonical trajectory value accepted by DataFlow cells."""
    if isinstance(value, Trajectory):
        return value
    if isinstance(value, Mapping):
        try:
            return Trajectory.from_dict(value)
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(value, str):
        candidate = value.strip()
        if candidate.startswith("{"):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                return None
            return as_trajectory(parsed)
        path = Path(candidate)
        try:
            is_jsonl_file = path.suffix.lower() == ".jsonl" and path.is_file()
        except OSError:
            return None
        if is_jsonl_file:
            try:
                return TrajectoryStore().load(path)
            except (OSError, TypeError, ValueError):
                return None
    return None


def as_trajectory_dict(value: Any) -> dict[str, Any] | None:
    """Return a validated canonical trajectory dictionary."""
    trajectory = as_trajectory(value)
    return trajectory.to_dict() if trajectory else None


def task_text(trajectory: Mapping[str, Any]) -> str:
    values: list[str] = []
    for raw in trajectory.get("messages") or []:
        if not isinstance(raw, Mapping) or raw.get("role") != "user":
            continue
        for content in raw.get("content") or []:
            if isinstance(content, Mapping) and content.get("type") == "text":
                values.append(str(content.get("text") or ""))
    return "\n".join(item for item in values if item)


def normal_success(trajectory: Mapping[str, Any]) -> bool:
    return trajectory.get("termination_reason") in {"finish", "environment_final"}


def steps(trajectory: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = trajectory.get("steps") or []
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def step_thought(step: Mapping[str, Any]) -> Any:
    action = step.get("action")
    return action.get("thought") if isinstance(action, Mapping) else None


def step_tool_ok(step: Mapping[str, Any]) -> bool | None:
    value = step.get("tool_ok")
    return value if isinstance(value, bool) else None


def step_error_code(step: Mapping[str, Any]) -> str | None:
    value = step.get("error_code")
    return str(value) if value is not None else None


def message_from_index(
    trajectory: Mapping[str, Any], index: Any,
) -> Message | None:
    if index is None:
        return None
    messages = trajectory.get("messages") or []
    try:
        value = messages[int(index)]
    except (IndexError, TypeError, ValueError):
        return None
    if isinstance(value, Message):
        return value
    if not isinstance(value, Mapping):
        return None
    try:
        return Message.from_dict(value)
    except (KeyError, TypeError, ValueError):
        return None


def observation_value(
    trajectory: Mapping[str, Any], step: Mapping[str, Any],
) -> Any:
    message = message_from_index(trajectory, step.get("observation_message_index"))
    if message is None:
        return None
    return [item.to_dict() for item in message.content]


def observation_text(
    trajectory: Mapping[str, Any], step: Mapping[str, Any],
) -> str:
    value = observation_value(trajectory, step)
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def observation_content(
    trajectory: Mapping[str, Any], step: Mapping[str, Any],
) -> tuple[TextContent | ImageContent, ...]:
    message = message_from_index(trajectory, step.get("observation_message_index"))
    return message.content if message is not None else ()
