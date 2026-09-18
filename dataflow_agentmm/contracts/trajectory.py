"""Canonical rollout trajectory contract (v2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from .agent import Message


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class EpisodeStep:
    """One parsed response and its control-level result."""

    index: int
    response_message_index: int
    action: Mapping[str, Any] | None
    observation_message_index: int | None = None
    parse_error: bool = False
    elapsed_ms: float | None = None
    tool_ok: bool | None = None
    error_code: str | None = None
    retryable: bool | None = None
    is_final: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 1:
            raise ValueError("step index must be a positive integer")
        if (
            isinstance(self.response_message_index, bool)
            or not isinstance(self.response_message_index, int)
            or self.response_message_index < 0
        ):
            raise ValueError("response_message_index must be non-negative")
        if self.observation_message_index is not None and (
            isinstance(self.observation_message_index, bool)
            or not isinstance(self.observation_message_index, int)
            or self.observation_message_index < 0
        ):
            raise ValueError("observation_message_index must be non-negative or null")
        if self.action is not None and not isinstance(self.action, Mapping):
            raise TypeError("action must be a mapping or null")
        for name in ("parse_error", "is_final"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        for name in ("tool_ok", "retryable"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean or null")

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "response_message_index": self.response_message_index,
            "action": dict(self.action) if self.action is not None else None,
            "observation_message_index": self.observation_message_index,
            "parse_error": self.parse_error,
            "elapsed_ms": self.elapsed_ms,
            "tool_ok": self.tool_ok,
            "error_code": self.error_code,
            "retryable": self.retryable,
            "is_final": self.is_final,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EpisodeStep":
        required = {
            "index", "response_message_index", "action",
            "observation_message_index", "parse_error", "elapsed_ms",
            "tool_ok", "error_code", "retryable", "is_final",
        }
        missing = required.difference(value)
        if missing:
            raise KeyError(f"missing EpisodeStep fields: {sorted(missing)}")
        if (
            isinstance(value["index"], bool)
            or not isinstance(value["index"], int)
            or isinstance(value["response_message_index"], bool)
            or not isinstance(value["response_message_index"], int)
        ):
            raise TypeError("step indexes must be integers")
        observation_index = value["observation_message_index"]
        if observation_index is not None and (
            isinstance(observation_index, bool) or not isinstance(observation_index, int)
        ):
            raise TypeError("observation_message_index must be an integer or null")
        elapsed = value["elapsed_ms"]
        if elapsed is not None and (
            isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
        ):
            raise TypeError("elapsed_ms must be numeric or null")
        action = value.get("action")
        if action is not None and not isinstance(action, Mapping):
            raise TypeError("action must be a mapping or null")
        for name in ("parse_error", "is_final"):
            if not isinstance(value.get(name, False), bool):
                raise TypeError(f"{name} must be a boolean")
        for name in ("tool_ok", "retryable"):
            item = value.get(name)
            if item is not None and not isinstance(item, bool):
                raise TypeError(f"{name} must be a boolean or null")
        return cls(
            index=value["index"],
            response_message_index=value["response_message_index"],
            action=dict(action) if action is not None else None,
            observation_message_index=(
                observation_index
                if observation_index is not None
                else None
            ),
            parse_error=bool(value.get("parse_error", False)),
            elapsed_ms=(
                float(elapsed)
                if elapsed is not None
                else None
            ),
            tool_ok=value.get("tool_ok"),
            error_code=(
                str(value["error_code"])
                if value.get("error_code") is not None
                else None
            ),
            retryable=value.get("retryable"),
            is_final=bool(value.get("is_final", False)),
        )


@dataclass(frozen=True)
class Trajectory:
    """One rollout of a required Task; verification lives outside this value."""

    episode_id: str
    task_id: str
    env_id: str
    messages: tuple[Message, ...]
    steps: tuple[EpisodeStep, ...]
    final_answer: str | None
    termination_reason: str
    started_at: str
    completed_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("Trajectory schema_version must be 2")
        for name in (
            "episode_id", "task_id", "env_id", "termination_reason",
            "started_at", "completed_at",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        messages = tuple(self.messages)
        steps = tuple(self.steps)
        if any(not isinstance(item, Message) for item in messages):
            raise TypeError("trajectory messages must contain Message values")
        if any(not isinstance(item, EpisodeStep) for item in steps):
            raise TypeError("trajectory steps must contain EpisodeStep values")
        if self.final_answer is not None and not isinstance(self.final_answer, str):
            raise TypeError("final_answer must be a string or null")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("trajectory metadata must be a mapping")
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "steps", steps)

    @property
    def success(self) -> bool:
        return self.termination_reason in {"finish", "environment_final"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "env_id": self.env_id,
            "messages": [message.to_dict() for message in self.messages],
            "steps": [step.to_dict() for step in self.steps],
            "final_answer": self.final_answer,
            "termination_reason": self.termination_reason,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "success": self.success,
            "num_steps": len(self.steps),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Trajectory":
        required = {
            "schema_version", "episode_id", "task_id", "env_id", "messages",
            "steps", "final_answer", "termination_reason", "started_at",
            "completed_at", "success", "num_steps", "metadata",
        }
        missing = required.difference(value)
        if missing:
            raise KeyError(f"missing Trajectory fields: {sorted(missing)}")
        extra = set(value).difference(required)
        if extra:
            raise ValueError(f"unsupported Trajectory fields: {sorted(extra)}")
        for name in (
            "episode_id", "task_id", "env_id", "termination_reason",
            "started_at", "completed_at",
        ):
            if not isinstance(value[name], str):
                raise TypeError(f"{name} must be a string")
        if value["final_answer"] is not None and not isinstance(
            value["final_answer"], str
        ):
            raise TypeError("final_answer must be a string or null")
        if not isinstance(value["messages"], list) or not isinstance(
            value["steps"], list
        ) or not isinstance(value["metadata"], Mapping):
            raise TypeError("messages, steps, or metadata has an invalid type")
        if not isinstance(value["success"], bool) or isinstance(
            value["num_steps"], bool
        ) or not isinstance(value["num_steps"], int):
            raise TypeError("success/num_steps has an invalid type")
        version = value.get("schema_version")
        if version != 2:
            raise ValueError(f"unsupported trajectory schema_version={version!r}")
        trajectory = cls(
            schema_version=2,
            episode_id=value["episode_id"],
            task_id=value["task_id"],
            env_id=value["env_id"],
            messages=tuple(Message.from_dict(item) for item in value["messages"]),
            steps=tuple(EpisodeStep.from_dict(item) for item in value["steps"]),
            final_answer=(
                value["final_answer"]
                if value.get("final_answer") is not None
                else None
            ),
            termination_reason=value["termination_reason"],
            started_at=value["started_at"],
            completed_at=value["completed_at"],
            metadata=dict(value.get("metadata") or {}),
        )
        if value.get("success") is not trajectory.success:
            raise ValueError("success does not match termination_reason")
        if value.get("num_steps") != len(trajectory.steps):
            raise ValueError("num_steps does not match steps")
        return trajectory


__all__ = ["EpisodeStep", "Trajectory", "utc_now"]
