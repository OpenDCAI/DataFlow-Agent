"""Required task identity and optional private scenario contracts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from .agent import Message
from .environment import ReplayVerifier, Scenario


TASK_SCHEMA_VERSION = 2
TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")


def validate_task_id(task_id: str) -> None:
    if not isinstance(task_id, str) or not TASK_ID_PATTERN.fullmatch(task_id):
        raise ValueError(f"unsafe task_id={task_id!r}")


def _finite_score(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"{field_name} must be a finite number")
    return score


@dataclass(frozen=True)
class JudgeCriterion:
    """One equally weighted criterion in a task-authored Judge rubric."""

    id: str
    description: str

    def __post_init__(self) -> None:
        validate_task_id(self.id)
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("judge criterion description must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "description": self.description}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "JudgeCriterion":
        if not isinstance(value, Mapping):
            raise TypeError("judge criterion must be an object")
        if set(value) != {"id", "description"}:
            raise ValueError(
                "judge criterion must contain exactly id and description"
            )
        return cls(id=value["id"], description=value["description"])


@dataclass(frozen=True)
class JudgeReference:
    """Optional Task-owned score range and equally weighted Judge criteria."""

    score_min: float
    score_max: float
    criteria: tuple[JudgeCriterion, ...]

    def __post_init__(self) -> None:
        minimum = _finite_score(self.score_min, "judge_ref.score_range.min")
        maximum = _finite_score(self.score_max, "judge_ref.score_range.max")
        if minimum >= maximum:
            raise ValueError("judge_ref score range requires min < max")
        criteria = tuple(self.criteria)
        if not criteria or any(
            not isinstance(item, JudgeCriterion) for item in criteria
        ):
            raise ValueError(
                "judge_ref.criteria must contain at least one JudgeCriterion"
            )
        criterion_ids = [item.id for item in criteria]
        if len(set(criterion_ids)) != len(criterion_ids):
            raise ValueError("judge_ref criterion ids must be unique")
        object.__setattr__(self, "score_min", minimum)
        object.__setattr__(self, "score_max", maximum)
        object.__setattr__(self, "criteria", criteria)

    @property
    def criterion_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.criteria)

    def normalize(self, score: Any) -> float:
        value = _finite_score(score, "judge criterion score")
        if value < self.score_min or value > self.score_max:
            raise ValueError(
                f"judge criterion score {value} is outside "
                f"[{self.score_min}, {self.score_max}]"
            )
        return (value - self.score_min) / (self.score_max - self.score_min)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_range": {"min": self.score_min, "max": self.score_max},
            "criteria": [item.to_dict() for item in self.criteria],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "JudgeReference":
        if not isinstance(value, Mapping):
            raise TypeError("judge_ref must be an object")
        if set(value) != {"score_range", "criteria"}:
            raise ValueError(
                "judge_ref must contain exactly score_range and criteria"
            )
        score_range = value.get("score_range")
        if not isinstance(score_range, Mapping) or set(score_range) != {
            "min", "max"
        }:
            raise ValueError(
                "judge_ref.score_range must contain exactly min and max"
            )
        criteria = value.get("criteria")
        if not isinstance(criteria, list):
            raise TypeError("judge_ref.criteria must be a list")
        return cls(
            score_min=score_range["min"],
            score_max=score_range["max"],
            criteria=tuple(JudgeCriterion.from_dict(item) for item in criteria),
        )


@dataclass(frozen=True)
class Task:
    """One reusable task definition.

    A task may be rolled out any number of times.  ``messages`` are the full
    task-authored model input and may be empty or multimodal.  ``scenario`` is
    private runtime data and is intentionally excluded from normal
    serialization and trajectory records. ``judge_ref`` is optional public
    scoring metadata consumed by the separate Judge operator.
    """

    task_id: str
    env_id: str
    messages: tuple[Message, ...] = ()
    scenario: Scenario | None = field(default=None, repr=False, compare=False)
    judge_ref: JudgeReference | None = None

    def __post_init__(self) -> None:
        validate_task_id(self.task_id)
        if not isinstance(self.env_id, str) or not TASK_ID_PATTERN.fullmatch(self.env_id):
            raise ValueError(f"invalid env_id={self.env_id!r}")
        messages = tuple(self.messages)
        if any(not isinstance(message, Message) for message in messages):
            raise TypeError("task.messages must contain Message values")
        if self.scenario is not None and not isinstance(self.scenario, Scenario):
            raise TypeError("task.scenario must be Scenario or None")
        if self.judge_ref is not None and not isinstance(
            self.judge_ref, JudgeReference
        ):
            raise TypeError("task.judge_ref must be JudgeReference or None")
        object.__setattr__(self, "messages", messages)

    def to_dict(self, *, include_scenario_init: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": TASK_SCHEMA_VERSION,
            "task_id": self.task_id,
            "env_id": self.env_id,
            "messages": [message.to_dict() for message in self.messages],
        }
        if include_scenario_init and self.scenario is not None:
            value["scenario"] = {"init": dict(self.scenario.init)}
        if self.judge_ref is not None:
            value["judge_ref"] = self.judge_ref.to_dict()
        return value

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        scenario: Scenario | None = None,
    ) -> "Task":
        version = value.get("schema_version")
        if version != TASK_SCHEMA_VERSION:
            raise ValueError(f"unsupported task schema_version={version!r}")
        allowed = {
            "schema_version", "task_id", "env_id", "messages", "scenario",
            "judge_ref",
        }
        extra = set(value).difference(allowed)
        if extra:
            raise ValueError(f"unsupported task fields: {sorted(extra)}")
        messages = value.get("messages")
        if not isinstance(messages, list):
            raise TypeError("task.messages must be a list")
        task_id = value.get("task_id")
        env_id = value.get("env_id")
        if not isinstance(task_id, str) or not isinstance(env_id, str):
            raise TypeError("task_id and env_id must be strings")
        return cls(
            task_id=task_id,
            env_id=env_id,
            messages=tuple(Message.from_dict(item) for item in messages),
            scenario=scenario,
            judge_ref=(
                JudgeReference.from_dict(value["judge_ref"])
                if value.get("judge_ref") is not None
                else None
            ),
        )


@runtime_checkable
class TaskResolver(Protocol):
    """Explicit identity resolver used by pipelines and ReplayVerify."""

    def resolve(self, task_id: str, *, env_id: str | None = None) -> Task:
        ...


@runtime_checkable
class ReplayVerifierResolver(Protocol):
    """Resolve a lazy fresh-verifier factory for one Task identity."""

    def verifier_factory(
        self,
        task_id: str,
        *,
        env_id: str,
    ) -> "ReplayVerifierFactory | None":
        ...


ReplayVerifierFactory = Callable[[], ReplayVerifier]


__all__ = [
    "JudgeCriterion",
    "JudgeReference",
    "TASK_ID_PATTERN",
    "TASK_SCHEMA_VERSION",
    "Task",
    "TaskResolver",
    "ReplayVerifierFactory",
    "ReplayVerifierResolver",
    "validate_task_id",
]
