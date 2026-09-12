"""Lightweight environment and optional scenario contracts.

An environment is deliberately just a tool catalog plus a dispatcher.  State,
backend startup, and teardown are capabilities discovered by the runtime, not
requirements imposed on every integration.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, TYPE_CHECKING, runtime_checkable

from .agent import ToolResult, ToolSpec

if TYPE_CHECKING:
    from .trajectory import Trajectory


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
_MODALITIES = frozenset({"text", "image"})


def json_object(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    """Detach one JSON object and reject executable/non-serializable values."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    try:
        copied = json.loads(json.dumps(dict(value), ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain only JSON values") from exc
    if not isinstance(copied, dict):
        raise TypeError(f"{field_name} must be a JSON object")
    return copied


@dataclass(frozen=True)
class RuleSpec:
    """One solver-facing rule that applies to an environment."""

    rule_id: str
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.rule_id, str) or not _IDENTIFIER.fullmatch(self.rule_id):
            raise ValueError(f"invalid rule_id={self.rule_id!r}")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("rule description must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {"rule_id": self.rule_id, "description": self.description}


@dataclass(frozen=True)
class EnvironmentSpec:
    """Solver-facing registration metadata; it is not a state schema."""

    env_id: str
    name: str
    description: str
    rules: tuple[RuleSpec, ...] = ()
    modalities: tuple[str, ...] = ("text",)

    def __post_init__(self) -> None:
        if not isinstance(self.env_id, str) or not _IDENTIFIER.fullmatch(self.env_id):
            raise ValueError(f"invalid env_id={self.env_id!r}")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("environment name must be non-empty")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("environment description must be non-empty")
        rules = tuple(self.rules)
        if any(not isinstance(rule, RuleSpec) for rule in rules):
            raise TypeError("rules must contain RuleSpec values")
        if len({rule.rule_id for rule in rules}) != len(rules):
            raise ValueError("environment rule_id values must be unique")
        modalities = tuple(self.modalities)
        if not modalities or any(item not in _MODALITIES for item in modalities):
            raise ValueError(
                f"modalities must be a non-empty subset of {sorted(_MODALITIES)}"
            )
        if len(set(modalities)) != len(modalities):
            raise ValueError("environment modalities must be unique")
        object.__setattr__(self, "rules", rules)
        object.__setattr__(self, "modalities", modalities)

    def solver_context(self) -> dict[str, Any]:
        return {
            "env_id": self.env_id,
            "name": self.name,
            "description": self.description,
            "rules": [rule.to_dict() for rule in self.rules],
            "modalities": list(self.modalities),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.solver_context()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EnvironmentSpec":
        return cls(
            env_id=str(value["env_id"]),
            name=str(value["name"]),
            description=str(value["description"]),
            rules=tuple(
                RuleSpec(str(rule["rule_id"]), str(rule["description"]))
                for rule in value.get("rules") or ()
            ),
            modalities=tuple(str(item) for item in value.get("modalities") or ("text",)),
        )


@runtime_checkable
class Env(Protocol):
    """The complete mandatory Env surface.

    Implementations may additionally expose ``start(init, workspace)`` and
    ``close()``.  The runtime discovers those hooks with ``getattr``.
    Rollout and replay complete optional startup before reading ``tools()``;
    the resulting catalog is fixed for the episode.
    """

    def tools(self) -> Sequence[ToolSpec]:
        """Return the episode catalog; may require successful optional startup."""
        ...

    def call(self, tool_name: str, args: Mapping[str, Any]) -> ToolResult:
        ...


@dataclass(frozen=True)
class VerificationCheck:
    name: str
    passed: bool
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("verification check name must be non-empty")
        if not isinstance(self.passed, bool):
            raise TypeError("verification check passed must be a boolean")
        if not isinstance(self.detail, str):
            raise TypeError("verification check detail must be a string")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VerificationCheck":
        if not isinstance(value.get("name"), str) or not isinstance(
            value.get("passed"), bool
        ) or not isinstance(value.get("detail", ""), str):
            raise TypeError("VerificationCheck fields have invalid types")
        return cls(
            name=value["name"],
            passed=value["passed"],
            detail=value.get("detail", ""),
        )


@dataclass(frozen=True)
class VerificationResult:
    """Domain verifier result; ReplayVerify wraps it with replay status."""

    passed: bool
    reward: float
    reason: str = ""
    checks: tuple[VerificationCheck, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise TypeError("verification passed must be a boolean")
        if isinstance(self.reward, bool) or not isinstance(self.reward, (int, float)):
            raise TypeError("verification reward must be numeric")
        if not isinstance(self.reason, str):
            raise TypeError("verification reason must be a string")
        checks = tuple(self.checks)
        if any(not isinstance(item, VerificationCheck) for item in checks):
            raise TypeError("checks must contain VerificationCheck values")
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(self, "checks", checks)

    @classmethod
    def from_checks(
        cls,
        checks: Sequence[VerificationCheck],
        *,
        reason: str = "",
    ) -> "VerificationResult":
        values = tuple(checks)
        passed = bool(values) and all(item.passed for item in values)
        reward = sum(item.passed for item in values) / len(values) if values else 0.0
        return cls(passed=passed, reward=reward, reason=reason, checks=values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reward": self.reward,
            "reason": self.reason,
            "checks": [item.to_dict() for item in self.checks],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VerificationResult":
        if not isinstance(value.get("passed"), bool):
            raise TypeError("verification passed must be a boolean")
        if isinstance(value.get("reward"), bool) or not isinstance(
            value.get("reward"), (int, float)
        ):
            raise TypeError("verification reward must be numeric")
        if not isinstance(value.get("reason", ""), str) or not isinstance(
            value.get("checks", []), list
        ):
            raise TypeError("verification reason/checks have invalid types")
        return cls(
            passed=value["passed"],
            reward=value["reward"],
            reason=value.get("reason", ""),
            checks=tuple(
                VerificationCheck.from_dict(item)
                for item in value.get("checks") or ()
            ),
        )


@runtime_checkable
class ReplayVerifier(Protocol):
    """A task-bound verifier run only after an exact fresh replay."""

    def verify(self, env: Env, rollout: "Trajectory") -> VerificationResult:
        ...

@dataclass(frozen=True)
class Scenario:
    """Optional private information transferred into a fresh Env run."""

    init: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "init", json_object(self.init, "scenario.init"))


def start_env(
    env: Env,
    init: Mapping[str, Any] | None,
    workspace: Path,
) -> ToolResult | None:
    """Invoke optional startup before callers discover the episode's tools."""

    start = getattr(env, "start", None)
    if start is None:
        return None
    if not callable(start):
        raise TypeError("Env.start must be callable when present")
    result = start(init, workspace)
    if result is not None and not isinstance(result, ToolResult):
        raise TypeError("Env.start must return ToolResult or None")
    return result


def close_env(env: Env) -> None:
    """Invoke the optional teardown hook."""

    close = getattr(env, "close", None)
    if close is None:
        return
    if not callable(close):
        raise TypeError("Env.close must be callable when present")
    close()


__all__ = [
    "Env",
    "EnvironmentSpec",
    "RuleSpec",
    "ReplayVerifier",
    "Scenario",
    "VerificationCheck",
    "VerificationResult",
    "close_env",
    "json_object",
    "start_env",
]
