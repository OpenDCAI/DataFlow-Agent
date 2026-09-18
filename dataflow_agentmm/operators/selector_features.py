"""Named trajectory features for :class:`AgentMMTrajectorySelector`.

A feature is ``fn(trajectory, row) -> value``: ``trajectory`` is the canonical
trajectory dict and ``row`` the rest of its storage record (for example
``replay_verification`` or ``traj_overall``). Every built-in feature below is a
public function, so custom features can combine them, e.g. into a weighted score::

    from dataflow_agentmm.operators import register_selector_feature, selector_features as sf

    @register_selector_feature("quality_score")
    def quality_score(trajectory, row):
        return 0.4 * min(sf.num_steps(trajectory, row) / 5, 1) + 0.6 * sf.replay_passed(trajectory, row)
"""

from __future__ import annotations

import json
import math
from typing import Any, Callable, Mapping

from .utils.trajectory import observation_text, steps

FeatureFn = Callable[[Mapping[str, Any], Mapping[str, Any]], Any]

SELECTOR_FEATURES: dict[str, FeatureFn] = {}


def register_selector_feature(name: str, fn: FeatureFn | None = None, *, overwrite: bool = False):
    """Register ``fn(trajectory, row)`` under ``name``; usable as a decorator."""

    def register(function: FeatureFn) -> FeatureFn:
        if not callable(function):
            raise TypeError("selector feature must be callable")
        if name in SELECTOR_FEATURES and not overwrite:
            raise ValueError(f"selector feature {name!r} is already registered")
        SELECTOR_FEATURES[name] = function
        return function

    return register(fn) if fn is not None else register


def actions(trajectory: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Parsed actions in step order."""
    return [step["action"] for step in steps(trajectory) if isinstance(step.get("action"), Mapping)]


def _tool_actions(trajectory: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [a for a in actions(trajectory) if a.get("tool") and a.get("tool") != "finish"]


def _content_blocks(trajectory: Mapping[str, Any]):
    for message in trajectory.get("messages") or ():
        for block in message.get("content") or ():
            if isinstance(block, Mapping):
                yield block


def num_steps(trajectory: Mapping[str, Any], _row: Mapping[str, Any]) -> int:
    """Recorded steps, including parse failures and ``finish``."""
    return len(steps(trajectory))


def num_tool_calls(trajectory: Mapping[str, Any], _row: Mapping[str, Any]) -> int:
    """Env tool calls, excluding ``finish`` and unparsed responses."""
    return len(_tool_actions(trajectory))


def num_distinct_tools(trajectory: Mapping[str, Any], _row: Mapping[str, Any]) -> int:
    return len({a["tool"] for a in _tool_actions(trajectory)})


def num_tool_errors(trajectory: Mapping[str, Any], _row: Mapping[str, Any]) -> int:
    return sum(step.get("tool_ok") is False for step in steps(trajectory))


def num_invalid_tool_calls(trajectory: Mapping[str, Any], _row: Mapping[str, Any]) -> int:
    """Calls to a tool the Env does not expose."""
    return sum(step.get("error_code") == "unknown_tool" for step in steps(trajectory))


def max_repeated_action(trajectory: Mapping[str, Any], _row: Mapping[str, Any]) -> int:
    """How often the most repeated tool call (same name and arguments) occurs."""
    counts: dict[str, int] = {}
    for action in _tool_actions(trajectory):
        try:
            args = json.dumps(action.get("args", {}), sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            args = str(action.get("args"))
        key = f"{action.get('tool')}:{args}"
        counts[key] = counts.get(key, 0) + 1
    return max(counts.values(), default=0)


def num_parse_errors(trajectory: Mapping[str, Any], _row: Mapping[str, Any]) -> int:
    return sum(bool(step.get("parse_error")) for step in steps(trajectory))


def num_messages(trajectory: Mapping[str, Any], _row: Mapping[str, Any]) -> int:
    return len(trajectory.get("messages") or ())


def text_chars(trajectory: Mapping[str, Any], _row: Mapping[str, Any]) -> int:
    """Characters of text content across all messages."""
    return sum(len(str(b.get("text") or "")) for b in _content_blocks(trajectory) if b.get("type") == "text")


def num_images(trajectory: Mapping[str, Any], _row: Mapping[str, Any]) -> int:
    return sum(b.get("type") == "image" for b in _content_blocks(trajectory))


def avg_observation_len(trajectory: Mapping[str, Any], _row: Mapping[str, Any]) -> float:
    """Mean observation text length per step (0 for no steps)."""
    trajectory_steps = steps(trajectory)
    if not trajectory_steps:
        return 0.0
    return sum(len(observation_text(trajectory, step)) for step in trajectory_steps) / len(trajectory_steps)


def has_final_answer(trajectory: Mapping[str, Any], _row: Mapping[str, Any]) -> bool:
    answer = trajectory.get("final_answer")
    return isinstance(answer, str) and bool(answer.strip())


def termination_reason(trajectory: Mapping[str, Any], _row: Mapping[str, Any]) -> Any:
    return trajectory.get("termination_reason")


def is_finish(trajectory: Mapping[str, Any], _row: Mapping[str, Any]) -> bool:
    return trajectory.get("termination_reason") == "finish"


def is_success(trajectory: Mapping[str, Any], _row: Mapping[str, Any]) -> bool:
    """Ended by ``finish`` or by an Env-final result."""
    return trajectory.get("termination_reason") in ("finish", "environment_final")


def replay_status(_trajectory: Mapping[str, Any], row: Mapping[str, Any]) -> Any:
    """``replay_verification.status`` from the storage row, or ``None``."""
    value = row.get("replay_verification")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    return value.get("status") if isinstance(value, Mapping) else None


def replay_passed(trajectory: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    return replay_status(trajectory, row) == "passed"


def judge_score(_trajectory: Mapping[str, Any], row: Mapping[str, Any]) -> float | None:
    """Finite ``traj_overall`` from the storage row, or ``None``."""
    value = row.get("traj_overall")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def uses_tool(tool: str, *, min_calls: int = 1, successful: bool = False) -> FeatureFn:
    """Feature factory: ``tool`` was called at least ``min_calls`` times (optionally with ``tool_ok``)."""

    def feature(trajectory: Mapping[str, Any], _row: Mapping[str, Any]) -> bool:
        calls = sum(
            isinstance(step.get("action"), Mapping)
            and step["action"].get("tool") == tool
            and (not successful or step.get("tool_ok") is True)
            for step in steps(trajectory)
        )
        return calls >= min_calls

    return feature


BUILTIN_FEATURES: dict[str, FeatureFn] = {
    fn.__name__: fn
    for fn in (
        num_steps, num_tool_calls, num_distinct_tools, num_tool_errors,
        num_invalid_tool_calls, num_parse_errors, max_repeated_action, num_messages,
        text_chars, num_images, avg_observation_len, has_final_answer,
        termination_reason, is_finish, is_success, replay_status, replay_passed,
        judge_score,
    )
}
for _name, _fn in BUILTIN_FEATURES.items():
    register_selector_feature(_name, _fn)


__all__ = [
    "BUILTIN_FEATURES",
    "FeatureFn",
    "SELECTOR_FEATURES",
    "actions",
    "register_selector_feature",
    "uses_tool",
    *BUILTIN_FEATURES,
]
