"""Rule-based trajectory selector with optional feature switches.

Every built-in feature from :mod:`selector_features` is an optional keyword;
registered custom features are passed the same way. A condition is either a
value (``is_finish=True`` means equality) or a mapping of comparisons
(``num_steps={"gte": 2, "lte": 40}``). A trajectory is kept only when every
given condition holds; omitted switches are not applied.

Optional post-processing, in order and per ``group_by`` group: sort by one
feature (``sort_by``), drop near-duplicate action sets (``dedupe_threshold``),
and keep at most ``max_selected``. Weighted ranking is not built in: register a
feature that combines the public built-in feature functions and pass it as a
condition or as ``sort_by``.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage

from .selector_features import SELECTOR_FEATURES, FeatureFn, actions
from .utils.trajectory import as_trajectory_dict

_COMPARISONS = {
    "eq": lambda actual, expected: actual == expected,
    "ne": lambda actual, expected: actual != expected,
    "gt": lambda actual, expected: actual > expected,
    "gte": lambda actual, expected: actual >= expected,
    "lt": lambda actual, expected: actual < expected,
    "lte": lambda actual, expected: actual <= expected,
    "in": lambda actual, expected: actual in expected,
    "not_in": lambda actual, expected: actual not in expected,
}


def _condition(name: str, value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping) and value and set(value) <= set(_COMPARISONS):
        return dict(value)
    if isinstance(value, Mapping):
        raise ValueError(
            f"condition {name!r} must be a value or a mapping of {sorted(_COMPARISONS)}; got keys {sorted(value)}"
        )
    return {"eq": value}


def _holds(actual: Any, comparisons: Mapping[str, Any]) -> bool:
    try:
        return all(_COMPARISONS[op](actual, expected) for op, expected in comparisons.items())
    except TypeError:  # e.g. None compared with a number
        return False


def _action_set(trajectory: Mapping[str, Any]) -> set[str]:
    signatures = set()
    for action in actions(trajectory):
        try:
            args = json.dumps(action.get("args", {}), sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            args = str(action.get("args"))
        signatures.add(f"{action.get('tool')}({args})")
    return signatures


def _jaccard(left: set[str], right: set[str]) -> float:
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def _loads(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


@OPERATOR_REGISTRY.register()
class AgentMMTrajectorySelector(OperatorABC):
    """Keep trajectories satisfying every given feature condition."""

    def __init__(
        self,
        *,
        num_steps: Any = None,
        num_tool_calls: Any = None,
        num_distinct_tools: Any = None,
        num_tool_errors: Any = None,
        num_parse_errors: Any = None,
        num_messages: Any = None,
        text_chars: Any = None,
        num_images: Any = None,
        avg_observation_len: Any = None,
        termination_reason: Any = None,
        is_finish: Any = None,
        is_success: Any = None,
        replay_status: Any = None,
        replay_passed: Any = None,
        judge_score: Any = None,
        features: Mapping[str, FeatureFn] | None = None,
        sort_by: str | None = None,
        descending: bool = True,
        max_selected: int | None = None,
        dedupe_threshold: float | None = None,
        group_by: str | Sequence[str] | None = None,
        input_format: str = "auto",
        **conditions: Any,
    ):
        """Built-in switches and ``**conditions`` default to off; ``None`` never filters.

        ``features`` adds instance-local features on top of the registered ones.
        """

        builtin = {
            "num_steps": num_steps, "num_tool_calls": num_tool_calls,
            "num_distinct_tools": num_distinct_tools, "num_tool_errors": num_tool_errors,
            "num_parse_errors": num_parse_errors, "num_messages": num_messages,
            "text_chars": text_chars, "num_images": num_images,
            "avg_observation_len": avg_observation_len, "termination_reason": termination_reason,
            "is_finish": is_finish, "is_success": is_success, "replay_status": replay_status,
            "replay_passed": replay_passed, "judge_score": judge_score,
        }
        self.features = {**SELECTOR_FEATURES, **dict(features or {})}
        requested = {name: value for name, value in {**builtin, **conditions}.items() if value is not None}
        unknown = sorted((set(requested) | ({sort_by} if sort_by else set())) - set(self.features))
        if unknown:
            raise KeyError(f"unknown selector features: {unknown}; register them or pass features=")
        if input_format not in ("auto", "rows", "tree"):
            raise ValueError("input_format must be 'auto', 'rows', or 'tree'")
        if max_selected is not None and (isinstance(max_selected, bool) or not isinstance(max_selected, int) or max_selected < 1):
            raise ValueError("max_selected must be a positive integer")
        if dedupe_threshold is not None and not 0 <= dedupe_threshold <= 1:
            raise ValueError("dedupe_threshold must be within [0, 1]")
        self.logger = get_logger()
        self.conditions = {name: _condition(name, value) for name, value in requested.items()}
        self.sort_by = sort_by
        self.descending = descending
        self.max_selected = max_selected
        self.dedupe_threshold = dedupe_threshold
        self.group_by = (group_by,) if isinstance(group_by, str) else tuple(group_by or ())
        self.input_format = input_format

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        if lang == "zh":
            return "按可选字段条件（内置或自定义注册）交集筛选轨迹，可选排序、去重、分组与数量上限。"
        return "Keeps trajectories meeting every optional feature condition, with optional sort, dedupe, grouping, and cap."

    def feature_values(self, trajectory: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
        names = list(self.conditions) + ([self.sort_by] if self.sort_by and self.sort_by not in self.conditions else [])
        return {name: self.features[name](trajectory, row) for name in names}

    def select(
        self,
        trajectories: Sequence[Mapping[str, Any]],
        rows: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[tuple[int, dict[str, Any]]]:
        """Return ``(index, feature values)`` for kept trajectories in output order."""

        rows = list(rows) if rows is not None else [{} for _ in trajectories]
        values = [self.feature_values(t, r) for t, r in zip(trajectories, rows)]
        groups: dict[tuple, list[int]] = {}
        for index, (trajectory, row) in enumerate(zip(trajectories, rows)):
            if all(_holds(values[index][name], cmp) for name, cmp in self.conditions.items()):
                key = tuple(row.get(name, trajectory.get(name)) for name in self.group_by)
                groups.setdefault(key, []).append(index)

        kept: list[int] = []
        for members in groups.values():
            if self.sort_by:
                present = [i for i in members if values[i][self.sort_by] is not None]
                missing = [i for i in members if values[i][self.sort_by] is None]
                present.sort(key=lambda i: values[i][self.sort_by], reverse=self.descending)
                members = present + missing  # stable: ties and missing values keep input order
            chosen: list[int] = []
            chosen_sets: list[set[str]] = []
            for index in members:
                if self.max_selected is not None and len(chosen) >= self.max_selected:
                    break
                if self.dedupe_threshold is not None:
                    current = _action_set(trajectories[index])
                    if any(_jaccard(current, prior) > self.dedupe_threshold for prior in chosen_sets):
                        continue
                    chosen_sets.append(current)
                chosen.append(index)
            kept.extend(chosen)
        return [(index, values[index]) for index in kept]

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "trajectory",
        output_key: str = "selected_trajectories",
    ):
        dataframe = storage.read(output_type="dataframe")
        if input_key not in dataframe.columns:
            raise KeyError(f"input_key {input_key!r} not found in columns: {list(dataframe.columns)}")
        records = dataframe.to_dict(orient="records")
        input_format = self.input_format
        if input_format == "auto":
            sample = next((_loads(r[input_key]) for r in records if r.get(input_key) is not None), None)
            input_format = "tree" if isinstance(sample, Mapping) and "paths" in sample else "rows"

        if input_format == "tree":
            selected_lists: list[list[dict[str, Any]]] = []
            for record in records:
                tree = _loads(record.get(input_key))
                raw_paths = tree.get("paths") if isinstance(tree, Mapping) else None
                paths = [p for raw in raw_paths or () if (p := as_trajectory_dict(raw)) is not None]
                chosen = self.select(paths, [record] * len(paths))
                selected_lists.append([{**paths[i], "_select_features": v} for i, v in chosen])
            dataframe[output_key] = selected_lists
            dataframe[f"{output_key}_count"] = [len(items) for items in selected_lists]
            storage.write(dataframe)
            return [output_key]

        trajectories: list[Mapping[str, Any]] = []
        row_indices: list[int] = []
        for index, record in enumerate(records):
            trajectory = as_trajectory_dict(record.get(input_key))
            if trajectory is not None:
                trajectories.append(trajectory)
                row_indices.append(index)
        chosen = self.select(trajectories, [records[i] for i in row_indices])
        result = dataframe.iloc[[row_indices[i] for i, _ in chosen]].reset_index(drop=True)
        result["select_features"] = [v for _, v in chosen]
        storage.write(result)
        self.logger.info(f"[AgentMMTrajectorySelector] kept {len(result)}/{len(records)} rows")
        return [input_key]


__all__ = ["AgentMMTrajectorySelector"]
