"""Branching multimodal exploration over replayable controlled Envs."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage

from ..contracts import Message, Task, TaskResolver
from ..contracts.trajectory import Trajectory
from ..runtime_components import AgentRollout, RolloutConfig
from ..serving import ModelServing


_TERMINAL_REASONS = {"finish", "environment_final", "infrastructure_error"}


@OPERATOR_REGISTRY.register()
class AgentMMExploreTreeGenerator(OperatorABC):
    """Explore several actions per node and emit canonical leaf trajectories.

    The upstream DataFlow-Agent tree explorer samples and de-duplicates actions
    but executes sibling actions in one shared stateful sandbox session. This
    implementation keeps the useful search shape while reconstructing every
    child from a fresh Env by replaying its complete response prefix through
    :class:`AgentRollout`. It is therefore correct for stateful deterministic
    Envs without requiring checkpoint/restore support.

    Prefix replay costs O(number_of_nodes * average_depth). A future Env
    checkpoint capability may optimize that cost without changing this output
    contract.
    """

    def __init__(
        self,
        serving: ModelServing | None = None,
        *,
        task_resolver: TaskResolver,
        max_depth: int = 64,
        branching_factor: int = 3,
        max_children: int | None = None,
        max_nodes: int = 64,
        depth_threshold: int | None = None,
        deduplicate_actions: bool = True,
        max_workers: int = 4,
        system_prompt: str | None = None,
        include_tool_catalog: bool = True,
        max_observation_chars: int = 8000,
        validate_tool_names: bool = True,
        include_host_tools: bool = False,
        workspace_root: str | Path | None = None,
        workspace_retention: str = "ephemeral",
    ):
        if not isinstance(include_host_tools, bool):
            raise TypeError("include_host_tools must be a boolean")
        if (
            isinstance(max_depth, bool)
            or not isinstance(max_depth, int)
            or max_depth < 1
        ):
            raise ValueError("max_depth must be positive")
        for name, value in (
            ("branching_factor", branching_factor),
            ("max_nodes", max_nodes),
            ("max_workers", max_workers),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if max_children is not None and (
            isinstance(max_children, bool)
            or not isinstance(max_children, int)
            or max_children < 1
        ):
            raise ValueError("max_children must be positive or None")
        if depth_threshold is not None and (
            isinstance(depth_threshold, bool)
            or not isinstance(depth_threshold, int)
            or depth_threshold < 0
        ):
            raise ValueError("depth_threshold must be non-negative or None")

        self.logger = get_logger()
        self.serving = serving
        self.task_resolver = task_resolver
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.max_children = max_children or branching_factor
        self.max_nodes = max_nodes
        self.depth_threshold = depth_threshold
        self.deduplicate_actions = deduplicate_actions
        self.max_workers = max_workers
        self.system_prompt = system_prompt
        self.include_tool_catalog = include_tool_catalog
        self.max_observation_chars = max_observation_chars
        self.validate_tool_names = validate_tool_names
        self.include_host_tools = include_host_tools
        self.workspace_root = Path(workspace_root) if workspace_root else None
        self.workspace_retention = workspace_retention

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        if lang == "zh":
            return (
                "从同一多模态消息节点批量采样候选 action，通过 fresh Env 前缀重放"
                "构建状态正确的探索树，并输出标准叶子轨迹。"
            )
        return (
            "Builds a state-correct multimodal exploration tree with fresh-Env "
            "prefix replay and emits canonical leaf trajectories."
        )

    @staticmethod
    def _candidate_key(action: Mapping[str, Any] | None, response: str) -> str:
        if action is None:
            return "parse_error:" + response.strip()
        return json.dumps(
            {
                "tool": action.get("tool"),
                "args": action.get("args") or {},
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _observation_for_last_step(
        trajectory: Trajectory,
    ) -> Mapping[str, Any] | None:
        if not trajectory.steps:
            return None
        index = trajectory.steps[-1].observation_message_index
        if index is None or not 0 <= index < len(trajectory.messages):
            return None
        return trajectory.messages[index].to_dict()

    @classmethod
    def _child_node(
        cls,
        *,
        node_id: str,
        depth: int,
        response: str,
        action: Mapping[str, Any] | None,
        trajectory: Trajectory,
    ) -> dict[str, Any]:
        step = trajectory.steps[-1] if trajectory.steps else None
        return {
            "node_id": node_id,
            "depth": depth,
            "thought": action.get("thought") if action is not None else None,
            "action": dict(action) if action is not None else None,
            "raw_response": response if action is None else None,
            "parse_error": bool(step and step.parse_error),
            "tool_ok": step.tool_ok if step else None,
            "error_code": step.error_code if step else None,
            "observation": cls._observation_for_last_step(trajectory),
            "terminal": None,
            "children": [],
        }

    def _runner(self) -> AgentRollout:
        if self.serving is None:
            raise ValueError(
                "AgentMMExploreTreeGenerator requires a serving instance"
            )
        defaults = RolloutConfig()
        return AgentRollout(
            serving=self.serving,
            config=RolloutConfig(
                max_steps=self.max_depth,
                system_prompt=self.system_prompt or defaults.system_prompt,
                include_host_tools=self.include_host_tools,
                include_tool_catalog=self.include_tool_catalog,
                validate_tool_names=self.validate_tool_names,
                max_observation_chars=self.max_observation_chars,
                workspace_root=self.workspace_root,
                workspace_retention=self.workspace_retention,
            ),
        )

    def _run_task(self, task: Task) -> dict[str, Any]:
        depth_limit = self.max_depth
        runner = self._runner()
        root_trajectory = runner.run_responses(
            task,
            (),
            exhaustion_reason="max_steps",
        )
        root: dict[str, Any] = {
            "node_id": "root",
            "depth": 0,
            "terminal": None,
            "children": [],
        }
        counters = {"nodes": 0, "next_id": 1}
        leaves: list[Trajectory] = []

        def add_leaf(trajectory: Trajectory) -> None:
            leaves.append(trajectory)

        def next_node_id() -> str:
            value = f"node{counters['next_id']:06d}"
            counters["next_id"] += 1
            return value

        def expand(
            node: dict[str, Any],
            responses_prefix: tuple[str, ...],
            prefix_trajectory: Trajectory,
            depth: int,
        ) -> None:
            if prefix_trajectory.termination_reason in _TERMINAL_REASONS:
                node["terminal"] = prefix_trajectory.termination_reason
                add_leaf(prefix_trajectory)
                return
            if depth >= depth_limit:
                node["terminal"] = "depth_limit"
                add_leaf(prefix_trajectory)
                return
            if counters["nodes"] >= self.max_nodes:
                node["terminal"] = "node_budget"
                add_leaf(prefix_trajectory)
                return

            try:
                sampled = runner.sample_responses(
                    prefix_trajectory.messages,
                    self.branching_factor,
                )
            except Exception as exc:
                node["terminal"] = "sampling_error"
                node["error"] = f"{type(exc).__name__}: {exc}"
                add_leaf(prefix_trajectory)
                return

            candidates: list[tuple[str, Mapping[str, Any] | None]] = []
            seen: set[str] = set()
            child_cap = (
                1
                if self.depth_threshold is not None
                and depth >= self.depth_threshold
                else self.max_children
            )
            for response in sampled:
                action = runner.parse_action(response)
                key = self._candidate_key(action, response)
                if self.deduplicate_actions and key in seen:
                    continue
                seen.add(key)
                candidates.append((response, action))
                if len(candidates) >= child_cap:
                    break

            if not candidates:
                node["terminal"] = "no_candidates"
                add_leaf(prefix_trajectory)
                return

            for response, action in candidates:
                if counters["nodes"] >= self.max_nodes:
                    node["truncated"] = "node_budget"
                    break
                counters["nodes"] += 1
                child_depth = depth + 1
                child_prefix = responses_prefix + (response,)
                child_trajectory = runner.run_responses(
                    task,
                    child_prefix,
                    exhaustion_reason="max_steps",
                )
                child = self._child_node(
                    node_id=next_node_id(),
                    depth=child_depth,
                    response=response,
                    action=action,
                    trajectory=child_trajectory,
                )
                node["children"].append(child)

                if child_trajectory.termination_reason in _TERMINAL_REASONS:
                    child["terminal"] = child_trajectory.termination_reason
                    add_leaf(child_trajectory)
                elif child_depth >= depth_limit:
                    child["terminal"] = "depth_limit"
                    add_leaf(child_trajectory)
                elif counters["nodes"] >= self.max_nodes:
                    child["terminal"] = "node_budget"
                    add_leaf(child_trajectory)
                else:
                    expand(
                        child,
                        child_prefix,
                        child_trajectory,
                        child_depth,
                    )

            if not node["children"] and node.get("terminal") is None:
                node["terminal"] = "node_budget"
                add_leaf(prefix_trajectory)

        expand(root, (), root_trajectory, 0)
        path_values = [trajectory.to_dict() for trajectory in leaves]
        return {
            "schema_version": 2,
            "task": task.to_dict(),
            "tree": root,
            "paths": path_values,
            "num_nodes": counters["nodes"],
            "num_paths": len(leaves),
            "num_success_paths": sum(item.success for item in leaves),
            "max_depth": depth_limit,
            "branching_factor": self.branching_factor,
            "branch_state_strategy": "fresh_env_prefix_replay",
        }

    def _task_from_record(
        self,
        record: Mapping[str, Any],
        *,
        env_key: str,
        task_key: str,
        input_key: str | None,
    ) -> Task:
        task = self.task_resolver.resolve(
            str(record[task_key]), env_id=str(record[env_key])
        )
        if input_key and record.get(input_key) is not None:
            task = replace(
                task,
                messages=(Message.text("user", str(record[input_key])),),
            )
        return task

    def _run_record(
        self,
        record: Mapping[str, Any],
        *,
        env_key: str,
        task_key: str,
        input_key: str | None,
    ) -> dict[str, Any]:
        task = self._task_from_record(
            record,
            env_key=env_key,
            task_key=task_key,
            input_key=input_key,
        )
        return self._run_task(task)

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str | None = None,
        output_key: str = "tree",
        env_key: str = "env_id",
        task_key: str = "task_id",
    ):
        if self.serving is None:
            raise ValueError(
                "AgentMMExploreTreeGenerator requires a serving instance"
            )
        dataframe = storage.read(output_type="dataframe")
        required_keys: Sequence[str] = (env_key, task_key)
        for required in required_keys:
            if required not in dataframe.columns:
                raise KeyError(f"missing required input column: {required}")
        records = dataframe.to_dict(orient="records")
        results: list[dict[str, Any] | None] = [None] * len(records)

        def run_index(index: int) -> dict[str, Any]:
            try:
                return self._run_record(
                    records[index],
                    env_key=env_key,
                    task_key=task_key,
                    input_key=input_key,
                )
            except Exception as exc:
                self.logger.error(
                    f"[AgentMMExploreTreeGenerator] task {index} failed: {exc}"
                )
                return {
                    "schema_version": 2,
                    "tree": None,
                    "paths": [],
                    "num_nodes": 0,
                    "num_paths": 0,
                    "num_success_paths": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }

        if self.max_workers == 1:
            results = [run_index(index) for index in range(len(records))]
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {
                    pool.submit(run_index, index): index
                    for index in range(len(records))
                }
                for future in as_completed(futures):
                    results[futures[future]] = future.result()

        complete = [item for item in results if item is not None]
        dataframe[output_key] = complete
        storage.write(dataframe)
        total_paths = sum(item["num_paths"] for item in complete)
        self.logger.info(
            f"[AgentMMExploreTreeGenerator] built {len(complete)} trees, "
            f"{total_paths} leaf paths"
        )
        return [output_key]


__all__ = ["AgentMMExploreTreeGenerator"]
