"""DataFlow exploration operator over required v2 Tasks."""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage

from ..contracts import Message, Task, TaskResolver
from ..contracts.trajectory import Trajectory, utc_now
from ..runtime_components import AgentRollout, RolloutConfig
from ..serving import ModelServing
from ..storage import TrajectoryStore

_SAFE_KEY = re.compile(r"[^A-Za-z0-9._-]+")


@OPERATOR_REGISTRY.register()
class AgentMMExploreGenerator(OperatorABC):
    """Generate one tool-loop trajectory per explicitly resolved Task."""

    def __init__(
        self,
        serving: ModelServing | None = None,
        *,
        task_resolver: TaskResolver,
        max_steps: int = 64,
        max_workers: int = 8,
        system_prompt: str | None = None,
        include_tool_catalog: bool = True,
        max_observation_chars: int = 8000,
        validate_tool_names: bool = True,
        structured_actions: bool = False,
        action_format_retries: int = 0,
        include_host_tools: bool = False,
        trajectory_dir: str | Path | None = None,
        workspace_root: str | Path | None = None,
        workspace_retention: str = "ephemeral",
        checkpoint_dir: str | Path | None = None,
        max_retries: int = 0,
    ):
        """``checkpoint_dir`` saves each finished row immediately as
        ``<sample key>.jsonl`` and skips rows whose checkpoint is already a
        non-infrastructure result. ``trajectory_dir`` additionally outputs
        checkpoint paths instead of trajectory objects and doubles as the
        checkpoint directory when ``checkpoint_dir`` is unset. ``max_retries``
        re-runs rows that raise or end in ``infrastructure_error``.
        """
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        self.logger = get_logger()
        self.serving = serving
        self.task_resolver = task_resolver
        self.max_workers = max_workers
        defaults = RolloutConfig()
        self.config = RolloutConfig(
            max_steps=max_steps,
            system_prompt=system_prompt or defaults.system_prompt,
            include_host_tools=include_host_tools,
            include_tool_catalog=include_tool_catalog,
            validate_tool_names=validate_tool_names,
            structured_actions=structured_actions,
            action_format_retries=action_format_retries,
            max_observation_chars=max_observation_chars,
            workspace_root=Path(workspace_root) if workspace_root else None,
            workspace_retention=workspace_retention,  # type: ignore[arg-type]
        )
        self.trajectory_dir = Path(trajectory_dir) if trajectory_dir else None
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.trajectory_dir
        self.max_retries = max_retries

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        if lang == "zh":
            return "解析必需 Task，在轻量 Env 的统一 ToolLoop 中生成多模态轨迹。"
        return "Resolves required Tasks and generates multimodal ToolLoop trajectories."

    def _runner(self) -> AgentRollout:
        if self.serving is None:
            raise ValueError("AgentMMExploreGenerator requires a serving instance")
        return AgentRollout(serving=self.serving, config=self.config)

    def _task(
        self,
        record: dict[str, Any],
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
        record: dict[str, Any],
        *,
        env_key: str,
        task_key: str,
        input_key: str | None,
    ) -> Trajectory:
        return self._runner().run(self._task(
            record, env_key=env_key, task_key=task_key, input_key=input_key
        ))

    def _sample_keys(
        self,
        records: list[dict[str, Any]],
        *,
        env_key: str,
        task_key: str,
        sample_key: str | None,
    ) -> list[str]:
        """Stable per-row identities for checkpoint files and resume."""

        keys: list[str] = []
        seen: dict[str, int] = {}
        for record in records:
            if sample_key is not None:
                raw = str(record[sample_key])
            else:
                base = f"{record[env_key]}__{record[task_key]}"
                occurrence = seen.get(base, 0)
                seen[base] = occurrence + 1
                raw = f"{base}__{occurrence}"
            key = _SAFE_KEY.sub("_", raw).strip("._") or "sample"
            if key in keys:
                raise ValueError(f"duplicate sample key after sanitizing: {raw!r}")
            keys.append(key)
        return keys

    def _attempt(
        self,
        index: int,
        record: dict[str, Any],
        *,
        env_key: str,
        task_key: str,
        input_key: str | None,
    ) -> Trajectory:
        """Run one row, retrying exceptions and infrastructure failures."""

        errors: list[dict[str, Any]] = []
        last_failure: Trajectory | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                trajectory = self._run_record(
                    record, env_key=env_key, task_key=task_key, input_key=input_key
                )
            except Exception as exc:
                self.logger.error(
                    f"[AgentMMExploreGenerator] episode {index} attempt {attempt} failed: {exc}"
                )
                errors.append({"attempt": attempt, "error": f"{type(exc).__name__}: {exc}"})
                last_failure = None
                continue
            if trajectory.termination_reason != "infrastructure_error":
                if errors:
                    trajectory = replace(trajectory, metadata={
                        **trajectory.metadata, "attempts": attempt, "attempt_errors": errors,
                    })
                return trajectory
            errors.append({"attempt": attempt, "error": "infrastructure_error"})
            last_failure = trajectory
        if last_failure is not None:
            return replace(last_failure, metadata={
                **last_failure.metadata, "attempts": len(errors), "attempt_errors": errors,
            })
        task = self._task(record, env_key=env_key, task_key=task_key, input_key=input_key)
        timestamp = utc_now()
        return Trajectory(
            episode_id=f"{task.task_id}-failed-{index}",
            task_id=task.task_id,
            env_id=task.env_id,
            messages=task.messages,
            steps=(),
            final_answer=None,
            termination_reason="infrastructure_error",
            started_at=timestamp,
            completed_at=timestamp,
            metadata={
                "error": errors[-1]["error"],
                "attempts": len(errors),
                "attempt_errors": errors,
            },
        )

    def _load_checkpoint(self, path: Path) -> Trajectory | None:
        if not path.is_file():
            return None
        try:
            trajectory = TrajectoryStore().load(path)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.logger.warning(f"[AgentMMExploreGenerator] ignoring unreadable checkpoint {path}: {exc}")
            return None
        return None if trajectory.termination_reason == "infrastructure_error" else trajectory

    def _write_manifest(self, directory: Path, entry: dict[str, Any]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "run_manifest.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str | None = None,
        output_key: str = "trajectory",
        env_key: str = "env_id",
        task_key: str = "task_id",
        sample_key: str | None = None,
    ):
        if self.serving is None:
            raise ValueError("AgentMMExploreGenerator requires a serving instance")
        dataframe = storage.read(output_type="dataframe")
        for required in (env_key, task_key, *((sample_key,) if sample_key else ())):
            if required not in dataframe.columns:
                raise KeyError(f"missing required input column: {required}")
        records = dataframe.to_dict(orient="records")
        results: list[Trajectory | None] = [None] * len(records)
        checkpoint = self.checkpoint_dir
        paths: list[Path | None] = [None] * len(records)
        started_at = utc_now()
        resumed = 0
        if checkpoint is not None:
            keys = self._sample_keys(records, env_key=env_key, task_key=task_key, sample_key=sample_key)
            paths = [checkpoint / f"{key}.jsonl" for key in keys]
            for index, path in enumerate(paths):
                results[index] = self._load_checkpoint(path)
            resumed = sum(item is not None for item in results)
        pending = [index for index, item in enumerate(results) if item is None]
        store = TrajectoryStore()

        def finish(index: int, trajectory: Trajectory) -> None:
            results[index] = trajectory
            if paths[index] is not None:
                store.save(trajectory, paths[index])

        def attempt(index: int) -> Trajectory:
            return self._attempt(
                index, records[index], env_key=env_key, task_key=task_key, input_key=input_key
            )

        if self.max_workers == 1:
            for index in pending:
                finish(index, attempt(index))
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {pool.submit(attempt, index): index for index in pending}
                for future in as_completed(futures):
                    finish(futures[future], future.result())

        trajectories = [item for item in results if item is not None]
        if self.trajectory_dir is not None:
            outputs: list[Any] = [str(path) for path in paths]
        else:
            outputs = [trajectory.to_dict() for trajectory in trajectories]
        dataframe[output_key] = outputs
        storage.write(dataframe)
        completed = sum(item.success for item in trajectories)
        failed = sum(item.termination_reason == "infrastructure_error" for item in trajectories)
        if checkpoint is not None:
            usage: dict[str, int] = {}
            for index in pending:
                for name, value in dict(results[index].metadata.get("usage") or {}).items():
                    usage[name] = usage.get(name, 0) + int(value)
            self._write_manifest(checkpoint, {
                "started_at": started_at,
                "completed_at": utc_now(),
                "rows": len(records),
                "resumed": resumed,
                "ran": len(pending),
                "completed": completed,
                "infrastructure_errors": failed,
                "usage": usage,
                "config": {
                    "model": getattr(self.serving, "model", None),
                    "max_steps": self.config.max_steps,
                    "max_retries": self.max_retries,
                    "max_workers": self.max_workers,
                    "structured_actions": self.config.structured_actions,
                    "action_format_retries": self.config.action_format_retries,
                    "include_host_tools": self.config.include_host_tools,
                    "system_prompt_sha256": hashlib.sha256(
                        self.config.system_prompt.encode("utf-8")
                    ).hexdigest(),
                },
            })
        self.logger.info(
            f"[AgentMMExploreGenerator] {completed}/{len(trajectories)} completed "
            f"({resumed} resumed, {len(pending)} ran, {failed} infrastructure errors)"
        )
        return [output_key]


__all__ = ["AgentMMExploreGenerator"]
