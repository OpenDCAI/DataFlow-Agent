"""DataFlow wrapper for independent ReplayVerify."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage

from ..contracts import ContentLimits, Env, ReplayVerifierResolver, TaskResolver
from ..env.registry import make_env
from ..runtime_components import ReplayVerify, ReplayVerifyConfig
from .utils.trajectory import as_trajectory


@OPERATOR_REGISTRY.register()
class AgentMMReplayVerifier(OperatorABC):
    """Write replay results beside, never into, the original trajectory."""

    def __init__(
        self,
        *,
        task_resolver: TaskResolver,
        verifier_resolver: ReplayVerifierResolver,
        env_resolver: Callable[[str], Env] = make_env,
        max_workers: int = 8,
        workspace_root: str | Path | None = None,
        content_limits: ContentLimits = ContentLimits(),
    ):
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.logger = get_logger()
        self.max_workers = max_workers
        self.replay = ReplayVerify(
            task_resolver=task_resolver,
            verifier_resolver=verifier_resolver,
            env_resolver=env_resolver,
            config=ReplayVerifyConfig(
                workspace_root=Path(workspace_root) if workspace_root else None,
                content_limits=content_limits,
            ),
        )

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        if lang == "zh":
            return "严格重放轨迹控制结果，仅在完全一致后运行独立解析的 verifier。"
        return "Strictly replays control results, then runs an independently resolved verifier."

    def _verify_one(self, value: Any) -> dict[str, Any]:
        trajectory = as_trajectory(value)
        if trajectory is None:
            raise ValueError("trajectory is not a canonical DataFlow-AgentMM v2 trajectory")
        return self.replay.verify(trajectory).to_dict()

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "trajectory",
        output_key: str = "replay_verification",
    ):
        dataframe = storage.read(output_type="dataframe")
        if input_key not in dataframe.columns:
            raise KeyError(
                f"input_key {input_key!r} not found in columns: "
                f"{list(dataframe.columns)}"
            )
        values = dataframe[input_key].tolist()
        results: list[dict[str, Any] | None] = [None] * len(values)
        if self.max_workers == 1:
            results = [self._verify_one(value) for value in values]
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {
                    pool.submit(self._verify_one, value): index
                    for index, value in enumerate(values)
                }
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        results[index] = future.result()
                    except Exception as exc:
                        results[index] = {
                            "status": "error",
                            "passed": False,
                            "reward": 0.0,
                            "reason": f"worker crashed: {type(exc).__name__}: {exc}",
                            "replayed_steps": 0,
                            "divergence_step": None,
                            "checks": [],
                        }
        finalized = [item for item in results if item is not None]
        dataframe[output_key] = finalized
        storage.write(dataframe)
        status_counts: dict[str, int] = {}
        for item in finalized:
            status = str(item.get("status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        summary = ", ".join(
            f"{status}={count}" for status, count in sorted(status_counts.items())
        )
        self.logger.info(
            f"[AgentMMReplayVerifier] {len(finalized)} result(s): {summary}"
        )
        return [output_key]


__all__ = ["AgentMMReplayVerifier"]
