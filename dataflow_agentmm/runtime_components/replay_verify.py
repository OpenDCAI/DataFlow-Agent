"""Independent strict-replay verification operator core."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from ..contracts import (
    ContentLimits,
    Env,
    ReplayVerifierResolver,
    TaskResolver,
    TextContent,
    Trajectory,
    VerificationResult,
    close_env,
    start_env,
)
from ..env.registry import make_env
from .host import HostPolicy
from .tool_loop import ToolLoop, _safe_result


ReplayStatus = Literal["passed", "failed", "diverged", "not_applicable", "error"]


@dataclass(frozen=True)
class ReplayVerification:
    status: ReplayStatus
    passed: bool
    reward: float
    reason: str
    replayed_steps: int
    divergence_step: int | None = None
    checks: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "reward": self.reward,
            "reason": self.reason,
            "replayed_steps": self.replayed_steps,
            "divergence_step": self.divergence_step,
            "checks": [dict(check) for check in self.checks],
        }


@dataclass(frozen=True)
class ReplayVerifyConfig:
    workspace_root: Path | None = None
    content_limits: ContentLimits = ContentLimits()
    max_observation_chars: int = 8000


class ReplayVerify:
    """Resolve a Task, replay its actions, then run its task-bound verifier."""

    def __init__(
        self,
        *,
        task_resolver: TaskResolver,
        verifier_resolver: ReplayVerifierResolver,
        env_resolver: Callable[[str], Env] = make_env,
        config: ReplayVerifyConfig = ReplayVerifyConfig(),
    ):
        self.task_resolver = task_resolver
        self.verifier_resolver = verifier_resolver
        self.env_resolver = env_resolver
        self.config = config

    @staticmethod
    def _result(
        status: ReplayStatus,
        *,
        reason: str,
        replayed_steps: int = 0,
        divergence_step: int | None = None,
        verification: VerificationResult | None = None,
    ) -> ReplayVerification:
        return ReplayVerification(
            status=status,
            passed=verification.passed if verification is not None else status == "passed",
            reward=verification.reward if verification is not None else 0.0,
            reason=verification.reason if verification is not None else reason,
            replayed_steps=replayed_steps,
            divergence_step=divergence_step,
            checks=(
                tuple(check.to_dict() for check in verification.checks)
                if verification is not None
                else ()
            ),
        )

    @staticmethod
    def _assistant_text(rollout: Trajectory, message_index: int) -> str:
        if message_index < 0 or message_index >= len(rollout.messages):
            raise IndexError("response_message_index is outside messages")
        message = rollout.messages[message_index]
        if message.role != "assistant" or len(message.content) != 1:
            raise ValueError("response message must be one assistant text block")
        content = message.content[0]
        if not isinstance(content, TextContent):
            raise ValueError("response message must contain text")
        return content.text

    @staticmethod
    def _control(result: Any) -> tuple[bool, str | None, bool]:
        return (
            bool(result.ok),
            result.error.code if result.error is not None else None,
            bool(result.is_final),
        )

    def verify(self, rollout: Trajectory) -> ReplayVerification:
        try:
            task = self.task_resolver.resolve(
                rollout.task_id, env_id=rollout.env_id
            )
        except Exception as exc:
            return self._result(
                "error", reason=f"task resolution failed: {type(exc).__name__}: {exc}"
            )
        if task.task_id != rollout.task_id or task.env_id != rollout.env_id:
            return self._result("error", reason="resolved task identity mismatch")
        try:
            verifier_factory = self.verifier_resolver.verifier_factory(
                rollout.task_id,
                env_id=rollout.env_id,
            )
        except Exception as exc:
            return self._result(
                "error",
                reason=f"verifier resolution failed: {type(exc).__name__}: {exc}",
            )
        if verifier_factory is None:
            return self._result(
                "not_applicable",
                reason="task has no replay verifier",
            )

        root = self.config.workspace_root
        if root is not None:
            Path(root).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f"agentmm-replay-{rollout.env_id}-",
            dir=str(root) if root else None,
        ) as temporary:
            workspace = Path(temporary).resolve()
            env: Env | None = None
            replayed = 0
            try:
                env = self.env_resolver(rollout.env_id)
                initial = start_env(
                    env,
                    task.scenario.init if task.scenario is not None else None,
                    workspace,
                )
                if initial is not None:
                    initial = _safe_result(
                        initial,
                        content_limits=self.config.content_limits,
                        max_observation_chars=self.config.max_observation_chars,
                    )
                    if not initial.ok:
                        code = initial.error.code if initial.error is not None else "unknown"
                        raise RuntimeError(f"Env.start failed with {code}")
                    if initial.is_final:
                        raise RuntimeError("Env.start must not terminate an episode")

                host_enabled = bool(rollout.metadata.get("host_tools", False))
                loop = ToolLoop(
                    env,
                    host_policy=(
                        HostPolicy(
                            workspace=workspace,
                            content_limits=self.config.content_limits,
                        )
                        if host_enabled
                        else None
                    ),
                    validate_tool_names=bool(
                        rollout.metadata.get("validate_tool_names", True)
                    ),
                    content_limits=self.config.content_limits,
                    max_observation_chars=self.config.max_observation_chars,
                )
                terminal_kind: str | None = None
                terminal_answer: str | None = None
                previous_response_index = -1
                for ordinal, step in enumerate(rollout.steps, start=1):
                    if terminal_kind is not None:
                        return self._result(
                            "diverged",
                            reason=f"action appears after terminal {terminal_kind}",
                            replayed_steps=replayed,
                            divergence_step=step.index,
                        )
                    if step.index != ordinal:
                        return self._result(
                            "diverged",
                            reason=f"non-contiguous step index {step.index}, expected {ordinal}",
                            replayed_steps=replayed,
                            divergence_step=step.index,
                        )
                    if step.response_message_index <= previous_response_index:
                        return self._result(
                            "diverged",
                            reason="assistant response references are not strictly increasing",
                            replayed_steps=replayed,
                            divergence_step=step.index,
                        )
                    previous_response_index = step.response_message_index
                    try:
                        response = self._assistant_text(
                            rollout, step.response_message_index
                        )
                    except Exception as exc:
                        return self._result(
                            "diverged",
                            reason=f"invalid response reference: {exc}",
                            replayed_steps=replayed,
                            divergence_step=step.index,
                        )
                    parsed = loop.parse_action(response)
                    if step.parse_error:
                        if parsed is not None or step.action is not None:
                            return self._result(
                                "diverged",
                                reason="recorded parse error no longer parses as an error",
                                replayed_steps=replayed,
                                divergence_step=step.index,
                            )
                        result = loop.parse_failure()
                        execution_terminal = None
                        execution_answer = None
                    else:
                        if parsed is None or step.action is None or dict(parsed) != dict(step.action):
                            return self._result(
                                "diverged",
                                reason="stored action does not match the assistant response",
                                replayed_steps=replayed,
                                divergence_step=step.index,
                            )
                        execution = loop.execute(parsed)
                        result = execution.result
                        execution_terminal = execution.terminal_kind
                        execution_answer = execution.final_answer

                    expected = (step.tool_ok, step.error_code, step.is_final)
                    actual = self._control(result)
                    if expected != actual:
                        return self._result(
                            "diverged",
                            reason=f"control result mismatch: expected={expected}, actual={actual}",
                            replayed_steps=replayed,
                            divergence_step=step.index,
                        )
                    replayed += 1
                    terminal_kind = execution_terminal
                    terminal_answer = execution_answer

                if terminal_kind is None and rollout.termination_reason in {
                    "finish", "environment_final"
                }:
                    return self._result(
                        "diverged",
                        reason="rollout claims terminal completion without a terminal action",
                        replayed_steps=replayed,
                    )
                if terminal_kind is None and rollout.final_answer is not None:
                    return self._result(
                        "diverged",
                        reason="non-terminal rollout carries a final answer",
                        replayed_steps=replayed,
                    )
                if terminal_kind is not None:
                    if rollout.termination_reason != terminal_kind:
                        return self._result(
                            "diverged",
                            reason=(
                                "termination reason mismatch: "
                                f"{rollout.termination_reason!r} != {terminal_kind!r}"
                            ),
                            replayed_steps=replayed,
                        )
                    if terminal_kind == "finish" and rollout.final_answer != terminal_answer:
                        return self._result(
                            "diverged",
                            reason="finish answer does not match the replayed action",
                            replayed_steps=replayed,
                        )
                    if terminal_kind == "environment_final" and rollout.final_answer is not None:
                        return self._result(
                            "diverged",
                            reason="environment-final rollout must not carry a finish answer",
                            replayed_steps=replayed,
                        )

                verifier = verifier_factory()
                verification = verifier.verify(env, rollout)
                if not isinstance(verification, VerificationResult):
                    raise TypeError("replay verifier must return VerificationResult")
                return self._result(
                    "passed" if verification.passed else "failed",
                    reason=verification.reason,
                    replayed_steps=replayed,
                    verification=verification,
                )
            except Exception as exc:
                return self._result(
                    "error",
                    reason=f"replay failed: {type(exc).__name__}: {exc}",
                    replayed_steps=replayed,
                )
            finally:
                if env is not None:
                    try:
                        close_env(env)
                    except Exception:
                        pass


__all__ = [
    "ReplayStatus",
    "ReplayVerification",
    "ReplayVerify",
    "ReplayVerifyConfig",
]
