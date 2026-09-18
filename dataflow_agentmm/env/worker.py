"""Dedicated-interpreter worker for process-isolated DataFlow-AgentMM Env packs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..contracts import VerificationResult, close_env, start_env
from ..contracts.trajectory import Trajectory
from ..storage.task_store import StatePredicateVerifier
from .process_runtime import ENV_WORKER_VARIABLE
from .registry import ENVIRONMENTS, get_environment, load_environment_plugins


_PROTOCOL_VERSION = 2


class _Worker:
    def __init__(self, plugin_modules: Sequence[str]):
        self.plugin_modules = tuple(plugin_modules)
        self._loaded = False
        self._env = None
        self._env_id: str | None = None

    def _load(self) -> None:
        if self._loaded:
            return
        load_environment_plugins(modules=self.plugin_modules)
        self._loaded = True

    @staticmethod
    def _runtime_identity(env_id: str | None = None) -> dict[str, Any]:
        value: dict[str, Any] = {
            "pid": os.getpid(),
            "python_executable": str(Path(sys.executable).resolve()),
            "python_prefix": str(Path(sys.prefix).resolve()),
            "protocol_version": _PROTOCOL_VERSION,
        }
        if env_id is not None:
            value["env_id"] = env_id
        return value

    def _registration(self, env_id: str):
        self._load()
        return get_environment(env_id)

    def dispatch(self, request: Mapping[str, Any]) -> tuple[Any, bool]:
        operation = request.get("op")
        if operation == "shutdown":
            self._close_env()
            return self._runtime_identity(self._env_id), True

        if operation == "describe":
            self._load()
            environments = []
            for env_id, registration in sorted(ENVIRONMENTS.items()):
                environments.append({
                    "spec": registration.spec.to_dict(),
                })
            return {
                **self._runtime_identity(),
                "environments": environments,
            }, False

        if operation == "create_env":
            env_id = str(request["env_id"])
            self._close_env()
            registration = self._registration(env_id)
            env = registration.factory()
            if not callable(getattr(env, "tools", None)) or not callable(
                getattr(env, "call", None)
            ):
                close_env(env)
                raise TypeError(f"environment factory returned invalid {env_id!r}")
            self._env = env
            self._env_id = env_id
            return self._runtime_identity(env_id), False

        if operation == "close_env":
            identity = self._runtime_identity(self._env_id)
            self._close_env()
            return identity, False

        if operation == "tools":
            env = self._require_env()
            return [tool.to_dict() for tool in env.tools()], False

        if operation == "start":
            env = self._require_env()
            init = request.get("init")
            if init is not None and not isinstance(init, Mapping):
                raise TypeError("init must be an object or null")
            workspace = Path(str(request["workspace"])).resolve()
            result = start_env(env, dict(init) if init is not None else None, workspace)
            return result.to_dict() if result is not None else None, False

        if operation == "call":
            env = self._require_env()
            args = request.get("args") or {}
            if not isinstance(args, Mapping):
                raise TypeError("tool args must be an object")
            return env.call(str(request["tool_name"]), dict(args)).to_dict(), False

        if operation == "verify":
            env = self._require_env()
            binding = request.get("binding")
            if not isinstance(binding, Mapping):
                raise TypeError("verification binding must be an object")
            trajectory = Trajectory.from_dict(request["trajectory"])
            verify_task = getattr(env, "verify_task", None)
            if callable(verify_task):
                result = verify_task(dict(binding), trajectory)
            elif binding.get("kind") == "state_predicate":
                snapshot = getattr(env, "snapshot", None)
                if not callable(snapshot):
                    raise TypeError("state predicate verifier requires snapshot()")
                value = snapshot()
                if not isinstance(value, Mapping):
                    raise TypeError("snapshot must be an object")
                result = StatePredicateVerifier.evaluate(binding, value)
            else:
                raise TypeError(
                    f"Env cannot verify binding kind {binding.get('kind')!r}"
                )
            if not isinstance(result, VerificationResult):
                raise TypeError("Env verifier must return VerificationResult")
            return result.to_dict(), False

        raise KeyError(f"unsupported Env worker operation: {operation!r}")

    def _require_env(self):
        if self._env is None:
            raise RuntimeError("no environment has been created in this worker")
        return self._env

    def _close_env(self) -> None:
        env = self._env
        self._env = None
        self._env_id = None
        if env is not None:
            close_env(env)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--plugin-module",
        action="append",
        default=[],
        help="Python module exposing a register() hook; may be repeated.",
    )
    return value


def main(argv: list[str] | None = None) -> int:
    os.environ[ENV_WORKER_VARIABLE] = "1"
    args = parser().parse_args(argv)
    worker = _Worker(args.plugin_module)
    protocol_output = sys.stdout
    stop = False
    try:
        for raw_line in sys.stdin:
            request_id: Any = None
            try:
                request = json.loads(raw_line)
                if not isinstance(request, Mapping):
                    raise TypeError("request must be a JSON object")
                request_id = request.get("id")
                if request.get("protocol_version") != _PROTOCOL_VERSION:
                    raise ValueError("protocol version mismatch")
                with redirect_stdout(sys.stderr):
                    result, stop = worker.dispatch(request)
                response = {
                    "protocol_version": _PROTOCOL_VERSION,
                    "id": request_id,
                    "ok": True,
                    "result": result,
                }
            except Exception as exc:
                response = {
                    "protocol_version": _PROTOCOL_VERSION,
                    "id": request_id,
                    "ok": False,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(limit=12),
                    },
                }
            protocol_output.write(json.dumps(response, ensure_ascii=False) + "\n")
            protocol_output.flush()
            if stop:
                break
    finally:
        with redirect_stdout(sys.stderr):
            worker._close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
