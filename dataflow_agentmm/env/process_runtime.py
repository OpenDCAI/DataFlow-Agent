"""Process-isolated adapters for concrete DataFlow-AgentMM environment packs.

The DataFlow-AgentMM process owns orchestration and model serving.  Concrete Env packs
are imported only by a worker started with the dedicated Env interpreter.  A
small newline-delimited JSON protocol carries descriptions, lifecycle calls,
tool calls, and live verifier requests across that boundary.
"""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..contracts import EnvironmentSpec, ToolResult, ToolSpec, VerificationResult
from ..contracts.trajectory import Trajectory
from .registry import register_env


ENV_MODE_VARIABLE = "DATAFLOW_AGENTMM_ENV_MODE"
ENV_PYTHON_VARIABLE = "DATAFLOW_AGENTMM_ENV_PYTHON"
ENV_NAME_VARIABLE = "DATAFLOW_AGENTMM_ENV_NAME"
ENV_WORKER_VARIABLE = "DATAFLOW_AGENTMM_ENV_WORKER"
ENV_TIMEOUT_VARIABLE = "DATAFLOW_AGENTMM_ENV_RPC_TIMEOUT"
DEFAULT_ENV_NAME = "dataflow-agentmm-envs"
DEFAULT_RPC_TIMEOUT = 180.0
_PROTOCOL_VERSION = 2

_MODEL_ROUTING_VARIABLES = {
    "API_URL",
    "BASE_URL",
    "MODEL",
    "SERVING_BACKEND",
    "KIGRESS_BIZ_SCENE",
    "KIGRESS_LLM_MODEL",
}
_SECRET_VARIABLE_SUFFIXES = (
    "_API_KEY",
    "_CREDENTIAL",
    "_CREDENTIALS",
    "_PASSWORD",
    "_SECRET",
    "_TOKEN",
    "_USER_KEY",
)


class EnvironmentProcessError(RuntimeError):
    """Raised when the dedicated Env worker cannot satisfy an RPC request."""


def should_use_isolated_environments() -> bool:
    """Return whether repository Env registration must use the worker process.

    Isolation is the default.  ``inprocess`` is retained only for the worker
    itself and focused Env-pack development/tests.
    """

    if os.environ.get(ENV_WORKER_VARIABLE) == "1":
        return False
    mode = os.environ.get(ENV_MODE_VARIABLE, "process").strip().lower()
    if mode == "process":
        return True
    if mode == "inprocess":
        return False
    raise ValueError(
        f"{ENV_MODE_VARIABLE} must be 'process' or 'inprocess', got {mode!r}"
    )


def _python_candidate(prefix: Path, env_name: str) -> Path:
    executable = "python.exe" if os.name == "nt" else "python"
    scripts = "Scripts" if os.name == "nt" else "bin"
    return prefix / env_name / scripts / executable


def resolve_isolated_env_python(
    value: str | os.PathLike[str] | None = None,
    *,
    require_distinct: bool = True,
) -> Path:
    """Resolve and validate the interpreter used for concrete Env workers."""

    explicit = value or os.environ.get(ENV_PYTHON_VARIABLE)
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    else:
        env_name = os.environ.get(ENV_NAME_VARIABLE, DEFAULT_ENV_NAME).strip()
        if not env_name:
            raise ValueError(f"{ENV_NAME_VARIABLE} must be non-empty")

        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            active = Path(conda_prefix).expanduser()
            candidates.append(_python_candidate(active.parent, env_name))
            candidates.append(_python_candidate(active.parent.parent / "envs", env_name))

        current_prefix = Path(sys.prefix)
        candidates.append(_python_candidate(current_prefix.parent, env_name))
        candidates.append(
            _python_candidate(current_prefix.parent.parent / "envs", env_name)
        )

        conda_exe = os.environ.get("CONDA_EXE")
        if conda_exe:
            conda_root = Path(conda_exe).expanduser().resolve().parent.parent
            candidates.append(_python_candidate(conda_root / "envs", env_name))

    checked: list[str] = []
    selected: Path | None = None
    for candidate in candidates:
        resolved = candidate.resolve()
        label = str(resolved)
        if label in checked:
            continue
        checked.append(label)
        if resolved.is_file() and os.access(resolved, os.X_OK):
            selected = resolved
            break
    if selected is None:
        hint = ", ".join(checked) or "<none>"
        raise FileNotFoundError(
            "dedicated Env Python was not found; create the shared Env Conda "
            f"or set {ENV_PYTHON_VARIABLE}. Checked: {hint}"
        )

    if require_distinct and selected == Path(sys.executable).resolve():
        raise EnvironmentProcessError(
            "Env Python resolves to the DataFlow-AgentMM interpreter; process mode "
            "requires two distinct Conda environments"
        )
    return selected


def _rpc_timeout() -> float:
    raw = os.environ.get(ENV_TIMEOUT_VARIABLE)
    if raw is None:
        return DEFAULT_RPC_TIMEOUT
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{ENV_TIMEOUT_VARIABLE} must be numeric") from exc
    if value <= 0:
        raise ValueError(f"{ENV_TIMEOUT_VARIABLE} must be positive")
    return value


def _is_model_or_secret_variable(name: str) -> bool:
    """Return whether a parent variable must not enter an Env worker."""

    normalized = name.strip().upper()
    return normalized in _MODEL_ROUTING_VARIABLES or normalized.endswith(
        _SECRET_VARIABLE_SUFFIXES
    )


@dataclass(frozen=True)
class _WorkerLauncher:
    python: Path
    plugin_modules: tuple[str, ...]
    project_root: Path
    timeout: float

    def command(self) -> list[str]:
        command = [
            str(self.python),
            "-B",
            "-m",
            "dataflow_agentmm.env.worker",
        ]
        for module in self.plugin_modules:
            command.extend(("--plugin-module", module))
        return command

    def environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        for name in tuple(environment):
            if _is_model_or_secret_variable(name):
                environment.pop(name, None)
        environment[ENV_WORKER_VARIABLE] = "1"
        environment[ENV_MODE_VARIABLE] = "inprocess"
        environment["PYTHONNOUSERSITE"] = "1"
        environment.pop("PYTHONHOME", None)
        environment.pop("VIRTUAL_ENV", None)

        source_roots = (
            self.project_root,
            self.project_root / "dataflow-agentmm",
        )
        environment["PYTHONPATH"] = os.pathsep.join(
            str(path.resolve()) for path in source_roots if path.exists()
        )

        env_prefix = self.python.parent.parent
        environment["CONDA_PREFIX"] = str(env_prefix)
        environment["CONDA_DEFAULT_ENV"] = env_prefix.name
        env_bin = str(self.python.parent)
        remaining_path = [
            item
            for item in environment.get("PATH", "").split(os.pathsep)
            if item and Path(item).resolve() != self.python.parent.resolve()
        ]
        environment["PATH"] = os.pathsep.join((env_bin, *remaining_path))
        return environment


class _WorkerClient:
    def __init__(self, launcher: _WorkerLauncher):
        self.launcher = launcher
        self._lock = threading.Lock()
        self._request_id = 0
        self._closed = False
        try:
            self._process = subprocess.Popen(
                launcher.command(),
                cwd=launcher.project_root,
                env=launcher.environment(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                bufsize=0,
                start_new_session=True,
            )
        except OSError as exc:
            raise EnvironmentProcessError(
                f"cannot start Env worker with {launcher.python}: {exc}"
            ) from exc
        if self._process.stdin is None or self._process.stdout is None:
            self._terminate()
            raise EnvironmentProcessError("Env worker pipes were not created")

    @property
    def pid(self) -> int:
        return self._process.pid

    def request(self, operation: str, **payload: Any) -> Any:
        with self._lock:
            if self._closed:
                raise EnvironmentProcessError("Env worker is already closed")
            return self._request_locked(operation, payload)

    def _request_locked(self, operation: str, payload: Mapping[str, Any]) -> Any:
        return_code = self._process.poll()
        if return_code is not None:
            raise EnvironmentProcessError(
                f"Env worker exited before {operation!r} (code={return_code})"
            )
        self._request_id += 1
        request_id = self._request_id
        request = {
            "protocol_version": _PROTOCOL_VERSION,
            "id": request_id,
            "op": operation,
            **dict(payload),
        }
        encoded = (json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8")
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(encoded)
        except (BrokenPipeError, OSError) as exc:
            raise EnvironmentProcessError(
                f"cannot write {operation!r} to Env worker: {exc}"
            ) from exc

        assert self._process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(self._process.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + self.launcher.timeout
        ignored_lines: list[str] = []
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    suffix = (
                        f"; non-protocol stdout={ignored_lines[-3:]!r}"
                        if ignored_lines
                        else ""
                    )
                    raise EnvironmentProcessError(
                        f"Env worker timed out during {operation!r}{suffix}"
                    )
                line = self._process.stdout.readline()
                if not line:
                    raise EnvironmentProcessError(
                        f"Env worker closed stdout during {operation!r} "
                        f"(code={self._process.poll()})"
                    )
                try:
                    response = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    ignored_lines.append(line.decode("utf-8", errors="replace").strip())
                    continue
                if not isinstance(response, Mapping) or response.get("id") != request_id:
                    ignored_lines.append(str(response)[:500])
                    continue
                if response.get("protocol_version") != _PROTOCOL_VERSION:
                    raise EnvironmentProcessError("Env worker protocol version mismatch")
                if response.get("ok") is True:
                    return response.get("result")
                error = response.get("error")
                if isinstance(error, Mapping):
                    error_type = str(error.get("type") or "WorkerError")
                    message = str(error.get("message") or "unknown Env worker error")
                    detail = str(error.get("traceback") or "").strip()
                    suffix = f"\n{detail}" if detail else ""
                    raise EnvironmentProcessError(
                        f"Env worker {error_type} during {operation!r}: {message}{suffix}"
                    )
                raise EnvironmentProcessError(
                    f"Env worker failed during {operation!r}: {error!r}"
                )
        finally:
            selector.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                if self._process.poll() is None:
                    self._request_locked("shutdown", {})
            except Exception:
                pass
            finally:
                self._closed = True
                self._terminate()

    def _terminate(self) -> None:
        if self._process.poll() is None:
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)
        else:
            self._process.wait()


def _environment_spec(value: Mapping[str, Any]) -> EnvironmentSpec:
    return EnvironmentSpec.from_dict(value)


def _launcher(
    *,
    env_python: str | os.PathLike[str] | None,
    plugin_modules: Sequence[str],
    project_root: str | os.PathLike[str],
    require_distinct: bool,
) -> _WorkerLauncher:
    modules = tuple(str(item).strip() for item in plugin_modules if str(item).strip())
    if not modules:
        raise ValueError("at least one Env plugin module is required")
    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Env project root does not exist: {root}")
    return _WorkerLauncher(
        python=resolve_isolated_env_python(
            env_python,
            require_distinct=require_distinct,
        ),
        plugin_modules=modules,
        project_root=root,
        timeout=_rpc_timeout(),
    )


def _one_shot(launcher: _WorkerLauncher, operation: str, **payload: Any) -> Any:
    client = _WorkerClient(launcher)
    try:
        return client.request(operation, **payload)
    finally:
        client.close()


class IsolatedEnv:
    """One episode-scoped proxy backed by a persistent Env worker process."""

    def __init__(
        self,
        *,
        env_id: str,
        spec: EnvironmentSpec,
        launcher: _WorkerLauncher,
    ):
        self.env_id = env_id
        self.spec = spec
        self._launcher = launcher
        self._client = _WorkerClient(launcher)
        self._closed = False
        try:
            identity = self._client.request("create_env", env_id=env_id)
        except Exception:
            self._client.close()
            raise
        if not isinstance(identity, Mapping):
            self._client.close()
            raise EnvironmentProcessError("Env worker returned an invalid identity")
        self.runtime_identity = dict(identity)
        if self.runtime_identity.get("env_id") != env_id:
            self._client.close()
            raise EnvironmentProcessError("Env worker created the wrong environment")

    def tools(self) -> Sequence[ToolSpec]:
        values = self._client.request("tools")
        if not isinstance(values, list):
            raise EnvironmentProcessError("Env worker returned an invalid tool catalog")
        return tuple(
            ToolSpec(
                name=str(value["name"]),
                description=str(value["description"]),
                operation_type=str(value["operation_type"]),  # type: ignore[arg-type]
                input_schema=dict(value["input_schema"]),
            )
            for value in values
        )

    def start(
        self,
        init: Mapping[str, Any] | None,
        workspace: Path,
    ) -> ToolResult | None:
        result = self._client.request(
            "start",
            init=dict(init) if init is not None else None,
            workspace=str(Path(workspace).resolve()),
        )
        if result is None:
            return None
        if not isinstance(result, Mapping):
            raise EnvironmentProcessError("Env worker returned an invalid start result")
        return ToolResult.from_dict(result)

    def call(self, tool_name: str, args: Mapping[str, Any]) -> ToolResult:
        result = self._client.request(
            "call",
            tool_name=tool_name,
            args=dict(args),
        )
        if not isinstance(result, Mapping):
            raise EnvironmentProcessError("Env worker returned an invalid tool result")
        return ToolResult.from_dict(result)

    def verify_task(
        self,
        binding: Mapping[str, Any],
        rollout: Trajectory,
    ) -> VerificationResult:
        result = self._client.request(
            "verify",
            binding=dict(binding),
            trajectory=rollout.to_dict(),
        )
        if not isinstance(result, Mapping):
            raise EnvironmentProcessError("Env worker returned invalid verification")
        return VerificationResult.from_dict(result)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._client.request("close_env")
        except Exception:
            pass
        finally:
            self._client.close()


@dataclass(frozen=True)
class IsolatedEnvFactory:
    env_id: str
    spec: EnvironmentSpec
    launcher: _WorkerLauncher

    def __call__(self) -> IsolatedEnv:
        return IsolatedEnv(
            env_id=self.env_id,
            spec=self.spec,
            launcher=self.launcher,
        )


def register_isolated_environments(
    *,
    plugin_modules: Sequence[str],
    project_root: str | os.PathLike[str],
    env_python: str | os.PathLike[str] | None = None,
    require_distinct: bool = True,
) -> tuple[str, ...]:
    """Discover Env registrations in the dedicated interpreter and proxy them."""

    launcher = _launcher(
        env_python=env_python,
        plugin_modules=plugin_modules,
        project_root=project_root,
        require_distinct=require_distinct,
    )
    description = _one_shot(launcher, "describe")
    if not isinstance(description, Mapping):
        raise EnvironmentProcessError("Env worker returned an invalid description")
    environments = description.get("environments")
    if not isinstance(environments, list) or not environments:
        raise EnvironmentProcessError("Env worker discovered no environments")

    registered: list[str] = []
    for item in environments:
        if not isinstance(item, Mapping) or not isinstance(item.get("spec"), Mapping):
            raise EnvironmentProcessError("Env worker returned a malformed descriptor")
        spec = _environment_spec(item["spec"])
        register_env(
            spec.env_id,
            IsolatedEnvFactory(
                env_id=spec.env_id,
                spec=spec,
                launcher=launcher,
            ),
            name=spec.name,
            description=spec.description,
            rules=spec.rules,
            modalities=spec.modalities,
        )
        registered.append(spec.env_id)
    return tuple(registered)


__all__ = [
    "DEFAULT_ENV_NAME",
    "ENV_MODE_VARIABLE",
    "ENV_NAME_VARIABLE",
    "ENV_PYTHON_VARIABLE",
    "EnvironmentProcessError",
    "IsolatedEnv",
    "IsolatedEnvFactory",
    "register_isolated_environments",
    "resolve_isolated_env_python",
    "should_use_isolated_environments",
]
