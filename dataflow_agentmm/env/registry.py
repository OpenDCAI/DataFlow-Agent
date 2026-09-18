"""Lightweight Env registry and plugin discovery."""

from __future__ import annotations

import importlib
import os
import threading
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Callable, Sequence

from ..contracts import Env, EnvironmentSpec, RuleSpec, close_env


EnvFactory = Callable[[], Env]


@dataclass(frozen=True)
class EnvironmentRegistration:
    """Exactly the information needed to construct and describe an Env."""

    spec: EnvironmentSpec
    factory: EnvFactory

    def __post_init__(self) -> None:
        if not callable(self.factory):
            raise TypeError("environment factory must be callable")


ENVIRONMENTS: dict[str, EnvironmentRegistration] = {}
ENVIRONMENT_ENTRY_POINT_GROUP = "dataflow_agentmm.environments"
_LOADED_ENTRY_POINTS: set[str] = set()
_LOADED_MODULES: set[str] = set()
_REGISTRY_LOCK = threading.RLock()


def register_env(
    env_id: str,
    factory: EnvFactory,
    *,
    description: str,
    rules: Sequence[RuleSpec] = (),
    modalities: Sequence[str] = ("text",),
    name: str | None = None,
) -> EnvironmentRegistration:
    """Register a factory without coupling it to tasks or verifiers."""

    registration = EnvironmentRegistration(
        spec=EnvironmentSpec(
            env_id=env_id,
            name=name or env_id,
            description=description,
            rules=tuple(rules),
            modalities=tuple(modalities),
        ),
        factory=factory,
    )
    with _REGISTRY_LOCK:
        if env_id in ENVIRONMENTS:
            raise ValueError(f"environment {env_id!r} is already registered")
        ENVIRONMENTS[env_id] = registration
    return registration


def _call_registration_hook(value: object, source: str) -> None:
    if not callable(value):
        raise TypeError(f"environment plugin {source!r} must expose a callable")
    value()


def load_environment_plugins(*, modules: tuple[str, ...] | None = None) -> None:
    """Load installed Env plugins and optional source-tree plugin modules."""

    with _REGISTRY_LOCK:
        requested = modules
        if requested is None:
            requested = tuple(
                item.strip()
                for item in os.environ.get(
                    "DATAFLOW_AGENTMM_ENV_PLUGINS", ""
                ).split(",")
                if item.strip()
            )
        for module_name in requested:
            if module_name in _LOADED_MODULES:
                continue
            module = importlib.import_module(module_name)
            _call_registration_hook(getattr(module, "register", None), module_name)
            _LOADED_MODULES.add(module_name)

        for plugin in entry_points(group=ENVIRONMENT_ENTRY_POINT_GROUP):
            identity = f"{plugin.name}:{plugin.value}"
            if identity in _LOADED_ENTRY_POINTS:
                continue
            _call_registration_hook(plugin.load(), identity)
            _LOADED_ENTRY_POINTS.add(identity)


def get_environment(env_id: str) -> EnvironmentRegistration:
    if env_id not in ENVIRONMENTS:
        load_environment_plugins()
    try:
        return ENVIRONMENTS[env_id]
    except KeyError as exc:
        raise KeyError(
            f"unknown environment {env_id!r}; available: {sorted(ENVIRONMENTS)}"
        ) from exc


def get_environment_spec(env_id: str) -> EnvironmentSpec:
    return get_environment(env_id).spec


def make_env(env_id: str) -> Env:
    env = get_environment(env_id).factory()
    if not isinstance(env, Env):
        try:
            close_env(env)  # type: ignore[arg-type]
        finally:
            raise TypeError(
                f"factory for {env_id!r} must return an object with tools() and call()"
            )
    return env


__all__ = [
    "ENVIRONMENTS",
    "ENVIRONMENT_ENTRY_POINT_GROUP",
    "EnvironmentRegistration",
    "EnvFactory",
    "get_environment",
    "get_environment_spec",
    "load_environment_plugins",
    "make_env",
    "register_env",
]
