"""Small JSON Task store with explicit verifier binding."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..contracts import (
    Env,
    ImageContent,
    ReplayVerifier,
    ReplayVerifierFactory,
    Scenario,
    Task,
    TextContent,
    VerificationCheck,
    VerificationResult,
    validate_task_id,
)
from ..contracts.environment import json_object
from ..contracts.trajectory import Trajectory


STATE_PREDICATE_VERIFIER_KIND = "state_predicate"
MAX_TEXT_REF_BYTES = 512 * 1024
TEXT_REF_MEDIA_TYPES = frozenset({"text/plain", "text/markdown"})
PREDICATE_OPERATORS = frozenset(
    {"equals", "not_equals", "gte", "lte", "empty", "not_empty"}
)
VerifierBuilder = Callable[
    [Mapping[str, Any]],
    ReplayVerifier,
]


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


class JsonTaskStore:
    """Resolve one strict v2 Task per ``<task_id>.json`` file.

    Verifier descriptors remain inert JSON and separate from Scenario. A
    caller must supply a trusted builder for their ``kind``; ``verifier_factory``
    creates a fresh instance for every ReplayVerify call.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        verifier_builders: Mapping[str, VerifierBuilder] | None = None,
    ):
        self.root = Path(root).resolve()
        self.verifier_builders = dict(verifier_builders or {})

    def _validate_record(
        self,
        value: Mapping[str, Any],
        *,
        task_id: str,
        source: str,
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError(f"task must be a JSON object: {source}")
        record = dict(value)
        allowed = {
            "schema_version",
            "task_id",
            "env_id",
            "messages",
            "scenario",
            "verification",
            "judge_ref",
        }
        extra = set(record).difference(allowed)
        if extra:
            raise ValueError(
                f"unsupported v2 task fields in {source}: {sorted(extra)}"
            )
        if record.get("schema_version") != 2:
            raise ValueError(f"task is not schema_version 2: {source}")
        if record.get("task_id") != task_id:
            raise ValueError(f"task id/filename mismatch in {source}")
        return record

    def _task_from_record(
        self,
        value: Mapping[str, Any],
        *,
        task_id: str,
        source: str,
    ) -> Task:
        record = self._validate_record(value, task_id=task_id, source=source)
        messages = record.get("messages")
        if not isinstance(messages, list):
            raise TypeError("task.messages must be a list")
        materialized_messages = _json_copy(messages)
        for message in materialized_messages:
            if not isinstance(message, Mapping):
                raise TypeError("task message must be an object")
            content_items = message.get("content")
            if not isinstance(content_items, list):
                raise TypeError("task message content must be a list")
            for index, content in enumerate(content_items):
                if not isinstance(content, Mapping) or content.get("type") not in {
                    "image_ref",
                    "text_ref",
                }:
                    continue
                relative = Path(str(content.get("relative_path") or ""))
                if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(
                        f"{content.get('type')} must be a confined relative path"
                    )
                path = (self.root / relative).resolve(strict=True)
                if not path.is_relative_to(self.root):
                    raise ValueError(f"{content.get('type')} escapes the task store")
                payload = path.read_bytes()
                expected = str(content.get("sha256") or "")
                if hashlib.sha256(payload).hexdigest() != expected:
                    raise ValueError(
                        f"{content.get('type')} sha256 mismatch: {relative}"
                    )
                if content.get("type") == "image_ref":
                    content_items[index] = ImageContent.from_bytes(
                        payload,
                        str(content["media_type"]),
                        detail=str(content.get("detail") or "original"),
                    ).to_dict()
                    continue

                extra = set(content).difference(
                    {"type", "relative_path", "media_type", "sha256", "encoding"}
                )
                if extra:
                    raise ValueError(
                        f"text_ref has unsupported fields: {sorted(extra)}"
                    )
                media_type = content.get("media_type")
                if media_type not in TEXT_REF_MEDIA_TYPES:
                    raise ValueError(
                        "text_ref media_type must be text/plain or text/markdown"
                    )
                encoding = content.get("encoding", "utf-8")
                if encoding != "utf-8":
                    raise ValueError("text_ref encoding must be utf-8")
                if len(payload) > MAX_TEXT_REF_BYTES:
                    raise ValueError(
                        f"text_ref exceeds {MAX_TEXT_REF_BYTES} bytes: {relative}"
                    )
                try:
                    text = payload.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(f"text_ref is not valid utf-8: {relative}") from exc
                content_items[index] = TextContent(text).to_dict()
        record["messages"] = materialized_messages
        raw_scenario = record.get("scenario")
        scenario: Scenario | None = None
        if raw_scenario is not None:
            if not isinstance(raw_scenario, Mapping):
                raise TypeError("task.scenario must be an object or null")
            extra = set(raw_scenario).difference({"init"})
            if extra:
                raise ValueError(f"unsupported scenario fields: {sorted(extra)}")
            if "init" not in raw_scenario:
                raise ValueError("task.scenario must contain init")
            init = json_object(raw_scenario["init"], "scenario.init")
            if not init:
                raise ValueError("omit task.scenario instead of using an empty init")
            scenario = Scenario(init=init)

        raw_binding = record.get("verification")
        if raw_binding is not None:
            if not isinstance(raw_binding, Mapping):
                raise TypeError("task.verification must be an object or null")
            binding = json_object(raw_binding, "task.verification")
            kind = binding.get("kind")
            if not isinstance(kind, str) or not kind:
                raise ValueError("task.verification.kind must be non-empty")
            if kind not in self.verifier_builders:
                raise ValueError(f"no verifier builder registered for kind {kind!r}")

        task_record = {
            key: record[key]
            for key in (
                "schema_version", "task_id", "env_id", "messages", "judge_ref"
            )
            if key in record
        }
        return Task.from_dict(task_record, scenario=scenario)

    def load(self, task_id: str) -> Task:
        validate_task_id(task_id)
        path = self.root / f"{task_id}.json"
        if not path.is_file():
            raise KeyError(f"task not found: {task_id!r} in {self.root}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError(f"task must be a JSON object: {path}")
        return self._task_from_record(value, task_id=task_id, source=str(path))

    def resolve(self, task_id: str, *, env_id: str | None = None) -> Task:
        task = self.load(task_id)
        if env_id is not None and task.env_id != env_id:
            raise ValueError(
                f"task {task_id!r} belongs to {task.env_id!r}, not {env_id!r}"
            )
        return task

    def verifier_factory(
        self,
        task_id: str,
        *,
        env_id: str,
    ) -> ReplayVerifierFactory | None:
        validate_task_id(task_id)
        path = self.root / f"{task_id}.json"
        if not path.is_file():
            raise KeyError(f"task not found: {task_id!r} in {self.root}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        record = self._validate_record(raw, task_id=task_id, source=str(path))
        if record.get("env_id") != env_id:
            raise ValueError(
                f"task {task_id!r} belongs to {record.get('env_id')!r}, not {env_id!r}"
            )
        binding = record.get("verification")
        if binding is None:
            return None
        # Reuse full Task validation so malformed records fail identically.
        self._task_from_record(record, task_id=task_id, source=str(path))
        kind = str(binding["kind"])
        detached_binding = _json_copy(binding)

        def factory() -> ReplayVerifier:
            verifier = self.verifier_builders[kind](_json_copy(detached_binding))
            if not isinstance(verifier, ReplayVerifier):
                raise TypeError("verifier builder returned an invalid verifier")
            return verifier

        return factory

    def list(self, *, env_id: str | None = None) -> tuple[Task, ...]:
        if not self.root.exists():
            return ()
        tasks = tuple(self.load(path.stem) for path in sorted(self.root.glob("*.json")))
        if len({task.task_id for task in tasks}) != len(tasks):
            raise ValueError("duplicate task ids")
        return tuple(task for task in tasks if env_id is None or task.env_id == env_id)

    def save_record(
        self,
        record: Mapping[str, Any],
        *,
        overwrite: bool = False,
    ) -> Path:
        """Atomically persist already-authored v2 JSON data."""

        task_id = str(record.get("task_id") or "")
        validate_task_id(task_id)
        detached = _json_copy(record)
        # Validate all structure and verifier bindings before touching disk.
        self._task_from_record(
            detached,
            task_id=task_id,
            source=f"record {task_id!r}",
        )
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / f"{task_id}.json"
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{task_id}.", suffix=".tmp", dir=self.root
        )
        temporary_path = Path(temporary)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(detached, ensure_ascii=False, indent=2) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.replace(destination)
        except Exception:
            raise
        finally:
            temporary_path.unlink(missing_ok=True)
        return destination


class CompositeTaskResolver:
    """Resolve colliding task IDs through an explicit Env-to-store mapping."""

    def __init__(self, stores: Mapping[str, JsonTaskStore]):
        self.stores = dict(stores)

    def resolve(self, task_id: str, *, env_id: str | None = None) -> Task:
        if env_id is not None:
            try:
                store = self.stores[env_id]
            except KeyError as exc:
                raise KeyError(f"no task store registered for {env_id!r}") from exc
            return store.resolve(task_id, env_id=env_id)
        matches: list[Task] = []
        for candidate_env, store in self.stores.items():
            try:
                matches.append(store.resolve(task_id, env_id=candidate_env))
            except KeyError:
                continue
        if not matches:
            raise KeyError(f"unknown task {task_id!r}")
        if len(matches) != 1:
            raise ValueError(
                f"task_id {task_id!r} is ambiguous; pass env_id explicitly"
            )
        return matches[0]

    def verifier_factory(
        self,
        task_id: str,
        *,
        env_id: str,
    ) -> ReplayVerifierFactory | None:
        try:
            store = self.stores[env_id]
        except KeyError as exc:
            raise KeyError(f"no task store registered for {env_id!r}") from exc
        return store.verifier_factory(task_id, env_id=env_id)


def _path_value(snapshot: Any, path: Sequence[str | int]) -> Any:
    current = snapshot
    for part in path:
        if isinstance(part, int):
            if not isinstance(current, Sequence) or isinstance(current, (str, bytes)):
                raise KeyError(part)
            current = current[part]
        else:
            if not isinstance(current, Mapping) or part not in current:
                raise KeyError(part)
            current = current[part]
    return current


class StatePredicateVerifier:
    """Task-bound interpreter for final-state predicate JSON."""

    def __init__(self, spec: Mapping[str, Any]):
        self.spec = json_object(spec, "state predicate verifier")

    def verify(self, env: Env, rollout: Trajectory) -> VerificationResult:
        remote_verify = getattr(env, "verify_task", None)
        if callable(remote_verify):
            result = remote_verify(self.spec, rollout)
            if not isinstance(result, VerificationResult):
                raise TypeError("remote verifier must return VerificationResult")
            return result
        snapshot_fn = getattr(env, "snapshot", None)
        if not callable(snapshot_fn):
            return VerificationResult(
                passed=False,
                reward=0.0,
                reason="state_predicate requires an Env snapshot() capability",
            )
        try:
            snapshot = snapshot_fn()
        except Exception as exc:
            return VerificationResult(
                passed=False,
                reward=0.0,
                reason=f"snapshot failed: {type(exc).__name__}: {exc}",
            )
        if not isinstance(snapshot, Mapping):
            return VerificationResult(False, 0.0, "snapshot must be an object")
        return self.evaluate(self.spec, snapshot)

    @staticmethod
    def evaluate(
        spec: Mapping[str, Any],
        snapshot: Mapping[str, Any],
    ) -> VerificationResult:
        if spec.get("kind") != STATE_PREDICATE_VERIFIER_KIND:
            return VerificationResult(False, 0.0, "invalid verifier kind")
        if spec.get("aggregation") != "all":
            return VerificationResult(False, 0.0, "aggregation must be all")
        checks = spec.get("checks")
        if not isinstance(checks, list) or not checks:
            return VerificationResult(False, 0.0, "checks must be a non-empty list")
        results: list[VerificationCheck] = []
        for index, check in enumerate(checks):
            if not isinstance(check, Mapping):
                return VerificationResult(False, 0.0, f"check[{index}] is invalid")
            check_id = str(check.get("check_id") or f"check_{index}")
            predicate = check.get("predicate")
            if not isinstance(predicate, Mapping):
                return VerificationResult(False, 0.0, f"{check_id} has no predicate")
            path = predicate.get("path")
            op = predicate.get("op")
            if not isinstance(path, list) or not path or op not in PREDICATE_OPERATORS:
                return VerificationResult(False, 0.0, f"{check_id} has invalid predicate")
            try:
                actual = _path_value(snapshot, path)
                expected = predicate.get("value")
                passed = {
                    "equals": lambda: actual == expected,
                    "not_equals": lambda: actual != expected,
                    "gte": lambda: actual >= expected,
                    "lte": lambda: actual <= expected,
                    "empty": lambda: not actual,
                    "not_empty": lambda: bool(actual),
                }[str(op)]()
            except (KeyError, IndexError, TypeError, ValueError):
                passed = False
            results.append(VerificationCheck(
                name=check_id,
                passed=bool(passed),
                detail=f"{'.'.join(map(str, path))} {op}",
            ))
        return VerificationResult.from_checks(results)


def state_predicate_verifier_builder(
    binding: Mapping[str, Any],
) -> ReplayVerifier:
    return StatePredicateVerifier(binding)


class LiveEnvVerifier:
    """Delegate a custom binding to the still-live local or isolated Env."""

    def __init__(self, binding: Mapping[str, Any]):
        self.binding = json_object(binding, "live Env verifier binding")

    def verify(self, env: Env, rollout: Trajectory) -> VerificationResult:
        verify_task = getattr(env, "verify_task", None)
        if not callable(verify_task):
            return VerificationResult(
                passed=False,
                reward=0.0,
                reason=(
                    f"Env does not implement verifier kind "
                    f"{self.binding.get('kind')!r}"
                ),
            )
        result = verify_task(self.binding, rollout)
        if not isinstance(result, VerificationResult):
            raise TypeError("Env.verify_task must return VerificationResult")
        return result


def live_env_verifier_builder(
    binding: Mapping[str, Any],
) -> ReplayVerifier:
    return LiveEnvVerifier(binding)


__all__ = [
    "JsonTaskStore",
    "CompositeTaskResolver",
    "LiveEnvVerifier",
    "PREDICATE_OPERATORS",
    "STATE_PREDICATE_VERIFIER_KIND",
    "StatePredicateVerifier",
    "VerifierBuilder",
    "state_predicate_verifier_builder",
    "live_env_verifier_builder",
]
