"""Per-scope model usage accounting shared by serving adapters and the runtime.

Callers open a scope with :func:`track_usage`; provider adapters report each
completed request with :func:`record_usage`. Scopes live in a ``ContextVar``,
so concurrent episodes on separate threads keep separate totals. Work handed to
another thread must run inside ``contextvars.copy_context()`` to be counted.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Mapping

_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens")


class UsageMeter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = 0
        self.tokens = dict.fromkeys(_FIELDS, 0)
        self._reported: set[str] = set()

    def add(self, **tokens: int | None) -> None:
        with self._lock:
            self.requests += 1
            for name, value in tokens.items():
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    self.tokens[name] += value
                    self._reported.add(name)

    def to_dict(self) -> dict[str, int]:
        """Request count plus only the token fields a provider actually reported."""
        with self._lock:
            return {"requests": self.requests, **{k: self.tokens[k] for k in _FIELDS if k in self._reported}}


_ACTIVE: ContextVar[tuple[UsageMeter, ...]] = ContextVar("dataflow_mm_agent_usage", default=())


@contextmanager
def track_usage() -> Iterator[UsageMeter]:
    """Count requests made in this context; nested scopes all receive them."""
    meter = UsageMeter()
    token = _ACTIVE.set(_ACTIVE.get() + (meter,))
    try:
        yield meter
    finally:
        _ACTIVE.reset(token)


def record_usage(
    *,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    reasoning_tokens: int | None = None,
) -> None:
    for meter in _ACTIVE.get():
        meter.add(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
        )


def _field(source: Any, name: str) -> Any:
    return source.get(name) if isinstance(source, Mapping) else getattr(source, name, None)


def record_openai_usage(response: Any) -> None:
    usage = _field(response, "usage")
    details = _field(usage, "completion_tokens_details") if usage is not None else None
    record_usage(
        prompt_tokens=_field(usage, "prompt_tokens") if usage is not None else None,
        completion_tokens=_field(usage, "completion_tokens") if usage is not None else None,
        total_tokens=_field(usage, "total_tokens") if usage is not None else None,
        reasoning_tokens=_field(details, "reasoning_tokens") if details is not None else None,
    )


def record_gemini_usage(response: Mapping[str, Any]) -> None:
    usage = response.get("usageMetadata") if isinstance(response, Mapping) else None
    usage = usage if isinstance(usage, Mapping) else {}
    record_usage(
        prompt_tokens=usage.get("promptTokenCount"),
        completion_tokens=usage.get("candidatesTokenCount"),
        total_tokens=usage.get("totalTokenCount"),
        reasoning_tokens=usage.get("thoughtsTokenCount"),
    )


__all__ = [
    "UsageMeter",
    "record_gemini_usage",
    "record_openai_usage",
    "record_usage",
    "track_usage",
]
