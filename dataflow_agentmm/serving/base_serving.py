"""Common model-serving interface consumed by DataFlow-AgentMM runtimes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from ..contracts import Message


class ModelResponseFormatError(RuntimeError):
    """The provider returned a response, but not usable assistant text."""


class ModelServing(ABC):
    @abstractmethod
    def generate_messages(
        self,
        conversations: Sequence[Sequence[Message]],
    ) -> list[str]:
        raise NotImplementedError

    def generate(self, messages: Sequence[Message]) -> str:
        results = self.generate_messages([messages])
        if len(results) != 1:
            raise RuntimeError("serving must return exactly one result per request")
        return results[0]

    def generate_messages_with_options(
        self,
        conversations: Sequence[Sequence[Message]],
        request_options: Sequence[Mapping[str, Any] | None],
    ) -> list[str]:
        """Generate with per-conversation provider options when supported.

        The default intentionally preserves compatibility with deterministic,
        local, and non-OpenAI serving implementations. Provider adapters that
        support constrained decoding can override this method and apply the
        options independently to each request.
        """

        if len(conversations) != len(request_options):
            raise ValueError(
                "conversations and request_options must have equal length"
            )
        return self.generate_messages(conversations)

    def health_check(self) -> bool:
        return True

    def cleanup(self) -> None:
        return None


__all__ = ["ModelResponseFormatError", "ModelServing"]
