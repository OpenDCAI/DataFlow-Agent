"""Deterministic serving implementation for tests and local smoke runs."""

from __future__ import annotations

from typing import Sequence

from ..contracts import Message
from .base_serving import ModelServing


class ScriptedServing(ModelServing):
    def __init__(self, responses: Sequence[str]):
        self._responses = list(responses)
        self.requests: list[tuple[Message, ...]] = []

    def generate_messages(
        self,
        conversations: Sequence[Sequence[Message]],
    ) -> list[str]:
        results = []
        for conversation in conversations:
            self.requests.append(tuple(conversation))
            if not self._responses:
                raise RuntimeError("ScriptedServing has no response left")
            results.append(self._responses.pop(0))
        return results


__all__ = ["ScriptedServing"]
