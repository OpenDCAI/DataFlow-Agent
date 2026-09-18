"""Request-time context projection for replayed history.

A policy decides which recorded messages a live request carries. It never edits
the trajectory: everything stays in the canonical record, so replay, export, and
review still see the full history.

The message list is split into three parts:

* head — the system prompt and the task messages, always kept;
* prefix — the replayed history, the only part this policy trims;
* tail — continuation messages and everything produced after them, always kept.

The prefix is grouped into turns (one assistant action plus the observations it
produced). The newest ``keep_last_steps`` turns survive complete, so the model
still sees the arguments it used, not just the resulting screen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from ..contracts import ImageContent, Message, TextContent

Summarizer = Callable[[tuple[Message, ...]], str]

DEFAULT_ELISION_NOTE = (
    "[{turns} earlier step(s) of the restored history are omitted from this "
    "request; the full record is kept in the trajectory]"
)


def _has_image(message: Message) -> bool:
    return any(isinstance(item, ImageContent) for item in message.content)


def _text_length(message: Message) -> int:
    return sum(len(item.text) for item in message.content if isinstance(item, TextContent))


@dataclass(frozen=True)
class ContextPolicy:
    """Which replayed messages a live request keeps.

    ``keep_last_steps=None`` keeps the whole prefix; ``0`` keeps no turn at all,
    which with ``keep_latest_image_observation`` reproduces the minimal
    "current state only" projection.
    """

    keep_last_steps: int | None = 3
    keep_latest_image_observation: bool = True
    max_prefix_chars: int | None = None
    elision_note: str = DEFAULT_ELISION_NOTE
    summarizer: Summarizer | None = None

    def __post_init__(self) -> None:
        if self.keep_last_steps is not None and (
            isinstance(self.keep_last_steps, bool) or not isinstance(self.keep_last_steps, int)
            or self.keep_last_steps < 0
        ):
            raise ValueError("keep_last_steps must be a non-negative integer or None")
        if self.max_prefix_chars is not None and (
            isinstance(self.max_prefix_chars, bool)
            or not isinstance(self.max_prefix_chars, int)
            or self.max_prefix_chars < 0
        ):
            raise ValueError("max_prefix_chars must be a non-negative integer or None")

    @staticmethod
    def _turns(prefix: Sequence[Message]) -> list[list[int]]:
        """Group prefix indexes into assistant-led turns, keeping stray messages."""

        turns: list[list[int]] = []
        for index, message in enumerate(prefix):
            if message.role == "assistant" or not turns:
                turns.append([index])
            else:
                turns[-1].append(index)
        return turns

    def project(
        self,
        messages: Sequence[Message],
        *,
        boundary: int,
        task_message_count: int,
    ) -> tuple[Message, ...]:
        """Return the messages a live request should carry."""

        head_end = min(1 + task_message_count, boundary)
        head = list(messages[:head_end])
        prefix = list(messages[head_end:boundary])
        tail = list(messages[boundary:])
        if self.keep_last_steps is None or not prefix:
            return tuple([*head, *prefix, *tail])

        turns = self._turns(prefix)
        kept_turns = turns[len(turns) - self.keep_last_steps:] if self.keep_last_steps else []
        kept: set[int] = {index for turn in kept_turns for index in turn}
        pinned: int | None = None
        if self.keep_latest_image_observation:
            candidates = [
                index for index in reversed(range(len(prefix)))
                if index not in kept and prefix[index].role == "observation"
            ]
            pinned = next(
                (index for index in candidates if _has_image(prefix[index])),
                # Text-only Envs still benefit from their latest observation.
                candidates[0] if candidates else None,
            )
        if self.max_prefix_chars is not None:
            budget = self.max_prefix_chars - (
                _text_length(prefix[pinned]) if pinned is not None else 0
            )
            while len(kept_turns) > 1 and sum(
                _text_length(prefix[index]) for index in kept
            ) > budget:
                dropped = kept_turns.pop(0)
                kept.difference_update(dropped)

        elided = len(turns) - len(kept_turns)
        if not elided:
            return tuple([*head, *prefix, *tail])
        middle: list[Message] = []
        note = (
            self.summarizer(tuple(
                prefix[index] for turn in turns[:len(turns) - len(kept_turns)] for index in turn
            ))
            if self.summarizer is not None
            else self.elision_note.format(turns=elided)
        )
        if note:
            middle.append(Message.text("user", note, name="context.elided"))
        if pinned is not None:
            middle.append(prefix[pinned])
        return tuple([
            *head, *middle, *[prefix[index] for index in sorted(kept)], *tail,
        ])


MINIMAL_RESTORED_CONTEXT = ContextPolicy(keep_last_steps=0)
"""Current state only: no replayed turn, just the newest image observation."""


__all__ = ["ContextPolicy", "DEFAULT_ELISION_NOTE", "MINIMAL_RESTORED_CONTEXT", "Summarizer"]
