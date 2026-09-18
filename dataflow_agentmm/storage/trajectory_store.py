"""Trajectory JSONL persistence.

The trajectory data model lives in :mod:`dataflow_agentmm.contracts.trajectory`;
this module contains only its storage implementation.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

from ..contracts.trajectory import Trajectory


class TrajectoryStore:
    """Persist one trajectory per JSONL line with images as raw base64."""

    def save(self, trajectory: Trajectory, path: Path | str) -> Path:
        """Write exactly one trajectory to a JSONL file."""
        return self.save_many((trajectory,), path)

    def save_many(
        self,
        trajectories: Iterable[Trajectory],
        path: Path | str,
    ) -> Path:
        """Atomically write trajectories as one compact JSON object per line."""
        destination = Path(path)
        if destination.suffix.lower() != ".jsonl":
            raise ValueError("new trajectory files must use the .jsonl suffix")
        destination.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for trajectory in trajectories:
            value = trajectory.to_dict()
            lines.append(json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
            ))
        payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
        self._atomic_write_bytes(destination, payload)
        return destination

    def load(self, path: Path | str) -> Trajectory:
        """Load a single-record JSONL file."""
        trajectories = self.load_many(path)
        if len(trajectories) != 1:
            raise ValueError(
                f"expected exactly one trajectory, found {len(trajectories)}"
            )
        return trajectories[0]

    def load_many(self, path: Path | str) -> tuple[Trajectory, ...]:
        """Load all records from a canonical trajectory JSONL file."""
        source = Path(path).resolve(strict=True)
        if source.suffix.lower() != ".jsonl":
            raise ValueError("trajectory path must end in .jsonl")
        values = []
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid trajectory JSONL at line {line_number}: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(
                        f"trajectory JSONL line {line_number} is not an object"
                    )
                values.append(value)
        return tuple(Trajectory.from_dict(value) for value in values)

    @staticmethod
    def _atomic_write_bytes(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=str(path.parent)
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise


__all__ = ["TrajectoryStore"]
