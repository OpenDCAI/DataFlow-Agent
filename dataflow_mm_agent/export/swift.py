"""Convert canonical trajectories to ms-swift ``messages`` JSONL.

Each trajectory becomes one row: ``{"messages": [...], "images": [...]}`` plus
``episode_id``/``task_id``/``env_id`` for traceability (ms-swift drops unknown
columns). The recorded conversation is kept as the model saw it:

* ``system``/``user``/``assistant`` messages keep their role and text; the
  assistant text is the raw recorded response (the runtime JSON action).
* An ``observation`` answering an assistant turn becomes ``tool_response``.
  One with no preceding assistant turn (e.g. ``env.start``) becomes ``user``
  with the same ``[tool NAME observation]`` header the serving layer sends.
* Every image block becomes an ``<image>`` tag, and ``images`` lists the images
  in tag order, as local file paths or ``data:`` URIs.

Messages after the last assistant turn cannot be trained on and are dropped; a
trajectory without any assistant turn is skipped. No other selection is applied.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping

from ..contracts.trajectory import Trajectory

ImageMode = Literal["file", "base64"]
_SUFFIXES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
}


@dataclass
class SwiftExportReport:
    output: Path
    exported: int = 0
    images: int = 0
    skipped: list[str] = field(default_factory=list)


def _image_reference(block: Mapping[str, Any], mode: ImageMode, directory: Path | None) -> str:
    media_type = str(block.get("media_type") or "image/png")
    data = str(block.get("data") or "")
    if not data:
        raise ValueError("image block has no inline data")
    if mode == "base64":
        return f"data:{media_type};base64,{data}"
    if directory is None:
        raise ValueError("image_mode='file' requires image_dir")
    payload = base64.b64decode(data, validate=True)
    path = directory.resolve() / (hashlib.sha256(payload).hexdigest() + _SUFFIXES.get(media_type, ".png"))
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return str(path)


def trajectory_to_swift(
    trajectory: Trajectory | Mapping[str, Any],
    *,
    image_mode: ImageMode = "base64",
    image_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    """Return one ms-swift row, or ``None`` when there is no assistant turn."""

    if image_mode not in ("file", "base64"):
        raise ValueError("image_mode must be 'file' or 'base64'")
    value = trajectory.to_dict() if isinstance(trajectory, Trajectory) else trajectory
    messages: list[tuple[str, str, list[Mapping[str, Any]]]] = []
    for message in value.get("messages") or ():
        role = message.get("role")
        parts: list[str] = []
        image_blocks: list[Mapping[str, Any]] = []
        for block in message.get("content") or ():
            if block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif block.get("type") == "image":
                image_blocks.append(block)
                parts.append("<image>")
            else:
                raise ValueError(f"unsupported content block type {block.get('type')!r}")
        text = "\n".join(part for part in parts if part)
        if role in ("system", "user", "assistant"):
            messages.append((role, text, image_blocks))
        elif role == "observation":
            if messages and messages[-1][0] in ("assistant", "tool_response"):
                messages.append(("tool_response", text, image_blocks))
            else:
                header = f"[tool {message.get('name') or 'environment'} observation]"
                messages.append(("user", f"{header}\n{text}", image_blocks))
        else:
            raise ValueError(f"unsupported message role {role!r}")

    last = max((i for i, (role, _, _) in enumerate(messages) if role == "assistant"), default=None)
    if last is None:
        return None
    kept = messages[: last + 1]
    directory = Path(image_dir) if image_dir is not None else None
    return {
        "messages": [{"role": role, "content": text} for role, text, _ in kept],
        "images": [
            _image_reference(block, image_mode, directory)
            for _, _, blocks in kept
            for block in blocks
        ],
        "episode_id": value.get("episode_id"),
        "task_id": value.get("task_id"),
        "env_id": value.get("env_id"),
    }


def _records(source: Path) -> Iterator[Any]:
    if source.suffix.lower() == ".jsonl":
        with source.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{source}:{number}: invalid JSON: {exc}") from exc
        return
    value = json.loads(source.read_text(encoding="utf-8"))
    yield from value if isinstance(value, list) else (value,)


def _trajectory(record: Any, trajectory_key: str) -> Mapping[str, Any]:
    if isinstance(record, Mapping) and "messages" not in record and trajectory_key in record:
        record = record[trajectory_key]
    if isinstance(record, str):
        record = json.loads(record)
    if not isinstance(record, Mapping) or not isinstance(record.get("messages"), list):
        raise ValueError("record is neither a trajectory nor a row with a trajectory")
    return record


def export_swift_jsonl(
    sources: Iterable[str | Path],
    output: str | Path,
    *,
    trajectory_key: str = "trajectory",
    image_mode: ImageMode = "file",
    image_dir: str | Path | None = None,
    force: bool = False,
) -> SwiftExportReport:
    """Convert trajectory JSON/JSONL files or pipeline-row JSONL files to one ms-swift JSONL.

    With ``image_mode="file"`` images are written once per content hash to
    ``image_dir`` (default ``<output stem>_images/`` next to the output) and
    referenced by absolute path; ``"base64"`` inlines ``data:`` URIs instead.
    """

    destination = Path(output)
    if destination.suffix.lower() != ".jsonl":
        raise ValueError("output must be a .jsonl file")
    if destination.exists() and not force:
        raise FileExistsError(f"{destination} exists; pass force=True to replace it")
    if image_mode == "file" and image_dir is None:
        image_dir = destination.with_name(destination.stem + "_images")
    report = SwiftExportReport(output=destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for source in map(Path, sources):
                for index, record in enumerate(_records(source)):
                    label = f"{source}#{index}"
                    try:
                        row = trajectory_to_swift(
                            _trajectory(record, trajectory_key),
                            image_mode=image_mode,
                            image_dir=image_dir,
                        )
                    except ValueError as exc:
                        report.skipped.append(f"{label}: {exc}")
                        continue
                    if row is None:
                        report.skipped.append(f"{label}: no assistant turn")
                        continue
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    report.exported += 1
                    report.images += len(row["images"])
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return report


__all__ = ["SwiftExportReport", "export_swift_jsonl", "trajectory_to_swift"]
