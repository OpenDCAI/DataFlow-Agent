"""Small, policy-governed host tools scoped to one episode workspace."""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..contracts import (
    ContentLimits,
    ImageContent,
    TextContent,
    ToolResult,
    ToolSpec,
    validate_content,
)


_IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp", "image/bmp"}
_TEXT_TYPES = {
    "application/json",
    "application/javascript",
    "application/xml",
    "application/x-yaml",
}


@dataclass(frozen=True)
class HostPolicy:
    workspace: Path
    max_list_entries: int = 500
    max_read_bytes: int = 1_000_000
    content_limits: ContentLimits = ContentLimits()

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace", Path(self.workspace).resolve())

    def resolve(self, raw: str, *, require_directory: bool | None = None) -> Path:
        if not raw or raw in {".", "/"}:
            candidate = self.workspace
        else:
            path = Path(raw)
            if path.is_absolute() or "://" in raw:
                raise ValueError("only episode-workspace-relative paths are allowed")
            candidate = self.workspace / path
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError("source does not exist") from exc
        if resolved != self.workspace and self.workspace not in resolved.parents:
            raise PermissionError("source resolves outside the episode workspace")
        if require_directory is True and not resolved.is_dir():
            raise NotADirectoryError("source is not a directory")
        if require_directory is False and not resolved.is_file():
            raise IsADirectoryError("source is not a file")
        return resolved


class HostTools:
    def __init__(self, policy: HostPolicy):
        self.policy = policy

    def tools(self) -> Sequence[ToolSpec]:
        return (
            ToolSpec(
                name="host.list",
                description="List entries in a directory inside the episode workspace.",
                operation_type="query",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string", "default": "."}},
                    "additionalProperties": False,
                },
            ),
            ToolSpec(
                name="host.open",
                description=(
                    "Open a text or image file inside the episode workspace and "
                    "return its actual model-visible content."
                ),
                operation_type="query",
                input_schema={
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "range": {"type": "string"},
                        "detail": {
                            "type": "string",
                            "enum": ["auto", "high", "low", "original"],
                        },
                    },
                    "required": ["source"],
                    "additionalProperties": False,
                },
            ),
        )

    def call(self, name: str, args: Mapping[str, Any]) -> ToolResult:
        try:
            if name == "host.list":
                return self._list(str(args.get("path") or "."))
            if name == "host.open":
                return self._open(
                    str(args.get("source") or ""),
                    line_range=(str(args["range"]) if args.get("range") else None),
                    detail=str(args.get("detail") or "auto"),
                )
            return ToolResult.failure("unknown_tool", f"unknown host tool: {name}")
        except FileNotFoundError as exc:
            return ToolResult.failure("not_found", str(exc))
        except PermissionError as exc:
            return ToolResult.failure("outside_workspace", str(exc))
        except (ValueError, IsADirectoryError, NotADirectoryError) as exc:
            return ToolResult.failure("invalid_source", str(exc))
        except OSError as exc:
            return ToolResult.failure("io_error", str(exc), retryable=True)

    def _list(self, raw: str) -> ToolResult:
        directory = self.policy.resolve(raw, require_directory=True)
        entries = []
        values = sorted(directory.iterdir(), key=lambda item: item.name)
        truncated = len(values) > self.policy.max_list_entries
        for item in values[: self.policy.max_list_entries]:
            is_symlink = item.is_symlink()
            try:
                resolved = item.resolve(strict=True)
                inside = (
                    not is_symlink
                    and (
                        resolved == self.policy.workspace
                        or self.policy.workspace in resolved.parents
                    )
                )
            except OSError:
                inside = False
            if is_symlink:
                entry_type = "symlink"
            elif item.is_dir():
                entry_type = "directory"
            elif item.is_file():
                entry_type = "file"
            else:
                entry_type = "other"
            entries.append({
                "name": item.name,
                "type": entry_type,
                "size": item.stat().st_size if inside and entry_type == "file" else None,
                "readable": inside,
            })
        import json

        return ToolResult.success((TextContent(json.dumps({
            "path": raw,
            "entries": entries,
            "truncated": truncated,
        }, ensure_ascii=False)),))

    def _open(
        self,
        raw: str,
        *,
        line_range: str | None,
        detail: str,
    ) -> ToolResult:
        path = self.policy.resolve(raw, require_directory=False)
        size = path.stat().st_size
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if media_type in _IMAGE_TYPES:
            if size > self.policy.content_limits.max_image_bytes:
                return ToolResult.failure("too_large", "image exceeds configured byte limit")
            image = ImageContent.from_bytes(path.read_bytes(), media_type, detail=detail)
            validate_content((image,), self.policy.content_limits)
            return ToolResult.success((image,), metadata={"source": path.name})
        if not (media_type.startswith("text/") or media_type in _TEXT_TYPES):
            return ToolResult.failure(
                "unsupported_media",
                f"unsupported file media type: {media_type}",
            )
        if size > self.policy.max_read_bytes:
            return ToolResult.failure("too_large", "text file exceeds configured byte limit")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult.failure("unsupported_media", "file is not UTF-8 text or a supported image")
        if line_range:
            text = self._slice_lines(text, line_range)
        if len(text) > self.policy.content_limits.max_text_chars:
            text = text[: self.policy.content_limits.max_text_chars] + "\n...[truncated]"
        return ToolResult.success((TextContent(text),), metadata={"source": path.name})

    @staticmethod
    def _slice_lines(text: str, raw: str) -> str:
        try:
            start_raw, end_raw = raw.split(":", 1)
            start = int(start_raw)
            end = int(end_raw)
        except (ValueError, TypeError) as exc:
            raise ValueError("range must use 1-based start:end syntax") from exc
        if start < 1 or end < start:
            raise ValueError("range must satisfy 1 <= start <= end")
        return "\n".join(text.splitlines()[start - 1:end])
