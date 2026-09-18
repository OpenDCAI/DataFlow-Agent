"""Provider-neutral multimodal message and tool contracts.

Images are first-class message content.  They are never represented as local
paths in the agent-facing API; a serving adapter decides how to deliver their
base64 payload to a concrete model provider.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence, Union

import jsonschema


_IMAGE_SIGNATURES = {
    "image/png": lambda value: value.startswith(b"\x89PNG\r\n\x1a\n"),
    "image/jpeg": lambda value: value.startswith(b"\xff\xd8\xff"),
    "image/gif": lambda value: value.startswith((b"GIF87a", b"GIF89a")),
    "image/webp": lambda value: (
        len(value) >= 12
        and value[:4] == b"RIFF"
        and value[8:12] == b"WEBP"
    ),
    "image/bmp": lambda value: value.startswith(b"BM"),
}


@dataclass(frozen=True)
class TextContent:
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text content must be a string")

    def to_dict(self) -> dict[str, Any]:
        return {"type": "text", "text": self.text}


@dataclass(frozen=True)
class ImageContent:
    """An inline image represented by raw base64 without a data-URI prefix."""

    media_type: str
    data: str
    detail: str = "auto"

    def __post_init__(self) -> None:
        if not isinstance(self.media_type, str) or not isinstance(self.data, str):
            raise TypeError("image media_type and data must be strings")
        if not isinstance(self.detail, str):
            raise TypeError("image detail must be a string")
        if self.media_type not in _IMAGE_SIGNATURES:
            raise ValueError(f"unsupported image media type: {self.media_type}")
        if self.detail not in {"auto", "high", "low", "original"}:
            raise ValueError(f"unsupported image detail: {self.detail}")
        if self.data.startswith("data:"):
            raise ValueError("ImageContent.data must be raw base64, not a data URI")
        self.decoded()

    def decoded(self) -> bytes:
        try:
            return base64.b64decode(self.data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("image data is not valid base64") from exc

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.decoded()).hexdigest()

    @classmethod
    def from_bytes(
        cls,
        payload: bytes,
        media_type: str,
        *,
        detail: str = "auto",
    ) -> "ImageContent":
        return cls(
            media_type=media_type,
            data=base64.b64encode(payload).decode("ascii"),
            detail=detail,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "image",
            "media_type": self.media_type,
            "data": self.data,
            "detail": self.detail,
        }


Content = Union[TextContent, ImageContent]


def content_from_dict(value: Mapping[str, Any]) -> Content:
    if not isinstance(value, Mapping):
        raise TypeError("content must be an object")
    kind = value.get("type")
    if kind == "text":
        if set(value) != {"type", "text"} or not isinstance(value.get("text"), str):
            raise ValueError("text content must contain exactly type and string text")
        return TextContent(value["text"])
    if kind == "image":
        extra = set(value).difference({"type", "media_type", "data", "detail"})
        if extra or not isinstance(value.get("media_type"), str) or not isinstance(
            value.get("data"), str
        ):
            raise ValueError("image content has invalid fields")
        return ImageContent(
            media_type=value["media_type"],
            data=value["data"],
            detail=value.get("detail") or "auto",
        )
    raise ValueError(f"unsupported content type: {kind!r}")


@dataclass(frozen=True)
class Message:
    role: str
    content: tuple[Content, ...]
    name: str | None = None

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "observation"}:
            raise ValueError(f"unsupported message role: {self.role}")
        content = tuple(self.content)
        if any(not isinstance(item, (TextContent, ImageContent)) for item in content):
            raise TypeError("message content must contain canonical content values")
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError("message name must be a string or null")
        object.__setattr__(self, "content", content)

    @classmethod
    def of(
        cls,
        role: str,
        content: Iterable[Content],
        *,
        name: str | None = None,
    ) -> "Message":
        return cls(role=role, content=tuple(content), name=name)

    @classmethod
    def text(
        cls,
        role: str,
        text: str,
        *,
        name: str | None = None,
    ) -> "Message":
        return cls(role=role, content=(TextContent(text),), name=name)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "role": self.role,
            "content": [item.to_dict() for item in self.content],
        }
        if self.name is not None:
            result["name"] = self.name
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Message":
        if not isinstance(value, Mapping):
            raise TypeError("message must be an object")
        extra = set(value).difference({"role", "content", "name"})
        if extra:
            raise ValueError(f"unsupported message fields: {sorted(extra)}")
        if not isinstance(value.get("role"), str):
            raise TypeError("message role must be a string")
        if not isinstance(value.get("content"), list):
            raise TypeError("message content must be a list")
        if value.get("name") is not None and not isinstance(value.get("name"), str):
            raise TypeError("message name must be a string or null")
        return cls.of(
            role=value["role"],
            content=(content_from_dict(item) for item in value["content"]),
            name=value.get("name"),
        )


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    operation_type: Literal["query", "mutation", "unknown"] = "unknown"
    input_schema: Mapping[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or any(
            character.isspace() for character in self.name
        ):
            raise ValueError("tool name must be non-empty and contain no whitespace")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("tool description must be non-empty")
        if self.operation_type not in {"query", "mutation", "unknown"}:
            raise ValueError(
                "tool operation_type must be 'query', 'mutation', or 'unknown'"
            )
        if not isinstance(self.input_schema, Mapping) or self.input_schema.get("type") != "object":
            raise ValueError("tool input_schema must describe a JSON object")
        try:
            jsonschema.Draft202012Validator.check_schema(dict(self.input_schema))
        except jsonschema.SchemaError as exc:
            raise ValueError(f"invalid tool input_schema: {exc.message}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "operation_type": self.operation_type,
            "input_schema": dict(self.input_schema),
        }


@dataclass(frozen=True)
class ToolError:
    code: str
    message: str
    retryable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("tool error code must be non-empty")
        if not isinstance(self.message, str):
            raise TypeError("tool error message must be a string")
        if not isinstance(self.retryable, bool):
            raise TypeError("tool error retryable must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolError":
        if not isinstance(value.get("code"), str) or not isinstance(
            value.get("message"), str
        ) or not isinstance(value.get("retryable"), bool):
            raise TypeError("ToolError fields have invalid types")
        return cls(
            code=value["code"],
            message=value["message"],
            retryable=value["retryable"],
        )


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: tuple[Content, ...] = ()
    error: ToolError | None = None
    is_final: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.ok, bool) or not isinstance(self.is_final, bool):
            raise TypeError("ToolResult ok/is_final must be booleans")
        content = tuple(self.content)
        if any(not isinstance(item, (TextContent, ImageContent)) for item in content):
            raise TypeError("ToolResult content must be canonical content")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("ToolResult metadata must be a mapping")
        if self.ok and self.error is not None:
            raise ValueError("successful ToolResult cannot carry an error")
        if not self.ok and self.error is None:
            raise ValueError("failed ToolResult must carry a ToolError")
        object.__setattr__(self, "content", content)

    @classmethod
    def success(
        cls,
        content: Iterable[Content] = (),
        *,
        is_final: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            ok=True,
            content=tuple(content),
            is_final=is_final,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def failure(
        cls,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        content: Iterable[Content] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            ok=False,
            content=tuple(content),
            error=ToolError(code, message, retryable),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "content": [item.to_dict() for item in self.content],
            "error": self.error.to_dict() if self.error else None,
            "is_final": self.is_final,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolResult":
        required = {"ok", "content", "error", "is_final", "metadata"}
        missing = required.difference(value)
        if missing:
            raise KeyError(f"missing ToolResult fields: {sorted(missing)}")
        if not isinstance(value["ok"], bool) or not isinstance(value["is_final"], bool):
            raise TypeError("ToolResult ok/is_final must be booleans")
        if not isinstance(value["content"], list):
            raise TypeError("ToolResult content must be a list")
        if not isinstance(value["metadata"], Mapping):
            raise TypeError("ToolResult metadata must be an object")
        raw_error = value.get("error")
        return cls(
            ok=value["ok"],
            content=tuple(
                content_from_dict(item) for item in value["content"]
            ),
            error=ToolError.from_dict(raw_error) if raw_error else None,
            is_final=value["is_final"],
            metadata=dict(value["metadata"]),
        )


@dataclass(frozen=True)
class ContentLimits:
    max_text_chars: int = 64_000
    max_image_bytes: int = 32 * 1024 * 1024
    max_image_pixels: int = 40_000_000

    def __post_init__(self) -> None:
        for name in ("max_text_chars", "max_image_bytes", "max_image_pixels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def validate_content(
    items: Sequence[Content],
    limits: ContentLimits = ContentLimits(),
) -> None:
    """Validate content before it is appended to model-visible history."""

    for item in items:
        if isinstance(item, TextContent):
            if len(item.text) > limits.max_text_chars:
                raise ValueError("text content exceeds configured character limit")
            continue
        payload = item.decoded()
        if len(payload) > limits.max_image_bytes:
            raise ValueError("image content exceeds configured byte limit")
        signature = _IMAGE_SIGNATURES[item.media_type]
        if not signature(payload):
            raise ValueError("image bytes do not match declared media type")
        try:
            from PIL import Image

            with Image.open(io.BytesIO(payload)) as image:
                width, height = image.size
                image.verify()
        except ImportError:
            continue
        except Exception as exc:
            raise ValueError("image payload cannot be decoded") from exc
        if width * height > limits.max_image_pixels:
            raise ValueError("image content exceeds configured pixel limit")
