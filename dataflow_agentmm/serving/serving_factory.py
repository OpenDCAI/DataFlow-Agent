"""Construct a serving backend from explicit or environment configuration."""

from __future__ import annotations

import os
from typing import Mapping

from .base_serving import ModelServing
from .gemini_serving import DEFAULT_GEMINI_BASE_URL, GeminiServing
from .openai_serving import OpenAICompatibleServing


def create_model_serving(
    *,
    backend: str,
    model: str,
    base_url: str | None,
    api_key: str = "EMPTY",
    timeout: float = 1800,
    max_tokens: int | None = None,
    temperature: float | None = None,
    max_workers: int = 1,
    max_images_per_request: int | None = 8,
    request_options: Mapping[str, object] | None = None,
) -> ModelServing:
    normalized = backend.strip().lower().replace("-", "_")
    if normalized in {"openai", "openai_compatible"}:
        return OpenAICompatibleServing(
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_tokens=max_tokens,
            temperature=temperature,
            max_workers=max_workers,
            max_images_per_request=max_images_per_request,
            request_options=request_options,
        )
    if normalized == "gemini":
        if not api_key.strip() or api_key == "EMPTY":
            raise ValueError("Gemini serving requires GEMINI_API_KEY or GOOGLE_API_KEY")
        return GeminiServing(
            model=model,
            base_url=base_url or DEFAULT_GEMINI_BASE_URL,
            api_key=api_key,
            timeout=timeout,
            max_tokens=max_tokens,
            temperature=temperature,
            max_workers=max_workers,
            max_images_per_request=max_images_per_request,
            request_options=request_options,
        )
    raise ValueError(f"unsupported serving backend: {backend!r}")


def create_model_serving_from_env(
    *,
    environ: Mapping[str, str] | None = None,
    timeout: float = 1800,
    max_tokens: int | None = None,
    temperature: float | None = None,
    max_workers: int = 1,
    max_images_per_request: int | None = 8,
) -> ModelServing:
    values = os.environ if environ is None else environ
    model = values.get("MODEL", "").strip()
    if not model:
        raise ValueError("MODEL is required")
    backend = values.get("SERVING_BACKEND", "openai")
    normalized = backend.strip().lower().replace("-", "_")
    api_key = (
        values.get("GOOGLE_API_KEY")
        or values.get("GEMINI_API_KEY")
        or ""
        if normalized == "gemini"
        else values.get("DF_API_KEY", "EMPTY")
    )
    return create_model_serving(
        backend=backend,
        model=model,
        base_url=values.get("API_URL") or None,
        api_key=api_key,
        timeout=timeout,
        max_tokens=max_tokens,
        temperature=temperature,
        max_workers=max_workers,
        max_images_per_request=max_images_per_request,
    )


__all__ = ["create_model_serving", "create_model_serving_from_env"]
