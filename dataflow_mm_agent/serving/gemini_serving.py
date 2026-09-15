"""Google Gemini ``generateContent`` REST serving."""

from __future__ import annotations

import contextvars
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from ..contracts import ImageContent, Message, TextContent
from .base_serving import ModelResponseFormatError, ModelServing
from .usage import record_gemini_usage


GeminiTransport = Callable[
    [str, Mapping[str, str], Mapping[str, Any], float],
    Mapping[str, Any],
]
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiServing(ModelServing):
    """Call the official Gemini REST shape with canonical multimodal messages."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = DEFAULT_GEMINI_BASE_URL,
        api_key: str,
        timeout: float = 1800,
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_workers: int = 1,
        max_images_per_request: int | None = 8,
        malformed_response_retries: int = 2,
        request_options: Mapping[str, Any] | None = None,
        transport: GeminiTransport | None = None,
    ):
        if not model.strip():
            raise ValueError("Gemini model must be non-empty")
        if not base_url.strip():
            raise ValueError("Gemini base_url must be non-empty")
        if not api_key.strip():
            raise ValueError("Gemini API key must be non-empty")
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        if max_images_per_request is not None and max_images_per_request < 1:
            raise ValueError("max_images_per_request must be positive or None")
        if malformed_response_retries < 0:
            raise ValueError("malformed_response_retries must be non-negative")

        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_workers = max_workers
        self.max_images_per_request = max_images_per_request
        self.malformed_response_retries = malformed_response_retries
        self.request_options = dict(request_options or {})
        self.transport = transport or self._post_json

    @staticmethod
    def endpoint(base_url: str, model: str) -> str:
        """Accept a route root, a model URL, or a URL containing a placeholder."""
        encoded_model = quote(model, safe="._-")
        endpoint = base_url.strip().rstrip("/")
        placeholders = ("{model}", "{model_id}")
        for placeholder in placeholders:
            if placeholder in endpoint:
                return endpoint.replace(placeholder, encoded_model)
        if endpoint.endswith(":generateContent"):
            return endpoint
        if endpoint.endswith("/models"):
            return f"{endpoint}/{encoded_model}:generateContent"
        if re.search(r"/v\d+(?:alpha|beta)?$", endpoint):
            return f"{endpoint}/models/{encoded_model}:generateContent"
        return f"{endpoint}/v1beta/models/{encoded_model}:generateContent"

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

    @classmethod
    def provider_request(
        cls,
        messages: Sequence[Message],
        *,
        max_images_per_request: int | None = None,
    ) -> dict[str, Any]:
        if max_images_per_request is not None and max_images_per_request < 1:
            raise ValueError("max_images_per_request must be positive or None")
        image_positions = [
            (message_index, content_index, message.role)
            for message_index, message in enumerate(messages)
            for content_index, item in enumerate(message.content)
            if isinstance(item, ImageContent)
        ]
        kept_images = {
            (message_index, content_index)
            for message_index, content_index, _role in image_positions
        }
        if (
            max_images_per_request is not None
            and len(image_positions) > max_images_per_request
        ):
            # Task-authored reference images arrive in ordinary user messages;
            # tool screenshots arrive in observation messages.  Keep the task
            # input stable across the whole rollout and spend the remaining
            # budget on the newest visual observations.  Dropping the oldest
            # images globally would make an image-to-artifact task forget its
            # references precisely when iterative editing becomes useful.
            task_images = [
                position for position in image_positions if position[2] != "observation"
            ]
            observation_images = [
                position for position in image_positions if position[2] == "observation"
            ]
            if len(task_images) >= max_images_per_request:
                selected = task_images[:max_images_per_request]
            else:
                remaining = max_images_per_request - len(task_images)
                selected = task_images + observation_images[-remaining:]
            kept_images = {
                (message_index, content_index)
                for message_index, content_index, _role in selected
            }

        system_parts: list[dict[str, Any]] = []
        contents: list[dict[str, Any]] = []
        for message_index, message in enumerate(messages):
            parts: list[dict[str, Any]] = []
            if message.role == "observation":
                parts.append({
                    "text": f"[tool {message.name or 'environment'} observation]",
                })
            omitted_here = 0
            for content_index, item in enumerate(message.content):
                if isinstance(item, TextContent):
                    if item.text:
                        parts.append({"text": item.text})
                    continue
                if (message_index, content_index) not in kept_images:
                    omitted_here += 1
                    continue
                parts.append({
                    "inlineData": {
                        "mimeType": item.media_type,
                        "data": item.data,
                    },
                })
            if omitted_here:
                parts.append({
                    "text": (
                        f"[{omitted_here} earlier image(s) omitted to respect "
                        "the serving request limit]"
                    ),
                })
            if not parts:
                parts.append({"text": ""})

            if message.role == "system":
                system_parts.extend(parts)
                continue
            role = "model" if message.role == "assistant" else "user"
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": parts})

        if not contents:
            contents.append({"role": "user", "parts": [{"text": ""}]})
        request: dict[str, Any] = {"contents": contents}
        if system_parts:
            request["systemInstruction"] = {"parts": system_parts}
        return request

    @staticmethod
    def _response_text(
        response: Mapping[str, Any],
        *,
        allow_structured_thought_fallback: bool = False,
    ) -> str:
        candidates = response.get("candidates")
        if not isinstance(candidates, Sequence) or not candidates:
            feedback = response.get("promptFeedback")
            raise RuntimeError(f"Gemini response has no candidates: {feedback!r}")
        candidate = candidates[0]
        if not isinstance(candidate, Mapping):
            raise RuntimeError("Gemini response candidate is not an object")
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, Mapping) else None
        if not isinstance(parts, Sequence):
            raise RuntimeError("Gemini response candidate has no content parts")
        text = "".join(
            str(part.get("text") or "")
            for part in parts
            if isinstance(part, Mapping) and not bool(part.get("thought"))
        )
        if text:
            return text

        # Some Gemini gateways occasionally mark the complete JSON response as
        # a thought part and omit the ordinary text part.  Never expose or run
        # arbitrary hidden reasoning: this fallback is enabled only for a
        # schema-constrained request and only accepts an exact action object.
        thought_text = "".join(
            str(part.get("text") or "")
            for part in parts
            if isinstance(part, Mapping) and bool(part.get("thought"))
        ).strip()
        if allow_structured_thought_fallback and thought_text:
            try:
                action = json.loads(thought_text)
            except json.JSONDecodeError:
                action = None
            if (
                isinstance(action, Mapping)
                and isinstance(action.get("tool"), str)
                and isinstance(action.get("args"), Mapping)
            ):
                return thought_text

        finish_reason = candidate.get("finishReason")
        raise ModelResponseFormatError(
            "Gemini response contains no ordinary text"
            f" (finish_reason={finish_reason!r}, thought_chars={len(thought_text)})"
        )

    @staticmethod
    def _post_json(
        endpoint: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
    ) -> Mapping[str, Any]:
        request = Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=dict(headers),
            method="POST",
        )
        raw = ""
        for attempt in range(3):
            try:
                with urlopen(request, timeout=timeout) as response:
                    raw = response.read().decode("utf-8")
                break
            except HTTPError as exc:
                detail = exc.read(4096).decode("utf-8", errors="replace")
                if exc.code in {429, 502, 503, 504} and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(
                    f"Gemini API returned HTTP {exc.code}: {detail}"
                ) from exc
            except TimeoutError as exc:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError("Gemini API request timed out") from exc
            except URLError as exc:
                raise RuntimeError(f"Gemini API request failed: {exc.reason}") from exc
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Gemini API returned invalid JSON") from exc
        if not isinstance(value, Mapping):
            raise RuntimeError("Gemini API response must be a JSON object")
        return value

    @staticmethod
    def _gemini_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
        """Project an action JSON Schema onto Gemini's supported subset.

        AgentRollout offers an exact ``oneOf`` branch per tool.  Sending that
        full union to Gemini is both unnecessarily large and prone to provider
        schema-complexity limits.  Gemini still receives the exact tool catalog
        in the prompt, while this compact schema guarantees one JSON action and
        constrains the selected tool name.  ToolLoop remains authoritative for
        per-tool argument validation.
        """

        detached = json.loads(json.dumps(dict(schema), ensure_ascii=False))
        branches = detached.get("oneOf")
        if detached.get("type") == "object" and isinstance(branches, list):
            tool_names: list[str] = []
            for branch in branches:
                if not isinstance(branch, Mapping):
                    break
                properties = branch.get("properties")
                tool = properties.get("tool") if isinstance(properties, Mapping) else None
                name = tool.get("const") if isinstance(tool, Mapping) else None
                if not isinstance(name, str):
                    break
                tool_names.append(name)
            else:
                if tool_names:
                    return {
                        "type": "object",
                        "properties": {
                            "thought": {"type": "string"},
                            "tool": {"type": "string", "enum": tool_names},
                            "args": {"type": "object"},
                        },
                        "required": ["thought", "tool", "args"],
                    }

        def normalize(value: Any) -> Any:
            if isinstance(value, list):
                return [normalize(item) for item in value]
            if not isinstance(value, Mapping):
                return value
            result: dict[str, Any] = {}
            for key, item in value.items():
                if key == "oneOf":
                    result["anyOf"] = normalize(item)
                elif key == "const":
                    result["enum"] = [normalize(item)]
                else:
                    result[str(key)] = normalize(item)
            return result

        return normalize(detached)

    @classmethod
    def _structured_generation_config(
        cls,
        request_options: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Translate the runtime's OpenAI-shaped JSON schema to Gemini REST."""

        remaining = dict(request_options or {})
        response_format = remaining.pop("response_format", None)
        if response_format is None:
            return {}, remaining
        if not isinstance(response_format, Mapping):
            raise TypeError("response_format must be an object")
        if response_format.get("type") != "json_schema":
            raise ValueError("Gemini only supports json_schema response_format")
        descriptor = response_format.get("json_schema")
        schema = descriptor.get("schema") if isinstance(descriptor, Mapping) else None
        if not isinstance(schema, Mapping):
            raise TypeError("response_format.json_schema.schema must be an object")
        return {
            "responseFormat": {
                "text": {
                    "mimeType": "application/json",
                    "schema": cls._gemini_schema(schema),
                },
            },
        }, remaining

    def _generate_one(
        self,
        messages: Sequence[Message],
        request_options: Mapping[str, Any] | None = None,
    ) -> str:
        payload = self.provider_request(
            messages,
            max_images_per_request=self.max_images_per_request,
        )
        generation_config: dict[str, Any] = {}
        if self.max_tokens is not None:
            generation_config["maxOutputTokens"] = self.max_tokens
        if self.temperature is not None:
            generation_config["temperature"] = self.temperature
        base_options = dict(self.request_options)
        base_generation = base_options.pop("generationConfig", None)
        if base_generation is not None:
            if not isinstance(base_generation, Mapping):
                raise TypeError("request_options.generationConfig must be an object")
            generation_config.update(base_generation)
        structured, per_request = self._structured_generation_config(request_options)
        per_generation = per_request.pop("generationConfig", None)
        if per_generation is not None:
            if not isinstance(per_generation, Mapping):
                raise TypeError("per-request generationConfig must be an object")
            generation_config.update(per_generation)
        generation_config.update(structured)
        payload.update(base_options)
        payload.update(per_request)
        if generation_config:
            payload["generationConfig"] = generation_config
        for attempt in range(self.malformed_response_retries + 1):
            response = self.transport(
                self.endpoint(self.base_url, self.model),
                self.headers,
                payload,
                self.timeout,
            )
            record_gemini_usage(response)
            try:
                return self._response_text(
                    response,
                    allow_structured_thought_fallback=bool(structured),
                )
            except ModelResponseFormatError:
                if (
                    attempt >= self.malformed_response_retries
                ):
                    raise
        raise AssertionError("unreachable")

    def generate_messages(
        self,
        conversations: Sequence[Sequence[Message]],
    ) -> list[str]:
        return self.generate_messages_with_options(
            conversations,
            [None] * len(conversations),
        )

    def generate_messages_with_options(
        self,
        conversations: Sequence[Sequence[Message]],
        request_options: Sequence[Mapping[str, Any] | None],
    ) -> list[str]:
        if len(conversations) != len(request_options):
            raise ValueError(
                "conversations and request_options must have equal length"
            )
        if self.max_workers == 1 or len(conversations) <= 1:
            return [
                self._generate_one(messages, options)
                for messages, options in zip(conversations, request_options)
            ]
        results = [""] * len(conversations)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    contextvars.copy_context().run, self._generate_one, messages, options
                ): index
                for index, (messages, options) in enumerate(
                    zip(conversations, request_options)
                )
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return results


__all__ = ["DEFAULT_GEMINI_BASE_URL", "GeminiServing", "GeminiTransport"]
