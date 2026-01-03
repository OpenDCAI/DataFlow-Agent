import base64
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage

from dataflow_agent.llm_callers.base import BaseLLMCaller
from dataflow_agent.logger import get_logger
from dataflow_agent.state import MainState

log = get_logger(__name__)


class VisionLLMCaller(BaseLLMCaller):
    """
    视觉 LLM 调用器（精简版）

    - understanding: 支持图像输入 -> 文本输出（通过 chat/completions 多模态消息）
    - generation/edit: 图像生成/编辑链路已剥离，此能力在当前代码仓库不可用
    """

    def __init__(self, state: MainState, vlm_config: Dict[str, Any], **kwargs):
        super().__init__(state, **kwargs)
        self.vlm_config = vlm_config
        self.mode = vlm_config.get("mode", "understanding")
        self.temperature = kwargs.get("temperature", 0.1)

    async def call(self, messages: List[BaseMessage], bind_post_tools: bool = False) -> AIMessage:
        log.info(f"VisionLLM 调用，模型: {self.model_name}, 模式: {self.mode}")

        if self.mode in {"generation", "edit"}:
            raise RuntimeError(
                "VisionLLMCaller: image generation/edit is not available in this repository."
            )

        return await self._call_image_understanding(messages)

    async def _call_image_understanding(self, messages: List[BaseMessage]) -> AIMessage:
        ROLE_MAP = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "tool": "tool",
        }

        processed_messages: List[Dict[str, Any]] = []
        for msg in messages:
            lc_role = getattr(msg, "type", "human")
            role = ROLE_MAP.get(lc_role, "user")
            processed_messages.append({"role": role, "content": msg.content})

        if self.vlm_config.get("input_image") and processed_messages:
            b64, fmt = self._encode_image(self.vlm_config["input_image"])
            last_msg = processed_messages[-1]
            if last_msg["role"] == "user":
                last_msg["content"] = [
                    {"type": "text", "text": last_msg["content"]},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{fmt};base64,{b64}"},
                    },
                ]

        payload = {
            "model": self.model_name,
            "messages": processed_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        response_data = await self._post_chat_completions(payload)
        content = response_data["choices"][0]["message"]["content"]
        return AIMessage(content=content)

    def _encode_image(self, image_path: str) -> tuple[str, str]:
        with open(image_path, "rb") as file_handle:
            raw = file_handle.read()
        b64 = base64.b64encode(raw).decode("utf-8")

        ext = image_path.rsplit(".", 1)[-1].lower()
        if ext in {"jpg", "jpeg"}:
            fmt = "jpeg"
        elif ext == "png":
            fmt = "png"
        else:
            raise ValueError(f"Unsupported image format: {ext}")

        return b64, fmt

    async def _post_chat_completions(self, payload: dict) -> dict:
        import httpx

        base_url = (self.state.request.chat_api_url or "").rstrip("/")
        url = f"{base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.state.request.api_key}",
            "Content-Type": "application/json",
        }

        timeout_value = self.vlm_config.get("timeout", 120)
        if isinstance(timeout_value, str):
            try:
                timeout = float(timeout_value)
            except ValueError:
                timeout = 120
        else:
            timeout = float(timeout_value)

        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()
