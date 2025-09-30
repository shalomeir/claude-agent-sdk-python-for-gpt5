"""OpenAI transport used by the SDK to talk to GPT models."""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, AsyncIterable

from ..types import ClaudeAgentOptions


class OpenAITransport:
    """High level transport that proxies requests to the OpenAI Responses API."""

    def __init__(self, options: ClaudeAgentOptions) -> None:
        self._options = options
        self._client: Any | None = options.openai_client

    async def generate(
        self, prompt: str, *, history: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Generate response messages for a prompt."""

        if not isinstance(prompt, str):
            raise TypeError("OpenAITransport only supports string prompts")

        start_time = time.perf_counter()
        client = await self._ensure_client()

        request_payload = self._build_request(prompt, history)
        response = await client.responses.create(**request_payload)

        duration_ms = int((time.perf_counter() - start_time) * 1000)
        text = self._extract_text(response)
        usage = self._extract_usage(response)
        session_id = str(uuid.uuid4())

        assistant_message = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": request_payload.get("model", ""),
                "content": [{"type": "text", "text": text}],
            },
        }

        result_message = {
            "type": "result",
            "subtype": "success",
            "duration_ms": duration_ms,
            "duration_api_ms": duration_ms,
            "is_error": False,
            "num_turns": 1,
            "session_id": session_id,
            "total_cost_usd": None,
            "usage": usage,
            "result": text,
        }

        yield assistant_message
        yield result_message

    async def stream(self, prompts: AsyncIterable[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
        """Placeholder for streaming prompts (not yet implemented)."""

        raise NotImplementedError("Streaming prompts are not yet supported with OpenAI models")

    async def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client

        from openai import AsyncOpenAI

        api_key = self._options.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "An OpenAI API key must be provided via ClaudeAgentOptions.openai_api_key "
                "or the OPENAI_API_KEY environment variable."
            )

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if self._options.openai_organization:
            client_kwargs["organization"] = self._options.openai_organization
        if self._options.openai_project:
            client_kwargs["project"] = self._options.openai_project

        self._client = AsyncOpenAI(**client_kwargs)
        return self._client

    def _build_request(
        self, prompt: str, history: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        model = self._options.model or "gpt-5-codex"
        input_messages = []

        if history:
            input_messages.extend(history)

        system_prompt = self._options.system_prompt
        if isinstance(system_prompt, dict):
            system_text = system_prompt.get("append")
        else:
            system_text = system_prompt

        if system_text:
            input_messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": str(system_text)}],
                }
            )

        input_messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        )

        payload: dict[str, Any] = {
            "model": model,
            "input": input_messages,
        }

        if self._options.temperature is not None:
            payload["temperature"] = self._options.temperature

        if self._options.max_output_tokens is not None:
            payload["max_output_tokens"] = self._options.max_output_tokens

        if self._options.response_format is not None:
            if isinstance(self._options.response_format, str):
                payload["response_format"] = {"type": self._options.response_format}
            else:
                payload["response_format"] = self._options.response_format

        if self._options.extra_headers:
            payload["extra_headers"] = self._options.extra_headers

        return payload

    def _extract_text(self, response: Any) -> str:
        if hasattr(response, "output_text"):
            text = getattr(response, "output_text")
            if text is not None:
                return str(text)

        # Fallback to checking output blocks
        output = getattr(response, "output", None)
        if output:
            for item in output:
                if isinstance(item, dict):
                    content = item.get("content")
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "output_text":
                                text_block = block.get("text")
                                if isinstance(text_block, str):
                                    return text_block

        raise ValueError("Unable to extract text from OpenAI response")

    def _extract_usage(self, response: Any) -> dict[str, Any] | None:
        usage = getattr(response, "usage", None)
        if not usage:
            return None

        if isinstance(usage, dict):
            return usage

        usage_dict = {}
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            if hasattr(usage, key):
                usage_dict[key] = getattr(usage, key)

        return usage_dict or None


__all__ = ["OpenAITransport"]
