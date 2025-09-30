"""High level client for multi-turn conversations with OpenAI GPT models."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ._internal.message_parser import parse_message
from ._internal.openai_transport import OpenAITransport
from .types import ClaudeAgentOptions, Message


class ClaudeSDKClient:
    """Minimal async client that keeps conversation state in memory."""

    def __init__(
        self,
        options: ClaudeAgentOptions | None = None,
        transport: OpenAITransport | None = None,
    ) -> None:
        self.options = options or ClaudeAgentOptions()
        self._transport = transport or OpenAITransport(self.options)
        self._history: list[dict[str, Any]] = []
        self._last_messages: list[Message] = []

    async def __aenter__(self) -> "ClaudeSDKClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def query(self, prompt: str) -> None:
        """Send a new user prompt and store the resulting assistant messages."""

        if not isinstance(prompt, str):
            raise TypeError("ClaudeSDKClient only supports string prompts")

        responses: list[Message] = []
        raw_messages: list[dict[str, Any]] = []

        async for raw in self._transport.generate(prompt, history=self._history):
            raw_messages.append(raw)
            responses.append(parse_message(raw))

        self._last_messages = responses
        self._append_turn(prompt, raw_messages)

    async def receive_response(self) -> AsyncIterator[Message]:
        """Yield the messages produced by the most recent query."""

        for message in self._last_messages:
            yield message

    def _append_turn(self, prompt: str, raw_messages: list[dict[str, Any]]) -> None:
        self._history.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        )

        for raw in raw_messages:
            if raw.get("type") != "assistant":
                continue
            content = raw.get("message", {}).get("content")
            if not isinstance(content, list):
                continue
            self._history.append({"role": "assistant", "content": content})

    async def disconnect(self) -> None:  # pragma: no cover - API parity helper
        """Provided for API compatibility with the Claude CLI client."""

        self._last_messages = []


__all__ = ["ClaudeSDKClient"]
