"""Internal client implementation backed by the OpenAI Responses API."""

from collections.abc import AsyncIterator
from typing import Any

from ..types import ClaudeAgentOptions, Message
from .message_parser import parse_message
from .openai_transport import OpenAITransport


class InternalClient:
    """Simplified internal client that proxies to the OpenAI transport."""

    async def process_query(
        self,
        prompt: str,
        options: ClaudeAgentOptions,
        transport: OpenAITransport | None = None,
    ) -> AsyncIterator[Message]:
        """Process a query and yield parsed messages."""

        if not isinstance(prompt, str):
            raise TypeError("query() currently only supports string prompts when using OpenAI")

        chosen_transport = transport or OpenAITransport(options)

        async for message in chosen_transport.generate(prompt):
            yield parse_message(message)
