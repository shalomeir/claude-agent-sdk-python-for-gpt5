"""Query function for one-shot interactions with OpenAI GPT models."""

import os
from collections.abc import AsyncIterator

from ._internal.client import InternalClient
from ._internal.openai_transport import OpenAITransport
from .types import ClaudeAgentOptions, Message


async def query(
    *,
    prompt: str,
    options: ClaudeAgentOptions | None = None,
    transport: OpenAITransport | None = None,
) -> AsyncIterator[Message]:
    """
    Query the OpenAI Responses API for a single prompt.

    Args:
        prompt: Text prompt to send to the model.
        options: Optional configuration object. If omitted a default
            ``ClaudeAgentOptions`` instance is created.
        transport: Optionally supply a custom ``OpenAITransport`` instance.

    Yields:
        ``Message`` objects representing the model response followed by an
        execution summary.

    Example:
        ```python
        async for message in query(prompt="Say hello"):
            print(message)
        ```

    """
    if options is None:
        options = ClaudeAgentOptions()

    os.environ["CLAUDE_CODE_ENTRYPOINT"] = "sdk-py"

    client = InternalClient()

    async for message in client.process_query(
        prompt=prompt, options=options, transport=transport
    ):
        yield message
