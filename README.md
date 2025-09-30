# GPT Agent SDK for Python

This repository provides a lightweight Python interface for talking to
OpenAI's GPT models (including `gpt-5` and `gpt-5-codex`). It started life as
the Anthropic Claude Agent SDK; the transport layer has been replaced so the
same high level API now talks to OpenAI's Responses API instead of the Claude
CLI.

## Installation

```bash
pip install -e .
```

### Requirements

- Python 3.10+
- An OpenAI API key (set `OPENAI_API_KEY` or pass it through
  `ClaudeAgentOptions.openai_api_key`)

## Quick start

```python
import anyio
from claude_agent_sdk import query


async def main() -> None:
    async for message in query(
        prompt="Write a short haiku about asynchronous Python",
    ):
        print(message)


anyio.run(main)
```

The SDK yields fully typed message objects. In the example above the first
message will be an `AssistantMessage` with a `TextBlock` containing the model's
response, followed by a `ResultMessage` with timing and token usage details (if
provided by the API).

## Conversation client

For multi-turn interactions use `ClaudeSDKClient`:

```python
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient


options = ClaudeAgentOptions(model="gpt-5-codex", temperature=0.4)


async def run_conversation() -> None:
    async with ClaudeSDKClient(options=options) as client:
        await client.query("Please introduce yourself")
        async for message in client.receive_response():
            print(message)

        await client.query("Thanks! Can you explain how streaming works?")
        async for message in client.receive_response():
            print(message)


anyio.run(run_conversation)
```

The client keeps the conversation history in memory and re-sends it to the API
for every new prompt.

## Configuration

`ClaudeAgentOptions` now exposes the most common OpenAI configuration knobs:

| Option | Description |
| ------ | ----------- |
| `model` | Target OpenAI model (defaults to `gpt-5-codex`). |
| `openai_api_key` | API key. If not provided the SDK falls back to `OPENAI_API_KEY`. |
| `openai_organization` / `openai_project` | Optional organisation/project scoping. |
| `temperature` | Sampling temperature forwarded to the Responses API. |
| `max_output_tokens` | Hard limit for generated tokens. |
| `response_format` | Either the string name of a response format (`"text"`, `"json_object"`, …) or a dict matching the OpenAI API schema. |
| `system_prompt` | Optional system prompt string. |

Legacy Claude-specific fields are still present on `ClaudeAgentOptions` for API
compatibility but they are ignored by the OpenAI backend.

## Testing

The test-suite runs without contacting the real API by injecting a mock
transport. You can execute it locally with:

```bash
pytest
```

## License

MIT
