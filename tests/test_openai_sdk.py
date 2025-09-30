"""Integration tests for the OpenAI backed agent SDK."""

from __future__ import annotations

import anyio

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, query
from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock


class StubTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def generate(self, prompt: str, *, history=None):
        self.calls.append({"prompt": prompt, "history": history})
        batch = self.responses.pop(0) if self.responses else []
        for response in batch:
            yield response


class FakeUsage:
    def __init__(self) -> None:
        self.input_tokens = 12
        self.output_tokens = 24


class FakeResponse:
    def __init__(self) -> None:
        self.output_text = "Hello from GPT"
        self.usage = FakeUsage()


class FakeResponsesAPI:
    def __init__(self) -> None:
        self.last_kwargs = None

    async def create(self, **kwargs):  # type: ignore[override]
        self.last_kwargs = kwargs
        return FakeResponse()


class FakeAsyncOpenAI:
    def __init__(self) -> None:
        self.responses = FakeResponsesAPI()


def test_query_with_stub_transport():
    async def _run() -> None:
        responses = [[
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "model": "gpt-5-codex",
                    "content": [{"type": "text", "text": "42"}],
                },
            },
            {
                "type": "result",
                "subtype": "success",
                "duration_ms": 10,
                "duration_api_ms": 10,
                "is_error": False,
                "num_turns": 1,
                "session_id": "test-session",
                "total_cost_usd": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "result": "42",
            },
        ]]

        transport = StubTransport(responses)
        options = ClaudeAgentOptions(model="gpt-5-codex")

        messages = []
        async for message in query(
            prompt="What is the answer?", options=options, transport=transport
        ):
            messages.append(message)

        assert isinstance(messages[0], AssistantMessage)
        assert isinstance(messages[0].content[0], TextBlock)
        assert messages[0].content[0].text == "42"
        assert isinstance(messages[1], ResultMessage)
        assert transport.calls[0]["prompt"] == "What is the answer?"

    anyio.run(_run)


def test_openai_transport_integration():
    async def _run() -> None:
        options = ClaudeAgentOptions(
            model="gpt-5",
            openai_api_key="dummy",
            openai_client=FakeAsyncOpenAI(),
            temperature=0.3,
            max_output_tokens=256,
            response_format="text",
        )

        from claude_agent_sdk._internal.openai_transport import OpenAITransport

        transport = OpenAITransport(options)

        raw_messages = []
        async for item in transport.generate("hello"):
            raw_messages.append(item)

        assert raw_messages[0]["message"]["content"][0]["text"] == "Hello from GPT"
        assert raw_messages[1]["usage"]["output_tokens"] == 24

        responses_api = options.openai_client.responses
        assert responses_api.last_kwargs["model"] == "gpt-5"
        assert responses_api.last_kwargs["temperature"] == 0.3
        assert responses_api.last_kwargs["max_output_tokens"] == 256

    anyio.run(_run)


def test_claude_sdk_client_tracks_history():
    async def _run() -> None:
        first = [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "model": "gpt-5-codex",
                    "content": [{"type": "text", "text": "Hi there"}],
                },
            },
        ]
        second = [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "model": "gpt-5-codex",
                    "content": [{"type": "text", "text": "Another reply"}],
                },
            },
        ]

        transport = StubTransport([first, second])
        options = ClaudeAgentOptions(model="gpt-5-codex")

        async with ClaudeSDKClient(options=options, transport=transport) as client:
            await client.query("Hello")
            messages = [message async for message in client.receive_response()]
            assert isinstance(messages[0], AssistantMessage)

            await client.query("What next?")
            assert len(transport.calls) == 2
            assert transport.calls[1]["history"]
            history = transport.calls[1]["history"]
            assert history[0]["role"] == "user"
            assert history[1]["role"] == "assistant"

    anyio.run(_run)
