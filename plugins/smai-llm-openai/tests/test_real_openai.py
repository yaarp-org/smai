"""Opt-in live OpenAI API smoke test.

Runs only when ``OPENAI_API_KEY`` is set in the environment, per the
Task 3.F5 brief: opt-in live-call test runs locally with real
credentials. CI never sets this — keeps the suite at zero cloud cost.

Marked with :mod:`pytest.mark.credentialed` per the no-credentials-in-CI
convention (root ``pyproject.toml``).

The test exercises one ``chat.completions.create`` round-trip + one
tool-call round-trip against the configured model (default:
``gpt-4o-mini`` — the cheapest GPT-4-class tier; override via
``OPENAI_LIVE_MODEL_ID``).
"""

from __future__ import annotations

import os

import pytest
from smai_core.plugins import (
    NormalizedMessage,
    TextContent,
    ToolDefinition,
)
from smai_llm_openai import OpenAIProvider

_SKIP_REASON = "live OpenAI test — set OPENAI_API_KEY to run"


@pytest.mark.credentialed
@pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason=_SKIP_REASON)
async def test_live_round_trip() -> None:
    model_id = os.environ.get("OPENAI_LIVE_MODEL_ID", "gpt-4o-mini")
    provider = OpenAIProvider(model_id=model_id)
    response = await provider.call(
        system="You are a terse oracle. Reply with one short sentence.",
        messages=[
            NormalizedMessage(
                role="user",
                content=[TextContent(text="Reply with the single word: pong.")],
            )
        ],
        max_tokens=32,
    )
    assert response.message.role == "assistant"
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    assert response.stop_reason in ("end_turn", "max_tokens", "tool_use")


@pytest.mark.credentialed
@pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason=_SKIP_REASON)
async def test_live_tool_use_round_trip() -> None:
    model_id = os.environ.get("OPENAI_LIVE_MODEL_ID", "gpt-4o-mini")
    provider = OpenAIProvider(model_id=model_id)
    response = await provider.call(
        system="You must call the echo tool with text='hi'.",
        messages=[
            NormalizedMessage(
                role="user",
                content=[TextContent(text="please echo 'hi'")],
            )
        ],
        tools=[
            ToolDefinition(
                name="echo",
                description="Echo the input back to the caller.",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        ],
        max_tokens=128,
    )
    # Just verify the response shape — model behavior on tool selection
    # is non-deterministic. Either a text reply or a tool_use is fine.
    assert response.message.role == "assistant"
    assert response.usage.input_tokens > 0
