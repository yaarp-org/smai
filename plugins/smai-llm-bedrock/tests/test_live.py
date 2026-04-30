"""Opt-in live AWS Bedrock smoke test.

Runs only when ``BEDROCK_LIVE_TESTS=1`` is set in the environment, per
the Task 2.A1 brief: an opt-in live-call test runs locally with AWS
credentials. CI never sets this flag — keeps the test suite at zero
cloud cost.

The test exercises a single ``Converse`` round-trip against the
configured model (default: ``us.anthropic.claude-haiku-4-5-20251001-v1:0``
— the cheapest Claude tier; override via ``BEDROCK_LIVE_MODEL_ID``).
"""

from __future__ import annotations

import os

import pytest
from smai_core.plugins import NormalizedMessage, TextContent
from smai_llm_bedrock import BedrockProvider

_SKIP_REASON = "live AWS Bedrock test — set BEDROCK_LIVE_TESTS=1 to run"


@pytest.mark.credentialed
@pytest.mark.skipif(
    os.environ.get("BEDROCK_LIVE_TESTS") != "1",
    reason=_SKIP_REASON,
)
async def test_live_round_trip() -> None:
    region = os.environ.get("AWS_REGION", "us-east-1")
    model_id = os.environ.get(
        "BEDROCK_LIVE_MODEL_ID",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    )
    provider = BedrockProvider(region=region, model_id=model_id)
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
