"""Round-6 item C: the Bedrock client is constructed with an explicit
``botocore.config.Config`` (connect/read timeouts + bounded internal
retries), and the per-call ``asyncio.wait_for`` ceiling turns a wedged
call into a retryable :class:`LlmProviderUnavailable` instead of hanging
the inline worker."""

from __future__ import annotations

import time
from typing import Any

import pytest
from _bedrock_fakes import FakeBedrockClient  # type: ignore[import-not-found]
from smai_core.plugins import LlmProviderUnavailable
from smai_llm_bedrock import BedrockProvider
from smai_llm_bedrock._provider import _build_bedrock_client


def test_real_client_carries_botocore_config() -> None:
    client = _build_bedrock_client(
        "us-east-1",
        connect_timeout_seconds=7,
        read_timeout_seconds=99,
        max_retries=3,
    )
    cfg = client.meta.config
    assert cfg.connect_timeout == 7
    assert cfg.read_timeout == 99
    # botocore normalizes ``max_attempts`` into ``total_max_attempts`` (= +1).
    assert cfg.retries["mode"] == "standard"
    assert cfg.retries.get("total_max_attempts") == 4 or cfg.retries.get("max_attempts") == 3


class _SlowClient(FakeBedrockClient):
    """Fake whose ``converse`` blocks its worker thread long enough for
    the ``wait_for`` ceiling to fire first."""

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        time.sleep(0.5)
        return super().converse(**kwargs)


async def _instant(_seconds: float) -> None:
    return None


async def test_send_surfaces_retryable_error_on_ceiling() -> None:
    provider = BedrockProvider(
        region="us-east-1",
        model_id="us.anthropic.claude-opus-4-6-v1",
        bedrock_client=_SlowClient(),
        sleep=_instant,
    )
    provider._call_timeout_seconds = 0.05  # shrink the hard ceiling for the test
    with pytest.raises(LlmProviderUnavailable):
        await provider._send({"modelId": "x"})


async def test_call_retries_once_then_propagates_on_repeated_ceiling() -> None:
    backoffs: list[float] = []

    async def _record(seconds: float) -> None:
        backoffs.append(seconds)

    provider = BedrockProvider(
        region="us-east-1",
        model_id="us.anthropic.claude-opus-4-6-v1",
        bedrock_client=_SlowClient(),
        sleep=_record,
    )
    provider._call_timeout_seconds = 0.05
    with pytest.raises(LlmProviderUnavailable):
        await provider.call(system="s", messages=[])
    # The §4.5 transient-retry path fired exactly once before propagating.
    assert len(backoffs) == 1
