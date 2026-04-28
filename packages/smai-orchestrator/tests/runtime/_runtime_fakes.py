"""Fake LlmProvider for runtime-package tests.

The engine `_helpers` module ships :class:`FakeCompute` /
:class:`FakeArtifactStore` but no fake :class:`LlmProvider` — the
engine substrate doesn't exercise LLM calls. The runtime tests need
one to satisfy the :class:`InstantiatedPlugins.llm_providers` map; it
lives here to avoid widening the engine helpers' surface for one
caller.
"""

from __future__ import annotations

from typing import ClassVar

from smai_core.plugins import (
    CacheConfig,
    LlmCapabilities,
    LlmProvider,
    ModelResponse,
    NormalizedMessage,
    TextContent,
    TokenUsage,
    ToolDefinition,
)


class FakeLlmProvider:
    """Minimal :class:`LlmProvider` Protocol-conforming stub.

    Returns a canned single-text-block response for every ``call``;
    tests that assert on call arguments inspect :attr:`call_log`.
    """

    name: str = "fake-llm"
    capabilities: ClassVar[LlmCapabilities] = LlmCapabilities(
        supports_caching=False,
        context_window=128_000,
        max_output_tokens=4_096,
        model_id="fake-llm-v1",
    )

    def __init__(self, response_text: str = "ok") -> None:
        self._response_text = response_text
        self.call_log: list[dict[str, object]] = []

    async def call(  # noqa: PLR0913
        self,
        system: str,
        messages: list[NormalizedMessage],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        cache_config: CacheConfig | None = None,
    ) -> ModelResponse:
        self.call_log.append(
            {
                "system": system,
                "messages": list(messages),
                "tools": list(tools) if tools else None,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "cache_config": cache_config,
            }
        )
        return ModelResponse(
            message=NormalizedMessage(
                role="assistant",
                content=[TextContent(text=self._response_text)],
            ),
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )


# Sanity check at import time; if the Protocol drifts we fail fast
# rather than at first test invocation.
assert isinstance(FakeLlmProvider("init"), LlmProvider)


__all__ = ["FakeLlmProvider"]
