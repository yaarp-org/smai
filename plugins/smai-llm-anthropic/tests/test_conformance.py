""":class:`LlmProviderConformance` against :class:`AnthropicProvider`.

Subclass the conformance suite shipped under
``smai_core.plugins.conformance`` and override :meth:`make_provider` to
return a real plugin instance backed by an in-process
:class:`FakeAnthropicClient` — same shape as the Bedrock plugin's
settled fault-injection pattern.
"""

from __future__ import annotations

from _f5_anthropic_fakes import FakeAnthropicClient  # type: ignore[import-not-found]
from smai_core.plugins.conformance.test_llm_provider import LlmProviderConformance
from smai_llm_anthropic import AnthropicProvider


class TestAnthropicConformance(LlmProviderConformance):
    """Drive the §4.7 conformance suite against :class:`AnthropicProvider`.

    The plugin's :class:`FakeAnthropicClient` is the in-process stand-in
    for ``anthropic.AsyncAnthropic``; its in-memory outcome queue is
    the seam through which
    :meth:`AnthropicProvider._conformance_inject_fault` stages the
    Anthropic-shaped errors (status_code 429 / 503 / 400 / 401) and
    canned tool-use responses the suite verifies.
    """

    def make_provider(self) -> AnthropicProvider:
        return AnthropicProvider(
            model_id="claude-opus-4-7",
            anthropic_client=FakeAnthropicClient(),
            transient_backoff_seconds=0.0,
        )
