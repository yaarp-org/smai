""":class:`LlmProviderConformance` against :class:`OpenAIProvider`.

Subclass the conformance suite and override :meth:`make_provider` to
return a real plugin instance backed by an in-process
:class:`FakeOpenAIClient`.
"""

from __future__ import annotations

from _f5_openai_fakes import FakeOpenAIClient  # type: ignore[import-not-found]
from smai_core.plugins.conformance.test_llm_provider import LlmProviderConformance
from smai_llm_openai import OpenAIProvider


class TestOpenAIConformance(LlmProviderConformance):
    """Drive the §4.7 conformance suite against :class:`OpenAIProvider`.

    The plugin's :class:`FakeOpenAIClient` is the in-process stand-in
    for ``openai.AsyncOpenAI``; its in-memory outcome queue is the
    seam through which
    :meth:`OpenAIProvider._conformance_inject_fault` stages the
    OpenAI-shaped errors (status_code 429 / 503 / 400 / 401) and
    canned tool-use responses the suite verifies.
    """

    def make_provider(self) -> OpenAIProvider:
        return OpenAIProvider(
            model_id="gpt-4o",
            openai_client=FakeOpenAIClient(),
            transient_backoff_seconds=0.0,
        )
