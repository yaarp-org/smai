""":class:`LlmProviderConformance` against :class:`BedrockProvider`.

Per the Task 2.A1 brief: subclass the conformance suite shipped under
``smai_core.plugins.conformance`` and override :meth:`make_provider` to
return a real plugin instance. Per the carry-forward note from Task 1.9,
this task **settles** the ``_conformance_inject_fault`` synthesis pattern
by overriding the underlying client with a :class:`FakeBedrockClient`
and queuing Bedrock-specific exceptions through it.
"""

from __future__ import annotations

from _bedrock_fakes import FakeBedrockClient  # type: ignore[import-not-found]
from smai_core.plugins.conformance.test_llm_provider import LlmProviderConformance
from smai_llm_bedrock import BedrockProvider


class TestBedrockConformance(LlmProviderConformance):
    """Drive the §4.7 conformance suite against :class:`BedrockProvider`.

    The plugin's :class:`FakeBedrockClient` is the in-process stand-in
    for the boto3 ``bedrock-runtime`` client; its in-memory outcome
    queue is the seam through which
    :meth:`BedrockProvider._conformance_inject_fault` stages the
    Bedrock-side exceptions (``ThrottlingException``,
    ``ValidationException``, ``ServiceUnavailableException``,
    ``AccessDeniedException``) the suite's error-class translation
    tests verify.
    """

    def make_provider(self) -> BedrockProvider:
        return BedrockProvider(
            region="us-east-1",
            model_id="us.anthropic.claude-opus-4-7-v1",
            bedrock_client=FakeBedrockClient(),
            # Fast-forward the §4.5 retry-once 30s backoff so the
            # ``test_transient_retry_once`` contract runs under a
            # second. The conformance suite's contract is
            # "retry happened"; the wall-clock measurement is
            # informational (per the comment on
            # ``test_transient_retry_once``).
            transient_backoff_seconds=0.0,
        )
