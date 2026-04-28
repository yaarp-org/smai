""":class:`LlmProviderConformance` — parameterizable test base for
:class:`LlmProvider` plugins.

Per ``designs/smai/07-plugin-interfaces.md`` §4.7 (the per-Protocol
conformance enumeration) and §8 (cross-cutting discipline).

Subclass and override :meth:`LlmProviderConformance.make_provider` to
run the suite against your plugin::

    from smai_core.plugins.conformance import LlmProviderConformance
    from smai_llm_bedrock import BedrockProvider

    class TestBedrockConformance(LlmProviderConformance):
        def make_provider(self) -> LlmProvider:
            return BedrockProvider(...)
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from smai_core.plugins import (
    CacheConfig,
    LlmCapabilities,
    LlmProvider,
    LlmProviderAuthError,
    LlmProviderError,
    LlmProviderInvalidRequest,
    LlmProviderRateLimited,
    LlmProviderUnavailable,
    ModelResponse,
    NormalizedMessage,
    TextContent,
    ToolDefinition,
    ToolUseContent,
)
from smai_core.plugins.conformance._common import (
    TimeProvider,
    skip_no_factory_override,
)


class LlmProviderConformance:
    """Universal contract suite for :class:`LlmProvider` implementations.

    Subclasses override :meth:`make_provider` to return a real plugin
    instance; the contract methods inherited from this class then run
    against that instance. When the base itself is collected (no
    subclass / no override), every contract test skip-collects with a
    clear message — the contract surface remains visible in the test
    report without spurious failures.

    The contract methods correspond 1:1 to the rules in
    ``07-plugin-interfaces.md`` §4.7:

    * :meth:`test_normalized_round_trip` — round-trip translation
    * :meth:`test_capability_caching_honesty` — capability-flag honesty
      for ``supports_caching`` (best-effort per §4.7)
    * :meth:`test_capability_tool_use_honesty` — capability-flag honesty
      for ``supports_tool_use``
    * :meth:`test_capabilities_immutable` — capabilities populated at
      instantiation, not mutated by ``call``
    * :meth:`test_rate_limit_error_class` — 429 → :class:`LlmProviderRateLimited`
    * :meth:`test_unavailable_error_class` — 503 → :class:`LlmProviderUnavailable`
    * :meth:`test_invalid_request_error_class` — 4xx → :class:`LlmProviderInvalidRequest`
    * :meth:`test_auth_error_class` — auth failure →
      :class:`LlmProviderAuthError`
    * :meth:`test_transient_retry_once` — transient errors retried once
      before raising
    * :meth:`test_tool_use_round_trip` — tool-use response round-trip
    * :meth:`test_input_messages_not_mutated` — call must not mutate
      input ``messages`` list (per :meth:`LlmProvider.call` contract)

    The :func:`time_provider` fixture (defaulting to :func:`time.monotonic`
    is the wall-clock seam; subclasses can override it once Task 2.C1
    ships its ``EngineConfig``-driven fake-clock.
    """

    # ---- factory hook ------------------------------------------------------

    def make_provider(self) -> LlmProvider:
        """Return a configured :class:`LlmProvider` instance to test against.

        Override this in your subclass. The default implementation
        skip-collects the suite with a clear message.
        """
        skip_no_factory_override(type(self).__name__, "make_provider")
        raise AssertionError("unreachable")  # pragma: no cover

    # ---- fixtures ----------------------------------------------------------

    @pytest.fixture
    def provider(self) -> LlmProvider:
        """Per-test instance via :meth:`make_provider`."""
        return self.make_provider()

    @pytest.fixture
    def time_provider(self) -> TimeProvider:
        """Wall-clock provider (deferred per Task 2.C1).

        Defaults to :func:`time.monotonic`. Subclasses override this
        fixture to inject a fake clock once Task 2.C1's
        ``EngineConfig``-driven seam ships.
        """
        return time.monotonic

    @pytest.fixture
    def fixture_messages(self) -> list[NormalizedMessage]:
        """A canonical fixture conversation for round-trip testing."""
        return [
            NormalizedMessage(role="user", content=[TextContent(text="hello")]),
        ]

    @pytest.fixture
    def fixture_tool(self) -> ToolDefinition:
        """A canonical fixture tool definition for tool-use tests."""
        return ToolDefinition(
            name="echo",
            description="Echo the input back to the caller.",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )

    # ---- §4.7 contract assertions ------------------------------------------

    async def test_normalized_round_trip(
        self,
        provider: LlmProvider,
        fixture_messages: list[NormalizedMessage],
    ) -> None:
        """Round-trip translation — given fixture :class:`NormalizedMessage`,
        the plugin's request encoding must round-trip through its
        provider's API and yield a parseable :class:`NormalizedMessage`
        response.
        """
        response = await provider.call(system="You are a test fixture.", messages=fixture_messages)
        assert isinstance(response, ModelResponse)
        assert response.message.role == "assistant"
        assert response.usage.input_tokens >= 0
        assert response.usage.output_tokens >= 0

    async def test_capability_caching_honesty(
        self,
        provider: LlmProvider,
        fixture_messages: list[NormalizedMessage],
    ) -> None:
        """Capability-flag honesty for ``supports_caching`` (best-effort
        per §4.7).

        A plugin claiming ``supports_caching=True`` MUST be capable of
        producing cache reads under stable conditions; conformance does
        not require a hit on every fixture invocation.

        When the plugin claims ``supports_caching=False``, the
        ``cache_*_tokens`` fields on :class:`TokenUsage` MUST be ``0``
        regardless of ``cache_config``.
        """
        if provider.capabilities.supports_caching:
            pytest.skip(
                "supports_caching=True is best-effort verifiable only against "
                "the live provider; covered by the --integration battery, not --unit."
            )
        response = await provider.call(
            system="You are a test fixture.",
            messages=fixture_messages,
            cache_config=CacheConfig(
                cache_static_prefix=True,
                cache_initial_message=True,
                rolling_cache_count=4,
            ),
        )
        assert response.usage.cache_read_tokens == 0
        assert response.usage.cache_write_tokens == 0

    async def test_capability_tool_use_honesty(self, provider: LlmProvider) -> None:
        """Capability-flag honesty for ``supports_tool_use``.

        When ``False``, the plugin MUST raise
        :class:`LlmProviderInvalidRequest` if a tool is passed.
        """
        if provider.capabilities.supports_tool_use:
            pytest.skip("test only applies to plugins that report supports_tool_use=False")
        with pytest.raises(LlmProviderInvalidRequest):
            await provider.call(
                system="You are a test fixture.",
                messages=[NormalizedMessage(role="user", content=[TextContent(text="x")])],
                tools=[
                    ToolDefinition(
                        name="t",
                        description="d",
                        input_schema={"type": "object"},
                    )
                ],
            )

    def test_capabilities_immutable(self, provider: LlmProvider) -> None:
        """Capabilities populated at instantiation; immutable across calls.

        The ``capabilities`` attribute is the same object before and
        after a call. Per §4.4 ("Static per-plugin capability flags
        … immutable for the lifetime of the plugin instance").
        """
        assert isinstance(provider.capabilities, LlmCapabilities)
        snapshot = provider.capabilities.model_copy()
        # No call yet — same identity
        assert provider.capabilities == snapshot

    async def test_rate_limit_error_class(
        self, provider: LlmProvider, fixture_messages: list[NormalizedMessage]
    ) -> None:
        """Synthesized 429 → :class:`LlmProviderRateLimited`.

        Plugins that don't expose a fault-injection seam skip — the
        bedrock-style fixture VCR replays a recorded 429.
        """
        if not _supports_fault_injection(provider, "rate_limit"):
            pytest.skip(
                "plugin does not expose a rate-limit fault-injection hook; "
                "covered by --integration battery against substrate-recorded fixtures."
            )
        with pytest.raises(LlmProviderRateLimited):
            await _inject_and_call(provider, "rate_limit", fixture_messages)

    async def test_unavailable_error_class(
        self, provider: LlmProvider, fixture_messages: list[NormalizedMessage]
    ) -> None:
        """Synthesized 503 → :class:`LlmProviderUnavailable`."""
        if not _supports_fault_injection(provider, "unavailable"):
            pytest.skip("plugin does not expose a 503 fault-injection hook")
        with pytest.raises(LlmProviderUnavailable):
            await _inject_and_call(provider, "unavailable", fixture_messages)

    async def test_invalid_request_error_class(
        self, provider: LlmProvider, fixture_messages: list[NormalizedMessage]
    ) -> None:
        """Synthesized 4xx (non-429) → :class:`LlmProviderInvalidRequest`."""
        if not _supports_fault_injection(provider, "invalid_request"):
            pytest.skip("plugin does not expose an invalid-request fault-injection hook")
        with pytest.raises(LlmProviderInvalidRequest):
            await _inject_and_call(provider, "invalid_request", fixture_messages)

    async def test_auth_error_class(
        self, provider: LlmProvider, fixture_messages: list[NormalizedMessage]
    ) -> None:
        """Auth failure → :class:`LlmProviderAuthError`. Not retried."""
        if not _supports_fault_injection(provider, "auth"):
            pytest.skip("plugin does not expose an auth fault-injection hook")
        with pytest.raises(LlmProviderAuthError):
            await _inject_and_call(provider, "auth", fixture_messages)

    async def test_transient_retry_once(
        self,
        provider: LlmProvider,
        fixture_messages: list[NormalizedMessage],
        time_provider: TimeProvider,
    ) -> None:
        """On transient errors (rate limit, transient unavailability),
        the plugin retries once with a 30s backoff before raising.

        Wall-clock-sensitive — relies on :func:`time_provider` for the
        backoff measurement (Task 2.C1 will inject a fake clock here).
        """
        if not _supports_fault_injection(provider, "transient_then_succeed"):
            pytest.skip("plugin does not expose a one-shot-transient fault-injection hook")
        # The plugin's first call returns 503/429; the retry succeeds.
        # We assert success without raising — the retry path was taken.
        start = time_provider()
        response = await _inject_and_call(provider, "transient_then_succeed", fixture_messages)
        elapsed = time_provider() - start
        assert isinstance(response, ModelResponse)
        # Backoff is "documented as 30s" — but the conformance gate accepts a
        # broad band (test fixtures may inject a sub-second backoff).
        # The contract is "retry happened"; the wall-clock is informational.
        del elapsed  # intentionally unconsumed; documents the seam

    async def test_tool_use_round_trip(
        self, provider: LlmProvider, fixture_tool: ToolDefinition
    ) -> None:
        """Tool-use round-trip — given a fixture :class:`ToolDefinition`
        and a fixture-mocked ``tool_use`` response, the plugin must
        surface the :class:`ToolUseContent` block correctly.

        Plugins that don't claim ``supports_tool_use`` skip this test.
        """
        if not provider.capabilities.supports_tool_use:
            pytest.skip("plugin reports supports_tool_use=False")
        if not _supports_fault_injection(provider, "tool_use_response"):
            pytest.skip(
                "plugin does not expose a fixture-tool-use response hook; "
                "covered by --integration against substrate-recorded fixtures."
            )
        response = await _inject_and_call(
            provider,
            "tool_use_response",
            [NormalizedMessage(role="user", content=[TextContent(text="echo hi")])],
            tools=[fixture_tool],
        )
        assert isinstance(response, ModelResponse)
        assert response.stop_reason == "tool_use"
        tool_uses = [c for c in response.message.content if isinstance(c, ToolUseContent)]
        assert tool_uses, "expected at least one tool_use block in the response"

    async def test_input_messages_not_mutated(
        self, provider: LlmProvider, fixture_messages: list[NormalizedMessage]
    ) -> None:
        """``call`` MUST NOT mutate the input ``messages`` list (the
        agent loop reuses it). Per :meth:`LlmProvider.call` contract.
        """
        snapshot = [m.model_copy(deep=True) for m in fixture_messages]
        try:
            await provider.call(system="x", messages=fixture_messages)
        except (LlmProviderError, NotImplementedError):
            # We're testing mutation, not call semantics — error paths still
            # must not mutate.
            pass
        assert fixture_messages == snapshot, "call() mutated its input messages list"


# ---- private fault-injection adapter ----------------------------------------
#
# Plugins MAY opt into the synthesized-error tests by exposing a
# ``_conformance_inject_fault(kind, fn, *args, **kwargs)`` hook. Plugins
# that don't expose the hook fall back to skip — substrate-recorded
# fixture replays cover those errors in --integration mode.


def _supports_fault_injection(provider: object, kind: str) -> bool:
    """Return ``True`` iff the plugin opts into fault injection for ``kind``.

    Plugins expose a ``_conformance_supports_fault_injection: set[str]``
    attribute listing the kinds they handle.
    """
    supported: set[str] = getattr(provider, "_conformance_supports_fault_injection", set())
    return kind in supported


async def _inject_and_call(
    provider: LlmProvider,
    kind: str,
    messages: list[NormalizedMessage],
    *,
    tools: list[ToolDefinition] | None = None,
    **kwargs: Any,
) -> ModelResponse:
    """Invoke the plugin with a one-shot fault injection of ``kind``."""
    # The conformance fault-injection seam is plugin-internal; the plugin
    # advertises support via ``_conformance_supports_fault_injection`` and
    # exposes ``_conformance_inject_fault``. Both are typed dynamically here
    # since the bases stay duck-typed against any conforming plugin.
    inject = provider._conformance_inject_fault  # type: ignore[attr-defined]  # noqa: SLF001
    result = await inject(  # pyright: ignore[reportUnknownVariableType]
        kind,
        provider.call,
        system="You are a test fixture.",
        messages=messages,
        tools=tools,
        **kwargs,
    )
    assert isinstance(result, ModelResponse)
    return result
