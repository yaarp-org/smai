""":class:`AnthropicProvider` — :class:`LlmProvider` adapter for the
Anthropic native API.

Per ``07-plugin-interfaces.md`` §4 (the full Protocol surface) and
``04-agents.md`` §5 (prompt-caching integration). The Phase 3 plugin
per Task 3.F5; mirrors the shape of :class:`smai_llm_bedrock.BedrockProvider`
since Bedrock Converse is itself a hosted-Anthropic adapter.

Construction reads ``ANTHROPIC_API_KEY`` from the environment via the
``anthropic`` SDK's default chain — there is deliberately no
``api_key`` argument so credentials never enter shell history.
``ANTHROPIC_BASE_URL`` is honored similarly (proxy / self-hosted
deployments). Per-call:

1. Translate the normalized inputs into Anthropic native shape (see
   ``_translation``).
2. Apply ``cache_control`` markers when ``capabilities.supports_caching``
   is True and ``cache_config`` was passed.
3. Invoke ``AsyncAnthropic.messages.create`` — already async-native, no
   ``asyncio.to_thread`` workaround needed (unlike Bedrock's boto3).
4. On transient errors (rate-limit / unavailable / connection), sleep
   30s and retry once. If the retry also fails, propagate.
5. Translate the success response back into :class:`ModelResponse`.
"""

from __future__ import annotations

import asyncio
import os
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar, cast

from smai_core.plugins import (
    CacheConfig,
    LlmCapabilities,
    LlmProviderAuthError,
    LlmProviderError,
    LlmProviderInvalidRequest,
    ModelResponse,
    NormalizedMessage,
    ToolDefinition,
)

from smai_llm_anthropic._capabilities import lookup_capabilities
from smai_llm_anthropic._errors import is_transient, translate_sdk_error
from smai_llm_anthropic._translation import (
    apply_cache_control,
    from_anthropic_response,
    to_anthropic_messages,
    to_anthropic_tools,
)

# Per §4.5: plugins MUST retry transient errors exactly once with a 30s
# backoff. The seam for fast-forward in tests is the ``sleep`` ctor arg.
_DEFAULT_TRANSIENT_BACKOFF_SECONDS = 30.0

# Round-6 item C — the Anthropic SDK defaults to a ~10-minute per-request
# timeout; pass an explicit, config-exposed value so a stalled call
# doesn't pin the inline worker for ten minutes.
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0

# Sensible default model. Surfaced in the §3.F5 brief: pick a current
# Opus tier; the per-task model selection (DEC-022) typically overrides.
_DEFAULT_MODEL_ID = "claude-opus-4-7"

# Type alias for the per-call async client surface. The plugin works
# duck-typed against any object that exposes
# ``messages.create(**kwargs) -> Message`` (or dict, for fakes).
_AnthropicClient = Any


class AnthropicProvider:
    """Anthropic native-API adapter implementing :class:`LlmProvider`.

    Constructor::

        AnthropicProvider()  # default model + ANTHROPIC_API_KEY from env
        AnthropicProvider(model_id="claude-opus-4-7")
        AnthropicProvider(
            model_id="...",
            anthropic_client=fake_client,  # tests only
            capabilities=...,              # override the default lookup
            sleep=lambda s: None,          # tests only — bypass real backoff
        )
    """

    name: str = "anthropic"

    # Conformance fault-injection contract per Task 1.9 / Bedrock's
    # settled pattern (carry-forward note from `07` §4.7).
    _conformance_supports_fault_injection: ClassVar[set[str]] = {
        "rate_limit",
        "unavailable",
        "invalid_request",
        "auth",
        "transient_then_succeed",
        "tool_use_response",
    }

    capabilities: LlmCapabilities

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        *,
        anthropic_client: _AnthropicClient | None = None,
        capabilities: LlmCapabilities | None = None,
        max_tokens_default: int = 4096,
        transient_backoff_seconds: float = _DEFAULT_TRANSIENT_BACKOFF_SECONDS,
        request_timeout_seconds: float = _DEFAULT_REQUEST_TIMEOUT_SECONDS,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._model_id = model_id
        self.capabilities = capabilities or lookup_capabilities(model_id)
        self._client: _AnthropicClient = anthropic_client or _build_anthropic_client(
            request_timeout_seconds
        )
        self._max_tokens_default = max_tokens_default
        self._transient_backoff_seconds = transient_backoff_seconds
        self._sleep = sleep or asyncio.sleep

    @property
    def model_id(self) -> str:
        """The Anthropic model ID this instance is configured for.

        Per-task model selection (DEC-022) is owned by the agent loop;
        the agent constructs one :class:`AnthropicProvider` per per-task
        ``model_id`` and routes accordingly. Exposed read-only here so
        the agent can verify the routing it built.
        """
        return self._model_id

    async def call(
        self,
        system: str,
        messages: list[NormalizedMessage],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        cache_config: CacheConfig | None = None,
    ) -> ModelResponse:
        request = self._build_request(
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            cache_config=cache_config,
        )
        try:
            response = await self._send(request)
        except LlmProviderError as exc:
            if not is_transient(exc):
                raise
            await self._sleep(self._transient_backoff_seconds)
            response = await self._send(request)
        return from_anthropic_response(response)

    async def credentials_for_subprocess(self) -> dict[str, str]:
        """Return the Anthropic API key as the env var the SDK reads.

        Per the :class:`smai_core.plugins.LlmProvider` Protocol's
        substrate-dispatch contract: the orchestrator calls this
        per-dispatch and merges the result into the sandboxed
        agent-runtime container's env. The Anthropic SDK reads
        ``ANTHROPIC_API_KEY`` from the env if no explicit ``api_key``
        argument is passed to the client constructor — the same chain
        :func:`_build_anthropic_client` relies on at host-side
        construction.

        Resolution order: the live client's ``.api_key`` attribute (when
        set by the SDK from any source) takes precedence over the host's
        ``ANTHROPIC_API_KEY`` env var. Either source is acceptable;
        absence of both raises :class:`LlmProviderAuthError`.
        """
        api_key = cast("Any", getattr(self._client, "api_key", None))
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LlmProviderAuthError(
                "no Anthropic API key available on the host; "
                "set ANTHROPIC_API_KEY before dispatching a sandboxed agent role"
            )
        return {"ANTHROPIC_API_KEY": str(api_key)}

    # --- internal -----------------------------------------------------------

    def _build_request(
        self,
        *,
        system: str,
        messages: list[NormalizedMessage],
        tools: list[ToolDefinition] | None,
        max_tokens: int,
        temperature: float | None,
        cache_config: CacheConfig | None,
    ) -> dict[str, Any]:
        anthropic_messages = to_anthropic_messages(messages)
        anthropic_tools = to_anthropic_tools(tools) if tools else None

        # ``system`` starts as a plain string. We promote it to typed
        # blocks only when caching is requested — the Anthropic API
        # accepts both shapes, but an unmarked string is the simpler
        # baseline.
        system_blocks: list[dict[str, Any]] | None = None
        if cache_config is not None and self.capabilities.supports_caching:
            if cache_config.cache_static_prefix and system:
                system_blocks = [{"type": "text", "text": system}]
            apply_cache_control(
                system_blocks=system_blocks,
                anthropic_messages=anthropic_messages,
                tools=anthropic_tools,
                cache_config=cache_config,
                capabilities=self.capabilities,
            )

        request: dict[str, Any] = {
            "model": self._model_id,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
        }
        if system:
            request["system"] = system_blocks if system_blocks is not None else system
        if anthropic_tools is not None:
            request["tools"] = anthropic_tools
        if temperature is not None:
            request["temperature"] = temperature
        return request

    async def _send(self, request: dict[str, Any]) -> Any:
        try:
            messages_api = getattr(self._client, "messages", None)
            if messages_api is None:
                raise LlmProviderInvalidRequest(
                    "Anthropic client missing 'messages' attribute; "
                    "expected an AsyncAnthropic-shaped object"
                )
            create = getattr(messages_api, "create", None)
            if create is None:
                raise LlmProviderInvalidRequest("Anthropic client.messages missing 'create' method")
            return await create(**request)
        except LlmProviderError:
            raise
        except BaseException as exc:
            raise translate_sdk_error(exc) from exc

    # --- conformance fault-injection ---------------------------------------

    async def _conformance_inject_fault(
        self,
        kind: str,
        fn: Callable[..., Awaitable[ModelResponse]],
        **kwargs: Any,
    ) -> ModelResponse:
        """Stage a one-shot Anthropic-side outcome and invoke ``fn``.

        Settles the §4.7 synthesized-error contract for the conformance
        suite — same shape as the Bedrock plugin's settled pattern. The
        supported kinds (listed on
        :attr:`_conformance_supports_fault_injection`) map to
        Anthropic-shaped errors / canned responses via
        :func:`_build_outcomes`. Fault injection only works against an
        in-process :class:`FakeAnthropicClient`; passing a real SDK
        client raises :class:`RuntimeError`.
        """
        outcomes = _build_outcomes(kind)
        queue = self._client_queue()
        for outcome in outcomes:
            queue.append(outcome)
        return await fn(**kwargs)

    def _client_queue(self) -> deque[Any]:
        messages_api = getattr(self._client, "messages", None)
        queue = getattr(messages_api, "_conformance_queue", None)
        if not isinstance(queue, deque):
            raise RuntimeError(
                "AnthropicProvider fault injection requires a FakeAnthropicClient; "
                "pass `anthropic_client=FakeAnthropicClient(...)` to the constructor"
            )
        return cast("deque[Any]", queue)


# --- module-level helpers ---------------------------------------------------


def _build_anthropic_client(
    request_timeout_seconds: float = _DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> Any:
    """Construct a real :class:`anthropic.AsyncAnthropic` client.

    Lazily imported so the plugin module is importable in environments
    without the ``anthropic`` SDK installed (e.g., ``pyright`` on a CI
    runner that hasn't synced workspace dependencies yet).

    Per round-6 item C an explicit ``timeout`` is passed (the SDK
    default is ~10 minutes — far too long for an inline worker).
    """
    try:
        import anthropic  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover — declared dep
        raise LlmProviderError(
            "smai-llm-anthropic requires the `anthropic` SDK; install with "
            "`pip install smai-llm-anthropic`"
        ) from exc
    factory: Any = anthropic.AsyncAnthropic
    return factory(timeout=request_timeout_seconds)


def _build_outcomes(kind: str) -> list[Any]:
    """Return the queued outcomes for a fault-injection kind.

    Each outcome is either an :class:`Exception` (the fake raises it) or
    a :class:`dict` (the fake returns it from ``messages.create``). The
    retry-once contract is settled here: kinds that expect propagation
    queue *two* identical errors; ``transient_then_succeed`` queues an
    error followed by a canned success.
    """
    if kind == "rate_limit":
        return [_make_status_error(429, "Rate exceeded")] * 2
    if kind == "unavailable":
        return [_make_status_error(503, "Service unavailable")] * 2
    if kind == "invalid_request":
        return [_make_status_error(400, "Invalid request")]
    if kind == "auth":
        return [_make_status_error(401, "Not authorized")]
    if kind == "transient_then_succeed":
        return [
            _make_status_error(429, "Rate exceeded — once"),
            _DEFAULT_SUCCESS_RESPONSE,
        ]
    if kind == "tool_use_response":
        return [_TOOL_USE_RESPONSE]
    raise ValueError(f"unknown fault-injection kind: {kind!r}")


class _StandinAnthropicError(Exception):
    """Status-bearing exception used for fault injection.

    The conformance fakes use this rather than the real SDK exception
    classes so that the plugin's classification logic exercises the
    ``status_code`` path (the documented contract) rather than
    class-based dispatch. Real SDK exceptions also carry
    ``status_code`` — same translation behavior.
    """

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _make_status_error(status: int, message: str) -> Exception:
    return _StandinAnthropicError(status, message)


# Canned successful Message response — minimal shape that
# ``from_anthropic_response`` accepts.
_DEFAULT_SUCCESS_RESPONSE: dict[str, Any] = {
    "id": "msg_fixture",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 7,
        "output_tokens": 3,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    },
}


# Canned tool_use response per §4.7's tool_use_round_trip contract.
_TOOL_USE_RESPONSE: dict[str, Any] = {
    "id": "msg_fixture_tu",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "tool_use",
            "id": "tu_fixture",
            "name": "echo",
            "input": {"text": "hi"},
        }
    ],
    "stop_reason": "tool_use",
    "usage": {
        "input_tokens": 12,
        "output_tokens": 8,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    },
}


__all__ = ["AnthropicProvider"]
