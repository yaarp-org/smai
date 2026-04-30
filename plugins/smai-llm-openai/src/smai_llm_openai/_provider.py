""":class:`OpenAIProvider` — :class:`LlmProvider` adapter for the OpenAI
Chat Completions API.

Per ``07-plugin-interfaces.md`` §4 (the full Protocol surface) and
``04-agents.md`` §5 (prompt-caching integration; OpenAI's automatic
caching is opaque to the caller — see ``_capabilities``). The Phase 3
plugin per Task 3.F5.

Construction reads ``OPENAI_API_KEY`` from the environment via the
``openai`` SDK's default chain — there is deliberately no ``api_key``
argument so credentials never enter shell history. ``OPENAI_BASE_URL``
is honored similarly (Azure OpenAI / proxy / self-hosted deployments).

Per-call:

1. Translate the normalized inputs into OpenAI chat-completions shape
   (see ``_translation``). The system prompt becomes a leading
   ``{"role": "system", ...}`` message; tool_use / tool_result blocks
   become flat ``tool_calls`` lists / role=tool messages respectively.
2. ``cache_config`` is ignored (``capabilities.supports_caching =
   False``) — the §4.3 contract permits silent ignore.
3. Invoke ``AsyncOpenAI.chat.completions.create`` — already async-native.
4. On transient errors (rate-limit / unavailable / connection), sleep
   30s and retry once. If the retry also fails, propagate.
5. Translate the success response back into :class:`ModelResponse`.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar, cast

from smai_core.plugins import (
    CacheConfig,
    LlmCapabilities,
    LlmProviderError,
    LlmProviderInvalidRequest,
    ModelResponse,
    NormalizedMessage,
    ToolDefinition,
)

from smai_llm_openai._capabilities import lookup_capabilities
from smai_llm_openai._errors import is_transient, translate_sdk_error
from smai_llm_openai._translation import (
    from_openai_response,
    to_openai_messages,
    to_openai_tools,
)

# Per §4.5: plugins MUST retry transient errors exactly once with a 30s
# backoff. The seam for fast-forward in tests is the ``sleep`` ctor arg.
_DEFAULT_TRANSIENT_BACKOFF_SECONDS = 30.0

# Sensible default model. Per-task model selection (DEC-022) typically
# overrides; this is the "you forgot to specify" fallback.
_DEFAULT_MODEL_ID = "gpt-4o"

# Type alias for the per-call async client. Duck-typed against any
# object exposing ``chat.completions.create(**kwargs) -> ChatCompletion``.
_OpenAIClient = Any


class OpenAIProvider:
    """OpenAI chat-completions adapter implementing :class:`LlmProvider`.

    Constructor::

        OpenAIProvider()  # default model + OPENAI_API_KEY from env
        OpenAIProvider(model_id="gpt-4o")
        OpenAIProvider(
            model_id="...",
            openai_client=fake_client,    # tests only
            capabilities=...,             # override the default lookup
            sleep=lambda s: None,         # tests only — bypass real backoff
        )
    """

    name: str = "openai"

    # Conformance fault-injection contract per the Bedrock-settled pattern.
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
        openai_client: _OpenAIClient | None = None,
        capabilities: LlmCapabilities | None = None,
        max_tokens_default: int = 4096,
        transient_backoff_seconds: float = _DEFAULT_TRANSIENT_BACKOFF_SECONDS,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._model_id = model_id
        self.capabilities = capabilities or lookup_capabilities(model_id)
        self._client: _OpenAIClient = openai_client or _build_openai_client()
        self._max_tokens_default = max_tokens_default
        self._transient_backoff_seconds = transient_backoff_seconds
        self._sleep = sleep or asyncio.sleep

    @property
    def model_id(self) -> str:
        """The OpenAI model ID this instance is configured for.

        Per-task model selection (DEC-022) is owned by the agent loop;
        the agent constructs one :class:`OpenAIProvider` per per-task
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
        # OpenAI does not support explicit caching markers; the §4.3
        # contract permits silent ignore.
        del cache_config

        request = self._build_request(
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        try:
            response = await self._send(request)
        except LlmProviderError as exc:
            if not is_transient(exc):
                raise
            await self._sleep(self._transient_backoff_seconds)
            response = await self._send(request)
        return from_openai_response(response)

    # --- internal -----------------------------------------------------------

    def _build_request(
        self,
        *,
        system: str,
        messages: list[NormalizedMessage],
        tools: list[ToolDefinition] | None,
        max_tokens: int,
        temperature: float | None,
    ) -> dict[str, Any]:
        openai_messages = to_openai_messages(system=system, messages=messages)
        openai_tools = to_openai_tools(tools) if tools else None

        request: dict[str, Any] = {
            "model": self._model_id,
            "messages": openai_messages,
            "max_tokens": max_tokens,
        }
        if openai_tools is not None:
            request["tools"] = openai_tools
        if temperature is not None:
            request["temperature"] = temperature
        return request

    async def _send(self, request: dict[str, Any]) -> Any:
        try:
            chat_api = getattr(self._client, "chat", None)
            if chat_api is None:
                raise LlmProviderInvalidRequest(
                    "OpenAI client missing 'chat' attribute; expected an AsyncOpenAI-shaped object"
                )
            completions = getattr(chat_api, "completions", None)
            if completions is None:
                raise LlmProviderInvalidRequest(
                    "OpenAI client.chat missing 'completions' attribute"
                )
            create = getattr(completions, "create", None)
            if create is None:
                raise LlmProviderInvalidRequest(
                    "OpenAI client.chat.completions missing 'create' method"
                )
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
        """Stage a one-shot OpenAI-side outcome and invoke ``fn``.

        Settles the §4.7 synthesized-error contract for the conformance
        suite — same shape as the Bedrock plugin's settled pattern.
        """
        outcomes = _build_outcomes(kind)
        queue = self._client_queue()
        for outcome in outcomes:
            queue.append(outcome)
        return await fn(**kwargs)

    def _client_queue(self) -> deque[Any]:
        chat = getattr(self._client, "chat", None)
        completions = getattr(chat, "completions", None)
        queue = getattr(completions, "_conformance_queue", None)
        if not isinstance(queue, deque):
            raise RuntimeError(
                "OpenAIProvider fault injection requires a FakeOpenAIClient; "
                "pass `openai_client=FakeOpenAIClient(...)` to the constructor"
            )
        return cast("deque[Any]", queue)


# --- module-level helpers ---------------------------------------------------


def _build_openai_client() -> Any:
    """Construct a real :class:`openai.AsyncOpenAI` client.

    Lazily imported so the plugin module is importable in environments
    without the ``openai`` SDK installed.
    """
    try:
        import openai  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover — declared dep
        raise LlmProviderError(
            "smai-llm-openai requires the `openai` SDK; install with `pip install smai-llm-openai`"
        ) from exc
    factory: Any = openai.AsyncOpenAI
    return factory()


def _build_outcomes(kind: str) -> list[Any]:
    """Return the queued outcomes for a fault-injection kind."""
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


class _StandinOpenAIError(Exception):
    """Status-bearing exception used for fault injection.

    Real OpenAI SDK exceptions also carry ``status_code`` — same
    translation behavior on the production path.
    """

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _make_status_error(status: int, message: str) -> Exception:
    return _StandinOpenAIError(status, message)


# Canned successful ChatCompletion response.
_DEFAULT_SUCCESS_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl_fixture",
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
    },
}


# Canned tool-call response per §4.7's tool_use_round_trip contract.
_TOOL_USE_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl_fixture_tu",
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_fixture",
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "arguments": '{"text": "hi"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "total_tokens": 20,
    },
}


__all__ = ["OpenAIProvider"]
