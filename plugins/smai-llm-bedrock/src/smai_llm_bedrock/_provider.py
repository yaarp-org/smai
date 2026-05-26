""":class:`BedrockProvider` — :class:`LlmProvider` adapter for AWS Bedrock.

Per ``07-plugin-interfaces.md`` §4 (the full Protocol surface),
``04-agents.md`` §5 (prompt-caching integration), DEC-005 (custom
Converse loop carry-forward) and DEC-019 (caching pattern). The Phase 2
reference plugin per the implementation_plan §3.3 Task 2.A1 brief.

Construction reads AWS credentials from the boto3 default chain
(``AWS_PROFILE``, ``AWS_REGION``, IAM role, env-var credentials) — there
is deliberately no ``api_key`` argument so credentials never enter shell
history. Per-call:

1. Translate the normalized inputs into Bedrock Converse request shape
   (see ``_translation``).
2. Apply ``cachePoint`` markers when ``capabilities.supports_caching``
   is True and ``cache_config`` was passed.
3. Invoke ``bedrock-runtime`` ``Converse``. boto3's client is
   synchronous; we run it in a worker thread via :func:`asyncio.to_thread`
   so ``call`` stays an ``async def`` per the Protocol.
4. On transient errors (rate-limit / unavailable), sleep 30s and retry
   once. If the retry also fails, propagate the canonical error.
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
    LlmProviderAuthError,
    LlmProviderError,
    LlmProviderInvalidRequest,
    LlmProviderUnavailable,
    ModelResponse,
    NormalizedMessage,
    ToolDefinition,
)

from smai_llm_bedrock._capabilities import lookup_capabilities
from smai_llm_bedrock._errors import is_transient, translate_client_error
from smai_llm_bedrock._translation import (
    apply_cache_points,
    from_converse_response,
    to_converse_messages,
    to_converse_tool_config,
)

# Documented retry-once backoff per §4.5. Plugins are required to retry
# transient errors exactly once with a 30s backoff. The conformance
# suite's ``test_transient_retry_once`` is wall-clock-sensitive but
# accepts a broad band; production runs the literal 30s. The seam for
# Task 2.C1's fake-clock injection is the :func:`_sleep` callable
# below — overridable via the ``_sleep`` constructor arg for tests.
_DEFAULT_TRANSIENT_BACKOFF_SECONDS = 30.0

# Round-6 item C — client-level timeout / retry ceiling. A bare
# ``boto3.client("bedrock-runtime")`` has *no* connect/read timeout, so
# a stalled call hangs forever; because the planner / supervisor run
# inline in the worker, that wedges the worker. We construct the client
# with an explicit :class:`botocore.config.Config` (connect/read
# timeouts + bounded internal retries) AND wrap the ``to_thread`` call
# in :func:`asyncio.wait_for` with a hard ceiling so the ``await``
# always unblocks. (Cancelling a ``to_thread`` does NOT kill the
# underlying boto3 call — the worker thread keeps running until the
# call returns — but it unblocks the loop so a timeout surfaces as a
# retryable :class:`LlmProviderUnavailable`; the orphan thread dies
# when the call eventually returns.)
_DEFAULT_CONNECT_TIMEOUT_SECONDS = 10.0
# 90s comfortably covers a full ~4k-token generation; capping internal
# retries at 2 keeps the hard ceiling at 90*(2+1)+30 = 300s and the
# retry-once worst case at ~2*300 + backoff ≈ 10min, rather than the
# ~21min the old 120s/4-retry defaults allowed. All config-overridable.
_DEFAULT_READ_TIMEOUT_SECONDS = 90.0
_DEFAULT_MAX_RETRIES = 2
_HARD_CEILING_SLACK_SECONDS = 30.0

# Default request shape echoed back when no real call is made (used
# by the in-process FakeBedrockClient in tests). Never reached at
# runtime — the production path always goes through the boto3 client.

# Type alias for the per-call sync Bedrock invocation. The plugin works
# duck-typed against any object that exposes ``.converse(**kwargs) ->
# dict[str, Any]`` — production passes a boto3
# ``bedrock-runtime`` client; tests pass a ``FakeBedrockClient``.
_BedrockClient = Any  # boto3 clients are dynamically generated; no static type


class BedrockProvider:
    """Bedrock Converse adapter implementing :class:`LlmProvider`.

    Constructor::

        BedrockProvider(region="us-east-1", model_id="us.anthropic.claude-opus-4-6-v1")
        BedrockProvider(
            region="us-east-1",
            model_id="...",
            bedrock_client=fake_client,  # tests only
            capabilities=...,             # override the default lookup
            sleep=lambda s: None,         # tests only — bypass real backoff
        )
    """

    # Per :class:`smai_core.plugins.LlmProvider` Protocol §4.3, ``name`` is
    # an instance attribute. The constant assignment provides the
    # canonical "bedrock" value at the class level — Python treats this
    # as a class-level default that is also visible as ``self.name`` on
    # every instance. ``ClassVar`` would tell the type checker the
    # attribute is class-only and is incompatible with the Protocol.
    # The ``: str`` annotation matches the form used by sibling
    # plugins (smai-store-sqlite, smai-artifacts-localfs,
    # smai-compute-localgpu).
    name: str = "bedrock"

    # --- conformance fault-injection contract per Task 1.9 -----------------
    #
    # The conformance suite's ``_inject_and_call`` calls
    # ``provider._conformance_inject_fault(kind, fn, **kwargs)``. The
    # provider is responsible for staging a one-shot Bedrock-side outcome
    # (a synthesized exception or a synthesized response) and then
    # delegating to ``fn`` (which is ``provider.call``). The conformance
    # suite reads the resulting :class:`LlmProviderError` subclass / the
    # :class:`ModelResponse` to confirm error-class translation.
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
        region: str,
        model_id: str,
        *,
        bedrock_client: _BedrockClient | None = None,
        capabilities: LlmCapabilities | None = None,
        transient_backoff_seconds: float = _DEFAULT_TRANSIENT_BACKOFF_SECONDS,
        connect_timeout_seconds: float = _DEFAULT_CONNECT_TIMEOUT_SECONDS,
        read_timeout_seconds: float = _DEFAULT_READ_TIMEOUT_SECONDS,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._region = region
        self._model_id = model_id
        self.capabilities = capabilities or lookup_capabilities(model_id)
        self._client: _BedrockClient = bedrock_client or _build_bedrock_client(
            region,
            connect_timeout_seconds=connect_timeout_seconds,
            read_timeout_seconds=read_timeout_seconds,
            max_retries=max_retries,
        )
        self._transient_backoff_seconds = transient_backoff_seconds
        # Hard ceiling for the per-call ``await``: every internal botocore
        # retry costs up to ``read_timeout`` seconds, plus slack.
        self._call_timeout_seconds = read_timeout_seconds * (max_retries + 1) + (
            _HARD_CEILING_SLACK_SECONDS
        )
        self._sleep = sleep or asyncio.sleep

    @property
    def model_id(self) -> str:
        """The Bedrock model ID this instance is configured for.

        Per-task model selection (v1's ``getModelForTask``) is owned by
        Task 2.B1's agent loop; the agent constructs one
        :class:`BedrockProvider` per per-task model_id and routes
        accordingly. Exposed read-only here so the agent can verify
        the routing it built.
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
        request = self._build_converse_request(
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
        return from_converse_response(response)

    async def credentials_for_subprocess(self) -> dict[str, str]:
        """Resolve AWS credentials via the boto3 default chain and return
        them as env vars suitable for projecting into a sandbox container.

        Per the :class:`smai_core.plugins.LlmProvider` Protocol's
        substrate-dispatch contract: the orchestrator calls this
        per-dispatch and merges the result into the agent-runtime
        container's env so PydanticAI's Bedrock client inside the sandbox
        can authenticate. Resolution happens via a fresh ``boto3.Session``
        rather than the cached ``self._client`` because the client may be
        a fake (conformance tests inject one) and because boto3 chains
        SSO / profile / instance-metadata to a frozen access-key pair
        only through ``session.get_credentials().get_frozen_credentials()``.

        Raises :class:`LlmProviderAuthError` if the chain returns no
        credentials.
        """
        try:
            import boto3  # noqa: PLC0415  # pyright: ignore[reportMissingTypeStubs]
        except ImportError as exc:  # pragma: no cover — declared dep
            raise LlmProviderError(
                "smai-llm-bedrock requires boto3; install with `pip install smai-llm-bedrock`"
            ) from exc
        session = cast("Any", boto3).Session(region_name=self._region)
        creds = await asyncio.to_thread(session.get_credentials)
        if creds is None:
            raise LlmProviderAuthError(
                "no AWS credentials resolved via the boto3 default chain on the host; "
                "configure ~/.aws/credentials, AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY, "
                "or an SSO / IAM-role-based profile before dispatching a sandboxed agent role"
            )
        frozen = await asyncio.to_thread(creds.get_frozen_credentials)
        env: dict[str, str] = {
            "AWS_REGION": self._region,
            "AWS_DEFAULT_REGION": self._region,
            "AWS_ACCESS_KEY_ID": frozen.access_key,
            "AWS_SECRET_ACCESS_KEY": frozen.secret_key,
        }
        if frozen.token:
            env["AWS_SESSION_TOKEN"] = frozen.token
        return env

    # --- internal -----------------------------------------------------------

    def _build_converse_request(
        self,
        *,
        system: str,
        messages: list[NormalizedMessage],
        tools: list[ToolDefinition] | None,
        max_tokens: int,
        temperature: float | None,
        cache_config: CacheConfig | None,
    ) -> dict[str, Any]:
        converse_messages = to_converse_messages(messages)
        system_blocks: list[dict[str, Any]] = [{"text": system}]
        tool_config = to_converse_tool_config(tools) if tools else None

        if cache_config is not None:
            apply_cache_points(
                system_blocks=system_blocks,
                converse_messages=converse_messages,
                tool_config=tool_config,
                cache_config=cache_config,
                capabilities=self.capabilities,
            )

        inference_config: dict[str, Any] = {"maxTokens": max_tokens}
        if temperature is not None:
            inference_config["temperature"] = temperature

        request: dict[str, Any] = {
            "modelId": self._model_id,
            "system": system_blocks,
            "messages": converse_messages,
            "inferenceConfig": inference_config,
        }
        if tool_config is not None:
            request["toolConfig"] = tool_config
        return request

    async def _send(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            # Hard ceiling on the ``await`` — cancelling the ``to_thread``
            # does not kill the boto3 call (the orphan thread runs until
            # the call returns), but it unblocks the loop so a wedged
            # call surfaces as a retryable error rather than hanging the
            # inline worker forever (round-6 item C).
            raw_response = await asyncio.wait_for(
                asyncio.to_thread(self._client.converse, **request),
                timeout=self._call_timeout_seconds,
            )
        except TimeoutError as exc:
            raise LlmProviderUnavailable(
                f"Bedrock Converse exceeded the {self._call_timeout_seconds:.0f}s "
                f"client-call ceiling; treating as transient"
            ) from exc
        except Exception as exc:
            raise self._classify_exception(exc) from exc
        if not isinstance(raw_response, dict):
            raise LlmProviderInvalidRequest(
                f"Bedrock Converse returned non-dict response: {type(raw_response).__name__}"
            )
        return cast(dict[str, Any], raw_response)

    def _classify_exception(self, exc: BaseException) -> LlmProviderError:
        # botocore is an optional runtime dependency — we want the
        # plugin to import even before boto3 is installed (it'll fail
        # at construct-time if the user actually tries to talk to
        # Bedrock). Importing here keeps module-load-time clean.
        client_error_cls, no_creds_cls, endpoint_err_cls = _load_botocore_exceptions()

        if isinstance(exc, no_creds_cls):
            out_auth = LlmProviderAuthError(str(exc))
            out_auth.__cause__ = exc
            return out_auth
        if isinstance(exc, endpoint_err_cls):
            out_unavail = LlmProviderUnavailable(str(exc))
            out_unavail.__cause__ = exc
            return out_unavail
        if isinstance(exc, client_error_cls):
            return translate_client_error(cast(Any, exc))
        if isinstance(exc, LlmProviderError):
            return exc
        # Unknown exception class — wrap as base so the caller still
        # sees an :class:`LlmProviderError` per §4.5.
        out = LlmProviderError(f"unexpected Bedrock error: {exc!r}")
        out.__cause__ = exc
        return out

    # --- conformance fault-injection ---------------------------------------

    async def _conformance_inject_fault(
        self,
        kind: str,
        fn: Callable[..., Awaitable[ModelResponse]],
        **kwargs: Any,
    ) -> ModelResponse:
        """Stage a one-shot Bedrock-side outcome and invoke ``fn``.

        Settles the §4.7 synthesized-error contract for the
        :class:`LlmProvider` conformance suite. The supported kinds —
        listed on :attr:`_conformance_supports_fault_injection` — map to
        Bedrock-specific ``ClientError`` instances or canned response
        dicts via :func:`_build_outcomes`. The fault-injection only
        works against an in-process :class:`FakeBedrockClient`
        (constructed via ``BedrockProvider(..., bedrock_client=fake)``);
        passing a real boto3 client raises :class:`RuntimeError`.

        Per the carry-forward note from Task 1.9: this method **settles**
        the synthesis pattern. Future plugin tasks (anthropic / openai)
        replicate the same shape — one ``_build_outcomes`` table, one
        provider-specific fake client.
        """
        outcomes = _build_outcomes(kind)
        queue = self._client_queue()
        for outcome in outcomes:
            queue.append(outcome)
        return await fn(**kwargs)

    def _client_queue(self) -> deque[Any]:
        queue = getattr(self._client, "_conformance_queue", None)
        if not isinstance(queue, deque):
            raise RuntimeError(
                "BedrockProvider fault injection requires a FakeBedrockClient; "
                "pass `bedrock_client=FakeBedrockClient(...)` to the constructor"
            )
        return cast("deque[Any]", queue)


# --- module-level helpers ---------------------------------------------------


def _build_bedrock_client(
    region: str,
    *,
    connect_timeout_seconds: float = _DEFAULT_CONNECT_TIMEOUT_SECONDS,
    read_timeout_seconds: float = _DEFAULT_READ_TIMEOUT_SECONDS,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> Any:
    """Construct a real ``bedrock-runtime`` boto3 client.

    Lazily imported so the plugin module is importable in environments
    without boto3 installed (e.g., ``pyright`` on a CI runner that hasn't
    synced workspace dependencies yet).

    Per round-6 item C the client carries an explicit
    :class:`botocore.config.Config` — a bare ``boto3.client(...)`` has no
    connect/read timeout, so a stalled call would hang the inline worker
    forever.
    """
    try:
        import boto3  # noqa: PLC0415  # pyright: ignore[reportMissingTypeStubs]
        import botocore.config  # noqa: PLC0415  # pyright: ignore[reportMissingTypeStubs]
    except ImportError as exc:  # pragma: no cover — declared dep
        raise LlmProviderError(
            "smai-llm-bedrock requires boto3; install with `pip install smai-llm-bedrock`"
        ) from exc
    boto_config = cast("Any", botocore.config.Config)(
        connect_timeout=connect_timeout_seconds,
        read_timeout=read_timeout_seconds,
        retries={"max_attempts": max_retries, "mode": "standard"},
    )
    factory = cast("Any", boto3).client
    client: Any = factory("bedrock-runtime", region_name=region, config=boto_config)
    return client


def _build_outcomes(kind: str) -> list[Any]:
    """Return the queued outcomes for a fault-injection kind.

    Each outcome is either a :class:`Exception` (the fake client raises
    it) or a :class:`dict` (the fake client returns it from
    ``.converse``). The retry-once contract is settled here: kinds that
    expect propagation queue *two* identical errors (so the retry also
    raises); ``transient_then_succeed`` queues an error followed by a
    canned success.
    """
    if kind == "rate_limit":
        return [_make_client_error("ThrottlingException", "Rate exceeded")] * 2
    if kind == "unavailable":
        return [_make_client_error("ServiceUnavailableException", "Service unavailable")] * 2
    if kind == "invalid_request":
        return [_make_client_error("ValidationException", "Invalid request")]
    if kind == "auth":
        return [_make_client_error("AccessDeniedException", "Not authorized")]
    if kind == "transient_then_succeed":
        return [
            _make_client_error("ThrottlingException", "Rate exceeded — once"),
            _DEFAULT_SUCCESS_RESPONSE,
        ]
    if kind == "tool_use_response":
        return [_TOOL_USE_RESPONSE]
    raise ValueError(f"unknown fault-injection kind: {kind!r}")


def _make_client_error(code: str, message: str) -> Exception:
    """Build a botocore :class:`ClientError` with the given AWS error code."""
    client_error_cls, _, _ = _load_botocore_exceptions()
    if client_error_cls is _StandinClientError:
        return _StandinClientError(code, message)
    # botocore.ClientError takes ``error_response`` + ``operation_name``;
    # the cast lifts the BaseException type back to the dynamically-imported
    # subclass so the kwargs are accepted at the call site.
    return cast("Any", client_error_cls)(
        error_response={"Error": {"Code": code, "Message": message}},
        operation_name="Converse",
    )


class _StandinClientError(Exception):
    """Fallback used only when botocore is not importable.

    Duck-typed against the same ``response['Error']['Code']`` shape
    :func:`translate_client_error` reads. Never instantiated in
    production — boto3 is a declared runtime dep — but kept so the
    plugin module imports cleanly even on a pre-``uv sync`` environment.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.response: dict[str, Any] = {"Error": {"Code": code, "Message": message}}


class _StandinPlaceholder(Exception):
    """Never-instantiated placeholder used only when botocore is unavailable."""


def _load_botocore_exceptions() -> tuple[
    type[BaseException], type[BaseException], type[BaseException]
]:
    """Return ``(ClientError, NoCredentialsError, EndpointConnectionError)``.

    Falls back to local placeholder classes when botocore is not
    installed (purely defensive — boto3 is a declared runtime dep).
    The fallback keeps module-import clean on a pre-``uv sync`` env.
    """
    try:
        from botocore.exceptions import (  # noqa: PLC0415  # pyright: ignore[reportMissingTypeStubs]
            ClientError,
            EndpointConnectionError,
            NoCredentialsError,
        )
    except ImportError:  # pragma: no cover — declared dep
        return _StandinClientError, _StandinPlaceholder, _StandinPlaceholder
    return (
        cast("type[BaseException]", ClientError),
        cast("type[BaseException]", NoCredentialsError),
        cast("type[BaseException]", EndpointConnectionError),
    )


# Canned successful Converse response — minimal shape that
# ``from_converse_response`` accepts.
_DEFAULT_SUCCESS_RESPONSE: dict[str, Any] = {
    "output": {
        "message": {
            "role": "assistant",
            "content": [{"text": "ok"}],
        }
    },
    "stopReason": "end_turn",
    "usage": {
        "inputTokens": 7,
        "outputTokens": 3,
        "cacheReadInputTokens": 0,
        "cacheWriteInputTokens": 0,
    },
}


# Canned tool_use response per §4.7's tool_use_round_trip contract.
_TOOL_USE_RESPONSE: dict[str, Any] = {
    "output": {
        "message": {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "tu_fixture",
                        "name": "echo",
                        "input": {"text": "hi"},
                    }
                }
            ],
        }
    },
    "stopReason": "tool_use",
    "usage": {
        "inputTokens": 12,
        "outputTokens": 8,
        "cacheReadInputTokens": 0,
        "cacheWriteInputTokens": 0,
    },
}


__all__ = ["BedrockProvider"]
