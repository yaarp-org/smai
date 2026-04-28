""":class:`LlmProvider` Protocol — single seam through which all agent roles
reach an LLM.

Per ``designs/smai/07-plugin-interfaces.md`` §4 and DEC-020 (model abstraction
layer carry-forward; v1's normalized conversation representation lifts
substantively into v2).

The Protocol is single-method — ``call(...)`` — by deliberate design (per
§4.3): the agent loop's between-turn deterministic logic (context truncation,
status writes, lint-on-write, validation polling, nudge injection) is engine
code, not plugin code; keeping ``call()`` as the only Protocol method
preserves the engine's ownership of the loop.

Provider-specific shapes (Bedrock's ``Message`` / ``ContentBlock``,
Anthropic's ``MessageParam``, OpenAI's ``ChatCompletionMessageParam``) are
translated into and out of the normalized types inside plugin code; agent
code never imports a provider SDK type directly.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

# === Normalized types (§4.2) =================================================


class ToolDefinition(BaseModel):
    """JSON-schema-shaped tool definition (§4.2)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    input_schema: dict[str, Any]


class TextContent(BaseModel):
    """Text content block (§4.2)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    text: str


class ToolUseContent(BaseModel):
    """Tool-use content block — model invokes a tool (§4.2)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ToolResultContent(BaseModel):
    """Tool-result content block — the agent loop returns the tool's output
    to the model (§4.2).

    ``content`` is ``str`` only in v1 (matching v1's narrowing — agent
    tools return plain text); a future widening to
    ``Union[str, list[NormalizedContent]]`` is non-breaking for existing
    call sites.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


NormalizedContent = TextContent | ToolUseContent | ToolResultContent


class NormalizedMessage(BaseModel):
    """Provider-agnostic message — one turn in the conversation (§4.2)."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: list[NormalizedContent]


class TokenUsage(BaseModel):
    """Per-call token accounting (§4.2 / §4.6).

    ``cache_read_tokens`` / ``cache_write_tokens`` are 0 when the model
    does not support caching or caching is disabled.
    """

    model_config = ConfigDict(extra="forbid")

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


StopReason = Literal["end_turn", "tool_use", "max_tokens"]


class ModelResponse(BaseModel):
    """A single model response (§4.2).

    ``message.role`` is always ``"assistant"``. ``stop_reason`` collapses
    provider-specific values that the loop does not differentiate
    (``stop_sequence``, ``content_filtered`` → ``end_turn``); same logic
    as v1.
    """

    model_config = ConfigDict(extra="forbid")

    message: NormalizedMessage
    stop_reason: StopReason
    usage: TokenUsage


# === Cache-config (§4.3) =====================================================


class CacheConfig(BaseModel):
    """Per-call cache-marker placement intent (§4.3).

    Per DEC-020, v1 settled :class:`CacheConfig` on the constructor. v2
    promotes it to per-call so the same plugin instance can serve
    different agent roles with different caching strategies. When
    :attr:`LlmCapabilities.supports_caching` is ``True``, the plugin
    translates the intent into provider-specific cache markers (Bedrock
    ``cachePoint``, Anthropic ``cache_control``, etc.) and manages
    between-turn cache-marker placement and cleanup within a single
    agent session. When ``False``, the field is ignored.
    """

    model_config = ConfigDict(extra="forbid")

    cache_static_prefix: bool = False
    cache_initial_message: bool = False
    rolling_cache_count: int = 0


# === Capability flags (§4.4) =================================================


class LlmCapabilities(BaseModel):
    """Static per-plugin capability flags (§4.4).

    Populated at instantiation; immutable for the lifetime of the plugin
    instance. Maps directly to v1's ``MODEL_REGISTRY`` entries
    (``packages/shared/src/model-registry.ts``); per-task model
    selection (v1's ``getModelForTask()``) becomes an
    orchestration / agent-config concern in v2.
    """

    model_config = ConfigDict(extra="forbid")

    supports_caching: bool
    context_window: int
    max_output_tokens: int
    supports_tool_use: bool = True
    model_id: str


# === Error contract (§4.5) ===================================================


class LlmProviderError(Exception):
    """Base class for all :class:`LlmProvider` errors (§4.5)."""


class LlmProviderRateLimited(LlmProviderError):
    """Throttling / rate-limit error from the provider (§4.5).

    On the first transient error, plugins MUST retry once with a 30s
    backoff. If the retry succeeds, the success result is returned to
    the caller; if the retry also raises a rate-limit error, that
    error is propagated to the caller.
    """


class LlmProviderUnavailable(LlmProviderError):
    """Service-unavailable / 503 error (§4.5).

    Same retry-once-with-30s-backoff pattern as
    :class:`LlmProviderRateLimited`: retry once, propagate on second
    failure.
    """


class LlmProviderInvalidRequest(LlmProviderError):
    """Validation error — bad request shape, exceeds context window,
    forbidden content, etc. (§4.5).

    Not retried. Surfaces immediately.
    """


class LlmProviderAuthError(LlmProviderError):
    """Auth failure — bad API key, missing IAM role, etc. (§4.5).

    Not retried.
    """


# === Protocol shape (§4.3) ===================================================


@runtime_checkable
class LlmProvider(Protocol):
    """Send messages to an LLM and get a completion, optionally with tool use.

    Plugins register via::

        [project.entry-points."smai.llm_providers"]
        <name> = "<module>:<class>"

    Implementations encapsulate:

    * SDK / API translation (``NormalizedMessage`` <-> provider format)
    * Retry on transient errors (throttling, 503) — see §4.5
    * Prompt-caching mechanics, when ``capabilities.supports_caching`` is True
    * Provider-specific auth (API key, IAM role, etc.) read from
      plugin-level config, not from method arguments
    """

    name: str
    """Canonical name (e.g., ``"bedrock"``, ``"anthropic"``, ``"openai"``).
    Matches the entry-point key. Must be unique across loaded plugins;
    collisions error at startup."""

    capabilities: LlmCapabilities
    """Static per-plugin capability flags. Populated at instantiation;
    immutable for the lifetime of the plugin instance. See §4.4."""

    async def call(
        self,
        system: str,
        messages: list[NormalizedMessage],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        cache_config: CacheConfig | None = None,
    ) -> ModelResponse:
        """Single-shot model call.

        Multi-turn loops are agent-loop concern — the agent loop calls
        ``.call()`` once per turn.

        Implementations MUST:

        * Translate messages and tools into the provider's native shape.
        * Respect ``cache_config`` when ``capabilities.supports_caching``
          is ``True``; ignore it (silently) otherwise.
        * On transient errors (rate limit, transient unavailability),
          retry once with a 30s backoff. If the retry succeeds, return
          the success result; if the retry also fails with a transient
          error, propagate the error to the caller. See §4.5.
        * Return a :class:`ModelResponse` with ``.usage`` populated.
          ``cache_*_tokens`` MUST be ``0`` when caching is unsupported
          or disabled.

        Implementations MAY:

        * Apply provider-specific request-shaping that doesn't change
          semantics (e.g., region selection, endpoint variant).

        Implementations MUST NOT:

        * Mutate the input ``messages`` list (the agent loop reuses it).
        * Apply context truncation (that is agent-loop concern).
        * Do retries beyond the documented transient pattern.
        """
        ...
