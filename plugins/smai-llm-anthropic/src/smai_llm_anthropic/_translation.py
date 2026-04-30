"""Normalized ``<->`` Anthropic native-API translation.

Per ``07-plugin-interfaces.md`` §4.2. The Anthropic native shape is
nearly identical to the normalized shape (Bedrock Converse is itself a
hosted-Anthropic adapter, which is why the Bedrock plugin's translation
module is structurally similar). The differences:

* Anthropic content blocks use ``"type": "text"`` / ``"type": "tool_use"``
  / ``"type": "tool_result"`` discriminators (matching our normalized
  literals exactly).
* Tool-use IDs and tool-result IDs use ``id`` / ``tool_use_id`` (no
  ``toolUseId`` camelCase).
* Tool definitions are flat ``{"name": ..., "description": ...,
  "input_schema": ...}`` (no ``toolSpec`` wrapper).
* The Anthropic API takes ``system`` as a *string* (or list of typed
  blocks for cache markers). We send it as a string by default and
  promote it to a typed-block list when ``cache_static_prefix=True``.
* Caching is via ``cache_control`` markers (Anthropic's ``ephemeral``
  cache), not ``cachePoint`` blocks. Markers attach to the LAST item
  of the cached prefix per the Anthropic docs.

Anthropic request shape (subset relevant here)::

    {
      "model": "claude-opus-4-7",
      "system": "..." | [{"type": "text", "text": "...",
                          "cache_control": {"type": "ephemeral"}?}],
      "messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "..."},
            {"type": "tool_result", "tool_use_id": "...",
             "content": "...", "is_error": ...}
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "..."},
            {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
        ]}
      ],
      "tools": [{"name": "...", "description": "...",
                 "input_schema": {...},
                 "cache_control": {"type": "ephemeral"}?}],
      "max_tokens": ...,
      "temperature": ...?
    }

Response shape::

    {
      "id": "msg_...",
      "type": "message",
      "role": "assistant",
      "content": [{"type": "text", ...} | {"type": "tool_use", ...}],
      "stop_reason": "end_turn" | "tool_use" | "max_tokens" |
                     "stop_sequence",
      "usage": {"input_tokens": ..., "output_tokens": ...,
                "cache_read_input_tokens": ...,
                "cache_creation_input_tokens": ...}
    }
"""

from __future__ import annotations

from typing import Any, cast

from smai_core.plugins import (
    CacheConfig,
    LlmCapabilities,
    LlmProviderInvalidRequest,
    ModelResponse,
    NormalizedContent,
    NormalizedMessage,
    StopReason,
    TextContent,
    TokenUsage,
    ToolDefinition,
    ToolUseContent,
)

_CACHE_CONTROL_EPHEMERAL: dict[str, Any] = {"type": "ephemeral"}


def to_anthropic_messages(messages: list[NormalizedMessage]) -> list[dict[str, Any]]:
    """Translate normalized messages into Anthropic ``messages``.

    The output is a fresh list of fresh dicts — the input list and its
    pydantic models are not mutated, satisfying the
    "MUST NOT mutate the input messages list" contract on ``call``.
    """
    return [_to_anthropic_message(m) for m in messages]


def _to_anthropic_message(msg: NormalizedMessage) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for block in msg.content:
        if isinstance(block, TextContent):
            content.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolUseContent):
            content.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": dict(block.input),
                }
            )
        else:
            # ToolResultContent
            tool_result: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": block.tool_use_id,
                "content": block.content,
            }
            if block.is_error:
                tool_result["is_error"] = True
            content.append(tool_result)
    return {"role": msg.role, "content": content}


def to_anthropic_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Translate normalized tool definitions into Anthropic ``tools``."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": dict(t.input_schema),
        }
        for t in tools
    ]


def apply_cache_control(
    *,
    system_blocks: list[dict[str, Any]] | None,
    anthropic_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    cache_config: CacheConfig,
    capabilities: LlmCapabilities,
) -> list[dict[str, Any]] | None:
    """Insert ``cache_control`` markers in-place per §4.3 / ``04-agents.md`` §5.

    No-op when ``capabilities.supports_caching`` is False — returns the
    incoming ``system_blocks`` unchanged.

    Anthropic's cache-marker placement follows the docs' "mark the last
    item of the prefix you want to cache" rule. The mapping from the
    spec's :class:`CacheConfig` to Anthropic markers:

    * ``cache_static_prefix`` — mark the last tool definition (if any)
      AND the system prompt (promoted to a list of typed blocks if
      currently a string).
    * ``cache_initial_message`` — mark the last content block of the
      first user message.
    * ``rolling_cache_count`` — mark the last content block of the
      ``rolling_cache_count`` most-recent user messages, walking newest
      → oldest, skipping index 0 (owned by ``cache_initial_message``
      above).
    """
    if not capabilities.supports_caching:
        return system_blocks

    if cache_config.cache_static_prefix:
        if tools:
            tools[-1]["cache_control"] = _CACHE_CONTROL_EPHEMERAL
        if system_blocks:
            system_blocks[-1]["cache_control"] = _CACHE_CONTROL_EPHEMERAL

    if (
        cache_config.cache_initial_message
        and anthropic_messages
        and anthropic_messages[0].get("role") == "user"
    ):
        first_content = anthropic_messages[0].get("content")
        if isinstance(first_content, list) and first_content:
            last_block = cast("list[dict[str, Any]]", first_content)[-1]
            last_block["cache_control"] = _CACHE_CONTROL_EPHEMERAL

    rolling = cache_config.rolling_cache_count
    if rolling > 0:
        placed = 0
        for idx in range(len(anthropic_messages) - 1, 0, -1):
            if placed >= rolling:
                break
            msg = anthropic_messages[idx]
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, list) and content:
                last_block = cast("list[dict[str, Any]]", content)[-1]
                last_block["cache_control"] = _CACHE_CONTROL_EPHEMERAL
                placed += 1

    return system_blocks


def from_anthropic_response(response: Any) -> ModelResponse:
    """Translate an Anthropic ``Message`` response into a :class:`ModelResponse`.

    Accepts either a duck-typed Anthropic SDK ``Message`` object (with
    ``.content`` / ``.stop_reason`` / ``.usage`` attributes) or a dict
    with the same keys — the fake client used by the conformance tests
    queues canned dicts, while the real SDK returns SDK objects.
    """
    role = _get(response, "role", "assistant")
    if role != "assistant":
        raise LlmProviderInvalidRequest(
            f"Anthropic response role must be 'assistant', got {role!r}"
        )

    raw_content: Any = _get(response, "content", [])
    content_blocks: list[NormalizedContent] = []
    if isinstance(raw_content, list):
        for block in cast("list[Any]", raw_content):
            normalized = _from_anthropic_content(block)
            if normalized is not None:
                content_blocks.append(normalized)

    stop_reason = _normalize_stop_reason(_get(response, "stop_reason", None))
    usage = _from_anthropic_usage(_get(response, "usage", None))

    return ModelResponse(
        message=NormalizedMessage(role="assistant", content=content_blocks),
        stop_reason=stop_reason,
        usage=usage,
    )


def _from_anthropic_content(block: Any) -> NormalizedContent | None:
    btype = _get(block, "type", None)
    if btype == "text":
        text = _get(block, "text", "")
        if isinstance(text, str) and text:
            return TextContent(text=text)
        return None
    if btype == "tool_use":
        tool_use_id = str(_get(block, "id", ""))
        name = str(_get(block, "name", ""))
        raw_input: Any = _get(block, "input", {})
        input_dict: dict[str, Any] = (
            cast("dict[str, Any]", raw_input) if isinstance(raw_input, dict) else {}
        )
        return ToolUseContent(
            id=tool_use_id,
            name=name,
            input=dict(input_dict),
        )
    return None


def _normalize_stop_reason(raw: Any) -> StopReason:
    """Collapse provider-specific stop reasons per §4.2.

    Anthropic's ``stop_sequence`` collapses to ``end_turn``; ``tool_use``
    and ``max_tokens`` map directly.
    """
    if raw == "tool_use":
        return "tool_use"
    if raw == "max_tokens":
        return "max_tokens"
    return "end_turn"


def _from_anthropic_usage(raw: Any) -> TokenUsage:
    if raw is None:
        return TokenUsage(input_tokens=0, output_tokens=0)
    return TokenUsage(
        input_tokens=_int_field(raw, "input_tokens"),
        output_tokens=_int_field(raw, "output_tokens"),
        cache_read_tokens=_int_field(raw, "cache_read_input_tokens"),
        cache_write_tokens=_int_field(raw, "cache_creation_input_tokens"),
    )


def _int_field(raw: Any, key: str) -> int:
    val = _get(raw, key, 0)
    if isinstance(val, bool):
        return 0
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    return 0


def _get(obj: Any, key: str, default: Any) -> Any:
    """Read ``key`` from ``obj`` whether it is a dict or an SDK model.

    The Anthropic SDK returns Pydantic-derived ``Message`` objects whose
    fields are attributes; the in-process fake queues plain dicts. Both
    shapes round-trip through the same translation function.
    """
    if isinstance(obj, dict):
        return cast("dict[str, Any]", obj).get(key, default)
    val = getattr(obj, key, default)
    return val


__all__ = [
    "apply_cache_control",
    "from_anthropic_response",
    "to_anthropic_messages",
    "to_anthropic_tools",
]
