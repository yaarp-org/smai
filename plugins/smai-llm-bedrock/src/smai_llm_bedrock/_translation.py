"""Normalized ``<->`` Bedrock Converse translation.

Per ``07-plugin-interfaces.md`` §4.2 (the normalized types are the contract;
provider-specific shapes — Bedrock's ``Message`` / ``ContentBlock`` —
stay inside plugin code, never leak to agent code) and DEC-020.

Bedrock Converse request shape (subset relevant here)::

    {
      "modelId": "us.anthropic.claude-opus-4-7-v1",
      "system": [{"text": "..."}, {"cachePoint": {"type": "default"}}?],
      "messages": [
        {"role": "user", "content": [
            {"text": "..."},
            {"toolResult": {"toolUseId": "...", "content": [...], "status": "..."}},
            {"cachePoint": {"type": "default"}}?
        ]},
        {"role": "assistant", "content": [
            {"text": "..."},
            {"toolUse": {"toolUseId": "...", "name": "...", "input": {...}}}
        ]}
      ],
      "toolConfig": {"tools": [{"toolSpec": {...}}, {"cachePoint": ...}?]}?,
      "inferenceConfig": {"maxTokens": ..., "temperature": ...?}
    }

Response shape::

    {
      "output": {"message": {"role": "assistant", "content": [...]}},
      "stopReason": "end_turn" | "tool_use" | "max_tokens" | ...,
      "usage": {"inputTokens": ..., "outputTokens": ...,
                "cacheReadInputTokens": ..., "cacheWriteInputTokens": ...}
    }

The translation table mirrors the v1 ``packages/shared/src/model-client.ts``
implementation (per DEC-020 carry-forward).
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


def to_converse_messages(messages: list[NormalizedMessage]) -> list[dict[str, Any]]:
    """Translate normalized messages into Bedrock Converse ``messages``.

    The output is a fresh list of fresh dicts — the input list and its
    pydantic models are not mutated, satisfying the
    "MUST NOT mutate the input messages list" contract on ``call``.
    """
    return [_to_converse_message(m) for m in messages]


def _to_converse_message(msg: NormalizedMessage) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for block in msg.content:
        if isinstance(block, TextContent):
            content.append({"text": block.text})
        elif isinstance(block, ToolUseContent):
            content.append(
                {
                    "toolUse": {
                        "toolUseId": block.id,
                        "name": block.name,
                        "input": dict(block.input),
                    }
                }
            )
        else:
            # ToolResultContent
            content.append(
                {
                    "toolResult": {
                        "toolUseId": block.tool_use_id,
                        "content": [{"text": block.content}],
                        "status": "error" if block.is_error else "success",
                    }
                }
            )
    return {"role": msg.role, "content": content}


def to_converse_tool_config(tools: list[ToolDefinition]) -> dict[str, Any]:
    """Translate normalized tool definitions into Bedrock ``toolConfig``."""
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": {"json": dict(t.input_schema)},
                }
            }
            for t in tools
        ]
    }


def apply_cache_points(
    *,
    system_blocks: list[dict[str, Any]],
    converse_messages: list[dict[str, Any]],
    tool_config: dict[str, Any] | None,
    cache_config: CacheConfig,
    capabilities: LlmCapabilities,
) -> None:
    """Insert ``cachePoint`` markers in-place per §4.3 / ``04-agents.md`` §5.

    No-op when ``capabilities.supports_caching`` is False. Per the
    Protocol contract: "Respect ``cache_config`` when
    ``capabilities.supports_caching`` is True; ignore it (silently)
    otherwise."

    The placement matches v1's ``ConverseClient.applyCacheHints`` plus
    the v2 expansion to also cache the system prompt under
    ``cache_static_prefix`` (per the spec language in §4.3:
    "tool definitions and system prompt"; v1 only cached tools).
    """
    if not capabilities.supports_caching:
        return

    if cache_config.cache_static_prefix:
        if tool_config is not None:
            tools_list = tool_config.get("tools")
            if isinstance(tools_list, list):
                cast("list[Any]", tools_list).append({"cachePoint": {"type": "default"}})
        if system_blocks:
            system_blocks.append({"cachePoint": {"type": "default"}})

    if (
        cache_config.cache_initial_message
        and converse_messages
        and converse_messages[0].get("role") == "user"
    ):
        first_content = converse_messages[0].get("content")
        if isinstance(first_content, list):
            cast("list[Any]", first_content).append({"cachePoint": {"type": "default"}})

    rolling = cache_config.rolling_cache_count
    if rolling > 0:
        placed = 0
        # Walk newest → oldest, skipping index 0 which is owned by
        # cache_initial_message above.
        for idx in range(len(converse_messages) - 1, 0, -1):
            if placed >= rolling:
                break
            msg = converse_messages[idx]
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                cast("list[Any]", content).append({"cachePoint": {"type": "default"}})
                placed += 1


def from_converse_response(response: dict[str, Any]) -> ModelResponse:
    """Translate a Bedrock Converse response into a :class:`ModelResponse`.

    Raises :class:`LlmProviderInvalidRequest` if the response shape is
    fundamentally unparseable (no ``output.message``) — this should never
    happen for a successful HTTP 200, but a guard prevents a downstream
    ``KeyError`` from masquerading as an :class:`LlmProviderError`.
    """
    output_raw: Any = response.get("output")
    if not isinstance(output_raw, dict):
        raise LlmProviderInvalidRequest(
            f"Bedrock Converse response missing output.message; raw={response!r}"
        )
    output = cast("dict[str, Any]", output_raw)
    message_raw: Any = output.get("message")
    if not isinstance(message_raw, dict):
        raise LlmProviderInvalidRequest(
            f"Bedrock Converse response missing output.message; raw={response!r}"
        )
    message = cast("dict[str, Any]", message_raw)
    role = message.get("role")
    if role != "assistant":
        raise LlmProviderInvalidRequest(
            f"Bedrock Converse response role must be 'assistant', got {role!r}"
        )
    raw_content: Any = message.get("content")
    content_blocks: list[NormalizedContent] = []
    if isinstance(raw_content, list):
        for block in cast("list[Any]", raw_content):
            if not isinstance(block, dict):
                continue
            normalized = _from_converse_content(cast("dict[str, Any]", block))
            if normalized is not None:
                content_blocks.append(normalized)

    stop_reason = _normalize_stop_reason(response.get("stopReason"))
    usage = _from_converse_usage(response.get("usage"))

    return ModelResponse(
        message=NormalizedMessage(role="assistant", content=content_blocks),
        stop_reason=stop_reason,
        usage=usage,
    )


def _from_converse_content(block: dict[str, Any]) -> NormalizedContent | None:
    text_val: Any = block.get("text")
    if isinstance(text_val, str) and text_val:
        return TextContent(text=text_val)
    tool_use_raw: Any = block.get("toolUse")
    if isinstance(tool_use_raw, dict):
        tu = cast("dict[str, Any]", tool_use_raw)
        tool_use_id: Any = tu.get("toolUseId", "")
        name: Any = tu.get("name", "")
        raw_input: Any = tu.get("input", {})
        input_dict: dict[str, Any] = (
            cast("dict[str, Any]", raw_input) if isinstance(raw_input, dict) else {}
        )
        return ToolUseContent(
            id=str(tool_use_id),
            name=str(name),
            input=dict(input_dict),
        )
    # Skip ``cachePoint`` and other non-content blocks (parity with v1).
    return None


def _normalize_stop_reason(raw: Any) -> StopReason:
    """Collapse provider-specific stop reasons per §4.2.

    ``stop_sequence``, ``content_filtered``, ``guardrail_intervened``,
    etc. all collapse to ``end_turn`` (same as v1).
    """
    if raw == "tool_use":
        return "tool_use"
    if raw == "max_tokens":
        return "max_tokens"
    return "end_turn"


def _from_converse_usage(raw: Any) -> TokenUsage:
    if not isinstance(raw, dict):
        return TokenUsage(input_tokens=0, output_tokens=0)
    raw_dict = cast("dict[str, Any]", raw)
    return TokenUsage(
        input_tokens=_int_field(raw_dict, "inputTokens"),
        output_tokens=_int_field(raw_dict, "outputTokens"),
        cache_read_tokens=_int_field(raw_dict, "cacheReadInputTokens"),
        cache_write_tokens=_int_field(raw_dict, "cacheWriteInputTokens"),
    )


def _int_field(raw: dict[str, Any], key: str) -> int:
    val = raw.get(key, 0)
    if isinstance(val, bool):  # bool is a subclass of int — exclude
        return 0
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    return 0


__all__ = [
    "apply_cache_points",
    "from_converse_response",
    "to_converse_messages",
    "to_converse_tool_config",
]
