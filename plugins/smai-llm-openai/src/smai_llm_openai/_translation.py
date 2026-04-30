"""Normalized ``<->`` OpenAI chat-completions translation.

Per ``07-plugin-interfaces.md`` §4.2. The OpenAI chat-completions API
has a *different* shape than Anthropic / Bedrock-Converse — the
translation here is more involved than for those providers:

* **System prompt** is a `system`-role message in the `messages` array
  (not a separate top-level field).
* **Tool calls** appear as a flat ``tool_calls`` list on the assistant
  message, separate from ``content`` (which carries the text reply).
  Anthropic / Bedrock interleave text and tool_use blocks within
  ``content`` — OpenAI does not.
* **Tool results** are role=``"tool"`` messages keyed by
  ``tool_call_id`` (not user messages with tool_result content blocks).
* **Tool-call arguments** are *JSON-encoded strings* on the wire, not
  inline dicts — we ``json.loads`` on the way in and ``json.dumps`` on
  the way out.
* **Tool definitions** wrap in ``{"type": "function", "function": {...}}``;
  the ``parameters`` field carries the JSON schema (Anthropic uses
  ``input_schema``).
* **Stop reasons** are ``finish_reason`` on the choice (``stop`` /
  ``tool_calls`` / ``length`` / ``content_filter``); we collapse to
  the normalized ``StopReason`` literal.
* **Usage** uses ``prompt_tokens`` / ``completion_tokens`` (not
  ``input_tokens`` / ``output_tokens``); cache fields surface under
  ``prompt_tokens_details.cached_tokens`` (newer SDK versions) — we
  surface them as ``cache_read_tokens`` and leave ``cache_write_tokens``
  at 0 (OpenAI's automatic caching is read-only from the caller's POV).

A NormalizedMessage that mixes text + tool_use blocks (Anthropic-shape)
must be flattened: text becomes the ``content`` string; tool_use
becomes one entry per ``tool_calls`` list element.

A NormalizedMessage with role=user containing ``tool_result`` blocks
must be *split* into multiple OpenAI messages: one role=tool message
per tool_result, plus an optional role=user message for any leftover
text. We preserve ordering by emitting the role=tool messages in
sequence.

OpenAI request shape (subset relevant here)::

    {
      "model": "gpt-4o",
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..." | None,
         "tool_calls": [
            {"id": "call_X", "type": "function",
             "function": {"name": "echo",
                          "arguments": "{\"text\": \"hi\"}"}}
         ]},
        {"role": "tool", "tool_call_id": "call_X", "content": "..."},
      ],
      "tools": [{"type": "function",
                 "function": {"name": "echo", "description": "...",
                              "parameters": {...}}}],
      "max_tokens": ...,
      "temperature": ...?
    }

Response shape::

    {
      "id": "chatcmpl_...",
      "model": "...",
      "choices": [
        {
          "index": 0,
          "message": {"role": "assistant",
                      "content": "..." | None,
                      "tool_calls": [...]?},
          "finish_reason": "stop" | "tool_calls" | "length" |
                           "content_filter"
        }
      ],
      "usage": {"prompt_tokens": ..., "completion_tokens": ...,
                "total_tokens": ...,
                "prompt_tokens_details": {"cached_tokens": ...}?}
    }
"""

from __future__ import annotations

import json
from typing import Any, cast

from smai_core.plugins import (
    LlmProviderInvalidRequest,
    ModelResponse,
    NormalizedContent,
    NormalizedMessage,
    StopReason,
    TextContent,
    TokenUsage,
    ToolDefinition,
    ToolResultContent,
    ToolUseContent,
)


def to_openai_messages(*, system: str, messages: list[NormalizedMessage]) -> list[dict[str, Any]]:
    """Translate ``(system, messages)`` into OpenAI's flat ``messages``.

    The system prompt becomes the leading ``{"role": "system",
    "content": <str>}`` entry (omitted if ``system`` is empty). User /
    assistant / tool-result messages translate per
    :func:`_to_openai_messages_for_normalized`. Output is a fresh list
    of fresh dicts — the input ``messages`` list and its pydantic
    models are not mutated, satisfying the
    "MUST NOT mutate the input messages list" contract on ``call``.
    """
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for msg in messages:
        out.extend(_to_openai_messages_for_normalized(msg))
    return out


def _to_openai_messages_for_normalized(msg: NormalizedMessage) -> list[dict[str, Any]]:
    """Translate one normalized message into one or more OpenAI messages.

    A user/assistant message with mixed text + tool_use blocks becomes
    a single OpenAI message (text → ``content``, tool_use →
    ``tool_calls``). A user message with tool_result blocks becomes a
    sequence of role=tool messages (one per result), with any remaining
    text content emitted as a leading role=user message.
    """
    if msg.role == "assistant":
        return [_assistant_to_openai(msg)]

    # role == "user"
    text_blocks: list[TextContent] = []
    tool_results: list[ToolResultContent] = []
    has_other = False
    for block in msg.content:
        if isinstance(block, TextContent):
            text_blocks.append(block)
        elif isinstance(block, ToolResultContent):
            tool_results.append(block)
        else:
            # ToolUseContent on a user message is not a normal shape;
            # OpenAI doesn't support it. The agent loop emits tool_use
            # only on assistant messages. Treat as malformed input.
            has_other = True

    if has_other:
        raise LlmProviderInvalidRequest(
            "OpenAI chat-completions does not support tool_use blocks on user messages"
        )

    out: list[dict[str, Any]] = []
    if text_blocks:
        out.append(
            {
                "role": "user",
                "content": _join_text_blocks(text_blocks),
            }
        )
    for result in tool_results:
        tool_msg: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": result.tool_use_id,
            "content": result.content,
        }
        out.append(tool_msg)
    return out


def _assistant_to_openai(msg: NormalizedMessage) -> dict[str, Any]:
    text_blocks: list[TextContent] = []
    tool_uses: list[ToolUseContent] = []
    for block in msg.content:
        if isinstance(block, TextContent):
            text_blocks.append(block)
        elif isinstance(block, ToolUseContent):
            tool_uses.append(block)
        else:
            # ToolResultContent on assistant is not valid in either
            # SMAI's shape or OpenAI's; surface as invalid request.
            raise LlmProviderInvalidRequest(
                "tool_result block on assistant role is not a valid normalized shape"
            )

    out: dict[str, Any] = {"role": "assistant"}
    # OpenAI accepts content=None when only tool_calls are present.
    out["content"] = _join_text_blocks(text_blocks) if text_blocks else None
    if tool_uses:
        out["tool_calls"] = [
            {
                "id": tu.id,
                "type": "function",
                "function": {
                    "name": tu.name,
                    "arguments": json.dumps(tu.input),
                },
            }
            for tu in tool_uses
        ]
    return out


def _join_text_blocks(blocks: list[TextContent]) -> str:
    return "".join(b.text for b in blocks)


def to_openai_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Translate normalized tool definitions into OpenAI ``tools``."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": dict(t.input_schema),
            },
        }
        for t in tools
    ]


def from_openai_response(response: Any) -> ModelResponse:
    """Translate an OpenAI ``ChatCompletion`` response into a
    :class:`ModelResponse`.

    Accepts either a duck-typed OpenAI SDK ``ChatCompletion`` object
    (with ``.choices`` / ``.usage`` attributes) or a dict with the
    same keys — the fake client used by the conformance tests queues
    canned dicts, while the real SDK returns SDK objects.
    """
    choices: Any = _get(response, "choices", [])
    if not isinstance(choices, list) or not choices:
        raise LlmProviderInvalidRequest(f"OpenAI response missing choices; raw={response!r}")
    choice = cast("list[Any]", choices)[0]

    raw_message: Any = _get(choice, "message", None)
    if raw_message is None:
        raise LlmProviderInvalidRequest(f"OpenAI response choice missing message; raw={response!r}")

    role = _get(raw_message, "role", None)
    if role != "assistant":
        raise LlmProviderInvalidRequest(f"OpenAI response role must be 'assistant', got {role!r}")

    content_blocks: list[NormalizedContent] = []
    text: Any = _get(raw_message, "content", None)
    if isinstance(text, str) and text:
        content_blocks.append(TextContent(text=text))

    tool_calls: Any = _get(raw_message, "tool_calls", None)
    if isinstance(tool_calls, list):
        for tc in cast("list[Any]", tool_calls):
            block = _from_openai_tool_call(tc)
            if block is not None:
                content_blocks.append(block)

    finish_reason = _get(choice, "finish_reason", None)
    has_tool_calls = isinstance(tool_calls, list) and len(cast("list[Any]", tool_calls)) > 0
    stop_reason = _normalize_finish_reason(finish_reason, has_tool_calls=has_tool_calls)

    usage = _from_openai_usage(_get(response, "usage", None))

    return ModelResponse(
        message=NormalizedMessage(role="assistant", content=content_blocks),
        stop_reason=stop_reason,
        usage=usage,
    )


def _from_openai_tool_call(tc: Any) -> ToolUseContent | None:
    """Translate one OpenAI ``tool_calls[i]`` entry into a
    :class:`ToolUseContent` block.

    Defensive about ``arguments`` parsing — OpenAI surfaces these as
    JSON strings, but malformed JSON should not crash the loop. Falls
    back to an empty dict (matching the conservative no-op surface
    contract — the agent will see the tool was called and can decide
    how to respond).
    """
    tc_type = _get(tc, "type", "function")
    if tc_type != "function":
        return None
    tc_id: Any = _get(tc, "id", "")
    function: Any = _get(tc, "function", None)
    if function is None:
        return None
    name: Any = _get(function, "name", "")
    raw_args: Any = _get(function, "arguments", "{}")
    parsed_args: dict[str, Any]
    if isinstance(raw_args, str):
        decoded: Any
        try:
            decoded = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            decoded = {}
        parsed_args = cast("dict[str, Any]", decoded) if isinstance(decoded, dict) else {}
    elif isinstance(raw_args, dict):
        parsed_args = dict(cast("dict[str, Any]", raw_args))
    else:
        parsed_args = {}

    return ToolUseContent(
        id=str(tc_id),
        name=str(name),
        input=parsed_args,
    )


def _normalize_finish_reason(raw: Any, *, has_tool_calls: bool) -> StopReason:
    """Collapse OpenAI ``finish_reason`` values into the normalized
    :data:`StopReason`.

    OpenAI returns ``stop`` / ``tool_calls`` / ``length`` /
    ``content_filter`` / ``function_call`` (deprecated). Our mapping:

    * ``tool_calls`` → ``"tool_use"``
    * ``length`` → ``"max_tokens"``
    * everything else → ``"end_turn"`` (matching v1's collapse of
      ``stop_sequence`` / ``content_filter``).

    The ``has_tool_calls`` parameter is a defensive fallback: some
    OpenAI-compatible servers (notably older Ollama versions and a few
    local proxies) report ``finish_reason="stop"`` even when the
    response carries tool calls. Treat that as ``"tool_use"`` so the
    agent loop's tool-execution branch fires correctly.
    """
    if raw == "tool_calls":
        return "tool_use"
    if raw == "length":
        return "max_tokens"
    if has_tool_calls:
        return "tool_use"
    return "end_turn"


def _from_openai_usage(raw: Any) -> TokenUsage:
    if raw is None:
        return TokenUsage(input_tokens=0, output_tokens=0)
    cached = 0
    details: Any = _get(raw, "prompt_tokens_details", None)
    if details is not None:
        cached = _int_field(details, "cached_tokens")
    return TokenUsage(
        input_tokens=_int_field(raw, "prompt_tokens"),
        output_tokens=_int_field(raw, "completion_tokens"),
        cache_read_tokens=cached,
        # OpenAI does not surface a "cache write" counter — automatic
        # caching is opaque to the caller.
        cache_write_tokens=0,
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

    The OpenAI SDK returns Pydantic-derived objects whose fields are
    attributes; the in-process fake queues plain dicts.
    """
    if isinstance(obj, dict):
        return cast("dict[str, Any]", obj).get(key, default)
    val = getattr(obj, key, default)
    return val


__all__ = [
    "from_openai_response",
    "to_openai_messages",
    "to_openai_tools",
]
