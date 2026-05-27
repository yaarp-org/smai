"""Context truncation policy for the agent loop.

Per ``04-agents.md`` §7 / ``07-plugin-interfaces.md`` §4.4. The agent
loop owns truncation; the :class:`smai_core.plugins.LlmProvider` Protocol
explicitly forbids plugins from doing it. v1's two-stage safety margin
lifts unchanged.

Token estimation is approximate (~4 chars per token). The 10% reserved
gap between ``effective_limit`` and the raw context window absorbs
estimation error per §7's note. An exact ``count_tokens`` Protocol
method is candidate vN work (07 §12 Q13).

Middle-truncation only (per §7's "Middle-truncation is the only
strategy in v1"): preserve the head (task definition + initial context)
and the tail (recent work). The agent re-reads files from disk if it
needs middle content.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict
from smai_core.plugins import (
    LlmCapabilities,
    NormalizedContent,
    NormalizedMessage,
    TextContent,
    ToolDefinition,
    ToolUseContent,
)

# Char-count heuristic per §7 / 07 §12 Q13. Coarse but matches v1's
# stance; the 10% reserved gap below the context window absorbs
# estimation error.
_CHARS_PER_TOKEN = 4


class TruncationPolicy(BaseModel):
    """Two-stage safety margin (v1 carry-forward per ``04-agents.md`` §7).

    For Opus (200K window): ``effective_limit = 180K``, truncation fires
    at 153K (``= 180K * 0.85``). For 128K-window models: effective limit
    = 115.2K, truncation fires at ~98K. Proportionally scaled by
    ``capabilities.context_window``.
    """

    model_config = ConfigDict(extra="forbid")

    threshold_fraction: float = 0.90
    """Effective context-window limit = ``capabilities.context_window *
    threshold_fraction``. The remaining 10% is reserved for the
    assistant's response output plus a small safety margin for
    tokenization estimates."""

    truncate_at_fraction: float = 0.85
    """Truncation fires when estimated tokens >= ``effective_limit *
    truncate_at_fraction``. The 15% gap below the effective limit is
    for "the next user message plus the next assistant response will
    fit, even after we just truncated." """

    keep_head_messages: int = 2
    """The first 2 messages (initial user + first assistant response).
    The system prompt is separate (lives outside ``messages``) and is
    not counted here."""

    keep_tail_messages: int = 8
    """The last 8 messages — recent work."""


def estimate_message_tokens(message: NormalizedMessage) -> int:
    """Approximate token count for one normalized message.

    Counts the underlying text — :class:`TextContent` text,
    :class:`ToolUseContent` JSON-serialized input, :class:`ToolResultContent`
    content. Not exact; see module docstring.
    """
    chars = 0
    for block in message.content:
        chars += _content_chars(block)
    return _chars_to_tokens(chars)


def estimate_total_tokens(messages: list[NormalizedMessage]) -> int:
    """Approximate token count for a list of normalized messages."""
    return sum(estimate_message_tokens(m) for m in messages)


def estimate_overhead_tokens(*, system: str, tools: list[ToolDefinition] | None) -> int:
    """Approximate token count for the always-sent prefix.

    System prompt + tool definitions are sent on every call but live
    outside ``messages``. The truncation decision compares the *full*
    request size against the effective limit, so we have to fold them in.
    """
    chars = len(system)
    if tools:
        for tool in tools:
            chars += len(tool.name) + len(tool.description)
            chars += len(json.dumps(tool.input_schema, sort_keys=True))
    return _chars_to_tokens(chars)


def effective_limit(capabilities: LlmCapabilities, policy: TruncationPolicy) -> int:
    """Per §7: ``effective_limit = capabilities.context_window * threshold_fraction``."""
    return int(capabilities.context_window * policy.threshold_fraction)


def truncation_threshold(capabilities: LlmCapabilities, policy: TruncationPolicy) -> int:
    """Per §7: truncation fires at ``effective_limit * truncate_at_fraction``."""
    return int(effective_limit(capabilities, policy) * policy.truncate_at_fraction)


def should_truncate(
    *,
    messages: list[NormalizedMessage],
    capabilities: LlmCapabilities,
    policy: TruncationPolicy,
    system: str = "",
    tools: list[ToolDefinition] | None = None,
) -> bool:
    """Return ``True`` iff the conversation has crossed the trigger threshold."""
    estimated = estimate_total_tokens(messages) + estimate_overhead_tokens(
        system=system, tools=tools
    )
    return estimated >= truncation_threshold(capabilities, policy)


def truncate_messages(
    messages: list[NormalizedMessage], policy: TruncationPolicy
) -> list[NormalizedMessage]:
    """Drop the middle of the conversation, keeping head + tail.

    Per §7: "middle-truncation only ... preserve the head (task
    definition + initial context) and the tail (recent work)." If the
    conversation is short enough that ``head + tail >= len(messages)``,
    nothing is dropped — the policy is a ceiling, not a floor.

    Tool-use / tool-result pairing is best-effort: dropping the middle
    can leave a ``ToolUseContent`` from a kept assistant turn whose
    matching ``ToolResultContent`` lived in a dropped user turn (or
    vice versa). The Bedrock Converse API tolerates this for
    long-running sessions because providers don't validate
    cross-turn linkage; if a future provider strictly does we can
    pair-aware-truncate as a follow-up. For now the simple slice is
    consistent with v1.
    """
    head = max(0, policy.keep_head_messages)
    tail = max(0, policy.keep_tail_messages)
    if head + tail >= len(messages):
        return list(messages)
    head_slice = messages[:head]
    tail_slice = messages[-tail:] if tail > 0 else []
    return [*head_slice, *tail_slice]


# --- internal helpers -------------------------------------------------------


def _content_chars(block: NormalizedContent) -> int:
    if isinstance(block, TextContent):
        return len(block.text)
    if isinstance(block, ToolUseContent):
        # ``input`` is JSON-shaped; canonical-ish serialization gives a
        # stable char count without claiming to mirror the provider's
        # exact tokenizer treatment.
        return len(block.name) + len(_canonical_json(block.input))
    # NormalizedContent is the three-arm union; the remaining arm is
    # ToolResultContent.
    return len(block.content)


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _chars_to_tokens(chars: int) -> int:
    if chars <= 0:
        return 0
    # Round up — the truncation gate is a ceiling check.
    return (chars + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


__all__ = [
    "TruncationPolicy",
    "effective_limit",
    "estimate_message_tokens",
    "estimate_overhead_tokens",
    "estimate_total_tokens",
    "should_truncate",
    "truncate_messages",
    "truncation_threshold",
]
