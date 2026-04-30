"""Static capability metadata per Anthropic model.

Per ``07-plugin-interfaces.md`` §4.4 and Task 3.F5. Each
:class:`smai_core.plugins.LlmProvider` instance carries its own
:class:`LlmCapabilities`. Per-task model selection is the agent-config
concern (DEC-022) — this table covers the Claude tiers the agent fleet
selects from per ``04-agents.md`` §4.

Anthropic's native API supports prompt caching for Claude via
``cache_control: {"type": "ephemeral"}`` on system / message / tool
content blocks. All current Claude tiers support tool use.

Unknown model IDs fall through to a conservative default (caching on,
200k context, 4k output) — Claude is the dominant Anthropic model
family and any new ID is overwhelmingly likely to be a Claude variant.
The caller can always override by passing ``capabilities=...`` to the
:class:`AnthropicProvider` constructor.
"""

from __future__ import annotations

from smai_core.plugins import LlmCapabilities

# Conservative defaults for current Claude tiers. Concrete model IDs
# turn over with model generations; per-task selection lives in
# :mod:`smai_agents.model_selection` (DEC-022) — this table is the
# capability-flag mapping, not a model registry.
_ANTHROPIC_CAPABILITIES: dict[str, LlmCapabilities] = {
    "claude-opus-4-7": LlmCapabilities(
        supports_caching=True,
        context_window=200_000,
        max_output_tokens=16_384,
        supports_tool_use=True,
        model_id="claude-opus-4-7",
    ),
    "claude-opus-4-6": LlmCapabilities(
        supports_caching=True,
        context_window=200_000,
        max_output_tokens=16_384,
        supports_tool_use=True,
        model_id="claude-opus-4-6",
    ),
    "claude-sonnet-4-6": LlmCapabilities(
        supports_caching=True,
        context_window=200_000,
        max_output_tokens=16_384,
        supports_tool_use=True,
        model_id="claude-sonnet-4-6",
    ),
    "claude-haiku-4-5": LlmCapabilities(
        supports_caching=True,
        context_window=200_000,
        max_output_tokens=16_384,
        supports_tool_use=True,
        model_id="claude-haiku-4-5",
    ),
}


def lookup_capabilities(model_id: str) -> LlmCapabilities:
    """Return :class:`LlmCapabilities` for ``model_id``.

    Falls through to a Claude-shaped default when the ID is unknown.
    """
    entry = _ANTHROPIC_CAPABILITIES.get(model_id)
    if entry is not None:
        return entry
    # Date-suffixed variants (``claude-opus-4-7-20251001``) prefix-match
    # against the canonical alias.
    for prefix, caps in _ANTHROPIC_CAPABILITIES.items():
        if model_id.startswith(prefix):
            return caps.model_copy(update={"model_id": model_id})
    return LlmCapabilities(
        supports_caching=True,
        context_window=200_000,
        max_output_tokens=4_096,
        supports_tool_use=True,
        model_id=model_id,
    )


__all__ = ["lookup_capabilities"]
