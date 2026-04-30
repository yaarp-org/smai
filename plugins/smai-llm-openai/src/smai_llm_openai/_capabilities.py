"""Static capability metadata per OpenAI model.

Per ``07-plugin-interfaces.md`` §4.4 and Task 3.F5. Each
:class:`smai_core.plugins.LlmProvider` instance carries its own
:class:`LlmCapabilities`. Per-task model selection is the agent-config
concern (DEC-022) — this table covers the OpenAI tiers the agent fleet
might select from.

OpenAI's chat-completions API does **not** expose explicit prompt-cache
markers (the platform applies automatic caching server-side). Per the
``07-plugin-interfaces.md`` §4.3 contract: when ``supports_caching`` is
False, the plugin silently ignores ``cache_config``. Setting
``supports_caching=False`` is the honest signal — automatic
server-side caching is not equivalent to caller-controlled caching.

Unknown model IDs fall through to a conservative GPT-4-shaped default
(no caching, 128k context, 4k output). Caller can always override by
passing ``capabilities=...`` to the :class:`OpenAIProvider`
constructor.
"""

from __future__ import annotations

from smai_core.plugins import LlmCapabilities

_OPENAI_CAPABILITIES: dict[str, LlmCapabilities] = {
    "gpt-4o": LlmCapabilities(
        supports_caching=False,
        context_window=128_000,
        max_output_tokens=16_384,
        supports_tool_use=True,
        model_id="gpt-4o",
    ),
    "gpt-4o-mini": LlmCapabilities(
        supports_caching=False,
        context_window=128_000,
        max_output_tokens=16_384,
        supports_tool_use=True,
        model_id="gpt-4o-mini",
    ),
    "gpt-4-turbo": LlmCapabilities(
        supports_caching=False,
        context_window=128_000,
        max_output_tokens=4_096,
        supports_tool_use=True,
        model_id="gpt-4-turbo",
    ),
    # ``o1`` series — reasoning models. Tool use support varies by
    # model variant; we report ``True`` because the agent loop uses
    # tool-use for structured output (DEC-018) and needs the capability
    # claim. ``o1-mini`` and the latest ``o1`` both carry tool-use in
    # current production.
    "o1": LlmCapabilities(
        supports_caching=False,
        context_window=200_000,
        max_output_tokens=100_000,
        supports_tool_use=True,
        model_id="o1",
    ),
    "o1-mini": LlmCapabilities(
        supports_caching=False,
        context_window=128_000,
        max_output_tokens=65_536,
        supports_tool_use=True,
        model_id="o1-mini",
    ),
}


def lookup_capabilities(model_id: str) -> LlmCapabilities:
    """Return :class:`LlmCapabilities` for ``model_id``.

    Falls through to a conservative GPT-4-shaped default when the ID is
    unknown.
    """
    entry = _OPENAI_CAPABILITIES.get(model_id)
    if entry is not None:
        return entry
    # Date-suffixed variants (``gpt-4o-2024-08-06``) prefix-match the
    # canonical alias.
    for prefix, caps in _OPENAI_CAPABILITIES.items():
        if model_id.startswith(prefix):
            return caps.model_copy(update={"model_id": model_id})
    return LlmCapabilities(
        supports_caching=False,
        context_window=128_000,
        max_output_tokens=4_096,
        supports_tool_use=True,
        model_id=model_id,
    )


__all__ = ["lookup_capabilities"]
