""":class:`CacheConfig` defaults for the agent loop.

Per ``04-agents.md`` §5 / DEC-019. The agent loop expresses cache *intent*;
the :class:`smai_core.plugins.LlmProvider` plugin (when
``capabilities.supports_caching=True``) translates the intent into
provider-specific cache markers (Bedrock ``cachePoint``, Anthropic
``cache_control``). Non-caching plugins ignore the field, so the loop
never branches on substrate.

The loop's default is the multi-turn pattern: 2 static checkpoints (tool
definitions + initial user message) plus a 2-message rolling tail. The
single-call default lives next to :func:`smai_inline_agents.structured_call.structured_call`
per §5's final paragraph (no rolling on a one-message conversation).

The :class:`smai_core.plugins.CacheConfig` Pydantic model is defined in
``smai-core``; this module re-exports it for convenient imports and
adds the engine-side defaults.
"""

from __future__ import annotations

from smai_core.plugins import CacheConfig

# Per §5: "The 2-static + 2-rolling pattern. v1 settled this in
# prompt_caching.md §4." Default for multi-turn agents (planner, harness
# builder, technique implementer).
DEFAULT_CACHE_CONFIG = CacheConfig(
    cache_static_prefix=True,
    cache_initial_message=True,
    rolling_cache_count=2,
)


# Per §5 final paragraph: "Code review and contextual evaluator have no
# rolling conversation; ... rolling_cache_count=0 for these roles."
SINGLE_CALL_CACHE_CONFIG = CacheConfig(
    cache_static_prefix=True,
    cache_initial_message=True,
    rolling_cache_count=0,
)


__all__ = ["CacheConfig", "DEFAULT_CACHE_CONFIG", "SINGLE_CALL_CACHE_CONFIG"]
