"""LlmProvider plugin: Anthropic native API adapter.

Per ``07-plugin-interfaces.md`` §4 and the implementation_plan §3.4
Task 3.F5. Wraps the official ``anthropic`` SDK's
``AsyncAnthropic.messages.create`` surface.

Registered via the ``smai.llm_providers`` entry-point group::

    [project.entry-points."smai.llm_providers"]
    anthropic = "smai_llm_anthropic:AnthropicProvider"

Tier A integrators (the in-tree CLI / hosted backend) instantiate the
plugin through the entry-point discovery flow owned by ``smai-cli``;
Tier B integrators import :class:`AnthropicProvider` directly.
"""

from smai_llm_anthropic._provider import AnthropicProvider

__all__ = ["AnthropicProvider"]
