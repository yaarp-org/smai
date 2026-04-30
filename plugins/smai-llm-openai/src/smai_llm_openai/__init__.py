"""LlmProvider plugin: OpenAI Chat Completions adapter.

Per ``07-plugin-interfaces.md`` §4 and the implementation_plan §3.4
Task 3.F5. Wraps the official ``openai`` SDK's
``AsyncOpenAI.chat.completions.create`` surface.

Registered via the ``smai.llm_providers`` entry-point group::

    [project.entry-points."smai.llm_providers"]
    openai = "smai_llm_openai:OpenAIProvider"

Tier A integrators (the in-tree CLI / hosted backend) instantiate the
plugin through the entry-point discovery flow owned by ``smai-cli``;
Tier B integrators import :class:`OpenAIProvider` directly.
"""

from smai_llm_openai._provider import OpenAIProvider

__all__ = ["OpenAIProvider"]
