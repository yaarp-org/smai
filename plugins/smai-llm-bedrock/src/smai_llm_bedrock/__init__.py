"""LlmProvider plugin: AWS Bedrock Converse adapter.

Per ``07-plugin-interfaces.md`` §4 and DEC-005 / DEC-019 / DEC-020. The
Phase 2 reference :class:`smai_core.plugins.LlmProvider` implementation
per the implementation_plan §3.3 Task 2.A1.

Registered via the ``smai.llm_providers`` entry-point group::

    [project.entry-points."smai.llm_providers"]
    bedrock = "smai_llm_bedrock:BedrockProvider"

Tier A integrators (the in-tree CLI / hosted backend) instantiate the
plugin through the entry-point discovery flow owned by ``smai-cli`` (see
Task 2.C3); Tier B integrators import :class:`BedrockProvider` directly.
"""

from smai_llm_bedrock._provider import BedrockProvider

__all__ = ["BedrockProvider"]
