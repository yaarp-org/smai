"""Sandboxed-role host-side dispatcher factories.

After Step 8 Wave 1 of the agent-layer refactor (2026-05-27), only the
host-side dispatcher factories + their support modules remain here.
Inline-role agents (planner, code_reviewer, contextual_evaluator,
supervisor, screener, enricher) moved to
:mod:`smai_inline_agents.agents`.

Step 8 Wave 2 (next) relocates these files into
:mod:`smai_orchestrator.sandboxed_dispatch` and deletes the
:mod:`smai_agents` package.
"""

from smai_agents.agents.harness_builder import (
    make_dispatch_harness_build_sandboxed,
)
from smai_agents.agents.manifest_tool import (
    EMIT_HARNESS_MANIFEST_TOOL_NAME,
    make_emit_harness_manifest_tool,
)
from smai_agents.agents.technique_implementer import (
    make_dispatch_technique_implementation_sandboxed,
)

__all__ = [
    "EMIT_HARNESS_MANIFEST_TOOL_NAME",
    "make_dispatch_harness_build_sandboxed",
    "make_dispatch_technique_implementation_sandboxed",
    "make_emit_harness_manifest_tool",
]
