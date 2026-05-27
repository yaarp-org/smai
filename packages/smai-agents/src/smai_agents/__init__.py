"""Host-side dispatcher wrappers for sandboxed agent roles.

After Step 8 Wave 1 of the agent-layer refactor (2026-05-27), this
package houses ONLY the host-side dispatcher factories for the
sandboxed agent roles (harness builder, technique implementer) plus
their support modules (artifact publish, harness API reference,
manifest tool). The inline agent roles (planner, supervisor, code
reviewer, contextual evaluator, screener, enricher) and the agent
substrate (loop, structured_call, prompts, schemas, std_tools,
tools, between_turn, cache, model_selection, retry_context,
truncation, agent_session_telemetry) moved to :mod:`smai_inline_agents`.

Step 8 Wave 2 (next) will relocate the remaining sandboxed-dispatcher
files into :mod:`smai_orchestrator.sandboxed_dispatch` and delete this
package entirely. The Wave-1 boundary keeps this surface importable so
the orchestrator's specs continue to find the dispatcher factories at
the same import paths until Wave 2's cutover.
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
