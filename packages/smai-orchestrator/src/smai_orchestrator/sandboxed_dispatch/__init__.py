"""Host-side dispatcher factories for the sandboxed agent roles.

After Step 8 Wave 2 of the agent-layer refactor (2026-05-27), the
harness_builder and technique_implementer dispatcher wirings live
here. The sandbox-side mini-orchestrators (the code that runs INSIDE
the agent container) live in :mod:`smai_agent_runtime`; the
inline-role agents (planner, code_reviewer, etc.) live in
:mod:`smai_inline_agents`. The host-side wiring sits in this
package because the orchestrator is the host process.

Exports:

* :func:`make_dispatch_harness_build_sandboxed` (was
  ``smai_agents.agents.harness_builder``)
* :func:`make_dispatch_technique_implementation_sandboxed` (was
  ``smai_agents.agents.technique_implementer``)
* :func:`make_emit_harness_manifest_tool` +
  :data:`EMIT_HARNESS_MANIFEST_TOOL_NAME` (was
  ``smai_agents.agents.manifest_tool``; legacy in-process tool the
  pre-Step-7 harness_builder used to emit the manifest, retained for
  the regression test that exercises the validation surface)
"""

from smai_orchestrator.sandboxed_dispatch.harness_builder import (
    make_dispatch_harness_build_sandboxed,
)
from smai_orchestrator.sandboxed_dispatch.manifest_tool import (
    EMIT_HARNESS_MANIFEST_TOOL_NAME,
    make_emit_harness_manifest_tool,
)
from smai_orchestrator.sandboxed_dispatch.technique_implementer import (
    make_dispatch_technique_implementation_sandboxed,
)

__all__ = [
    "EMIT_HARNESS_MANIFEST_TOOL_NAME",
    "make_dispatch_harness_build_sandboxed",
    "make_dispatch_technique_implementation_sandboxed",
    "make_emit_harness_manifest_tool",
]
