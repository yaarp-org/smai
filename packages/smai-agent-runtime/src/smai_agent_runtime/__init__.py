"""Sandbox-side agent runtime.

Per ``designs/smai/agent_refactor/architectural_decisions.md`` §3 and
``compute_dispatch_decisions.md`` §6 (DEC-038): this package is consumed
*inside* the agent sandbox container image (``smai-agent-runtime:dev``)
and not by the host worker process. It ships the mini-orchestrator that
sequences each sandboxed role's workflow plus the PydanticAI Agent
calls that perform structured-output reasoning at each step.

At Step 3 of the refactor (this commit) the package ships the skeleton
only:

* :mod:`smai_agent_runtime.__main__` — ``python -m smai_agent_runtime
  --role <role> --cg-id <id>`` entry point that dispatches to a per-role
  ``main`` function.
* :mod:`smai_agent_runtime.harness_builder` — stub. Body lands in
  Step 4 of the refactor plan.
* :mod:`smai_agent_runtime.technique_implementer` — stub. Body lands in
  Step 7 of the refactor plan.

Non-implemented roles exit with a clear ``NotImplementedError`` -style
message rather than an import error, so the substrate gate (Step 3
acceptance criterion) can assert dispatch wiring without depending on
agent-reasoning code.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
