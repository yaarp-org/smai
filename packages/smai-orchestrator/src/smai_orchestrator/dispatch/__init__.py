"""Unified compute-dispatch factory (agent-refactor Step 2).

Per ``designs/smai/agent_refactor/compute_dispatch_decisions.md`` §3 and
``designs/smai/agent_refactor/architectural_decisions.md`` §5: agent and
experiment dispatches share substrate. The factory produces a
:data:`DispatchHandler` that stages workspace inputs, submits the
:class:`Compute` job, and returns the resulting :class:`JobHandle`. The
seed-run dispatcher (originally in :mod:`smai_orchestrator.specs.run_record`)
migrates onto this factory at Step 2's PR boundary; the agent-side
dispatchers (Steps 4 / 7) follow.

Round-10's declarative :class:`RetryPolicy` on :class:`DispatchAction`
and round-20's stderr-on-failure surface are both supported by the
same machinery the engine already provides; the factory wires them.
"""

from smai_orchestrator.dispatch._factory import (
    CommandSpec,
    WorkspaceInputs,
    WorkspaceOutputs,
    format_stderr_tail,
    make_compute_dispatcher,
)

__all__ = [
    "CommandSpec",
    "WorkspaceInputs",
    "WorkspaceOutputs",
    "format_stderr_tail",
    "make_compute_dispatcher",
]
