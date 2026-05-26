"""Technique-implementer ``main`` stub. Body lands in Step 7 of the
agent-layer refactor (see ``agent_refactor/implementation_plan.md``).
"""

from __future__ import annotations

import argparse

from smai_agent_runtime.errors import RoleNotImplementedError


def main(args: argparse.Namespace) -> int:
    """Entry point invoked by :mod:`smai_agent_runtime.__main__`.

    Step 7 of the refactor replaces this raise with the five-step
    mini-orchestrator per ``notes/research_report.md`` §6.2. Until then
    the stub raises so the entry point surfaces "not yet implemented"
    cleanly per the Step 3 acceptance criterion.
    """
    if args.entry_id is None:
        raise RoleNotImplementedError(
            "technique_implementer requires --entry-id (Step 7 will "
            "consume it to stage the workspace + dispatch the mini-"
            "orchestrator)"
        )
    raise RoleNotImplementedError(
        "technique_implementer mini-orchestrator lands in Step 7 of the "
        "agent-layer refactor (see "
        "designs/smai/agent_refactor/implementation_plan.md). The Step 3 "
        "package skeleton exercises only the entry-point dispatch."
    )
