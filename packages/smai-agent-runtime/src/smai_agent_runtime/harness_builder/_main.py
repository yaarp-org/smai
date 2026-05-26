"""Harness-builder ``main`` stub. Body lands in Step 4 of the
agent-layer refactor (see ``agent_refactor/implementation_plan.md``).
"""

from __future__ import annotations

import argparse

from smai_agent_runtime.errors import RoleNotImplementedError


def main(args: argparse.Namespace) -> int:
    """Entry point invoked by :mod:`smai_agent_runtime.__main__`.

    Step 4 of the refactor replaces this raise with the real mini-
    orchestrator (workflow generator + per-step PydanticAI Agent calls
    + scripted lint / validate / manifest-emit steps). Until then the
    stub raises so the entry point surfaces "not yet implemented"
    cleanly per the Step 3 acceptance criterion.
    """
    if args.cg_id is None:
        # Argparse already requires --role; --cg-id is per-role so the
        # entry point can't reject upstream. Surface the missing arg as
        # a not-yet-implemented diagnostic rather than an AttributeError
        # downstream.
        raise RoleNotImplementedError(
            "harness_builder requires --cg-id (Step 4 will consume it to "
            "stage the workspace + dispatch the mini-orchestrator)"
        )
    raise RoleNotImplementedError(
        "harness_builder mini-orchestrator lands in Step 4 of the "
        "agent-layer refactor (see "
        "designs/smai/agent_refactor/implementation_plan.md). The Step 3 "
        "package skeleton exercises only the entry-point dispatch."
    )
