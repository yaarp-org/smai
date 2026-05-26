"""Harness-builder sandboxed agent role.

Per ``designs/smai/agent_refactor/architectural_decisions.md`` §6, the
harness-builder runs inside the agent sandbox image as a mini-
orchestrator: a generator-derived workflow of scripted Python steps
plus PydanticAI ``Agent(output_type=...)`` calls per
``implementation_plan.md`` Step 4.

Step 3 of the refactor ships the package skeleton; Step 4 lands the
mini-orchestrator body. This module currently exposes a :func:`main`
stub that raises :class:`RoleNotImplementedError` so the entry-point
dispatch wiring is exercised without depending on agent-reasoning code.
"""

from __future__ import annotations

from smai_agent_runtime.harness_builder._main import main

__all__ = ["main"]
