"""Technique-implementer sandboxed agent role.

Per ``designs/smai/agent_refactor/architectural_decisions.md`` §6, the
technique-implementer runs inside the agent sandbox image as a five-
step mini-orchestrator per ``notes/research_report.md`` §6.2.

Step 3 of the refactor ships the package skeleton; Step 7 lands the
mini-orchestrator body. This module currently exposes a :func:`main`
stub that raises :class:`RoleNotImplementedError`.
"""

from __future__ import annotations

from smai_agent_runtime.technique_implementer._main import main

__all__ = ["main"]
