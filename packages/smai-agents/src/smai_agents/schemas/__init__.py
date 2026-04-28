"""Pydantic output schemas for single-call structured-output agents.

Per ``04-agents.md`` §6 (and DEC-018), each single-call agent role
defines its ``output_schema`` as a Pydantic model — the shape the
consumer wants. The schemas live alongside :mod:`smai_agents.agents` so
both the role-shaped wrappers (which call :func:`structured_call`) and
downstream consumers (the orchestrator's gate-rule body and dispatch
handler in Task 2.C4) import the same locked types.

* :mod:`smai_agents.schemas.code_review` — :class:`CodeReviewResult` per §2.4.
* :mod:`smai_agents.schemas.contextual_verdict` — :class:`ContextualVerdict`
  per §2.5.
"""

from smai_agents.schemas.code_review import (
    CodeReviewResult,
    Finding,
)
from smai_agents.schemas.contextual_verdict import (
    ContextualVerdict,
    EntryRanking,
)

__all__ = [
    "CodeReviewResult",
    "ContextualVerdict",
    "EntryRanking",
    "Finding",
]
