""":class:`SupervisorDecision` — supervisor's locked output schema.

Per ``04-agents.md`` §2.6 (verbatim shape) and §6 (single-call structured
output via ``tool_use``). Consumed by:

* :func:`smai_inline_agents.agents.supervisor.run_supervisor_check` — the
  role-shaped wrapper that hands this to :func:`smai_inline_agents.structured_call`
  as ``output_schema``.
* :func:`smai_inline_agents.between_turn.maybe_run_supervisor_check` — the
  between-turn hook that translates the decision into either a nudge
  file write (``action='intervene'``) or a session-abort flag
  (``action='abort'``).

The three-action enum is committed by §2.6 verbatim:

* ``continue`` — no-op; the loop carries on.
* ``intervene`` — the supervisor wants a nudge delivered; ``nudge``
  carries the text written to the agent's nudge-file location.
* ``abort`` — the supervisor wants the loop terminated; the dispatch
  handler picks the abort up and routes the parent entity to a
  ``*_failed`` state per its pipeline-spec.

Per §15 OQ2 the *invocation lifecycle* (cadence, configurable thresholds,
per-call cost metering) is open; this module ships the *role* surface
verbatim per §2.6, leaving the lifecycle wiring to the engine /
``EngineConfig`` plumbing in Task 3.G4.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

SupervisorAction = Literal["continue", "intervene", "abort"]


class SupervisorDecision(BaseModel):
    """Locked supervisor verdict per ``04-agents.md`` §2.6.

    The ``action`` enum is the load-bearing surface; ``reason`` is
    free-text rationale (always populated, including for ``continue``
    so debug traces carry the supervisor's reasoning); ``nudge`` is
    populated **iff** ``action == 'intervene'`` and is the text the
    between-turn hook writes to the agent's nudge-file location.

    A Pydantic validator enforces the ``intervene`` ↔ ``nudge``
    invariant: ``action='intervene'`` requires a non-empty ``nudge``;
    ``action != 'intervene'`` requires ``nudge`` to be ``None``. v1's
    reasoning carries forward (DEC-018 family): silently accepting an
    inconsistent ``intervene`` with no nudge content is a bug shape we
    refuse to ship.
    """

    model_config = ConfigDict(extra="forbid")

    action: SupervisorAction
    reason: str
    nudge: str | None = None

    @model_validator(mode="after")
    def _nudge_required_iff_intervene(self) -> Self:
        if self.action == "intervene":
            if self.nudge is None or not self.nudge.strip():
                raise ValueError(
                    "SupervisorDecision: action='intervene' requires a non-empty nudge"
                )
        elif self.nudge is not None:
            raise ValueError(
                f"SupervisorDecision: action={self.action!r} must leave nudge=None "
                f"(nudge is only meaningful for action='intervene')"
            )
        return self


__all__ = ["SupervisorAction", "SupervisorDecision"]
