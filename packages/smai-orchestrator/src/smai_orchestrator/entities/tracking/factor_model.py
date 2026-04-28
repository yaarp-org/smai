""":class:`FactorModelRecord` — runtime instance of a
:class:`smai_core.entities.FactorModel`.

Per ``designs/smai/01-data-model.md`` §5.8. Lifecycle is **degenerate**
in v1 — purely organizational per DEC-014 / DEC-031 #5; the entity carries
no state machine, no dispatch, no lease, no retry counters. CAS exists
for safe edits to the cross-CG association set in vN if FactorModel
grows mutability. Per §5.2.4, the lease triple is explicitly omitted —
FactorModelRecord extends :class:`BasePipelineRecord` directly rather
than :class:`LeaseableRecord`.

Per §5.2.6, the ``transition_log`` event-sourcing pattern does not apply
here — there are no transitions to log, so ``last_write_wins`` (the
``created_at`` / ``updated_at`` shape on :class:`BasePipelineRecord`) is
sufficient.
"""

from __future__ import annotations

from pydantic import field_validator

from smai_orchestrator.entities.tracking._common import (
    BasePipelineRecord,
    validate_id_format,
)


class FactorModelRecord(BasePipelineRecord):
    """Runtime instance of a ``FactorModel`` (§5.8)."""

    # === Identity ===
    id: str

    # === Methodology references ===
    factor_model_id: str
    research_question: str

    _validate_ids = field_validator("id", "factor_model_id")(validate_id_format)


__all__ = ["FactorModelRecord"]
