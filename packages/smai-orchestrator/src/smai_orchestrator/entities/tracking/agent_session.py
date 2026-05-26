"""Public ``AgentSessionRecord`` — per-invocation agent-loop telemetry row.

Per ``designs/decisions.md`` DEC-033 #2 (the ``agent_sessions`` normalized
cost / telemetry table) and the observability-hardening pass that wires the
:class:`smai_core.plugins.MetadataStore` CRUD methods
(``create_agent_session`` / ``update_agent_session`` / ``get_agent_session``)
which return / accept this shape.

This is the *public* record the Protocol references. The SQLite reference
plugin keeps a private row-mapping struct in ``smai_store_sqlite._records``
for its own table-IO bookkeeping (it carries extra columns like the
auto-increment ``id`` on ``transition_log``); the canonical wire shape for
the plugin Protocol is this one.

Scope note: the per-token USD **rate** columns on the underlying
``agent_sessions`` table are out of scope for the current pass — cost
tracking (``run_costs``, the rate table) is backlogged — so they are not
modelled here. They stay ``NULL`` in the table. When the cost-rate work
lands, this record gains the three ``*_rate_usd_per_million`` fields.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator
from smai_core.plugins import JobHandle

from smai_orchestrator.entities.tracking._common import validate_id_format


class AgentSessionRecord(BaseModel):
    """One agent-loop invocation's telemetry.

    Created at dispatch start (``started_at = now()``, ``ended_at = None``,
    token/turn counts ``0``), UPDATEd on the loop's per-turn cadence with
    the running totals, and closed (``ended_at`` set) when the loop
    returns.

    ``parent_kind`` / ``parent_id`` is the polymorphic FK to the
    pipeline-tracking entity that owns the invocation (e.g.
    ``("proposal", "<proposal_id>")`` for the planner). Kept as a wide
    ``str`` rather than a ``Literal`` for forward-compat — see
    ``smai_store_sqlite._records.AgentSessionParentKind`` for the
    candidate enumeration.

    ``compute_job_handle`` cross-references the sandbox the session ran
    in for sandboxed roles (harness_builder, technique_implementer).
    NULL for inline roles (planner, supervisor, reviewers) and for
    legacy pre-Step-4-sub-PR-B rows. Lets operator queries correlate a
    cost-ledger row with ``Compute.logs(handle)`` retrieval after the
    parent record's ``*_job_handle`` has been overwritten by a later
    re-dispatch. Persisted as a JSON column via migration 0006 (per
    agent_refactor design note D2 / ``agent_session_jobhandle.md``).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    parent_kind: str
    parent_id: str

    agent_role: str
    llm_provider: str | None = None
    model_id: str | None = None

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    turn_count: int = 0

    started_at: datetime
    ended_at: datetime | None = None

    compute_job_handle: JobHandle | None = None

    _validate_ids = field_validator("id", "parent_id")(validate_id_format)


__all__ = ["AgentSessionRecord"]
