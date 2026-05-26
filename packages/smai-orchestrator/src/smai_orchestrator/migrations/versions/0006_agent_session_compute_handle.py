"""Add ``compute_job_handle`` column to ``agent_sessions`` (agent refactor Step 4 sub-PR B).

Revision ID: 0006_agent_session_handle
Revises: 0005_entry_dispatch_attempt
Create Date: 2026-05-26

Agent-layer refactor Step 4 sub-PR B (host-side dispatch substrate +
``smai-agent-runtime`` mini-orchestrator skeleton): the sandboxed
roles (harness_builder, technique_implementer) now run their agent
loop inside a Compute sandbox dispatched via
:func:`smai_orchestrator.dispatch.make_compute_dispatcher`. Each
dispatch creates an ``agent_sessions`` row at start (DEC-033 #2 cost
ledger). This column records the :class:`JobHandle` of the sandbox
the session ran in, denormalized into the cost row so operator
queries can correlate token spend with ``Compute.logs(handle)``
retrieval after the parent record's ``*_job_handle`` has been
overwritten by a later re-dispatch.

The authoritative dispatch-state :class:`JobHandle` continues to live
on the parent record (``cgs.harness_job_handle`` for harness_builder,
``entries.implementation_job_handle`` for technique_implementer,
``runs.compute_job_handle`` for seed runs). This column is a
cross-reference for the cost-ledger row's lifetime, not the source of
truth for dispatch state.

NULL for inline-role sessions (planner, supervisor, code_reviewer,
contextual_evaluator) which never run in a sandbox. Backfill: every
legacy row keeps NULL, since pre-refactor sandboxed-role sessions did
not yet exist (round 14 ran harness_builder / technique_implementer
in-process; pre-round-14 those sessions also lived inside the
orchestrator worker).

Forward-only per DEC-036; ``downgrade()`` raises so a stray
``alembic downgrade`` is loud rather than silent. Dual-mode add-column
pattern matches 0003 / 0004 / 0005: in offline (``--sql``) mode the
ADD COLUMN renders unconditionally; in online mode we inspect the live
table and skip when the column is already present (the 0001 baseline's
``MetaData.create_all(checkfirst=True)`` covers the column for fresh
databases once the canonical SQLAlchemy declaration in
``smai_orchestrator.migrations.metadata`` includes it).

JSON column type matches the existing ``*_job_handle`` columns
(``cgs.harness_job_handle``, ``entries.implementation_job_handle``,
``runs.compute_job_handle``, ``proposals.planner_job_handle``,
``papers.{fetcher,screener,planner}_job_handle``).
``none_as_null=True`` maps Python ``None`` to SQL ``NULL`` (rather
than the JSON literal ``'null'``) so ``IS NULL`` / ``IS NOT NULL``
predicates work as expected on the column.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import context, op

revision: str = "0006_agent_session_handle"
down_revision: str | None = "0005_entry_dispatch_attempt"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    if context.is_offline_mode():
        op.add_column(
            "agent_sessions",
            sa.Column(
                "compute_job_handle",
                sa.JSON(none_as_null=True),
                nullable=True,
            ),
        )
        return
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {col["name"] for col in inspector.get_columns("agent_sessions")}
    if "compute_job_handle" not in columns:
        op.add_column(
            "agent_sessions",
            sa.Column(
                "compute_job_handle",
                sa.JSON(none_as_null=True),
                nullable=True,
            ),
        )


def downgrade() -> None:
    raise NotImplementedError(
        "smai_orchestrator.migrations: down-migrations are not implemented in v2 "
        "(documented design-time deferral per Task 3.H2 / DEC-036). Roll back by "
        "restoring from a database backup."
    )
