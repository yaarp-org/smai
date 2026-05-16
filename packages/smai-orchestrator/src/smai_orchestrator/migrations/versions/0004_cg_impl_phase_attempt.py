"""Add ``implementation_phase_attempt`` column to the ``cgs`` table (round 10).

Revision ID: 0004_cg_impl_phase_attempt
Revises: 0003_cg_symbolic_id
Create Date: 2026-05-13

Round 10 (declarative retry bookkeeping): the CG ``implementing`` state's
dispatch (harness builder) had no retry-budget terminal pre-round-10, so
a dispatch-handler failure infinitely rolled back to ``draft`` and re-
dispatched. The round-10 refactor lifts retry bookkeeping into the
engine via a :class:`RetryPolicy` declaration on
:class:`DispatchAction`; the CG ``implementing`` state declares one with
this new counter as its ``attempt_counter_field``.

The column lands with default 0 and ``nullable=False`` to match the
other ``*_attempt`` counters on the same table (``code_review_attempt``).
Backfill is trivial since the default applies to every legacy row.

Forward-only per DEC-036; ``downgrade()`` raises so a stray
``alembic downgrade`` is loud rather than silent. Same dual-mode shape
as 0003: in offline (``--sql``) mode the ADD COLUMN renders
unconditionally; in online mode we inspect the live table and skip when
the column is already present (the 0001 baseline's
``MetaData.create_all(checkfirst=True)`` covers the column for fresh
databases since the canonical SQLAlchemy declaration already includes
it post-round-10).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import context, op

revision: str = "0004_cg_impl_phase_attempt"
down_revision: str | None = "0003_cg_symbolic_id"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    if context.is_offline_mode():
        op.add_column(
            "cgs",
            sa.Column(
                "implementation_phase_attempt",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
        )
        return
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {col["name"] for col in inspector.get_columns("cgs")}
    if "implementation_phase_attempt" not in columns:
        op.add_column(
            "cgs",
            sa.Column(
                "implementation_phase_attempt",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
        )


def downgrade() -> None:
    raise NotImplementedError(
        "smai_orchestrator.migrations: down-migrations are not implemented in v2 "
        "(documented design-time deferral per Task 3.H2 / DEC-036). Roll back by "
        "restoring from a database backup."
    )
