"""Add ``entry_dispatch_attempt`` column to the ``entries`` table (round 11).

Revision ID: 0005_entry_dispatch_attempt
Revises: 0004_cg_impl_phase_attempt
Create Date: 2026-05-16

Round 11 (close the ``cg_entries.implementing`` unbounded retry loop):
the entry ``implementing`` state's dispatch (technique implementer) had
no retry-budget terminal pre-round-11, so a dispatch-handler failure
infinitely rolled back to ``pending`` and re-dispatched. Round 10 left
it ``retry_policy=None`` deliberately because the existing
``implementation_attempt`` counter is owned by the CG-level
review-retry gate (double-count hazard). The fix is a separate counter:
this column backs the :class:`RetryPolicy` declared on the entry
``implementing`` ``DispatchAction``.

The column lands with default 0 and ``nullable=False`` to match the
other ``*_attempt`` counters on the same table
(``implementation_attempt``). Backfill is trivial since the default
applies to every legacy row.

Forward-only per DEC-036; ``downgrade()`` raises so a stray
``alembic downgrade`` is loud rather than silent. Same dual-mode shape
as 0003 / 0004: in offline (``--sql``) mode the ADD COLUMN renders
unconditionally; in online mode we inspect the live table and skip when
the column is already present (the 0001 baseline's
``MetaData.create_all(checkfirst=True)`` covers the column for fresh
databases since the canonical SQLAlchemy declaration already includes
it post-round-11).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import context, op

revision: str = "0005_entry_dispatch_attempt"
down_revision: str | None = "0004_cg_impl_phase_attempt"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    if context.is_offline_mode():
        op.add_column(
            "entries",
            sa.Column(
                "entry_dispatch_attempt",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
        )
        return
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {col["name"] for col in inspector.get_columns("entries")}
    if "entry_dispatch_attempt" not in columns:
        op.add_column(
            "entries",
            sa.Column(
                "entry_dispatch_attempt",
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
