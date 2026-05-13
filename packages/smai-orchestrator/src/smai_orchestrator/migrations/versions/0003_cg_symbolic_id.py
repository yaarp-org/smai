"""Add ``symbolic_id`` column to the ``cgs`` table (round-9 fix A).

Revision ID: 0003_cg_symbolic_id
Revises: 0001_initial_schema
Create Date: 2026-05-13

Round-9 fix A: pre-round-9 the CG record's ``id`` was constructed as
``f"{proposal_id}--{draft_cg_id}"`` (~80 chars in production), which
busted through the 64-char id-format cap and crashed
:class:`ComparisonGroupRecord` Pydantic validation at registration time
— wedging the proposal in ``designed`` until the registration retry-
exhausted terminal gate fired. Round-9 generates the primary CG id as
a fresh ULID-shaped string instead and preserves the planner's
human-readable draft id on a new ``symbolic_id`` column for listings.

The column is nullable so that backfill is trivial — pre-0003 rows
keep ``NULL``; post-0003 rows always populate it via the registration
handler. Length 128 (not 64) to admit planner symbolic names that
would exceed the id-format cap.

Forward-only per DEC-036; ``downgrade()`` raises so a stray
``alembic downgrade`` is loud rather than silent.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import context, op

revision: str = "0003_cg_symbolic_id"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # The 0001 baseline uses ``MetaData.create_all(checkfirst=True)`` against
    # the canonical SQLAlchemy declarations, which already include
    # ``symbolic_id`` post-round-9 — so on fresh databases the column lands
    # in step with 0001 and 0003 would error on the duplicate add. Inspect
    # the live table first and skip the ``ADD COLUMN`` when the column is
    # already present (the legacy-DB upgrade path is the only one that
    # needs the explicit add). Mirrors the spirit of 0001's
    # ``checkfirst=True``.
    #
    # In offline (``--sql``) mode :func:`op.get_bind` returns a MockConnection
    # that the reflective inspector can't introspect; emit the ADD COLUMN
    # unconditionally so the rendered SQL covers the legacy upgrade path
    # (this matches the spirit of 0001's offline-rendered CREATE TABLE
    # statements, which also represent the legacy upgrade path).
    if context.is_offline_mode():
        op.add_column(
            "cgs",
            sa.Column("symbolic_id", sa.String(128), nullable=True),
        )
        return
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {col["name"] for col in inspector.get_columns("cgs")}
    if "symbolic_id" not in columns:
        op.add_column(
            "cgs",
            sa.Column("symbolic_id", sa.String(128), nullable=True),
        )


def downgrade() -> None:
    raise NotImplementedError(
        "smai_orchestrator.migrations: down-migrations are not implemented in v2 "
        "(documented design-time deferral per Task 3.H2 / DEC-036). Roll back by "
        "restoring from a database backup."
    )
