"""Add ``context_kind`` column to ``techniques`` (planner-refactor Step 2).

Revision ID: 0007_context_kind
Revises: 0006_agent_session_handle
Create Date: 2026-05-27

Planner-refactor Step 2: closes
``agent_refactor/upstream_requirements.md`` §1. Adds the
``techniques.context_kind`` column so :class:`TechniqueRef` becomes
self-describing (paper_extract / proposal / reviewer_attested / standard)
instead of relying on the legacy
``_resolve_context_kind``-style inference at the implementer side.

Per D10 (``planner_refactor/notes/07_settled_decisions.md`` D10): NO data
backfill. SMAI is pre-v1; the only existing :class:`TechniqueRef` data is
the project owner's own test fixtures, which were dropped — every new write
populates ``context_kind`` explicitly (the :class:`TechniqueRef`
``_validate_context_kind`` validator enforces it as a required field).

Forward-only per DEC-036; ``downgrade()`` raises so a stray
``alembic downgrade`` is loud rather than silent. Dual-mode add-column
pattern matches 0003-0006: in offline (``--sql``) mode the ADD COLUMN
renders unconditionally; in online mode we inspect the live table and
skip when the column is already present (the 0001 baseline's
``MetaData.create_all(checkfirst=True)`` covers the column for fresh
databases once the canonical SQLAlchemy declaration in
``smai_orchestrator.migrations.metadata`` includes it).

Closed-v1 enum (``paper_extract | proposal | reviewer_attested | standard``)
stored as ``String(32)``; mirrors the storage discipline applied to
``fidelity_anchor_kind`` (also ``String(32)``).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import context, op

revision: str = "0007_context_kind"
down_revision: str | None = "0006_agent_session_handle"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    if context.is_offline_mode():
        op.add_column(
            "techniques",
            sa.Column("context_kind", sa.String(32), nullable=True),
        )
        return
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {col["name"] for col in inspector.get_columns("techniques")}
    if "context_kind" not in columns:
        op.add_column(
            "techniques",
            sa.Column("context_kind", sa.String(32), nullable=True),
        )


def downgrade() -> None:
    raise NotImplementedError(
        "smai_orchestrator.migrations: down-migrations are not implemented in v2 "
        "(documented design-time deferral per Task 3.H2 / DEC-036). Roll back by "
        "restoring from a database backup."
    )
