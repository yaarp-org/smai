"""Opt-in tenant-aware schema extension — adds ``tenant_id`` columns +
indexes to every pipeline-tracking table per `07-plugin-interfaces.md`
§5.5 / §5.6.8 (Task 3.G2).

Revision ID: 0002_tenant_aware_schema
Revises: <separate root, depends_on=0001_initial_schema>
Create Date: 2026-04-29

Branch shape per Task 3.G2: this revision is a **separate-root** branch
labeled ``tenant_aware`` with ``depends_on="0001_initial_schema"``. That
keeps it OUT of the default ``upgrade head`` chain — operators must
target it explicitly via ``upgrade tenant_aware@head`` (which Alembic
walks from the depends_on dependency, applying 0001 first then 0002).

Two consumer paths:

* :class:`PostgresStore(tenant_aware=True)` — its ``migrate()`` calls
  :func:`smai_orchestrator.migrations.upgrade_to_head(engine,
  branch="tenant_aware")` automatically.
* ``smai migrate --upgrade-to=tenant_aware`` — the CLI verb operators
  invoke before flipping ``tenant_aware=True`` on a long-running
  Postgres deployment.

Default OSS ``smai migrate`` (no flag) targets ``default@head`` and
**never** touches this revision.

Schema shape: nullable ``tenant_id VARCHAR(64)`` on every
pipeline-tracking table (cgs, entries, runs, proposals, papers) plus a
composite index ``(tenant_id, created_at, <pk>)`` on each — anchors the
``ROW_NUMBER() OVER (PARTITION BY tenant_id ORDER BY created_at, id)``
window-function ordering the plugin's tenant-aware
``_paginate_predicate`` uses (`07` §5.5 / §5.6.8). Operators populating
``tenant_id`` is their concern — the OSS plugin treats NULL as a single
``"<no-tenant>"`` partition.

Forward-only per the design-time deferral (DEC-036). Calling
``downgrade()`` raises so a stray ``alembic downgrade`` is loud.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
#
# Separate-root branch (``down_revision=None``) labeled
# ``tenant_aware``; ``depends_on`` makes Alembic require 0001 to be
# applied first whenever ``tenant_aware@head`` is the upgrade target.
revision: str = "0002_tenant_aware_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = ("tenant_aware",)
depends_on: str | Sequence[str] | None = "0001_initial_schema"


# Per-table primary-key column name. ``papers`` uses ``arxiv_id`` per
# `07-plugin-interfaces.md` §5.3 (paper records do not get a separate
# ULID); the rest use ``id``.
_PIPELINE_TABLES: dict[str, str] = {
    "cgs": "id",
    "entries": "id",
    "runs": "id",
    "proposals": "id",
    "papers": "arxiv_id",
}


def upgrade() -> None:
    """Add ``tenant_id`` + composite index to every pipeline-tracking table.

    Idempotent in spirit (Alembic's ``alembic_version`` short-circuits
    re-runs at the engine level); this implementation does NOT use
    ``IF NOT EXISTS`` since Alembic's stamping discipline handles the
    re-run case at the revision level.
    """
    for table_name, pk_column in _PIPELINE_TABLES.items():
        op.add_column(
            table_name,
            sa.Column("tenant_id", sa.String(64), nullable=True),
        )
        op.create_index(
            f"ix_{table_name}_tenant_created_{pk_column}",
            table_name,
            ["tenant_id", "created_at", pk_column],
        )


def downgrade() -> None:
    """Down-migrations are documented, NOT implemented (DEC-036).

    Forward-only per the Task 3.H2 design-time deferral. Operators
    rolling back a tenant-aware extension restore from a backup taken
    before the 0002 upgrade; flipping ``PostgresStore(tenant_aware=
    False)`` on an already-extended schema is operationally fine (the
    column / index sit unused) but the Alembic version row pins
    ``0002_tenant_aware_schema``, so ``smai migrate --check`` against
    the default branch flags drift.
    """
    raise NotImplementedError(
        "smai_orchestrator.migrations: down-migrations are not implemented in v2 "
        "(documented design-time deferral per Task 3.H2 / Task 3.G2). Roll back "
        "by restoring from a database backup."
    )
