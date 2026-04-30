"""Shared SQLAlchemy schema + Alembic env for SMAI v2 metadata storage.

Per Task 3.H2 / DEC-036: the schema and the Alembic env that drives it
live here so every :class:`MetadataStore` plugin imports from one
place. Pre-3.H2 the schema lived inside ``smai-store-sqlite`` and the
Postgres plugin cross-imported it; that lift cleared with this task.

Public surface:

* :data:`metadata` and the per-table :class:`Table` objects (re-exported
  from :mod:`.metadata`).
* :func:`upgrade_to_head` / :func:`is_at_head` /
  :func:`render_offline_sql` / :func:`prune_retention_tables` (the
  programmatic Alembic helpers from :mod:`.runner`).

Operational notes (rollback policy, retention defaults) live in
``MIGRATIONS.md`` next to this file.
"""

from __future__ import annotations

from smai_orchestrator.migrations.metadata import (
    ENTITY_PK_COLUMN,
    ENTITY_TABLE,
    RETENTION_TABLES,
    agent_sessions_table,
    cgs_table,
    entries_table,
    factor_models_table,
    metadata,
    papers_table,
    proposals_table,
    run_costs_table,
    runs_table,
    techniques_table,
    transition_log_table,
)
from smai_orchestrator.migrations.runner import (
    DEFAULT_BRANCH,
    DEFAULT_RETENTION_DAYS,
    TENANT_AWARE_BRANCH,
    get_current_revision,
    get_head_revision,
    is_at_head,
    prune_retention_tables,
    render_offline_sql,
    upgrade_to_head,
)

__all__ = [
    "DEFAULT_BRANCH",
    "DEFAULT_RETENTION_DAYS",
    "ENTITY_PK_COLUMN",
    "ENTITY_TABLE",
    "RETENTION_TABLES",
    "TENANT_AWARE_BRANCH",
    "agent_sessions_table",
    "cgs_table",
    "entries_table",
    "factor_models_table",
    "get_current_revision",
    "get_head_revision",
    "is_at_head",
    "metadata",
    "papers_table",
    "proposals_table",
    "prune_retention_tables",
    "render_offline_sql",
    "run_costs_table",
    "runs_table",
    "techniques_table",
    "transition_log_table",
    "upgrade_to_head",
]
