"""Re-export shim for the shared SMAI metadata schema.

Per Task 3.H2 (DEC-036's Alembic deferral): the canonical schema
lives at :mod:`smai_orchestrator.migrations.metadata`. Both the SQLite
reference plugin (this package) and the Postgres plugin import the
same :class:`MetaData` / :class:`Table` set from there.

This module preserves the prior import surface (``from
smai_store_sqlite._schema import metadata, cgs_table, ...``) so any
external caller that pinned that path keeps working. New code should
import from :mod:`smai_orchestrator.migrations` directly.
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

__all__ = [
    "ENTITY_PK_COLUMN",
    "ENTITY_TABLE",
    "RETENTION_TABLES",
    "agent_sessions_table",
    "cgs_table",
    "entries_table",
    "factor_models_table",
    "metadata",
    "papers_table",
    "proposals_table",
    "run_costs_table",
    "runs_table",
    "techniques_table",
    "transition_log_table",
]
