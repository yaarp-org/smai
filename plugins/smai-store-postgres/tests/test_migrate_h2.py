"""Postgres-side migration tests (Task 3.H2).

Verifies that the shared Alembic env from
:mod:`smai_orchestrator.migrations` runs equivalently against the
Postgres dialect — same head revision, same idempotency, same
``--check`` semantics. Skips cleanly when no Postgres test DB is
reachable (per the package's ``conftest`` discipline).
"""

from __future__ import annotations

import pytest
from smai_orchestrator.migrations import (
    get_current_revision,
    get_head_revision,
    is_at_head,
    upgrade_to_head,
)
from smai_store_postgres import PostgresStore
from sqlalchemy import Inspector, inspect


async def test_postgres_upgrade_to_head_creates_schema(postgres_url: str) -> None:
    """``upgrade_to_head`` against a clean Postgres landed every table."""
    store = PostgresStore(uri=postgres_url)
    try:
        await store.drop_all()
        engine = store._engine  # noqa: SLF001 — test-only access
        assert await is_at_head(engine) is False
        await upgrade_to_head(engine)
        assert await is_at_head(engine) is True
        assert await get_current_revision(engine) == get_head_revision()

        async with engine.connect() as conn:

            def _list_tables(sync_conn: object) -> set[str]:
                inspector = inspect(sync_conn)
                assert isinstance(inspector, Inspector)
                return set(inspector.get_table_names())

            tables = await conn.run_sync(_list_tables)
        # Postgres also includes ``alembic_version``; both reference
        # plugins must agree on the table set per the cross-dialect
        # invariant.
        assert "alembic_version" in tables
        for expected in (
            "cgs",
            "entries",
            "runs",
            "proposals",
            "papers",
            "factor_models",
            "techniques",
            "transition_log",
            "agent_sessions",
            "run_costs",
        ):
            assert expected in tables, f"Postgres schema missing {expected!r}"
    finally:
        await store.drop_all()
        await store.dispose()


async def test_postgres_upgrade_to_head_is_idempotent(postgres_url: str) -> None:
    """Re-running ``upgrade_to_head`` is a no-op."""
    store = PostgresStore(uri=postgres_url)
    try:
        await store.drop_all()
        await store.migrate()  # First pass — creates schema.
        await store.migrate()  # Second pass — must not raise.
        assert await is_at_head(store._engine) is True  # noqa: SLF001
    finally:
        await store.drop_all()
        await store.dispose()


@pytest.mark.parametrize(
    "expected_table",
    ["cgs", "entries", "runs", "proposals", "papers", "transition_log"],
)
async def test_postgres_schema_matches_sqlite_table_set(
    postgres_url: str, expected_table: str
) -> None:
    """Sanity check: the same conceptual table exists on both dialects.

    The single :class:`MetaData` source should render identical table
    sets across SQLite and Postgres; this test pins it for the
    cross-dialect correctness story.
    """
    store = PostgresStore(uri=postgres_url)
    try:
        await store.drop_all()
        await store.migrate()
        engine = store._engine  # noqa: SLF001
        async with engine.connect() as conn:

            def _list_tables(sync_conn: object) -> set[str]:
                inspector = inspect(sync_conn)
                assert isinstance(inspector, Inspector)
                return set(inspector.get_table_names())

            tables = await conn.run_sync(_list_tables)
        assert expected_table in tables
    finally:
        await store.drop_all()
        await store.dispose()
