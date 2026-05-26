"""Unit tests for :mod:`smai_orchestrator.migrations` (Task 3.H2).

Covers:

* :func:`get_head_revision` returns the single shipped revision id.
* :func:`upgrade_to_head` against an empty SQLite ``:memory:`` engine
  creates every table in :data:`metadata` and stamps ``alembic_version``.
* Re-running ``upgrade_to_head`` is a no-op (boot-time idempotency).
* :func:`is_at_head` agrees with the database state (False before
  upgrade; True after).
* :func:`render_offline_sql` emits dialect-correct SQL for both
  SQLite and Postgres and references every table on the shared
  :class:`MetaData`.
* The initial revision's ``downgrade()`` raises
  :class:`NotImplementedError` per the documented forward-only policy.
* :func:`prune_retention_tables` deletes rows older than the cutoff.

Filename uses the ``_h2_`` prefix per the brief's cross-package
fixture-hygiene rule.
"""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType

import pytest
from smai_orchestrator.migrations import (
    DEFAULT_RETENTION_DAYS,
    RETENTION_TABLES,
    get_current_revision,
    get_head_revision,
    is_at_head,
    metadata,
    prune_retention_tables,
    render_offline_sql,
    transition_log_table,
    upgrade_to_head,
)
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import create_async_engine


def _load_initial_revision_module() -> ModuleType:
    """Load the initial Alembic revision file for direct introspection.

    Alembic revision filenames start with a digit (``0001_*.py``), so
    they aren't importable as plain Python modules — we go through
    :func:`importlib.util.spec_from_file_location`.
    """
    versions_dir = Path(__file__).resolve().parents[1] / (
        "src/smai_orchestrator/migrations/versions"
    )
    revision_path = versions_dir / "0001_initial_schema.py"
    spec = importlib.util.spec_from_file_location("h2_initial_schema", revision_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_head_revision_is_latest_default_revision() -> None:
    """The shipped default-branch head walks to the latest revision.

    Round 10 added ``0004_cg_impl_phase_attempt`` downstream of
    ``0003_cg_symbolic_id`` to add the CG record's
    ``implementation_phase_attempt`` counter. Round 11 added
    ``0005_entry_dispatch_attempt`` downstream of that to add the
    ``EntryRecord.entry_dispatch_attempt`` counter. Agent-refactor
    Step 4 sub-PR B (D2) added ``0006_agent_session_handle``
    downstream of that for the ``agent_sessions.compute_job_handle``
    cross-reference column. The tenant_aware branch (``0002``) sits on
    a separate root and does not affect the default head.
    """
    assert get_head_revision() == "0006_agent_session_handle"


async def test_upgrade_to_head_creates_every_table() -> None:
    """A fresh DB lands every table from ``metadata`` plus ``alembic_version``."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        # Empty DB: not at head.
        assert await is_at_head(engine) is False
        await upgrade_to_head(engine)
        assert await is_at_head(engine) is True

        async with engine.connect() as conn:

            def _list_tables(sync_conn: object) -> set[str]:
                from sqlalchemy import Inspector
                from sqlalchemy import inspect as _inspect

                inspector = _inspect(sync_conn)
                assert isinstance(inspector, Inspector)
                return set(inspector.get_table_names())

            tables = await conn.run_sync(_list_tables)

        expected = {t.name for t in metadata.tables.values()} | {"alembic_version"}
        assert tables == expected
    finally:
        await engine.dispose()


async def test_upgrade_to_head_is_idempotent() -> None:
    """Running upgrade twice is a no-op."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        await upgrade_to_head(engine)
        # Second pass — must not raise (Alembic skips when already at head).
        await upgrade_to_head(engine)
        # And the schema is still reachable.
        async with engine.connect() as conn:
            await conn.execute(select(transition_log_table))
    finally:
        await engine.dispose()


async def test_get_current_revision_against_empty_db() -> None:
    """Empty DB has no ``alembic_version`` row → ``None``."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        assert await get_current_revision(engine) is None
    finally:
        await engine.dispose()


def test_render_offline_sqlite_emits_create_table_statements() -> None:
    """SQLite offline render references every table."""
    sql = render_offline_sql("sqlite:///:memory:")
    for table in metadata.tables:
        assert f"CREATE TABLE {table}" in sql, f"missing CREATE TABLE for {table!r}"


def test_render_offline_postgres_uses_timezone_aware_timestamps() -> None:
    """Postgres render emits ``TIMESTAMP WITH TIME ZONE`` for ``DateTime(timezone=True)``."""
    sql = render_offline_sql("postgresql://u@h/db")
    assert "TIMESTAMP WITH TIME ZONE" in sql


def test_initial_revision_downgrade_raises() -> None:
    """The forward-only policy: ``downgrade()`` raises a clear error."""
    revision = _load_initial_revision_module()
    with pytest.raises(NotImplementedError, match="not implemented"):
        revision.downgrade()


def test_initial_revision_metadata_fields() -> None:
    """The revision id matches the head and ``down_revision`` is None."""
    revision = _load_initial_revision_module()
    assert revision.revision == "0001_initial_schema"
    assert revision.down_revision is None


def _load_revision_module(filename: str, mod_name: str) -> ModuleType:
    """Load an Alembic revision file by name for direct introspection.

    Generic counterpart to :func:`_load_initial_revision_module` —
    revision filenames start with a digit so they are not importable as
    plain modules.
    """
    versions_dir = Path(__file__).resolve().parents[1] / (
        "src/smai_orchestrator/migrations/versions"
    )
    spec = importlib.util.spec_from_file_location(mod_name, versions_dir / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_revision_0005_entry_dispatch_attempt_metadata_fields() -> None:
    """Round-11 revision 0005 chains off 0004 and is forward-only.

    Mirrors the round-10 ``0004_cg_impl_phase_attempt`` shape: a
    single ADD COLUMN, chained off the prior default-branch head, with
    a ``downgrade()`` that raises per the forward-only policy (DEC-036).
    """
    revision = _load_revision_module("0005_entry_dispatch_attempt.py", "h2_rev_0005")
    assert revision.revision == "0005_entry_dispatch_attempt"
    assert revision.down_revision == "0004_cg_impl_phase_attempt"
    with pytest.raises(NotImplementedError, match="not implemented"):
        revision.downgrade()


async def test_upgrade_to_head_lands_entry_dispatch_attempt_column() -> None:
    """After ``upgrade_to_head`` the ``entries`` table carries the
    round-11 ``entry_dispatch_attempt`` column (the ``RetryPolicy``
    counter for the entry-spec ``implementing`` dispatch action)."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        await upgrade_to_head(engine)

        async with engine.connect() as conn:

            def _entry_columns(sync_conn: object) -> set[str]:
                from sqlalchemy import Inspector
                from sqlalchemy import inspect as _inspect

                inspector = _inspect(sync_conn)
                assert isinstance(inspector, Inspector)
                return {col["name"] for col in inspector.get_columns("entries")}

            columns = await conn.run_sync(_entry_columns)

        assert "entry_dispatch_attempt" in columns
    finally:
        await engine.dispose()


async def test_prune_retention_tables_deletes_old_rows() -> None:
    """Rows older than ``cutoff_days`` are deleted; newer rows remain."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        await upgrade_to_head(engine)
        now = datetime(2026, 4, 29, tzinfo=UTC)
        # Seed two rows: one old (200 days), one recent (10 days).
        async with engine.begin() as conn:
            await conn.execute(
                insert(transition_log_table).values(
                    entity_kind="cg",
                    entity_id="cg_old",
                    from_state="draft",
                    to_state="implementing",
                    occurred_at=now - timedelta(days=200),
                )
            )
            await conn.execute(
                insert(transition_log_table).values(
                    entity_kind="cg",
                    entity_id="cg_new",
                    from_state="draft",
                    to_state="implementing",
                    occurred_at=now - timedelta(days=10),
                )
            )

        # Default retention for transition_log is 90 days; the old row
        # should be deleted, the recent row kept.
        deleted = await prune_retention_tables(engine, now=now)
        assert deleted["transition_log"] == 1
        assert deleted["agent_sessions"] == 0
        assert deleted["run_costs"] == 0

        async with engine.connect() as conn:
            remaining = (
                (await conn.execute(select(transition_log_table.c.entity_id))).scalars().all()
            )
        assert remaining == ["cg_new"]
    finally:
        await engine.dispose()


async def test_prune_zero_days_disables_retention() -> None:
    """A 0-day window keeps all rows."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        await upgrade_to_head(engine)
        now = datetime(2026, 4, 29, tzinfo=UTC)
        async with engine.begin() as conn:
            await conn.execute(
                insert(transition_log_table).values(
                    entity_kind="cg",
                    entity_id="cg_ancient",
                    from_state="draft",
                    to_state="implementing",
                    occurred_at=now - timedelta(days=5_000),
                )
            )
        deleted = await prune_retention_tables(
            engine,
            retention_days={"transition_log": 0},
            now=now,
        )
        assert deleted["transition_log"] == 0
    finally:
        await engine.dispose()


async def test_prune_unknown_table_name_raises() -> None:
    """Operator typo in retention_policies surfaces as ValueError."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        await upgrade_to_head(engine)
        with pytest.raises(ValueError, match="unknown retention table"):
            await prune_retention_tables(
                engine,
                retention_days={"transitionlog_typo": 30},  # missing underscore
            )
    finally:
        await engine.dispose()


def test_default_retention_days_covers_all_retention_tables() -> None:
    """Every retention-targeted table has a default."""
    assert set(DEFAULT_RETENTION_DAYS) == set(RETENTION_TABLES)
