"""Programmatic Alembic helpers used by :class:`MetadataStore` plugins
and by the ``smai migrate`` CLI verb.

Per Task 3.H2 / DEC-036: Alembic runs against the shared
:class:`MetaData` declared in :mod:`.metadata`. The plugin's
:meth:`MetadataStore.migrate` calls :func:`upgrade_to_head` against its
:class:`~sqlalchemy.ext.asyncio.AsyncEngine`; the CLI calls the same
plus :func:`is_at_head` (for ``--check``) and :func:`render_offline_sql`
(for ``--dry-run``).

Boot-time idempotency. The standard Alembic pattern — ``upgrade head``
against an already-head schema is a no-op — holds. The first call
against a fresh DB creates ``alembic_version`` + every table; the
second call observes ``alembic_version`` at head and exits cleanly.
The first call against an unstamped pre-3.H2 schema (i.e., one that
was created with ``metadata.create_all`` before this task landed)
relies on the initial revision's ``checkfirst=True`` for safety —
``CREATE TABLE IF NOT EXISTS`` semantics keep the upgrade non-
destructive, and the version table is stamped at the head after the
first successful pass.

Async wiring. Alembic's :class:`~alembic.runtime.environment.EnvironmentContext`
operates on a sync DBAPI connection; we drive it from
:class:`~sqlalchemy.ext.asyncio.AsyncConnection` via
:meth:`AsyncConnection.run_sync`. The async/sync seam stays inside this
module — plugins and CLI just call the public ``async def`` helpers.
"""

from __future__ import annotations

import contextlib
import io
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import delete

from smai_orchestrator.migrations.metadata import RETENTION_TABLES

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection
    from sqlalchemy.ext.asyncio import AsyncEngine


# Default retention windows (DEC-033 #1 / #2). Conservative by design:
# transition_log is the audit trail (90 days = enough for incident
# investigation past one quarter); agent_sessions is the cost ledger
# at agent-invocation granularity (180 days = covers two quarters of
# spend retros); run_costs is the cost ledger at GPU-job granularity
# (365 days = annual cost review). Operators override per deployment
# via :attr:`EngineConfig.retention_policies`.
DEFAULT_RETENTION_DAYS: dict[str, int] = {
    "transition_log": 90,
    "agent_sessions": 180,
    "run_costs": 365,
}


def _migrations_dir() -> Path:
    """Resolve the directory containing :mod:`env.py` + ``versions/``."""
    return Path(__file__).resolve().parent


def _build_config(*, url: str | None = None) -> Config:
    """Construct an Alembic :class:`Config` pointed at this package.

    ``url`` is set on the returned :class:`Config` when provided so the
    offline / ``--sql`` path can render dialect-correct SQL without
    instantiating an engine. Online callers leave it ``None`` and pass
    a connection through ``cfg.attributes["connection"]``.
    """
    here = _migrations_dir()
    cfg = Config(str(here / "alembic.ini"))
    cfg.set_main_option("script_location", str(here))
    if url is not None:
        cfg.set_main_option("sqlalchemy.url", url)
    return cfg


def _do_upgrade(connection: Connection, cfg: Config) -> None:
    """Sync helper used inside :meth:`AsyncConnection.run_sync`."""
    cfg.attributes["connection"] = connection
    command.upgrade(cfg, "head")


async def upgrade_to_head(engine: AsyncEngine) -> None:
    """Run ``alembic upgrade head`` against ``engine``.

    Idempotent: the upgrade against a head schema is a no-op (Alembic
    consults ``alembic_version`` and skips revisions whose
    ``down_revision`` chain is already applied).
    """
    cfg = _build_config()
    async with engine.begin() as conn:
        await conn.run_sync(_do_upgrade, cfg)


def get_head_revision() -> str:
    """Return the ``rev_id`` of the most recent revision on this branch."""
    cfg = _build_config()
    script = ScriptDirectory.from_config(cfg)
    head = script.get_current_head()
    if head is None:
        raise RuntimeError(
            "smai_orchestrator.migrations: no Alembic revisions found under "
            f"{_migrations_dir() / 'versions'}."
        )
    return head


def _do_get_current(connection: Connection) -> str | None:
    ctx = MigrationContext.configure(connection)
    return ctx.get_current_revision()


async def get_current_revision(engine: AsyncEngine) -> str | None:
    """Return the database's ``alembic_version`` row, or ``None`` if absent."""
    async with engine.connect() as conn:
        return await conn.run_sync(_do_get_current)


async def is_at_head(engine: AsyncEngine) -> bool:
    """``True`` iff the database's revision matches the migrations' head."""
    current = await get_current_revision(engine)
    if current is None:
        return False
    return current == get_head_revision()


def render_offline_sql(url: str, *, target: str = "head") -> str:
    """Render ``alembic upgrade <target>`` SQL for ``url`` without executing.

    Honors the dialect inferred from the URL — SQLite emits SQLite SQL,
    Postgres (``postgresql://...`` or ``postgresql+asyncpg://...``)
    emits Postgres SQL.

    The ``target`` defaults to ``"head"``; callers can pass ``"base:head"``
    for the full from-scratch upgrade emission. The Alembic ``--sql``
    flag is what powers ``smai migrate --dry-run``.
    """
    cfg = _build_config(url=_strip_async_driver(url))
    buf = io.StringIO()
    # Alembic emits offline SQL to ``stdout`` by default; redirect it
    # into ``buf`` so the CLI can post-process / pipe the result. Using
    # ``redirect_stdout`` keeps us off Alembic's private
    # ``config.output_buffer`` attribute, which has shifted across
    # minor versions.
    with contextlib.redirect_stdout(buf):
        command.upgrade(cfg, target, sql=True)
    return buf.getvalue()


def _strip_async_driver(url: str) -> str:
    """Convert async-driver URL to its sync equivalent for offline SQL.

    Alembic's offline / ``--sql`` mode does not actually open a
    connection, but it does parse the URL through SQLAlchemy's URL
    parser to pick a dialect. The async drivers (``+asyncpg``,
    ``+aiosqlite``) parse fine for dialect inference, but stripping
    keeps the dialect picker on the simplest mainline path.
    """
    return (
        url.replace("postgresql+asyncpg://", "postgresql://")
        .replace("sqlite+aiosqlite:///", "sqlite:///")
        .replace("postgresql+psycopg://", "postgresql://")
    )


def _do_prune(
    connection: Connection,
    *,
    cutoff: datetime,
    table_names: list[str] | None,
) -> dict[str, int]:
    """Delete rows older than ``cutoff`` per :data:`RETENTION_TABLES`."""
    deleted: dict[str, int] = {}
    targets = table_names if table_names is not None else list(RETENTION_TABLES.keys())
    for name in targets:
        if name not in RETENTION_TABLES:
            raise ValueError(
                f"smai_orchestrator.migrations: unknown retention table {name!r}; "
                f"allowed: {sorted(RETENTION_TABLES)}"
            )
        table, ts_col = RETENTION_TABLES[name]
        result = connection.execute(delete(table).where(ts_col < cutoff))
        deleted[name] = result.rowcount or 0
    return deleted


async def prune_retention_tables(
    engine: AsyncEngine,
    *,
    retention_days: dict[str, int] | None = None,
    now: datetime | None = None,
) -> dict[str, int]:
    """Delete rows older than the per-table retention window.

    ``retention_days`` maps table name → window in days. Tables absent
    from the map fall through to :data:`DEFAULT_RETENTION_DAYS`. Set a
    table's value to ``0`` to skip it (interpreted as "no retention
    limit"); the prune pass leaves every row in place.

    Returns the deletion counts by table. Callers can log / surface
    the totals; the function itself prints nothing.

    Per DEC-033 retention guidance: never auto-prune at boot; this is
    a deliberate operator action, fired via ``smai migrate --prune``.
    """
    policy = dict(DEFAULT_RETENTION_DAYS)
    if retention_days is not None:
        policy.update(retention_days)
    when = now if now is not None else datetime.now(UTC)
    deleted: dict[str, int] = {}
    for table_name, days in policy.items():
        if table_name not in RETENTION_TABLES:
            # Operator misconfiguration in retention_policies — surface
            # the typo rather than silently skipping.
            raise ValueError(
                f"smai_orchestrator.migrations: unknown retention table "
                f"{table_name!r}; allowed: {sorted(RETENTION_TABLES)}"
            )
        if days <= 0:
            deleted[table_name] = 0
            continue
        cutoff = when - timedelta(days=days)
        async with engine.begin() as conn:
            result = await conn.run_sync(
                _do_prune,
                cutoff=cutoff,
                table_names=[table_name],
            )
        deleted[table_name] = result[table_name]
    return deleted


__all__ = [
    "DEFAULT_RETENTION_DAYS",
    "get_current_revision",
    "get_head_revision",
    "is_at_head",
    "prune_retention_tables",
    "render_offline_sql",
    "upgrade_to_head",
]
