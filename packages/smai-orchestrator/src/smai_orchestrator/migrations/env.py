"""Alembic environment script for the SMAI metadata schema.

Per Task 3.H2 (DEC-036's Alembic deferral cashed in). The env supports
two execution modes:

* **Embedded / online**. The plugin's :meth:`MetadataStore.migrate`
  enters an :class:`~sqlalchemy.ext.asyncio.AsyncEngine` connection,
  hands it to Alembic via ``config.attributes["connection"]``, and
  runs the migrations synchronously on the bound connection (the
  outer plugin layer is responsible for ``conn.run_sync(...)``). This
  is the path :class:`SqliteStore` and :class:`PostgresStore` use at
  boot and that ``smai migrate`` uses against a configured
  ``MetadataStore``.
* **Offline / SQL emit**. ``smai migrate --dry-run`` configures
  Alembic with ``as_sql=True`` and a target URL; Alembic emits the SQL
  to stdout without touching a database. Useful for review before a
  prod deployment.

The ``target_metadata`` is the shared :class:`MetaData` from
:mod:`smai_orchestrator.migrations.metadata`. Both plugins (and any
future :class:`MetadataStore` plugin) share one schema-of-record so
revisions land once and apply to every dialect.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from alembic import context

from smai_orchestrator.migrations.metadata import metadata as target_metadata

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection


config = context.config


def run_migrations_offline() -> None:
    """Render migration SQL without binding to a live connection.

    Honors the ``--sql`` Alembic flag (set programmatically by
    ``smai migrate --dry-run``). The dialect is inferred from the
    configured URL — SQLite and Postgres render different SQL for the
    same schema declarations (e.g., ``TIMESTAMP WITH TIME ZONE`` vs.
    SQLite's text-encoded datetime), so callers pass an explicit URL
    when they want dialect-correct emission.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Don't compare types — the schema is dialect-agnostic at the
        # MetaData layer (e.g., ``DateTime(timezone=True)`` renders
        # differently per dialect; type-comparison would flag every
        # such cell as drift on non-baseline dialects).
        compare_type=False,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Execute the pending migrations on ``connection`` (sync DBAPI)."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=False,
        # SQLite ``ALTER TABLE`` is limited; render in batch mode so
        # any future ALTER-shaped revision works against SQLite. The
        # initial revision is pure ``CREATE TABLE`` so this is a no-op
        # today, but later revisions need it.
        render_as_batch=connection.dialect.name == "sqlite",
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against an injected sync :class:`Connection`.

    The async-engine path is at the outer caller (``runner.py``):
    ``await engine.run_sync(do_run_migrations)``. Alembic's
    :class:`~alembic.runtime.environment.EnvironmentContext` is
    designed against the sync DBAPI, so we receive a sync connection
    here and the async wiring stays out of this module.
    """
    connection = config.attributes.get("connection")
    if connection is None:
        raise RuntimeError(
            "smai_orchestrator.migrations.env: no connection in config.attributes — "
            "run via smai_orchestrator.migrations.runner (online) or "
            "with --sql (offline)."
        )
    do_run_migrations(cast("Connection", connection))


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
