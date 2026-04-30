"""Initial schema — captures the v1 schema declared in
:mod:`smai_orchestrator.migrations.metadata`.

Revision ID: 0001_initial_schema
Revises: <none>
Create Date: 2026-04-29

The initial revision delegates to the shared :class:`MetaData`'s
``create_all`` rather than re-declaring every ``op.create_table(...)``.
Two reasons:

1. **Single source of truth.** The :class:`MetaData` declarations in
   ``metadata.py`` are already the canonical schema; the plugin
   conformance suite tests against that shape directly. Re-declaring
   the tables here would create a divergence-risk surface that
   ``check_deps``-style tests would have to police forever.
2. **Dialect-correct emission.** ``create_all`` renders SQLite-flavor
   SQL on aiosqlite and Postgres-flavor SQL on asyncpg from the same
   declarations — exactly what we want from a cross-plugin baseline
   revision.

Future revisions DO use explicit ``op.add_column`` / ``op.create_index``
calls per the standard Alembic pattern; the ``create_all`` shortcut is
specific to the baseline.

Per Task 3.H2 / the implementation plan §3.4 acceptance:
**downgrade is documented but not implemented**. Production deployments
that need to roll back a schema change in v2 do so by restoring from
backup; v3 may revisit. Calling ``downgrade()`` raises
:class:`NotImplementedError` so a stray ``alembic downgrade`` fails loud.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

from smai_orchestrator.migrations.metadata import metadata as target_metadata

# revision identifiers, used by Alembic.
#
# Per Task 3.G2: the canonical OSS schema lives on the ``default``
# branch. The opt-in tenant-aware extension (revision
# ``0002_tenant_aware_schema``) is a separate-root revision on the
# ``tenant_aware`` branch that ``depends_on`` this one — Alembic
# enforces 0001 ships before 0002 when ``tenant_aware@head`` is the
# upgrade target.
revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = ("default",)
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create every table + index declared on the shared :class:`MetaData`.

    Idempotent: ``checkfirst=True`` matches the prior boot-time
    ``metadata.create_all(checkfirst=True)`` semantics from the
    pre-3.H2 plugin code, so re-running against an existing-but-
    unstamped database does not error. (The post-3.H2 path stamps the
    revision after the first successful upgrade so subsequent boots
    skip ``upgrade`` entirely; the ``checkfirst`` is the safety net
    for first-time-stamped databases that were created before 3.H2.)
    """
    bind = op.get_bind()
    target_metadata.create_all(bind, checkfirst=True)


def downgrade() -> None:
    """Down-migrations are documented, NOT implemented (Task 3.H2 / DEC-036).

    Per the design-time deferral in the implementation plan:
    rollbacks are out of scope for v2; production deployments roll
    back via backup restore. Calling this raises so a stray
    ``alembic downgrade`` is loud rather than silent.
    """
    raise NotImplementedError(
        "smai_orchestrator.migrations: down-migrations are not implemented in v2 "
        "(documented design-time deferral per Task 3.H2). Roll back by restoring "
        "from a database backup."
    )
