"""End-to-end CLI tests for ``smai migrate`` (Task 3.H2).

Exercises:

* Plain ``smai migrate`` upgrades an empty SQLite file to head.
* Re-running ``smai migrate`` against the head DB is idempotent.
* ``smai migrate --check`` exits 0 on a current DB and 1 on an empty
  one.
* ``smai migrate --dry-run`` prints the SQL Alembic would emit and
  does NOT touch the DB.
* ``smai migrate --prune`` deletes rows older than the configured
  retention window.

Each test points the verb at a per-test SQLite file via a ``smai.yaml``
override so the tests don't pollute ``~/.smai/state.db``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from smai_cli.main import app
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import create_async_engine
from typer.testing import CliRunner


def _write_smai_yaml(
    tmp_path: Path,
    *,
    sqlite_path: Path,
    retention: dict[str, int] | None = None,
) -> Path:
    """Write a minimal smai.yaml pointing the metadata store at ``sqlite_path``."""
    cfg: dict[str, object] = {
        "engine": {
            "poll_interval_seconds": 10,
            "worker_count": 1,
            "fair_scheduling": "off",
        },
        "plugins": {
            "llm_provider": "bedrock",
            "metadata_store": "sqlite",
            "artifact_store": "localfs",
            "compute": "localgpu",
            "llm_provider_config": {
                "region": "us-east-1",
                "model_id": "us.anthropic.claude-opus-4-6-v1",
            },
            "metadata_store_config": {
                "uri": f"sqlite+aiosqlite:///{sqlite_path}",
            },
            "artifact_store_config": {},
            "compute_config": {},
        },
        "pipelines": ["smai_cg_execution", "smai_cg_entries"],
    }
    if retention is not None:
        engine_section = cfg["engine"]
        assert isinstance(engine_section, dict)
        engine_section["retention_policies"] = retention
    smai_yaml = tmp_path / "smai.yaml"
    smai_yaml.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return smai_yaml


@pytest.fixture
def smai_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Override ``$SMAI_HOME`` so the verb's filesystem-defaults helper
    points at a per-test root.

    The migrate verb honors :func:`_apply_dev_filesystem_defaults`'s
    sqlite-at-``$SMAI_HOME/state.db`` fallback when ``smai.yaml``
    omits ``metadata_store_config.uri``. Most tests pass an explicit
    URI through ``smai.yaml`` so the fallback isn't exercised, but the
    home override keeps any incidental directory creation contained
    to the temp tree.
    """
    monkeypatch.setenv("SMAI_HOME", str(tmp_path / "smai_home"))
    yield tmp_path


def test_migrate_upgrades_empty_db_to_head(smai_home: Path) -> None:
    sqlite_path = smai_home / "state.db"
    smai_yaml = _write_smai_yaml(smai_home, sqlite_path=sqlite_path)
    runner = CliRunner()
    result = runner.invoke(app, ["migrate", "-c", str(smai_yaml)])
    assert result.exit_code == 0, result.output
    assert "schema upgraded to head" in result.output
    # The exact head revision id moves as the default chain grows; assert
    # a default-branch revision id appears (the tenant_aware branch's
    # revision id has ``tenant_aware`` in it, which we rule out below).
    assert "0006_agent_session_handle" in result.output
    # File now exists and is non-empty.
    assert sqlite_path.exists()
    assert sqlite_path.stat().st_size > 0


def test_migrate_is_idempotent(smai_home: Path) -> None:
    sqlite_path = smai_home / "state.db"
    smai_yaml = _write_smai_yaml(smai_home, sqlite_path=sqlite_path)
    runner = CliRunner()
    first = runner.invoke(app, ["migrate", "-c", str(smai_yaml)])
    assert first.exit_code == 0
    second = runner.invoke(app, ["migrate", "-c", str(smai_yaml)])
    assert second.exit_code == 0
    assert "schema upgraded to head" in second.output


def test_migrate_warns_when_resolved_store_is_in_memory(smai_home: Path) -> None:
    """A smai.yaml with an empty ``metadata_store_config`` resolves to an
    in-memory SQLite database (the `smai init` shape). `smai migrate`
    still succeeds but warns loudly that the effect is transient."""
    cfg = {
        "engine": {"poll_interval_seconds": 10, "worker_count": 1, "fair_scheduling": "off"},
        "plugins": {
            "llm_provider": "bedrock",
            "metadata_store": "sqlite",
            "artifact_store": "localfs",
            "compute": "localgpu",
            "llm_provider_config": {
                "region": "us-east-1",
                "model_id": "us.anthropic.claude-opus-4-6-v1",
            },
            "metadata_store_config": {},
            "artifact_store_config": {},
            "compute_config": {},
        },
        "pipelines": ["smai_cg_execution", "smai_cg_entries"],
    }
    smai_yaml = smai_home / "smai.yaml"
    smai_yaml.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(app, ["migrate", "-c", str(smai_yaml)])
    assert result.exit_code == 0, result.output
    combined = (result.output or "") + (result.stderr or "")
    assert "in-memory" in combined
    assert "no lasting effect" in combined


def test_migrate_check_exits_one_on_empty_db(smai_home: Path) -> None:
    sqlite_path = smai_home / "state.db"
    smai_yaml = _write_smai_yaml(smai_home, sqlite_path=sqlite_path)
    runner = CliRunner()
    result = runner.invoke(app, ["migrate", "--check", "-c", str(smai_yaml)])
    assert result.exit_code == 1
    # Stderr capture: typer.testing folds stderr into result.output by
    # default. The "NOT at head" message lands in either stream
    # depending on Typer/Click version; be tolerant.
    combined = (result.output or "") + (result.stderr or "")
    assert "NOT at head" in combined
    assert "<unstamped>" in combined


def test_migrate_check_exits_zero_after_upgrade(smai_home: Path) -> None:
    sqlite_path = smai_home / "state.db"
    smai_yaml = _write_smai_yaml(smai_home, sqlite_path=sqlite_path)
    runner = CliRunner()
    runner.invoke(app, ["migrate", "-c", str(smai_yaml)])
    result = runner.invoke(app, ["migrate", "--check", "-c", str(smai_yaml)])
    assert result.exit_code == 0, result.output
    assert "schema at head" in result.output


def test_migrate_dry_run_emits_create_statements_and_skips_db(smai_home: Path) -> None:
    sqlite_path = smai_home / "state.db"
    smai_yaml = _write_smai_yaml(smai_home, sqlite_path=sqlite_path)
    runner = CliRunner()
    result = runner.invoke(app, ["migrate", "--dry-run", "-c", str(smai_yaml)])
    assert result.exit_code == 0, result.output
    assert "CREATE TABLE cgs" in result.output
    assert "CREATE TABLE alembic_version" in result.output
    # --dry-run must NOT create the SQLite file.
    assert not sqlite_path.exists()


def test_migrate_flags_are_mutually_exclusive(smai_home: Path) -> None:
    sqlite_path = smai_home / "state.db"
    smai_yaml = _write_smai_yaml(smai_home, sqlite_path=sqlite_path)
    runner = CliRunner()
    result = runner.invoke(app, ["migrate", "--check", "--dry-run", "-c", str(smai_yaml)])
    assert result.exit_code != 0
    combined = (result.output or "") + (result.stderr or "")
    assert "mutually exclusive" in combined


def test_migrate_upgrade_to_tenant_aware_applies_0002(smai_home: Path) -> None:
    """``smai migrate --upgrade-to=tenant_aware`` runs the opt-in 0002
    revision (Task 3.G2 / `07` §5.5 / §5.6.8).

    Default ``smai migrate`` (no flag) targets ``default@head`` and
    stamps the DB at ``0001_initial_schema``; the explicit ``--upgrade-
    to=tenant_aware`` walks the depends_on chain (0001 first, then
    0002) and stamps at ``0002_tenant_aware_schema``. The schema then
    carries a ``tenant_id`` column on every pipeline-tracking table.
    """
    sqlite_path = smai_home / "state.db"
    smai_yaml = _write_smai_yaml(smai_home, sqlite_path=sqlite_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["migrate", "--upgrade-to", "tenant_aware", "-c", str(smai_yaml)],
    )
    assert result.exit_code == 0, result.output
    assert "tenant_aware@head" in result.output
    assert "0002_tenant_aware_schema" in result.output

    # Verify the tenant_id column landed on a representative table.
    async def _check() -> bool:
        engine = create_async_engine(f"sqlite+aiosqlite:///{sqlite_path}")
        try:
            from sqlalchemy import text

            async with engine.connect() as conn:
                result_rows = await conn.execute(text("PRAGMA table_info(proposals)"))
                columns = [row[1] for row in result_rows.all()]
                return "tenant_id" in columns
        finally:
            await engine.dispose()

    assert asyncio.run(_check()), "tenant_id column missing after --upgrade-to=tenant_aware"


def test_migrate_default_does_not_apply_tenant_aware(smai_home: Path) -> None:
    """Default ``smai migrate`` (no ``--upgrade-to``) targets only the
    ``default`` branch — it must NOT apply the opt-in 0002 revision.

    Regression guard for the branch-selection wiring: before Task 3.G2
    the canonical OSS schema was the only chain in the migrations env;
    after 3.G2 there are two branches, and the default upgrade must not
    accidentally pick up the tenant_aware extension.
    """
    sqlite_path = smai_home / "state.db"
    smai_yaml = _write_smai_yaml(smai_home, sqlite_path=sqlite_path)
    runner = CliRunner()
    result = runner.invoke(app, ["migrate", "-c", str(smai_yaml)])
    assert result.exit_code == 0, result.output
    # Default-branch head (agent-refactor Step 4 sub-PR B grew the chain
    # to 0006); the tenant_aware branch revision id sits on the side
    # branch and is NOT picked up by a no-flag ``smai migrate``.
    assert "0006_agent_session_handle" in result.output
    assert "tenant_aware" not in result.output

    async def _check() -> bool:
        engine = create_async_engine(f"sqlite+aiosqlite:///{sqlite_path}")
        try:
            from sqlalchemy import text

            async with engine.connect() as conn:
                result_rows = await conn.execute(text("PRAGMA table_info(proposals)"))
                columns = [row[1] for row in result_rows.all()]
                return "tenant_id" in columns
        finally:
            await engine.dispose()

    assert not asyncio.run(_check()), (
        "tenant_id column unexpectedly present after default migrate — "
        "the tenant_aware branch leaked into the default upgrade chain"
    )


def test_migrate_prune_deletes_old_rows(smai_home: Path) -> None:
    sqlite_path = smai_home / "state.db"
    # Tighten the retention window to 1 day so the seeded "100 days
    # old" row is firmly out of policy.
    smai_yaml = _write_smai_yaml(
        smai_home,
        sqlite_path=sqlite_path,
        retention={"transition_log": 1},
    )
    runner = CliRunner()
    runner.invoke(app, ["migrate", "-c", str(smai_yaml)])

    # Seed an old row directly via SQLAlchemy; the CLI is what we're
    # testing, not the seed path.
    async def _seed() -> None:
        from smai_orchestrator.migrations import transition_log_table

        engine = create_async_engine(f"sqlite+aiosqlite:///{sqlite_path}")
        try:
            async with engine.begin() as conn:
                await conn.execute(
                    insert(transition_log_table).values(
                        entity_kind="cg",
                        entity_id="cg_old",
                        from_state="draft",
                        to_state="implementing",
                        occurred_at=datetime.now(UTC) - timedelta(days=100),
                    )
                )
        finally:
            await engine.dispose()

    asyncio.run(_seed())

    result = runner.invoke(app, ["migrate", "--prune", "-c", str(smai_yaml)])
    assert result.exit_code == 0, result.output
    assert "transition_log: deleted 1 row" in result.output

    async def _count() -> int:
        from smai_orchestrator.migrations import transition_log_table

        engine = create_async_engine(f"sqlite+aiosqlite:///{sqlite_path}")
        try:
            async with engine.connect() as conn:
                rows = (await conn.execute(select(transition_log_table.c.id))).all()
            return len(rows)
        finally:
            await engine.dispose()

    assert asyncio.run(_count()) == 0
