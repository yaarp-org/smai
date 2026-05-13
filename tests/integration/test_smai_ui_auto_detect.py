"""Auto-detect rule tests for ``smai ui --with-worker`` (Task 4.L1).

Per ``12-ui-process.md`` §4.3 / §9.3: the verb infers ``--with-worker``
from the resolved plugin shape unless the user passes the flag
explicitly. Sqlite + localfs → on; anything else → off; mixed →
conservative-no-worker. The inferred decision MUST be surfaced via a
loud startup-log line so the user can override on the next launch.

These tests exercise the rule at two levels:

* :func:`smai_cli.main._infer_with_worker` directly — pure function
  over a :class:`RuntimeConfig`, fast and deterministic.
* The verb flow end-to-end via :class:`typer.testing.CliRunner`,
  with :class:`Runtime.start_in_band` and :class:`uvicorn.Server.serve`
  monkeypatched so the verb logs its decision and exits without
  binding a port.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from _4_l1_fakes import (  # type: ignore[import-not-found]
    build_dev_smai_yaml,
    build_postgres_smai_yaml,
)
from smai_cli.main import _infer_with_worker
from smai_orchestrator import (
    PluginSelection,
    RuntimeConfig,
)
from smai_orchestrator.engine.config import EngineConfig
from typer.testing import CliRunner


def _make_config(*, metadata_store: str, artifact_store: str) -> RuntimeConfig:
    return RuntimeConfig(
        engine=EngineConfig(),
        plugins=PluginSelection(
            llm_provider="bedrock",
            metadata_store=metadata_store,
            artifact_store=artifact_store,
            compute="localgpu",
        ),
        pipelines=["smai_cg_execution", "smai_cg_entries"],
    )


# === _infer_with_worker — direct rule tests ==================================


def test_auto_detect_dev_shape_returns_true() -> None:
    """sqlite + localfs → in-process worker (Case A laptop)."""
    config = _make_config(metadata_store="sqlite", artifact_store="localfs")
    assert _infer_with_worker(config) is True


def test_auto_detect_postgres_returns_false() -> None:
    """postgres + s3 → no in-process worker (Case B remote-data)."""
    config = _make_config(metadata_store="postgres", artifact_store="s3")
    assert _infer_with_worker(config) is False


def test_auto_detect_mixed_postgres_localfs_returns_false() -> None:
    """Mixed shape (postgres + localfs) is conservative-no-worker.

    Per `12` §4.3 table: any drift toward production plugins flips
    the rule off; mixed configs require the explicit ``--with-worker``
    opt-in.
    """
    config = _make_config(metadata_store="postgres", artifact_store="localfs")
    assert _infer_with_worker(config) is False


def test_auto_detect_mixed_sqlite_s3_returns_false() -> None:
    """Mixed sqlite + s3 also flips off (any non-localfs artifact store)."""
    config = _make_config(metadata_store="sqlite", artifact_store="s3")
    assert _infer_with_worker(config) is False


# === Verb-level tests (loud-log discipline) ==================================


def test_smai_ui_logs_in_process_worker_on_sqlite_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dev-shaped config logs ``starting in-process worker`` on stderr.

    Monkey-patches :meth:`Runtime.start_in_band` and
    :class:`uvicorn.Server` so the verb body executes through the
    auto-detect branch + startup log, then short-circuits the actual
    serve loop.
    """
    # ``smai ui``'s dev-filesystem defaults push the resolved sqlite URI
    # at ``$SMAI_HOME/state.db`` (overriding the ``:memory:`` URI in
    # the test smai.yaml, since the dev-filesystem hook is layered as
    # ``flag_overrides`` — highest precedence). Pin SMAI_HOME at a tmp
    # path so the strict ``_check_schema_at_head`` pre-flight runs
    # against a fresh database that needs no migration (state.db doesn't
    # exist, so ``get_current_revision`` returns ``None`` — but here we
    # need ``current==head`` for the verb to pass). Stamp the empty
    # database to head before invoking so the pre-flight passes.
    smai_home = tmp_path / "smai_home"
    monkeypatch.setenv("SMAI_HOME", str(smai_home))

    cfg_path = tmp_path / "smai.yaml"
    cfg_path.write_text(build_dev_smai_yaml(), encoding="utf-8")

    _patch_short_circuit(monkeypatch)

    # Run the migration so the dev-filesystem-defaulted state.db lands
    # at head; otherwise the strict ``--with-worker`` pre-flight fails.
    import asyncio  # noqa: PLC0415

    from smai_orchestrator.migrations import upgrade_to_head  # noqa: PLC0415
    from sqlalchemy.ext.asyncio import create_async_engine  # noqa: PLC0415

    async def _migrate() -> None:
        smai_home.mkdir(parents=True, exist_ok=True)
        engine = create_async_engine(f"sqlite+aiosqlite:///{smai_home / 'state.db'}")
        try:
            await upgrade_to_head(engine)
        finally:
            await engine.dispose()

    asyncio.run(_migrate())

    from smai_cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["ui", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.output + (result.stderr or "")
    combined = result.output + (result.stderr or "")
    assert "starting in-process worker" in combined
    assert "sqlite" in combined and "localfs" in combined


def test_smai_ui_logs_no_worker_on_postgres_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A postgres + s3 config logs ``NOT starting in-process worker``."""
    cfg_path = tmp_path / "smai.yaml"
    cfg_path.write_text(build_postgres_smai_yaml(), encoding="utf-8")

    _patch_short_circuit(monkeypatch)

    from smai_cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["ui", "--config", str(cfg_path)])
    # `--no-worker` mode skips schema-at-head — postgres URI here
    # points at a non-existent test database; pre-flights must be
    # soft so the verb still boots.
    assert result.exit_code == 0, result.output + (result.stderr or "")
    combined = result.output + (result.stderr or "")
    assert "NOT starting in-process worker" in combined
    assert "postgres" in combined


def test_smai_ui_explicit_with_worker_overrides_auto_detect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--with-worker`` flag wins over auto-detect even on postgres."""
    cfg_path = tmp_path / "smai.yaml"
    cfg_path.write_text(build_postgres_smai_yaml(), encoding="utf-8")

    _patch_short_circuit(monkeypatch)
    _patch_skip_strict_preflights(monkeypatch)

    from smai_cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["ui", "--with-worker", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.output + (result.stderr or "")
    combined = result.output + (result.stderr or "")
    assert "starting in-process worker" in combined
    # No "auto-detected" mention — the user passed the flag explicitly.
    assert "auto-detected" not in combined


# === Helpers =================================================================


def _patch_short_circuit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace :meth:`Runtime.start_in_band` + :class:`uvicorn.Server` so
    the verb body reaches its startup log + exits without binding."""

    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager

    from smai_cli import main as cli_main

    @asynccontextmanager
    async def _fake_start_in_band(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        del args, kwargs

        class _FakeRuntime:
            event_broker = None
            workspace_root = Path("/tmp")

            @property
            def config(self) -> Any:
                from smai_orchestrator import (  # noqa: PLC0415
                    PluginSelection,
                    RuntimeConfig,
                )
                from smai_orchestrator.engine.config import EngineConfig  # noqa: PLC0415

                return RuntimeConfig(
                    engine=EngineConfig(worker_count=1),
                    plugins=PluginSelection(
                        llm_provider="bedrock",
                        metadata_store="sqlite",
                        artifact_store="localfs",
                        compute="localgpu",
                    ),
                    pipelines=["smai_cg_execution"],
                )

            @property
            def plugins(self) -> Any:
                class _P:
                    metadata_store = type(
                        "_S",
                        (),
                        {"capabilities": type("_C", (), {"supports_leasing": True})()},
                    )()

                return _P()

            @property
            def worker_id(self) -> str:
                return "fake-worker"

        yield _FakeRuntime()

    monkeypatch.setattr(cli_main.Runtime, "start_in_band", _fake_start_in_band)

    class _FakeUvicornServer:
        def __init__(self, config: Any) -> None:
            del config
            self.should_exit = False

        async def serve(self) -> None:
            return None

    class _FakeUvicornConfig:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

    class _FakeUvicornModule:
        Server = _FakeUvicornServer
        Config = _FakeUvicornConfig

    import sys

    monkeypatch.setitem(sys.modules, "uvicorn", _FakeUvicornModule())


def _patch_skip_strict_preflights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the strict pre-flights so a postgres test config doesn't try
    to actually connect to the database during auto-detect verification."""

    async def _ok(_runtime_config: Any) -> None:
        return None

    from smai_cli import main as cli_main

    monkeypatch.setattr(cli_main, "_check_schema_at_head", _ok)
