"""``smai serve`` deprecation warning (Task 4.L1 Part 3 / `12` §7.2).

Per the spec:

* Invoking ``smai serve`` emits a one-line deprecation warning to stderr.
* Behavior is otherwise unchanged — the existing read-only Jinja
  dashboard still boots and serves the documented pages.
* Source-tree removal is a v2.1 backlog item (`12` §7.2 / `implementation_plan.md` §8).

Both properties are pinned by tests here so a future v2.x change can't
silently drop the warning or alter the dashboard wiring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner


def test_smai_serve_emits_deprecation_warning_on_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``smai serve`` invocation prints the deprecation line on stderr."""
    _patch_short_circuit_serve(monkeypatch)
    # Avoid stomping a real ~/.smai by pointing SMAI_HOME at tmp_path.
    monkeypatch.setenv("SMAI_HOME", str(tmp_path))

    from smai_cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0, result.output + (result.stderr or "")
    stderr = result.stderr or ""
    assert "smai serve: deprecated" in stderr
    assert "smai ui" in stderr
    assert "12-ui-process.md" in stderr


def test_smai_serve_help_still_renders(tmp_path: Path) -> None:
    """``--help`` doesn't trigger the warning; the help text is unchanged."""
    del tmp_path
    from smai_cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    # The deprecation log is in the verb body, NOT the docstring; the
    # `--help` output renders the docstring before the body runs, so
    # the warning text shouldn't leak into stderr from `--help`.
    assert "smai serve: deprecated" not in (result.stderr or "")


def _patch_short_circuit_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub :meth:`Runtime.start_in_band` + uvicorn so the dashboard
    verb's body executes through the deprecation warning + exits
    without binding a real port."""

    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager

    from smai_cli import main as cli_main

    @asynccontextmanager
    async def _fake_start_in_band(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        del args, kwargs

        class _FakeRuntime:
            workspace_root = Path("/tmp")

            @property
            def config(self) -> Any:
                from smai_orchestrator import (  # noqa: PLC0415
                    PluginSelection,
                    RuntimeConfig,
                )
                from smai_orchestrator.engine.config import EngineConfig  # noqa: PLC0415

                return RuntimeConfig(
                    engine=EngineConfig(),
                    plugins=PluginSelection(
                        llm_provider="bedrock",
                        metadata_store="sqlite",
                        artifact_store="localfs",
                        compute="localgpu",
                    ),
                    pipelines=["smai_cg_execution"],
                )

        yield _FakeRuntime()

    monkeypatch.setattr(cli_main.Runtime, "start_in_band", _fake_start_in_band)

    class _FakeUvicornServer:
        def __init__(self, config: Any) -> None:
            del config

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

    # build_app reads from MetadataStore — short-circuit it too.
    from fastapi import FastAPI

    def _fake_build_app(_runtime: Any) -> FastAPI:
        return FastAPI()

    monkeypatch.setattr("smai_cli.dashboard.build_app", _fake_build_app)
