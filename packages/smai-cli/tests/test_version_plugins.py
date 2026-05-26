"""Tests for ``smai version`` and ``smai plugins`` (the diagnostic verbs)."""

from __future__ import annotations

import json

from smai_cli.config import PHASE_2_DEFAULT_PIPELINES
from smai_cli.main import app
from typer.testing import CliRunner


def test_version_prints_known_packages_in_text_format() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0, result.output
    output = result.output
    for pkg in (
        "smai-cli",
        "smai-core",
        "smai-orchestrator",
        "smai-agents",
        "smai-agent-runtime",
        "smai-runtime",
    ):
        assert pkg in output, f"missing package {pkg!r} in version output"


def test_version_json_format_round_trips_as_dict() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["version", "--format", "json"])
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert "smai-cli" in parsed
    assert "smai-core" in parsed
    assert "smai-orchestrator" in parsed


def test_plugins_lists_phase_2_specs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["plugins"])
    assert result.exit_code == 0, result.output
    output = result.output
    for spec_name in PHASE_2_DEFAULT_PIPELINES:
        assert spec_name in output, f"missing spec {spec_name!r} in plugins output"


def test_plugins_lists_each_entry_point_namespace() -> None:
    """Every namespace per `07-plugin-interfaces.md` §3.2 is rendered.

    This isn't an assertion on which plugins are installed (that
    depends on the workspace) — just that each interface group is
    surfaced.
    """
    runner = CliRunner()
    result = runner.invoke(app, ["plugins"])
    assert result.exit_code == 0, result.output
    for group in (
        "smai.llm_providers",
        "smai.metadata_stores",
        "smai.artifact_stores",
        "smai.computes",
    ):
        assert group in result.output


def test_plugins_json_format_has_required_keys() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["plugins", "--format", "json"])
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert "discovered" in parsed
    assert "registered_pipeline_specs" in parsed
    # The four namespaces all appear in 'discovered'.
    for group in (
        "smai.llm_providers",
        "smai.metadata_stores",
        "smai.artifact_stores",
        "smai.computes",
    ):
        assert group in parsed["discovered"]
