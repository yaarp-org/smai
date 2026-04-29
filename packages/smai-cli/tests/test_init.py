"""``smai init`` writes a starter ``smai.yaml`` + ``experiment.yaml``."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from smai_cli.config import dev_defaults, load_runtime_config
from smai_cli.main import app
from typer.testing import CliRunner


def test_init_creates_starter_files(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "smai.yaml").is_file()
    assert (tmp_path / "experiment.yaml").is_file()


def test_init_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    (tmp_path / "smai.yaml").write_text("# existing\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code != 0
    # Existing file is preserved.
    assert (tmp_path / "smai.yaml").read_text() == "# existing\n"


def test_init_force_overwrites(tmp_path: Path) -> None:
    (tmp_path / "smai.yaml").write_text("# old\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(app, ["init", str(tmp_path), "--force"])
    assert result.exit_code == 0, result.output
    assert "# old" not in (tmp_path / "smai.yaml").read_text()


def test_starter_smai_yaml_round_trips_through_load_runtime_config(
    tmp_path: Path,
) -> None:
    """The starter ``smai.yaml`` parses cleanly via :func:`load_runtime_config`."""
    runner = CliRunner()
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.output
    cfg = load_runtime_config(
        config_path=tmp_path / "smai.yaml",
        env={},
        defaults=dev_defaults(),
    )
    assert cfg.plugins.metadata_store == "sqlite"
    assert cfg.plugins.llm_provider == "bedrock"


def test_starter_experiment_yaml_is_a_valid_dsl_document(tmp_path: Path) -> None:
    """The starter ``experiment.yaml`` parses through the DSL adapter
    (technique-registration is documented as a separate concern; we
    don't compile it here, just shape-check it).
    """
    pytest.importorskip("smai_core")
    from smai_core import DslDocumentAdapter  # noqa: PLC0415

    runner = CliRunner()
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.output
    payload = yaml.safe_load((tmp_path / "experiment.yaml").read_text())
    document = DslDocumentAdapter.validate_python(payload, context={"smai_mode": "dsl"})
    assert document is not None
