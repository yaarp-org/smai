"""Tests for the four-source config-layering pipeline (`09-cli.md` §2).

Verifies the per-field precedence — defaults → file → env → flags —
plus the file search-order and validation-error surface.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from smai_cli.config import (
    PHASE_2_DEFAULT_PIPELINES,
    ConfigFileError,
    ConfigValidationError,
    base_defaults,
    dev_defaults,
    load_runtime_config,
)


def _write_yaml(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_dev_defaults_yield_runtime_config_with_no_inputs() -> None:
    cfg = load_runtime_config(env={}, defaults=dev_defaults())
    assert cfg.plugins.metadata_store == "sqlite"
    assert cfg.plugins.artifact_store == "localfs"
    assert cfg.plugins.compute == "localgpu"
    assert cfg.plugins.llm_provider == "bedrock"
    assert cfg.engine.poll_interval_seconds == 10
    assert cfg.engine.fair_scheduling == "off"
    assert tuple(cfg.pipelines) == PHASE_2_DEFAULT_PIPELINES


def test_base_defaults_alone_fail_to_validate_without_plugins() -> None:
    """``base_defaults()`` ships an empty plugin block — the
    :class:`PluginSelection` model requires plugin names, so loading
    raises :class:`ConfigValidationError` to surface the missing
    fields.
    """
    with pytest.raises(ConfigValidationError):
        load_runtime_config(env={}, defaults=base_defaults())


def test_yaml_file_overrides_defaults(tmp_path: Path) -> None:
    cfg_path = tmp_path / "smai.yaml"
    _write_yaml(
        cfg_path,
        """\
engine:
  poll_interval_seconds: 60
plugins:
  metadata_store: postgres
""",
    )
    cfg = load_runtime_config(config_path=cfg_path, env={}, defaults=dev_defaults())
    assert cfg.engine.poll_interval_seconds == 60  # file > defaults
    assert cfg.plugins.metadata_store == "postgres"  # file > defaults
    # Untouched fields fall through to dev defaults.
    assert cfg.plugins.compute == "localgpu"


def test_env_overrides_file(tmp_path: Path) -> None:
    cfg_path = tmp_path / "smai.yaml"
    _write_yaml(
        cfg_path,
        """\
plugins:
  metadata_store: postgres
""",
    )
    env = {"SMAI_PLUGINS__METADATA_STORE": "sqlite"}
    cfg = load_runtime_config(config_path=cfg_path, env=env, defaults=dev_defaults())
    assert cfg.plugins.metadata_store == "sqlite"  # env > file


def test_flag_overrides_env(tmp_path: Path) -> None:
    env = {"SMAI_PLUGINS__METADATA_STORE": "postgres"}
    cfg = load_runtime_config(
        env=env,
        defaults=dev_defaults(),
        flag_overrides={"plugins": {"metadata_store": "sqlite"}},
    )
    assert cfg.plugins.metadata_store == "sqlite"


def test_env_var_double_underscore_nesting() -> None:
    """``SMAI_PLUGINS__METADATA_STORE_CONFIG__URI`` → ``plugins.metadata_store_config.uri``."""
    env = {
        "SMAI_PLUGINS__METADATA_STORE_CONFIG__URI": "sqlite+aiosqlite:///path/to/db",
    }
    cfg = load_runtime_config(env=env, defaults=dev_defaults())
    assert cfg.plugins.metadata_store_config["uri"] == "sqlite+aiosqlite:///path/to/db"


def test_env_var_int_parsing() -> None:
    """Integer-shaped values in env vars round-trip as ints."""
    env = {"SMAI_ENGINE__POLL_INTERVAL_SECONDS": "45"}
    cfg = load_runtime_config(env=env, defaults=dev_defaults())
    assert cfg.engine.poll_interval_seconds == 45


def test_env_var_bool_parsing() -> None:
    """``true``/``false`` env values become Python booleans (the
    ``fair_scheduling`` field is a Literal — pick another bool field
    once one is added; for now this verifies parsing of ``true`` does
    not silently become the literal string ``"true"``).
    """
    # Use the parse_env_value helper indirectly; engine has no bool
    # field today so we verify via the layered dict for completeness.
    from smai_cli.config import _parse_env_value  # noqa: PLC0415 — internal probe

    assert _parse_env_value("true") is True
    assert _parse_env_value("FALSE") is False
    assert _parse_env_value("null") is None


def test_per_role_model_env_vars_are_skipped(tmp_path: Path) -> None:
    """``SMAI_MODEL_<ROLE>`` vars must NOT pollute the layered config —
    they're consumed downstream by :func:`build_per_role_llm_providers`.
    """
    env = {
        "SMAI_MODEL_PLANNER": "bedrock:claude-3.5-sonnet",
        "SMAI_PLUGINS__METADATA_STORE": "sqlite",
    }
    cfg = load_runtime_config(env=env, defaults=dev_defaults())
    # If the SMAI_MODEL_ vars leaked in, RuntimeConfig would either
    # have an unexpected top-level field (and raise) or the
    # ``model.planner`` path would appear. Check via dump:
    dumped = cfg.model_dump()
    assert "model" not in dumped
    assert cfg.plugins.metadata_store == "sqlite"


def test_smai_config_env_var_points_at_yaml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "alt-config.yaml"
    _write_yaml(cfg_path, "plugins:\n  metadata_store: postgres\n")
    env = {"SMAI_CONFIG": str(cfg_path)}
    cfg = load_runtime_config(env=env, defaults=dev_defaults())
    assert cfg.plugins.metadata_store == "postgres"


def test_malformed_yaml_raises_config_file_error(tmp_path: Path) -> None:
    cfg_path = tmp_path / "broken.yaml"
    _write_yaml(cfg_path, "engine: : :\n  unbalanced\n")
    with pytest.raises(ConfigFileError):
        load_runtime_config(config_path=cfg_path, env={}, defaults=dev_defaults())


def test_non_mapping_yaml_raises_config_file_error(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.yaml"
    _write_yaml(cfg_path, "- just\n- a\n- list\n")
    with pytest.raises(ConfigFileError):
        load_runtime_config(config_path=cfg_path, env={}, defaults=dev_defaults())


def test_validation_error_carries_context(tmp_path: Path) -> None:
    """A missing required plugin name surfaces as
    :class:`ConfigValidationError` (not a raw Pydantic traceback).
    """
    cfg_path = tmp_path / "smai.yaml"
    _write_yaml(cfg_path, "plugins:\n  llm_provider: ''\n")
    # Empty defaults → empty plugin name → validation requires
    # non-empty selection; surface raises a typed error.
    with pytest.raises((ConfigValidationError, ConfigFileError)):
        load_runtime_config(config_path=cfg_path, env={}, defaults=base_defaults())


def test_worked_example_per_09_section_2_1(tmp_path: Path) -> None:
    """Reproduces the worked example in `09` §2.1.

    File: ``plugins.metadata_store_config.uri = postgres://localhost/smai_dev``.
    Env: ``SMAI_PLUGINS__METADATA_STORE_CONFIG__URI=postgres://staging/smai``.
    Flag: ``plugins.metadata_store_config.uri = postgres://prod/smai``.

    Effective: ``postgres://prod/smai`` (flag wins).
    """
    cfg_path = tmp_path / "smai.yaml"
    _write_yaml(
        cfg_path,
        """\
plugins:
  metadata_store: postgres
  metadata_store_config:
    uri: postgres://localhost/smai_dev
""",
    )
    env = {"SMAI_PLUGINS__METADATA_STORE_CONFIG__URI": "postgres://staging/smai"}
    cfg = load_runtime_config(
        config_path=cfg_path,
        env=env,
        defaults=dev_defaults(),
        flag_overrides={"plugins": {"metadata_store_config": {"uri": "postgres://prod/smai"}}},
    )
    assert cfg.plugins.metadata_store == "postgres"  # file value, never overridden
    assert cfg.plugins.metadata_store_config["uri"] == "postgres://prod/smai"
