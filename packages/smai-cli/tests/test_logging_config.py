"""Round-6 item 6: ``smai dev`` / ``smai start`` configure logging,
honoring ``--verbose`` and ``SMAI_LOG_LEVEL`` (the latter was doc-only
fiction before)."""

from __future__ import annotations

import logging

import pytest
from smai_cli.main import _configure_logging, _resolve_log_level


def test_verbose_count_maps_to_levels() -> None:
    assert _resolve_log_level(0) == logging.WARNING
    assert _resolve_log_level(1) == logging.INFO
    assert _resolve_log_level(2) == logging.DEBUG
    assert _resolve_log_level(5) == logging.DEBUG


def test_smai_log_level_env_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMAI_LOG_LEVEL", "INFO")
    assert _resolve_log_level(0) == logging.INFO
    monkeypatch.setenv("SMAI_LOG_LEVEL", "debug")  # case-insensitive
    assert _resolve_log_level(0) == logging.DEBUG
    # An explicit --verbose flag overrides the env var.
    assert _resolve_log_level(1) == logging.INFO
    # Garbage env value falls back to WARNING (no crash).
    monkeypatch.setenv("SMAI_LOG_LEVEL", "nonsense")
    assert _resolve_log_level(0) == logging.WARNING
    monkeypatch.delenv("SMAI_LOG_LEVEL")
    assert _resolve_log_level(0) == logging.WARNING


def test_configure_logging_sets_root_level() -> None:
    root = logging.getLogger()
    saved = root.level
    try:
        _configure_logging(2)
        assert root.level == logging.DEBUG
        _configure_logging(1)
        assert root.level == logging.INFO
        _configure_logging(0)
        assert root.level == logging.WARNING
    finally:
        root.setLevel(saved)
