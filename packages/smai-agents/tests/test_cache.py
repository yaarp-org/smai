""":data:`DEFAULT_CACHE_CONFIG` matches ``04-agents.md`` §5 verbatim."""

from __future__ import annotations

from smai_agents import (
    DEFAULT_CACHE_CONFIG,
    SINGLE_CALL_CACHE_CONFIG,
    CacheConfig,
)


def test_default_cache_config_matches_v1_pattern() -> None:
    """§5: 'cache_static_prefix=True, cache_initial_message=True,
    rolling_cache_count=2.'"""
    assert DEFAULT_CACHE_CONFIG.cache_static_prefix is True
    assert DEFAULT_CACHE_CONFIG.cache_initial_message is True
    assert DEFAULT_CACHE_CONFIG.rolling_cache_count == 2


def test_single_call_cache_config_disables_rolling() -> None:
    """§5 final paragraph: 'rolling_cache_count=0 for these roles.'"""
    assert SINGLE_CALL_CACHE_CONFIG.cache_static_prefix is True
    assert SINGLE_CALL_CACHE_CONFIG.cache_initial_message is True
    assert SINGLE_CALL_CACHE_CONFIG.rolling_cache_count == 0


def test_default_cache_config_is_immutable_per_session() -> None:
    """A fresh ``CacheConfig`` does not share state with the module
    constant — Pydantic models are copy-on-construct."""
    custom = CacheConfig()
    assert custom != DEFAULT_CACHE_CONFIG  # default-default has all False
    assert DEFAULT_CACHE_CONFIG.cache_static_prefix is True
