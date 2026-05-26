"""Tests for :func:`build_agent` + :class:`BedrockCacheSettings`.

Sub-PR D thread 3: the opportunistic boolean cache gate from sub-PR C1
is replaced by a structured per-step :class:`BedrockCacheSettings`
shape (notes/pydantic_ai_bedrock_caching.md). These tests pin the
projection from the structured shape to PydanticAI 1.102.0's
:class:`BedrockModelSettings` flags.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError
from smai_agent_runtime.agent_reasoning import (
    DEFAULT_BEDROCK_CACHE_SETTINGS,
    NO_BEDROCK_CACHE,
    BedrockCacheSettings,
    build_agent,
)
from smai_agent_runtime.agent_reasoning.agents import (
    _settings_are_all_off,
    _to_bedrock_model_settings,
)


class _DummyOutput(BaseModel):
    answer: str


def test_default_settings_match_subpr_c1_behavior() -> None:
    """The default :class:`BedrockCacheSettings` reproduces sub-PR C1's
    opportunistic-mode behavior verbatim — instructions + tool
    definitions cached at 5m, messages disabled. Behavior-preserving
    refactor at the default; structured shape enables per-step
    variation when a future cost-tuning pass needs it."""
    assert DEFAULT_BEDROCK_CACHE_SETTINGS.cache_instructions == "5m"
    assert DEFAULT_BEDROCK_CACHE_SETTINGS.cache_tool_definitions == "5m"
    assert DEFAULT_BEDROCK_CACHE_SETTINGS.cache_messages == "off"


def test_no_bedrock_cache_sentinel_disables_every_field() -> None:
    """The explicit opt-out sentinel zeroes every TTL — handy for tests
    that need to assert the no-caching projection path."""
    assert NO_BEDROCK_CACHE.cache_instructions == "off"
    assert NO_BEDROCK_CACHE.cache_tool_definitions == "off"
    assert NO_BEDROCK_CACHE.cache_messages == "off"
    assert _settings_are_all_off(NO_BEDROCK_CACHE)


def test_settings_reject_extra_fields() -> None:
    """``BedrockCacheSettings`` is frozen + extra=forbid. A typo on a
    field name surfaces at config time rather than silently falling
    through to default."""
    with pytest.raises(ValidationError):
        BedrockCacheSettings(cache_instr="5m")  # type: ignore[call-arg]


def test_to_bedrock_model_settings_omits_off_fields() -> None:
    """The projection omits any field whose TTL is ``"off"`` — PydanticAI's
    documented idiom for the "no cache for this surface" branch."""
    partial = BedrockCacheSettings(
        cache_instructions="1h",
        cache_tool_definitions="off",
        cache_messages="off",
    )
    projected = _to_bedrock_model_settings(partial)
    payload = dict(projected)
    assert payload == {"bedrock_cache_instructions": "1h"}


def test_to_bedrock_model_settings_emits_every_field_when_set() -> None:
    """Every field with a non-``"off"`` TTL projects to the matching
    :class:`BedrockModelSettings` ``bedrock_cache_*`` key."""
    full = BedrockCacheSettings(
        cache_instructions="5m",
        cache_tool_definitions="1h",
        cache_messages="5m",
    )
    payload = dict(_to_bedrock_model_settings(full))
    assert payload == {
        "bedrock_cache_instructions": "5m",
        "bedrock_cache_tool_definitions": "1h",
        "bedrock_cache_messages": "5m",
    }


def test_build_agent_bedrock_path_passes_model_settings_with_default() -> None:
    """Default settings produce a PydanticAI Agent whose model_settings
    carry the 5m instructions + tool_definitions flags (sub-PR C1
    behavior preserved)."""
    agent = build_agent(
        provider="bedrock",
        model_id="us.anthropic.claude-sonnet-4-6",
        output_type=_DummyOutput,
        system_prompt="test",
    )
    # PydanticAI 1.102.0 stores settings on the agent as
    # ``Agent._model_settings``; inspect via the public attribute exposed
    # for tests (``model_settings``).
    settings = getattr(agent, "model_settings", None)
    assert settings is not None
    settings_dict = dict(settings)
    assert settings_dict.get("bedrock_cache_instructions") == "5m"
    assert settings_dict.get("bedrock_cache_tool_definitions") == "5m"


def test_build_agent_bedrock_path_with_no_cache_skips_settings() -> None:
    """Passing :data:`NO_BEDROCK_CACHE` skips the model_settings
    construction entirely; the Agent runs without cache flags."""
    agent = build_agent(
        provider="bedrock",
        model_id="us.anthropic.claude-sonnet-4-6",
        output_type=_DummyOutput,
        system_prompt="test",
        bedrock_cache_settings=NO_BEDROCK_CACHE,
    )
    settings = getattr(agent, "model_settings", None)
    # PydanticAI's Agent.model_settings is None when no settings were
    # passed at construction (vs an empty BedrockModelSettings).
    assert settings is None


def test_build_agent_non_bedrock_provider_ignores_cache_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anthropic / OpenAI providers ignore the Bedrock-only cache flags
    — the factory takes the no-settings branch."""
    # PydanticAI's Anthropic provider construction reads ANTHROPIC_API_KEY
    # eagerly; the test cares about ``model_settings``, not the client
    # construction path, so satisfy the key check with a placeholder.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-placeholder")
    agent = build_agent(
        provider="anthropic",
        model_id="claude-sonnet-4-6",
        output_type=_DummyOutput,
        system_prompt="test",
        bedrock_cache_settings=DEFAULT_BEDROCK_CACHE_SETTINGS,
    )
    settings = getattr(agent, "model_settings", None)
    assert settings is None


def test_build_agent_supports_per_step_cache_variation() -> None:
    """Two call sites can pass different :class:`BedrockCacheSettings`
    to the same factory — the per-step config matrix the sub-PR D
    refactor enables (a diagnose step might want longer-TTL tool
    cache than a body-generation step, for instance)."""
    body_step = build_agent(
        provider="bedrock",
        model_id="us.anthropic.claude-opus-4-6-v1",
        output_type=_DummyOutput,
        system_prompt="body step",
        bedrock_cache_settings=BedrockCacheSettings(
            cache_instructions="5m",
            cache_tool_definitions="5m",
        ),
    )
    diagnose_step = build_agent(
        provider="bedrock",
        model_id="us.anthropic.claude-opus-4-6-v1",
        output_type=_DummyOutput,
        system_prompt="diagnose step",
        bedrock_cache_settings=BedrockCacheSettings(
            cache_instructions="1h",
            cache_tool_definitions="1h",
        ),
    )
    body_settings = dict(getattr(body_step, "model_settings", {}) or {})
    diagnose_settings = dict(getattr(diagnose_step, "model_settings", {}) or {})
    assert body_settings.get("bedrock_cache_instructions") == "5m"
    assert diagnose_settings.get("bedrock_cache_instructions") == "1h"
