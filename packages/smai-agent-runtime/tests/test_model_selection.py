"""Per-step model selection tests (D3 mechanism).

Asserts the six-layer precedence chain documented in
:mod:`smai_agent_runtime.agent_reasoning.model_selection`. The chain is
the load-bearing contract for sub-PR D's host-side override map plus
operator env-var shortcuts.
"""

from __future__ import annotations

import pytest
from smai_agent_runtime.agent_reasoning import (
    SANDBOXED_ROLE_DEFAULTS,
    get_model_for_step,
)
from smai_agent_runtime.agent_reasoning.model_selection import ModelSelectionError


def test_falls_back_to_role_default_when_no_override() -> None:
    """Layer 6: bare role default surfaces when no env / no overrides."""
    result = get_model_for_step("harness_builder", "harness_body_generation", env={})
    assert result == SANDBOXED_ROLE_DEFAULTS["harness_builder"]


def test_flat_override_round7_shape_wins_over_default() -> None:
    """Layer 5: round-7's flat ``{role: model_id}`` shape projects to a
    ``(bedrock, model_id)`` tuple."""
    result = get_model_for_step(
        "harness_builder",
        "harness_body_generation",
        overrides={"harness_builder": "us.anthropic.claude-sonnet-4-6"},
        env={},
    )
    assert result == ("bedrock", "us.anthropic.claude-sonnet-4-6")


def test_nested_default_sentinel_wins_over_role_default() -> None:
    """Layer 4: the ``_default`` sentinel inside the nested form."""
    result = get_model_for_step(
        "harness_builder",
        "harness_body_generation",
        overrides={
            "harness_builder": {"_default": "us.anthropic.claude-sonnet-4-6"},
        },
        env={},
    )
    assert result == ("bedrock", "us.anthropic.claude-sonnet-4-6")


def test_nested_step_entry_wins_over_default_sentinel() -> None:
    """Layer 3: per-step entry beats ``_default``."""
    result = get_model_for_step(
        "harness_builder",
        "diagnose_on_failure",
        overrides={
            "harness_builder": {
                "_default": "us.anthropic.claude-opus-4-6-v1",
                "diagnose_on_failure": "us.anthropic.claude-sonnet-4-6",
            },
        },
        env={},
    )
    assert result == ("bedrock", "us.anthropic.claude-sonnet-4-6")


def test_smai_model_role_env_wins_over_nested_step_entry() -> None:
    """Layer 2: ``SMAI_MODEL_<ROLE>`` env beats config-supplied overrides."""
    result = get_model_for_step(
        "harness_builder",
        "diagnose_on_failure",
        overrides={
            "harness_builder": {"diagnose_on_failure": "us.anthropic.claude-sonnet-4-6"},
        },
        env={"SMAI_MODEL_HARNESS_BUILDER": "bedrock:us.anthropic.claude-opus-4-6-v1"},
    )
    assert result == ("bedrock", "us.anthropic.claude-opus-4-6-v1")


def test_smai_model_role_step_env_wins_over_role_env() -> None:
    """Layer 1: the most-specific env var wins over the role-wide env."""
    result = get_model_for_step(
        "harness_builder",
        "diagnose_on_failure",
        env={
            "SMAI_MODEL_HARNESS_BUILDER": "bedrock:role-fallback",
            "SMAI_MODEL_HARNESS_BUILDER__DIAGNOSE_ON_FAILURE": "bedrock:step-specific",
        },
    )
    assert result == ("bedrock", "step-specific")


def test_malformed_env_value_raises_typed_error() -> None:
    """Malformed env override surfaces a typed error so the caller can
    distinguish 'configuration broken' from 'no override set'."""
    with pytest.raises(ModelSelectionError):
        get_model_for_step(
            "harness_builder",
            "harness_body_generation",
            env={"SMAI_MODEL_HARNESS_BUILDER": "no_colon_here"},
        )
