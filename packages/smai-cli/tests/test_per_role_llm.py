"""Tests for :func:`smai_cli.per_role_llm.build_per_role_llm_providers`.

Per `04-agents.md` §4 / Task 2.D2 brief: per-role per-model
divergence lives in the CLI; one :class:`LlmProvider` instance per
unique ``(provider, model_id)`` tuple, shared across roles that
resolve to the same tuple.
"""

from __future__ import annotations

from typing import cast

import pytest
from smai_agents.model_selection import TASK_DEFAULTS, TaskRole
from smai_cli.per_role_llm import UnsupportedLlmProvider, build_per_role_llm_providers
from smai_orchestrator import DEFAULT_TASK_ROLES, PluginSelection


def _selection(**overrides: object) -> PluginSelection:
    base: dict[str, object] = {
        "llm_provider": "bedrock",
        "metadata_store": "sqlite",
        "artifact_store": "localfs",
        "compute": "localgpu",
        "llm_provider_config": {"region": "us-east-1"},
    }
    base.update(overrides)
    return PluginSelection.model_validate(base)


def test_default_resolution_uses_task_defaults_table() -> None:
    """No env / no overrides → :data:`TASK_DEFAULTS` selects the model
    per role; tuple-keyed cache shares instances across same-tuple
    siblings.
    """
    providers = build_per_role_llm_providers(
        _selection(),
        env={},
    )
    assert set(providers.keys()) == set(DEFAULT_TASK_ROLES)
    # Roles that resolve to the same tuple share a single instance.
    # In TASK_DEFAULTS, "code_reviewer" is Opus and "harness_builder"
    # is Opus — same model_id → same instance.
    assert providers["code_reviewer"] is providers["harness_builder"]
    assert providers["code_reviewer"] is providers["planner"]
    # Sonnet-tier roles share an instance among themselves but differ
    # from the Opus instance.
    assert providers["contextual_evaluator"] is providers["screener"]
    assert providers["contextual_evaluator"] is not providers["code_reviewer"]


def test_env_var_override_routes_one_role_to_a_distinct_instance() -> None:
    env = {"SMAI_MODEL_PLANNER": "bedrock:custom-model-id"}
    providers = build_per_role_llm_providers(_selection(), env=env)
    # planner now points at a different (provider, model_id) than
    # everyone else — so its instance is distinct from harness_builder.
    assert providers["planner"] is not providers["harness_builder"]


def test_env_takes_precedence_over_explicit_overrides() -> None:
    """Round-7 reordering: ``SMAI_MODEL_<ROLE>`` wins over the config
    override map (per :func:`get_model_for_task`'s resolution order)."""
    env = {"SMAI_MODEL_PLANNER": "bedrock:env-route"}
    overrides: dict[TaskRole, tuple[str, str]] = {
        cast(TaskRole, "planner"): ("bedrock", "config-route"),
    }
    providers = build_per_role_llm_providers(_selection(), env=env, overrides=overrides)
    assert getattr(providers["planner"], "model_id", None) == "env-route"


def test_config_override_beats_default_when_no_env() -> None:
    """With no env var, the config override map selects the model."""
    overrides: dict[TaskRole, tuple[str, str]] = {
        cast(TaskRole, "planner"): ("bedrock", "config-route"),
    }
    providers = build_per_role_llm_providers(_selection(), env={}, overrides=overrides)
    assert getattr(providers["planner"], "model_id", None) == "config-route"
    # Other roles are untouched -> still on the TASK_DEFAULTS model.
    assert getattr(providers["harness_builder"], "model_id", None) != "config-route"


def test_unsupported_provider_raises() -> None:
    """Unknown provider names raise :class:`UnsupportedLlmProvider`.

    Phase 3 Task 3.F5 lifted the previous bedrock-only constraint;
    ``anthropic`` / ``openai`` are now supported. An unrecognised name
    (e.g., ``"vertex"``) still raises so the CLI surface fails-fast
    rather than mid-agent-loop.
    """
    overrides: dict[TaskRole, tuple[str, str]] = {
        cast(TaskRole, "planner"): ("vertex", "gemini-2.5"),
    }
    with pytest.raises(UnsupportedLlmProvider):
        build_per_role_llm_providers(_selection(), env={}, overrides=overrides)


def test_anthropic_provider_constructs_via_overrides() -> None:
    """Anthropic-rooted overrides build a real
    :class:`smai_llm_anthropic.AnthropicProvider` instance —
    bedrock-only kwargs (``region``) are silently dropped on the way
    in."""
    from smai_llm_anthropic import AnthropicProvider  # noqa: PLC0415

    overrides: dict[TaskRole, tuple[str, str]] = {
        cast(TaskRole, "planner"): ("anthropic", "claude-opus-4-7"),
    }
    providers = build_per_role_llm_providers(_selection(), env={}, overrides=overrides)
    planner = providers["planner"]
    assert isinstance(planner, AnthropicProvider)
    assert planner.model_id == "claude-opus-4-7"


def test_openai_provider_constructs_via_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same shape against OpenAI.

    The OpenAI SDK demands ``OPENAI_API_KEY`` at construction time
    (the Anthropic SDK is lazy about this check). We seed a sentinel
    key so the constructor does not error before
    :func:`build_per_role_llm_providers` returns the instance.
    """
    from smai_llm_openai import OpenAIProvider  # noqa: PLC0415

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    overrides: dict[TaskRole, tuple[str, str]] = {
        cast(TaskRole, "planner"): ("openai", "gpt-4o"),
    }
    providers = build_per_role_llm_providers(_selection(), env={}, overrides=overrides)
    planner = providers["planner"]
    assert isinstance(planner, OpenAIProvider)
    assert planner.model_id == "gpt-4o"


def test_base_kwargs_pass_through_minus_model_id() -> None:
    """``selection.llm_provider_config`` keys (e.g., ``region``) flow
    into every constructed instance; ``model_id`` is per-instance.
    """
    selection = _selection(
        llm_provider_config={"region": "us-west-2", "model_id": "ignored"},
    )
    providers = build_per_role_llm_providers(selection, env={})
    # Every role resolves via TASK_DEFAULTS, NOT the
    # ``model_id="ignored"`` kwarg (which is a CLI mis-shape — model
    # selection always flows through get_model_for_task).
    expected_models = {model_id for _, model_id in TASK_DEFAULTS.values()}
    actual_models = {getattr(p, "model_id", None) for p in providers.values()}
    assert actual_models == expected_models, (
        f"per-role model_id resolution failed; expected {expected_models}, got {actual_models}"
    )


def test_default_task_roles_match_smai_agents_literal() -> None:
    """Defensive: re-assert :data:`DEFAULT_TASK_ROLES` is in lockstep
    with :data:`TASK_DEFAULTS` — surfaces a literal drift in either
    package as a deterministic failure.
    """
    assert set(DEFAULT_TASK_ROLES) == set(TASK_DEFAULTS.keys())
