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


def test_explicit_overrides_take_precedence_over_env() -> None:
    """The ``overrides`` dict wins (per :func:`get_model_for_task`'s
    resolution order)."""
    env = {"SMAI_MODEL_PLANNER": "bedrock:env-route"}
    overrides: dict[TaskRole, tuple[str, str]] = {
        cast(TaskRole, "planner"): ("bedrock", "explicit-route"),
    }
    providers = build_per_role_llm_providers(_selection(), env=env, overrides=overrides)
    # The explicit override is shared with any other role that
    # resolves to the same tuple — but "explicit-route" is unique to
    # planner here, so its instance is distinct.
    planner_provider = providers["planner"]
    assert getattr(planner_provider, "model_id", None) == "explicit-route"


def test_unsupported_provider_raises() -> None:
    """Phase 2 only ships ``bedrock``; ``anthropic``/``openai`` raise."""
    overrides: dict[TaskRole, tuple[str, str]] = {
        cast(TaskRole, "planner"): ("anthropic", "claude-3.7"),
    }
    with pytest.raises(UnsupportedLlmProvider):
        build_per_role_llm_providers(_selection(), env={}, overrides=overrides)


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
