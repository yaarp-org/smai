""":data:`TASK_DEFAULTS` and :func:`get_model_for_task` per ``04-agents.md`` §4."""

from __future__ import annotations

import pytest
from smai_inline_agents import (
    TASK_DEFAULTS,
    ModelSelectionError,
    TaskRole,
    get_model_for_task,
)

# --- TASK_DEFAULTS verbatim ---------------------------------------------------


def test_task_defaults_covers_all_eight_roles() -> None:
    """§4: eight roles — five fleet roles + supervisor + screener + enricher."""
    expected = {
        "planner",
        "harness_builder",
        "technique_implementer",
        "code_reviewer",
        "contextual_evaluator",
        "supervisor",
        "screener",
        "enricher",
    }
    assert set(TASK_DEFAULTS.keys()) == expected


def test_task_defaults_heavy_roles_use_opus() -> None:
    """§4: planner / harness_builder / technique_implementer /
    code_reviewer all default to the Opus tier (claude-opus-4-6-v1)."""
    for role in (
        "planner",
        "harness_builder",
        "technique_implementer",
        "code_reviewer",
    ):
        provider, model_id = TASK_DEFAULTS[role]
        assert provider == "bedrock"
        assert model_id == "us.anthropic.claude-opus-4-6-v1"


def test_task_defaults_bounded_roles_use_sonnet() -> None:
    """§4: contextual_evaluator / supervisor / screener / enricher use
    the Sonnet tier (claude-sonnet-4-6) — bounded single-call roles."""
    for role in (
        "contextual_evaluator",
        "supervisor",
        "screener",
        "enricher",
    ):
        provider, model_id = TASK_DEFAULTS[role]
        assert provider == "bedrock"
        assert model_id == "us.anthropic.claude-sonnet-4-6"


# --- Resolution order ---------------------------------------------------------


def test_env_takes_precedence_over_config_override() -> None:
    """Round-7: ``SMAI_MODEL_<ROLE>`` wins over the config override map."""
    overrides: dict[TaskRole, tuple[str, str]] = {
        "planner": ("bedrock", "config-model"),
    }
    env = {"SMAI_MODEL_PLANNER": "bedrock:env-model"}
    assert get_model_for_task("planner", overrides=overrides, env=env) == (
        "bedrock",
        "env-model",
    )


def test_config_override_takes_precedence_over_default() -> None:
    """The config override map (``engine.role_models``) beats TASK_DEFAULTS
    when no env var is set."""
    overrides: dict[TaskRole, tuple[str, str]] = {
        "planner": ("bedrock", "us.anthropic.claude-sonnet-4-6"),
    }
    assert get_model_for_task("planner", overrides=overrides, env={}) == (
        "bedrock",
        "us.anthropic.claude-sonnet-4-6",
    )


def test_env_var_takes_precedence_over_default() -> None:
    """SMAI_MODEL_<ROLE> overrides TASK_DEFAULTS when no config override."""
    env = {"SMAI_MODEL_HARNESS_BUILDER": "bedrock:qwen.qwen3-coder-next"}
    assert get_model_for_task("harness_builder", env=env) == (
        "bedrock",
        "qwen.qwen3-coder-next",
    )


def test_env_var_format_must_be_provider_colon_model() -> None:
    """Malformed env var raises :class:`ModelSelectionError`."""
    env = {"SMAI_MODEL_PLANNER": "no-colon"}
    with pytest.raises(ModelSelectionError):
        get_model_for_task("planner", env=env)


def test_env_var_rejects_empty_provider_or_model() -> None:
    for raw in (":model", "provider:", " : ", ":"):
        env = {"SMAI_MODEL_PLANNER": raw}
        with pytest.raises(ModelSelectionError):
            get_model_for_task("planner", env=env)


def test_falls_through_to_task_defaults() -> None:
    """No override + no env var = TASK_DEFAULTS."""
    assert get_model_for_task("planner", env={}) == TASK_DEFAULTS["planner"]
    assert get_model_for_task("supervisor", env={}) == TASK_DEFAULTS["supervisor"]


def test_env_var_strips_whitespace() -> None:
    """``"  bedrock  :  modelid  "`` is a real value users set; whitespace
    around the separator should not break the parser."""
    env = {"SMAI_MODEL_PLANNER": " bedrock : us.anthropic.claude-opus-4-6-v1 "}
    assert get_model_for_task("planner", env=env) == (
        "bedrock",
        "us.anthropic.claude-opus-4-6-v1",
    )
