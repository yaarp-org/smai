"""Tests for the prompt-config loader (Task 2.B2 / §10.1 / §10.2)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from smai_agents.model_selection import TaskRole
from smai_agents.prompts import (
    PromptConfigNotFound,
    PromptConfigValidationError,
    clear_prompt_config_cache,
    load_prompt_config,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Each test starts with a clean process-local cache."""
    clear_prompt_config_cache()


# All eight TaskRoles ship a base.yaml.
_ALL_ROLES: list[TaskRole] = [
    "planner",
    "harness_builder",
    "technique_implementer",
    "code_reviewer",
    "contextual_evaluator",
    "supervisor",
    "screener",
    "enricher",
]


@pytest.mark.parametrize("role", _ALL_ROLES)
def test_base_yaml_loads_for_all_roles(role: TaskRole) -> None:
    """Every shipped TaskRole has a valid base.yaml."""
    config = load_prompt_config(role)
    assert config.role == role
    assert config.system_prompt
    assert config.initial_user_message_template
    assert config.layer_chain == [f"{role}/base"]


def test_harness_builder_loads_with_base_only() -> None:
    """Acceptance bullet — harness builder base loads cleanly."""
    config = load_prompt_config("harness_builder")
    assert config.role == "harness_builder"
    tool_names = [t.name for t in config.tools]
    # Base ships full inventory + finish.
    assert "read_file" in tool_names
    assert "run_experiment" in tool_names
    assert "finish" in tool_names


def test_harness_builder_loads_with_variant() -> None:
    """Acceptance bullet — harness builder + a sample variant compose
    cleanly with the layer chain reflecting the merge order."""
    config = load_prompt_config("harness_builder", variant_name="lint_first")
    assert config.layer_chain == [
        "harness_builder/base",
        "harness_builder/lint_first",
    ]
    # Variant overrode system_prompt.
    assert "lint-first" in config.system_prompt
    # Replace-semantics on tools — the variant's tool list dropped `execute`.
    tool_names = [t.name for t in config.tools]
    assert "execute" not in tool_names
    assert "run_experiment" in tool_names


def test_planner_novel_technique_variant() -> None:
    config = load_prompt_config("planner", variant_name="novel_technique")
    assert config.layer_chain == ["planner/base", "planner/novel_technique"]
    assert "novel_technique" in config.system_prompt


def test_planner_paper_ingestion_variant() -> None:
    config = load_prompt_config("planner", variant_name="paper_ingestion")
    assert config.layer_chain == ["planner/base", "planner/paper_ingestion"]


def test_contextual_evaluator_compare_to_baseline_variant() -> None:
    config = load_prompt_config("contextual_evaluator", variant_name="compare_to_baseline")
    assert config.layer_chain == [
        "contextual_evaluator/base",
        "contextual_evaluator/compare_to_baseline",
    ]


def test_contextual_evaluator_compare_to_target_variant() -> None:
    config = load_prompt_config("contextual_evaluator", variant_name="compare_to_target")
    assert config.layer_chain == [
        "contextual_evaluator/base",
        "contextual_evaluator/compare_to_target",
    ]


def test_missing_variant_raises(tmp_path: Path) -> None:
    with pytest.raises(PromptConfigNotFound):
        load_prompt_config("harness_builder", variant_name="does_not_exist")


def test_override_layer_wins(tmp_path: Path) -> None:
    """Override layer system_prompt overrides variant + base."""
    override = {
        "layer": "override",
        "name": "deployment/harness_builder",
        "role": "harness_builder",
        "system_prompt": "OVERRIDE-SPECIFIC PROMPT",
    }
    (tmp_path / "harness_builder.yaml").write_text(yaml.safe_dump(override))

    config = load_prompt_config(
        "harness_builder",
        variant_name="lint_first",
        overrides_dir=tmp_path,
    )
    assert config.system_prompt == "OVERRIDE-SPECIFIC PROMPT"
    assert config.layer_chain == [
        "harness_builder/base",
        "harness_builder/lint_first",
        "deployment/harness_builder",
    ]


def test_override_can_replace_tools(tmp_path: Path) -> None:
    """§10.2 verbatim: override CAN replace the tool list."""
    override = {
        "layer": "override",
        "name": "tenant_acme/harness_builder",
        "role": "harness_builder",
        "tools": [
            {"name": "read_file"},
            {"name": "finish"},
        ],
    }
    (tmp_path / "harness_builder.yaml").write_text(yaml.safe_dump(override))

    config = load_prompt_config("harness_builder", overrides_dir=tmp_path)
    tool_names = [t.name for t in config.tools]
    assert tool_names == ["read_file", "finish"]


def test_override_inherits_when_field_null(tmp_path: Path) -> None:
    """An override that only sets `notes` leaves system_prompt etc. inherited."""
    override = {
        "layer": "override",
        "name": "tenant/notes_only",
        "role": "harness_builder",
        "notes": "we only document, not change",
    }
    (tmp_path / "harness_builder.yaml").write_text(yaml.safe_dump(override))

    base_config = load_prompt_config("harness_builder")
    overridden = load_prompt_config("harness_builder", overrides_dir=tmp_path)

    assert overridden.system_prompt == base_config.system_prompt
    assert overridden.tools == base_config.tools


def test_override_absent_when_file_missing(tmp_path: Path) -> None:
    """An overrides_dir without a matching <role>.yaml is silently no-op."""
    config = load_prompt_config("harness_builder", overrides_dir=tmp_path)
    assert config.layer_chain == ["harness_builder/base"]


def test_malformed_yaml_raises(tmp_path: Path) -> None:
    """A YAML parse error surfaces as PromptConfigValidationError at load."""
    (tmp_path / "harness_builder.yaml").write_text(
        "layer: override\n  bad indent: oops\n",
    )
    with pytest.raises(PromptConfigValidationError):
        load_prompt_config("harness_builder", overrides_dir=tmp_path)


def test_yaml_with_wrong_layer_kind_raises(tmp_path: Path) -> None:
    """An override file whose `layer` says `base` is rejected."""
    bad = {
        "layer": "base",
        "name": "wrong",
        "role": "harness_builder",
        "system_prompt": "x",
    }
    (tmp_path / "harness_builder.yaml").write_text(yaml.safe_dump(bad))
    with pytest.raises(PromptConfigValidationError):
        load_prompt_config("harness_builder", overrides_dir=tmp_path)


def test_yaml_with_extra_field_raises(tmp_path: Path) -> None:
    """`extra='forbid'` on PromptLayer — typos surface at load, not at use."""
    bad = {
        "layer": "override",
        "name": "x",
        "role": "harness_builder",
        "systme_prompt": "typo",  # intentional typo — extra='forbid' should reject
    }
    (tmp_path / "harness_builder.yaml").write_text(yaml.safe_dump(bad))
    with pytest.raises(PromptConfigValidationError):
        load_prompt_config("harness_builder", overrides_dir=tmp_path)


def test_yaml_with_role_mismatch_raises(tmp_path: Path) -> None:
    """An override claiming role=planner under harness_builder.yaml fails."""
    bad = {
        "layer": "override",
        "name": "x",
        "role": "planner",
        "system_prompt": "x",
    }
    (tmp_path / "harness_builder.yaml").write_text(yaml.safe_dump(bad))
    with pytest.raises(PromptConfigValidationError):
        load_prompt_config("harness_builder", overrides_dir=tmp_path)


def test_single_call_agent_has_structured_output_tool() -> None:
    """Single-call agents (code_reviewer, etc.) ship with structured_output_tool set."""
    cfg = load_prompt_config("code_reviewer")
    assert cfg.structured_output_tool is not None
    assert cfg.structured_output_tool.name == "submit_review"
    assert cfg.tools == []  # single-call agents expose no inventory


def test_multi_turn_agent_has_no_structured_output_tool() -> None:
    cfg = load_prompt_config("harness_builder")
    assert cfg.structured_output_tool is None
