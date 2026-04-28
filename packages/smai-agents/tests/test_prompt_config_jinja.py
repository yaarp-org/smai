"""Tests for Jinja2 rendering of initial_user_message_template (§10.1)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from smai_agents.prompts import (
    PromptConfigValidationError,
    clear_prompt_config_cache,
    load_prompt_config,
    render_initial_user_message,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_prompt_config_cache()


def test_render_with_all_variables() -> None:
    """Harness builder's base template renders with the expected vars."""
    config = load_prompt_config("harness_builder")
    rendered = render_initial_user_message(
        config,
        cg_id="cg-123",
        workspace_path="/tmp/ws",
    )
    assert "cg-123" in rendered
    assert "/tmp/ws" in rendered


def test_render_strict_undefined_raises() -> None:
    """Missing variable surfaces as PromptConfigValidationError."""
    config = load_prompt_config("harness_builder")
    with pytest.raises(PromptConfigValidationError) as excinfo:
        render_initial_user_message(config, cg_id="cg-123")
    # The error should name the missing variable.
    assert "workspace_path" in str(excinfo.value)


def test_render_extra_variable_is_silently_ignored() -> None:
    """Jinja2 happily accepts extra kwargs that the template doesn't use."""
    config = load_prompt_config("harness_builder")
    rendered = render_initial_user_message(
        config,
        cg_id="cg-1",
        workspace_path="/x",
        unused_var="ignored",
    )
    assert "ignored" not in rendered


def test_render_template_syntax_error_raises(tmp_path: Path) -> None:
    """Malformed Jinja2 syntax surfaces at render time."""
    override = {
        "layer": "override",
        "name": "broken",
        "role": "harness_builder",
        "initial_user_message_template": "{{ unclosed",
    }
    (tmp_path / "harness_builder.yaml").write_text(yaml.safe_dump(override))

    config = load_prompt_config("harness_builder", overrides_dir=tmp_path)
    with pytest.raises(PromptConfigValidationError):
        render_initial_user_message(config, cg_id="cg-1", workspace_path="/x")


def test_render_substitutes_loop_variables(tmp_path: Path) -> None:
    """Jinja2 control flow (loops) works as expected."""
    override = {
        "layer": "override",
        "name": "loop_demo",
        "role": "harness_builder",
        "initial_user_message_template": (
            "Items:\n{% for item in items %}- {{ item }}\n{% endfor %}"
        ),
    }
    (tmp_path / "harness_builder.yaml").write_text(yaml.safe_dump(override))

    config = load_prompt_config("harness_builder", overrides_dir=tmp_path)
    rendered = render_initial_user_message(config, items=["alpha", "beta", "gamma"])
    assert "- alpha" in rendered
    assert "- beta" in rendered
    assert "- gamma" in rendered


def test_render_planner_paper_ingestion_variant() -> None:
    """The paper_ingestion variant inherits base's template."""
    config = load_prompt_config("planner", variant_name="paper_ingestion")
    rendered = render_initial_user_message(
        config,
        source_kind="paper",
        source_id="2401.12345",
    )
    assert "paper" in rendered
    assert "2401.12345" in rendered
