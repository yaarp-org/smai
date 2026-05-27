"""Factor-type-aware framing — DEC-017 / 10-runtime-and-templates.md §9.

Per the Task 2.B3 brief: "additive baseline (entry with technique_id=null)
gets framed differently in the harness-builder prompt than substitutive
baseline; verify via rendered initial user message."

The harness builder always builds the same code shape; what differs is
the initial user message's framing — the prompt template branches on
``factor_type`` so the agent knows whether the extension point has a
working default (additive) or is a mandatory slot (substitutive).

Sub-PR E cutover note: the previous two
``test_run_harness_builder_session_*`` cases that drove the in-process
session-runner via an ``inline_runner`` seam are gone — the harness
builder no longer runs in-process. Equivalent end-to-end framing
coverage will land alongside Step 6's real-LLM dogfooding (recorded
fixture for the sandbox-side prompt path); the template-rendering tests
below already pin the additive vs substitutive branch.
"""

from __future__ import annotations

from smai_inline_agents import (
    load_prompt_config,
    render_initial_user_message,
)

# === Direct template rendering — additive vs substitutive framing ===========


def test_harness_builder_template_renders_additive_framing() -> None:
    """The base template's branch for ``factor_type=additive`` contains
    the additive-specific framing language."""
    config = load_prompt_config("harness_builder")
    rendered = render_initial_user_message(
        config,
        cg_id="cg-add",
        workspace_path="/tmp/ws",
        factor_type="additive",
        factor_dimension="augmentation",
        manifest_artifact_path="cg-add/harness/manifest.json",
    )
    assert "additive" in rendered
    assert "augmentation" in rendered
    # Additive-specific language: "no-op" / "as-is" / "optional=true".
    assert "optional=true" in rendered
    assert "substitutive" not in rendered.lower() or "**substitutive**" not in rendered


def test_harness_builder_template_renders_substitutive_framing() -> None:
    """The base template's branch for ``factor_type=substitutive``
    contains the substitutive-specific framing language."""
    config = load_prompt_config("harness_builder")
    rendered = render_initial_user_message(
        config,
        cg_id="cg-sub",
        workspace_path="/tmp/ws",
        factor_type="substitutive",
        factor_dimension="architecture",
        manifest_artifact_path="cg-sub/harness/manifest.json",
    )
    assert "substitutive" in rendered
    assert "architecture" in rendered
    # Substitutive-specific language: "mandatory" / "optional=false".
    assert "optional=false" in rendered
    assert "mandatory slots" in rendered


# === Technique implementer template: the three context kinds ===============


def test_technique_implementer_template_renders_method_description_context() -> None:
    config = load_prompt_config("technique_implementer")
    rendered = render_initial_user_message(
        config,
        cg_id="cg-1",
        entry_id="entry-1",
        technique_id="tq-1",
        technique_name="cutout",
        workspace_path="/tmp/ws",
        factor_type="additive",
        factor_dimension="augmentation",
        context_kind="method_description",
        grounding_path="techniques/tq-1/method_description.json",
        review_feedback=None,
        implementation_attempt=0,
    )
    assert "method description" in rendered
    assert "novel-technique pipeline" in rendered


def test_technique_implementer_template_renders_description_only_context() -> None:
    config = load_prompt_config("technique_implementer")
    rendered = render_initial_user_message(
        config,
        cg_id="cg-1",
        entry_id="entry-2",
        technique_id="tq-dropout",
        technique_name="dropout",
        workspace_path="/tmp/ws",
        factor_type="additive",
        factor_dimension="regularization",
        context_kind="description_only",
        grounding_path=None,
        review_feedback=None,
        implementation_attempt=0,
    )
    assert "standard technique" in rendered
    assert "library APIs" in rendered


def test_technique_implementer_template_renders_paper_extract_context() -> None:
    config = load_prompt_config("technique_implementer")
    rendered = render_initial_user_message(
        config,
        cg_id="cg-1",
        entry_id="entry-3",
        technique_id="tq-mixup",
        technique_name="mixup",
        workspace_path="/tmp/ws",
        factor_type="additive",
        factor_dimension="augmentation",
        context_kind="paper_extract",
        grounding_path="papers/2401.12345/techniques/mixup/method_extraction.json",
        review_feedback=None,
        implementation_attempt=0,
    )
    assert "paper extract" in rendered
    assert "PaperFidelityAnchor" in rendered
