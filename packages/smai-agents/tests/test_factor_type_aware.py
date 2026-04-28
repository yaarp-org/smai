"""Factor-type-aware framing — DEC-017 / 10-runtime-and-templates.md §9.

Per the Task 2.B3 brief: "additive baseline (entry with technique_id=null)
gets framed differently in the harness-builder prompt than substitutive
baseline; verify via rendered initial user message."

The harness builder always builds the same code shape; what differs is
the initial user message's framing — the prompt template branches on
``factor_type`` so the agent knows whether the extension point has a
working default (additive) or is a mandatory slot (substitutive).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _agent_fakes import StubArtifactStore, StubLlmProvider  # type: ignore[import-not-found]
from _b2_fakes import FakeCompute  # type: ignore[import-not-found]
from _b3_fakes import (  # type: ignore[import-not-found]
    make_harness_contract,
)
from smai_agents import (
    AgentLoopConfig,
    AgentOutcome,
    AgentSession,
    load_prompt_config,
    render_initial_user_message,
    run_harness_builder_session,
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


# === Through the dispatch session: rendered message lands in the LLM call ==


@pytest.mark.asyncio
async def test_run_harness_builder_session_threads_factor_type_into_initial_message(
    tmp_path: Path,
) -> None:
    """The factor type from HarnessContract.body.factor.type lands in
    the rendered initial user message that the agent sees on its first
    turn."""
    contract = make_harness_contract(
        factor_type="substitutive",
        factor_name="architecture",
    )
    workspace = tmp_path / "ws"

    captured: list[AgentSession] = []

    async def _capture(session: AgentSession) -> AgentOutcome:
        captured.append(session)
        return AgentOutcome(
            kind="finished",
            turn_count=0,
            usage_total=session.usage_total,
            finish_success=True,
            finish_summary="captured",
        )

    await run_harness_builder_session(
        workspace_path=workspace,
        harness_contract=contract,
        cg_id="cg-substitutive",
        llm=StubLlmProvider([]),
        artifact_store=StubArtifactStore(),
        compute=FakeCompute(),
        manifest_artifact_path="cg-substitutive/harness/manifest.json",
        config=AgentLoopConfig(status_write_every_turns=0),
        runner=_capture,
    )

    assert len(captured) == 1
    session = captured[0]
    # First message is the rendered initial user message.
    first_msg = session.messages[0]
    assert first_msg.role == "user"
    body = first_msg.content[0]
    text = getattr(body, "text", "")
    # Substitutive framing surfaces.
    assert "substitutive" in text
    assert "architecture" in text
    assert "mandatory slots" in text


@pytest.mark.asyncio
async def test_run_harness_builder_session_additive_threading(
    tmp_path: Path,
) -> None:
    """Additive contracts thread the additive-specific framing through."""
    contract = make_harness_contract(
        factor_type="additive",
        factor_name="augmentation",
    )
    workspace = tmp_path / "ws"

    captured: list[AgentSession] = []

    async def _capture(session: AgentSession) -> AgentOutcome:
        captured.append(session)
        return AgentOutcome(
            kind="finished",
            turn_count=0,
            usage_total=session.usage_total,
            finish_success=True,
            finish_summary="captured",
        )

    await run_harness_builder_session(
        workspace_path=workspace,
        harness_contract=contract,
        cg_id="cg-additive",
        llm=StubLlmProvider([]),
        artifact_store=StubArtifactStore(),
        compute=FakeCompute(),
        manifest_artifact_path="cg-additive/harness/manifest.json",
        config=AgentLoopConfig(status_write_every_turns=0),
        runner=_capture,
    )

    assert len(captured) == 1
    text = getattr(captured[0].messages[0].content[0], "text", "")
    assert "additive" in text
    assert "augmentation" in text
    assert "optional=true" in text
    assert "no contribution" in text or "as-is" in text


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
