"""Tests for the planner agent (`04-agents.md` §2.1, Task 3.E1).

Two surfaces under test:

* :func:`run_planner_session` — drives the loop end-to-end against a
  stubbed :class:`LlmProvider` returning canned tool calls; asserts
  the buffer ends up populated and the design plan persists to
  :class:`ArtifactStore`.
* Variant selection — both ``novel_technique`` and ``paper_ingestion``
  variants exercise their tool surfaces.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _agent_fakes import StubArtifactStore, StubLlmProvider  # type: ignore[import-not-found]
from _agent_helpers import model_response  # type: ignore[import-not-found]
from smai_agents import (
    AgentLoopConfig,
    PlannerBuffer,
    PlannerInput,
    PlannerSessionResult,
    clear_prompt_config_cache,
    run_planner_session,
)


@pytest.fixture(autouse=True)
def _reset_prompt_cache() -> None:
    """Reset the prompt-config cache before each test."""
    clear_prompt_config_cache()


def _set_classification_call(
    *,
    factor_dimension: str = "augmentation",
    factor_type: str = "additive",
) -> tuple[str, str, dict]:
    return (
        "tu-set-classification",
        "set_classification",
        {
            "factor_dimension": factor_dimension,
            "factor_type": factor_type,
            "rationale": "test rationale",
        },
    )


def _draft_create_technique_call(
    *,
    symbolic_name: str = "tech-cutout",
    name: str = "Cutout",
    standard: bool = False,
) -> tuple[str, str, dict]:
    payload: dict = {
        "symbolic_name": symbolic_name,
        "name": name,
        "description": f"{name} test description",
        "category": "augmentation",
        "compatible_factor_types": ["additive"],
        "standard": standard,
        "affects_extension_points": ["train_transforms"],
    }
    if not standard:
        payload["fidelity_anchor"] = {"kind": "proposal", "proposal_id": "prop-1"}
    return (f"tu-create-{symbolic_name}", "draft_create_technique", payload)


def _draft_comparison_call() -> tuple[str, str, dict]:
    return (
        "tu-draft-comparison",
        "draft_comparison",
        {
            "id": "cg-1",
            "hypothesis": "Cutout improves accuracy on CIFAR-10.",
            "factor_dimension": "augmentation",
            "factor_type": "additive",
            "factor_description": "augmentation policy",
            "entries": [
                {
                    "id": "entry-baseline",
                    "is_baseline": True,
                    "level": {
                        "factor": "augmentation",
                        "name": "absent",
                        "technique_symbolic_name": None,
                    },
                },
                {
                    "id": "entry-cutout",
                    "is_baseline": False,
                    "level": {
                        "factor": "augmentation",
                        "name": "cutout",
                        "technique_symbolic_name": "tech-cutout",
                    },
                },
            ],
        },
    )


def _set_conditions_call() -> tuple[str, str, dict]:
    return (
        "tu-set-conditions",
        "set_conditions",
        {
            "cg_id": "cg-1",
            "conditions": {
                "dataset": {"name": "cifar10", "split": "train", "version": "v1"},
                "optimization": {"optimizer": "sgd", "lr": 0.1},
                "seeds": [1, 2, 3],
            },
        },
    )


def _draft_assertion_call() -> tuple[str, str, dict]:
    return (
        "tu-draft-assertion",
        "draft_assertion",
        {
            "cg_id": "cg-1",
            "validation": {
                "metric": {"ref": "accuracy", "kind": "atomic"},
                "direction": "higher_is_better",
                "aggregation_method": "mean",
                "comparison_rule": "compare_to_baseline",
                "threshold": 0.01,
                "seed_count_required": 3,
            },
        },
    )


def _finalize_plan_call() -> tuple[str, str, dict]:
    return ("tu-finalize", "finalize_plan", {})


def _finish_call(success: bool = True) -> tuple[str, str, dict]:
    return ("tu-finish", "finish", {"success": success, "summary": "test finish"})


# === Novel-technique variant ===============================================


@pytest.mark.asyncio
async def test_novel_technique_variant_finalizes_plan(tmp_path: Path) -> None:
    """End-to-end: novel-technique variant drives finalize_plan + finish.

    The agent loop walks through a 7-call sequence:
    ``set_classification`` → ``draft_create_technique`` →
    ``draft_comparison`` → ``set_conditions`` → ``draft_assertion`` →
    ``finalize_plan`` → ``finish``. After the loop returns we inspect
    the buffer and the artifact store.
    """
    artifact_store = StubArtifactStore()

    responses = [
        model_response(tool_uses=[_set_classification_call()], stop_reason="tool_use"),
        model_response(
            tool_uses=[_draft_create_technique_call(symbolic_name="tech-cutout")],
            stop_reason="tool_use",
        ),
        model_response(tool_uses=[_draft_comparison_call()], stop_reason="tool_use"),
        model_response(tool_uses=[_set_conditions_call()], stop_reason="tool_use"),
        model_response(tool_uses=[_draft_assertion_call()], stop_reason="tool_use"),
        model_response(tool_uses=[_finalize_plan_call()], stop_reason="tool_use"),
        model_response(tool_uses=[_finish_call(True)], stop_reason="tool_use"),
    ]
    llm = StubLlmProvider(responses)
    planner_input = PlannerInput(
        variant="novel_technique",
        proposal_id="prop-1",
        submission_kind="novel_technique",
        technique_description="Test cutout technique",
        pool_summary="(empty pool)",
    )

    result = await run_planner_session(
        input=planner_input,
        llm=llm,  # type: ignore[arg-type]
        workspace_path=tmp_path / "ws",
        artifact_store=artifact_store,  # type: ignore[arg-type]
        config=AgentLoopConfig(max_turns=10),
    )

    assert isinstance(result, PlannerSessionResult)
    assert result.outcome.kind == "finished"
    assert result.outcome.finish_success is True
    assert result.buffer.finalized is True
    assert len(result.buffer.comparison_groups) == 1
    cg = result.buffer.comparison_groups[0]
    assert cg.id == "cg-1"
    assert len(cg.entries) == 2
    assert cg.validation is not None
    assert cg.validation.metric == {"ref": "accuracy", "kind": "atomic"}
    # Buffer was persisted.
    assert result.artifact_key.endswith("design_plan.json")
    assert result.artifact_key in artifact_store._data
    persisted = json.loads(artifact_store._data[result.artifact_key])
    assert persisted["finalized"] is True
    assert len(persisted["comparison_groups"]) == 1
    assert persisted["techniques"]["tech-cutout"]["fidelity_anchor"]["kind"] == "proposal"


@pytest.mark.asyncio
async def test_novel_technique_variant_finalize_returns_errors_on_missing_validation(
    tmp_path: Path,
) -> None:
    """finalize_plan should reject a buffer with no ValidationCriteria.

    The agent calls draft_comparison + finalize_plan but skips
    draft_assertion; finalize_plan must return an error result so the
    agent sees the rule failure.
    """
    artifact_store = StubArtifactStore()
    responses = [
        model_response(tool_uses=[_set_classification_call()], stop_reason="tool_use"),
        model_response(
            tool_uses=[_draft_create_technique_call(symbolic_name="tech-cutout")],
            stop_reason="tool_use",
        ),
        model_response(tool_uses=[_draft_comparison_call()], stop_reason="tool_use"),
        model_response(tool_uses=[_set_conditions_call()], stop_reason="tool_use"),
        # Skip draft_assertion!
        model_response(tool_uses=[_finalize_plan_call()], stop_reason="tool_use"),
        # Agent sees the error and finishes early.
        model_response(tool_uses=[_finish_call(False)], stop_reason="tool_use"),
    ]
    llm = StubLlmProvider(responses)
    planner_input = PlannerInput(
        variant="novel_technique",
        proposal_id="prop-1",
        submission_kind="novel_technique",
        technique_description="Test",
        pool_summary="",
    )

    result = await run_planner_session(
        input=planner_input,
        llm=llm,  # type: ignore[arg-type]
        workspace_path=tmp_path / "ws",
        artifact_store=artifact_store,  # type: ignore[arg-type]
        config=AgentLoopConfig(max_turns=10),
        max_finalize_reprompts=0,  # this test exercises finalize-rejection, not re-prompting
    )
    # Agent finished but the buffer was NOT finalized — finalize_plan
    # rejected the structure-incomplete buffer.
    assert result.outcome.kind == "finished"
    assert result.buffer.finalized is False
    assert result.artifact_key not in artifact_store._data


@pytest.mark.asyncio
async def test_novel_technique_search_techniques_matches_pool(tmp_path: Path) -> None:
    """search_techniques substring-matches against the pool snapshot."""
    artifact_store = StubArtifactStore()
    pool_summary = "tech-vgg | VGG | category=architecture | standard=False\n"
    pool_summary += "tech-resnet | ResNet | category=architecture | standard=False"
    responses = [
        model_response(
            tool_uses=[
                ("tu-search", "search_techniques", {"query": "vgg", "limit": 5}),
            ],
            stop_reason="tool_use",
        ),
        model_response(tool_uses=[_finish_call(True)], stop_reason="tool_use"),
    ]
    llm = StubLlmProvider(responses)
    planner_input = PlannerInput(
        variant="novel_technique",
        proposal_id="prop-1",
        submission_kind="novel_technique",
        technique_description="Test",
        pool_summary=pool_summary,
    )
    result = await run_planner_session(
        input=planner_input,
        llm=llm,  # type: ignore[arg-type]
        workspace_path=tmp_path / "ws",
        artifact_store=artifact_store,  # type: ignore[arg-type]
        config=AgentLoopConfig(max_turns=5),
        max_finalize_reprompts=0,
    )
    # The search_techniques tool result was appended to the conversation
    # as a tool_result with the matching line. Inspect only the
    # tool_result blocks (the pool_summary is also in the initial user
    # message, so a naive substring check would always match).
    assert result.outcome.kind == "finished"
    assert len(llm.calls) == 2
    second_turn_msgs = llm.calls[1]["messages"]
    tool_result_text = ""
    for msg in second_turn_msgs:  # type: ignore[union-attr]
        for block in msg["content"]:  # type: ignore[index]
            if block.get("type") == "tool_result":  # type: ignore[union-attr]
                # tool_result.content can be a string or a list of blocks.
                content = block.get("content")  # type: ignore[union-attr]
                if isinstance(content, str):
                    tool_result_text += content
                elif isinstance(content, list):
                    for sub in content:  # type: ignore[var-annotated]
                        if isinstance(sub, dict) and "text" in sub:
                            tool_result_text += sub["text"]  # type: ignore[arg-type]
    assert "VGG" in tool_result_text
    assert "ResNet" not in tool_result_text  # not matched by "vgg" query


@pytest.mark.asyncio
async def test_novel_technique_draft_create_rejects_unanchored_non_standard(
    tmp_path: Path,
) -> None:
    """A non-standard TechniqueRef must carry a fidelity_anchor (DEC-032)."""
    artifact_store = StubArtifactStore()
    responses = [
        model_response(
            tool_uses=[
                (
                    "tu-bad",
                    "draft_create_technique",
                    {
                        "symbolic_name": "tech-broken",
                        "name": "Broken",
                        "description": "missing anchor",
                        "category": "test",
                        "compatible_factor_types": ["additive"],
                        "standard": False,
                        # No fidelity_anchor!
                    },
                ),
            ],
            stop_reason="tool_use",
        ),
        model_response(tool_uses=[_finish_call(False)], stop_reason="tool_use"),
    ]
    llm = StubLlmProvider(responses)
    planner_input = PlannerInput(
        variant="novel_technique",
        proposal_id="prop-1",
        pool_summary="",
    )
    result = await run_planner_session(
        input=planner_input,
        llm=llm,  # type: ignore[arg-type]
        workspace_path=tmp_path / "ws",
        artifact_store=artifact_store,  # type: ignore[arg-type]
        config=AgentLoopConfig(max_turns=5),
        max_finalize_reprompts=0,
    )
    assert result.outcome.kind == "finished"
    # The technique should NOT be in the buffer (the tool returned
    # is_error=True before mutating).
    assert "tech-broken" not in result.buffer.techniques


# === Paper-ingestion variant ===============================================


@pytest.mark.asyncio
async def test_paper_ingestion_variant_finalizes(tmp_path: Path) -> None:
    """End-to-end: paper-ingestion variant drafts a paper-anchored
    technique and calls finalize_paper_techniques."""
    artifact_store = StubArtifactStore()
    responses = [
        model_response(
            tool_uses=[
                (
                    "tu-create",
                    "draft_create_technique",
                    {
                        "symbolic_name": "tech-novel-arch",
                        "name": "NovelArch",
                        "description": "paper contribution",
                        "category": "architecture",
                        "compatible_factor_types": ["substitutive"],
                        "standard": False,
                        "fidelity_anchor": {
                            "kind": "paper",
                            "doi": "arxiv:2401.12345",
                            "arxiv_id": "2401.12345",
                        },
                    },
                ),
            ],
            stop_reason="tool_use",
        ),
        model_response(
            tool_uses=[("tu-finalize", "finalize_paper_techniques", {})],
            stop_reason="tool_use",
        ),
        model_response(tool_uses=[_finish_call(True)], stop_reason="tool_use"),
    ]
    llm = StubLlmProvider(responses)
    planner_input = PlannerInput(
        variant="paper_ingestion",
        paper_arxiv_id="2401.12345",
        pool_summary="",
    )
    result = await run_planner_session(
        input=planner_input,
        llm=llm,  # type: ignore[arg-type]
        workspace_path=tmp_path / "ws",
        artifact_store=artifact_store,  # type: ignore[arg-type]
        config=AgentLoopConfig(max_turns=10),
    )
    assert result.outcome.kind == "finished"
    assert result.buffer.finalized is True
    # Paper-ingestion variant produces TechniqueRefs only — no CGs.
    assert len(result.buffer.comparison_groups) == 0
    assert "tech-novel-arch" in result.buffer.techniques
    # Buffer landed at the paper-keyed artifact path.
    assert result.artifact_key.endswith("planner_buffer.json")
    assert "papers/2401.12345" in result.artifact_key


@pytest.mark.asyncio
async def test_paper_ingestion_variant_rejects_empty_buffer(tmp_path: Path) -> None:
    """finalize_paper_techniques requires at least one paper-anchored
    TechniqueRef per `08` §5.4."""
    artifact_store = StubArtifactStore()
    responses = [
        model_response(
            tool_uses=[("tu-finalize", "finalize_paper_techniques", {})],
            stop_reason="tool_use",
        ),
        model_response(tool_uses=[_finish_call(False)], stop_reason="tool_use"),
    ]
    llm = StubLlmProvider(responses)
    planner_input = PlannerInput(
        variant="paper_ingestion",
        paper_arxiv_id="2401.12345",
        pool_summary="",
    )
    result = await run_planner_session(
        input=planner_input,
        llm=llm,  # type: ignore[arg-type]
        workspace_path=tmp_path / "ws",
        artifact_store=artifact_store,  # type: ignore[arg-type]
        config=AgentLoopConfig(max_turns=5),
        max_finalize_reprompts=0,
    )
    assert result.outcome.kind == "finished"
    assert result.buffer.finalized is False


# === Tool registry / variant selection ======================================


@pytest.mark.asyncio
async def test_variant_selects_distinct_tool_surfaces(tmp_path: Path) -> None:
    """The novel-technique variant declares CG-drafting tools; the
    paper-ingestion variant does NOT (per DEC-032's narrowing)."""
    # Drive both variants for one turn each and inspect the registered
    # tools the LLM saw.
    novel_responses = [
        model_response(text="ok", stop_reason="end_turn"),
    ]
    paper_responses = [
        model_response(text="ok", stop_reason="end_turn"),
    ]

    novel_llm = StubLlmProvider(novel_responses)
    novel_input = PlannerInput(
        variant="novel_technique",
        proposal_id="prop-1",
        pool_summary="",
    )
    await run_planner_session(
        input=novel_input,
        llm=novel_llm,  # type: ignore[arg-type]
        workspace_path=tmp_path / "ws-nt",
        artifact_store=None,
        config=AgentLoopConfig(max_turns=2),
        max_finalize_reprompts=0,
    )
    novel_tools = {t["name"] for t in novel_llm.calls[0]["tools"]}  # type: ignore[index, union-attr]
    assert "draft_comparison" in novel_tools
    assert "set_conditions" in novel_tools
    assert "draft_assertion" in novel_tools
    assert "finalize_plan" in novel_tools
    assert "finalize_paper_techniques" not in novel_tools

    paper_llm = StubLlmProvider(paper_responses)
    paper_input = PlannerInput(
        variant="paper_ingestion",
        paper_arxiv_id="2401.99999",
        pool_summary="",
    )
    await run_planner_session(
        input=paper_input,
        llm=paper_llm,  # type: ignore[arg-type]
        workspace_path=tmp_path / "ws-pi",
        artifact_store=None,
        config=AgentLoopConfig(max_turns=2),
        max_finalize_reprompts=0,
    )
    paper_tools = {t["name"] for t in paper_llm.calls[0]["tools"]}  # type: ignore[index, union-attr]
    # Per DEC-032 narrowing — these are absent.
    assert "draft_comparison" not in paper_tools
    assert "set_conditions" not in paper_tools
    assert "draft_assertion" not in paper_tools
    assert "finalize_plan" not in paper_tools
    # Present in paper-ingestion variant.
    assert "draft_create_technique" in paper_tools
    assert "draft_ensure_technique" in paper_tools
    assert "finalize_paper_techniques" in paper_tools


@pytest.mark.asyncio
async def test_planner_buffer_round_trip(tmp_path: Path) -> None:
    """The persisted buffer parses cleanly back as a :class:`PlannerBuffer`."""
    artifact_store = StubArtifactStore()
    responses = [
        model_response(tool_uses=[_set_classification_call()], stop_reason="tool_use"),
        model_response(
            tool_uses=[_draft_create_technique_call(symbolic_name="tech-cutout")],
            stop_reason="tool_use",
        ),
        model_response(tool_uses=[_draft_comparison_call()], stop_reason="tool_use"),
        model_response(tool_uses=[_set_conditions_call()], stop_reason="tool_use"),
        model_response(tool_uses=[_draft_assertion_call()], stop_reason="tool_use"),
        model_response(tool_uses=[_finalize_plan_call()], stop_reason="tool_use"),
        model_response(tool_uses=[_finish_call(True)], stop_reason="tool_use"),
    ]
    llm = StubLlmProvider(responses)
    planner_input = PlannerInput(
        variant="novel_technique",
        proposal_id="prop-1",
        pool_summary="",
    )
    result = await run_planner_session(
        input=planner_input,
        llm=llm,  # type: ignore[arg-type]
        workspace_path=tmp_path / "ws",
        artifact_store=artifact_store,  # type: ignore[arg-type]
        config=AgentLoopConfig(max_turns=10),
    )
    raw = artifact_store._data[result.artifact_key]
    parsed = PlannerBuffer.model_validate_json(raw)
    assert parsed.finalized is True
    assert len(parsed.comparison_groups) == 1
    assert parsed.proposal_id == "prop-1"


# === Round-6 friction B: finalize re-prompting ==============================


@pytest.mark.asyncio
async def test_planner_reprompts_when_buffer_not_finalized(tmp_path: Path) -> None:
    """The loop returns ``end_turn`` (no tool) with the buffer un-finalized
    → :func:`run_planner_session` re-prompts up to the bound, then returns
    ``finalized=False`` with the re-prompt count recorded."""
    # 1 initial run + 3 re-prompts = 4 loop runs; each ends immediately.
    responses = [
        model_response(text="nothing more to say", stop_reason="end_turn") for _ in range(4)
    ]
    llm = StubLlmProvider(responses)
    planner_input = PlannerInput(variant="novel_technique", proposal_id="prop-1", pool_summary="")
    result = await run_planner_session(
        input=planner_input,
        llm=llm,  # type: ignore[arg-type]
        workspace_path=tmp_path / "ws",
        artifact_store=None,
        config=AgentLoopConfig(max_turns=4),
        max_finalize_reprompts=3,
    )
    assert result.buffer.finalized is False
    assert result.finalize_reprompts == 3
    assert len(llm.calls) == 4  # one LLM call per loop run
    # The re-prompt nudges reference the finalize tool.
    assert "finalize_plan" in json.dumps(llm.calls[-1]["messages"])


@pytest.mark.asyncio
async def test_planner_reprompt_succeeds_on_second_run(tmp_path: Path) -> None:
    """The first loop run stops without finalizing; the re-prompt drives a
    full draft → finalize → finish run that succeeds."""
    artifact_store = StubArtifactStore()
    responses = [
        model_response(text="hmm, done?", stop_reason="end_turn"),  # run 1: stops short
        model_response(tool_uses=[_set_classification_call()], stop_reason="tool_use"),
        model_response(
            tool_uses=[_draft_create_technique_call(symbolic_name="tech-cutout")],
            stop_reason="tool_use",
        ),
        model_response(tool_uses=[_draft_comparison_call()], stop_reason="tool_use"),
        model_response(tool_uses=[_set_conditions_call()], stop_reason="tool_use"),
        model_response(tool_uses=[_draft_assertion_call()], stop_reason="tool_use"),
        model_response(tool_uses=[_finalize_plan_call()], stop_reason="tool_use"),
        model_response(tool_uses=[_finish_call(True)], stop_reason="tool_use"),
    ]
    llm = StubLlmProvider(responses)
    planner_input = PlannerInput(
        variant="novel_technique",
        proposal_id="prop-1",
        submission_kind="novel_technique",
        technique_description="t",
        pool_summary="",
    )
    result = await run_planner_session(
        input=planner_input,
        llm=llm,  # type: ignore[arg-type]
        workspace_path=tmp_path / "ws",
        artifact_store=artifact_store,  # type: ignore[arg-type]
        config=AgentLoopConfig(max_turns=20),
        max_finalize_reprompts=3,
    )
    assert result.buffer.finalized is True
    assert result.finalize_reprompts == 1
    assert result.artifact_key in artifact_store._data


@pytest.mark.asyncio
async def test_planner_finish_before_finalize_not_accepted(tmp_path: Path) -> None:
    """Calling ``finish`` before ``finalize_plan`` succeeded does NOT count
    as a successful finalize — :func:`run_planner_session` keeps re-prompting
    until the bound is exhausted."""
    responses = [
        model_response(tool_uses=[_finish_call(True)], stop_reason="tool_use") for _ in range(4)
    ]
    llm = StubLlmProvider(responses)
    planner_input = PlannerInput(variant="novel_technique", proposal_id="prop-1", pool_summary="")
    result = await run_planner_session(
        input=planner_input,
        llm=llm,  # type: ignore[arg-type]
        workspace_path=tmp_path / "ws",
        artifact_store=None,
        config=AgentLoopConfig(max_turns=4),
        max_finalize_reprompts=3,
    )
    assert result.buffer.finalized is False
    assert result.finalize_reprompts == 3


@pytest.mark.asyncio
async def test_planner_persists_conversation_trace(tmp_path: Path) -> None:
    """``run_loop`` serializes the conversation to the planner trace key on
    loop exit (round-6 item 5 — nothing wrote it before)."""
    from smai_agents.agents.planner import DEFAULT_PLANNER_TRACE_KEY_TEMPLATE  # noqa: PLC0415

    artifact_store = StubArtifactStore()
    llm = StubLlmProvider([model_response(text="done", stop_reason="end_turn")])
    planner_input = PlannerInput(
        variant="novel_technique", proposal_id="prop-trace", pool_summary=""
    )
    await run_planner_session(
        input=planner_input,
        llm=llm,  # type: ignore[arg-type]
        workspace_path=tmp_path / "ws",
        artifact_store=artifact_store,  # type: ignore[arg-type]
        config=AgentLoopConfig(max_turns=2),
        max_finalize_reprompts=0,
    )
    key = DEFAULT_PLANNER_TRACE_KEY_TEMPLATE.format(proposal_id="prop-trace")
    assert key in artifact_store._data
    parsed = json.loads(artifact_store._data[key])
    assert isinstance(parsed, list) and parsed
    assert parsed[0]["role"] == "user"


# === Round-8 friction A: finalize_plan projects the buffer ==================
#
# Pre-round-8 ``_structural_check_novel_technique`` only verified field
# *presence* on controlled_conditions — a buffer with ``dataset="MNIST"``
# (bare string) sailed past finalize_plan and wedged the proposal at
# registration time. round-8 wires the same Pydantic projection the
# registration handler uses so type/shape errors surface in-loop and the
# re-prompt machinery lets the agent self-correct.


def _set_conditions_call_with(conditions: dict) -> tuple[str, str, dict]:
    """Variant of :func:`_set_conditions_call` with custom conditions."""
    return (
        "tu-set-conditions",
        "set_conditions",
        {"cg_id": "cg-1", "conditions": conditions},
    )


@pytest.mark.asyncio
async def test_finalize_plan_rejects_dataset_as_bare_string(tmp_path: Path) -> None:
    """The reproduced bug: ``controlled_conditions.dataset = "MNIST"`` (a
    bare string instead of the typed dict) fails projection — finalize_plan
    must return is_error mentioning the field path so the agent can fix it."""
    artifact_store = StubArtifactStore()
    bad_conditions = {
        "dataset": "MNIST",  # WRONG — must be {"name": "MNIST", ...}
        "optimization": {"optimizer": "sgd", "lr": 0.1},
        "seeds": [0, 1],
    }
    responses = [
        model_response(tool_uses=[_set_classification_call()], stop_reason="tool_use"),
        model_response(
            tool_uses=[_draft_create_technique_call(symbolic_name="tech-cutout")],
            stop_reason="tool_use",
        ),
        model_response(tool_uses=[_draft_comparison_call()], stop_reason="tool_use"),
        model_response(
            tool_uses=[_set_conditions_call_with(bad_conditions)], stop_reason="tool_use"
        ),
        model_response(tool_uses=[_draft_assertion_call()], stop_reason="tool_use"),
        model_response(tool_uses=[_finalize_plan_call()], stop_reason="tool_use"),
        model_response(tool_uses=[_finish_call(False)], stop_reason="tool_use"),
    ]
    llm = StubLlmProvider(responses)
    planner_input = PlannerInput(
        variant="novel_technique",
        proposal_id="prop-bad-ds",
        submission_kind="novel_technique",
        technique_description="t",
        pool_summary="",
    )
    result = await run_planner_session(
        input=planner_input,
        llm=llm,  # type: ignore[arg-type]
        workspace_path=tmp_path / "ws",
        artifact_store=artifact_store,  # type: ignore[arg-type]
        config=AgentLoopConfig(max_turns=10),
        max_finalize_reprompts=0,
    )
    # The buffer was NOT finalized — projection rejected it.
    assert result.buffer.finalized is False
    assert result.artifact_key not in artifact_store._data
    # The finalize_plan tool result mentions the field path so the agent
    # can correct on its next turn (a tool_result with is_error=True is
    # what the re-prompt machinery / live agent actually sees).
    second_call_msgs = llm.calls[6]["messages"]  # 7th LLM call sees finalize_plan's tool_result
    tool_result_text = ""
    for msg in second_call_msgs:  # type: ignore[union-attr]
        for block in msg["content"]:  # type: ignore[index]
            if block.get("type") == "tool_result":  # type: ignore[union-attr]
                if block.get("is_error") is not True:  # type: ignore[union-attr]
                    continue
                content = block.get("content")  # type: ignore[union-attr]
                if isinstance(content, str):
                    tool_result_text += content
                elif isinstance(content, list):
                    for sub in content:  # type: ignore[var-annotated]
                        if isinstance(sub, dict) and "text" in sub:
                            tool_result_text += sub["text"]  # type: ignore[arg-type]
    assert "controlled_conditions.dataset" in tool_result_text
    # A hint for what the agent should put in the dataset slot.
    assert "dict" in tool_result_text.lower() or "{" in tool_result_text


@pytest.mark.asyncio
async def test_finalize_plan_rejects_optimization_not_a_dict(tmp_path: Path) -> None:
    """A second projection failure shape — ``optimization`` as a list
    instead of a dict — must also be surfaced via the tool result with the
    field path. Confirms the projection check catches multiple error
    shapes, not just the dataset-string one."""
    artifact_store = StubArtifactStore()
    bad_conditions = {
        "dataset": {"name": "MNIST", "split": "train"},
        "optimization": ["sgd", 0.1],  # WRONG — must be a dict
        "seeds": [0, 1],
    }
    responses = [
        model_response(tool_uses=[_set_classification_call()], stop_reason="tool_use"),
        model_response(
            tool_uses=[_draft_create_technique_call(symbolic_name="tech-cutout")],
            stop_reason="tool_use",
        ),
        model_response(tool_uses=[_draft_comparison_call()], stop_reason="tool_use"),
        model_response(
            tool_uses=[_set_conditions_call_with(bad_conditions)], stop_reason="tool_use"
        ),
        model_response(tool_uses=[_draft_assertion_call()], stop_reason="tool_use"),
        model_response(tool_uses=[_finalize_plan_call()], stop_reason="tool_use"),
        model_response(tool_uses=[_finish_call(False)], stop_reason="tool_use"),
    ]
    llm = StubLlmProvider(responses)
    planner_input = PlannerInput(
        variant="novel_technique",
        proposal_id="prop-bad-opt",
        submission_kind="novel_technique",
        technique_description="t",
        pool_summary="",
    )
    result = await run_planner_session(
        input=planner_input,
        llm=llm,  # type: ignore[arg-type]
        workspace_path=tmp_path / "ws",
        artifact_store=artifact_store,  # type: ignore[arg-type]
        config=AgentLoopConfig(max_turns=10),
        max_finalize_reprompts=0,
    )
    assert result.buffer.finalized is False
    second_call_msgs = llm.calls[6]["messages"]  # 7th LLM call sees finalize_plan's tool_result
    tool_result_text = ""
    for msg in second_call_msgs:  # type: ignore[union-attr]
        for block in msg["content"]:  # type: ignore[index]
            if block.get("type") == "tool_result" and block.get("is_error") is True:  # type: ignore[union-attr]
                content = block.get("content")  # type: ignore[union-attr]
                if isinstance(content, str):
                    tool_result_text += content
                elif isinstance(content, list):
                    for sub in content:  # type: ignore[var-annotated]
                        if isinstance(sub, dict) and "text" in sub:
                            tool_result_text += sub["text"]  # type: ignore[arg-type]
    assert "controlled_conditions.optimization" in tool_result_text


@pytest.mark.asyncio
async def test_finalize_plan_accepts_well_shaped_controlled_conditions(
    tmp_path: Path,
) -> None:
    """A buffer with the typed-dict controlled_conditions shape projects
    cleanly and ``finalize_plan`` succeeds — confirms the round-8
    projection check doesn't reject the happy-path shape that
    ``test_novel_technique_variant_finalizes_plan`` already establishes."""
    artifact_store = StubArtifactStore()
    good_conditions = {
        "dataset": {"name": "MNIST", "split": "standard"},
        "optimization": {"optimizer": "sgd", "lr": 0.1},
        "seeds": [0, 1],
    }
    responses = [
        model_response(tool_uses=[_set_classification_call()], stop_reason="tool_use"),
        model_response(
            tool_uses=[_draft_create_technique_call(symbolic_name="tech-cutout")],
            stop_reason="tool_use",
        ),
        model_response(tool_uses=[_draft_comparison_call()], stop_reason="tool_use"),
        model_response(
            tool_uses=[_set_conditions_call_with(good_conditions)], stop_reason="tool_use"
        ),
        model_response(tool_uses=[_draft_assertion_call()], stop_reason="tool_use"),
        model_response(tool_uses=[_finalize_plan_call()], stop_reason="tool_use"),
        model_response(tool_uses=[_finish_call(True)], stop_reason="tool_use"),
    ]
    llm = StubLlmProvider(responses)
    planner_input = PlannerInput(
        variant="novel_technique",
        proposal_id="prop-ok",
        submission_kind="novel_technique",
        technique_description="t",
        pool_summary="",
    )
    result = await run_planner_session(
        input=planner_input,
        llm=llm,  # type: ignore[arg-type]
        workspace_path=tmp_path / "ws",
        artifact_store=artifact_store,  # type: ignore[arg-type]
        config=AgentLoopConfig(max_turns=10),
    )
    assert result.buffer.finalized is True
    assert result.artifact_key in artifact_store._data


# === Round-9 fix B: finalize_plan runs the orchestrator-level record check =


def _draft_comparison_call_with_entries(entries: list[dict]) -> tuple[str, str, dict]:
    """Variant of :func:`_draft_comparison_call` with overridable entries.

    Used to inject a record-level Pydantic failure (e.g. an entry id with
    whitespace) that the methodology projection accepts but
    :class:`EntryRecord`'s id-format validator rejects.
    """
    base = _draft_comparison_call()
    payload = dict(base[2])
    payload["entries"] = entries
    return (base[0], base[1], payload)


@pytest.mark.asyncio
async def test_finalize_plan_rejects_entry_id_with_whitespace(tmp_path: Path) -> None:
    """Round-9 fix B: an entry id with embedded whitespace passes the
    methodology layer's ``Entry.id: str`` (no format constraint) but fails
    :class:`smai_orchestrator.entities.tracking.EntryRecord`'s
    ``validate_id_format``. The planner's ``finalize_plan`` runs the
    orchestrator-level dry-run record check after the methodology
    projection succeeds, so this failure surfaces in-loop with the field
    path."""
    artifact_store = StubArtifactStore()
    bad_entries = [
        {
            "id": "entry baseline with space",  # WRONG — whitespace rejected
            "is_baseline": True,
            "level": {
                "factor": "augmentation",
                "name": "absent",
                "technique_symbolic_name": None,
            },
        },
        {
            "id": "entry-cutout",
            "is_baseline": False,
            "level": {
                "factor": "augmentation",
                "name": "cutout",
                "technique_symbolic_name": "tech-cutout",
            },
        },
    ]
    responses = [
        model_response(tool_uses=[_set_classification_call()], stop_reason="tool_use"),
        model_response(
            tool_uses=[_draft_create_technique_call(symbolic_name="tech-cutout")],
            stop_reason="tool_use",
        ),
        model_response(
            tool_uses=[_draft_comparison_call_with_entries(bad_entries)],
            stop_reason="tool_use",
        ),
        model_response(tool_uses=[_set_conditions_call()], stop_reason="tool_use"),
        model_response(tool_uses=[_draft_assertion_call()], stop_reason="tool_use"),
        model_response(tool_uses=[_finalize_plan_call()], stop_reason="tool_use"),
        model_response(tool_uses=[_finish_call(False)], stop_reason="tool_use"),
    ]
    llm = StubLlmProvider(responses)
    planner_input = PlannerInput(
        variant="novel_technique",
        proposal_id="prop-bad-entry",
        submission_kind="novel_technique",
        technique_description="t",
        pool_summary="",
    )
    result = await run_planner_session(
        input=planner_input,
        llm=llm,  # type: ignore[arg-type]
        workspace_path=tmp_path / "ws",
        artifact_store=artifact_store,  # type: ignore[arg-type]
        config=AgentLoopConfig(max_turns=10),
        max_finalize_reprompts=0,
    )
    assert result.buffer.finalized is False
    assert result.artifact_key not in artifact_store._data
    # The finalize_plan tool result mentions the offending entry id +
    # the entries.<id> path so the agent can correct on its next turn.
    second_call_msgs = llm.calls[6]["messages"]
    tool_result_text = ""
    for msg in second_call_msgs:  # type: ignore[union-attr]
        for block in msg["content"]:  # type: ignore[index]
            if block.get("type") == "tool_result" and block.get("is_error") is True:  # type: ignore[union-attr]
                content = block.get("content")  # type: ignore[union-attr]
                if isinstance(content, str):
                    tool_result_text += content
                elif isinstance(content, list):
                    for sub in content:  # type: ignore[var-annotated]
                        if isinstance(sub, dict) and "text" in sub:
                            tool_result_text += sub["text"]  # type: ignore[arg-type]
    assert "entry baseline with space" in tool_result_text
    # Field path mentions "entries" — the dry-run helper includes it in
    # its loc string.
    assert "entries" in tool_result_text


@pytest.mark.asyncio
async def test_finalize_plan_reprompts_self_correct_after_bad_dataset(
    tmp_path: Path,
) -> None:
    """The end-to-end self-correct path: first call ships a bad dataset →
    projection rejects → re-prompt fires → agent re-issues set_conditions
    with the typed-dict shape → finalize_plan succeeds. The bug under fix
    was that the planner had no in-loop signal to correct on; this test
    confirms the signal is delivered AND the re-prompt machinery can pick
    up the corrected buffer to a successful finalize."""
    artifact_store = StubArtifactStore()
    bad = {
        "dataset": "MNIST",
        "optimization": {"optimizer": "sgd", "lr": 0.1},
        "seeds": [0, 1],
    }
    good = {
        "dataset": {"name": "MNIST", "split": "standard"},
        "optimization": {"optimizer": "sgd", "lr": 0.1},
        "seeds": [0, 1],
    }
    responses = [
        # First loop run: classification → technique → CG → BAD conditions
        # → assertion → finalize (rejected by projection) → finish(False).
        model_response(tool_uses=[_set_classification_call()], stop_reason="tool_use"),
        model_response(
            tool_uses=[_draft_create_technique_call(symbolic_name="tech-cutout")],
            stop_reason="tool_use",
        ),
        model_response(tool_uses=[_draft_comparison_call()], stop_reason="tool_use"),
        model_response(tool_uses=[_set_conditions_call_with(bad)], stop_reason="tool_use"),
        model_response(tool_uses=[_draft_assertion_call()], stop_reason="tool_use"),
        model_response(tool_uses=[_finalize_plan_call()], stop_reason="tool_use"),
        model_response(tool_uses=[_finish_call(False)], stop_reason="tool_use"),
        # Round-6 re-prompt fires; second loop run corrects conditions
        # then re-finalizes.
        model_response(tool_uses=[_set_conditions_call_with(good)], stop_reason="tool_use"),
        model_response(tool_uses=[_finalize_plan_call()], stop_reason="tool_use"),
        model_response(tool_uses=[_finish_call(True)], stop_reason="tool_use"),
    ]
    llm = StubLlmProvider(responses)
    planner_input = PlannerInput(
        variant="novel_technique",
        proposal_id="prop-self-correct",
        submission_kind="novel_technique",
        technique_description="t",
        pool_summary="",
    )
    result = await run_planner_session(
        input=planner_input,
        llm=llm,  # type: ignore[arg-type]
        workspace_path=tmp_path / "ws",
        artifact_store=artifact_store,  # type: ignore[arg-type]
        config=AgentLoopConfig(max_turns=20),
        max_finalize_reprompts=3,
    )
    assert result.buffer.finalized is True
    assert result.finalize_reprompts == 1
    assert result.artifact_key in artifact_store._data
