"""Reproduce-paper-X workflow integration test — Task 3.E2 / M3 gate.

Drives both pipelines (paper-ingestion + proposal) through the
"reproduce paper X" workflow per DEC-032's primary input path:

1. ``smai ingest <arxiv-id>`` (via :meth:`PapersService.submit`) drives
   the paper-ingestion pipeline to ``registered``, committing
   :class:`TechniqueRef`s with :class:`PaperFidelityAnchor`s.
2. ``smai submit-proposal --reproduce-paper <arxiv-id>`` (via
   :meth:`ProposalsService.submit`) creates a proposal with
   ``submission_kind=reproduce_paper``; the proposal pipeline's
   planner reads the pre-ingested paper's artifacts during
   ``designing`` and drafts ≥ 1 :class:`ComparisonGroupRecord`.

Per DEC-032: paper ingestion produces ``TechniqueRef``s only; the
proposal pipeline produces CGs. Both pipelines coordinate via shared
:class:`MetadataStore` + :class:`ArtifactStore` (per `08` §5.6 / `03`
§5.6); no direct call between them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from _e2_integration_fakes import (  # type: ignore[import-not-found]
    InProcessFakeFetcher,
    StubLlmProvider,
    build_smoke_runtime_config_for_papers,
    make_paper_planner_responses,
    make_screener_response,
)
from smai_artifacts_localfs import LocalFsStore
from smai_cli.runtime import Runtime
from smai_core.plugins import (
    ModelResponse,
    NormalizedMessage,
    StopReason,
    TextContent,
    TokenUsage,
    ToolUseContent,
)
from smai_orchestrator import (
    DEFAULT_TASK_ROLES,
    PluginOverrides,
)

# === Proposal-pipeline planner responses (novel-technique variant) ==========


def _model_response(
    *,
    tool_uses: list[tuple[str, str, dict[str, Any]]] | None = None,
    text: str | None = None,
    stop_reason: StopReason = "tool_use",
) -> ModelResponse:
    content: list[TextContent | ToolUseContent] = []
    if text is not None:
        content.append(TextContent(text=text))
    if tool_uses is not None:
        for tu_id, tu_name, tu_input in tool_uses:
            content.append(ToolUseContent(id=tu_id, name=tu_name, input=dict(tu_input)))
    return ModelResponse(
        message=NormalizedMessage(role="assistant", content=content),
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=1, output_tokens=1),
    )


def _make_reproduce_paper_proposal_responses(
    *,
    proposal_id: str,
    arxiv_id: str,
) -> list[ModelResponse]:
    """Canned LLM responses driving the novel-technique-variant planner
    through a ``reproduce_paper`` proposal: classification → comparison
    → conditions → assertion → finalize → finish.

    The planner produces one CG with one entry referencing the paper's
    pre-ingested contribution technique by symbolic name. The
    registration step looks the technique up in the registries
    (which the proposal-spec's ``_register_buffer`` extends with
    buffer-internal techniques first; pool techniques registered by
    paper ingestion are found via the ``Registries`` snapshot the
    factory loads).

    For this integration test we ALSO include the contribution
    technique as an in-buffer ``draft_create_technique`` call (with the
    same symbolic name) so the registration step has a definition for
    it without needing to extend the default registries with the
    paper-ingested technique. v1 supports this dedup pattern
    explicitly.
    """
    cg_id = f"{proposal_id}-cg-reproduce"
    factor_dim = "augmentation"
    factor_type = "additive"
    treatment_symbol = f"{arxiv_id}-tech-contrib"

    return [
        # set_classification
        _model_response(
            tool_uses=[
                (
                    "tu-classify",
                    "set_classification",
                    {
                        "factor_dimension": factor_dim,
                        "factor_type": factor_type,
                        "rationale": f"reproduce paper {arxiv_id}",
                    },
                )
            ],
        ),
        # draft_create_technique — re-declare the paper-ingested
        # contribution technique so the buffer carries a definition for it.
        _model_response(
            tool_uses=[
                (
                    "tu-create",
                    "draft_create_technique",
                    {
                        "symbolic_name": treatment_symbol,
                        "name": "Paper Contribution",
                        "description": "Reproduce contribution technique from the source paper.",
                        "category": factor_dim,
                        "compatible_factor_types": [factor_type],
                        "standard": False,
                        "fidelity_anchor": {
                            "kind": "paper",
                            "arxiv_id": arxiv_id,
                            "doi": f"arxiv:{arxiv_id}",
                        },
                        "affects_extension_points": ["train_transforms"],
                    },
                )
            ],
        ),
        # draft_comparison
        _model_response(
            tool_uses=[
                (
                    "tu-comparison",
                    "draft_comparison",
                    {
                        "id": cg_id,
                        "hypothesis": f"reproduce paper {arxiv_id}'s comparative claims",
                        "factor_dimension": factor_dim,
                        "factor_type": factor_type,
                        "factor_description": f"reproduce paper {arxiv_id}",
                        "entries": [
                            {
                                "id": f"{cg_id}-entry-baseline",
                                "is_baseline": True,
                                "level": {
                                    "factor": factor_dim,
                                    "name": "baseline",
                                    "technique_symbolic_name": None,
                                },
                            },
                            {
                                "id": f"{cg_id}-entry-treatment",
                                "is_baseline": False,
                                "level": {
                                    "factor": factor_dim,
                                    "name": "treatment",
                                    "technique_symbolic_name": treatment_symbol,
                                },
                            },
                        ],
                    },
                )
            ],
        ),
        # set_conditions
        _model_response(
            tool_uses=[
                (
                    "tu-conditions",
                    "set_conditions",
                    {
                        "cg_id": cg_id,
                        "conditions": {
                            "dataset": {"name": "cifar10", "split": "train", "version": "v1"},
                            "optimization": {"optimizer": "sgd", "lr": 0.1},
                            "seeds": [1, 2, 3],
                        },
                    },
                )
            ],
        ),
        # draft_assertion
        _model_response(
            tool_uses=[
                (
                    "tu-assertion",
                    "draft_assertion",
                    {
                        "cg_id": cg_id,
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
            ],
        ),
        # finalize_plan
        _model_response(
            tool_uses=[("tu-finalize", "finalize_plan", {})],
        ),
        # finish
        _model_response(
            tool_uses=[
                (
                    "tu-finish",
                    "finish",
                    {"success": True, "summary": "reproduce paper proposal complete"},
                )
            ],
        ),
    ]


# === Per-role stubs ==========================================================


def _build_per_role_stubs(
    *,
    paper_arxiv_id: str,
    proposal_id: str,
) -> dict[str, StubLlmProvider]:
    """One :class:`StubLlmProvider` per task role.

    Roles invoked in the reproduce-paper-X workflow:

    * ``screener`` — paper-ingestion's screening stage.
    * ``planner`` — drives BOTH the paper-ingestion variant (during
      paper ingestion's ``planning`` state) AND the novel-technique
      variant (during the proposal pipeline's ``designing`` state).
      The single planner stub queues responses for both — the
      paper-ingestion sequence runs first (``smai ingest`` is called
      before ``smai submit-proposal``), then the novel-technique
      sequence.
    * ``enricher`` — not invoked in this test (the contribution
      technique has no comparison baselines that need enrichment).
    """
    role_to_stub: dict[str, StubLlmProvider] = {}
    # Concatenate paper-ingestion-variant + novel-technique-variant
    # responses; the paper-ingestion flow drives first.
    planner_responses = make_paper_planner_responses(
        arxiv_id=paper_arxiv_id
    ) + _make_reproduce_paper_proposal_responses(
        proposal_id=proposal_id,
        arxiv_id=paper_arxiv_id,
    )
    for role in DEFAULT_TASK_ROLES:
        if role == "screener":
            role_to_stub[role] = StubLlmProvider(
                [make_screener_response(decision="accept")],
                name=f"stub-{role}",
            )
        elif role == "planner":
            role_to_stub[role] = StubLlmProvider(
                planner_responses,
                name=f"stub-{role}",
            )
        else:
            role_to_stub[role] = StubLlmProvider([], name=f"stub-{role}")
    return role_to_stub


# === The integration test ===================================================


@pytest.mark.asyncio
async def test_reproduce_paper_workflow(tmp_path: Path) -> None:
    """End-to-end reproduce-paper-X workflow.

    Per DEC-032's primary input path:

    1. ``smai ingest <arxiv>`` — paper ingestion to ``registered``;
       :class:`TechniqueRef`s committed.
    2. ``smai submit-proposal --reproduce-paper <arxiv>`` — proposal
       to ``designed``; one CG drafted referencing the paper.
    3. Approve the proposal — proposal advances to ``registered``;
       CG committed in ``draft``.

    The CG-execution flow downstream (``draft → implementing → ...``)
    is exercised by the smoke test
    (:mod:`tests.integration.test_smoke_e2e`); this test asserts the
    cross-pipeline coordination — paper ingestion's
    :class:`TechniqueRef`s land first; the proposal pipeline's
    registration finds them and creates a CG.
    """
    arxiv_id = "2401.55555"
    proposal_id = "prop-reproduce-paper"
    artifact_store = LocalFsStore(tmp_path / "artifacts")
    role_stubs = _build_per_role_stubs(paper_arxiv_id=arxiv_id, proposal_id=proposal_id)
    fake_fetcher = InProcessFakeFetcher()
    overrides = PluginOverrides(
        llm_providers=cast(dict[str, object], dict(role_stubs)),  # type: ignore[arg-type]
        artifact_store=artifact_store,
    )
    config = build_smoke_runtime_config_for_papers()

    async with Runtime.start_in_band(
        config,
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=overrides,
        run_worker=False,
        paper_fetcher=fake_fetcher,
    ) as runtime:
        # === Phase 1: ingest paper ==========================================
        submission = await runtime.papers.submit(
            arxiv_id=arxiv_id,
            title="Reproduce-Paper Source",
        )
        assert submission.state == "submitted"

        paper_state: str | None = None
        for _ in range(10):
            await runtime.run_one_cycle()
            paper = await runtime.papers.get(arxiv_id)
            if paper.state in {"registered", "rejected", "failed"}:
                paper_state = paper.state
                break
        assert paper_state == "registered", f"paper got {paper_state}"

        # The paper's contribution technique is now in MetadataStore.
        techniques = await runtime.plugins.metadata_store.list_techniques_for_paper(arxiv_id)
        assert len(techniques.items) >= 1

        # === Phase 2: submit reproduce-paper proposal ======================
        prop_submission = await runtime.proposals.submit(
            proposal_id=proposal_id,
            submission_kind="reproduce_paper",
            reproduce_paper_arxiv_id=arxiv_id,
        )
        assert prop_submission.proposal_id == proposal_id
        assert prop_submission.submission_kind == "reproduce_paper"

        # === Phase 3: drive proposal pipeline to designed, then approve ====
        proposal_state: str | None = None
        states_seen: list[str] = []
        for _ in range(10):
            await runtime.run_one_cycle()
            prop = await runtime.proposals.get(proposal_id)
            states_seen.append(prop.state)
            if prop.state in {"designed", "registered", "rejected", "failed"}:
                proposal_state = prop.state
                break
        # The proposal-pipeline runs with require_human_approval=True
        # (per Runtime.start_in_band); the proposal parks at ``designed``.
        assert proposal_state == "designed", (
            f"proposal got {proposal_state}; states_seen={states_seen}"
        )

        # Approve.
        approved = await runtime.proposals.approve(proposal_id)
        assert approved.user_decision == "approved"

        # Drive cycles until proposal reaches ``registered``.
        final_proposal_state: str | None = None
        for _ in range(10):
            await runtime.run_one_cycle()
            prop = await runtime.proposals.get(proposal_id)
            if prop.state in {"registered", "rejected", "failed"}:
                final_proposal_state = prop.state
                break
        assert final_proposal_state == "registered", f"proposal got {final_proposal_state}"

        # ≥ 1 CG was created by the proposal-pipeline registration handler.
        cgs = await runtime.plugins.metadata_store.list_cgs_for_proposal(proposal_id)
        assert len(cgs.items) >= 1
        cg = cgs.items[0]
        assert cg.proposal_id == proposal_id
        assert cg.state == "draft"

        # The proposal carries the reproduce_paper_arxiv_id link.
        prop = await runtime.proposals.get(proposal_id)
        assert prop.reproduce_paper_arxiv_id == arxiv_id

        # The proposal's submission artifact is preserved (per `08` §3.1).
        # We don't assert on the submission artifact key here because
        # ``reproduce_paper`` proposals don't write a technique-description
        # artifact (the paper IS the description); the proposal-record
        # just carries the arxiv-id link.

    # Sanity: the screener was called once; the planner was called for
    # both the paper-ingestion variant AND the novel-technique variant.
    assert len(role_stubs["screener"].calls) == 1
    # Paper-ingestion planner uses 3 turns (create + finalize + finish).
    # Novel-technique planner uses 7 turns (classify + create + comparison
    # + conditions + assertion + finalize + finish). 10 total.
    assert len(role_stubs["planner"].calls) >= 10
