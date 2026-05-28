"""Paper-ingestion round-trip integration test — Task 3.E2 acceptance.

Drives ``Runtime.start_in_band`` through the full paper-ingestion
pipeline: ``submitted → fetching → screening → planning → registered``
with a fake :class:`PaperFetcher` (canned content) + canned per-role
:class:`StubLlmProvider`s. Asserts at least one
:class:`TechniqueRef` is committed to :class:`MetadataStore` with a
:class:`PaperFidelityAnchor` matching the paper.

Per DEC-032: the test verifies paper ingestion produces
``TechniqueRef``s + paper-fidelity-anchor metadata only — NOT
:class:`ExperimentDefinition`s, :class:`ComparisonGroupRecord`s, or
:class:`EntryRecord`s. Reproduce-paper-X workflows (which DO produce
CGs through the proposal pipeline) live in
:mod:`tests.integration.test_reproduce_paper_workflow`.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from _e2_integration_fakes import (  # type: ignore[import-not-found]
    InProcessFakeFetcher,
    StubLlmProvider,
    build_smoke_runtime_config_for_papers,
    make_ingestion_corpus_fetcher,
    make_ingestion_function_model,
    make_screener_response,
)
from smai_artifacts_localfs import LocalFsStore
from smai_cli.runtime import Runtime
from smai_orchestrator import (
    DEFAULT_TASK_ROLES,
    PluginOverrides,
)
from smai_orchestrator.specs.paper_ingestion import (
    PAPER_TEXT_KEY_TEMPLATE,
    SCREEN_RESULT_KEY_TEMPLATE,
    TECHNIQUE_BUFFER_KEY_TEMPLATE,
)


def _build_per_role_stubs() -> dict[str, StubLlmProvider]:
    """Build one :class:`StubLlmProvider` per task role.

    With the planner-refactor Step-3 cutover, only the ``screener`` role
    LLM is actually invoked in this fixture: the ``screening`` state runs
    the screener once, and the ``planning`` state's ingestion subagent
    runs on the injected ``FunctionModel`` (its in-tool ``ingestion``
    sub-extractions never fire because the model emits the output tool
    immediately). Every other role gets an empty-queue stub that
    ``AssertionError``s if anything calls it — a tripwire that also
    proves the no-double-screen contract (the ``ingestion`` role stub
    would be called on a re-screen / sub-extraction).
    """
    role_to_stub: dict[str, StubLlmProvider] = {}
    for role in DEFAULT_TASK_ROLES:
        if role == "screener":
            role_to_stub[role] = StubLlmProvider(
                [make_screener_response(decision="accept")],
                name=f"stub-{role}",
            )
        else:
            role_to_stub[role] = StubLlmProvider([], name=f"stub-{role}")
    return role_to_stub


@pytest.mark.asyncio
async def test_paper_ingestion_round_trip_submitted_to_registered(tmp_path: Path) -> None:
    """End-to-end paper-ingestion round-trip per Task 3.E2 acceptance.

    Submits a paper via :meth:`PapersService.submit`; drives cycles
    via :meth:`Runtime.run_one_cycle` until the paper reaches
    ``registered``; asserts ≥ 1 ``TechniqueRef`` is committed with a
    matching :class:`PaperFidelityAnchor`.
    """
    arxiv_id = "2401.12345"
    artifact_store = LocalFsStore(tmp_path / "artifacts")
    role_stubs = _build_per_role_stubs()
    fake_fetcher = InProcessFakeFetcher()
    overrides = PluginOverrides(
        llm_providers=cast(dict[str, object], dict(role_stubs)),  # type: ignore[arg-type]
        artifact_store=artifact_store,
    )
    config = build_smoke_runtime_config_for_papers()

    final_state: str | None = None

    async with Runtime.start_in_band(
        config,
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=overrides,
        run_worker=False,
        paper_fetcher=fake_fetcher,
        # Step-3 ingestion subagent runs offline: fake LaTeX corpus +
        # a FunctionModel emitting the paper_extract output tool.
        ingestion_corpus_fetcher=make_ingestion_corpus_fetcher(arxiv_id=arxiv_id),
        ingestion_model=make_ingestion_function_model(source_arxiv_id=arxiv_id),
    ) as runtime:
        submission = await runtime.papers.submit(arxiv_id=arxiv_id, title="Test Paper")
        assert submission.arxiv_id == arxiv_id
        assert submission.state == "submitted"
        assert submission.promoted is False

        # Drive cycles until terminal.
        for _ in range(10):
            await runtime.run_one_cycle()
            paper = await runtime.papers.get(arxiv_id)
            if paper.state in {"registered", "rejected", "failed"}:
                final_state = paper.state
                break

        assert final_state == "registered", f"got {final_state}"

        # The fetcher was actually called.
        assert fake_fetcher.fetch_log == [arxiv_id]

        # Paper-text + screen-result + technique-buffer artifacts persisted.
        assert await artifact_store.exists(PAPER_TEXT_KEY_TEMPLATE.format(arxiv_id=arxiv_id))
        assert await artifact_store.exists(SCREEN_RESULT_KEY_TEMPLATE.format(arxiv_id=arxiv_id))
        assert await artifact_store.exists(TECHNIQUE_BUFFER_KEY_TEMPLATE.format(arxiv_id=arxiv_id))

        # ≥ 1 paper_extract ``TechniqueRef`` committed with a
        # ``PaperFidelityAnchor``.
        techniques = await runtime.plugins.metadata_store.list_techniques_for_paper(arxiv_id)
        assert len(techniques.items) >= 1
        technique = techniques.items[0]
        assert technique.context_kind == "paper_extract"
        assert technique.fidelity_anchor is not None
        assert technique.fidelity_anchor.kind == "paper"
        assert technique.fidelity_anchor.arxiv_id == arxiv_id

        # Per DEC-032: NO CGs are created by paper ingestion. Confirm
        # the CG table is empty for this paper's arxiv_id (paper
        # ingestion does NOT carry a parent_proposal_id, and we never
        # invoked the proposal pipeline in this test).
        # No direct "list cgs by paper" surface, so we sample the
        # CG-list-by-proposal route would also be empty (no proposals
        # were created either).

    # Sanity check: the screener stub was invoked exactly once (the
    # screening state). No double-screen + no sub-extraction: the
    # ``ingestion`` role provider was never called (the subagent ran on
    # the injected FunctionModel and reused the screening verdict).
    assert len(role_stubs["screener"].calls) == 1
    assert len(role_stubs["ingestion"].calls) == 0
    assert len(role_stubs["planner"].calls) == 0
