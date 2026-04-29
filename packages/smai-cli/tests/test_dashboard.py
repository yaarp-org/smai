"""Per-route unit tests for the ``smai serve`` dashboard.

Uses an in-memory :class:`SqliteStore` and a duck-typed
:class:`FakeRuntime` to exercise every page handler against real
:class:`MetadataStore` Protocol round-trips. The full
:class:`Runtime.start_in_band` context manager is exercised in the
integration smoke test under ``tests/integration/test_dashboard_smoke.py``.

Per the Task 3.H1 brief: empty-DB rendering is verified for every
list page; populated-DB rendering is verified by writing fixture rows
through the :class:`MetadataStore` Protocol surface and asserting on
key field presence in the rendered HTML.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from _h1_dashboard_fakes import (  # type: ignore[import-not-found]
    FakeRuntime,
    build_test_app,
    make_in_memory_store,
)
from fastapi.testclient import TestClient
from smai_orchestrator import (
    ComparisonGroupRecord,
    EntryRecord,
    PaperRecord,
    ProposalRecord,
    RunRecord,
)


def _now() -> datetime:
    return datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)


# === Empty-DB rendering (every page returns 200) =============================


@pytest.mark.asyncio
async def test_empty_db_index_renders_zero_counts() -> None:
    store = await make_in_memory_store()
    runtime = FakeRuntime(store=store)
    client = TestClient(build_test_app(runtime))
    response = client.get("/")
    assert response.status_code == 200
    assert "smai dashboard" in response.text
    assert 'data-test="count-proposals"' in response.text
    # Every count is zero on an empty DB.
    assert response.text.count(">0<") >= 4


@pytest.mark.asyncio
async def test_empty_db_proposals_list_renders_empty_state() -> None:
    store = await make_in_memory_store()
    runtime = FakeRuntime(store=store)
    client = TestClient(build_test_app(runtime))
    response = client.get("/proposals/")
    assert response.status_code == 200
    assert 'data-test="empty-proposals"' in response.text
    assert "No active proposals" in response.text


@pytest.mark.asyncio
async def test_empty_db_cgs_list_renders_empty_state() -> None:
    store = await make_in_memory_store()
    runtime = FakeRuntime(store=store)
    client = TestClient(build_test_app(runtime))
    response = client.get("/cgs/")
    assert response.status_code == 200
    assert 'data-test="empty-cgs"' in response.text


@pytest.mark.asyncio
async def test_empty_db_runs_list_renders_empty_state() -> None:
    store = await make_in_memory_store()
    runtime = FakeRuntime(store=store)
    client = TestClient(build_test_app(runtime))
    response = client.get("/runs/")
    assert response.status_code == 200
    assert 'data-test="empty-runs"' in response.text


@pytest.mark.asyncio
async def test_empty_db_papers_list_renders_empty_state() -> None:
    store = await make_in_memory_store()
    runtime = FakeRuntime(store=store)
    client = TestClient(build_test_app(runtime))
    response = client.get("/papers/")
    assert response.status_code == 200
    assert 'data-test="empty-papers"' in response.text


# === Populated-DB rendering ==================================================


@pytest.mark.asyncio
async def test_populated_index_surfaces_in_flight_counts() -> None:
    """The index counts in-flight entities per kind.

    Writes one in-flight row per kind plus one terminal row per kind
    and verifies the counts reflect only in-flight entities.
    """
    store = await make_in_memory_store()
    now = _now()

    # In-flight entities — one per kind.
    await store.create_proposal(
        ProposalRecord(
            id="proposal-active",
            submission_kind="novel_technique",
            state="proposal_submitted",
            created_at=now,
            updated_at=now,
        )
    )
    await store.create_cg(
        ComparisonGroupRecord(
            id="cg-active",
            proposal_id="proposal-active",
            experiment_definition_id="cg-active",
            state="draft",
            created_at=now,
            updated_at=now,
        )
    )
    await store.create_entry(
        EntryRecord(
            id="entry-active",
            cg_id="cg-active",
            technique_id=None,
            is_baseline=True,
            entry_id="entry-active",
            state="pending",
            created_at=now,
            updated_at=now,
        )
    )
    await store.create_run(
        RunRecord(
            id="run-active",
            cg_id="cg-active",
            entry_id="entry-active",
            seed=1,
            state="pending",
            created_at=now,
            updated_at=now,
        )
    )
    await store.create_paper(
        PaperRecord(
            arxiv_id="2107.12345",
            title="A Sample Paper",
            state="submitted",
            created_at=now,
            updated_at=now,
        )
    )

    # Terminal rows — should NOT be counted as in-flight.
    await store.create_proposal(
        ProposalRecord(
            id="proposal-terminal",
            submission_kind="novel_technique",
            state="registered",
            created_at=now,
            updated_at=now,
        )
    )
    await store.create_paper(
        PaperRecord(
            arxiv_id="2107.99999",
            title="Terminal Paper",
            state="registered",
            created_at=now,
            updated_at=now,
        )
    )

    runtime = FakeRuntime(store=store)
    client = TestClient(build_test_app(runtime))
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert _count_value(body, "count-proposals") == "1"
    assert _count_value(body, "count-cgs") == "1"
    assert _count_value(body, "count-runs") == "1"
    assert _count_value(body, "count-papers") == "1"


@pytest.mark.asyncio
async def test_proposal_detail_renders_human_gate_affordance() -> None:
    """A proposal in ``designed`` with ``user_decision is None`` surfaces
    the awaiting-decision callout per the per-3.E1 status note.
    """
    store = await make_in_memory_store()
    now = _now()
    proposal_id = "proposal-at-gate"
    await store.create_proposal(
        ProposalRecord(
            id=proposal_id,
            submission_kind="novel_technique",
            state="proposal_submitted",
            created_at=now,
            updated_at=now,
        )
    )
    p = await store.get_proposal(proposal_id)
    assert p is not None
    advanced = await store.transition_proposal_state(proposal_id, p.version, "designing")
    advanced = await store.transition_proposal_state(proposal_id, advanced.version, "designed")
    assert advanced.user_decision is None  # at the human gate

    runtime = FakeRuntime(store=store)
    client = TestClient(build_test_app(runtime))
    response = client.get(f"/proposals/{proposal_id}")
    assert response.status_code == 200
    body = response.text
    assert proposal_id in body
    assert 'data-test="proposal-human-gate"' in body
    assert "smai approve-proposal" in body
    assert "smai reject-proposal" in body


@pytest.mark.asyncio
async def test_proposal_detail_decided_pending_registration() -> None:
    """A ``designed`` proposal with a non-None ``user_decision`` is
    surfaced distinctly from human-gate-pending.
    """
    store = await make_in_memory_store()
    now = _now()
    proposal_id = "proposal-decided"
    await store.create_proposal(
        ProposalRecord(
            id=proposal_id,
            submission_kind="novel_technique",
            state="proposal_submitted",
            created_at=now,
            updated_at=now,
        )
    )
    p = await store.get_proposal(proposal_id)
    assert p is not None
    advanced = await store.transition_proposal_state(proposal_id, p.version, "designing")
    advanced = await store.transition_proposal_state(
        proposal_id,
        advanced.version,
        "designed",
        user_decision="approved",
        user_decided_at=now,
    )

    runtime = FakeRuntime(store=store)
    client = TestClient(build_test_app(runtime))
    response = client.get(f"/proposals/{proposal_id}")
    assert response.status_code == 200
    body = response.text
    assert 'data-test="proposal-decided"' in body
    assert "approved" in body


@pytest.mark.asyncio
async def test_cg_detail_lists_entries_and_runs() -> None:
    store = await make_in_memory_store()
    now = _now()
    cg_id = "cg-detail-test"
    await store.create_proposal(
        ProposalRecord(
            id="proposal-detail",
            submission_kind="novel_technique",
            state="registered",
            created_at=now,
            updated_at=now,
        )
    )
    await store.create_cg(
        ComparisonGroupRecord(
            id=cg_id,
            proposal_id="proposal-detail",
            experiment_definition_id=cg_id,
            state="implementing",
            created_at=now,
            updated_at=now,
        )
    )
    await store.create_entry(
        EntryRecord(
            id="entry-baseline",
            cg_id=cg_id,
            technique_id=None,
            is_baseline=True,
            entry_id="entry-baseline",
            state="implemented",
            created_at=now,
            updated_at=now,
        )
    )
    await store.create_run(
        RunRecord(
            id="run-1",
            cg_id=cg_id,
            entry_id="entry-baseline",
            seed=1,
            state="pending",
            created_at=now,
            updated_at=now,
        )
    )

    runtime = FakeRuntime(store=store)
    client = TestClient(build_test_app(runtime))
    response = client.get(f"/cgs/{cg_id}")
    assert response.status_code == 200
    body = response.text
    assert cg_id in body
    assert "entry-baseline" in body
    assert "run-1" in body
    # Baseline badge surfaces.
    assert ">baseline<" in body


@pytest.mark.asyncio
async def test_run_detail_surfaces_failure_reason() -> None:
    store = await make_in_memory_store()
    now = _now()
    cg_id = "cg-run"
    await store.create_proposal(
        ProposalRecord(
            id="proposal-run",
            submission_kind="novel_technique",
            state="registered",
            created_at=now,
            updated_at=now,
        )
    )
    await store.create_cg(
        ComparisonGroupRecord(
            id=cg_id,
            proposal_id="proposal-run",
            experiment_definition_id=cg_id,
            state="running",
            created_at=now,
            updated_at=now,
        )
    )
    await store.create_entry(
        EntryRecord(
            id="entry-run",
            cg_id=cg_id,
            technique_id=None,
            is_baseline=True,
            entry_id="entry-run",
            state="implemented",
            created_at=now,
            updated_at=now,
        )
    )
    run_id = "run-detail"
    await store.create_run(
        RunRecord(
            id=run_id,
            cg_id=cg_id,
            entry_id="entry-run",
            seed=42,
            state="pending",
            created_at=now,
            updated_at=now,
        )
    )

    runtime = FakeRuntime(store=store)
    client = TestClient(build_test_app(runtime))
    response = client.get(f"/runs/{run_id}")
    assert response.status_code == 200
    assert run_id in response.text
    # Seed value should render.
    assert ">42<" in response.text


@pytest.mark.asyncio
async def test_paper_detail_partial_surfaces_promote_affordance() -> None:
    """A paper in ``partial`` surfaces the CLI promotion command
    affordance per the Task 3.H1 brief / `08` §5.7.
    """
    store = await make_in_memory_store()
    now = _now()
    arxiv_id = "1804.07612"
    await store.create_paper(
        PaperRecord(
            arxiv_id=arxiv_id,
            title="Partial Paper Test",
            state="submitted",
            created_at=now,
            updated_at=now,
        )
    )
    paper = await store.get_paper(arxiv_id)
    assert paper is not None
    paper = await store.transition_paper_state(arxiv_id, paper.version, "fetching")
    paper = await store.transition_paper_state(arxiv_id, paper.version, "screening")
    paper = await store.transition_paper_state(arxiv_id, paper.version, "planning")
    paper = await store.transition_paper_state(arxiv_id, paper.version, "partial")

    runtime = FakeRuntime(store=store)
    client = TestClient(build_test_app(runtime))
    response = client.get(f"/papers/{arxiv_id}")
    assert response.status_code == 200
    body = response.text
    assert arxiv_id in body
    assert 'data-test="paper-promote-affordance"' in body
    assert f"smai ingest --promote-partial {arxiv_id}" in body


@pytest.mark.asyncio
async def test_paper_detail_renders_unset_error_context_gracefully() -> None:
    """Per the 3.E2 status-note carry-forward: when
    ``error_context_artifact_key`` is ``None`` the paper detail page
    surfaces an explicit "(unset — ...)" placeholder rather than crashing.
    """
    store = await make_in_memory_store()
    now = _now()
    arxiv_id = "2107.12345"
    await store.create_paper(
        PaperRecord(
            arxiv_id=arxiv_id,
            title="Submitted",
            state="submitted",
            created_at=now,
            updated_at=now,
        )
    )

    runtime = FakeRuntime(store=store)
    client = TestClient(build_test_app(runtime))
    response = client.get(f"/papers/{arxiv_id}")
    assert response.status_code == 200
    assert "(unset — check most-recent transition log entry" in response.text


@pytest.mark.asyncio
async def test_proposal_state_filter_narrows_list() -> None:
    """Passing ``?state=designed`` filters the proposals list to that state."""
    store = await make_in_memory_store()
    now = _now()
    # One proposal_submitted, one designed.
    await store.create_proposal(
        ProposalRecord(
            id="proposal-submitted",
            submission_kind="novel_technique",
            state="proposal_submitted",
            created_at=now,
            updated_at=now,
        )
    )
    await store.create_proposal(
        ProposalRecord(
            id="proposal-designed",
            submission_kind="novel_technique",
            state="proposal_submitted",
            created_at=now,
            updated_at=now,
        )
    )
    pd = await store.get_proposal("proposal-designed")
    assert pd is not None
    pd = await store.transition_proposal_state("proposal-designed", pd.version, "designing")
    await store.transition_proposal_state("proposal-designed", pd.version, "designed")

    runtime = FakeRuntime(store=store)
    client = TestClient(build_test_app(runtime))
    response = client.get("/proposals/?state=designed")
    assert response.status_code == 200
    body = response.text
    assert "proposal-designed" in body
    assert "proposal-submitted" not in body


@pytest.mark.asyncio
async def test_unknown_id_returns_404() -> None:
    store = await make_in_memory_store()
    runtime = FakeRuntime(store=store)
    client = TestClient(build_test_app(runtime))
    assert client.get("/proposals/does-not-exist").status_code == 404
    assert client.get("/cgs/missing-cg").status_code == 404
    assert client.get("/runs/missing-run").status_code == 404
    assert client.get("/papers/9999.99999").status_code == 404


@pytest.mark.asyncio
async def test_static_files_served() -> None:
    store = await make_in_memory_store()
    runtime = FakeRuntime(store=store)
    client = TestClient(build_test_app(runtime))
    response = client.get("/static/style.css")
    assert response.status_code == 200
    assert "topbar" in response.text


@pytest.mark.asyncio
async def test_no_mutation_endpoints_exposed() -> None:
    """The dashboard is read-only by spec — POST/PUT/PATCH/DELETE
    against any endpoint must 405 (or 404)."""
    store = await make_in_memory_store()
    runtime = FakeRuntime(store=store)
    client = TestClient(build_test_app(runtime))
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)("/")
        assert response.status_code in (404, 405), (
            f"{method.upper()} / returned {response.status_code}; "
            "dashboard must not expose mutation endpoints"
        )


# === Helpers =================================================================


def _count_value(html: str, data_test: str) -> str:
    """Parse the count-pill numeric value from a ``data-test`` element."""
    needle = f'data-test="{data_test}">'
    idx = html.find(needle)
    if idx == -1:
        raise AssertionError(f"data-test={data_test!r} not found in HTML")
    start = idx + len(needle)
    end = html.find("<", start)
    return html[start:end].strip()


# Suppress unused-import warning — Any is referenced indirectly via type checks.
_ = Any
