"""Integration smoke test for the ``smai serve`` dashboard.

Per Task 3.H1 acceptance: boots the dashboard against a populated
:class:`MetadataStore` and verifies every page returns 200. Uses the
real :func:`Runtime.start_in_band` context manager so the dashboard
is exercised against the same plugin substrate ``smai dev`` would.

The populated state is built by reusing the paper-ingestion round-trip
flow (Task 3.E2's integration fixture) — this drives a paper through
``submitted → fetching → screening → planning → registered`` against
a real :class:`SqliteStore` + :class:`LocalFsStore`. We then mount the
FastAPI app on the same Runtime and walk every page.

Empty-DB rendering is covered separately by the per-route unit tests
under ``packages/smai-cli/tests/test_dashboard.py``; this integration
test focuses on the populated-DB happy path against the real plugin
stack.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from _e2_integration_fakes import (  # type: ignore[import-not-found]
    InProcessFakeFetcher,
    StubLlmProvider,
    build_smoke_runtime_config_for_papers,
    make_paper_planner_responses,
    make_screener_response,
)
from fastapi.testclient import TestClient
from smai_artifacts_localfs import LocalFsStore
from smai_cli.dashboard import build_app
from smai_cli.runtime import Runtime
from smai_orchestrator import (
    DEFAULT_TASK_ROLES,
    ComparisonGroupRecord,
    EntryRecord,
    PluginOverrides,
    ProposalRecord,
    RunRecord,
)


def _now() -> datetime:
    return datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)


def _build_per_role_stubs(arxiv_id: str) -> dict[str, StubLlmProvider]:
    """Per-role :class:`StubLlmProvider` map for the paper-ingestion drive.

    Mirrors :mod:`tests.integration.test_paper_ingestion_round_trip`'s
    setup — we only exercise the paper-ingestion flow here so only
    ``screener`` / ``planner`` need canned responses; everything else
    gets an empty-queue tripwire.
    """
    role_to_stub: dict[str, StubLlmProvider] = {}
    for role in DEFAULT_TASK_ROLES:
        if role == "screener":
            role_to_stub[role] = StubLlmProvider(
                [make_screener_response(decision="accept")],
                name=f"stub-{role}",
            )
        elif role == "planner":
            role_to_stub[role] = StubLlmProvider(
                make_paper_planner_responses(arxiv_id=arxiv_id),
                name=f"stub-{role}",
            )
        else:
            role_to_stub[role] = StubLlmProvider([], name=f"stub-{role}")
    return role_to_stub


_PAGES_TO_VISIT: tuple[str, ...] = (
    "/",
    "/proposals/",
    "/cgs/",
    "/runs/",
    "/papers/",
)


@pytest.mark.asyncio
async def test_dashboard_smoke_against_populated_runtime(tmp_path: Path) -> None:
    """Drive a paper through ingestion, then walk every dashboard page.

    Acceptance per the Task 3.H1 brief: ``smai serve`` boots; the
    dashboard renders against a populated :class:`MetadataStore`;
    every page loads without errors.
    """
    arxiv_id = "2401.99999"
    artifact_store = LocalFsStore(tmp_path / "artifacts")
    role_stubs = _build_per_role_stubs(arxiv_id)
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
        # Drive the paper through the ingestion pipeline.
        await runtime.papers.submit(arxiv_id=arxiv_id, title="Smoke Test Paper")
        for _ in range(10):
            await runtime.run_one_cycle()
            paper = await runtime.papers.get(arxiv_id)
            if paper.state in {"registered", "rejected", "failed"}:
                break

        # Plus seed a proposal-with-CG-with-entry-with-run quintet so
        # every list page surfaces real fixture rows. We write directly
        # via :class:`MetadataStore` rather than driving the proposal
        # pipeline (which would require LLM stubs we haven't canned for
        # the planner-novel-technique variant in this test).
        store = runtime.plugins.metadata_store
        now = _now()
        await store.create_proposal(
            ProposalRecord(
                id="proposal-smoke",
                submission_kind="novel_technique",
                state="proposal_submitted",
                created_at=now,
                updated_at=now,
            )
        )
        await store.create_cg(
            ComparisonGroupRecord(
                id="cg-smoke",
                proposal_id="proposal-smoke",
                experiment_definition_id="cg-smoke",
                state="draft",
                created_at=now,
                updated_at=now,
            )
        )
        await store.create_entry(
            EntryRecord(
                id="entry-smoke",
                cg_id="cg-smoke",
                technique_id=None,
                is_baseline=True,
                entry_id="entry-smoke",
                state="implemented",
                created_at=now,
                updated_at=now,
            )
        )
        await store.create_run(
            RunRecord(
                id="run-smoke",
                cg_id="cg-smoke",
                entry_id="entry-smoke",
                seed=1,
                state="pending",
                created_at=now,
                updated_at=now,
            )
        )

        # Boot the dashboard against the same Runtime and walk every page.
        app = build_app(runtime)
        client = TestClient(app)

        for path in _PAGES_TO_VISIT:
            response = client.get(path)
            assert response.status_code == 200, (
                f"{path} returned {response.status_code}: {response.text[:200]}"
            )
            # The base template's brand banner should always render.
            assert "smai dashboard" in response.text

        # Per-entity detail pages — exercise each kind via the seeded
        # fixture rows.
        proposal_response = client.get("/proposals/proposal-smoke")
        assert proposal_response.status_code == 200
        assert "proposal-smoke" in proposal_response.text

        cg_response = client.get("/cgs/cg-smoke")
        assert cg_response.status_code == 200
        assert "cg-smoke" in cg_response.text
        assert "entry-smoke" in cg_response.text
        assert "run-smoke" in cg_response.text

        run_response = client.get("/runs/run-smoke")
        assert run_response.status_code == 200
        assert "run-smoke" in run_response.text

        paper_response = client.get(f"/papers/{arxiv_id}")
        assert paper_response.status_code == 200
        assert arxiv_id in paper_response.text
        # The paper round-trip ingested ≥ 1 technique; the detail page
        # should surface it in the registered-techniques table.
        assert "registered techniques" in paper_response.text

        # Static CSS reachable.
        css_response = client.get("/static/style.css")
        assert css_response.status_code == 200
